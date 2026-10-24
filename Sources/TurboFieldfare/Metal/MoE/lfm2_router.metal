#include <metal_stdlib>
using namespace metal;

// LFM2.5 MoE decode path. Deliberately separate from `moe.metal`: Gemma 4 uses
// a top-8 / GELU router and expert stream, LFM2.5 uses a top-4 / SiLU one.
// Keeping the LFM kernels in their own module leaves every Gemma pipeline and
// its function constants untouched.
//
// Router: an FP32 64 x 2048 GEMV produces FP32 logits. Selection scores are
// `sigmoid(logit) + expert_bias`; the bias is never part of the emitted weight.
// Weights are the unbiased sigmoid values of the four selected experts,
// normalized with `sum + 1e-6` and scaled by `routedScalingFactor`.
//
// Dense FFN and routed experts share `lfm2_silu` = `x * sigmoid(x)`.

constant constexpr uint kLFM2GroupSize = 64;
constant constexpr uint kLFM2MaxStreamedExperts = 4;
constant constexpr float kLFM2RoutingEpsilon = 1e-6f;

struct LFM2ExpertOffsets {
    uint gate_W_off;
    uint gate_s_off;
    uint gate_b_off;
    uint up_W_off;
    uint up_s_off;
    uint up_b_off;
    uint down_W_off;
    uint down_s_off;
    uint down_b_off;
};

struct LFM2RoutedBlobs {
    device const uint8_t* blob[kLFM2MaxStreamedExperts];
};

static inline float lfm2_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

static inline float lfm2_silu(float x) {
    return x / (1.0f + exp(-x));
}

// One SIMD group per expert row. Weights are FP32, activations FP16, and the
// dot product accumulates in FP32.
kernel void lfm2_router_gemv_f32(
    device const float* W [[buffer(0)]],
    device const half* x [[buffer(1)]],
    device float* out_logits [[buffer(2)]],
    constant uint& num_experts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint e = tg_idx * 8u + sg_idx;
    if (e >= num_experts) return;

    device const float* row = W + uint(e) * D;
    float acc = 0.0f;
    for (uint i = lane; i < D; i += 32u) {
        acc = fma(row[i], float(x[i]), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) out_logits[e] = acc;
}

// Single-thread top-4 selection. `top_idx` stays ordered best-first and ties
// prefer the lower expert index; because experts are visited in ascending
// order, a score equal to the current worst can never improve the tie-break.
kernel void lfm2_router_select_top4(
    device const float* logits [[buffer(0)]],
    device const float* expert_bias [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    device half* out_weights [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    constant uint& top_k [[buffer(5)]],
    constant float& routed_scaling_factor [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0u || top_k != kLFM2MaxStreamedExperts) return;

    uint top_idx[kLFM2MaxStreamedExperts];
    float top_score[kLFM2MaxStreamedExperts];
    float top_sigmoid[kLFM2MaxStreamedExperts];
    for (uint i = 0; i < kLFM2MaxStreamedExperts; ++i) {
        top_idx[i] = 0u;
        top_score[i] = -INFINITY;
        top_sigmoid[i] = 0.0f;
    }

    for (uint e = 0; e < num_experts; ++e) {
        const float unbiased = lfm2_sigmoid(logits[e]);
        const float score = unbiased + expert_bias[e];
        if (score <= top_score[kLFM2MaxStreamedExperts - 1u]) continue;

        uint pos = kLFM2MaxStreamedExperts;
        for (uint i = 0; i < kLFM2MaxStreamedExperts; ++i) {
            if (score > top_score[i] ||
                (score == top_score[i] && e < top_idx[i])) {
                pos = i;
                break;
            }
        }
        if (pos >= kLFM2MaxStreamedExperts) continue;
        for (uint i = kLFM2MaxStreamedExperts - 1u; i > pos; --i) {
            top_idx[i] = top_idx[i - 1u];
            top_score[i] = top_score[i - 1u];
            top_sigmoid[i] = top_sigmoid[i - 1u];
        }
        top_idx[pos] = e;
        top_score[pos] = score;
        top_sigmoid[pos] = unbiased;
    }

    float sum = 0.0f;
    for (uint i = 0; i < kLFM2MaxStreamedExperts; ++i) {
        sum += top_sigmoid[i];
    }
    const float denominator = sum + kLFM2RoutingEpsilon;
    for (uint i = 0; i < kLFM2MaxStreamedExperts; ++i) {
        out_indices[i] = top_idx[i];
        out_weights[i] = half(top_sigmoid[i] / denominator * routed_scaling_factor);
    }
}

// Prefill router: one threadgroup per token row. Each thread accumulates whole
// expert rows in FP32, then thread 0 runs the same top-4 selection the decode
// kernel uses — `sigmoid(logit) + expert_bias` for selection, unbiased sigmoid
// values normalized with `sum + 1e-6` and scaled by `routedScalingFactor`.
// Gemma's `prefill_router_gemma4_block` is deliberately not reused: it applies
// the Gemma softmax/gain contract and unpacks INT4-affine router weights.
kernel void prefill_lfm2_router_block(
    device const float* W                   [[buffer(0)]],   // [num_experts, D] FP32
    device const float* expert_bias         [[buffer(1)]],   // [num_experts] FP32
    device const half*  hidden              [[buffer(2)]],   // [T, hidden_stride]
    device uint*        out_indices         [[buffer(3)]],   // [T, top_k]
    device half*        out_weights         [[buffer(4)]],   // [T, top_k]
    constant uint&      T                   [[buffer(5)]],
    constant uint&      num_experts         [[buffer(6)]],
    constant uint&      D                   [[buffer(7)]],
    constant uint&      top_k               [[buffer(8)]],
    constant uint&      hidden_stride       [[buffer(9)]],
    constant float&     routed_scaling_factor [[buffer(10)]],
    uint                row                 [[threadgroup_position_in_grid]],
    uint                tid                 [[thread_position_in_threadgroup]],
    uint                tg_size             [[threads_per_threadgroup]]
) {
    if (row >= T || top_k != kLFM2MaxStreamedExperts) return;

    threadgroup float scores[256];
    const uint NE = min(num_experts, 256u);
    device const half* row_hidden = hidden + row * hidden_stride;

    for (uint e = tid; e < NE; e += tg_size) {
        device const float* w_row = W + uint(e) * D;
        float acc = 0.0f;
        for (uint i = 0; i < D; ++i) {
            acc = fma(w_row[i], float(row_hidden[i]), acc);
        }
        scores[e] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid != 0u) return;

    uint top_idx[kLFM2MaxStreamedExperts];
    float top_score[kLFM2MaxStreamedExperts];
    float top_sigmoid[kLFM2MaxStreamedExperts];
    for (uint i = 0; i < kLFM2MaxStreamedExperts; ++i) {
        top_idx[i] = 0u;
        top_score[i] = -INFINITY;
        top_sigmoid[i] = 0.0f;
    }

    for (uint e = 0; e < NE; ++e) {
        const float unbiased = lfm2_sigmoid(scores[e]);
        const float score = unbiased + expert_bias[e];
        if (score <= top_score[kLFM2MaxStreamedExperts - 1u]) continue;

        uint pos = kLFM2MaxStreamedExperts;
        for (uint i = 0; i < kLFM2MaxStreamedExperts; ++i) {
            if (score > top_score[i] ||
                (score == top_score[i] && e < top_idx[i])) {
                pos = i;
                break;
            }
        }
        if (pos >= kLFM2MaxStreamedExperts) continue;
        for (uint i = kLFM2MaxStreamedExperts - 1u; i > pos; --i) {
            top_idx[i] = top_idx[i - 1u];
            top_score[i] = top_score[i - 1u];
            top_sigmoid[i] = top_sigmoid[i - 1u];
        }
        top_idx[pos] = e;
        top_score[pos] = score;
        top_sigmoid[pos] = unbiased;
    }

    float sum = 0.0f;
    for (uint i = 0; i < kLFM2MaxStreamedExperts; ++i) {
        sum += top_sigmoid[i];
    }
    const float denominator = sum + kLFM2RoutingEpsilon;
    for (uint i = 0; i < kLFM2MaxStreamedExperts; ++i) {
        out_indices[row * top_k + i] = top_idx[i];
        out_weights[row * top_k + i] =
            half(top_sigmoid[i] / denominator * routed_scaling_factor);
    }
}

kernel void lfm2_silu_mul_fp16(
    device const half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    out[tid] = half(lfm2_silu(float(gate[tid])) * float(up[tid]));
}

// One SIMD group per affine INT4 row. Four adjacent groups are loaded as
// aligned 32-bit chunks; remaining groups use one byte per lane. Mirrors
// `moe_int4_gemv_row_simd_dev_vec`.
static inline float lfm2_int4_gemv_row_simd_dev_vec(
    device const uint8_t* W,
    device const bfloat* S,
    device const bfloat* B,
    device const half* x,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups = N / kLFM2GroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W + uint(row) * row_bytes;
    device const bfloat* s_row = S + uint(row) * n_groups;
    device const bfloat* b_row = B + uint(row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        const uint w4 = *((device const uint*)(W_row + byte_base));
        const uint g = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const uint b0 = w4 & 0xFFu;
        const uint b1 = (w4 >> 8) & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        const float e0 = float(xa.x), e1 = float(xa.y);
        const float e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y);
        const float e6 = float(xb.z), e7 = float(xb.w);
        float dot = 0.0f;
        dot = fma(float(b0 & 0x0Fu), e0, dot); dot = fma(float(b0 >> 4), e1, dot);
        dot = fma(float(b1 & 0x0Fu), e2, dot); dot = fma(float(b1 >> 4), e3, dot);
        dot = fma(float(b2 & 0x0Fu), e4, dot); dot = fma(float(b2 >> 4), e5, dot);
        dot = fma(float(b3 & 0x0Fu), e6, dot); dot = fma(float(b3 >> 4), e7, dot);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint8_t byte = W_row[g * (kLFM2GroupSize / 2) + lane];
        const float x0 = float(x[g * kLFM2GroupSize + lane * 2u]);
        const float x1 = float(x[g * kLFM2GroupSize + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        acc = fma(s, dot, acc);
        acc = fma(b, x0 + x1, acc);
    }
    return simd_sum(acc);
}

// Gate and up rows share activation loads. Two 16-bit loads assemble each
// 4-byte weight chunk because packed sub-tensor offsets need only be 2-byte
// aligned. Mirrors `moe_int4_gate_up_rows_simd_dev_vec_u16load`.
static inline float2 lfm2_int4_gate_up_rows_simd_dev_vec_u16load(
    device const uint8_t* gateW,
    device const bfloat* gateS,
    device const bfloat* gateB,
    device const uint8_t* upW,
    device const bfloat* upS,
    device const bfloat* upB,
    device const half* x,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups = N / kLFM2GroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* gW_row = gateW + uint(row) * row_bytes;
    device const uint8_t* uW_row = upW + uint(row) * row_bytes;
    device const bfloat* gS_row = gateS + uint(row) * n_groups;
    device const bfloat* gB_row = gateB + uint(row) * n_groups;
    device const bfloat* uS_row = upS + uint(row) * n_groups;
    device const bfloat* uB_row = upB + uint(row) * n_groups;

    float g_acc = 0.0f;
    float u_acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* gp = (device const ushort*)(gW_row + byte_base);
        device const ushort* up = (device const ushort*)(uW_row + byte_base);
        const uint gw4 = uint(gp[0]) | (uint(gp[1]) << 16);
        const uint uw4 = uint(up[0]) | (uint(up[1]) << 16);
        const uint g = blk * 4u + (lane >> 3);
        const float gs = float(gS_row[g]);
        const float gb = float(gB_row[g]);
        const float us = float(uS_row[g]);
        const float ub = float(uB_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const float e0 = float(xa.x), e1 = float(xa.y);
        const float e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y);
        const float e6 = float(xb.z), e7 = float(xb.w);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;

        const uint gb0 = gw4 & 0xFFu;
        const uint gb1 = (gw4 >> 8) & 0xFFu;
        const uint gb2 = (gw4 >> 16) & 0xFFu;
        const uint gb3 = (gw4 >> 24) & 0xFFu;
        float g_dot = 0.0f;
        g_dot = fma(float(gb0 & 0x0Fu), e0, g_dot); g_dot = fma(float(gb0 >> 4), e1, g_dot);
        g_dot = fma(float(gb1 & 0x0Fu), e2, g_dot); g_dot = fma(float(gb1 >> 4), e3, g_dot);
        g_dot = fma(float(gb2 & 0x0Fu), e4, g_dot); g_dot = fma(float(gb2 >> 4), e5, g_dot);
        g_dot = fma(float(gb3 & 0x0Fu), e6, g_dot); g_dot = fma(float(gb3 >> 4), e7, g_dot);

        const uint ub0 = uw4 & 0xFFu;
        const uint ub1 = (uw4 >> 8) & 0xFFu;
        const uint ub2 = (uw4 >> 16) & 0xFFu;
        const uint ub3 = (uw4 >> 24) & 0xFFu;
        float u_dot = 0.0f;
        u_dot = fma(float(ub0 & 0x0Fu), e0, u_dot); u_dot = fma(float(ub0 >> 4), e1, u_dot);
        u_dot = fma(float(ub1 & 0x0Fu), e2, u_dot); u_dot = fma(float(ub1 >> 4), e3, u_dot);
        u_dot = fma(float(ub2 & 0x0Fu), e4, u_dot); u_dot = fma(float(ub2 >> 4), e5, u_dot);
        u_dot = fma(float(ub3 & 0x0Fu), e6, u_dot); u_dot = fma(float(ub3 >> 4), e7, u_dot);

        g_acc = fma(gs, g_dot, g_acc);
        g_acc = fma(gb, sum, g_acc);
        u_acc = fma(us, u_dot, u_acc);
        u_acc = fma(ub, sum, u_acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float gs = float(gS_row[g]);
        const float gb = float(gB_row[g]);
        const float us = float(uS_row[g]);
        const float ub = float(uB_row[g]);
        const uint8_t gbv = gW_row[g * (kLFM2GroupSize / 2) + lane];
        const uint8_t ubv = uW_row[g * (kLFM2GroupSize / 2) + lane];
        const float x0 = float(x[g * kLFM2GroupSize + lane * 2u]);
        const float x1 = float(x[g * kLFM2GroupSize + lane * 2u + 1u]);
        const float sum = x0 + x1;
        float g_dot = fma(float(uint(gbv & 0x0Fu)), x0, 0.0f);
        g_dot = fma(float(uint(gbv >> 4)), x1, g_dot);
        float u_dot = fma(float(uint(ubv & 0x0Fu)), x0, 0.0f);
        u_dot = fma(float(uint(ubv >> 4)), x1, u_dot);
        g_acc = fma(gs, g_dot, g_acc);
        g_acc = fma(gb, sum, g_acc);
        u_acc = fma(us, u_dot, u_acc);
        u_acc = fma(ub, sum, u_acc);
    }
    return float2(simd_sum(g_acc), simd_sum(u_acc));
}

// One SIMD group per (slot, intermediate) gate/up row pair.
kernel void lfm2_moe_phase1_top4(
    device const LFM2RoutedBlobs& routed [[buffer(0)]],
    constant LFM2ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= top_k * F) return;
    const uint slot = rowg / F;
    if (slot >= kLFM2MaxStreamedExperts) return;
    const uint f = rowg % F;

    device const uint8_t* base = routed.blob[slot];
    const LFM2ExpertOffsets re = routed_offsets;
    device const uint8_t* gW = base + re.gate_W_off;
    device const uint8_t* uW = base + re.up_W_off;
    device const bfloat* gS = (device const bfloat*)(base + re.gate_s_off);
    device const bfloat* uS = (device const bfloat*)(base + re.up_s_off);
    device const bfloat* gB = (device const bfloat*)(base + re.gate_b_off);
    device const bfloat* uB = (device const bfloat*)(base + re.up_b_off);

    const float2 gu = lfm2_int4_gate_up_rows_simd_dev_vec_u16load(
        gW, gS, gB, uW, uS, uB, x, f, D, lane);
    if (lane == 0) acts[slot * F + f] = half(lfm2_silu(gu.x) * gu.y);
}

// One threadgroup per output element. Four SIMD groups each own one routed
// slot's down-projection row; lane 0 of the first group combines the weighted
// partials with the residual in FP32.
kernel void lfm2_moe_phase2_top4(
    device const LFM2RoutedBlobs& routed [[buffer(0)]],
    constant LFM2ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* acts [[buffer(2)]],
    device const half* routing_w [[buffer(3)]],
    device const half* residual [[buffer(4)]],
    device half* y [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& F [[buffer(7)]],
    uint d [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[kLFM2MaxStreamedExperts];
    if (d >= D) return;

    device const uint8_t* base = routed.blob[sg_idx];
    const LFM2ExpertOffsets re = routed_offsets;
    device const uint8_t* dW = base + re.down_W_off;
    device const bfloat* dS = (device const bfloat*)(base + re.down_s_off);
    device const bfloat* dB = (device const bfloat*)(base + re.down_b_off);
    device const half* act_slot = acts + sg_idx * F;

    const float value = lfm2_int4_gemv_row_simd_dev_vec(
        dW, dS, dB, act_slot, d, F, lane);
    if (lane == 0) partial[sg_idx] = float(routing_w[sg_idx]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0 && lane == 0) {
        float acc = float(residual[d]);
        acc += partial[0]; acc += partial[1];
        acc += partial[2]; acc += partial[3];
        y[d] = half(acc);
    }
}
