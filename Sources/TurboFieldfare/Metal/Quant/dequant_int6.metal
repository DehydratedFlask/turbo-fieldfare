#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_int6 — MLX `affine` 6-bit dequant (LFM2.5's tied embedding/lm-head).
//
// Layout (per row of length N):
//   W       : N*3/4 bytes. Four 6-bit values per three bytes, least-significant
//             bits first. Triple k holds components 4k..4k+3:
//               v0 = b0 & 0x3f
//               v1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0f) << 2)
//               v2 = ((b1 >> 4) & 0x0f) | ((b2 & 0x03) << 4)
//               v3 = (b2 >> 2) & 0x3f
//   scales  : N/64 BF16, one per group of 64.
//   biases  : N/64 BF16, one per group of 64.
//   value   : w[i] = float(v[i]) * scale[i/64] + bias[i/64].
//
// A row stride of N*3/4 is a multiple of 3, not necessarily of 2 or 4, so every
// weight access here is a single byte load: the payload is byte-addressed.
// Requires N % 64 == 0, validated at the wrapper.
// ============================================================================

constant constexpr uint kInt6GroupSize = 64;
constant constexpr uint kInt6RowNumerator = 3u;   // bytes per four values
constant constexpr uint kInt6RowDenominator = 4u; // values per triple

struct int6_triple {
    uint v0;
    uint v1;
    uint v2;
    uint v3;
};

/// Decodes one packed triple (three bytes) into four 6-bit values.
static inline int6_triple int6_decode(uint b0, uint b1, uint b2) {
    int6_triple t;
    t.v0 = b0 & 0x3f;
    t.v1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0f) << 2);
    t.v2 = ((b1 >> 4) & 0x0f) | ((b2 & 0x03) << 4);
    t.v3 = (b2 >> 2) & 0x3f;
    return t;
}

/// One thread per output element; decodes the selected row only.
kernel void embed_lookup_int6(
    device const uint8_t* table     [[buffer(0)]],   // [V, D*3/4] packed 6-bit
    device const bfloat*  scales    [[buffer(1)]],   // [V, D/64] BF16
    device const bfloat*  biases    [[buffer(2)]],   // [V, D/64] BF16
    device half*          out       [[buffer(3)]],   // [D] FP16
    constant uint&        token_id  [[buffer(4)]],
    constant uint&        D         [[buffer(5)]],
    constant float&       out_scale [[buffer(6)]],   // pass 1.0 to disable
    uint                  gid       [[thread_position_in_grid]]
) {
    if (gid >= D) return;
    const uint groups_per_row = D / kInt6GroupSize;
    const uint row_bytes = D / kInt6RowDenominator * kInt6RowNumerator;
    device const uint8_t* row_q = table  + uint(token_id) * row_bytes;
    device const bfloat*  row_s = scales + uint(token_id) * groups_per_row;
    device const bfloat*  row_b = biases + uint(token_id) * groups_per_row;

    const uint triple = gid >> 2;
    const uint slot   = gid & 3u;
    device const uint8_t* p = row_q + triple * kInt6RowNumerator;
    const int6_triple t = int6_decode(uint(p[0]), uint(p[1]), uint(p[2]));
    const uint v = slot == 0u ? t.v0
                 : slot == 1u ? t.v1
                 : slot == 2u ? t.v2
                 :              t.v3;
    const float s = float(row_s[gid / kInt6GroupSize]);
    const float b = float(row_b[gid / kInt6GroupSize]);
    out[gid] = half((float(v) * s + b) * out_scale);
}

// y[m] = sum_{n} W[m, n] * x[n]. One threadgroup per output row: a single
// 32-lane SIMD group decodes one 3-byte triple (four values) per lane, so the
// 16 lanes of a half-warp cover one 64-element group and both halves cover
// two consecutive groups per step. Each lane keeps its own group's scale and
// bias, so the per-group affine dequant stays inside the lane's dot; the
// `simd_sum` at the end is across groups, and every lane contributes the same
// weighted terms regardless of grouping. Odd group counts take a half-warp
// tail so an N of 64 (one group) still produces the full row sum.
kernel void dequant_int6_gemv_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],
    device half*          y      [[buffer(4)]],
    constant uint&        M      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx;
    if (row >= M) return;
    const uint n_groups  = N / kInt6GroupSize;
    const uint row_bytes = N / kInt6RowDenominator * kInt6RowNumerator;
    const uint group_bytes = kInt6GroupSize / kInt6RowDenominator * kInt6RowNumerator;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;

    const uint half_warp = lane >> 4;   // which of the two groups in a step
    const uint slot      = lane & 15u;  // triple within that group

    float acc = 0.0f;
    const uint full_steps = n_groups >> 1;
    for (uint step = 0; step < full_steps; ++step) {
        const uint g = step * 2u + half_warp;
        device const uint8_t* p = W_row + g * group_bytes + slot * kInt6RowNumerator;
        const int6_triple t = int6_decode(uint(p[0]), uint(p[1]), uint(p[2]));
        const uint base = g * kInt6GroupSize + slot * 4u;
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        float dot = 0.0f;
        dot = fma(float(t.v0) * s + b, float(x[base]),      dot);
        dot = fma(float(t.v1) * s + b, float(x[base + 1u]), dot);
        dot = fma(float(t.v2) * s + b, float(x[base + 2u]), dot);
        dot = fma(float(t.v3) * s + b, float(x[base + 3u]), dot);
        acc += dot;
    }
    if ((n_groups & 1u) != 0u && half_warp == 0u) {
        const uint g = n_groups - 1u;
        device const uint8_t* p = W_row + g * group_bytes + slot * kInt6RowNumerator;
        const int6_triple t = int6_decode(uint(p[0]), uint(p[1]), uint(p[2]));
        const uint base = g * kInt6GroupSize + slot * 4u;
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        float dot = 0.0f;
        dot = fma(float(t.v0) * s + b, float(x[base]),      dot);
        dot = fma(float(t.v1) * s + b, float(x[base + 1u]), dot);
        dot = fma(float(t.v2) * s + b, float(x[base + 2u]), dot);
        dot = fma(float(t.v3) * s + b, float(x[base + 3u]), dot);
        acc += dot;
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        y[row] = half(acc);
    }
}
