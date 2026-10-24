#include <metal_stdlib>
using namespace metal;

constant uint FC_ROPE_HEAD_DIM [[function_constant(50)]];
constant uint FC_ROPE_NUM_HEADS [[function_constant(51)]];
constant uint FC_ROPE_ROTATED_PAIRS [[function_constant(52)]];
constant bool FC_ROPE_USE_FC [[function_constant(53)]];

static inline uint rope_head_dim(constant uint& runtime_value) {
    return (is_function_constant_defined(FC_ROPE_USE_FC) &&
            FC_ROPE_USE_FC &&
            is_function_constant_defined(FC_ROPE_HEAD_DIM))
        ? FC_ROPE_HEAD_DIM
        : runtime_value;
}
static inline uint rope_num_heads(constant uint& runtime_value) {
    return (is_function_constant_defined(FC_ROPE_USE_FC) &&
            FC_ROPE_USE_FC &&
            is_function_constant_defined(FC_ROPE_NUM_HEADS))
        ? FC_ROPE_NUM_HEADS
        : runtime_value;
}

static inline uint rope_rotated_pairs(constant uint& runtime_value) {
    return (is_function_constant_defined(FC_ROPE_USE_FC) &&
            FC_ROPE_USE_FC &&
            is_function_constant_defined(FC_ROPE_ROTATED_PAIRS))
        ? FC_ROPE_ROTATED_PAIRS
        : runtime_value;
}

static inline void apply_neox_pair(
    device half* head,
    uint pair,
    uint half_dim,
    uint frequency_divisor,
    float position,
    float theta
) {
    const float exponent = -float(2u * pair) / float(frequency_divisor);
    const float angle = position * pow(theta, exponent);
    const float cosine = cos(angle);
    const float sine = sin(angle);
    const uint lower = pair;
    const uint upper = half_dim + pair;
    const float x0 = float(head[lower]);
    const float x1 = float(head[upper]);
    head[lower] = half(x0 * cosine - x1 * sine);
    head[upper] = half(x0 * sine + x1 * cosine);
}

kernel void rope_default_neox(
    device half* data [[buffer(0)]],
    constant uint& position [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant float& theta [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint pair = gid.x;
    const uint head_index = gid.y;
    const uint token_index = gid.z;
    const uint dimension = rope_head_dim(head_dim);
    const uint heads = rope_num_heads(num_heads);
    const uint half_dimension = dimension / 2u;
    if (pair >= half_dimension || head_index >= heads) return;

    device half* head = data
        + token_index * heads * dimension
        + head_index * dimension;
    apply_neox_pair(head, pair, half_dimension, dimension,
                    float(position), theta);
}

// ============================================================================
// LFM2.5 Q/K epilogue: per-head standard RMSNorm (no unit offset, BF16 gain)
// fused with full NeoX rotate-half RoPE.
//
// Q uses `q_heads` heads and K uses `kv_heads`; both use `head_dim`. One
// threadgroup handles one head, so Q and K share a single dispatch: heads
// [0, q_heads) read Q, heads [q_heads, q_heads + kv_heads) read K.
//
// Per head:
//   inv      = rsqrt(mean(x^2) + eps)                      // eps = 1e-5 for LFM
//   x[i]    *= inv * weight[i]
//   angle(i) = position * theta^(-2i / head_dim)
//   out[i]          = x[i] * cos - x[i + half] * sin
//   out[i + half]   = x[i] * sin + x[i + half] * cos
//
// FP32 accumulation, FP16 output. `out` may alias `x`: each thread reads its
// whole pair before writing either element.
// ============================================================================

constant constexpr uint kLFM2QKMaxSimdGroups = 8;

// Same two-stage block reduction as rmsnorm.metal, kept local so rope.metal
// compiles on its own. Returns rsqrt(mean(x^2) + eps) broadcast to all threads.
static inline float lfm2_qk_block_inv(
    device const half* x,
    uint  D,
    float eps,
    uint  lid,
    uint  lsize,
    uint  simd_lane_id,
    uint  simd_group_id,
    uint  simdgroups,
    threadgroup float* partial
) {
    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        float v = float(x[i]);
        acc = fma(v, v, acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        v = simd_sum(v);
        if (simd_lane_id == 0) {
            float mean_sq = v / float(D);
            partial[0] = rsqrt(mean_sq + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void lfm2_qk_norm_rope(
    device const half*   q          [[buffer(0)]],   // [q_heads * head_dim] FP16
    device const half*   k          [[buffer(1)]],   // [kv_heads * head_dim] FP16
    device const bfloat* q_weight   [[buffer(2)]],   // [head_dim] BF16
    device const bfloat* k_weight   [[buffer(3)]],   // [head_dim] BF16
    device       half*   q_out      [[buffer(4)]],   // [q_heads * head_dim] FP16
    device       half*   k_out      [[buffer(5)]],   // [kv_heads * head_dim] FP16
    constant     uint&   q_heads    [[buffer(6)]],
    constant     uint&   kv_heads   [[buffer(7)]],
    constant     uint&   head_dim   [[buffer(8)]],
    constant     uint&   position   [[buffer(9)]],
    constant     float&  theta      [[buffer(10)]],
    constant     float&  eps        [[buffer(11)]],
    uint  head             [[threadgroup_position_in_grid]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    if (head >= q_heads + kv_heads) return;

    const uint dim = head_dim;
    const uint half_dim = dim / 2u;
    const bool is_q = head < q_heads;
    const uint idx = is_q ? head : head - q_heads;
    device const half* x = (is_q ? q : k) + idx * dim;
    device const bfloat* w = is_q ? q_weight : k_weight;
    device half* o = (is_q ? q_out : k_out) + idx * dim;

    threadgroup float partial[kLFM2QKMaxSimdGroups];
    const float inv = lfm2_qk_block_inv(x, dim, eps, lid, lsize,
                                        simd_lane_id, simd_group_id,
                                        simdgroups, partial);

    const float pos = float(position);
    for (uint i = lid; i < half_dim; i += lsize) {
        const float x0 = float(x[i]) * inv * float(w[i]);
        const float x1 = float(x[half_dim + i]) * inv * float(w[half_dim + i]);
        const float angle = pos * pow(theta, -2.0f * float(i) / float(dim));
        const float cosine = cos(angle);
        const float sine = sin(angle);
        o[i] = half(x0 * cosine - x1 * sine);
        o[half_dim + i] = half(x0 * sine + x1 * cosine);
    }
}

kernel void rope_proportional_neox(
    device half* data [[buffer(0)]],
    constant uint& position [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant float& theta [[buffer(4)]],
    constant uint& rotated_pairs [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint pair = gid.x;
    const uint head_index = gid.y;
    const uint token_index = gid.z;
    const uint dimension = rope_head_dim(head_dim);
    const uint heads = rope_num_heads(num_heads);
    const uint active_pairs = rope_rotated_pairs(rotated_pairs);
    if (pair >= active_pairs || head_index >= heads) return;

    const uint half_dimension = dimension / 2u;
    device half* head = data
        + token_index * heads * dimension
        + head_index * dimension;
    apply_neox_pair(head, pair, half_dimension, dimension,
                    float(position), theta);
}
