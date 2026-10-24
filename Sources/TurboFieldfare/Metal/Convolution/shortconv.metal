#include <metal_stdlib>
using namespace metal;

// ============================================================================
// shortconv — LFM2.5 depthwise causal short convolution, kernel size 3.
//
//   Bx   = B * x                      (computed by the caller)
//   conv = w0 * Bx[t-2] + w1 * Bx[t-1] + w2 * Bx[t]
//
// Layout:
//   bx      : [tokens, channels] row-major, FP16.
//   weight  : [channels, 3] FP16, `weight[channel * 3 + tap]`; tap 0 is the
//             oldest position, tap 2 the current one. No bias. The installed
//             pack stores these taps BF16; convert them once at load to FP16.
//             Binding the pack's BF16 tensor here is not diagnosed and yields
//             plausible-looking wrong numbers.
//   state   : [2, channels] FP16, oldest first. Decode shifts it in place; the
//             prefill variant reads `stateIn` and writes `stateOut` so callers
//             can keep replay history. `stateIn` and `stateOut` must not alias
//             in prefill: the `t == 0` stateOut write would race the `t == 1`
//             stateIn read. Both rows hold gated values `Bx`, not conv outputs.
//
// There is no activation and no cross-channel mixing: every channel is an
// independent 3-tap filter.
// ============================================================================

// One thread per channel. Each thread reads both state rows before writing the
// shifted pair, so the in-place update is race-free channel by channel.
kernel void lfm2_shortconv_decode(
    device const half* bx       [[buffer(0)]],   // [channels]
    device const half* weight   [[buffer(1)]],   // [channels, 3]
    device half*       state    [[buffer(2)]],   // [2, channels], oldest first
    device half*       out      [[buffer(3)]],   // [channels]
    constant uint&     channels [[buffer(4)]],
    uint               c        [[thread_position_in_grid]]
) {
    if (c >= channels) return;

    const half oldest   = state[c];
    const half previous = state[channels + c];
    const half current  = bx[c];

    const float w0 = float(weight[c * 3u]);
    const float w1 = float(weight[c * 3u + 1u]);
    const float w2 = float(weight[c * 3u + 2u]);
    out[c] = half(w0 * float(oldest) + w1 * float(previous) + w2 * float(current));

    state[c]            = previous;
    state[channels + c] = current;
}

// One thread per (token, channel). Causal taps read earlier rows of the chunk,
// except at the first two positions, where the incoming state supplies the
// left context. `stateOut` receives the last two gated rows of the chunk.
kernel void lfm2_shortconv_prefill(
    device const half* bx       [[buffer(0)]],   // [tokens, channels]
    device const half* weight   [[buffer(1)]],   // [channels, 3]
    device const half* stateIn  [[buffer(2)]],   // [2, channels], oldest first
    device half*       out      [[buffer(3)]],   // [tokens, channels]
    device half*       stateOut [[buffer(4)]],   // [2, channels], oldest first
    constant uint&     tokens   [[buffer(5)]],
    constant uint&     channels [[buffer(6)]],
    uint2              gid      [[thread_position_in_grid]]
) {
    const uint t = gid.y;
    const uint c = gid.x;
    if (t >= tokens || c >= channels) return;

    const uint stride = channels;
    const half current  = bx[t * stride + c];
    const half previous = (t >= 1u) ? bx[(t - 1u) * stride + c]
                                    : stateIn[stride + c];
    // t == 0: oldest is the row two positions before the chunk.
    // t == 1: oldest is the row one position before the chunk.
    const half oldest   = (t >= 2u) ? bx[(t - 2u) * stride + c]
                        : (t == 1u) ? stateIn[stride + c]
                                    : stateIn[c];

    const float w0 = float(weight[c * 3u]);
    const float w1 = float(weight[c * 3u + 1u]);
    const float w2 = float(weight[c * 3u + 2u]);
    out[t * stride + c] = half(w0 * float(oldest)
                             + w1 * float(previous)
                             + w2 * float(current));

    // A single thread per channel publishes the new two-row state so the
    // write is not repeated `tokens` times.
    if (t == 0u) {
        stateOut[c] = (tokens >= 2u) ? bx[(tokens - 2u) * stride + c]
                                     : stateIn[stride + c];
        stateOut[stride + c] = bx[(tokens - 1u) * stride + c];
    }
}

// The `in_proj` GEMV writes `[B | C | x]` per row, each `channels` wide. The
// two helpers below extract the gated products without materialising the
// three sub-vectors: `Bx = B * x` before the convolution and `C * conv` after
// it. `proj_stride` is `3 * channels`.
kernel void lfm2_conv_bx_mul_fp16(
    device const half* proj        [[buffer(0)]],   // [tokens, 3, channels]
    device half*       bx          [[buffer(1)]],   // [tokens, channels]
    constant uint&     tokens      [[buffer(2)]],
    constant uint&     channels    [[buffer(3)]],
    constant uint&     proj_stride [[buffer(4)]],
    uint2              gid         [[thread_position_in_grid]]
) {
    const uint c = gid.x;
    const uint t = gid.y;
    if (t >= tokens || c >= channels) return;
    const uint base = t * proj_stride;
    bx[t * channels + c] = half(float(proj[base + c]) * float(proj[base + 2u * channels + c]));
}

kernel void lfm2_conv_gate_mul_fp16(
    device const half* proj        [[buffer(0)]],   // [tokens, 3, channels]
    device const half* conv        [[buffer(1)]],   // [tokens, channels]
    device half*       out         [[buffer(2)]],   // [tokens, channels]
    constant uint&     tokens      [[buffer(3)]],
    constant uint&     channels    [[buffer(4)]],
    constant uint&     proj_stride [[buffer(5)]],
    uint2              gid         [[thread_position_in_grid]]
) {
    const uint c = gid.x;
    const uint t = gid.y;
    if (t >= tokens || c >= channels) return;
    const uint base = t * proj_stride;
    out[t * channels + c] = half(float(proj[base + channels + c]) * float(conv[t * channels + c]));
}

// Residual accumulate: `hidden += branch`. Kept beside the convolution ops
// because the LFM graph uses it between the operator, dense/MoE branch, and
// the next layer's norm.
kernel void lfm2_residual_add_fp16(
    device half*       hidden [[buffer(0)]],
    device const half* branch [[buffer(1)]],
    constant uint&     count  [[buffer(2)]],
    uint               tid    [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    hidden[tid] = half(float(hidden[tid]) + float(branch[tid]));
}
