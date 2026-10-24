import Metal

/// LFM2.5 depthwise causal short convolution, kernel size 3, no bias.
///
/// `Bx` is the caller's gated value `B * x`. The state is the two preceding
/// `Bx` rows, oldest first, and the output is
///
///   conv = w0 * Bx[t-2] + w1 * Bx[t-1] + w2 * Bx[t]
///
/// with weight layout `[channels, 3]`, `weight[channel * 3 + tap]`, FP16. The
/// installed pack stores these taps BF16; convert them once at load to FP16.
/// Binding the pack's BF16 tensor to this `half*` slot is not diagnosed and
/// yields plausible-looking wrong numbers. The cache stores gated values, not
/// conv outputs, and there is no activation inside these kernels. Task 15's
/// runner owns the rewind ring; this type only owns one two-row state.
final class ShortConv {

    private let decodePipeline: MTLComputePipelineState
    private let prefillPipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.decodePipeline = try context.pipeline("lfm2_shortconv_decode")
        self.prefillPipeline = try context.pipeline("lfm2_shortconv_prefill")
    }

    /// One token. Reads both old state rows, writes the output, then shifts the
    /// state in place so it holds `[Bx[t-1], Bx[t]]` afterwards.
    func encodeDecode(commandBuffer: MTLCommandBuffer,
                      bx: MTLBuffer, bxOffset: Int = 0,
                      weight: MTLBuffer, weightOffset: Int = 0,
                      state: MTLBuffer, stateOffset: Int = 0,
                      out: MTLBuffer, outOffset: Int = 0,
                      channels: UInt32) {
        guard channels > 0 else { return }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(decodePipeline)
        encoder.setBuffer(bx,     offset: bxOffset,     index: 0)
        encoder.setBuffer(weight, offset: weightOffset, index: 1)
        encoder.setBuffer(state,  offset: stateOffset,  index: 2)
        encoder.setBuffer(out,    offset: outOffset,    index: 3)
        var channelsValue = channels
        encoder.setBytes(&channelsValue, length: MemoryLayout<UInt32>.size, index: 4)

        let width = min(Int(decodePipeline.maxTotalThreadsPerThreadgroup), 256)
        encoder.dispatchThreads(
            MTLSize(width: Int(channels), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// `tokens` positions at once. `stateIn[0]` is `Bx[t-2]` and `stateIn[1]` is
    /// `Bx[t-1]` for the first chunk position; `stateOut` receives the chunk's
    /// last two gated rows, oldest first. `stateIn` is read-only. `stateIn` and
    /// `stateOut` must not alias: the `t == 0` `stateOut` write races the
    /// `t == 1` `stateIn` read, so pass a distinct buffer (or non-overlapping
    /// offsets) and copy `stateOut` into the live state afterwards.
    func encodePrefill(commandBuffer: MTLCommandBuffer,
                       bx: MTLBuffer, bxOffset: Int = 0,
                       weight: MTLBuffer, weightOffset: Int = 0,
                       stateIn: MTLBuffer, stateInOffset: Int = 0,
                       out: MTLBuffer, outOffset: Int = 0,
                       stateOut: MTLBuffer, stateOutOffset: Int = 0,
                       tokens: UInt32, channels: UInt32) {
        guard tokens > 0, channels > 0 else { return }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(prefillPipeline)
        encoder.setBuffer(bx,       offset: bxOffset,       index: 0)
        encoder.setBuffer(weight,   offset: weightOffset,   index: 1)
        encoder.setBuffer(stateIn,  offset: stateInOffset,  index: 2)
        encoder.setBuffer(out,      offset: outOffset,      index: 3)
        encoder.setBuffer(stateOut, offset: stateOutOffset, index: 4)
        var tokensValue = tokens
        var channelsValue = channels
        encoder.setBytes(&tokensValue,   length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&channelsValue, length: MemoryLayout<UInt32>.size, index: 6)

        encoder.dispatchThreads(
            MTLSize(width: Int(channels), height: Int(tokens), depth: 1),
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encoder.endEncoding()
    }
}
