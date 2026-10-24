import Metal

/// Elementwise helpers for the LFM2.5 short-convolution layer and residual
/// stream. They are kept out of `ShortConv` because that type owns only the
/// two-row convolution state, and out of the Gemma fusion kernels because the
/// LFM graph composes them directly per chunk.
final class LFM2ConvOps {
    private let bxPSO: MTLComputePipelineState
    private let gatePSO: MTLComputePipelineState
    private let residualAddPSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.bxPSO = try context.pipeline("lfm2_conv_bx_mul_fp16")
        self.gatePSO = try context.pipeline("lfm2_conv_gate_mul_fp16")
        self.residualAddPSO = try context.pipeline("lfm2_residual_add_fp16")
    }

    /// `bx[token, c] = proj[token, c] * proj[token, 2 * channels + c]`, where
    /// `proj` is the `[B | C | x]` output of the convolution's `in_proj`.
    func encodeBxMul(commandBuffer: MTLCommandBuffer,
                     proj: MTLBuffer, projOffset: Int = 0,
                     bx: MTLBuffer, bxOffset: Int = 0,
                     tokens: Int, channels: Int) {
        guard tokens > 0, channels > 0 else { return }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(bxPSO)
        encoder.setBuffer(proj, offset: projOffset, index: 0)
        encoder.setBuffer(bx, offset: bxOffset, index: 1)
        var tokensValue = UInt32(tokens)
        var channelsValue = UInt32(channels)
        var strideValue = UInt32(3 * channels)
        encoder.setBytes(&tokensValue, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&channelsValue, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&strideValue, length: MemoryLayout<UInt32>.stride, index: 4)
        dispatch(encoder, pso: bxPSO, tokens: tokens, channels: channels)
    }

    /// `out[token, c] = proj[token, channels + c] * conv[token, c]`.
    func encodeGateMul(commandBuffer: MTLCommandBuffer,
                       proj: MTLBuffer, projOffset: Int = 0,
                       conv: MTLBuffer, convOffset: Int = 0,
                       out: MTLBuffer, outOffset: Int = 0,
                       tokens: Int, channels: Int) {
        guard tokens > 0, channels > 0 else { return }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(gatePSO)
        encoder.setBuffer(proj, offset: projOffset, index: 0)
        encoder.setBuffer(conv, offset: convOffset, index: 1)
        encoder.setBuffer(out, offset: outOffset, index: 2)
        var tokensValue = UInt32(tokens)
        var channelsValue = UInt32(channels)
        var strideValue = UInt32(3 * channels)
        encoder.setBytes(&tokensValue, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&channelsValue, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&strideValue, length: MemoryLayout<UInt32>.stride, index: 5)
        dispatch(encoder, pso: gatePSO, tokens: tokens, channels: channels)
    }

    func encodeResidualAdd(commandBuffer: MTLCommandBuffer,
                           hidden: MTLBuffer, hiddenOffset: Int = 0,
                           branch: MTLBuffer, branchOffset: Int = 0,
                           count: Int) {
        guard count > 0 else { return }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(residualAddPSO)
        encoder.setBuffer(hidden, offset: hiddenOffset, index: 0)
        encoder.setBuffer(branch, offset: branchOffset, index: 1)
        var countValue = UInt32(count)
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 2)
        let width = min(Int(residualAddPSO.maxTotalThreadsPerThreadgroup), 256)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width,
                                                               height: 1,
                                                               depth: 1))
        encoder.endEncoding()
    }

    private func dispatch(_ encoder: MTLComputeCommandEncoder,
                          pso: MTLComputePipelineState,
                          tokens: Int,
                          channels: Int) {
        let width = min(Int(pso.maxTotalThreadsPerThreadgroup), 16)
        encoder.dispatchThreads(MTLSize(width: channels, height: tokens, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width,
                                                               height: 16,
                                                               depth: 1))
        encoder.endEncoding()
    }
}
