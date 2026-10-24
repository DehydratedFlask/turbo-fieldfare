import Metal

/// LFM2.5 Q/K epilogue: per-head standard RMSNorm fused with full NeoX RoPE.
///
///   inv       = 1 / sqrt(mean(x^2) + eps)          // standard, no unit offset
///   angle(i)  = position * theta^(-2i / headDim)
///   out[i]         = x[i] * inv * w[i] * cos - x[i + half] * inv * w[i + half] * sin
///   out[i + half]  = x[i] * inv * w[i] * sin + x[i + half] * inv * w[i + half] * cos
///
/// `qWeight` / `kWeight` are BF16 `headDim` gain vectors (the installed LFM
/// `q_layernorm` / `k_layernorm` tensors); activations and outputs are FP16.
/// Q runs over `qHeads` heads and K over `kvHeads`. One dispatch covers both,
/// one threadgroup per head. There is no value norm.
///
/// LFM norm epsilon is `1e-5`; it is passed in explicitly rather than being
/// baked into the kernel so callers own the convention.
final class LFM2QKNormRoPE {

    private let pipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pipeline = try context.pipeline("lfm2_qk_norm_rope")
    }

    func encode(commandBuffer: MTLCommandBuffer,
                q: MTLBuffer, qOffset: Int = 0,
                k: MTLBuffer, kOffset: Int = 0,
                qWeight: MTLBuffer, qWeightOffset: Int = 0,
                kWeight: MTLBuffer, kWeightOffset: Int = 0,
                qOut: MTLBuffer, qOutOffset: Int = 0,
                kOut: MTLBuffer, kOutOffset: Int = 0,
                qHeads: UInt32, kvHeads: UInt32, headDim: UInt32,
                position: UInt32, theta: Float, eps: Float) {
        precondition(headDim.isMultiple(of: 2), "head_dim must be even")
        precondition(headDim > 0, "head_dim must be positive")
        let totalHeads = Int(qHeads) + Int(kvHeads)
        guard totalHeads > 0 else { return }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(q,       offset: qOffset,       index: 0)
        encoder.setBuffer(k,       offset: kOffset,       index: 1)
        encoder.setBuffer(qWeight, offset: qWeightOffset, index: 2)
        encoder.setBuffer(kWeight, offset: kWeightOffset, index: 3)
        encoder.setBuffer(qOut,    offset: qOutOffset,    index: 4)
        encoder.setBuffer(kOut,    offset: kOutOffset,    index: 5)
        var qHeadsValue = qHeads
        var kvHeadsValue = kvHeads
        var headDimValue = headDim
        var positionValue = position
        var thetaValue = theta
        var epsValue = eps
        encoder.setBytes(&qHeadsValue,  length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&kvHeadsValue, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.setBytes(&headDimValue, length: MemoryLayout<UInt32>.size, index: 8)
        encoder.setBytes(&positionValue, length: MemoryLayout<UInt32>.size, index: 9)
        encoder.setBytes(&thetaValue,   length: MemoryLayout<Float>.size,  index: 10)
        encoder.setBytes(&epsValue,     length: MemoryLayout<Float>.size,  index: 11)

        let width = min(Int(pipeline.maxTotalThreadsPerThreadgroup), 256)
        encoder.dispatchThreadgroups(
            MTLSize(width: totalHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
