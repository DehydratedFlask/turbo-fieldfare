import Metal

/// LFM2.5 prefill route selection.
///
/// One threadgroup per token computes FP32 router logits, then thread 0 applies
/// the decode router's contract: selection scores `sigmoid(logit) + bias`,
/// emitted weights are the *unbiased* sigmoid values normalized with
/// `sum + 1e-6` and scaled by `routedScalingFactor`. The Gemma prefill router
/// applies a different (softmax/gain) contract and unpacks INT4 router
/// weights, so it cannot be reused.
final class LFM2PrefillRouter {
    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline(
            "prefill_lfm2_router_block",
            constants: [],
            maxTotalThreadsPerThreadgroup: 256)
    }

    func encodeBlock(commandBuffer: MTLCommandBuffer,
                     weights: MTLBuffer, weightsOffset: Int = 0,
                     bias: MTLBuffer, biasOffset: Int = 0,
                     hidden: MTLBuffer, hiddenOffset: Int = 0,
                     outIndices: MTLBuffer, outIndicesOffset: Int = 0,
                     outWeights: MTLBuffer, outWeightsOffset: Int = 0,
                     queryCount: UInt32,
                     numExperts: UInt32,
                     d: UInt32,
                     topK: UInt32,
                     hiddenStrideElements: UInt32,
                     routedScalingFactor: Float) {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(numExperts > 0 && numExperts <= 256,
                     "numExperts must be in 1...256")
        precondition(topK > 0 && topK <= 64, "topK must be in 1...64")
        precondition(hiddenStrideElements >= d, "hidden stride is too small")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pso)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(bias, offset: biasOffset, index: 1)
        encoder.setBuffer(hidden, offset: hiddenOffset, index: 2)
        encoder.setBuffer(outIndices, offset: outIndicesOffset, index: 3)
        encoder.setBuffer(outWeights, offset: outWeightsOffset, index: 4)
        var queryCountValue = queryCount
        var numExpertsValue = numExperts
        var dValue = d
        var topKValue = topK
        var strideValue = hiddenStrideElements
        var scaleValue = routedScalingFactor
        encoder.setBytes(&queryCountValue, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&numExpertsValue, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&dValue, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&topKValue, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.setBytes(&strideValue, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.setBytes(&scaleValue, length: MemoryLayout<Float>.stride, index: 10)
        let width = min(max(Int(numExperts), 32),
                        Int(pso.maxTotalThreadsPerThreadgroup))
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(queryCount), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
