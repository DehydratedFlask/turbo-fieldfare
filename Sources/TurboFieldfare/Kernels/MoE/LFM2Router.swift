import Metal

/// LFM2.5 mixture-of-experts router.
///
/// Production shape is fixed by the installed LFM2.5-8B-A1B pack: 64 experts,
/// hidden 2048. The router projection is FP32 (the pack stores
/// `feed_forward.gate.weight` and `feed_forward.expert_bias` as FP32), while
/// the hidden state is FP16.
///
/// Selection scores `sigmoid(logit) + expert_bias`; the expert bias only
/// influences which experts are chosen. Emitted routing weights are the
/// unbiased sigmoid values of the selected experts, normalized with
/// `sum + 1e-6`, then multiplied by `routedScalingFactor`. Selection ties break
/// toward the lower expert index.
final class LFM2Router {
    static let numExperts: UInt32 = 64
    static let hiddenDimension: UInt32 = 2048
    static let topK: UInt32 = 4

    private let gemvPSO: MTLComputePipelineState
    private let selectPSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.gemvPSO = try context.pipeline(
            "lfm2_router_gemv_f32",
            constants: [],
            maxTotalThreadsPerThreadgroup: 256)
        self.selectPSO = try context.pipeline("lfm2_router_select_top4")
    }

    /// `logits[e] = sum_i W[e, i] * x[i]`, with FP32 accumulation.
    func encodeLogits(commandBuffer: MTLCommandBuffer,
                      weights: MTLBuffer, weightsOffset: Int = 0,
                      x: MTLBuffer, xOffset: Int = 0,
                      logits: MTLBuffer, logitsOffset: Int = 0) {
        let experts = Int(Self.numExperts)
        let dimension = Int(Self.hiddenDimension)
        let weightBytes = experts * dimension * MemoryLayout<Float>.stride
        let inputBytes = dimension * MemoryLayout<Float16>.stride
        let outputBytes = experts * MemoryLayout<Float>.stride
        precondition(weightsOffset >= 0 && weightsOffset + weightBytes <= weights.length,
                     "router weights range exceeds buffer")
        precondition(xOffset >= 0 && xOffset + inputBytes <= x.length,
                     "router input range exceeds buffer")
        precondition(logitsOffset >= 0 && logitsOffset + outputBytes <= logits.length,
                     "router logits range exceeds buffer")

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        var numExperts = Self.numExperts
        var d = Self.hiddenDimension
        encoder.setComputePipelineState(gemvPSO)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(x, offset: xOffset, index: 1)
        encoder.setBuffer(logits, offset: logitsOffset, index: 2)
        encoder.setBytes(&numExperts, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&d, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: (experts + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Selects the top `topK` experts and writes FP16 normalized weights.
    func encodeSelect(commandBuffer: MTLCommandBuffer,
                      logits: MTLBuffer, logitsOffset: Int = 0,
                      bias: MTLBuffer, biasOffset: Int = 0,
                      indices: MTLBuffer, indicesOffset: Int = 0,
                      weights: MTLBuffer, weightsOffset: Int = 0,
                      topK: UInt32, routedScalingFactor: Float) {
        precondition(topK == Self.topK,
                     "LFM2 top-4 router requires topK == 4, got \(topK)")
        let experts = Int(Self.numExperts)
        let logitBytes = experts * MemoryLayout<Float>.stride
        let indexBytes = Int(topK) * MemoryLayout<UInt32>.stride
        let weightBytes = Int(topK) * MemoryLayout<Float16>.stride
        precondition(logitsOffset >= 0 && logitsOffset + logitBytes <= logits.length,
                     "router logits range exceeds buffer")
        precondition(biasOffset >= 0 && biasOffset + logitBytes <= bias.length,
                     "router bias range exceeds buffer")
        precondition(indicesOffset >= 0 && indicesOffset + indexBytes <= indices.length,
                     "router index range exceeds buffer")
        precondition(weightsOffset >= 0 && weightsOffset + weightBytes <= weights.length,
                     "router weight range exceeds buffer")

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        var numExperts = Self.numExperts
        var k = topK
        var scale = routedScalingFactor
        encoder.setComputePipelineState(selectPSO)
        encoder.setBuffer(logits, offset: logitsOffset, index: 0)
        encoder.setBuffer(bias, offset: biasOffset, index: 1)
        encoder.setBuffer(indices, offset: indicesOffset, index: 2)
        encoder.setBuffer(weights, offset: weightsOffset, index: 3)
        encoder.setBytes(&numExperts, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&k, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&scale, length: MemoryLayout<Float>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
