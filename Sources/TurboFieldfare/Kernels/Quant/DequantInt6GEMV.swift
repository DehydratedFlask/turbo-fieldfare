import Metal

/// MLX-affine INT6 matrix-vector multiplication.
///
/// LFM2.5's tied embedding/lm-head is 6-bit affine: four values per three
/// bytes, least-significant bits first, with one BF16 scale/bias pair per group
/// of 64 columns. A resident `[rows, columns]` tensor stores
/// `columns * 6 / 32` packed `UInt32` columns per row, the same byte count as
/// `columns * 3 / 4`. The payload is byte-addressed — a 3-bytes-per-4-values
/// row stride is only 3-aligned — so no buffer alignment beyond Metal's is
/// required.
///
/// One threadgroup (a single 32-lane SIMD group) handles one output row, and
/// each lane decodes one packed triple, so no partial-row bookkeeping is needed.
final class DequantInt6GEMV {
    private let pipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pipeline = try context.pipeline(
            "dequant_int6_gemv_simd",
            constants: [],
            maxTotalThreadsPerThreadgroup: 32)
    }

    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer,
                weightsOffset: Int = 0,
                scales: MTLBuffer,
                scalesOffset: Int = 0,
                biases: MTLBuffer,
                biasesOffset: Int = 0,
                x: MTLBuffer,
                xOffset: Int = 0,
                y: MTLBuffer,
                yOffset: Int = 0,
                m: UInt32,
                n: UInt32) {
        precondition(n % UInt32(Quantization.groupSize) == 0,
                     "N must be a multiple of \(Quantization.groupSize)")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var mValue = m
        var nValue = n
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)

        let threadgroupSize = MTLSize(width: 32, height: 1, depth: 1)
        let threadgroupCount = MTLSize(width: Int(m), height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadgroupCount,
                                     threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
}
