import Metal

/// LFM2.5 dense (shared) feed-forward: `down(silu(gate(x)) * up(x))` with
/// INT4-affine weights.
///
/// Sibling of `SharedExpertInt4` with the only behavioural difference being the
/// gated activation: LFM2.5 uses SiLU (`x * sigmoid(x)`), not Gemma's
/// `gelu_pytorch_tanh`. Projections and scratch buffers follow the same layout
/// contract so callers can share allocation code.
final class LFM2SharedExpert {
    private let int4: DequantInt4GEMV
    private let siluMulPSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.int4 = try DequantInt4GEMV(context: context)
        self.siluMulPSO = try context.pipeline("lfm2_silu_mul_fp16")
    }

    func encode(commandBuffer cb: MTLCommandBuffer,
                x: MTLBuffer, xOffset: Int = 0,
                gate: SharedExpertProjection,
                up: SharedExpertProjection,
                down: SharedExpertProjection,
                y: MTLBuffer, yOffset: Int = 0,
                scratchGate: MTLBuffer, scratchGateOffset: Int = 0,
                scratchUp: MTLBuffer, scratchUpOffset: Int = 0,
                scratchAct: MTLBuffer, scratchActOffset: Int = 0) throws {
        guard gate.rows == up.rows, gate.cols == up.cols,
              down.rows == gate.cols, down.cols == gate.rows else {
            throw SharedExpertError.dimensionMismatch(
                "gate=(\(gate.rows),\(gate.cols)) up=(\(up.rows),\(up.cols)) down=(\(down.rows),\(down.cols))")
        }
        let intermediate = Int(gate.rows)
        let required = intermediate * MemoryLayout<Float16>.stride
        guard scratchGateOffset >= 0, scratchGateOffset + required <= scratchGate.length,
              scratchUpOffset >= 0, scratchUpOffset + required <= scratchUp.length,
              scratchActOffset >= 0, scratchActOffset + required <= scratchAct.length else {
            throw SharedExpertError.scratchTooSmall(
                "need \(required) bytes per intermediate buffer")
        }
        let inputBytes = Int(gate.cols) * MemoryLayout<Float16>.stride
        let outputBytes = Int(down.rows) * MemoryLayout<Float16>.stride
        guard xOffset >= 0, xOffset + inputBytes <= x.length else {
            throw SharedExpertError.scratchTooSmall("input range exceeds x buffer")
        }
        guard yOffset >= 0, yOffset + outputBytes <= y.length else {
            throw SharedExpertError.scratchTooSmall("output range exceeds y buffer")
        }

        int4.encode(commandBuffer: cb,
                    weights: gate.weights, weightsOffset: gate.weightsOffset,
                    scales: gate.scales, scalesOffset: gate.scalesOffset,
                    biases: gate.biases, biasesOffset: gate.biasesOffset,
                    x: x, xOffset: xOffset,
                    y: scratchGate, yOffset: scratchGateOffset,
                    m: gate.rows, n: gate.cols)
        int4.encode(commandBuffer: cb,
                    weights: up.weights, weightsOffset: up.weightsOffset,
                    scales: up.scales, scalesOffset: up.scalesOffset,
                    biases: up.biases, biasesOffset: up.biasesOffset,
                    x: x, xOffset: xOffset,
                    y: scratchUp, yOffset: scratchUpOffset,
                    m: up.rows, n: up.cols)

        guard let encoder = cb.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(siluMulPSO)
        encoder.setBuffer(scratchGate, offset: scratchGateOffset, index: 0)
        encoder.setBuffer(scratchUp, offset: scratchUpOffset, index: 1)
        encoder.setBuffer(scratchAct, offset: scratchActOffset, index: 2)
        var count = UInt32(intermediate)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
        let width = min(siluMulPSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(MTLSize(width: intermediate, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()

        int4.encode(commandBuffer: cb,
                    weights: down.weights, weightsOffset: down.weightsOffset,
                    scales: down.scales, scalesOffset: down.scalesOffset,
                    biases: down.biases, biasesOffset: down.biasesOffset,
                    x: scratchAct, xOffset: scratchActOffset,
                    y: y, yOffset: yOffset,
                    m: down.rows, n: down.cols)
    }
}
