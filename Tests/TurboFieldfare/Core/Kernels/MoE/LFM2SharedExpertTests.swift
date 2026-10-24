import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// LFM2.5 dense (shared) feed-forward: `down(silu(gate(x)) * up(x))` with
/// INT4-affine weights. The kernel is a sibling of `SharedExpertInt4` that
/// swaps GELU for SiLU; this suite pins SiLU against an independent CPU path.
///
/// LFM2 dense FFNs in the brief are 6 x 8 / 8 x 6, smaller than an affine group
/// (64). Helpers therefore embed the logical matrix in the top-left corner of a
/// 64 x 64 affine block and zero-fill the rest: the extra rows/columns are
/// multiplied by zeros, so the first `hidden` outputs equal the logical FFN.
@Suite struct LFM2SharedExpertTests {

    private static let groupWidth = Quantization.groupSize

    @Test func denseSiLUGatedFFNMatchesCPU() throws {
        let (gate, up, down) = makeSmallDenseFFN(rows: 6, hidden: 8, seed: 7)
        let x: [Float16] = (0..<8).map { Float16(Double($0) / 8) }
        let y = try runLFM2SharedExpert(gate, up, down, x)
        let expected = cpuSiluFFN(gate, up, down, x)
        for i in 0..<8 { #expect(abs(y[i] - expected[i]) < 0.02, "index \(i)") }
    }

    /// A rejected GELU implementation would disagree noticeably on this input,
    /// where the gate projection summed negative/positive values.
    @Test func denseActivationIsSiluNotGelu() throws {
        let hidden = 8
        let rows = 6
        // gate = identity-ish on the first six inputs; up = identity; down =
        // first six rows of the identity. Then y[i] = silu(x[i]) for i < 6.
        var gate = [[Float]](repeating: [Float](repeating: 0, count: hidden),
                             count: rows)
        var up = gate
        var down = [[Float]](repeating: [Float](repeating: 0, count: rows),
                             count: hidden)
        for i in 0..<rows {
            gate[i][i] = 1
            up[i][i] = 1
            down[i][i] = 1
        }
        let x: [Float16] = (0..<hidden).map { Float16(Double($0) / 8) }
        let y = try runLFM2SharedExpert(gate, up, down, x)
        // x[0] is zero, where SiLU and GELU agree; the remaining inputs are
        // strictly positive and separate the two activations.
        for i in 1..<rows {
            let xv = Float(x[i])
            let silu = xv / (1 + exp(-xv))
            let gelu = 0.5 * xv * (1 + tanh(0.7978845608028654
                * (xv + 0.044715 * xv * xv * xv)))
            // `up` is the identity too, so the routed value is silu(x) * x.
            #expect(abs(y[i] - silu * xv) < 0.02, "index \(i)")
            // At x = 0.125 the two activations are within 2e-4; larger inputs
            // separate them cleanly.
            if i >= 2 {
                #expect(abs(y[i] - gelu * xv) > 1e-3, "gelu would match index \(i)")
            }
        }
    }

    // MARK: - CPU reference

    private func makeSmallDenseFFN(rows: Int,
                                   hidden: Int,
                                   seed: UInt64) -> ([[Float]], [[Float]], [[Float]]) {
        var rng = SplitMix64(seed: seed)
        func matrix(_ r: Int, _ c: Int) -> [[Float]] {
            (0..<r).map { _ in (0..<c).map { _ in rng.uniform(-0.4, 0.4) } }
        }
        return (matrix(rows, hidden), matrix(rows, hidden), matrix(hidden, rows))
    }

    private func cpuSiluFFN(_ gate: [[Float]],
                            _ up: [[Float]],
                            _ down: [[Float]],
                            _ x: [Float16]) -> [Float] {
        let width = Self.groupWidth
        let xPadded = Self.paddedVector(x)
        let gateOutput = DequantInt4GemvRef.apply(
            weightRows: Self.paddedRows(gate), x: xPadded, n: width)
        let upOutput = DequantInt4GemvRef.apply(
            weightRows: Self.paddedRows(up), x: xPadded, n: width)
        let acts = zip(gateOutput, upOutput).map { gateValue, upValue in
            Float(Float16(Self.silu(gateValue) * upValue))
        }
        let output = DequantInt4GemvRef.apply(
            weightRows: Self.paddedRows(down), x: acts, n: width)
        return Array(output.prefix(x.count))
    }

    /// Runs the production kernel over the padded 64 x 64 block.
    private func runLFM2SharedExpert(_ gate: [[Float]],
                                     _ up: [[Float]],
                                     _ down: [[Float]],
                                     _ x: [Float16]) throws -> [Float] {
        let width = Self.groupWidth
        let context = try MetalContext()
        let kernel = try LFM2SharedExpert(context: context)
        let xBuffer = try #require(Fp16Buffer.make(
            context.device, halves: Self.paddedVector(x).map { Float16($0) }))
        let yBuffer = try #require(Fp16Buffer.make(context.device, count: width))
        let gateScratch = try #require(Fp16Buffer.make(context.device, count: width))
        let upScratch = try #require(Fp16Buffer.make(context.device, count: width))
        let actScratch = try #require(Fp16Buffer.make(context.device, count: width))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        try kernel.encode(
            commandBuffer: commandBuffer,
            x: xBuffer,
            gate: Self.projection(context, rows: Self.paddedRows(gate)),
            up: Self.projection(context, rows: Self.paddedRows(up)),
            down: Self.projection(context, rows: Self.paddedRows(down)),
            y: yBuffer,
            scratchGate: gateScratch,
            scratchUp: upScratch,
            scratchAct: actScratch)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        return Array(Fp16Buffer.read(yBuffer, count: width).prefix(x.count))
    }

    // MARK: - Padding / buffer helpers

    private static func paddedVector(_ x: [Float16]) -> [Float] {
        precondition(x.count <= groupWidth)
        var values = [Float](repeating: 0, count: groupWidth)
        for i in x.indices { values[i] = Float(x[i]) }
        return values
    }

    /// Zero-pads the logical matrix into the top-left of a 64 x 64 affine block.
    private static func paddedRows(_ rows: [[Float]]) -> [Quantization.Int4AffineRow] {
        let columns = rows.first?.count ?? 0
        precondition(rows.count <= groupWidth && columns <= groupWidth)
        return (0..<groupWidth).map { row in
            var values = [Float](repeating: 0, count: groupWidth)
            if row < rows.count {
                for column in 0..<columns { values[column] = rows[row][column] }
            }
            return Quantization.quantizeInt4Affine(values)
        }
    }

    private static func projection(
        _ context: MetalContext,
        rows: [Quantization.Int4AffineRow]
    ) -> SharedExpertProjection {
        let packed = rows.flatMap(\.packed)
        let scales = rows.flatMap(\.scales)
        let biases = rows.flatMap(\.biases)
        return SharedExpertProjection(
            weights: context.device.makeBuffer(
                bytes: packed, length: packed.count,
                options: .storageModeShared)!,
            scales: context.device.makeBuffer(
                bytes: scales, length: scales.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared)!,
            biases: context.device.makeBuffer(
                bytes: biases, length: biases.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared)!,
            rows: UInt32(groupWidth),
            cols: UInt32(groupWidth))
    }

    private static func silu(_ x: Float) -> Float {
        x / (1 + exp(-x))
    }
}
