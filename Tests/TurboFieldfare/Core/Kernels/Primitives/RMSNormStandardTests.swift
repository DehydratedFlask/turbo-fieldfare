import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Standard (no unit offset) RMSNorm — the variant LFM2.5's
/// `operator_norm` / `ffn_norm` / `q_layernorm` / `k_layernorm` use:
///
///   y[i] = weight[i] * x[i] / sqrt(mean(x^2) + eps)
///
/// Gemma's `rmsnorm_bf16w` multiplies by `(1 + weight)`; these tests pin the
/// difference and the BF16 weight binding the installed LFM pack requires.
@Suite struct RMSNormStandardTests {

    @Test func standardRMSNormDoesNotAddUnitOffset() throws {
        let x: [Float16] = [3, 4]
        let w: [Float16] = [2, 0.5]
        let y = try runStandardRMSNorm(x, weight: w, eps: 1e-5)
        let rms = sqrt(Float(9 + 16) / 2)
        #expect(abs(y[0] - 2 * 3 / rms) < 1e-3)
        #expect(abs(y[1] - 0.5 * 4 / rms) < 1e-3)
    }

    /// Random data at the LFM hidden width, compared against an independent
    /// Accelerate reference over BF16-round-tripped weights.
    @Test func standardRMSNormMatchesReference() throws {
        let d = 2048
        let eps: Float = 1e-5
        var rng = SeedTree(0x9A17).key("rmsnorm-standard-d\(d)")
        let xFp32 = (0..<d).map { _ in rng.uniform(-1.0, 1.0) }
        let wFp32 = (0..<d).map { _ in rng.uniform(0.5, 1.5) }
        let x = xFp32.map { Float16($0) }
        let w = wFp32.map { Float16($0) }
        let wRef = w.map { Quantization.bf16ToFloat(Quantization.bf16Bits(Float($0))) }

        let y = try runStandardRMSNorm(x, weight: w, eps: eps)
        let ref = RmsNormRef.apply(x: x.map { Float($0) }, weight: wRef, eps: eps)
        let relErr = RelError.compute(actual: y, reference: ref)
        #expect(relErr < Tolerance.fp16Reduction,
                "standard D=\(d): relErr=\(relErr) maxAbsDiff=\(RelError.maxAbsDiff(y, ref))")
    }

    // MARK: - Helpers

    private func runStandardRMSNorm(_ x: [Float16],
                                    weight: [Float16],
                                    eps: Float) throws -> [Float] {
        let context = try MetalContext()
        let kernel = try RMSNorm(context: context)
        let xBuf = try #require(Fp16Buffer.make(context.device, halves: x))
        let yBuf = try #require(Fp16Buffer.make(context.device, count: x.count))
        let wBuf = try Self.bf16Buffer(context.device, weight)
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        kernel.encodeStandard(commandBuffer: commandBuffer,
                              x: xBuf, weight: wBuf, out: yBuf,
                              d: UInt32(x.count), eps: eps)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return Fp16Buffer.read(yBuf, count: x.count)
    }

    /// Test inputs arrive as `Float16` for readability; bind the production
    /// tensor type (BF16) by converting to raw BF16 bits first.
    private static func bf16Buffer(_ device: MTLDevice,
                                   _ weights: [Float16]) throws -> MTLBuffer {
        let bits = weights.map { Quantization.bf16Bits(Float($0)) }
        let buffer = try #require(device.makeBuffer(
            length: bits.count * MemoryLayout<UInt16>.size,
            options: .storageModeShared))
        let pointer = buffer.contents().bindMemory(to: UInt16.self,
                                                   capacity: bits.count)
        for i in bits.indices { pointer[i] = bits[i] }
        return buffer
    }
}
