import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// LFM2.5 applies a standard (no unit offset) RMSNorm to every Q and K head
/// with a shared BF16 gain, then full NeoX rotate-half RoPE:
///
///   angle(i) = position * theta^(-2i / headDim)
///   out[i]        = x[i] * cos - x[i + headDim/2] * sin
///   out[i+half]   = x[i] * sin + x[i + headDim/2] * cos
///
/// The reference below is an independent FP32 implementation; weights are
/// BF16-round-tripped before both sides see them so the comparison is against
/// the production tensor type.
@Suite struct LFM2QKNormRoPETests {

    private static let eps: Float = 1e-5

    @Test func qkNormAndNeoXRoPEMatchCPU() throws {
        let q = (0..<64).map { Float16(sin(Float($0))) }
        let k = (0..<64).map { Float16(cos(Float($0))) }
        let qw = (0..<64).map { Float16(1 + Float($0) / 64) }
        let kw = (0..<64).map { Float16(1 - Float($0) / 128) }
        let (qo, ko) = try runQKNormRoPE(q, k, qw, kw, heads: 1, kvHeads: 1,
                                          headDim: 64, position: 7, theta: 1e7)
        let (eq, ek) = cpuQKNormRoPE(q, k, qw, kw, headDim: 64,
                                     position: 7, theta: 1e7)
        for i in 0..<64 {
            #expect(abs(qo[i] - eq[i]) < 2e-3)
            #expect(abs(ko[i] - ek[i]) < 2e-3)
        }
    }

    @Test func neoxRoPEPairsFirstAndSecondHalf() {
        // 8 in one lane and zeros elsewhere has unit RMS, so the standard norm
        // leaves the vector unchanged before the rotation.
        var q = [Float16](repeating: 0, count: 64)
        q[0] = 8
        let (qo, _) = try! runQKNormRoPE(q, q, unitWeights(), unitWeights(),
                                          heads: 1, kvHeads: 1, headDim: 64,
                                          position: 1, theta: 1e7)
        #expect(abs(qo[0] - 8 * cos(1)) < 2e-3)
        #expect(abs(qo[32] - 8 * sin(1)) < 2e-3)
    }

    /// Unequal head counts exercise the separate Q and K regions: Q must run
    /// over `qHeads` heads and K over `kvHeads`, each with its own weight.
    @Test func qkNormUsesQHeadsAndKVHeadsSeparately() throws {
        let headDim = 64
        let qHeads = 5
        let kvHeads = 3
        let q = (0..<(qHeads * headDim)).map {
            Float16(2 * sin(Float($0) * 0.37))
        }
        let k = (0..<(kvHeads * headDim)).map {
            Float16(2 * cos(Float($0) * 0.23))
        }
        let qw = (0..<headDim).map { Float16(0.75 + Float($0) / 128) }
        let kw = (0..<headDim).map { Float16(1.25 - Float($0) / 256) }

        let (qo, ko) = try runQKNormRoPE(q, k, qw, kw,
                                          heads: UInt32(qHeads),
                                          kvHeads: UInt32(kvHeads),
                                          headDim: UInt32(headDim),
                                          position: 13, theta: 1e7)
        let (eq, ek) = cpuQKNormRoPE(q, k, qw, kw, headDim: headDim,
                                     position: 13, theta: 1e7)
        #expect(qo.count == qHeads * headDim)
        #expect(ko.count == kvHeads * headDim)
        for i in 0..<qo.count {
            #expect(abs(qo[i] - eq[i]) < 2e-3, "Q index \(i)")
        }
        for i in 0..<ko.count {
            #expect(abs(ko[i] - ek[i]) < 2e-3, "K index \(i)")
        }
    }

    /// The norm is standard, not Gemma's unit-offset form. With unit weights,
    /// each head of constant `c` normalizes to `sign(c)` before RoPE.
    @Test func qkNormHasNoUnitOffset() throws {
        let headDim = 64
        let q = [Float16](repeating: 2, count: headDim)
        let (qo, _) = try runQKNormRoPE(q, q, unitWeights(), unitWeights(),
                                         heads: 1, kvHeads: 1,
                                         headDim: UInt32(headDim),
                                         position: 0, theta: 1e7)
        for i in 0..<headDim {
            #expect(abs(qo[i] - 1) < 2e-3, "index \(i)")
        }
    }

    // MARK: - Helpers

    private func runQKNormRoPE(_ q: [Float16],
                               _ k: [Float16],
                               _ qWeight: [Float16],
                               _ kWeight: [Float16],
                               heads: UInt32,
                               kvHeads: UInt32,
                               headDim: UInt32,
                               position: UInt32,
                               theta: Float) throws -> ([Float], [Float]) {
        let context = try MetalContext()
        let kernel = try LFM2QKNormRoPE(context: context)
        let qBuf = try #require(Fp16Buffer.make(context.device, halves: q))
        let kBuf = try #require(Fp16Buffer.make(context.device, halves: k))
        let qwBuf = try Self.bf16Buffer(context.device, qWeight)
        let kwBuf = try Self.bf16Buffer(context.device, kWeight)
        let qOutBuf = try #require(Fp16Buffer.make(context.device, count: q.count))
        let kOutBuf = try #require(Fp16Buffer.make(context.device, count: k.count))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        kernel.encode(commandBuffer: commandBuffer,
                      q: qBuf, k: kBuf,
                      qWeight: qwBuf, kWeight: kwBuf,
                      qOut: qOutBuf, kOut: kOutBuf,
                      qHeads: heads, kvHeads: kvHeads, headDim: headDim,
                      position: position, theta: theta, eps: Self.eps)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return (Fp16Buffer.read(qOutBuf, count: q.count),
                Fp16Buffer.read(kOutBuf, count: k.count))
    }

    private func cpuQKNormRoPE(_ q: [Float16],
                               _ k: [Float16],
                               _ qWeight: [Float16],
                               _ kWeight: [Float16],
                               headDim: Int,
                               position: Int,
                               theta: Float) -> ([Float], [Float]) {
        let qw = qWeight.map { Quantization.bf16ToFloat(Quantization.bf16Bits(Float($0))) }
        let kw = kWeight.map { Quantization.bf16ToFloat(Quantization.bf16Bits(Float($0))) }
        func apply(_ input: [Float16], _ weight: [Float]) -> [Float] {
            let heads = input.count / headDim
            let halfDim = headDim / 2
            var out = [Float](repeating: 0, count: input.count)
            for head in 0..<heads {
                let base = head * headDim
                var sumSq: Float = 0
                for i in 0..<headDim {
                    let v = Float(input[base + i])
                    sumSq += v * v
                }
                let inv = 1 / sqrt(sumSq / Float(headDim) + Self.eps)
                for i in 0..<halfDim {
                    let x0 = Float(input[base + i]) * inv * weight[i]
                    let x1 = Float(input[base + halfDim + i]) * inv * weight[halfDim + i]
                    let angle = Float(position)
                        * pow(theta, -2 * Float(i) / Float(headDim))
                    let c = cos(angle)
                    let s = sin(angle)
                    out[base + i] = x0 * c - x1 * s
                    out[base + halfDim + i] = x0 * s + x1 * c
                }
            }
            return out
        }
        return (apply(q, qw), apply(k, kw))
    }

    private func unitWeights() -> [Float16] {
        [Float16](repeating: 1, count: 64)
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
