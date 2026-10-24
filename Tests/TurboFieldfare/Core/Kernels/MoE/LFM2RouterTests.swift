import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// LFM2.5 router: 64-expert FP32 GEMV, sigmoid scoring, bias used only for
/// selection, top-4 by `(selection, -index)`, and FP16 weights that renormalize
/// the *unbiased* sigmoid values with `sum + 1e-6`.
@Suite struct LFM2RouterTests {

    /// The production router is fixed at 64 experts; shorter fixtures pad the
    /// logits with `-inf` and the bias with zero so only supplied experts are
    /// selectable.
    private static let numExperts = 64
    private static let topK = 4

    @Test func routerUsesBiasOnlyForSelectionAndNormalizesUnbiasedWeights() throws {
        let logits: [Float] = [0, 0, 0, 0, 10, 10]
        let bias: [Float] = [0, 0, 0, 0, 5, 5]
        let (indices, weights) = try runLFM2Router(logits, bias, topK: 4, scale: 1.0)
        #expect(indices == [4, 5, 0, 1])
        let p4 = 1 / (1 + exp(-10 as Float))
        let sum = 2 * p4 + 2 * 0.5 + 1e-6
        // Routing weights leave the kernel as FP16 (one half ULP at ~0.33 is
        // 2.4e-4), so storage precision bounds the comparison.
        #expect(abs(Float(weights[0]) - p4 / sum) < 2.5e-4)
        #expect(abs(Float(weights[1]) - p4 / sum) < 2.5e-4)
        #expect(abs(Float(weights[2]) - 0.5 / sum) < 2.5e-4)
        #expect(abs(Float(weights[3]) - 0.5 / sum) < 2.5e-4)
    }

    @Test func routerTieBreaksByLowerExpertIndex() throws {
        let (indices, _) = try runLFM2Router(
            [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0], topK: 4, scale: 1.0)
        #expect(indices == [0, 1, 2, 3])
    }

    /// `routedScalingFactor` multiplies the normalized weights and the bias is
    /// never part of the weight numerator.
    @Test func routedScalingFactorScalesNormalizedWeights() throws {
        let logits: [Float] = [0, 0, 0, 0, 10, 10]
        let bias: [Float] = [0, 0, 0, 0, 5, 5]
        let (indices, weights) = try runLFM2Router(logits, bias, topK: 4, scale: 2.5)
        #expect(indices == [4, 5, 0, 1])
        let p4 = 1 / (1 + exp(-10 as Float))
        let sum = 2 * p4 + 2 * 0.5 + 1e-6
        #expect(abs(Float(weights[0]) - 2.5 * p4 / sum) < 5e-4)
        #expect(abs(Float(weights[2]) - 2.5 * 0.5 / sum) < 5e-4)
    }

    @Test func fp32RouterGEMVMatchesCPU() throws {
        var rng = SplitMix64(seed: 0x11F2_0A3C)
        let weights = (0..<Self.numExperts).map { _ in
            (0..<2048).map { _ in rng.uniform(-0.05, 0.05) }
        }
        let x: [Float16] = (0..<2048).map { _ in
            Float16(rng.uniform(-1, 1))
        }
        let actual = try runLFM2RouterLogits(weights, x)
        var expected = [Float](repeating: 0, count: Self.numExperts)
        for e in 0..<Self.numExperts {
            var acc: Float = 0
            for i in 0..<2048 {
                acc += weights[e][i] * Float(x[i])
            }
            expected[e] = acc
        }
        for e in 0..<Self.numExperts {
            #expect(abs(actual[e] - expected[e]) < 1e-3, "expert \(e)")
        }
    }

    // MARK: - Helpers

    /// Runs only the selection stage on the supplied logits/bias.
    private func runLFM2Router(_ logits: [Float],
                               _ bias: [Float],
                               topK: Int,
                               scale: Float) throws -> ([UInt32], [Float16]) {
        precondition(logits.count == bias.count)
        precondition(logits.count <= Self.numExperts)
        var paddedLogits = [Float](repeating: -.infinity, count: Self.numExperts)
        var paddedBias = [Float](repeating: 0, count: Self.numExperts)
        for i in logits.indices {
            paddedLogits[i] = logits[i]
            paddedBias[i] = bias[i]
        }

        let context = try MetalContext()
        let kernel = try LFM2Router(context: context)
        let logitsBuffer = try #require(context.device.makeBuffer(
            bytes: paddedLogits,
            length: paddedLogits.count * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let biasBuffer = try #require(context.device.makeBuffer(
            bytes: paddedBias,
            length: paddedBias.count * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let indicesBuffer = try #require(context.device.makeBuffer(
            length: topK * MemoryLayout<UInt32>.stride,
            options: .storageModeShared))
        let weightsBuffer = try #require(Fp16Buffer.make(context.device, count: topK))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        kernel.encodeSelect(
            commandBuffer: commandBuffer,
            logits: logitsBuffer,
            bias: biasBuffer,
            indices: indicesBuffer,
            weights: weightsBuffer,
            topK: UInt32(topK),
            routedScalingFactor: scale)

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let pointer = indicesBuffer.contents().bindMemory(
            to: UInt32.self, capacity: topK)
        return ((0..<topK).map { pointer[$0] },
                Fp16Buffer.readHalf(weightsBuffer, count: topK))
    }

    private func runLFM2RouterLogits(_ weights: [[Float]],
                                     _ x: [Float16]) throws -> [Float] {
        precondition(weights.count == Self.numExperts)
        precondition(x.count == 2048)
        let flattened = weights.flatMap { $0 }

        let context = try MetalContext()
        let kernel = try LFM2Router(context: context)
        let weightBuffer = try #require(context.device.makeBuffer(
            bytes: flattened,
            length: flattened.count * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let xBuffer = try #require(Fp16Buffer.make(context.device, halves: x))
        let logitsBuffer = try #require(context.device.makeBuffer(
            length: Self.numExperts * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        kernel.encodeLogits(
            commandBuffer: commandBuffer,
            weights: weightBuffer,
            x: xBuffer,
            logits: logitsBuffer)

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let pointer = logitsBuffer.contents().bindMemory(
            to: Float.self, capacity: Self.numExperts)
        return (0..<Self.numExperts).map { pointer[$0] }
    }
}
