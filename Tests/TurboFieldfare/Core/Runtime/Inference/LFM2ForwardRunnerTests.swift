import Foundation
import Metal
import Testing
import TurboFieldfareFormat

@testable import TurboFieldfare

/// Decode, chunked prefill, and conv-rewind coverage for the LFM2.5 runner.
///
/// Every numeric case needs the installed `scratch/lfm2.5-24b-a2b.gturbo`
/// pack; without it they return early so a checkout with only the Gemma model
/// still runs the suite. The factory-routing case always runs.
@Suite(.serialized)
struct LFM2ForwardRunnerTests {
    private struct Fixture {
        let runner: LFM2ForwardRunner
        let device: MTLDevice
    }

    // MARK: - Factory routing (no model required)

    @Test func factoryPlansLFMRunnerForLFMFamily() throws {
        #expect(try RunnerFactory.plan(for: ModelFamily.lfm2Moe) == .lfm2Moe)
        #expect(try RunnerFactory.plan(for: ModelFamily.gemma4) == .gemma4)
    }

    // MARK: - New kernels without a model pack

    /// The LFM prefill router is new in this task. It must select and weight
    /// exactly like the decode router: sigmoid + bias for selection, unbiased
    /// sigmoid values normalized with `sum + 1e-6`, scaled by the routed
    /// scaling factor, ties toward the lower expert index.
    @Test func prefillRouterMatchesDecodeRouter() throws {
        let context = try MetalContext()
        let device = context.device
        let numExperts = Int(LFM2Router.numExperts)
        let d = Int(LFM2Router.hiddenDimension)

        var weights = [Float](repeating: 0, count: numExperts * d)
        for expert in 0..<numExperts {
            for column in 0..<d {
                weights[expert * d + column] =
                    Float(((expert * 37 + column * 11) % 101) - 50) * 0.000_7
            }
        }
        var hiddenValues = [Float16](repeating: 0, count: d)
        for index in 0..<d {
            hiddenValues[index] = Float16(Float((index % 29) - 14) * 0.02)
        }
        var bias = [Float](repeating: 0, count: numExperts)
        // Pin two experts so the case cannot depend on the deterministic
        // pseudo-random scores crossing near a top-4 boundary.
        bias[3] = 10.0
        bias[17] = 0.0

        let weightBuffer = try #require(device.makeBuffer(
            bytes: weights,
            length: weights.count * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let hiddenBuffer = try #require(device.makeBuffer(
            bytes: hiddenValues,
            length: hiddenValues.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared))
        let biasBuffer = try #require(device.makeBuffer(
            bytes: bias,
            length: bias.count * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let logits = try #require(device.makeBuffer(
            length: numExperts * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let decodeIndices = try #require(device.makeBuffer(
            length: 4 * MemoryLayout<UInt32>.stride,
            options: .storageModeShared))
        let decodeWeights = try #require(device.makeBuffer(
            length: 4 * MemoryLayout<Float16>.stride,
            options: .storageModeShared))
        let prefillIndices = try #require(device.makeBuffer(
            length: 4 * MemoryLayout<UInt32>.stride,
            options: .storageModeShared))
        let prefillWeights = try #require(device.makeBuffer(
            length: 4 * MemoryLayout<Float16>.stride,
            options: .storageModeShared))

        let decodeRouter = try LFM2Router(context: context)
        let decodeCB = try #require(context.queue.makeCommandBuffer())
        decodeRouter.encodeLogits(commandBuffer: decodeCB,
                                  weights: weightBuffer,
                                  x: hiddenBuffer,
                                  logits: logits)
        decodeRouter.encodeSelect(commandBuffer: decodeCB,
                                  logits: logits,
                                  bias: biasBuffer,
                                  indices: decodeIndices,
                                  weights: decodeWeights,
                                  topK: 4,
                                  routedScalingFactor: 1.0)
        decodeCB.commit()
        decodeCB.waitUntilCompleted()

        let prefillRouter = try LFM2PrefillRouter(context: context)
        let prefillCB = try #require(context.queue.makeCommandBuffer())
        prefillRouter.encodeBlock(commandBuffer: prefillCB,
                                  weights: weightBuffer,
                                  bias: biasBuffer,
                                  hidden: hiddenBuffer,
                                  outIndices: prefillIndices,
                                  outWeights: prefillWeights,
                                  queryCount: 1,
                                  numExperts: UInt32(numExperts),
                                  d: UInt32(d),
                                  topK: 4,
                                  hiddenStrideElements: UInt32(d),
                                  routedScalingFactor: 1.0)
        prefillCB.commit()
        prefillCB.waitUntilCompleted()

        let decodeIndexPointer = decodeIndices.contents()
            .bindMemory(to: UInt32.self, capacity: 4)
        let prefillIndexPointer = prefillIndices.contents()
            .bindMemory(to: UInt32.self, capacity: 4)
        let decodeWeightPointer = decodeWeights.contents()
            .bindMemory(to: Float16.self, capacity: 4)
        let prefillWeightPointer = prefillWeights.contents()
            .bindMemory(to: Float16.self, capacity: 4)
        for rank in 0..<4 {
            #expect(decodeIndexPointer[rank] == prefillIndexPointer[rank])
            #expect(abs(Float(decodeWeightPointer[rank])
                        - Float(prefillWeightPointer[rank])) <= 1e-3)
        }
        #expect((0..<4).contains { decodeIndexPointer[$0] == 3 })
    }

    /// The convolution gating helpers and residual add are new shaders. Check
    /// them against the literal two-token algebra rather than only through a
    /// 15 GB model.
    @Test func convOpsMatchTwoTokenAlgebra() throws {
        let context = try MetalContext()
        let device = context.device
        let tokens = 2
        let channels = 4
        let stride = 3 * channels

        let projValues: [Float16] = [
            1, 2, 3, 4,     10, 20, 30, 40,    0.5, 0.25, 0.125, 0.0625,
            2, 3, 4, 5,     1, 2, 3, 4,        1, 1, 1, 1,
        ]
        let convValues: [Float16] = [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5]
        let proj = try #require(device.makeBuffer(
            bytes: projValues,
            length: projValues.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared))
        let conv = try #require(device.makeBuffer(
            bytes: convValues,
            length: convValues.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared))
        let bx = try #require(device.makeBuffer(
            length: tokens * channels * MemoryLayout<Float16>.stride,
            options: .storageModeShared))
        let gated = try #require(device.makeBuffer(
            length: tokens * channels * MemoryLayout<Float16>.stride,
            options: .storageModeShared))
        let hiddenValues: [Float16] = [1, 2, 3, 4]
        let branchValues: [Float16] = [0.5, 0.25, 0.125, 0.0625]
        let hidden = try #require(device.makeBuffer(
            bytes: hiddenValues,
            length: hiddenValues.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared))
        let branch = try #require(device.makeBuffer(
            bytes: branchValues,
            length: branchValues.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared))

        let ops = try LFM2ConvOps(context: context)
        let cb = try #require(context.queue.makeCommandBuffer())
        ops.encodeBxMul(commandBuffer: cb,
                        proj: proj,
                        bx: bx,
                        tokens: tokens,
                        channels: channels)
        ops.encodeGateMul(commandBuffer: cb,
                          proj: proj,
                          conv: conv,
                          out: gated,
                          tokens: tokens,
                          channels: channels)
        ops.encodeResidualAdd(commandBuffer: cb,
                              hidden: hidden,
                              branch: branch,
                              count: channels)
        cb.commit()
        cb.waitUntilCompleted()

        let bxValues = bx.contents().bindMemory(to: Float16.self,
                                                capacity: tokens * channels)
        let gatedValues = gated.contents().bindMemory(to: Float16.self,
                                                      capacity: tokens * channels)
        let hiddenResult = hidden.contents().bindMemory(to: Float16.self,
                                                        capacity: channels)
        for token in 0..<tokens {
            for channel in 0..<channels {
                let base = token * stride
                let expectedBx = Float(projValues[base + channel])
                    * Float(projValues[base + 2 * channels + channel])
                let expectedGated = Float(projValues[base + channels + channel])
                    * Float(convValues[token * channels + channel])
                #expect(abs(Float(bxValues[token * channels + channel]) - expectedBx) <= 1e-3)
                #expect(abs(Float(gatedValues[token * channels + channel]) - expectedGated) <= 1e-3)
            }
        }
        for channel in 0..<channels {
            #expect(abs(Float(hiddenResult[channel])
                        - (Float(hiddenValues[channel]) + Float(branchValues[channel]))) <= 1e-3)
        }
    }

    // MARK: - Decode

    @Test func lfmDecodeProducesFiniteLogits() async throws {
        guard let fixture = try makeLFMRunner() else { return }
        let logits = try makeLogitsBuffer(device: fixture.device)
        try await fixture.runner.produce(token: 124_894, position: 0, into: logits)
        let values = readLogits(logits)
        #expect(values.count == 128_000)
        #expect(values.allSatisfy { $0.isFinite })
        #expect(fixture.runner.usesFusedGreedyHead == false)
        #expect(fixture.runner.lastGreedyToken == 0)
        #expect(fixture.runner.convRewindLimit == 128)
        #expect(fixture.runner.continuationPosition == 1)
    }

    @Test func lfmPrefillMatchesTokenByTokenDecode() async throws {
        guard let fixture = try makeLFMRunner() else { return }
        let prompt: [Int32] = [124_894, 100, 101, 102, 103]
        let prefillLogits = try await prefill(fixture,
                                               tokens: prompt,
                                               chunk: 32)
        let decodeLogits = try await decodeOneByOne(fixture, tokens: prompt)
        #expect(maxAbsDiff(prefillLogits, decodeLogits) < 0.05)
    }

    // MARK: - Chunking

    /// The public `PrefillRuntimeConfig` only admits the 32/64/128/256
    /// production chunk sizes, so this compares the smallest supported chunk
    /// (two chunks for 33 tokens) against 64 and 128 (one chunk each).
    @Test func lfmChunkSizeDoesNotChangeLogits() async throws {
        let prompt = Array(repeating: Int32(100), count: 33)
        guard let aFixture = try makeLFMRunner(),
              let bFixture = try makeLFMRunner(),
              let cFixture = try makeLFMRunner() else { return }
        let a = try await prefill(aFixture, tokens: prompt, chunk: 32)
        let b = try await prefill(bFixture, tokens: prompt, chunk: 64)
        let c = try await prefill(cFixture, tokens: prompt, chunk: 128)
        #expect(maxAbsDiff(a, b) < 0.05)
        #expect(maxAbsDiff(b, c) < 0.05)
    }

    // MARK: - Conv rewind

    @Test func rewindWithinConvLimitRestoresBothCaches() async throws {
        guard let fixture = try makeLFMRunner() else { return }
        _ = try await prefill(fixture,
                              tokens: Array(repeating: Int32(100), count: 200),
                              chunk: 128)
        try fixture.runner.rewind(to: 72)
        #expect(fixture.runner.continuationPosition == 72)
        try fixture.runner.prepareForContinuation(expectedPosition: 72)
        let again = try await prefill(fixture,
                                      tokens: Array(repeating: Int32(101), count: 20),
                                      chunk: 128,
                                      startPosition: 72)
        #expect(again.allSatisfy { $0.isFinite })
        #expect(fixture.runner.continuationPosition == 92)
    }

    @Test func rewindBeyondConvLimitThrowsWithoutMutating() async throws {
        guard let fixture = try makeLFMRunner() else { return }
        _ = try await prefill(fixture,
                              tokens: Array(repeating: Int32(100), count: 300),
                              chunk: 128)
        let before = fixture.runner.continuationPosition
        #expect(throws: PrefillError.self) {
            try fixture.runner.rewind(to: before - 129)
        }
        #expect(fixture.runner.continuationPosition == before)
    }

    @Test func rewindRejectsNonPositiveAndFutureTargets() async throws {
        guard let fixture = try makeLFMRunner() else { return }
        _ = try await prefill(fixture,
                              tokens: Array(repeating: Int32(100), count: 40),
                              chunk: 64)
        let before = fixture.runner.continuationPosition
        #expect(throws: PrefillError.self) { try fixture.runner.rewind(to: 0) }
        #expect(throws: PrefillError.self) {
            try fixture.runner.rewind(to: before + 1)
        }
        #expect(fixture.runner.continuationPosition == before)
    }

    // MARK: - Helpers

    private func makeLFMRunner(maxContext: Int = 512) throws -> Fixture? {
        guard let loaded = try Self.Harness.shared.load() else { return nil }
        let runner = try RunnerFactory.make(model: loaded.model,
                                            context: loaded.context,
                                            maxContext: maxContext)
        let lfm = try #require(runner as? LFM2ForwardRunner,
                               "factory returned \(type(of: runner)) for .lfm2Moe") as LFM2ForwardRunner
        return Fixture(runner: lfm, device: loaded.context.device)
    }

    private func makeLogitsBuffer(device: MTLDevice) throws -> MTLBuffer {
        try #require(device.makeBuffer(
            length: ArchConfig.lfm2_5_24B_A2B.vocabSize * MemoryLayout<Float16>.stride,
            options: .storageModeShared))
    }

    private func readLogits(_ buffer: MTLBuffer) -> [Float] {
        let count = buffer.length / MemoryLayout<Float16>.stride
        let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(pointer[$0]) }
    }

    private func prefill(_ fixture: Fixture,
                         tokens: [Int32],
                         chunk: Int,
                         startPosition: Int = 0) async throws -> [Float] {
        let logits = try makeLogitsBuffer(device: fixture.device)
        let config = PrefillRuntimeConfig.production(
            chunkTokens: PrefillRuntimeConfig.supportedChunkTokens(chunk))
        _ = try await fixture.runner.prefillChunked(
            tokens: tokens[...],
            startPosition: startPosition,
            outputMode: PrefillOutputMode.logits,
            config: config,
            into: logits,
            onProgress: { _ in })
        return readLogits(logits)
    }

    private func decodeOneByOne(_ fixture: Fixture,
                                tokens: [Int32]) async throws -> [Float] {
        fixture.runner.reset()
        var last: [Float] = []
        for (position, token) in tokens.enumerated() {
            let logits = try makeLogitsBuffer(device: fixture.device)
            try await fixture.runner.produce(token: token,
                                             position: position,
                                             into: logits)
            last = readLogits(logits)
        }
        return last
    }

    private func maxAbsDiff(_ lhs: [Float], _ rhs: [Float]) -> Float {
        guard lhs.count == rhs.count else { return .infinity }
        var maximum: Float = 0
        for index in lhs.indices {
            maximum = max(maximum, abs(lhs[index] - rhs[index]))
        }
        return maximum
    }

    /// One model load and one `MetalContext` for the whole suite. Loading the
    /// 15 GB pack once keeps the numeric cases affordable when it is present.
    private final class Harness: @unchecked Sendable {
        static let shared = Harness()

        private let lock = NSLock()
        private var context: MetalContext?
        private var model: Model?
        private var searched = false

        func load() throws -> (context: MetalContext, model: Model)? {
            lock.lock()
            defer { lock.unlock() }
            if searched {
                if let context, let model { return (context, model) }
                return nil
            }
            guard let url = Self.modelURL() else {
                searched = true
                return nil
            }
            let context = try MetalContext()
            let model = try Model.load(directoryURL: url,
                                       device: context.device,
                                       expecting: .lfm2_5_24B_A2B)
            self.context = context
            self.model = model
            searched = true
            return (context, model)
        }

        /// Walks up from this source file so the suite finds the checkout's
        /// `scratch` directory regardless of the working directory.
        private static func modelURL(from file: StaticString = #filePath) -> URL? {
            var directory = URL(fileURLWithPath: "\(file)").deletingLastPathComponent()
            while directory.path != "/" {
                let candidate = directory
                    .appendingPathComponent("scratch")
                    .appendingPathComponent("lfm2.5-24b-a2b.gturbo")
                if FileManager.default.fileExists(atPath: candidate.path) {
                    return candidate
                }
                directory = directory.deletingLastPathComponent()
            }
            return nil
        }
    }
}
