import Testing
import Foundation
import Metal
import TurboFieldfareFormat
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Shared raw-completion loop validation: stops, progress callbacks, tail
/// flush, and cancellation, via `ScriptedLogitProducer` (kernel-independent).
@Suite struct RawCompletionLoopTests {

    func automaton(_ seq: [Int32], end: Int32) -> @Sendable (Int32, Int) -> ScriptedLogitProducer.Step {
        let next: [Int32: Int32] = {
            var n: [Int32: Int32] = [:]
            for i in 0..<max(0, seq.count - 1) { n[seq[i]] = seq[i + 1] }
            if let last = seq.last { n[last] = end }
            return n
        }()
        let first = seq.first ?? end
        return { input, _ in .argmax(next[input] ?? first) }
    }

    struct Collected {
        var prefills: [(Int, Int)] = []
        var tokens: [(Int, Int32, String)] = []
        var tails: [String] = []
    }

    final class CountingProducer: LogitProducer, ContextWindowReporting, @unchecked Sendable {
        let vocabSize: Int
        let maxContext: Int
        private let step: @Sendable (Int32, Int) -> ScriptedLogitProducer.Step
        private var calls = 0
        private(set) var resetCalls = 0
        private(set) var produceCalls = 0

        init(vocabSize: Int,
             maxContext: Int = Int.max,
             step: @escaping @Sendable (Int32, Int) -> ScriptedLogitProducer.Step) {
            self.vocabSize = vocabSize
            self.maxContext = maxContext
            self.step = step
        }

        func reset() {
            resetCalls += 1
            calls = 0
            produceCalls = 0
        }

        func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
            produceCalls += 1
            let spec = step(token, calls)
            calls += 1
            let ptr = logits.contents().bindMemory(to: Float16.self, capacity: vocabSize)
            switch spec {
            case .argmax(let token):
                for i in 0..<vocabSize { ptr[i] = Float16(-30.0) }
                if Int(token) >= 0 && Int(token) < vocabSize {
                    ptr[Int(token)] = Float16(30.0)
                }
            case .vector(let values):
                for i in 0..<vocabSize {
                    ptr[i] = Float16(i < values.count ? values[i] : -30.0)
                }
            }
        }

    }

    final class ChunkedTestProducer: LogitProducer, ChunkedPrefillRunner, @unchecked Sendable {
        let vocabSize: Int
        private let firstToken: Int32
        private let seed: PrefillSeed
        private(set) var resetCalls = 0
        private(set) var produceCalls = 0
        private(set) var chunkedCalls = 0
        private(set) var lastOutputMode: PrefillOutputMode?
        private(set) var lastConfig: PrefillRuntimeConfig?

        init(vocabSize: Int,
             firstToken: Int32,
             seed: PrefillSeed = .logitsWritten) {
            self.vocabSize = vocabSize
            self.firstToken = firstToken
            self.seed = seed
        }

        func reset() {
            resetCalls += 1
            produceCalls = 0
            chunkedCalls = 0
            lastOutputMode = nil
            lastConfig = nil
        }

        func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
            produceCalls += 1
        }

        func prefillChunked(tokens: ArraySlice<Int32>,
                            startPosition: Int,
                            outputMode: PrefillOutputMode,
                            config: PrefillRuntimeConfig,
                            into logits: MTLBuffer,
                            onProgress: (Int) -> Void) async throws -> PrefillResult {
            chunkedCalls += 1
            lastOutputMode = outputMode
            lastConfig = config
            let ptr = logits.contents().bindMemory(to: Float16.self, capacity: vocabSize)
            for i in 0..<vocabSize { ptr[i] = Float16(-30.0) }
            ptr[Int(firstToken)] = Float16(30.0)
            onProgress(tokens.count)
            return PrefillResult(newPosition: startPosition + tokens.count,
                                 seed: seed)
        }
    }

    func runLoop(seq: [Int32], end: Int32, prompt: String = "go",
                         config: GenerationConfig) async throws -> (Collected, RawDecodeResult) {
        let ctx = try MetalContext()
        let tok = try await GFTokenizer.load()
        let producer = ScriptedLogitProducer(vocabSize: tok.vocabSize,
                                             step: automaton(seq, end: end))
        let promptIds = tok.encode(prompt, addBOS: true)
        let scratch = try RawCompletionScratch(context: ctx, vocab: tok.vocabSize)
        var collected = Collected()
        let result = try await runRawCompletion(producer: producer, tokenizer: tok,
                                                promptIds: promptIds, config: config,
                                                context: ctx, scratch: scratch,
                                                prefillConfig: .off) { progress in
            switch progress {
            case .prefill(let done, let total): collected.prefills.append((done, total))
            case .token(let index, let id, let delta): collected.tokens.append((index, id, delta))
            case .tail(let text): collected.tails.append(text)
            case .reasoning: break
            }
        }
        return (collected, result)
    }

    // MARK: - Family-neutral tokenizer

    /// Minimal `ModelTokenizer` standing in for the LFM2 tokenizer Task 17
    /// adds. It carries the LFM marker IDs so the loop is exercised against a
    /// family whose structural markers are not Gemma's.
    struct ScriptedTokenizer: ModelTokenizer {
        let family: ModelFamily
        let stopTokenIDs: Set<Int32>
        let structuralMarkerIDs: Set<Int32>

        var bosID: Int32 { 124_894 }
        var eosID: Int32 { 124_900 }
        var padID: Int32 { 124_893 }
        var endOfTurnID: Int32 { 124_900 }
        var toolResponseID: Int32 { 124_900 }
        var toolCallStartID: Int32 { 124_901 }
        var toolCallEndID: Int32 { 124_902 }
        var toolResponseEndID: Int32 { 124_900 }
        var vocabSize: Int { 128_000 }
        var chatTemplateIdentity: String { "scripted-lfm-v1" }

        func encode(_ text: String, addBOS: Bool) -> [Int32] { [] }
        func decode(_ ids: [Int32], skipSpecialTokens: Bool) -> String { "" }

        func makeStreamingDetokenizer() -> any StreamingDetokenizer {
            ScriptedDetokenizer(emittedIDs: structuralMarkerIDs)
        }

        func applyChatTemplate(messages: [ChatMessage],
                               tools: [ToolDefinition]?,
                               addGenerationPrompt: Bool) throws -> [Int32] { [] }

        func continuationPlan(cachedMessages: [ChatMessage],
                              cachedTokenIDs: [Int32],
                              incomingMessages: [ChatMessage],
                              tools: [ToolDefinition]?) throws -> ContinuationPlan {
            .replayAll([])
        }

        func makeAssistantDecoder(allowedTools: Set<String>) -> (any StructuredOutputDecoder)? {
            nil
        }
    }

    /// Emits marker text only for the IDs the tokenizer calls structural; every
    /// other token contributes its own text. Mirrors the LFM detokenizer's
    /// contract that think/tool markers are never stripped before the
    /// structured decoder sees them.
    struct ScriptedDetokenizer: StreamingDetokenizer {
        let emittedIDs: Set<Int32>

        mutating func push(_ tokenID: Int32) -> String {
            emittedIDs.contains(tokenID) ? "<\(tokenID)>" : "t\(tokenID)"
        }

        mutating func flush() -> String { "" }
    }

    @Test func rawCompletionPassesStructuralMarkersToTheDecoder() async throws {
        let context = try MetalContext()
        let tokenizer = ScriptedTokenizer(
            family: .lfm2Moe,
            stopTokenIDs: [124_900],
            structuralMarkerIDs: [124_901, 124_902])
        let producer = ScriptedLogitProducer(
            vocabSize: tokenizer.vocabSize,
            step: automaton([124_901, 124_902, 42], end: 124_900))
        let scratch = try RawCompletionScratch(
            context: context, vocab: tokenizer.vocabSize)
        var tokens: [(Int32, String)] = []

        _ = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: [124_894],
            config: GenerationConfig(maxNewTokens: 3, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .off) { progress in
                if case .token(_, let id, let delta) = progress {
                    tokens.append((id, delta))
                }
            }

        #expect(tokens.map(\.0) == [124_901, 124_902, 42])
        #expect(tokens.map(\.1) == ["<124901>", "<124902>", "t42"])
    }

    /// The loop must recognize the fused head through
    /// `FusedGreedyHeadReporting`, not a concrete Gemma runner type. The decoy
    /// logits make a fallback to sampling observable: `SampleToken` is the
    /// argmax of every logits buffer the producer writes.
    final class ScriptedFusedProducer: LogitProducer, ContextWindowReporting,
                                      FusedGreedyHeadReporting, @unchecked Sendable {
        static let sampleToken: Int32 = 7

        let vocabSize: Int
        let maxContext = Int.max
        let usesFusedGreedyHead = true
        var lastGreedyToken: UInt32 = 0
        private let greedyTokens: [Int32]
        private var produceCalls = 0

        init(vocabSize: Int, greedyTokens: [Int32]) {
            self.vocabSize = vocabSize
            self.greedyTokens = greedyTokens
        }

        func reset() {
            produceCalls = 0
            lastGreedyToken = UInt32(bitPattern: greedyTokens.first ?? 0)
        }

        func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
            let ptr = logits.contents().bindMemory(to: Float16.self, capacity: vocabSize)
            for i in 0..<vocabSize { ptr[i] = Float16(-30) }
            ptr[Int(Self.sampleToken)] = Float16(30)
            produceCalls += 1
            let index = min(max(produceCalls - 1, 0), greedyTokens.count - 1)
            lastGreedyToken = UInt32(bitPattern: greedyTokens[index])
        }
    }

    @Test func fusedGreedyHeadIsUsedThroughTheReportingProtocol() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let producer = ScriptedFusedProducer(
            vocabSize: tokenizer.vocabSize,
            greedyTokens: [124_910, 124_911, tokenizer.eosID])
        let scratch = try RawCompletionScratch(
            context: context, vocab: tokenizer.vocabSize)
        var tokens: [Int32] = []

        // The prompt is one token, so exactly one prefill `produce` call runs
        // before the loop reads `lastGreedyToken` for the first generated
        // token; every later call advances the scripted sequence.
        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: [tokenizer.bosID],
            config: GenerationConfig(maxNewTokens: 3, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .off) { progress in
                if case .token(_, let id, _) = progress { tokens.append(id) }
            }

        #expect(tokens == [124_910, 124_911])
        #expect(!tokens.contains(ScriptedFusedProducer.sampleToken))
        #expect(result.reason == .eos)
    }
}
