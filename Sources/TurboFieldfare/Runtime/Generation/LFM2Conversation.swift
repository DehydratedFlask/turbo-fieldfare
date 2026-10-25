import Foundation

public enum LFM2ConversationError: Error, Equatable, CustomStringConvertible {
    case closed
    case lineageBroken
    case busy
    case emptyTurn
    case imagesUnsupported
    case contextExhausted(prompt: Int, reserve: Int, maxContext: Int)
    case continuationMismatch

    public var description: String {
        switch self {
        case .closed: "the conversation is closed"
        case .lineageBroken: "the KV no longer matches the conversation"
        case .busy: "a generation is already running"
        case .emptyTurn: "the turn is empty"
        case .imagesUnsupported: "image input is not supported for lfm2_moe"
        case .contextExhausted(let prompt, let reserve, let maxContext):
            "the chat holds \(prompt) tokens and keeps \(reserve) free, "
                + "which exceeds the \(maxContext)-token context"
        case .continuationMismatch: "the retained prompt no longer matches the conversation"
        }
    }
}

/// Text-only LFM conversation. Owns the runner's KV and conv state, keeps the
/// user/assistant message history for re-rendering, and routes the model's
/// ` thinking` channel through `LFM2AssistantDecoder` so reasoning never reaches
/// the visible answer or storage.
public actor LFM2Conversation: ConversationSession {
    private let model: Model
    private let context: MetalContext
    private let tokenizer: LFM2Tokenizer
    private let runner: LFM2ForwardRunner
    private let scratch: RawCompletionScratch
    private let maxContext: Int

    private var messages: [ChatMessage] = []
    private var kvTokenIDs: [Int32] = []
    private var pending: [String] = []
    private var closed = false
    private var generating = false
    private var lineageBroken = false
    private var uncommittedBoundary: [Int32] = []
    private var boundaryNeedsReplay = false

    public init(model: Model,
                context: MetalContext,
                tokenizer: LFM2Tokenizer,
                runner: LFM2ForwardRunner,
                scratch: RawCompletionScratch,
                maxContext: Int) {
        self.model = model
        self.context = context
        self.tokenizer = tokenizer
        self.runner = runner
        self.scratch = scratch
        self.maxContext = maxContext
    }

    public var kvTokenCount: Int { kvTokenIDs.count }
    public var isGenerating: Bool { generating }
    public var hasStagedTurn: Bool { !pending.isEmpty }
    public var isClosed: Bool { closed }
    public var isUsable: Bool { !closed && !lineageBroken }

    public static func requireTextOnly(parts: [MultimodalContinuationPart],
                                       images: [URL]) throws {
        guard images.isEmpty else {
            throw LFM2ConversationError.imagesUnsupported
        }
        if parts.contains(where: { if case .image = $0 { true } else { false } }) {
            throw LFM2ConversationError.imagesUnsupported
        }
    }

    public func append(parts: [MultimodalContinuationPart],
                       images: [URL]) throws {
        try Self.requireTextOnly(parts: parts, images: images)
        guard !closed else { throw LFM2ConversationError.closed }
        var text = ""
        for part in parts {
            if case .text(let value) = part { text += value }
        }
        guard !text.isEmpty else { throw LFM2ConversationError.emptyTurn }
        pending.append(text)
    }

    public func send(parts: [MultimodalContinuationPart],
                     images: [URL],
                     config: GenerationConfig,
                     prefillConfig: PrefillRuntimeConfig,
                     checkCancellation: @Sendable () throws -> Void,
                     shouldStop: (@Sendable () -> Bool)?,
                     onProgress: (@Sendable (RawDecodeProgress) -> Void)?) async throws -> MultimodalTurnResult {
        try append(parts: parts, images: images)
        return try await generate(config: config,
                                  prefillConfig: prefillConfig,
                                  checkCancellation: checkCancellation,
                                  shouldStop: shouldStop,
                                  onProgress: onProgress)
    }

    private func generate(config: GenerationConfig,
                          prefillConfig: PrefillRuntimeConfig,
                          checkCancellation: @Sendable () throws -> Void,
                          shouldStop: (@Sendable () -> Bool)?,
                          onProgress: (@Sendable (RawDecodeProgress) -> Void)?) async throws -> MultimodalTurnResult {
        guard !closed else { throw LFM2ConversationError.closed }
        guard !lineageBroken else { throw LFM2ConversationError.lineageBroken }
        guard !generating else { throw LFM2ConversationError.busy }
        guard !pending.isEmpty else { throw LFM2ConversationError.emptyTurn }
        try checkCancellation()

        let text = pending.joined()
        pending.removeAll(keepingCapacity: true)
        let user = ChatMessage(role: .user, content: text)

        let promptIDs: [Int32]
        let start: RawCompletionStart
        let usedTokenOnlyHistory = !kvTokenIDs.isEmpty && messages.isEmpty
        let positionBefore = runner.continuationPosition
        if kvTokenIDs.isEmpty {
            promptIDs = try tokenizer.applyChatTemplate(
                messages: [user], tools: nil, addGenerationPrompt: true)
            start = .reset
        } else if usedTokenOnlyHistory {
            guard runner.continuationPosition == kvTokenIDs.count else {
                throw LFM2ConversationError.continuationMismatch
            }
            promptIDs = kvTokenIDs + tokenizer.encodeTextContinuation(userContent: text)
            start = .resume(cachedPromptTokens: kvTokenIDs.count)
        } else {
            let incoming = messages + [user]
            let full = try tokenizer.applyChatTemplate(
                messages: incoming, tools: nil, addGenerationPrompt: true)
            let plan = try tokenizer.continuationPlan(
                cachedMessages: messages,
                cachedTokenIDs: kvTokenIDs,
                incomingMessages: incoming,
                tools: nil)
            switch plan {
            case .reuse(let prefix, let suffix):
                if prefix == kvTokenIDs,
                   prefix.count == runner.continuationPosition {
                    promptIDs = prefix + suffix
                    start = .resume(cachedPromptTokens: prefix.count)
                } else {
                    promptIDs = try replay(full)
                    start = .reset
                }
            case .replayAll(let all):
                promptIDs = try replay(all)
                start = .reset
            }
        }

        guard ConversationGenerationReserve.fits(
            tokens: promptIDs.count, maxContext: maxContext) else {
            throw LFM2ConversationError.contextExhausted(
                prompt: promptIDs.count + ConversationGenerationReserve.turnEnvelope,
                reserve: ConversationGenerationReserve.tokens,
                maxContext: maxContext)
        }

        var runConfig = config
        runConfig.maxNewTokens = min(config.maxNewTokens,
                                     maxContext - promptIDs.count)
        let decoder = LFM2AssistantDecoder(allowedTools: [])
        var visible = ""
        var decodeError: (any Error)?
        var stop = false
        var lastReportedIndex = 0
        let result: RawDecodeResult
        do {
            result = try await runRawCompletion(
                producer: runner,
                tokenizer: tokenizer,
                promptIds: promptIDs,
                config: runConfig,
                context: context,
                scratch: scratch,
                prefillConfig: prefillConfig,
                start: start,
                shouldStop: {
                    if let shouldStop, shouldStop() { return true }
                    return stop
                },
                onProgress: { progress in
                    switch progress {
                    case .prefill(let done, let total):
                        onProgress?(.prefill(done: done, total: total))
                    case .reasoning(let flag):
                        onProgress?(.reasoning(flag))
                    case .token(let index, let id, let delta):
                        do {
                            let events = try decoder.consume(tokenID: id, delta: delta)
                            lastReportedIndex = index
                            for event in events {
                                switch event {
                                case .content(let chunk):
                                    visible += chunk
                                    onProgress?(.token(index: index, id: id, delta: chunk))
                                case .reasoning(let flag):
                                    onProgress?(.reasoning(flag))
                                case .toolCall:
                                    break
                                }
                            }
                        } catch {
                            decodeError = error
                            stop = true
                        }
                    case .tail(let tail):
                        do {
                            let events = try decoder.consumeTail(tail)
                            for event in events {
                                switch event {
                                case .content(let chunk):
                                    visible += chunk
                                    onProgress?(.token(index: lastReportedIndex, id: 0, delta: chunk))
                                case .reasoning(let flag):
                                    onProgress?(.reasoning(flag))
                                case .toolCall:
                                    break
                                }
                            }
                        } catch {
                            decodeError = error
                            stop = true
                        }
                    }
                })
        } catch {
            // A failed turn leaves an unknown amount of the KV advanced. Rewind
            // only within the runner's conv-state limit; a deeper turn condemns
            // the lineage, and the app shows New Chat instead of corrupting it.
            if (try? runner.rewind(to: positionBefore)) == nil {
                lineageBroken = true
            }
            pending.insert(text, at: 0)
            throw error
        }

        if let decodeError {
            if (try? runner.rewind(to: positionBefore)) == nil {
                lineageBroken = true
            }
            pending.insert(text, at: 0)
            throw decodeError
        }
        do {
            try decoder.finish()
        } catch {
            if (try? runner.rewind(to: positionBefore)) == nil {
                lineageBroken = true
            }
            pending.insert(text, at: 0)
            throw error
        }

        kvTokenIDs = result.kvBackedTokenIDs
        if !usedTokenOnlyHistory {
            messages = messages + [user,
                                   ChatMessage(role: .assistant, content: visible)]
        }
        uncommittedBoundary = result.uncommittedBoundaryTokenIDs
        boundaryNeedsReplay = result.reason == .maxTokens || result.reason == .cancelled

        return MultimodalTurnResult(
            text: visible,
            promptTokens: promptIDs.count,
            cachedTokens: result.cachedPromptTokens,
            computedPrefillTokens: result.computedPrefillTokens,
            kvTokens: kvTokenIDs.count,
            completionTokens: result.newTokens,
            reason: result.reason,
            prefillSeconds: result.prefillSeconds,
            decodeSeconds: result.decodeSeconds,
            promptTokenIDs: promptIDs,
            generatedTokenIDs: Array(result.kvBackedTokenIDs.dropFirst(promptIDs.count)),
            uncommittedBoundaryTokenIDs: uncommittedBoundary,
            boundaryNeedsReplay: boundaryNeedsReplay)
    }

    private func replay(_ full: [Int32]) throws -> [Int32] {
        runner.reset()
        return full
    }

    public func restore(lineage: MultimodalConversationLineage,
                        prefillConfig: PrefillRuntimeConfig,
                        checkCancellation: @Sendable () throws -> Void,
                        onProgress: (@Sendable (RawDecodeProgress) -> Void)?) async throws {
        guard !closed else { throw LFM2ConversationError.closed }
        guard !lineageBroken else { throw LFM2ConversationError.lineageBroken }
        guard !generating else { throw LFM2ConversationError.busy }
        guard kvTokenIDs.isEmpty else { throw LFM2ConversationError.busy }
        guard lineage.images.isEmpty else {
            throw LFM2ConversationError.imagesUnsupported
        }
        guard !lineage.tokenIDs.isEmpty else {
            throw LFM2ConversationError.emptyTurn
        }
        try lineage.validateTokenIDs(vocabSize: model.config.vocabSize)
        guard ConversationGenerationReserve.fitsLineage(
            tokens: lineage.promptPrefixCount, maxContext: maxContext) else {
            throw LFM2ConversationError.contextExhausted(
                prompt: lineage.promptPrefixCount
                    + ConversationGenerationReserve.turnEnvelope,
                reserve: ConversationGenerationReserve.tokens,
                maxContext: maxContext)
        }
        generating = true
        defer { generating = false }
        do {
            _ = try await runRawPrefill(
                producer: runner,
                promptIds: lineage.tokenIDs,
                multimodalInput: nil,
                prefillConfig: prefillConfig,
                start: .reset,
                outputMode: .greedyIfAvailable,
                seedUse: .discarded,
                historyReserve: lineage.tokenIDs.count,
                scratch: scratch,
                onProgress: { onProgress?($0) })
        } catch {
            runner.reset()
            kvTokenIDs.removeAll(keepingCapacity: true)
            uncommittedBoundary = []
            boundaryNeedsReplay = false
            throw error
        }
        kvTokenIDs = lineage.tokenIDs
        uncommittedBoundary = lineage.uncommittedBoundary
        boundaryNeedsReplay = lineage.boundaryNeedsReplay
    }

    public func reset() async {
        guard !generating else { return }
        runner.reset()
        kvTokenIDs.removeAll(keepingCapacity: true)
        messages.removeAll(keepingCapacity: true)
        pending.removeAll(keepingCapacity: true)
        uncommittedBoundary = []
        boundaryNeedsReplay = false
        lineageBroken = false
    }

    public func invalidate() async {
        await reset()
    }

    public func close() async {
        closed = true
        await reset()
    }
}
