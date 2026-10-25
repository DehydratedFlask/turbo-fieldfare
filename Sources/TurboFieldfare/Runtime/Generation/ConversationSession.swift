import Foundation

/// The app's family-neutral view of an open conversation. Gemma's
/// `MultimodalConversation` and LFM's `LFM2Conversation` each own a KV lineage;
/// the session switches on the loaded model's family rather than on a
/// compile-time default.
public protocol ConversationSession: AnyObject, Sendable {
    var kvTokenCount: Int { get async }
    var isGenerating: Bool { get async }
    var hasStagedTurn: Bool { get async }
    var isClosed: Bool { get async }
    var isUsable: Bool { get async }

    func invalidate() async
    func reset() async
    func close() async
    func append(parts: [MultimodalContinuationPart], images: [URL]) async throws
    func send(parts: [MultimodalContinuationPart],
              images: [URL],
              config: GenerationConfig,
              prefillConfig: PrefillRuntimeConfig,
              checkCancellation: @Sendable () throws -> Void,
              shouldStop: (@Sendable () -> Bool)?,
              onProgress: (@Sendable (RawDecodeProgress) -> Void)?) async throws -> MultimodalTurnResult
    func restore(lineage: MultimodalConversationLineage,
                 prefillConfig: PrefillRuntimeConfig,
                 checkCancellation: @Sendable () throws -> Void,
                 onProgress: (@Sendable (RawDecodeProgress) -> Void)?) async throws
}

public extension ConversationSession {
    func append(parts: [MultimodalContinuationPart]) async throws {
        try await append(parts: parts, images: [])
    }

    func send(parts: [MultimodalContinuationPart],
              images: [URL] = [],
              config: GenerationConfig,
              prefillConfig: PrefillRuntimeConfig = .defaultChunked,
              checkCancellation: @Sendable () throws -> Void = {},
              shouldStop: (@Sendable () -> Bool)? = nil,
              onProgress: (@Sendable (RawDecodeProgress) -> Void)? = nil) async throws -> MultimodalTurnResult {
        try await send(parts: parts, images: images, config: config,
                       prefillConfig: prefillConfig,
                       checkCancellation: checkCancellation,
                       shouldStop: shouldStop,
                       onProgress: onProgress)
    }

    func restore(lineage: MultimodalConversationLineage,
                 prefillConfig: PrefillRuntimeConfig = .defaultChunked,
                 checkCancellation: @Sendable () throws -> Void = {},
                 onProgress: (@Sendable (RawDecodeProgress) -> Void)? = nil) async throws {
        try await restore(lineage: lineage,
                          prefillConfig: prefillConfig,
                          checkCancellation: checkCancellation,
                          onProgress: onProgress)
    }
}
