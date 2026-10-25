import TurboFieldfareFormat

/// A chat message in the family-neutral shape every front end speaks. The
/// Gemma renderer is the reference shape; the LFM renderer accepts the same
/// messages so the CLI, server, and app never branch on family.
public typealias ChatMessage = GFTokenizer.Message

/// A tool definition in the family-neutral shape every front end speaks.
public typealias ToolDefinition = GFTokenizer.FunctionDefinition

/// Per-token detokenization contract of the generation loop.
///
/// The loop owns one existential and pushes every generated token, so the
/// family's own decoder decides what a token contributes: a Gemma byte-fallback
/// fragment, an LFM byte-level fragment, or nothing for a stripped special.
public protocol StreamingDetokenizer {
    mutating func push(_ tokenID: Int32) -> String
    mutating func flush() -> String
}

/// How a new request continues the token stream a KV cache already holds.
///
/// `reuse` means the KV is still valid and only `suffixTokens` must be
/// prefilled. Gemma returns an empty `prefixTokens` because its continuation
/// encoders append the bridge at the current KV position; a tokenizer that
/// compares full renders (LFM) returns the prefix it verified.
/// `replayAll` is the correctness fallback: reset and prefill the whole
/// rendered request.
public enum ContinuationPlan: Sendable, Equatable {
    case reuse(prefixTokens: [Int32], suffixTokens: [Int32])
    case replayAll([Int32])
}

/// Family-neutral tokenizer surface for generation, chat rendering, and
/// prompt continuation.
///
/// This is the target surface for every runtime path that used to name
/// `GFTokenizer`. The paths migrated so far are `runRawCompletion` and the
/// tokenizer/decoder factories; `ServerPromptCache`, the conversation identity
/// and lineage code, and `MultimodalConversation` still take a concrete
/// `GFTokenizer`, so a second family is not yet accepted end to end. Each
/// remaining path moves over as its own task adds a conformance, rather than
/// branching on family at the call site.
public protocol ModelTokenizer: Sendable {
    var family: ModelFamily { get }
    var bosID: Int32 { get }
    var eosID: Int32 { get }
    var padID: Int32 { get }
    var endOfTurnID: Int32 { get }
    var toolResponseID: Int32 { get }
    var toolCallStartID: Int32 { get }
    var toolCallEndID: Int32 { get }
    var toolResponseEndID: Int32 { get }
    /// Every token that ends a turn: EOS plus any family-specific markers.
    var stopTokenIDs: Set<Int32> { get }
    /// Tokens that structure assistant output. Streaming treats them as
    /// detokenizer barriers so held-back bytes cannot surface on the wrong
    /// side of a channel switch.
    var structuralMarkerIDs: Set<Int32> { get }
    var vocabSize: Int { get }
    /// Identity of the chat template whose rendering the conversation
    /// lineage depends on. A template change invalidates saved token links.
    var chatTemplateIdentity: String { get }

    func encode(_ text: String, addBOS: Bool) -> [Int32]
    func decode(_ ids: [Int32], skipSpecialTokens: Bool) -> String
    func makeStreamingDetokenizer() -> any StreamingDetokenizer
    func applyChatTemplate(messages: [ChatMessage], tools: [ToolDefinition]?,
                           addGenerationPrompt: Bool) throws -> [Int32]
    /// `cachedMessages` is the exact message sequence whose tokens the KV
    /// holds, including the assistant turn. `cachedTokenIDs` is exactly the
    /// token sequence the KV holds; an implementation may use it to find the
    /// reuse boundary. `incomingMessages` is the full new request. A Gemma
    /// implementation may return an empty `prefixTokens` meaning "append the
    /// suffix at the current KV position"; an LFM implementation returns the
    /// full rendered prefix it compared.
    func continuationPlan(cachedMessages: [ChatMessage],
                          cachedTokenIDs: [Int32],
                          incomingMessages: [ChatMessage],
                          tools: [ToolDefinition]?) throws -> ContinuationPlan
    /// The family's structured-output router when it has one; `nil` keeps the
    /// raw text stream.
    func makeAssistantDecoder(allowedTools: Set<String>) -> (any StructuredOutputDecoder)?
}

/// Streaming consumer of generated tokens that turns the family's markers into
/// routed assistant events (visible content, tool calls, reasoning state).
public protocol StructuredOutputDecoder: AnyObject, Sendable {
    func consume(tokenID: Int32, delta: String) throws -> [StructuredAssistantEvent]
    func consumeTail(_ text: String) throws -> [StructuredAssistantEvent]
    func finish() throws
    var hasToolCalls: Bool { get }
}
