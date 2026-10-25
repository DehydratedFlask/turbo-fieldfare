import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfare

/// `GFTokenizer` is the Gemma implementation of the family-neutral tokenizer
/// surface. These cases pin that adopting `ModelTokenizer` did not change any
/// Gemma behavior: the IDs, stop set, template identity, rendered prompt IDs,
/// continuation bridge, and per-token detokenization stay exactly what the
/// runtime emitted before the abstraction existed.
@Suite("ModelTokenizer")
struct ModelTokenizerTests {

    @Test func gemmaTokenizerConformsWithoutChangingBehavior() async throws {
        let tokenizer = try await GFTokenizer.load()
        #expect(tokenizer.family == .gemma4)
        #expect(tokenizer.chatTemplateIdentity == GFTokenizer.chatTemplateIdentity)
        #expect(tokenizer.stopTokenIDs == [tokenizer.eosID,
                                           tokenizer.endOfTurnID,
                                           tokenizer.toolResponseID])
        #expect(tokenizer.makeStreamingDetokenizer() is GFDetokenizer)
        #expect(tokenizer.makeAssistantDecoder(allowedTools: []) is StructuredAssistantDecoder)
        #expect(tokenizer.structuralMarkerIDs == [tokenizer.toolCallStartID,
                                                  tokenizer.toolCallEndID,
                                                  tokenizer.toolResponseID,
                                                  tokenizer.toolResponseEndID,
                                                  tokenizer.channelStartID,
                                                  tokenizer.channelEndID])
    }

    @Test func gemmaTemplateOverloadKeepsTheStringRendering() async throws {
        let tokenizer = try await GFTokenizer.load()
        let messages: [ChatMessage] = [
            .init(role: .system, content: "be brief"),
            .init(role: .user, content: "hello"),
        ]
        let ids = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, addGenerationPrompt: true)
        let rendered = try tokenizer.applyChatTemplate(messages)
        #expect(ids == tokenizer.encode(rendered, addBOS: false))
    }

    @Test func gemmaStreamingDetokenizerMatchesBatchDecode() async throws {
        let tokenizer = try await GFTokenizer.load()
        let ids = tokenizer.encode("the capital of France is", addBOS: false)
        var detokenizer = tokenizer.makeStreamingDetokenizer()
        var streamed = ""
        for id in ids {
            streamed += detokenizer.push(id)
        }
        streamed += detokenizer.flush()
        #expect(streamed == tokenizer.decode(ids, skipSpecialTokens: true))
    }

    @Test func gemmaTextContinuationReusesTheBridge() async throws {
        let tokenizer = try await GFTokenizer.load()
        let cached: [ChatMessage] = [
            .init(role: .user, content: "hi"),
            .init(role: .assistant, content: "hello"),
        ]
        let incoming = cached + [.init(role: .user, content: "again")]
        let plan = try tokenizer.continuationPlan(
            cachedMessages: cached,
            cachedTokenIDs: [1, 2, 3],
            incomingMessages: incoming,
            tools: nil)
        #expect(plan == .reuse(
            prefixTokens: [],
            suffixTokens: tokenizer.encodeTextContinuation(userContent: "again")))
    }

    @Test func gemmaContinuationReplaysWhatItCannotBridge() async throws {
        let tokenizer = try await GFTokenizer.load()
        let incoming: [ChatMessage] = [.init(role: .user, content: "hello")]
        let plan = try tokenizer.continuationPlan(
            cachedMessages: [],
            cachedTokenIDs: [],
            incomingMessages: incoming,
            tools: nil)
        #expect(plan == .replayAll(
            tokenizer.encode(try tokenizer.applyChatTemplate(incoming), addBOS: false)))
    }
}
