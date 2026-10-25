import Testing
@testable import TurboFieldfare
import TurboFieldfareFormat

@Suite("LFM streaming detokenizer")
struct LFM2DetokenizerTests {
    @Test func detokenizerBuffersSplitUTF8() {
        let tokenizer = ByteLevelStubTokenizer()
        var detokenizer = LFM2Detokenizer(tokenizer: tokenizer)
        // 0xE2 0x82 0xAC = "€", split across three byte tokens.
        #expect(detokenizer.push(1).isEmpty)
        #expect(detokenizer.push(2).isEmpty)
        #expect(detokenizer.push(3) == "€")
        #expect(detokenizer.flush().isEmpty)
    }

    @Test func detokenizerPreservesMarkersAndDropsSpecials() {
        let tokenizer = ByteLevelStubTokenizer()
        var detokenizer = LFM2Detokenizer(tokenizer: tokenizer)
        #expect(detokenizer.push(124_902) == "</think>")
        #expect(detokenizer.push(124_900).isEmpty)
    }
}

/// Minimal byte-level stand-in. The real LFM2Tokenizer arrives in Task 17; it
/// conforms to the same raw-byte protocol so the detokenizer can assemble
/// multi-byte UTF-8 before a token's String decoding would replace the bytes.
struct ByteLevelStubTokenizer: ModelTokenizer, LFM2ByteLevelVocabulary {
    var family: ModelFamily { .lfm2Moe }
    var bosID: Int32 { 124_894 }
    var eosID: Int32 { 124_900 }
    var padID: Int32 { 124_893 }
    var endOfTurnID: Int32 { 124_900 }
    var toolResponseID: Int32 { 124_900 }
    var toolCallStartID: Int32 { 124_905 }
    var toolCallEndID: Int32 { 124_906 }
    var toolResponseEndID: Int32 { 124_900 }
    var stopTokenIDs: Set<Int32> { [124_900] }
    var structuralMarkerIDs: Set<Int32> { [124_901, 124_902, 124_905, 124_906] }
    var vocabSize: Int { 128_000 }
    var chatTemplateIdentity: String { "lfm-stub" }

    func encode(_ text: String, addBOS: Bool) -> [Int32] { [] }

    func decode(_ ids: [Int32], skipSpecialTokens: Bool) -> String {
        ids.compactMap { id in
            let bytes = byteFragment(forTokenID: id) ?? []
            return String(decoding: bytes, as: UTF8.self)
        }.joined()
    }

    func makeStreamingDetokenizer() -> any StreamingDetokenizer {
        LFM2Detokenizer(tokenizer: self)
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
        LFM2AssistantDecoder(allowedTools: allowedTools)
    }

    func byteFragment(forTokenID id: Int32) -> [UInt8]? {
        switch id {
        case 1: [0xE2]
        case 2: [0x82]
        case 3: [0xAC]
        case 124_901: Array("<think>".utf8)
        case 124_902: Array("</think>".utf8)
        case 124_905: Array("<|tool_call_start|>".utf8)
        case 124_906: Array("<|tool_call_end|>".utf8)
        default: nil
        }
    }
}
