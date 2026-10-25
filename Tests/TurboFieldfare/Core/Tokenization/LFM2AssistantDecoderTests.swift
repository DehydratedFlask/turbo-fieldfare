import Testing
@testable import TurboFieldfare

@Suite("LFM assistant decoder")
struct LFM2AssistantDecoderTests {
    private func makeDecoder(allowedTools: Set<String> = []) -> LFM2AssistantDecoder {
        LFM2AssistantDecoder(allowedTools: allowedTools)
    }

    @Test func startsInReasoningAndSwitchesOnCloseMarker() throws {
        let decoder = makeDecoder()
        #expect(try decoder.consume(tokenID: 100, delta: "thinking") == [.reasoning(true)])
        #expect(try decoder.consume(tokenID: 124_902, delta: "") == [.reasoning(false)])
        #expect(try decoder.consume(tokenID: 101, delta: "answer") == [.content("answer")])
    }

    @Test func acceptsToolCallDuringReasoning() throws {
        let decoder = makeDecoder(allowedTools: ["weather"])
        _ = try decoder.consume(tokenID: 100, delta: "hmm")
        _ = try decoder.consume(tokenID: 124_905, delta: "")
        _ = try decoder.consume(tokenID: 200, delta: "[weather(city='Paris')]")
        let events = try decoder.consume(tokenID: 124_906, delta: "")
        #expect(events.contains { if case .toolCall = $0 { true } else { false } })
        #expect(decoder.hasToolCalls)
    }

    @Test func eosBeforeCloseEmitsNoVisibleContent() throws {
        let decoder = makeDecoder()
        _ = try decoder.consume(tokenID: 100, delta: "hidden")
        #expect(try decoder.consume(tokenID: 124_900, delta: "").isEmpty)
        #expect(throws: Never.self) { try decoder.finish() }
    }

    @Test func reasoningTextIsCappedAndDiscarded() throws {
        let decoder = makeDecoder()
        let chunk = String(repeating: "x", count: 1024)
        for _ in 0..<65 {
            _ = try decoder.consume(tokenID: 100, delta: chunk)
        }
        #expect(throws: LFM2AssistantDecoderError.self) { try decoder.finish() }
    }
}
