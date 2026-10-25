import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfare

@Suite("LFM ChatML renderer")
struct LFM2ChatRendererTests {
    private let renderer = LFM2ChatRenderer()

    @Test func rendersSystemUserAndGenerationPrompt() throws {
        let text = try renderer.render(
            messages: [.init(role: .system, content: "S"),
                       .init(role: .user, content: "U")],
            tools: nil, addGenerationPrompt: true)
        #expect(text == "<|startoftext|><|im_start|>system\nS<|im_end|>\n"
            + "<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n thinking")
    }

    @Test func rendersToolCallsWithPythonLiterals() throws {
        let call = GFTokenizer.HistoricalToolCall(
            id: "call_1",
            name: "weather",
            arguments: .object(["city": .string("Paris"), "metric": .bool(true),
                                 "limit": .integer(3), "note": .null]))
        let text = try renderer.render(
            messages: [.init(role: .assistant, content: nil, toolCalls: [call])],
            tools: nil, addGenerationPrompt: false)
        #expect(text.contains("<|tool_call_start|>[weather(city='Paris', limit=3, metric=True, note=None)]<|tool_call_end|>"))
    }

    @Test func rendersToolDefinitionsInTheSystemBlock() throws {
        let tool = GFTokenizer.FunctionDefinition(
            name: "weather",
            description: "Look up weather",
            parameters: .object([
                "type": .string("object"),
                "properties": .object(["city": .object(["type": .string("string")])]),
            ]))
        let text = try renderer.render(
            messages: [.init(role: .user, content: "U")],
            tools: [tool], addGenerationPrompt: false)
        #expect(text.hasPrefix("<|startoftext|><|im_start|>system\nList of tools: ["))
        #expect(text.contains("\"type\":\"function\""))
        #expect(text.contains("\"name\":\"weather\""))
        #expect(text.contains("<|im_start|>user\nU<|im_end|>\n"))
    }

    @Test func dropsHistoricalThinkingUnlessPreserved() throws {
        let text = try renderer.render(
            messages: [.init(role: .assistant,
                             content: "hidden</think>visible")],
            tools: nil, addGenerationPrompt: false)
        #expect(!text.contains("hidden"))
        #expect(text.contains("visible"))
    }

    @Test func rejectsASystemMessageThatIsNotFirst() {
        #expect(throws: LFM2ChatRendererError.self) {
            _ = try renderer.render(
                messages: [.init(role: .user, content: "U"),
                           .init(role: .system, content: "S")],
                tools: nil, addGenerationPrompt: false)
        }
    }
}
