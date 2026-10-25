import Foundation

public enum LFM2ChatRendererError: Error, CustomStringConvertible {
    case systemMustBeFirst
    case toolArgumentsNotMapping
    case invalidNumber

    public var description: String {
        switch self {
        case .systemMustBeFirst:
            return "system message must be first"
        case .toolArgumentsNotMapping:
            return "tool-call arguments must be a mapping"
        case .invalidNumber:
            return "tool-call argument is not a finite number"
        }
    }
}

/// Pure renderer for the pinned LFM2.5 ChatML template.
///
/// The template is small enough to render without a Jinja engine, and keeping
/// it as a pure function lets the exact framing be asserted without a 17 MB
/// tokenizer fixture. It is pinned to the installed `chat_template.jinja`
/// identity; a template revision fails load instead of drifting silently.
public struct LFM2ChatRenderer: Sendable {
    public init() {}

    public func render(messages: [ChatMessage],
                       tools: [ToolDefinition]?,
                       addGenerationPrompt: Bool) throws -> String {        var output = "<|startoftext|>"
        var firstRendered = 0

        if let first = messages.first, first.role == .system {
            let system = try renderSystem(first, tools: tools)
            if !system.isEmpty { output += system }
            firstRendered = 1
        } else if let tools, !tools.isEmpty {
            output += "<|im_start|>system\nList of tools: "
                + (try toolsJSON(tools)) + "<|im_end|>\n"
        }

        for message in messages.dropFirst(firstRendered) {
            guard message.role != .system else {
                throw LFM2ChatRendererError.systemMustBeFirst
            }
            output += try renderMessage(message)
        }

        if addGenerationPrompt {
            output += "<|im_start|>assistant\n<think>"
        }
        return output
    }

    /// Appends a user turn to a KV that already ends at an assistant turn's
    /// last visible token (the `<|im_end|>` stop was withheld). Used after a
    /// restored conversation, where the token history exists but the message
    /// text was never loaded.
    public func continuationPrompt(userContent: String) -> String {
        "<|im_end|>\n<|im_start|>user\n\(userContent)<|im_end|>\n"
            + "<|im_start|>assistant\n<think>"
    }

    private func renderSystem(_ message: ChatMessage,
                              tools: [ToolDefinition]?) throws -> String {
        var body = message.content ?? ""
        if let tools, !tools.isEmpty {
            if !body.isEmpty { body += "\n" }
            body += "List of tools: " + (try toolsJSON(tools))
        }
        guard !body.isEmpty else { return "" }
        return "<|im_start|>system\n\(body)<|im_end|>\n"
    }

    private func renderMessage(_ message: ChatMessage) throws -> String {
        var body = message.content ?? ""
        if message.role == .assistant {
            body = droppingHistoricalThinking(body)
        }
        if !message.toolCalls.isEmpty {
            body += try renderToolCalls(message.toolCalls)
        }
        return "<|im_start|>\(message.role.rawValue)\n\(body)<|im_end|>\n"
    }

    /// The v1 decoder never stores reasoning, but a caller may still replay a
    /// transcript that contains it. Match the template's default
    /// `preserve_thinking = false`: keep only the text after `</think>`.
    private func droppingHistoricalThinking(_ content: String) -> String {
        guard let close = content.range(of: "</think>", options: .backwards) else {
            return content
        }
        return String(content[close.upperBound...])
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func renderToolCalls(_ calls: [GFTokenizer.HistoricalToolCall]) throws -> String {
        let rendered = try calls.map { call in
            "\(call.name)(\(try pythonArguments(call.arguments)))"
        }
        return "<|tool_call_start|>[\(rendered.joined(separator: ", "))]<|tool_call_end|>"
    }

    private func pythonArguments(_ value: JSONValue) throws -> String {
        guard case .object(let pairs) = value else {
            throw LFM2ChatRendererError.toolArgumentsNotMapping
        }
        return try pairs.keys.sorted().map { key in
            "\(key)=\(try pythonValue(pairs[key]!))"
        }.joined(separator: ", ")
    }

    private func pythonValue(_ value: JSONValue) throws -> String {
        switch value {
        case .string(let text):
            let escaped = text
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "'", with: "\\'")
                .replacingOccurrences(of: "\n", with: "\\n")
                .replacingOccurrences(of: "\r", with: "\\r")
            return "'\(escaped)'"
        case .object, .array:
            return try value.encoded(sortedKeys: true)
        case .integer(let number):
            return String(number)
        case .unsignedInteger(let number):
            return String(number)
        case .decimal(let number):
            return NSDecimalNumber(decimal: number).stringValue
        case .number(let number):
            guard number.isFinite else { throw LFM2ChatRendererError.invalidNumber }
            return String(number)
        case .bool(let flag):
            return flag ? "True" : "False"
        case .null:
            return "None"
        }
    }

    private func toolsJSON(_ tools: [ToolDefinition]) throws -> String {
        let values = tools.map { tool in
            JSONValue.object([
                "type": .string("function"),
                "function": .object([
                    "name": .string(tool.name),
                    "description": .string(tool.description),
                    "parameters": tool.parameters,
                ]),
            ])
        }
        return try JSONValue.array(values).encoded(sortedKeys: true)
    }
}
