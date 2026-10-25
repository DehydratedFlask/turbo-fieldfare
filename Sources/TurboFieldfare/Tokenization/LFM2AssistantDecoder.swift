import Foundation

public enum LFM2AssistantDecoderError: Error, Equatable {
    /// The structured stream ended in a state that cannot be returned as a
    /// valid assistant turn (open tool call, oversized tool call).
    case malformed(detail: String)
    /// The suppressed reasoning channel exceeded its 64 KiB budget.
    case reasoningOverflow
}

/// Routes LFM's ChatML markers into visible content, tool calls, and the
/// reasoning channel.
///
/// The generation prompt already opens `<think>`, so a fresh decoder starts in
/// the reasoning channel. Reasoning text is counted (for the cap) and thrown
/// away: it is never returned as content or stored as history. `</think>`
/// switches to the visible channel. A tool call is accepted in either channel;
/// the turn ends as a tool-call turn when one is parsed.
public final class LFM2AssistantDecoder: StructuredOutputDecoder, @unchecked Sendable {
    public static let maximumReasoningBytes = 64 * 1024

    private enum Channel {
        case reasoning
        case visible
    }

    private static let thinkOpen = "<think>"
    private static let thinkClose = "</think>"
    private static let toolCallOpen = "<|tool_call_start|>"
    private static let toolCallClose = "<|tool_call_end|>"

    private let allowedTools: Set<String>
    private var channel: Channel = .reasoning
    private var announcedReasoning = false
    private var reasoningBytes = 0
    private var reasoningOverflowed = false
    private var toolBody: String?
    private var emittedCalls = 0
    private var ended = false
    private var failure: LFM2AssistantDecoderError?

    public init(allowedTools: Set<String>) {
        self.allowedTools = allowedTools
    }

    public func consume(tokenID: Int32, delta: String) throws -> [StructuredAssistantEvent] {
        try rethrowFailure()
        guard !ended else { return [] }

        switch tokenID {
        case 124_901:
            return route(stripping(Self.thinkOpen, from: delta))
        case 124_902:
            var events = route(stripping(Self.thinkClose, from: delta))
            if channel == .reasoning {
                channel = .visible
                events.append(.reasoning(false))
            }
            return events
        case 124_905:
            guard toolBody == nil else {
                throw fail(.malformed(detail: "nested tool call"))
            }
            let events = route(stripping(Self.toolCallOpen, from: delta))
            toolBody = ""
            return events
        case 124_906:
            guard var body = toolBody else {
                throw fail(.malformed(detail: "tool call end without start"))
            }
            body += stripping(Self.toolCallClose, from: delta)
            toolBody = nil
            return try parseToolCalls(body)
        case 124_900:
            let events = route(delta)
            ended = true
            return events
        default:
            if toolBody != nil {
                try appendToToolBody(delta)
                return []
            }
            return route(delta)
        }
    }

    public func consumeTail(_ text: String) throws -> [StructuredAssistantEvent] {
        try rethrowFailure()
        guard !ended, !text.isEmpty else { return [] }
        if toolBody != nil {
            try appendToToolBody(text)
            return []
        }
        return route(text)
    }

    public func finish() throws {
        try rethrowFailure()
        if toolBody != nil {
            throw LFM2AssistantDecoderError.malformed(detail: "unterminated tool call")
        }
        if reasoningOverflowed {
            throw LFM2AssistantDecoderError.reasoningOverflow
        }
    }

    public var hasToolCalls: Bool { emittedCalls > 0 }

    // MARK: - Routing

    private func route(_ text: String) -> [StructuredAssistantEvent] {
        guard !text.isEmpty else { return [] }
        switch channel {
        case .reasoning:
            var events: [StructuredAssistantEvent] = []
            if !announcedReasoning {
                announcedReasoning = true
                events.append(.reasoning(true))
            }
            guard !reasoningOverflowed else { return events }
            reasoningBytes += text.utf8.count
            if reasoningBytes > Self.maximumReasoningBytes {
                reasoningOverflowed = true
            }
            return events
        case .visible:
            return [.content(text)]
        }
    }

    private func appendToToolBody(_ text: String) throws {
        guard var body = toolBody else { return }
        body += text
        guard body.utf8.count <= LFMToolCallParser.maximumBytes else {
            throw fail(.malformed(detail: "tool call body too large"))
        }
        toolBody = body
    }

    private func parseToolCalls(_ body: String) throws -> [StructuredAssistantEvent] {
        let calls: [ParsedToolCall]
        do {
            calls = try LFMToolCallParser().parse(
                body, allowedTools: allowedTools, id: "call_\(emittedCalls)")
        } catch LFMToolCallParserError.malformed(detail: "no tool calls") {
            return []
        } catch {
            throw fail(.malformed(detail: "invalid tool call body"))
        }

        var events: [StructuredAssistantEvent] = []
        for call in calls {
            let id = "call_\(emittedCalls)"
            emittedCalls += 1
            events.append(.toolCall(ParsedToolCall(
                id: id,
                name: call.name,
                arguments: call.arguments,
                argumentsJSON: call.argumentsJSON)))
        }
        return events
    }

    /// Text held back from before a marker is returned in the marker token's
    /// delta; the marker's own decoded text is not content.
    private func stripping(_ marker: String, from delta: String) -> String {
        guard delta.hasSuffix(marker) else { return delta }
        return String(delta.dropLast(marker.count))
    }

    private func fail(_ error: LFM2AssistantDecoderError) -> LFM2AssistantDecoderError {
        failure = error
        return error
    }

    private func rethrowFailure() throws {
        if let failure { throw failure }
    }
}
