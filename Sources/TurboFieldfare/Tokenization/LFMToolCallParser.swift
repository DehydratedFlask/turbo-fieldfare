import Foundation

/// Failures raised while parsing the Python-literal body of an LFM tool call.
///
/// The parser never returns a partial result: any malformed call rejects the
/// whole body so callers cannot accidentally act on half of a model emission.
public enum LFMToolCallParserError: Error, Equatable {
    case malformed(detail: String)
    case unknownTool(String)
    case oversized
    case tooManyCalls
    case tooManyArguments
}

/// Parses the text LFM renders between `<|tool_call_start|>` and
/// `<|tool_call_end|>`, for example:
///
///     [weather(city='Paris', metric=True), search(query="x")]
///
/// Values are Python literals (`'single'` strings, `True`, `False`, `None`)
/// mixed with JSON (`"double"` strings, `true`, `false`, `null`, objects and
/// arrays). Every name must be present in `allowedTools`.
public struct LFMToolCallParser: Sendable {
    public static let maximumBytes = 256 * 1024
    public static let maximumCalls = 16
    public static let maximumArguments = 64

    public init() {}

    public func parse(_ text: String,
                      allowedTools: Set<String>,
                      id: String) throws -> [ParsedToolCall] {
        guard text.utf8.count <= Self.maximumBytes else {
            throw LFMToolCallParserError.oversized
        }
        var scanner = LFMToolCallScanner(text, allowedTools: allowedTools, id: id)
        return try scanner.parseCalls()
    }
}

/// Index-based scanner. Character indexing (rather than regex or repeated
/// `dropFirst`) keeps nested brackets and quoted commas unambiguous.
private struct LFMToolCallScanner {
    private let characters: [Character]
    private let allowedTools: Set<String>
    private let id: String
    private var index = 0

    init(_ text: String, allowedTools: Set<String>, id: String) {
        self.characters = Array(text)
        self.allowedTools = allowedTools
        self.id = id
    }

    private var isAtEnd: Bool { index >= characters.count }

    private var current: Character? {
        index < characters.count ? characters[index] : nil
    }

    // MARK: - Grammar

    mutating func parseCalls() throws -> [ParsedToolCall] {
        skipWhitespace()
        guard !isAtEnd else {
            throw LFMToolCallParserError.malformed(detail: "empty tool call body")
        }
        let hasOuterList = current == "["
        if hasOuterList { index += 1 }

        var calls: [ParsedToolCall] = []
        skipWhitespace()
        if hasOuterList, current == "]" {
            index += 1
        } else {
            while true {
                guard calls.count < LFMToolCallParser.maximumCalls else {
                    throw LFMToolCallParserError.tooManyCalls
                }
                calls.append(try parseCall())
                skipWhitespace()
                if current == "," {
                    index += 1
                    continue
                }
                break
            }
            if hasOuterList { try expect("]") }
        }

        skipWhitespace()
        guard isAtEnd else {
            throw LFMToolCallParserError.malformed(detail: "trailing input after tool calls")
        }
        guard !calls.isEmpty else {
            throw LFMToolCallParserError.malformed(detail: "no tool calls")
        }
        return calls
    }

    private mutating func parseCall() throws -> ParsedToolCall {
        let name = try parseName()
        guard allowedTools.contains(name) else {
            throw LFMToolCallParserError.unknownTool(name)
        }
        try expect("(")

        var arguments: [String: JSONValue] = [:]
        skipWhitespace()
        if current != ")" {
            while true {
                guard arguments.count < LFMToolCallParser.maximumArguments else {
                    throw LFMToolCallParserError.tooManyArguments
                }
                let key = try parseKey()
                try expect("=")
                let value = try parseValue()
                guard arguments.updateValue(value, forKey: key) == nil else {
                    throw LFMToolCallParserError.malformed(
                        detail: "duplicate argument key '\(key)'")
                }
                skipWhitespace()
                if current == "," {
                    index += 1
                    continue
                }
                break
            }
        }
        try expect(")")

        let encodedArguments = JSONValue.object(arguments)
        return ParsedToolCall(id: id,
                              name: name,
                              arguments: encodedArguments,
                              argumentsJSON: try encodedArguments.encoded())
    }

    private mutating func parseName() throws -> String {
        skipWhitespace()
        let start = index
        guard let first = current, isNameStart(first) else {
            throw LFMToolCallParserError.malformed(detail: "expected tool name")
        }
        index += 1
        while let character = current, isNameContinuation(character) {
            index += 1
        }
        return String(characters[start..<index])
    }

    /// Argument keys are bare identifiers, or quoted strings when the model
    /// emits a JSON-shaped mapping with an unusual key.
    private mutating func parseKey() throws -> String {
        skipWhitespace()
        if current == "'" || current == "\"" {
            return try parseString()
        }
        let start = index
        guard let first = current, isKeyContinuation(first) else {
            throw LFMToolCallParserError.malformed(detail: "expected argument key")
        }
        index += 1
        while let character = current, isKeyContinuation(character) {
            index += 1
        }
        return String(characters[start..<index])
    }

    private mutating func parseValue() throws -> JSONValue {
        skipWhitespace()
        guard let character = current else {
            throw LFMToolCallParserError.malformed(detail: "expected value")
        }
        switch character {
        case "'", "\"":
            return .string(try parseString())
        case "[":
            return try parseArray()
        case "{":
            return try parseObject()
        default:
            return try parseLiteralOrNumber()
        }
    }

    private mutating func parseArray() throws -> JSONValue {
        try expect("[")
        var values: [JSONValue] = []
        skipWhitespace()
        if current == "]" {
            index += 1
            return .array(values)
        }
        while true {
            values.append(try parseValue())
            skipWhitespace()
            if current == "," {
                index += 1
                continue
            }
            break
        }
        try expect("]")
        return .array(values)
    }

    private mutating func parseObject() throws -> JSONValue {
        try expect("{")
        var object: [String: JSONValue] = [:]
        skipWhitespace()
        if current == "}" {
            index += 1
            return .object(object)
        }
        while true {
            let key = try parseKey()
            try expect(":")
            let value = try parseValue()
            guard object.updateValue(value, forKey: key) == nil else {
                throw LFMToolCallParserError.malformed(
                    detail: "duplicate object key '\(key)'")
            }
            skipWhitespace()
            if current == "," {
                index += 1
                continue
            }
            break
        }
        try expect("}")
        return .object(object)
    }

    // MARK: - Scalars

    private mutating func parseLiteralOrNumber() throws -> JSONValue {
        let start = index
        while let character = current, character.isASCII, character.isLetter {
            index += 1
        }
        if index > start {
            let word = String(characters[start..<index])
            switch word {
            case "true", "True": return .bool(true)
            case "false", "False": return .bool(false)
            case "null", "None": return .null
            default:
                throw LFMToolCallParserError.malformed(detail: "unknown literal '\(word)'")
            }
        }
        return try parseNumber()
    }

    private mutating func parseNumber() throws -> JSONValue {
        skipWhitespace()
        let start = index
        if current == "-" || current == "+" { index += 1 }

        var sawDigit = false
        while let character = current, character.isASCIIDigit {
            index += 1
            sawDigit = true
        }
        var isInteger = true
        if current == "." {
            isInteger = false
            index += 1
            var sawFraction = false
            while let character = current, character.isASCIIDigit {
                index += 1
                sawFraction = true
            }
            guard sawFraction else {
                throw LFMToolCallParserError.malformed(detail: "expected digits after decimal point")
            }
        }
        if current == "e" || current == "E" {
            isInteger = false
            index += 1
            if current == "+" || current == "-" { index += 1 }
            var sawExponent = false
            while let character = current, character.isASCIIDigit {
                index += 1
                sawExponent = true
            }
            guard sawExponent else {
                throw LFMToolCallParserError.malformed(detail: "expected digits after exponent")
            }
        }
        guard sawDigit else {
            throw LFMToolCallParserError.malformed(detail: "expected value")
        }

        let text = String(characters[start..<index])
        if isInteger, let value = Int64(text) {
            return .integer(value)
        }
        guard let value = Double(text), value.isFinite else {
            throw LFMToolCallParserError.malformed(detail: "number out of range '\(text)'")
        }
        return .number(value)
    }

    private mutating func parseString() throws -> String {
        guard let quote = current, quote == "'" || quote == "\"" else {
            throw LFMToolCallParserError.malformed(detail: "expected string")
        }
        index += 1
        var result = ""
        while let character = current {
            if character == quote {
                index += 1
                return result
            }
            if character == "\\" {
                index += 1
                result += try parseEscape()
                continue
            }
            result.append(character)
            index += 1
        }
        throw LFMToolCallParserError.malformed(detail: "unterminated string")
    }

    private mutating func parseEscape() throws -> String {
        guard let escape = current else {
            throw LFMToolCallParserError.malformed(detail: "unterminated escape")
        }
        index += 1
        switch escape {
        case "\\": return "\\"
        case "'": return "'"
        case "\"": return "\""
        case "/": return "/"
        case "b": return "\u{8}"
        case "f": return "\u{c}"
        case "n": return "\n"
        case "r": return "\r"
        case "t": return "\t"
        case "u": return try parseUnicodeEscape()
        default:
            throw LFMToolCallParserError.malformed(detail: "unsupported escape '\\\(escape)'")
        }
    }

    private mutating func parseUnicodeEscape() throws -> String {
        let first = try readHexCodeUnit()
        if (0xD800...0xDBFF).contains(first) {
            guard current == "\\" else {
                throw LFMToolCallParserError.malformed(detail: "unpaired high surrogate")
            }
            index += 1
            guard current == "u" else {
                throw LFMToolCallParserError.malformed(detail: "unpaired high surrogate")
            }
            index += 1
            let second = try readHexCodeUnit()
            guard (0xDC00...0xDFFF).contains(second) else {
                throw LFMToolCallParserError.malformed(detail: "unpaired high surrogate")
            }
            let combined = 0x10000
                + (UInt32(first - 0xD800) << 10)
                + UInt32(second - 0xDC00)
            guard let scalar = Unicode.Scalar(combined) else {
                throw LFMToolCallParserError.malformed(detail: "invalid unicode escape")
            }
            return String(Character(scalar))
        }
        guard !(0xDC00...0xDFFF).contains(first),
              let scalar = Unicode.Scalar(UInt32(first)) else {
            throw LFMToolCallParserError.malformed(detail: "unpaired low surrogate")
        }
        return String(Character(scalar))
    }

    private mutating func readHexCodeUnit() throws -> UInt16 {
        var value: UInt16 = 0
        for _ in 0..<4 {
            guard let character = current, let digit = character.hexDigitValue else {
                throw LFMToolCallParserError.malformed(detail: "invalid unicode escape")
            }
            value = value << 4 | UInt16(digit)
            index += 1
        }
        return value
    }

    // MARK: - Primitives

    private mutating func skipWhitespace() {
        while let character = current, character.isWhitespace {
            index += 1
        }
    }

    private mutating func expect(_ character: Character) throws {
        skipWhitespace()
        guard current == character else {
            throw LFMToolCallParserError.malformed(detail: "expected '\(character)' at offset \(index)")
        }
        index += 1
    }

    private func isNameStart(_ character: Character) -> Bool {
        character.isASCII && (character.isLetter || character == "_")
    }

    private func isNameContinuation(_ character: Character) -> Bool {
        character.isASCII && (character.isLetter || character.isNumber || "_.-".contains(character))
    }

    private func isKeyContinuation(_ character: Character) -> Bool {
        character.isASCII && (character.isLetter || character.isNumber || "_.-$".contains(character))
    }
}

private extension Character {
    var isASCIIDigit: Bool {
        isASCII && ("0"..."9").contains(self)
    }
}
