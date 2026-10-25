import Foundation

/// Raw byte source for a ByteLevel BPE vocabulary.
///
/// `ModelTokenizer.decode([id])` cannot be used to stream a multi-byte
/// codepoint: an incomplete UTF-8 sequence has no `String` representation and
/// the reference decoder replaces it with U+FFFD. A byte-level family exposes
/// the token's payload bytes instead, so the streaming detokenizer can hold the
/// incomplete tail and commit the codepoint once the next token completes it.
public protocol LFM2ByteLevelVocabulary: Sendable {
    /// Payload bytes for one token, or `nil` when the token has no byte payload
    /// (specials) or does not come from the byte-level alphabet.
    func byteFragment(forTokenID id: Int32) -> [UInt8]?
}

/// Streaming detokenizer for the LFM ByteLevel BPE vocabulary.
///
/// Tokens are converted to bytes, accumulated in a small buffer, and only the
/// longest complete UTF-8 prefix is emitted. The trailing incomplete sequence
/// waits for the token that finishes it. `flush()` drops an incomplete trailing
/// sequence rather than inserting U+FFFD, matching a generation that stopped
/// before a codepoint was complete.
public struct LFM2Detokenizer: StreamingDetokenizer {
    private let tokenizer: any ModelTokenizer
    private let vocabulary: (any LFM2ByteLevelVocabulary)?
    private var pending: [UInt8] = []

    public init(tokenizer: any ModelTokenizer) {
        self.tokenizer = tokenizer
        self.vocabulary = tokenizer as? any LFM2ByteLevelVocabulary
    }

    public mutating func push(_ tokenID: Int32) -> String {
        let bytes: [UInt8]
        if let vocabulary, let fragment = vocabulary.byteFragment(forTokenID: tokenID) {
            bytes = fragment
        } else {
            bytes = Array(tokenizer.decode([tokenID], skipSpecialTokens: true).utf8)
        }
        guard !bytes.isEmpty else { return "" }
        pending.append(contentsOf: bytes)
        let complete = Self.completePrefixLength(pending)
        guard complete > 0 else { return "" }
        let text = String(decoding: pending[..<complete], as: UTF8.self)
        pending.removeFirst(complete)
        return text
    }

    public mutating func flush() -> String {
        let complete = Self.completePrefixLength(pending)
        let text = complete == 0
            ? ""
            : String(decoding: pending[..<complete], as: UTF8.self)
        pending.removeAll(keepingCapacity: true)
        return text
    }

    /// Length of the longest prefix that ends on a complete UTF-8 scalar.
    ///
    /// Only a trailing partial sequence is held back. Malformed bytes in the
    /// middle are left to `String(decoding:)`, which is the same replacement
    /// behavior a batch decode has.
    private static func completePrefixLength(_ bytes: [UInt8]) -> Int {
        guard let last = bytes.last else { return 0 }
        guard last >= 0x80 else { return bytes.count }

        var leadIndex = bytes.count - 1
        var continuations = 0
        while leadIndex >= 0, (0x80...0xBF).contains(bytes[leadIndex]), continuations < 4 {
            continuations += 1
            leadIndex -= 1
        }
        guard leadIndex >= 0 else { return bytes.count }

        let expected: Int
        switch bytes[leadIndex] {
        case 0xC2...0xDF: expected = 2
        case 0xE0...0xEF: expected = 3
        case 0xF0...0xF4: expected = 4
        default: return bytes.count
        }
        return bytes.count - leadIndex < expected ? leadIndex : bytes.count
    }
}
