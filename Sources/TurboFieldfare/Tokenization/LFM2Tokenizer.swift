import Foundation
import Hub
import Tokenizers
import TurboFieldfareFormat

public enum LFM2TokenizerError: Error, CustomStringConvertible {
    case missingIdentity
    case identityMismatch(String)
    case missingFile(String)
    case unsupportedDecoder(actual: String)
    case specialTokenMismatch(token: String, expected: Int32, resolved: Int32)
    case invalidTokenID(token: String, id: Int)

    public var description: String {
        switch self {
        case .missingIdentity:
            return "installed LFM tokenizer is missing identity.json; reinstall the model"
        case .identityMismatch(let detail):
            return "installed LFM tokenizer identity does not match this build: \(detail)"
        case .missingFile(let name):
            return "installed LFM tokenizer is missing \(name)"
        case .unsupportedDecoder(let actual):
            return "tokenizer decoder is not ByteLevel; found: \(actual)"
        case .specialTokenMismatch(let token, let expected, let resolved):
            return "tokenizer resolves \(token) to \(resolved), but the runtime "
                + "expects \(expected); the tokenizer does not match this build"
        case .invalidTokenID(let token, let id):
            return "tokenizer declares out-of-range ID \(id) for token \(token)"
        }
    }
}

/// LFM2.5 tokenizer wrapper around the installed ByteLevel BPE sidecar.
///
/// The added-token table is part of the architecture contract: the runtime
/// keys channel handling, tool parsing, and stopping on IDs, so a tokenizer
/// revision that renumbers any of 124893...124907 fails load instead of
/// producing subtly wrong framing.
public struct LFM2Tokenizer: ModelTokenizer, LFM2ByteLevelVocabulary, @unchecked Sendable {
    public static let chatTemplateIdentity = "lfm2.5-24b-a2b-chatml-v1"
    public static let bosTokenID: Int32 = 124_894
    public static let eosTokenID: Int32 = 124_900
    public static let padTokenID: Int32 = 124_893

    public let family: ModelFamily = .lfm2Moe
    public let bosID: Int32 = LFM2Tokenizer.bosTokenID
    public let eosID: Int32 = LFM2Tokenizer.eosTokenID
    public let padID: Int32 = LFM2Tokenizer.padTokenID
    public let endOfTurnID: Int32 = LFM2Tokenizer.eosTokenID
    public let toolResponseID: Int32 = LFM2Tokenizer.eosTokenID
    public let toolCallStartID: Int32 = 124_905
    public let toolCallEndID: Int32 = 124_906
    public let toolResponseEndID: Int32 = LFM2Tokenizer.eosTokenID
    public let stopTokenIDs: Set<Int32> = [LFM2Tokenizer.eosTokenID]
    public let structuralMarkerIDs: Set<Int32> = [124_901, 124_902, 124_905, 124_906]
    public let vocabSize: Int = 128_000
    public let chatTemplateIdentity: String

    private let tokenizer: any Tokenizer
    private let specialTokenIDs: Set<Int32>
    private let addedTokenIDs: Set<Int32>

    private static let expectedAddedTokens: [(id: Int32, text: String, special: Bool)] = [
        (124_893, "<|pad|>", true),
        (124_894, "<|startoftext|>", true),
        (124_895, "<|endoftext|>", true),
        (124_896, "<|fim_pre|>", true),
        (124_897, "<|fim_mid|>", true),
        (124_898, "<|fim_suf|>", true),
        (124_899, "<|im_start|>", true),
        (124_900, "<|im_end|>", true),
        (124_901, "<think>", false),
        (124_902, "</think>", false),
        (124_903, "<|tool_list_start|>", true),
        (124_904, "<|tool_list_end|>", true),
        (124_905, "<|tool_call_start|>", false),
        (124_906, "<|tool_call_end|>", false),
        (124_907, "<image>", true),
    ]

    public static func load(forModelDirectory modelDirectory: URL,
                            fileManager: FileManager = .default) async throws -> LFM2Tokenizer {
        let sidecar = modelDirectory.standardizedFileURL
            .appendingPathComponent("tokenizer", isDirectory: true)
        let identityURL = sidecar.appendingPathComponent(TokenizerIdentity.fileName)
        guard fileManager.fileExists(atPath: identityURL.path) else {
            throw LFM2TokenizerError.missingIdentity
        }
        let identity: TokenizerIdentity
        do {
            identity = try TokenizerIdentity.load(from: sidecar)
        } catch {
            throw LFM2TokenizerError.missingIdentity
        }
        guard identity.family == .lfm2Moe else {
            throw LFM2TokenizerError.identityMismatch(
                "family is \(identity.family.rawValue), expected \(ModelFamily.lfm2Moe.rawValue)")
        }
        guard identity.chatTemplateIdentity == chatTemplateIdentity else {
            throw LFM2TokenizerError.identityMismatch(
                "chat template is \(identity.chatTemplateIdentity), expected \(chatTemplateIdentity)")
        }
        try verify(identity: identity, in: sidecar, fileManager: fileManager)

        let hub = LanguageModelConfigurationFromHub(modelFolder: sidecar)
        guard let tokenizerConfig = try await hub.tokenizerConfig else {
            throw LFM2TokenizerError.missingFile("tokenizer_config.json")
        }
        let tokenizerData = try await hub.tokenizerData
        let underlying = try AutoTokenizer.from(
            tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
        return try LFM2Tokenizer(
            tokenizer: underlying, tokenizerData: tokenizerData, identity: identity)
    }

    public init(tokenizer: any Tokenizer,
                tokenizerData: Config,
                identity: TokenizerIdentity) throws {
        self.tokenizer = tokenizer
        self.chatTemplateIdentity = identity.chatTemplateIdentity
        try Self.verifyDecoderConfiguration(tokenizerData)

        var added: [Int32: (text: String, special: Bool)] = [:]
        for entry in tokenizerData["addedTokens"].array(or: []) {
            guard let rawID = entry["id"].integer(), let text = entry["content"].string() else {
                continue
            }
            guard let id = Int32(exactly: rawID) else {
                throw LFM2TokenizerError.invalidTokenID(token: text, id: rawID)
            }
            added[id] = (text, entry["special"].boolean(or: false))
        }
        for expected in Self.expectedAddedTokens {
            guard let actual = added[expected.id],
                  actual.text == expected.text,
                  actual.special == expected.special else {
                throw LFM2TokenizerError.specialTokenMismatch(
                    token: expected.text, expected: expected.id, resolved: expected.id)
            }
        }
        let expectedRange = Set(Self.expectedAddedTokens.map(\.id))
        guard Set(added.keys).intersection(expectedRange) == expectedRange else {
            throw LFM2TokenizerError.missingFile("added_tokens 124893...124907")
        }
        self.addedTokenIDs = Set(added.keys)
        self.specialTokenIDs = Set(added.filter { $0.value.special }.keys)

        guard tokenizer.bosTokenId == Int(Self.bosTokenID),
              tokenizer.eosTokenId == Int(Self.eosTokenID) else {
            throw LFM2TokenizerError.specialTokenMismatch(
                token: "BOS/EOS",
                expected: Self.bosTokenID,
                resolved: Int32(tokenizer.bosTokenId ?? -1))
        }
    }

    static func verifyDecoderConfiguration(_ tokenizerData: Config) throws {
        let decoder = tokenizerData["decoder"]
        guard decoder.type.string() == "ByteLevel" else {
            throw LFM2TokenizerError.unsupportedDecoder(actual: decoder.description)
        }
    }

    private static func verify(identity: TokenizerIdentity,
                               in sidecar: URL,
                               fileManager: FileManager) throws {
        let files: [(name: String, expected: String)] = [
            ("chat_template.jinja", identity.chatTemplateSHA256),
            ("tokenizer.json", identity.tokenizerSHA256),
            ("tokenizer_config.json", identity.tokenizerConfigSHA256),
        ]
        for file in files {
            let url = sidecar.appendingPathComponent(file.name)
            guard fileManager.fileExists(atPath: url.path) else {
                throw LFM2TokenizerError.missingFile(file.name)
            }
            let actual = try Sha256Verifier.hashFile(at: url)
            guard actual.lowercased() == file.expected.lowercased() else {
                throw LFM2TokenizerError.identityMismatch("SHA-256 of \(file.name)")
            }
        }
    }

    public func encode(_ text: String, addBOS: Bool = true) -> [Int32] {
        let base = tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
        return addBOS ? [bosID] + base : base
    }

    public func decode(_ ids: [Int32], skipSpecialTokens: Bool = true) -> String {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(ids.count)
        for id in ids {
            if skipSpecialTokens, specialTokenIDs.contains(id) { continue }
            if let fragment = byteFragment(forTokenID: id) {
                bytes.append(contentsOf: fragment)
            } else if let text = tokenizer.convertIdToToken(Int(id)) {
                bytes.append(contentsOf: Array(text.utf8))
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    public func byteFragment(forTokenID id: Int32) -> [UInt8]? {
        guard let text = tokenizer.convertIdToToken(Int(id)) else { return nil }
        if specialTokenIDs.contains(id) { return nil }
        if addedTokenIDs.contains(id) { return Array(text.utf8) }

        var bytes: [UInt8] = []
        bytes.reserveCapacity(text.utf8.count)
        for scalar in text.unicodeScalars {
            guard let byte = Self.byte(for: scalar) else {
                return Array(text.utf8)
            }
            bytes.append(byte)
        }
        return bytes
    }

    /// GPT-2's byte-to-unicode alphabet, inverted. Every byte without a
    /// printable identity is represented by a scalar starting at U+0100.
    private static func byte(for scalar: Unicode.Scalar) -> UInt8? {
        switch scalar.value {
        case 33...126, 161...172, 174...255:
            return UInt8(scalar.value)
        case 256...323:
            let offset = Int(scalar.value - 256)
            if offset < 33 { return UInt8(offset) }
            if offset < 67 { return UInt8(127 + offset - 33) }
            if offset == 67 { return 173 }
            return nil
        default:
            return nil
        }
    }

    public func makeStreamingDetokenizer() -> any StreamingDetokenizer {
        LFM2Detokenizer(tokenizer: self)
    }

    public func applyChatTemplate(messages: [ChatMessage],
                                  tools: [ToolDefinition]?,
                                  addGenerationPrompt: Bool) throws -> [Int32] {
        let text = try LFM2ChatRenderer().render(
            messages: messages, tools: tools, addGenerationPrompt: addGenerationPrompt)
        return encode(text, addBOS: false)
    }

    public func continuationPlan(cachedMessages: [ChatMessage],
                                 cachedTokenIDs: [Int32],
                                 incomingMessages: [ChatMessage],
                                 tools: [ToolDefinition]?) throws -> ContinuationPlan {
        let full = try applyChatTemplate(
            messages: incomingMessages, tools: tools, addGenerationPrompt: true)
        guard incomingMessages.count >= cachedMessages.count,
              Array(incomingMessages.prefix(cachedMessages.count)) == cachedMessages,
              full.count > cachedTokenIDs.count,
              Array(full.prefix(cachedTokenIDs.count)) == cachedTokenIDs else {
            return .replayAll(full)
        }
        let suffix = Array(full.dropFirst(cachedTokenIDs.count))
        guard suffix.first == 124_900 || suffix.first == 124_899 else {
            return .replayAll(full)
        }
        return .reuse(prefixTokens: cachedTokenIDs, suffixTokens: suffix)
    }

    public func makeAssistantDecoder(allowedTools: Set<String>) -> (any StructuredOutputDecoder)? {
        LFM2AssistantDecoder(allowedTools: allowedTools)
    }
}
