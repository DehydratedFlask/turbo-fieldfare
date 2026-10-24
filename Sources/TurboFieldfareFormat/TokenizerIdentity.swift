import Foundation

/// Identity of the tokenizer sidecars copied into an installed pack.
///
/// `tokenizer/identity.json` sits beside the files it describes. The runtime
/// uses it to prove that the template and tokenizer bytes it is about to load
/// are the ones the family's chat renderer was written against: rebuilding,
/// trimming, or swapping any of the three files changes a hash here and makes
/// the pack fail closed instead of silently rendering a different prompt.
///
/// Foundation-only so the runtime can read it without depending on the
/// repacker.
public struct TokenizerIdentity: Codable, Equatable, Sendable {
    public let family: ModelFamily
    public let chatTemplateIdentity: String
    public let chatTemplateSHA256: String
    public let tokenizerSHA256: String
    public let tokenizerConfigSHA256: String

    /// File name inside the pack's `tokenizer/` directory.
    public static let fileName = "identity.json"

    public init(family: ModelFamily,
                chatTemplateIdentity: String,
                chatTemplateSHA256: String,
                tokenizerSHA256: String,
                tokenizerConfigSHA256: String) {
        self.family = family
        self.chatTemplateIdentity = chatTemplateIdentity
        self.chatTemplateSHA256 = chatTemplateSHA256
        self.tokenizerSHA256 = tokenizerSHA256
        self.tokenizerConfigSHA256 = tokenizerConfigSHA256
    }

    /// Decodes `tokenizer/identity.json` from an installed pack.
    public static func load(from tokenizerDirectory: URL) throws -> TokenizerIdentity {
        let data = try Data(contentsOf: tokenizerDirectory
            .appendingPathComponent(Self.fileName))
        return try JSONDecoder().decode(TokenizerIdentity.self, from: data)
    }
}
