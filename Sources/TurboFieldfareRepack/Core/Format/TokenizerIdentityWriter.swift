import Foundation
import TurboFieldfareFormat

public extension TokenizerIdentity {
    /// The ChatML template the LFM2.5 renderer is pinned to. A pack whose
    /// template identity differs must not be rendered as LFM2.5 chat.
    static let lfm2ChatMLIdentity = "lfm2.5-24b-a2b-chatml-v1"

    /// Writes the sidecar for `family`, or nothing when the family keeps its
    /// identity compiled in.
    ///
    /// Gemma's tokenizer identity is a runtime constant and every existing
    /// install predates this file, so writing one would change the installed
    /// bytes and manifest of every Gemma pack. LFM installs have no such
    /// constant and must carry the sidecar.
    static func writeIfNeeded(for tokenizerDirectory: URL,
                              family: ModelFamily) throws -> TokenizerIdentity? {
        switch family {
        case .gemma4:
            return nil
        case .lfm2Moe:
            return try write(for: tokenizerDirectory, family: family)
        }
    }

    /// Hashes the copied `chat_template.jinja`, `tokenizer.json`, and
    /// `tokenizer_config.json` and writes `identity.json` atomically in the
    /// same directory.
    ///
    /// Missing or oversized inputs are hard errors: an LFM pack without all
    /// three files cannot be verified at runtime, and skipping the sidecar
    /// would defer that failure to a half-loaded model.
    static func write(for tokenizerDirectory: URL,
                      family: ModelFamily) throws -> TokenizerIdentity {
        guard family == .lfm2Moe else {
            throw RepackError.configurationInvalid(
                detail: "tokenizer identity sidecar is only written for LFM "
                    + "installs, not \(family.rawValue)")
        }
        let identity = TokenizerIdentity(
            family: family,
            chatTemplateIdentity: Self.lfm2ChatMLIdentity,
            chatTemplateSHA256: try hash(
                "chat_template.jinja", in: tokenizerDirectory, cap: sidecarCapBytes),
            tokenizerSHA256: try hash(
                "tokenizer.json", in: tokenizerDirectory, cap: tokenizerCapBytes),
            tokenizerConfigSHA256: try hash(
                "tokenizer_config.json", in: tokenizerDirectory, cap: sidecarCapBytes))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try Posix.atomicWrite(
            try encoder.encode(identity),
            to: tokenizerDirectory.appendingPathComponent(Self.fileName).path,
            durableIn: tokenizerDirectory.path)
        return identity
    }

    private static let tokenizerCapBytes: UInt64 = 64 * 1024 * 1024
    private static let sidecarCapBytes: UInt64 = 4 * 1024 * 1024

    private static func hash(_ filename: String,
                             in tokenizerDirectory: URL,
                             cap: UInt64) throws -> String {
        let path = tokenizerDirectory.appendingPathComponent(filename).path
        let descriptor = try Posix.openReadNoFollow(path)
        defer { close(descriptor) }
        let size = try Posix.fileSize(fd: descriptor, path: path)
        guard size <= cap else {
            throw RepackError.installStateCorrupt(
                path: path,
                detail: "size \(size) exceeds \(cap)-byte cap")
        }
        return try Sha256Stream.hashFileDescriptor(descriptor, displayPath: path)
    }
}
