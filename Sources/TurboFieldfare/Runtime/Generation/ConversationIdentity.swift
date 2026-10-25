import Foundation
import TurboFieldfareFormat

public enum ConversationIdentityError: Error, CustomStringConvertible {
    case snapshotHashUnavailable
    case missingTokenizerIdentity
    case familyMismatch(installed: ModelFamily, expected: ModelFamily)

    public var description: String {
        switch self {
        case .snapshotHashUnavailable:
            "the loaded model's manifest carries no source snapshot hash, so a "
                + "conversation recorded against it could not be told apart from "
                + "one recorded against a different checkpoint"
        case .missingTokenizerIdentity:
            "the installed model has no tokenizer/identity.json, so its chat "
                + "framing cannot be verified; reinstall the model"
        case .familyMismatch(let installed, let expected):
            "the tokenizer identity belongs to \(installed.rawValue), but the "
                + "model is \(expected.rawValue)"
        }
    }
}

/// Everything that has to be the same for a stored conversation's token IDs to
/// mean what they meant when they were recorded.
///
/// Not a cache key: a conversation owns its lineage and never has to match
/// anything. It is a refusal test. Token IDs index a vocabulary, sit at
/// positions a template chose, and stand in for images a preprocessor produced;
/// change any of those and the same integers describe a different conversation.
/// The four fields are the four ways that can happen — a different checkpoint,
/// a different model, a different chat framing, or different pixels out of the
/// same file — and a mismatch in any of them leaves the transcript readable and
/// says so rather than continuing on a lineage the model never had.
public struct ConversationIdentity: Sendable, Equatable, Codable {
    public let modelID: String
    public let sourceSnapshotHash: String
    public let templateIdentity: String
    public let imageProcessingVersion: Int

    public init(modelID: String,
                sourceSnapshotHash: String,
                templateIdentity: String,
                imageProcessingVersion: Int) {
        self.modelID = modelID
        self.sourceSnapshotHash = sourceSnapshotHash
        self.templateIdentity = templateIdentity
        self.imageProcessingVersion = imageProcessingVersion
    }

    /// The identity of a loaded model. Gemma keeps its constant identity, so
    /// existing conversations still load. LFM reads the installed sidecar and
    /// has no image processor (version 0).
    ///
    /// Throws rather than substituting a placeholder for a missing snapshot
    /// hash: an identity that cannot be established has to refuse replay, and a
    /// blank field would compare equal to the next blank field and admit it.
    public init(model: Model) throws {
        guard let snapshot = model.sourceSnapshotHash, !snapshot.isEmpty else {
            throw ConversationIdentityError.snapshotHashUnavailable
        }
        let framing = try Self.framing(at: model.directoryURL,
                                       family: model.config.family)
        self.init(modelID: model.modelID,
                  sourceSnapshotHash: snapshot,
                  templateIdentity: framing.identity,
                  imageProcessingVersion: framing.imageVersion)
    }

    /// The identity of an installed model, from its manifest and tokenizer
    /// sidecar alone.
    ///
    /// The app process needs this and never loads the model: the decode service
    /// does. Reading the manifest costs a few kilobytes and stays well inside
    /// the rule that no path holds model-derived data in the heap.
    public static func forModelDirectory(_ directoryURL: URL) throws -> ConversationIdentity {
        let family = try ManifestReader.peekFamily(directoryURL: directoryURL)
        guard let config = ArchConfig.canonical(for: family) else {
            throw ConversationIdentityError.familyMismatch(
                installed: family, expected: family)
        }
        let manifest = try ManifestReader.load(
            directoryURL: directoryURL, expecting: config)
        guard let snapshot = manifest.sourceSnapshotHash, !snapshot.isEmpty else {
            throw ConversationIdentityError.snapshotHashUnavailable
        }
        let framing = try framing(at: directoryURL, family: family)
        return ConversationIdentity(
            modelID: manifest.modelID,
            sourceSnapshotHash: snapshot,
            templateIdentity: framing.identity,
            imageProcessingVersion: framing.imageVersion)
    }

    private static func framing(
        at directoryURL: URL,
        family: ModelFamily
    ) throws -> (identity: String, imageVersion: Int) {
        switch family {
        case .gemma4:
            return (GFTokenizer.chatTemplateIdentity, VisionImageProcessing.version)
        case .lfm2Moe:
            let sidecar = directoryURL.standardizedFileURL
                .appendingPathComponent("tokenizer", isDirectory: true)
            guard FileManager.default.fileExists(
                atPath: sidecar.appendingPathComponent(TokenizerIdentity.fileName).path) else {
                throw ConversationIdentityError.missingTokenizerIdentity
            }
            let identity: TokenizerIdentity
            do {
                identity = try TokenizerIdentity.load(from: sidecar)
            } catch {
                throw ConversationIdentityError.missingTokenizerIdentity
            }
            guard identity.family == family else {
                throw ConversationIdentityError.familyMismatch(
                    installed: identity.family, expected: family)
            }
            return (identity.chatTemplateIdentity, 0)
        }
    }
}
