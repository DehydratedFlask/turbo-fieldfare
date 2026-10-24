import Foundation
import TurboFieldfareFormat

/// A pinned Hugging Face source the repacker knows how to install. `id` is the
/// selector string accepted by `--model-source`.
public struct ModelSourceDescriptor: Sendable, Equatable {
    public let id: String
    public let family: ModelFamily
    public let displayName: String
    public let repoID: String
    public let revision: String
    public let sourceIndexSHA256: String
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64
    public let reserveBytes: UInt64
    public let defaultFilename: String
    public let defaultContext: Int
    public let defaultExpertCacheSlots: Int
    public let supportsVision: Bool

    public init(id: String,
                family: ModelFamily,
                displayName: String,
                repoID: String,
                revision: String,
                sourceIndexSHA256: String,
                approximateDownloadBytes: UInt64,
                installedBytes: UInt64,
                reserveBytes: UInt64,
                defaultFilename: String,
                defaultContext: Int,
                defaultExpertCacheSlots: Int,
                supportsVision: Bool) {
        self.id = id
        self.family = family
        self.displayName = displayName
        self.repoID = repoID
        self.revision = revision
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.approximateDownloadBytes = approximateDownloadBytes
        self.installedBytes = installedBytes
        self.reserveBytes = reserveBytes
        self.defaultFilename = defaultFilename
        self.defaultContext = defaultContext
        self.defaultExpertCacheSlots = defaultExpertCacheSlots
        self.supportsVision = supportsVision
    }
}

/// The pinned sources this build can install. Adding an entry means the
/// importer was validated against a fresh upload of that checkpoint.
public enum ModelSourceCatalog {
    public static let gemma4 = ModelSourceDescriptor(
        id: "gemma4",
        family: .gemma4,
        displayName: "Gemma 4 26B-A4B IT 4-bit",
        repoID: "mlx-community/gemma-4-26b-a4b-it-4bit",
        revision: "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
        sourceIndexSHA256:
            "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13",
        approximateDownloadBytes: 14_620_479_420,
        installedBytes: 14_291_921_884,
        reserveBytes: 1_073_741_824,
        defaultFilename: "gemma4.gturbo",
        defaultContext: 16_384,
        defaultExpertCacheSlots: 16,
        supportsVision: true)

    /// LFM2.5 24B-A2B MLX 4-bit. `installedBytes` covers the planner's
    /// measured text model (13,567,221,760 bytes) plus manifest, layout,
    /// tokenizer, and receipt metadata, rounded up conservatively.
    public static let lfm2_5_24B_A2B = ModelSourceDescriptor(
        id: "lfm2_moe",
        family: .lfm2Moe,
        displayName: "LFM2.5 24B-A2B 4-bit",
        repoID: "LiquidAI/LFM2.5-24B-A2B-MLX-4bit",
        revision: "667408d494bde191c2cb9dd4f4549827816b9275",
        sourceIndexSHA256:
            "7887b03265aef7824c8e5b9a51667e486f889749240d8c59a95f3cd899e86e4a",
        approximateDownloadBytes: 13_567_188_992,
        installedBytes: 13_572_000_000,
        reserveBytes: 1_073_741_824,
        defaultFilename: "lfm2.5-24b-a2b.gturbo",
        defaultContext: 8_192,
        defaultExpertCacheSlots: 8,
        supportsVision: false)

    public static let all: [ModelSourceDescriptor] = [gemma4, lfm2_5_24B_A2B]

    public static func descriptor(id: String) -> ModelSourceDescriptor? {
        all.first { $0.id == id }
    }
}
