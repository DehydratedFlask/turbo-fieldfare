import TurboFieldfareFormat
import TurboFieldfareRepackCore

/// The models the app can install and run. Gemma 4 remains the default; LFM is
/// a second, explicitly selected family, never a hard-coded replacement.
public enum AppModelCatalog {
    public static let gemma4 = AppModelInstallDescriptor.default

    public static let lfm2_5 = AppModelInstallDescriptor(
        id: "lfm2.5-24b-a2b",
        family: .lfm2Moe,
        displayName: "LFM2.5 24B-A2B 4-bit",
        repoID: "LiquidAI/LFM2.5-24B-A2B-MLX-4bit",
        revision: "667408d494bde191c2cb9dd4f4549827816b9275",
        sourceIndexSHA256: "7887b03265aef7824c8e5b9a51667e486f889749240d8c59a95f3cd899e86e4a",
        approximateDownloadBytes: 13_567_188_992,
        installedBytes: 13_572_000_000,
        rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
        reserveBytes: 1_073_741_824,
        defaultFilename: "lfm2.5-24b-a2b.gturbo",
        defaultContext: 8_192,
        defaultExpertCacheSlots: 8,
        supportsVision: false)

    public static let all = [gemma4, lfm2_5]
    public static let `default` = gemma4

    public static func descriptor(id: String?) -> AppModelInstallDescriptor {
        guard let id, let match = all.first(where: { $0.id == id }) else {
            return gemma4
        }
        return match
    }

    public static func descriptor(family: ModelFamily) -> AppModelInstallDescriptor {
        all.first { $0.family == family } ?? gemma4
    }
}
