import Foundation
import TurboFieldfareFormat
import TurboFieldfareRepackCore

public struct AppModelInstallDescriptor: Equatable, Sendable, Identifiable {
    public let id: String
    public let family: ModelFamily
    public let displayName: String
    public let repoID: String
    public let revision: String
    public let sourceIndexSHA256: String
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64
    public let rangeStagingBytes: UInt64
    public let reserveBytes: UInt64
    public let defaultFilename: String
    public let defaultContext: Int
    public let defaultExpertCacheSlots: Int
    public let supportsVision: Bool

    public init(id: String = "gemma4",
                family: ModelFamily = .gemma4,
                displayName: String,
                repoID: String,
                revision: String,
                sourceIndexSHA256: String,
                approximateDownloadBytes: UInt64,
                installedBytes: UInt64,
                rangeStagingBytes: UInt64,
                reserveBytes: UInt64,
                defaultFilename: String = "gemma4.gturbo",
                defaultContext: Int = 16_384,
                defaultExpertCacheSlots: Int = 16,
                supportsVision: Bool = true) {
        self.id = id
        self.family = family
        self.displayName = displayName
        self.repoID = repoID
        self.revision = revision
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.approximateDownloadBytes = approximateDownloadBytes
        self.installedBytes = installedBytes
        self.rangeStagingBytes = rangeStagingBytes
        self.reserveBytes = reserveBytes
        self.defaultFilename = defaultFilename
        self.defaultContext = defaultContext
        self.defaultExpertCacheSlots = defaultExpertCacheSlots
        self.supportsVision = supportsVision
    }

    public var requiredFreeBytes: UInt64 {
        installedBytes + rangeStagingBytes + reserveBytes
    }

    public static let `default` = AppModelInstallDescriptor(
        id: "gemma4",
        family: .gemma4,
        displayName: "Gemma 4 26B-A4B IT 4-bit",
        repoID: "mlx-community/gemma-4-26b-a4b-it-4bit",
        revision: "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
        sourceIndexSHA256: "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13",
        approximateDownloadBytes: 14_620_479_420,
        installedBytes: 14_291_921_884,
        rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
        reserveBytes: 1_073_741_824,
        defaultFilename: "gemma4.gturbo",
        defaultContext: 16_384,
        defaultExpertCacheSlots: 16,
        supportsVision: true)

    public static let visionCompanion = AppModelInstallDescriptor(
        id: "gemma4-vision",
        family: .gemma4,
        displayName: "Gemma 4 Image Support",
        repoID: "mlx-community/gemma-4-26b-a4b-it-4bit",
        revision: "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
        sourceIndexSHA256: "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13",
        approximateDownloadBytes: 1_539_478_890,
        installedBytes: 1_144_373_248 + 4_194_304,
        rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
        reserveBytes: 1_073_741_824,
        defaultFilename: "gemma4.vision.gturbo",
        supportsVision: true)
}

public struct AppModelInstallRequirement: Equatable, Sendable {
    public let probePath: String
    public let requiredBytes: UInt64
    public let availableBytes: UInt64

    public init(probePath: String = "", requiredBytes: UInt64, availableBytes: UInt64) {
        self.probePath = probePath
        self.requiredBytes = requiredBytes
        self.availableBytes = availableBytes
    }

    public var canInstall: Bool { availableBytes >= requiredBytes }

    public var shortfallBytes: UInt64 {
        requiredBytes > availableBytes ? requiredBytes - availableBytes : 0
    }
}

public enum AppModelInstallReadiness: Equatable, Sendable {
    case installed
    case checking
    case ready(AppModelInstallRequirement)
    case insufficientSpace(AppModelInstallRequirement)
    case failed(String)

    public var requirement: AppModelInstallRequirement? {
        switch self {
        case .ready(let requirement), .insufficientSpace(let requirement):
            return requirement
        case .installed, .checking, .failed:
            return nil
        }
    }
}
