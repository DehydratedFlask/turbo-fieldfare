import Foundation
import TurboFieldfare
import TurboFieldfareFormat

public enum AppModelInstallationStatus: Equatable, Sendable {
    case missing
    case partial(String)
    case complete
}

public enum AppModelInstallationProbe {
    public static func status(
        at directory: URL,
        descriptor: AppModelInstallDescriptor? = nil
    ) -> AppModelInstallationStatus {
        let directory = directory.standardizedFileURL
        let manifestURL = directory.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            return .missing
        }

        do {
            let family = try ManifestReader.peekFamily(directoryURL: directory)
            let expected = descriptor ?? AppModelCatalog.descriptor(family: family)
            guard expected.family == family else {
                return .partial(
                    "installed model is \(family.rawValue), expected \(expected.family.rawValue)")
            }
            guard let config = ArchConfig.canonical(for: family) else {
                return .partial("unsupported model family: \(family.rawValue)")
            }
            let manifest = try ManifestReader.load(directoryURL: directory, expecting: config)
            let expectedSource = "sha256:" + expected.sourceIndexSHA256
            guard manifest.sourceSnapshotHash == expectedSource else {
                return .partial("installed checkpoint does not match \(expected.displayName)")
            }
            let layout = directory.appendingPathComponent("packed_experts/layout.json")
            guard FileManager.default.fileExists(atPath: layout.path) else {
                return .partial("packed_experts/layout.json is missing")
            }
            let receipt = try VerifiedInstallReceiptReader.load(directoryURL: directory)
            let manifestHash = try Sha256Verifier.hashFile(at: manifestURL, chunkBytes: 65_536)
            try VerifiedInstallReceiptReader.validateManifestBinding(
                receipt,
                directoryURL: directory,
                manifestSha256: manifestHash)
            return .complete
        } catch {
            return .partial("\(error)")
        }
    }
}
