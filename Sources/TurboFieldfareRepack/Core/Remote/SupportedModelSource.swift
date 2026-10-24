import Foundation

/// Gemma aliases kept for existing call sites. New code should read the
/// descriptor from `ModelSourceCatalog`.
public enum SupportedModelSource {
    public static let displayName = ModelSourceCatalog.gemma4.displayName
    public static let repoID = ModelSourceCatalog.gemma4.repoID
    public static let revision = ModelSourceCatalog.gemma4.revision
    public static let sourceIndexSHA256 = ModelSourceCatalog.gemma4.sourceIndexSHA256
    public static let approximateDownloadBytes = ModelSourceCatalog.gemma4.approximateDownloadBytes
    public static let installedBytes = ModelSourceCatalog.gemma4.installedBytes
    public static let reserveBytes = ModelSourceCatalog.gemma4.reserveBytes

    public static func installOptions(for descriptor: ModelSourceDescriptor,
                                      outputDirectory: URL,
                                      overwrite: Bool,
                                      token: String?,
                                      resume: Bool = false)
        -> RemoteStreamingRepackOptions {
        RemoteStreamingRepackOptions(
            repoID: descriptor.repoID,
            revision: descriptor.revision,
            outputDir: outputDirectory.path,
            token: token,
            requireKnownSource: true,
            minFreeReserveBytes: descriptor.reserveBytes,
            overwrite: overwrite,
            resume: resume)
    }

    public static func installOptions(outputDirectory: URL,
                                      overwrite: Bool,
                                      token: String?,
                                      resume: Bool = false)
        -> RemoteStreamingRepackOptions {
        installOptions(
            for: ModelSourceCatalog.gemma4,
            outputDirectory: outputDirectory,
            overwrite: overwrite,
            token: token,
            resume: resume)
    }
}
