import Foundation
import Testing
import TurboFieldfare
import TurboFieldfareFormat
@testable import TurboFieldfareDecodeService

@Suite("Decode-service family admission")
struct DecodeServiceFamilyTests {
    @Test func loadRefusalNamesTheLFMFamily() throws {
        let directory = try makePeekableLFMModel()
        defer { try? FileManager.default.removeItem(at: directory) }
        let family = try ManifestReader.peekFamily(directoryURL: directory)
        #expect(family == .lfm2Moe)
        #expect(DecodeLoadAdmission.refusal(
            maxContextTokens: 131_072,
            hostMemoryBytes: 8 << 30,
            environment: [:],
            expertCacheSlots: 8,
            config: ArchConfig.canonical(for: family)!) != nil)
    }
}

private func makePeekableLFMModel() throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("decode-lfm-\(UUID().uuidString).gturbo",
                                isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let arch = ArchConfig.lfm2_5_24B_A2B
    let archDict: [String: Any] = [
        "hiddenSize": arch.hiddenSize,
        "ffnIntermediate": arch.intermediateSize,
        "moeIntermediateSize": arch.moeIntermediateSize,
        "numHeads": arch.numHeads,
        "numKVHeads": arch.numKVHeads,
        "numFullKVHeads": arch.numFullKVHeads,
        "headDim": arch.headDim,
        "fullHeadDim": arch.fullHeadDim,
        "vocabSize": arch.vocabSize,
        "slidingWindow": arch.slidingWindow,
        "finalLogitSoftcap": arch.finalLogitSoftcap,
        "ropeTheta": arch.ropeTheta,
        "fullRopeTheta": arch.fullRopeTheta,
        "partialRotaryFactor": arch.partialRotaryFactor,
        "numLayers": arch.numLayers,
        "numExperts": arch.numExperts,
        "topKExperts": arch.topKExperts,
        "tieWordEmbeddings": arch.tieWordEmbeddings,
        "attentionKEqV": arch.attentionKEqV,
        "hiddenActivation": arch.hiddenActivation,
        "fullAttentionLayerMask": arch.fullAttentionLayerMask.map(Int.init),
        "family": ModelFamily.lfm2Moe.rawValue,
        "layerTypes": arch.layerTypes.map(\.rawValue),
        "moeLayerMask": arch.moeLayerMask.map(Int.init),
        "attentionScale": arch.attentionScale,
        "rmsNormUnitOffset": arch.rmsNormUnitOffset,
        "ropeStyle": arch.ropeStyle.rawValue,
        "convKernelSize": arch.convKernelSize,
        "routerKind": arch.routerKind.rawValue,
        "sharedExpert": arch.hasSharedExpert,
        "embeddingScaledBySqrtHidden": arch.embeddingScaledBySqrtHidden,
        "numDenseLayers": arch.numDenseLayers,
        "routerNormalizeTopK": arch.routerNormalizeTopK,
        "routerBias": arch.routerBias,
        "routedScalingFactor": arch.routedScalingFactor,
    ]
    let manifest: [String: Any] = [
        "magic": "GTURBO",
        "versionMajor": 1,
        "versionMinor": 0,
        "flags": [String: Bool](),
        "modelID": "test/lfm",
        "arch": archDict,
        "files": [String: Any](),
        "expertsPerLayer": arch.numExperts,
        "numLayers": arch.numLayers,
        "expertStride": arch.admission.expertStrideBytes,
    ]
    try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
        .write(to: directory.appendingPathComponent("manifest.json"))
    return directory
}
