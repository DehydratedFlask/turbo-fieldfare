import Foundation
import Testing
import TurboFieldfare
import TurboFieldfareFormat
@testable import TurboFieldfareAppCore

/// A complete-install fixture whose manifest and receipt describe the selected
/// family. The probe only hashes the manifest, so the payload files may be
/// empty; the layout must exist and the manifest must validate structurally.
func makeInstalledModelDirectory(
    family: ModelFamily,
    includeIdentity: Bool = true,
    identity: String = "lfm2.5-24b-a2b-chatml-v1"
) throws -> URL {
    let manager = FileManager.default
    let directory = manager.temporaryDirectory
        .appendingPathComponent("app-\(family.rawValue)-\(UUID().uuidString).gturbo",
                                isDirectory: true)
    let experts = directory.appendingPathComponent("packed_experts", isDirectory: true)
    let tokenizer = directory.appendingPathComponent("tokenizer", isDirectory: true)
    try manager.createDirectory(at: experts, withIntermediateDirectories: true)
    try manager.createDirectory(at: tokenizer, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: experts.appendingPathComponent("layout.json"))

    let arch = ArchConfig.canonical(for: family) ?? .gemma4_26B_A4B
    let descriptor = AppModelCatalog.descriptor(family: family)
    var files: [String: Any] = [
        "model_weights.bin": ["size": 0, "sha256": String(repeating: "0", count: 64)],
        "packed_experts/layout.json": ["size": 2, "sha256": String(repeating: "0", count: 64)],
    ]
    for layer in 0..<arch.numLayers where arch.moeLayerMask[layer] == 1 {
        files[String(format: "packed_experts/layer_%02d.bin", layer)] = [
            "size": 0,
            "sha256": String(repeating: "0", count: 64),
        ]
    }

    var archDict: [String: Any] = [
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
        "family": family.rawValue,
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
    if family == .gemma4 {
        archDict["family"] = nil
        archDict["layerTypes"] = nil
        archDict["moeLayerMask"] = nil
        archDict["attentionScale"] = nil
        archDict["rmsNormUnitOffset"] = nil
        archDict["ropeStyle"] = nil
        archDict["convKernelSize"] = nil
        archDict["routerKind"] = nil
        archDict["sharedExpert"] = nil
        archDict["embeddingScaledBySqrtHidden"] = nil
        archDict["numDenseLayers"] = nil
        archDict["routerNormalizeTopK"] = nil
        archDict["routerBias"] = nil
        archDict["routedScalingFactor"] = nil
    }

    let quant: [String: Any] = family == .lfm2Moe
        ? [
            "embedding": quantSlot(6),
            "attention": quantSlot(4),
            "sharedExpert": quantSlot(4),
            "routedExpert": quantSlot(4),
        ]
        : [
            "embedding": quantSlot(4),
            "attention": quantSlot(4),
            "router": quantSlot(8),
            "sharedExpert": quantSlot(4),
            "routedExpert": quantSlot(4),
        ]
    let manifest: [String: Any] = [
        "magic": "GTURBO",
        "versionMajor": 1,
        "versionMinor": 0,
        "flags": ["streamingPresent": true],
        "modelID": descriptor.id,
        "sourceSnapshotHash": "sha256:" + descriptor.sourceIndexSHA256,
        "quant": quant,
        "arch": archDict,
        "files": files,
        "expertsPerLayer": arch.numExperts,
        "numLayers": arch.numLayers,
        "expertStride": arch.admission.expertStrideBytes,
    ]
    let manifestData = try JSONSerialization.data(withJSONObject: manifest,
                                                  options: [.sortedKeys])
    let manifestURL = directory.appendingPathComponent("manifest.json")
    try manifestData.write(to: manifestURL)
    let manifestHash = Sha256Verifier.hashData(manifestData)
    let receipt: [String: Any] = [
        "schemaVersion": 1,
        "manifestSha256": manifestHash,
        "modelDirectoryPath": directory.standardizedFileURL.path,
        "verificationTimestamp": "2026-10-25T00:00:00Z",
        "toolVersion": "TurboFieldfareAppCoreTests",
        "files": [:],
    ]
    let receiptData = try JSONSerialization.data(withJSONObject: receipt,
                                                 options: [.sortedKeys])
    try receiptData.write(to: directory.appendingPathComponent("verified-install.json"))

    if includeIdentity {
        let identityJSON: [String: Any] = [
            "family": family.rawValue,
            "chatTemplateIdentity": identity,
            "chatTemplateSHA256": String(repeating: "0", count: 64),
            "tokenizerSHA256": String(repeating: "0", count: 64),
            "tokenizerConfigSHA256": String(repeating: "0", count: 64),
        ]
        try JSONSerialization.data(withJSONObject: identityJSON, options: [.sortedKeys])
            .write(to: tokenizer.appendingPathComponent("identity.json"))
    }
    return directory
}

private func quantSlot(_ bits: Int) -> [String: Any] {
    [
        "weightBits": bits,
        "scheme": "affine",
        "scaleType": "bf16",
        "biasType": "bf16",
        "groupSize": 64,
    ]
}
