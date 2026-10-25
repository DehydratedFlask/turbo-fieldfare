import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfare

@Suite("Conversation identity")
struct ConversationIdentityTests {
    @Test func lfmIdentityComesFromTheTokenizerSidecar() throws {
        let directory = try makeInstalledModelDirectory(family: .lfm2Moe)
        defer { try? FileManager.default.removeItem(at: directory) }
        let identity = try ConversationIdentity.forModelDirectory(directory)
        #expect(identity.templateIdentity == "lfm2.5-24b-a2b-chatml-v1")
        #expect(identity.imageProcessingVersion == 0)
    }

    @Test func gemmaIdentityIsUnchanged() throws {
        let directory = try makeInstalledModelDirectory(family: .gemma4)
        defer { try? FileManager.default.removeItem(at: directory) }
        let identity = try ConversationIdentity.forModelDirectory(directory)
        #expect(identity.templateIdentity == GFTokenizer.chatTemplateIdentity)
    }

    @Test func lfmIdentityFailsClosedWhenTheSidecarIsMissing() throws {
        let directory = try makeInstalledModelDirectory(family: .lfm2Moe,
                                                        includeIdentity: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(throws: ConversationIdentityError.self) {
            _ = try ConversationIdentity.forModelDirectory(directory)
        }
    }
}

private func makeInstalledModelDirectory(
    family: ModelFamily,
    includeIdentity: Bool = true
) throws -> URL {
    let manager = FileManager.default
    let directory = manager.temporaryDirectory
        .appendingPathComponent("identity-\(family.rawValue)-\(UUID().uuidString).gturbo",
                                isDirectory: true)
    try manager.createDirectory(at: directory, withIntermediateDirectories: true)
    let arch = ArchConfig.canonical(for: family) ?? .gemma4_26B_A4B
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
    ]
    if family == .lfm2Moe {
        archDict["family"] = family.rawValue
        archDict["layerTypes"] = arch.layerTypes.map(\.rawValue)
        archDict["moeLayerMask"] = arch.moeLayerMask.map(Int.init)
        archDict["attentionScale"] = arch.attentionScale
        archDict["rmsNormUnitOffset"] = arch.rmsNormUnitOffset
        archDict["ropeStyle"] = arch.ropeStyle.rawValue
        archDict["convKernelSize"] = arch.convKernelSize
        archDict["routerKind"] = arch.routerKind.rawValue
        archDict["sharedExpert"] = arch.hasSharedExpert
        archDict["embeddingScaledBySqrtHidden"] = arch.embeddingScaledBySqrtHidden
        archDict["numDenseLayers"] = arch.numDenseLayers
        archDict["routerNormalizeTopK"] = arch.routerNormalizeTopK
        archDict["routerBias"] = arch.routerBias
        archDict["routedScalingFactor"] = arch.routedScalingFactor
    }
    let manifest: [String: Any] = [
        "magic": "GTURBO",
        "versionMajor": 1,
        "versionMinor": 0,
        "flags": [String: Bool](),
        "modelID": "test/\(family.rawValue)",
        "sourceSnapshotHash": "sha256:" + String(repeating: "0", count: 64),
        "quant": family == .lfm2Moe
            ? ["embedding": quantSlot(6), "attention": quantSlot(4),
               "sharedExpert": quantSlot(4), "routedExpert": quantSlot(4)]
            : ["embedding": quantSlot(4), "attention": quantSlot(4),
               "router": quantSlot(8), "sharedExpert": quantSlot(4),
               "routedExpert": quantSlot(4)],
        "arch": archDict,
        "files": [
            "model_weights.bin": ["size": 0, "sha256": Sha256Verifier.hashData(Data())],
            "packed_experts/layout.json": ["size": 2, "sha256": Sha256Verifier.hashData(Data("{}".utf8))],
        ],
        "expertsPerLayer": arch.numExperts,
        "numLayers": arch.numLayers,
        "expertStride": arch.admission.expertStrideBytes,
    ]
    try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
        .write(to: directory.appendingPathComponent("manifest.json"))
    try Data().write(to: directory.appendingPathComponent("model_weights.bin"))
    let experts = directory.appendingPathComponent("packed_experts", isDirectory: true)
    try manager.createDirectory(at: experts, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: experts.appendingPathComponent("layout.json"))
    if includeIdentity {
        let tokenizer = directory.appendingPathComponent("tokenizer", isDirectory: true)
        try manager.createDirectory(at: tokenizer, withIntermediateDirectories: true)
        let identity: [String: Any] = [
            "family": family.rawValue,
            "chatTemplateIdentity": "lfm2.5-24b-a2b-chatml-v1",
            "chatTemplateSHA256": String(repeating: "0", count: 64),
            "tokenizerSHA256": String(repeating: "0", count: 64),
            "tokenizerConfigSHA256": String(repeating: "0", count: 64),
        ]
        try JSONSerialization.data(withJSONObject: identity, options: [.sortedKeys])
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
