import CryptoKit
import Foundation
import Testing
import TurboFieldfareFormat
import TurboFieldfareRepackCore

@Suite struct TokenizerIdentityTests {
    @Test func identityHashesMatchCopiedSidecarBytes() throws {
        let dir = try makeTokenizerSidecar(family: .lfm2Moe)
        defer { try? FileManager.default.removeItem(at: dir) }

        let identity = try TokenizerIdentity.write(for: dir, family: .lfm2Moe)
        let json = try JSONDecoder().decode(
            TokenizerIdentity.self,
            from: Data(contentsOf: dir.appendingPathComponent("identity.json")))

        #expect(json == identity)
        #expect(identity.family == .lfm2Moe)
        #expect(identity.chatTemplateIdentity == "lfm2.5-24b-a2b-chatml-v1")
        #expect(identity.chatTemplateSHA256.count == 64)
        #expect(identity.tokenizerSHA256.count == 64)
        #expect(identity.tokenizerConfigSHA256.count == 64)
        // The hashes must cover the bytes actually copied, not a template or a
        // normalized re-encoding of them.
        #expect(identity.chatTemplateSHA256 == sha256(chatTemplateBytes))
        #expect(identity.tokenizerSHA256 == sha256(tokenizerBytes))
        #expect(identity.tokenizerConfigSHA256 == sha256(tokenizerConfigBytes))
        // The runtime reads the same file through the format-only loader.
        #expect(try TokenizerIdentity.load(from: dir) == identity)
    }

    @Test func lfmSidecarRequiresEveryHashedFile() throws {
        let dir = try temporaryDirectory("lfm-missing-template")
        defer { try? FileManager.default.removeItem(at: dir) }
        try tokenizerBytes.write(to: dir.appendingPathComponent("tokenizer.json"))
        try tokenizerConfigBytes.write(to: dir.appendingPathComponent("tokenizer_config.json"))

        #expect(throws: RepackError.self) {
            try TokenizerIdentity.write(for: dir, family: .lfm2Moe)
        }
        #expect(!FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("identity.json").path))
    }

    @Test func visionInstallRejectsLFMTextManifestBeforeWriting() throws {
        let output = try temporaryDirectory("vision-output")
        let textModel = try writeInstalledLFMTextModel()
        defer {
            try? FileManager.default.removeItem(at: output)
            try? FileManager.default.removeItem(at: textModel)
        }

        #expect(throws: RepackError.self) {
            try RemoteVisionPackInstaller.ensureGemmaTextModel(
                at: textModel, outputDirectory: output)
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: output.path).isEmpty)
    }

    @Test func gemmaSidecarWritesNoIdentityFile() throws {
        let dir = try makeTokenizerSidecar(family: .gemma4)
        defer { try? FileManager.default.removeItem(at: dir) }

        _ = try TokenizerIdentity.writeIfNeeded(for: dir, family: .gemma4)
        #expect(!FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("identity.json").path))
    }
}

// MARK: - Fixtures shared with `RepackCLITests`

private let chatTemplateBytes = Data(
    "{% for message in messages %}{{ message.content }}{% endfor %}\n".utf8)
private let tokenizerBytes = Data(
    #"{"model":{"type":"BPE","vocab":{"<|startoftext|>":0}}}"#.utf8)
private let tokenizerConfigBytes = Data(
    #"{"bos_token":"<|startoftext|>","chat_template":"chatml"}"#.utf8)

/// Writes a tokenizer directory whose three hashed files hold fixed bytes.
private func makeTokenizerSidecar(family: ModelFamily) throws -> URL {
    let dir = try temporaryDirectory("tokenizer-sidecar-\(family.rawValue)")
    try chatTemplateBytes.write(to: dir.appendingPathComponent("chat_template.jinja"))
    try tokenizerBytes.write(to: dir.appendingPathComponent("tokenizer.json"))
    try tokenizerConfigBytes.write(
        to: dir.appendingPathComponent("tokenizer_config.json"))
    return dir
}

/// Writes a minimal installed LFM text pack. Not loadable as a model — only
/// `manifest.json` matters to the family check this exercises.
func writeInstalledLFMTextModel(in parent: URL? = nil) throws -> URL {
    let directory: URL
    if let parent {
        directory = parent
    } else {
        directory = try temporaryDirectory("lfm-text-model")
    }
    try FileManager.default.createDirectory(
        at: directory, withIntermediateDirectories: true)
    let manifest = GTurboManifestV1(
        flags: ["streamingPresent": true],
        modelID: "test/lfm2.5-24b-a2b",
        sourceSnapshotHash: "sha256:" + String(repeating: "a", count: 64),
        arch: GTurboManifestArchV1(
            hiddenSize: 64, ffnIntermediate: 64, moeIntermediateSize: 64,
            numHeads: 1, numKVHeads: 1, numFullKVHeads: 1,
            headDim: 64, fullHeadDim: 64, vocabSize: 128,
            slidingWindow: 0, finalLogitSoftcap: 0,
            ropeTheta: 10_000, fullRopeTheta: 10_000,
            partialRotaryFactor: 1, numLayers: 1, numExperts: 4,
            topKExperts: 1, tieWordEmbeddings: true, attentionKEqV: false,
            hiddenActivation: "silu", fullAttentionLayerMask: [1],
            family: .lfm2Moe,
            layerTypes: [.fullAttention],
            moeLayerMask: [1],
            attentionScale: 0.125,
            rmsNormUnitOffset: false,
            ropeStyle: .neoxFull,
            convKernelSize: 3,
            routerKind: .lfm2SigmoidTopKBias,
            sharedExpert: false,
            embeddingScaledBySqrtHidden: false,
            numDenseLayers: 0,
            routerNormalizeTopK: true,
            routerBias: true,
            routedScalingFactor: 1),
        quant: nil,
        files: [:],
        expertsPerLayer: 4, numLayers: 1, expertStride: 16_384,
        bitWidthOverridesHonored: nil)
    try GTurboManifestCodec.encode(manifest).write(
        to: directory.appendingPathComponent("manifest.json"))
    return directory
}

private func temporaryDirectory(_ tag: String) throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("tf-\(tag)-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}
