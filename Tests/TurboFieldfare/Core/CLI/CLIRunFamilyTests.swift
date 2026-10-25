import Foundation
import Testing
import TurboFieldfare
import TurboFieldfareFormat
@testable import TurboFieldfareCLICore

@Suite("CLI family dispatch")
struct CLIRunFamilyTests {
    @Test func lfmWithImagesFailsClosed() async throws {
        let directory = try makeMinimalInstalledModel(family: .lfm2Moe)
        defer { try? FileManager.default.removeItem(at: directory) }
        let args = try Args.parse([
            "--model", directory.path,
            "--chat-prompt", "hi",
            "--image", "/tmp/does-not-need-to-exist.png",
        ])
        let pipe = Pipe()
        let result = await run(args: args, stdout: .nullDevice,
                               stderr: pipe.fileHandleForWriting)
        pipe.fileHandleForWriting.closeFile()
        let message = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(),
                             as: UTF8.self)
        #expect(result.exitCode == 2)
        #expect(message.contains("lfm2_moe"))
        #expect(message.lowercased().contains("image"))
    }

    @Test func footerNamesModelAndFamily() {
        let footer = footerLine(modelID: "LiquidAI/LFM2.5-24B-A2B-MLX-4bit",
                                family: .lfm2Moe, reason: .eos,
                                prefillTokens: 12, newTokens: 4, decodeSeconds: 1.5)
        #expect(footer.contains("LiquidAI/LFM2.5-24B-A2B-MLX-4bit"))
        #expect(footer.contains("family=lfm2_moe"))
    }

    @Test func gemmaPromptInputStillUsesTheUserRole() throws {
        let args = try Args.parse(["--model", "gemma4.gturbo", "--prompt", "hi"])
        guard case .raw(let text) = try parseInput(args: args) else {
            Issue.record("expected raw prompt"); return
        }
        #expect(text == "hi")

        let chat = try Args.parse(["--model", "gemma4.gturbo", "--chat-prompt", "hi"])
        guard case .messages(let messages) = try parseInput(args: chat) else {
            Issue.record("expected messages"); return
        }
        #expect(messages.first?.role == .user)
    }
}

private func makeMinimalInstalledModel(family: ModelFamily) throws -> URL {
    let manager = FileManager.default
    let directory = manager.temporaryDirectory
        .appendingPathComponent("cli-\(family.rawValue)-\(UUID().uuidString).gturbo",
                                isDirectory: true)
    try manager.createDirectory(at: directory, withIntermediateDirectories: true)
    let arch = family == .lfm2Moe ? ArchConfig.lfm2_5_24B_A2B : ArchConfig.gemma4_26B_A4B
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
        archDict["convKernelSize"] = arch.convKernelSize
        archDict["routerKind"] = arch.routerKind.rawValue
        archDict["sharedExpert"] = arch.hasSharedExpert
        archDict["embeddingScaledBySqrtHidden"] = arch.embeddingScaledBySqrtHidden
        archDict["numDenseLayers"] = arch.numDenseLayers
        archDict["routerNormalizeTopK"] = arch.routerNormalizeTopK
        archDict["routerBias"] = arch.routerBias
    }
    let manifest: [String: Any] = [
        "magic": "GTURBO",
        "versionMajor": 1,
        "versionMinor": 0,
        "flags": [String: Bool](),
        "modelID": "test/\(family.rawValue)",
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
