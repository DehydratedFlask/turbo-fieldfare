import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

@Suite
struct ArchInfoTests {
    @Test func parsesFlatLFMMoEConfig() throws {
        let path = try writeLFMConfig()
        defer { removeConfig(at: path) }
        let info = try ArchInfo.load(configPath: path)
        #expect(info.family == .lfm2Moe)
        #expect(info.hiddenSize == 2048 && info.numLayers == 40)
        #expect(info.layerTypes?.count == 40)
        #expect(info.moeLayerMask?.prefix(2) == [0, 0])
        #expect(info.convKernelSize == 3)
        #expect(info.routerKind == .lfm2SigmoidTopKBias)
        #expect(info.routedScalingFactor == 1.0)

        // Everything design §9.2 derives from the flat config.
        #expect(info.headDim == 64 && info.fullHeadDim == 64)
        #expect(info.numKVHeads == 8 && info.numFullKVHeads == 8)
        #expect(info.ropeTheta == 10_000_000 && info.fullRopeTheta == 10_000_000)
        #expect(info.slidingWindow == 0 && info.finalLogitSoftcap == 0)
        #expect(info.partialRotaryFactor == 1.0)
        #expect(info.attentionScale == 0.125)
        #expect(info.rmsNormUnitOffset == false)
        #expect(info.ropeStyle == .neoxFull)
        #expect(info.hasSharedExpert == false)
        #expect(info.embeddingScaledBySqrtHidden == false)
        #expect(info.numDenseLayers == 2)
        #expect(info.routerNormalizeTopK == true)
        #expect(info.routerBias == true)
        #expect(info.attentionKEqV == false)
        #expect(info.hiddenActivation == "silu")
        #expect(info.maxPositionEmbeddings == 131_072)
        #expect(info.vocabSize == 128_000)
        #expect(info.numExperts == 64 && info.topKExperts == 4)
        #expect(info.moeLayerMask?.filter { $0 == 1 }.count == 38)
        #expect(info.layerTypes?.filter { $0 == .fullAttention }.count == 10)
        #expect(info.fullAttentionLayerMask[2] == 1)
        #expect(info.fullAttentionLayerMask[0] == 0)
    }

    @Test func rejectsUnknownModelType() throws {
        let path = try writeConfig(modelType: "llama")
        defer { removeConfig(at: path) }
        #expect(throws: RepackError.self) {
            _ = try ArchInfo.load(configPath: path)
        }
    }

    @Test func gemmaTextConfigStillParsesAsGemma() throws {
        let path = try writeJSONObject(gemmaConfig())
        defer { removeConfig(at: path) }
        let info = try ArchInfo.load(configPath: path)
        #expect(info.family == .gemma4)
        #expect(info.hiddenSize == 2816 && info.numLayers == 30)
        #expect(info.headDim == 128 && info.fullHeadDim == 256)
        #expect(info.ropeTheta == 10_000 && info.fullRopeTheta == 1_000_000)
        #expect(info.partialRotaryFactor == 0.25)
        #expect(info.layerTypes == nil)
        #expect(info.moeLayerMask == nil)
        #expect(info.convKernelSize == nil)
        #expect(info.routerKind == nil)
        #expect(info.maxPositionEmbeddings == nil)
    }

    // MARK: - Fixtures

    private func writeLFMConfig() throws -> String {
        try writeJSONObject(lfmConfig())
    }

    private func writeConfig(modelType: String) throws -> String {
        try writeJSONObject(["model_type": modelType])
    }

    private func writeJSONObject(_ object: [String: Any]) throws -> String {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("archinfo-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
        let path = directory.appendingPathComponent("config.json")
        try JSONSerialization.data(withJSONObject: object).write(to: path)
        return path.path
    }

    private func removeConfig(at path: String) {
        try? FileManager.default.removeItem(
            atPath: (path as NSString).deletingLastPathComponent)
    }

    private func lfmConfig() -> [String: Any] {
        let fullAttention = Set([2, 6, 10, 14, 18, 22, 26, 30, 34, 38])
        let layerTypes = (0..<40).map {
            fullAttention.contains($0) ? "full_attention" : "conv"
        }
        return [
            "model_type": "lfm2_moe",
            "hidden_size": 2048,
            "intermediate_size": 11_776,
            "moe_intermediate_size": 1_536,
            "num_hidden_layers": 40,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128_000,
            "max_position_embeddings": 131_072,
            "layer_types": layerTypes,
            "num_dense_layers": 2,
            "conv_L_cache": 3,
            "norm_eps": 1e-5,
            "rope_parameters": ["rope_theta": 10_000_000.0],
            "tie_word_embeddings": true,
            "norm_topk_prob": true,
            "use_expert_bias": true,
            "routed_scaling_factor": 1.0,
            "num_experts": 64,
            "num_experts_per_tok": 4,
        ]
    }

    private func gemmaConfig() -> [String: Any] {
        [
            "model_type": "gemma4",
            "text_config": [
                "hidden_size": 2816,
                "intermediate_size": 2112,
                "moe_intermediate_size": 704,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "num_global_key_value_heads": 2,
                "head_dim": 128,
                "global_head_dim": 256,
                "vocab_size": 262_144,
                "sliding_window": 1024,
                "final_logit_softcapping": 30.0,
                "num_hidden_layers": 30,
                "num_experts": 128,
                "top_k_experts": 8,
                "layer_types": [String](repeating: "sliding_attention", count: 30),
                "rope_parameters": [
                    "full_attention": [
                        "rope_theta": 1_000_000.0,
                        "partial_rotary_factor": 0.25,
                    ],
                    "sliding_attention": ["rope_theta": 10_000.0],
                ],
                "tie_word_embeddings": true,
                "attention_k_eq_v": true,
                "hidden_activation": "gelu_pytorch_tanh",
            ],
        ]
    }
}
