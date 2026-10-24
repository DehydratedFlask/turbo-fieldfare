import Foundation
import TurboFieldfareFormat

/// Architecture facts mirrored into `manifest.json -> arch`. Cross-checked by
/// the runtime loader at startup.
struct ArchInfo: Sendable, Equatable {
    let hiddenSize: Int
    let intermediateSize: Int          // shared expert FFN
    let moeIntermediateSize: Int       // per-expert FFN
    let numHeads: Int
    let numKVHeads: Int
    let numFullKVHeads: Int
    let headDim: Int
    let fullHeadDim: Int
    let vocabSize: Int
    let slidingWindow: Int
    let finalLogitSoftcap: Double
    let ropeTheta: Double
    let fullRopeTheta: Double
    let partialRotaryFactor: Double
    let numLayers: Int
    let numExperts: Int
    let topKExperts: Int
    let tieWordEmbeddings: Bool
    let attentionKEqV: Bool
    /// 1 if `full_attention`, 0 if `sliding_attention`. Indexed by layer.
    let fullAttentionLayerMask: [UInt8]
    let hiddenActivation: String

    // Family traits. Optional so a Gemma `ArchInfo` built without them keeps
    // the historical semantics; `family` itself defaults to `.gemma4`.
    let maxPositionEmbeddings: Int?
    let family: ModelFamily
    /// Per-layer mechanism. Nil for Gemma, where it derives from
    /// `fullAttentionLayerMask`.
    let layerTypes: [LayerType]?
    /// `1` where the layer has routed experts, `0` for a dense layer.
    let moeLayerMask: [UInt8]?
    let attentionScale: Double?
    let rmsNormUnitOffset: Bool?
    let ropeStyle: RopeStyle?
    let convKernelSize: Int?
    let routerKind: RouterKind?
    let hasSharedExpert: Bool?
    let embeddingScaledBySqrtHidden: Bool?
    let numDenseLayers: Int?
    let routerNormalizeTopK: Bool?
    let routerBias: Bool?
    let routedScalingFactor: Double?

    init(hiddenSize: Int,
         intermediateSize: Int,
         moeIntermediateSize: Int,
         numHeads: Int,
         numKVHeads: Int,
         numFullKVHeads: Int,
         headDim: Int,
         fullHeadDim: Int,
         vocabSize: Int,
         slidingWindow: Int,
         finalLogitSoftcap: Double,
         ropeTheta: Double,
         fullRopeTheta: Double,
         partialRotaryFactor: Double,
         numLayers: Int,
         numExperts: Int,
         topKExperts: Int,
         tieWordEmbeddings: Bool,
         attentionKEqV: Bool,
         fullAttentionLayerMask: [UInt8],
         hiddenActivation: String,
         maxPositionEmbeddings: Int? = nil,
         family: ModelFamily = .gemma4,
         layerTypes: [LayerType]? = nil,
         moeLayerMask: [UInt8]? = nil,
         attentionScale: Double? = nil,
         rmsNormUnitOffset: Bool? = nil,
         ropeStyle: RopeStyle? = nil,
         convKernelSize: Int? = nil,
         routerKind: RouterKind? = nil,
         hasSharedExpert: Bool? = nil,
         embeddingScaledBySqrtHidden: Bool? = nil,
         numDenseLayers: Int? = nil,
         routerNormalizeTopK: Bool? = nil,
         routerBias: Bool? = nil,
         routedScalingFactor: Double? = nil) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numFullKVHeads = numFullKVHeads
        self.headDim = headDim
        self.fullHeadDim = fullHeadDim
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.finalLogitSoftcap = finalLogitSoftcap
        self.ropeTheta = ropeTheta
        self.fullRopeTheta = fullRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.numLayers = numLayers
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionKEqV = attentionKEqV
        self.fullAttentionLayerMask = fullAttentionLayerMask
        self.hiddenActivation = hiddenActivation
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.family = family
        self.layerTypes = layerTypes
        self.moeLayerMask = moeLayerMask
        self.attentionScale = attentionScale
        self.rmsNormUnitOffset = rmsNormUnitOffset
        self.ropeStyle = ropeStyle
        self.convKernelSize = convKernelSize
        self.routerKind = routerKind
        self.hasSharedExpert = hasSharedExpert
        self.embeddingScaledBySqrtHidden = embeddingScaledBySqrtHidden
        self.numDenseLayers = numDenseLayers
        self.routerNormalizeTopK = routerNormalizeTopK
        self.routerBias = routerBias
        self.routedScalingFactor = routedScalingFactor
    }

    static func load(configPath: String) throws -> ArchInfo {
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "no text_config")
        }
        if root["text_config"] != nil {
            guard let textConfig = root["text_config"] as? [String: Any] else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "no text_config")
            }
            return try loadGemma(textConfig: textConfig, configPath: configPath)
        }
        let modelType = (root["model_type"] as? String) ?? "<missing>"
        guard modelType == ModelFamily.lfm2Moe.rawValue else {
            throw RepackError.unsupportedModelFamily(modelType)
        }
        return try loadLFMMoE(root: root, configPath: configPath)
    }

    /// Gemma's nested `text_config` path, unchanged by the family split.
    private static func loadGemma(textConfig tc: [String: Any],
                                  configPath: String) throws -> ArchInfo {
        func i(_ k: String) throws -> Int {
            guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func d(_ k: String) throws -> Double {
            guard let n = (tc[k] as? Double) ?? (tc[k] as? NSNumber)?.doubleValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        let layerTypes = (tc["layer_types"] as? [String]) ?? []
        let mask = layerTypes.map { UInt8($0 == "full_attention" ? 1 : 0) }
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        let ropeFull = (rope["full_attention"] as? [String: Any]) ?? [:]
        let ropeSWA  = (rope["sliding_attention"] as? [String: Any]) ?? [:]
        let prf = (ropeFull["partial_rotary_factor"] as? Double)
            ?? (ropeFull["partial_rotary_factor"] as? NSNumber)?.doubleValue ?? 0.25
        let fullTheta = (ropeFull["rope_theta"] as? Double)
            ?? (ropeFull["rope_theta"] as? NSNumber)?.doubleValue ?? 1_000_000.0
        let swaTheta = (ropeSWA["rope_theta"] as? Double)
            ?? (ropeSWA["rope_theta"] as? NSNumber)?.doubleValue ?? 10_000.0
        let kEqV = (tc["attention_k_eq_v"] as? Bool) ?? false
        let tie = (tc["tie_word_embeddings"] as? Bool) ?? false
        let act = (tc["hidden_activation"] as? String) ?? "gelu_pytorch_tanh"
        return ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: try i("intermediate_size"),
            moeIntermediateSize: try i("moe_intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            numFullKVHeads: try i("num_global_key_value_heads"),
            headDim: try i("head_dim"),
            fullHeadDim: try i("global_head_dim"),
            vocabSize: try i("vocab_size"),
            slidingWindow: try i("sliding_window"),
            finalLogitSoftcap: try d("final_logit_softcapping"),
            ropeTheta: swaTheta,
            fullRopeTheta: fullTheta,
            partialRotaryFactor: prf,
            numLayers: try i("num_hidden_layers"),
            numExperts: try i("num_experts"),
            topKExperts: try i("top_k_experts"),
            tieWordEmbeddings: tie,
            attentionKEqV: kEqV,
            fullAttentionLayerMask: mask,
            hiddenActivation: act,
            family: .gemma4)
    }

    /// LFM2.5 MoE's flat `config.json`, with the derived values from design
    /// §9.2.
    private static func loadLFMMoE(root: [String: Any],
                                   configPath: String) throws -> ArchInfo {
        func i(_ k: String) throws -> Int {
            guard let n = (root[k] as? Int) ?? (root[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func d(_ k: String) throws -> Double {
            guard let n = (root[k] as? Double) ?? (root[k] as? NSNumber)?.doubleValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func b(_ k: String) throws -> Bool {
            if let value = root[k] as? Bool { return value }
            if let number = root[k] as? NSNumber { return number.boolValue }
            throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
        }

        let hiddenSize = try i("hidden_size")
        let numHeads = try i("num_attention_heads")
        guard numHeads > 0 else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "num_attention_heads must be positive")
        }
        let headDim = hiddenSize / numHeads
        guard headDim > 0 else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "head_dim must be positive")
        }
        let numLayers = try i("num_hidden_layers")
        guard numLayers > 0 else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "num_hidden_layers must be positive")
        }
        let numKVHeads = try i("num_key_value_heads")

        guard let layerTypeNames = root["layer_types"] as? [String] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "missing layer_types")
        }
        guard layerTypeNames.count == numLayers else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "layer_types has \(layerTypeNames.count) entries for \(numLayers) layers")
        }
        let layerTypes = try layerTypeNames.map { name -> LayerType in
            guard let type = LayerType(rawValue: name) else {
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "unknown layer_types entry \(name)")
            }
            return type
        }
        let mask = layerTypes.map { UInt8($0 == .fullAttention ? 1 : 0) }

        let numDenseLayers = try i("num_dense_layers")
        guard numDenseLayers >= 0, numDenseLayers <= numLayers else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "num_dense_layers \(numDenseLayers) exceeds \(numLayers) layers")
        }
        let moeLayerMask = (0..<numLayers).map {
            UInt8($0 < numDenseLayers ? 0 : 1)
        }

        let ropeTheta: Double
        if let rope = root["rope_parameters"] as? [String: Any],
           let theta = (rope["rope_theta"] as? Double)
            ?? (rope["rope_theta"] as? NSNumber)?.doubleValue {
            ropeTheta = theta
        } else {
            ropeTheta = try d("rope_theta")
        }

        // The repacker does not carry epsilon into the manifest, but a config
        // without it is not the pinned checkpoint.
        _ = try d("norm_eps")

        return ArchInfo(
            hiddenSize: hiddenSize,
            intermediateSize: try i("intermediate_size"),
            moeIntermediateSize: try i("moe_intermediate_size"),
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            numFullKVHeads: numKVHeads,
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: try i("vocab_size"),
            slidingWindow: 0,
            finalLogitSoftcap: 0,
            ropeTheta: ropeTheta,
            fullRopeTheta: ropeTheta,
            partialRotaryFactor: 1.0,
            numLayers: numLayers,
            numExperts: try i("num_experts"),
            topKExperts: try i("num_experts_per_tok"),
            tieWordEmbeddings: try b("tie_word_embeddings"),
            attentionKEqV: false,
            fullAttentionLayerMask: mask,
            hiddenActivation: "silu",
            maxPositionEmbeddings: try i("max_position_embeddings"),
            family: .lfm2Moe,
            layerTypes: layerTypes,
            moeLayerMask: moeLayerMask,
            attentionScale: 1.0 / Double(headDim).squareRoot(),
            rmsNormUnitOffset: false,
            ropeStyle: .neoxFull,
            convKernelSize: try i("conv_L_cache"),
            routerKind: .lfm2SigmoidTopKBias,
            hasSharedExpert: false,
            embeddingScaledBySqrtHidden: false,
            numDenseLayers: numDenseLayers,
            routerNormalizeTopK: try b("norm_topk_prob"),
            routerBias: try b("use_expert_bias"),
            routedScalingFactor: try d("routed_scaling_factor"))
    }
}
