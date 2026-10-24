/// Architecture family of a `.gturbo` model. Absent in a v1 manifest means
/// `.gemma4`, so existing Gemma manifests keep their meaning.
public enum ModelFamily: String, Codable, CaseIterable, Sendable {
    case gemma4
    case lfm2Moe = "lfm2_moe"
}

/// Per-layer attention mechanism. Gemma manifests derive this from
/// `fullAttentionLayerMask`; LFM writes it explicitly.
public enum LayerType: String, Codable, Sendable {
    case conv
    case fullAttention = "full_attention"
    case slidingAttention = "sliding_attention"
}

/// Rotary position-embedding convention.
public enum RopeStyle: String, Codable, Sendable {
    case gemmaPartial = "gemma_partial"
    case neoxFull = "neox_full"
}

/// Router scoring convention for the mixture-of-experts gate.
public enum RouterKind: String, Codable, Sendable {
    case gemmaSoftmaxTopK = "gemma_softmax_topk"
    case lfm2SigmoidTopKBias = "lfm2_sigmoid_topk_bias"
}
