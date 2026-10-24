import TurboFieldfareFormat

/// Resolves installed tensor names to resident views. `Model` is the only
/// production conformer; tests use a tiny `MTLBuffer`-backed lookup so they do
/// not need a full LFM pack.
protocol LFM2TensorLookup {
    func tensor(named name: String) throws -> TensorView
}

extension Model: LFM2TensorLookup {}

/// LFM2.5 24B-A2B resident weights. Names match what `LFMRepackPlanner`
/// installs: the embedding and `embedding_norm` keep `model.`, while layer
/// tensors are stripped to `layers.N.*`.
///
/// Accessors check the layer type before resolving, so asking a conv layer for
/// an attention tensor fails with `ModelError.tensorNotFound` instead of
/// returning an unrelated view.
public struct LFM2Weights {
    public let config: ArchConfig
    private let lookup: any LFM2TensorLookup

    public init(model: Model) {
        self.config = model.config
        self.lookup = model
    }

    init(lookup: any LFM2TensorLookup, config: ArchConfig) {
        self.config = config
        self.lookup = lookup
    }

    // MARK: - Embedding and final norm

    public func embedding() throws -> TensorView {
        try lookup.tensor(named: "model.embed_tokens.weight")
    }

    public func embeddingNorm() throws -> TensorView {
        try lookup.tensor(named: "model.embedding_norm.weight")
    }

    // MARK: - Per-layer norms

    public func operatorNorm(layer L: Int) throws -> TensorView {
        let name = "layers.\(L).operator_norm.weight"
        try requireKnownLayer(L, name: name)
        return try lookup.tensor(named: name)
    }

    public func ffnNorm(layer L: Int) throws -> TensorView {
        let name = "layers.\(L).ffn_norm.weight"
        try requireKnownLayer(L, name: name)
        return try lookup.tensor(named: name)
    }

    // MARK: - Short convolution

    public func convIn(layer L: Int) throws -> TensorView {
        let name = "layers.\(L).conv.in_proj.weight"
        try requireConvLayer(L, name: name)
        return try lookup.tensor(named: name)
    }

    public func convOut(layer L: Int) throws -> TensorView {
        let name = "layers.\(L).conv.out_proj.weight"
        try requireConvLayer(L, name: name)
        return try lookup.tensor(named: name)
    }

    public func convWeight(layer L: Int) throws -> TensorView {
        let name = "layers.\(L).conv.conv.weight"
        try requireConvLayer(L, name: name)
        return try lookup.tensor(named: name)
    }

    // MARK: - Attention

    public func qProj(layer L: Int) throws -> TensorView {
        let name = "layers.\(L).self_attn.q_proj.weight"
        try requireAttentionLayer(L, name: name)
        return try lookup.tensor(named: name)
    }

    public func kProj(layer L: Int) throws -> TensorView {
        let name = "layers.\(L).self_attn.k_proj.weight"
        try requireAttentionLayer(L, name: name)
        return try lookup.tensor(named: name)
    }

    public func vProj(layer L: Int) throws -> TensorView {
        let name = "layers.\(L).self_attn.v_proj.weight"
        try requireAttentionLayer(L, name: name)
        return try lookup.tensor(named: name)
    }

    public func oProj(layer L: Int) throws -> TensorView {
        let name = "layers.\(L).self_attn.o_proj.weight"
        try requireAttentionLayer(L, name: name)
        return try lookup.tensor(named: name)
    }

    public func qLayernorm(layer L: Int) throws -> TensorView {
        let name = "layers.\(L).self_attn.q_layernorm.weight"
        try requireAttentionLayer(L, name: name)
        return try lookup.tensor(named: name)
    }

    public func kLayernorm(layer L: Int) throws -> TensorView {
        let name = "layers.\(L).self_attn.k_layernorm.weight"
        try requireAttentionLayer(L, name: name)
        return try lookup.tensor(named: name)
    }

    // MARK: - Dense FFN (leading dense layers only)

    public func denseGate(layer L: Int) throws -> TensorView {
        let name = "layers.\(L).feed_forward.gate_proj.weight"
        try requireDenseLayer(L, name: name)
        return try lookup.tensor(named: name)
    }

    public func denseUp(layer L: Int) throws -> TensorView {
        let name = "layers.\(L).feed_forward.up_proj.weight"
        try requireDenseLayer(L, name: name)
        return try lookup.tensor(named: name)
    }

    public func denseDown(layer L: Int) throws -> TensorView {
        let name = "layers.\(L).feed_forward.down_proj.weight"
        try requireDenseLayer(L, name: name)
        return try lookup.tensor(named: name)
    }

    // MARK: - MoE router (MoE layers only)

    public func routerGate(layer L: Int) throws -> TensorView {
        let name = "layers.\(L).feed_forward.gate.weight"
        try requireMoELayer(L, name: name)
        return try lookup.tensor(named: name)
    }

    public func expertBias(layer L: Int) throws -> TensorView {
        let name = "layers.\(L).feed_forward.expert_bias"
        try requireMoELayer(L, name: name)
        return try lookup.tensor(named: name)
    }

    // MARK: - Layer-type gates

    private func requireKnownLayer(_ layer: Int, name: String) throws {
        guard layer >= 0,
              layer < config.layerTypes.count,
              layer < config.numLayers else {
            throw ModelError.tensorNotFound(name: name)
        }
    }

    private func requireConvLayer(_ layer: Int, name: String) throws {
        try requireKnownLayer(layer, name: name)
        guard config.layerTypes[layer] == .conv else {
            throw ModelError.tensorNotFound(name: name)
        }
    }

    private func requireAttentionLayer(_ layer: Int, name: String) throws {
        try requireKnownLayer(layer, name: name)
        guard config.layerTypes[layer] != .conv else {
            throw ModelError.tensorNotFound(name: name)
        }
    }

    private func requireDenseLayer(_ layer: Int, name: String) throws {
        try requireKnownLayer(layer, name: name)
        guard layer < config.numDenseLayers else {
            throw ModelError.tensorNotFound(name: name)
        }
    }

    private func requireMoELayer(_ layer: Int, name: String) throws {
        guard layer >= 0,
              layer < config.numLayers,
              layer < config.moeLayerMask.count,
              config.moeLayerMask[layer] == 1 else {
            throw ModelError.tensorNotFound(name: name)
        }
    }
}
