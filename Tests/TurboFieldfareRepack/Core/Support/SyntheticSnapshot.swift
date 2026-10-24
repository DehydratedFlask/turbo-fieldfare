import Foundation
@testable import TurboFieldfareRepackCore

/// Synthesises a tiny MLX-affine-quantized safetensors snapshot inside a
/// temporary directory. The remote repack tests need only deterministic bytes
/// plus matching `config.json` and `model.safetensors.index.json` metadata.
enum SyntheticSnapshot {

    struct Snapshot {
        let shardPath: String
    }

    struct Arch {
        let hidden: Int = 128
        let moeIntermediate: Int = 64
        let intermediate: Int = 192
        let numHeads: Int = 2
        let numKVHeads: Int = 2
        let numGlobalKVHeads: Int = 2
        let headDim: Int = 32
        let globalHeadDim: Int = 64
        let vocab: Int = 512
        let numLayers: Int = 2
        let numExperts: Int = 2
        let topK: Int = 2
        let slidingWindow: Int = 128
        let groupSize: Int = 64
        // layer 0 = sliding, layer 1 = full
        let layerTypes: [String] = ["sliding_attention", "full_attention"]
    }

    /// Build the snapshot. `seed` controls the pseudo-random payload bytes so
    /// tests can pre-compute byte-fidelity expectations.
    static func build(at dir: String, seed: UInt64 = 0xA17B_EEF1_5FAC_E202) throws -> Snapshot {
        try? FileManager.default.removeItem(atPath: dir)
        try FileManager.default.createDirectory(atPath: dir,
                                                withIntermediateDirectories: true)

        let arch = Arch()
        var rng = SplitMix64(seed: seed)

        // Build the inventory: (name, dtype, shape, payload)
        var tensors: [(String, String, [Int], [UInt8])] = []
        tensors.reserveCapacity(64)

        // -- LM resident: embedding (4-bit, group=64)
        appendQuantizedWeight(name: "language_model.model.embed_tokens",
                              outerShape: [arch.vocab],
                              innerLogical: arch.hidden, bits: 4,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)

        for li in 0..<arch.numLayers {
            let prefix = "language_model.model.layers.\(li)"
            // q/k/v/o projections — 4-bit attention
            // SWA layer: heads*head_dim out; full: globalHeadDim heads (here equal kv configs).
            let isFull = arch.layerTypes[li] == "full_attention"
            let qOut = arch.numHeads * (isFull ? arch.globalHeadDim : arch.headDim)
            let kOut = (isFull ? arch.numGlobalKVHeads : arch.numKVHeads) * (isFull ? arch.globalHeadDim : arch.headDim)
            let vOut = kOut
            let oIn  = qOut
            appendQuantizedWeight(name: prefix + ".self_attn.q_proj",
                                  outerShape: [qOut], innerLogical: arch.hidden,
                                  bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".self_attn.k_proj",
                                  outerShape: [kOut], innerLogical: arch.hidden,
                                  bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            if !isFull {
                appendQuantizedWeight(name: prefix + ".self_attn.v_proj",
                                      outerShape: [vOut], innerLogical: arch.hidden,
                                      bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            }
            appendQuantizedWeight(name: prefix + ".self_attn.o_proj",
                                  outerShape: [arch.hidden], innerLogical: oIn,
                                  bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)

            // Per-head norms (BF16, no scale/bias)
            appendUnquantizedBF16(name: prefix + ".self_attn.q_norm.weight",
                                  shape: [isFull ? arch.globalHeadDim : arch.headDim],
                                  into: &tensors, rng: &rng)
            appendUnquantizedBF16(name: prefix + ".self_attn.k_norm.weight",
                                  shape: [isFull ? arch.globalHeadDim : arch.headDim],
                                  into: &tensors, rng: &rng)

            // Router proj — 8-bit affine
            appendQuantizedWeight(name: prefix + ".router.proj",
                                  outerShape: [arch.numExperts], innerLogical: arch.hidden,
                                  bits: 8, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendUnquantizedBF16(name: prefix + ".router.scale",
                                  shape: [arch.hidden], into: &tensors, rng: &rng)
            appendUnquantizedBF16(name: prefix + ".router.per_expert_scale",
                                  shape: [arch.numExperts], into: &tensors, rng: &rng)
            appendUnquantizedBF16(name: prefix + ".layer_scalar",
                                  shape: [1], into: &tensors, rng: &rng)

            // Shared-expert mlp — 8-bit affine
            appendQuantizedWeight(name: prefix + ".mlp.gate_proj",
                                  outerShape: [arch.intermediate], innerLogical: arch.hidden,
                                  bits: 8, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.up_proj",
                                  outerShape: [arch.intermediate], innerLogical: arch.hidden,
                                  bits: 8, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.down_proj",
                                  outerShape: [arch.hidden], innerLogical: arch.intermediate,
                                  bits: 8, groupSize: arch.groupSize, into: &tensors, rng: &rng)

            // Routed experts — 4-bit affine, leading dim = numExperts
            appendQuantizedWeight(name: prefix + ".experts.switch_glu.gate_proj",
                                  outerShape: [arch.numExperts, arch.moeIntermediate],
                                  innerLogical: arch.hidden, bits: 4,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".experts.switch_glu.up_proj",
                                  outerShape: [arch.numExperts, arch.moeIntermediate],
                                  innerLogical: arch.hidden, bits: 4,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".experts.switch_glu.down_proj",
                                  outerShape: [arch.numExperts, arch.hidden],
                                  innerLogical: arch.moeIntermediate, bits: 4,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)

            // Per-layer norms
            for norm in ["input_layernorm","post_attention_layernorm",
                         "pre_feedforward_layernorm","pre_feedforward_layernorm_2",
                         "post_feedforward_layernorm","post_feedforward_layernorm_1",
                         "post_feedforward_layernorm_2"] {
                appendUnquantizedBF16(name: prefix + "." + norm + ".weight",
                                      shape: [arch.hidden], into: &tensors, rng: &rng)
            }
        }
        // Final norm
        appendUnquantizedBF16(name: "language_model.model.norm.weight",
                              shape: [arch.hidden], into: &tensors, rng: &rng)

        // Multimodal tensors included to prove the text-only repacker drops them.
        appendUnquantizedBF16(name: "vision_tower.encoder.layers.0.input_layernorm.weight",
                              shape: [arch.hidden], into: &tensors, rng: &rng)
        appendQuantizedWeight(name: "vision_tower.encoder.layers.0.self_attn.q_proj.linear",
                              outerShape: [arch.hidden], innerLogical: arch.hidden, bits: 4,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)
        appendQuantizedWeight(name: "embed_vision.embedding_projection",
                              outerShape: [arch.hidden], innerLogical: arch.hidden, bits: 4,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)

        // -- Encode safetensors.
        let shardName = "model-00001-of-00001.safetensors"
        let shardPath = (dir as NSString).appendingPathComponent(shardName)
        try writeShard(path: shardPath, tensors: tensors)

        // -- Write config.json with bit-width overrides for mlp + router.
        var overrides: [String: [String: Any]] = [:]
        for li in 0..<arch.numLayers {
            let prefix = "language_model.model.layers.\(li)"
            for k in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "router.proj"] {
                overrides[prefix + "." + k] = ["bits": 8, "group_size": arch.groupSize]
            }
        }
        var quant: [String: Any] = [
            "bits": 4, "group_size": arch.groupSize, "mode": "affine"
        ]
        for (k, v) in overrides { quant[k] = v }

        let textConfig: [String: Any] = [
            "hidden_size": arch.hidden,
            "intermediate_size": arch.intermediate,
            "moe_intermediate_size": arch.moeIntermediate,
            "num_attention_heads": arch.numHeads,
            "num_key_value_heads": arch.numKVHeads,
            "num_global_key_value_heads": arch.numGlobalKVHeads,
            "head_dim": arch.headDim,
            "global_head_dim": arch.globalHeadDim,
            "vocab_size": arch.vocab,
            "num_hidden_layers": arch.numLayers,
            "num_experts": arch.numExperts,
            "top_k_experts": arch.topK,
            "sliding_window": arch.slidingWindow,
            "final_logit_softcapping": 30.0,
            "rope_parameters": [
                "sliding_attention": ["rope_theta": 10000.0, "rope_type": "default"],
                "full_attention":   ["rope_theta": 1000000.0, "rope_type": "proportional",
                                      "partial_rotary_factor": 0.25]
            ],
            "layer_types": arch.layerTypes,
            "tie_word_embeddings": true,
            "attention_k_eq_v": true,
            "hidden_activation": "gelu_pytorch_tanh"
        ]
        let config: [String: Any] = [
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            "quantization": quant,
            "text_config": textConfig
        ]
        let configData = try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
        try configData.write(to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent("config.json")))

        // -- Write model.safetensors.index.json.
        var weightMap: [String: String] = [:]
        for (name, _, _, _) in tensors { weightMap[name] = shardName }
        let indexObj: [String: Any] = [
            "metadata": ["format": "mlx"],
            "weight_map": weightMap
        ]
        let indexData = try JSONSerialization.data(withJSONObject: indexObj, options: [.sortedKeys])
        let indexPath = (dir as NSString).appendingPathComponent("model.safetensors.index.json")
        try indexData.write(to: URL(fileURLWithPath: indexPath))
        return Snapshot(shardPath: shardPath)
    }

    // MARK: - Tensor builders

    private static func appendQuantizedWeight(name: String,
                                              outerShape: [Int],
                                              innerLogical: Int,
                                              bits: Int,
                                              groupSize: Int,
                                              into tensors: inout [(String, String, [Int], [UInt8])],
                                              rng: inout SplitMix64) {
        precondition(innerLogical % groupSize == 0)
        let factor = 32 / bits
        precondition(innerLogical % factor == 0)
        let innerSource = innerLogical / factor
        let shape = outerShape + [innerSource]
        let elements = shape.reduce(1, *)
        var bytes = [UInt8](repeating: 0, count: elements * 4)
        for i in 0..<bytes.count { bytes[i] = UInt8(rng.next() & 0xFF) }
        tensors.append((name + ".weight", "U32", shape, bytes))

        let groups = innerLogical / groupSize
        let companionShape = outerShape + [groups]
        let companionElems = companionShape.reduce(1, *)
        var sb = [UInt8](repeating: 0, count: companionElems * 2)
        for i in 0..<sb.count { sb[i] = UInt8(rng.next() & 0xFF) }
        tensors.append((name + ".scales", "BF16", companionShape, sb))
        var bb = [UInt8](repeating: 0, count: companionElems * 2)
        for i in 0..<bb.count { bb[i] = UInt8(rng.next() & 0xFF) }
        tensors.append((name + ".biases", "BF16", companionShape, bb))
    }

    private static func appendUnquantizedBF16(name: String, shape: [Int],
                                              into tensors: inout [(String, String, [Int], [UInt8])],
                                              rng: inout SplitMix64) {
        let elements = shape.reduce(1, *)
        var bytes = [UInt8](repeating: 0, count: elements * 2)
        for i in 0..<bytes.count { bytes[i] = UInt8(rng.next() & 0xFF) }
        tensors.append((name, "BF16", shape, bytes))
    }

    // MARK: - Safetensors writer

    private static func writeShard(path: String,
                                   tensors: [(String, String, [Int], [UInt8])]) throws {
        var off: UInt64 = 0
        var headerEntries: [(String, [String: Any])] = []
        for (name, dtype, shape, bytes) in tensors {
            let begin = off
            let end = begin + UInt64(bytes.count)
            headerEntries.append((name, [
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [begin, end]
            ]))
            off = end
        }
        var headerDict: [String: Any] = [:]
        for (n, e) in headerEntries { headerDict[n] = e }
        headerDict["__metadata__"] = ["format": "mlx"]
        // Ensure deterministic key ordering for the header — JSONSerialization
        // sortedKeys handles that for us.
        let headerData = try JSONSerialization.data(withJSONObject: headerDict,
                                                    options: [.sortedKeys])
        // Pad header so payload starts on an 8-byte boundary (matches MLX
        // convention and trips fewer downstream surprises).
        var padded = headerData
        while padded.count % 8 != 0 { padded.append(0x20) } // space pad

        let fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0o644)
        precondition(fd >= 0, "open failed for \(path)")
        defer { close(fd) }
        var headerLenLE = UInt64(padded.count).littleEndian
        withUnsafeBytes(of: &headerLenLE) { raw in
            _ = write(fd, raw.baseAddress, 8)
        }
        padded.withUnsafeBytes { raw in
            _ = write(fd, raw.baseAddress, padded.count)
        }
        for (_, _, _, bytes) in tensors {
            bytes.withUnsafeBufferPointer { ptr in
                _ = write(fd, ptr.baseAddress, ptr.count)
            }
        }
    }

    // MARK: - Pinned LFM2.5 24B-A2B planning snapshot

    struct LFMSnapshot {
        let directory: String
        let shardPath: String
        let metadata: IndexLoader.SourceMetadata
        let arch: ArchInfo
        let headers: [Safetensors.Header]
    }

    private struct LFMTensor {
        let name: String
        let dtype: String
        let shape: [Int]
        let sizeBytes: UInt64
    }

    /// Writes the pinned LFM2.5 `config.json` and index, and builds a shard
    /// header with the checkpoint's exact payload shapes. The shard payload is
    /// never written: the repack planner only reads header metadata, so the
    /// 13.5 GB of source bytes stay virtual.
    static func buildLFM(at dir: String) throws -> LFMSnapshot {
        try? FileManager.default.removeItem(atPath: dir)
        try FileManager.default.createDirectory(atPath: dir,
                                                withIntermediateDirectories: true)

        let hidden = 2_048
        let vocab = 128_000
        let numLayers = 40
        let numExperts = 64
        let denseIntermediate = 11_776
        let moeIntermediate = 1_536
        let groupSize = 64
        let attentionLayers = Set(stride(from: 2, to: numLayers, by: 4))
        let layerTypes = (0..<numLayers).map {
            attentionLayers.contains($0) ? "full_attention" : "conv"
        }

        var tensors: [LFMTensor] = []
        func quantized(_ name: String,
                       outerShape: [Int],
                       logicalColumns: Int,
                       bits: Int) {
            precondition(logicalColumns * bits % 32 == 0)
            let packedColumns = logicalColumns * bits / 32
            let weightShape = outerShape + [packedColumns]
            tensors.append(LFMTensor(
                name: name + ".weight", dtype: "U32", shape: weightShape,
                sizeBytes: UInt64(weightShape.reduce(1, *)) * 4))
            let companionShape = outerShape + [logicalColumns / groupSize]
            let companionBytes = UInt64(companionShape.reduce(1, *)) * 2
            tensors.append(LFMTensor(
                name: name + ".scales", dtype: "BF16", shape: companionShape,
                sizeBytes: companionBytes))
            tensors.append(LFMTensor(
                name: name + ".biases", dtype: "BF16", shape: companionShape,
                sizeBytes: companionBytes))
        }
        func plain(_ name: String, dtype: String, shape: [Int],
                   elementBytes: Int) {
            tensors.append(LFMTensor(
                name: name, dtype: dtype, shape: shape,
                sizeBytes: UInt64(shape.reduce(1, *)) * UInt64(elementBytes)))
        }

        quantized("model.embed_tokens", outerShape: [vocab],
                  logicalColumns: hidden, bits: 6)
        plain("model.embedding_norm.weight", dtype: "BF16",
              shape: [hidden], elementBytes: 2)

        for layer in 0..<numLayers {
            let prefix = "model.layers.\(layer)"
            plain(prefix + ".operator_norm.weight", dtype: "BF16",
                  shape: [hidden], elementBytes: 2)
            plain(prefix + ".ffn_norm.weight", dtype: "BF16",
                  shape: [hidden], elementBytes: 2)
            if attentionLayers.contains(layer) {
                for (role, rows) in [("q_proj", hidden), ("k_proj", 512),
                                     ("v_proj", 512), ("out_proj", hidden)] {
                    quantized(prefix + ".self_attn." + role,
                              outerShape: [rows],
                              logicalColumns: hidden, bits: 4)
                }
                plain(prefix + ".self_attn.q_layernorm.weight", dtype: "BF16",
                      shape: [64], elementBytes: 2)
                plain(prefix + ".self_attn.k_layernorm.weight", dtype: "BF16",
                      shape: [64], elementBytes: 2)
            } else {
                quantized(prefix + ".conv.in_proj",
                          outerShape: [3 * hidden],
                          logicalColumns: hidden, bits: 4)
                quantized(prefix + ".conv.out_proj",
                          outerShape: [hidden],
                          logicalColumns: hidden, bits: 4)
                plain(prefix + ".conv.conv.weight", dtype: "BF16",
                      shape: [hidden, 3, 1], elementBytes: 2)
            }

            if layer < 2 {
                quantized(prefix + ".feed_forward.gate_proj",
                          outerShape: [denseIntermediate],
                          logicalColumns: hidden, bits: 4)
                quantized(prefix + ".feed_forward.up_proj",
                          outerShape: [denseIntermediate],
                          logicalColumns: hidden, bits: 4)
                quantized(prefix + ".feed_forward.down_proj",
                          outerShape: [hidden],
                          logicalColumns: denseIntermediate, bits: 4)
            } else {
                plain(prefix + ".feed_forward.gate.weight", dtype: "F32",
                      shape: [numExperts, hidden], elementBytes: 4)
                plain(prefix + ".feed_forward.expert_bias", dtype: "F32",
                      shape: [numExperts], elementBytes: 4)
                for (role, rows, columns) in [
                    ("gate_proj", moeIntermediate, hidden),
                    ("up_proj", moeIntermediate, hidden),
                    ("down_proj", hidden, moeIntermediate),
                ] {
                    quantized(prefix + ".feed_forward.switch_mlp." + role,
                              outerShape: [numExperts, rows],
                              logicalColumns: columns, bits: 4)
                }
            }
        }

        // Header offsets are virtual; no payload bytes are written.
        var header: [String: Any] = [:]
        var cursor: UInt64 = 0
        for tensor in tensors {
            let begin = cursor
            let end = begin + tensor.sizeBytes
            header[tensor.name] = [
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [begin, end],
            ]
            cursor = end
        }
        header["__metadata__"] = ["format": "mlx"]
        let headerData = try JSONSerialization.data(withJSONObject: header,
                                                    options: [.sortedKeys])
        var paddedHeader = headerData
        while paddedHeader.count % 8 != 0 { paddedHeader.append(0x20) }
        guard paddedHeader.count <= Int(Safetensors.maxHeaderBytes) else {
            throw RepackError.configurationInvalid(
                detail: "synthetic LFM header exceeds the safetensors cap")
        }

        let shardName = "model-00001-of-00001.safetensors"
        let shardPath = (dir as NSString).appendingPathComponent(shardName)
        let virtualFileSize = UInt64(8 + paddedHeader.count) + cursor
        let parsedHeader = try Safetensors.parseHeaderBytes(
            path: shardPath, fileSize: virtualFileSize,
            headerBytes: paddedHeader)

        let config: [String: Any] = [
            "architectures": ["Lfm2MoeForCausalLM"],
            "model_type": "lfm2_moe",
            "hidden_size": hidden,
            "intermediate_size": denseIntermediate,
            "moe_intermediate_size": moeIntermediate,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": vocab,
            "max_position_embeddings": 131_072,
            "num_hidden_layers": numLayers,
            "num_experts": numExperts,
            "num_experts_per_tok": 4,
            "num_dense_layers": 2,
            "layer_types": layerTypes,
            "conv_L_cache": 3,
            "norm_eps": 1e-5,
            "rope_parameters": ["rope_theta": 10_000_000.0],
            "tie_word_embeddings": true,
            "norm_topk_prob": true,
            "use_expert_bias": true,
            "routed_scaling_factor": 1.0,
            "quantization": [
                "bits": 4,
                "group_size": groupSize,
                "mode": "affine",
                "model.embed_tokens": ["bits": 6, "group_size": groupSize],
            ],
        ]
        let configData = try JSONSerialization.data(
            withJSONObject: config, options: [.sortedKeys])
        let configPath = (dir as NSString).appendingPathComponent("config.json")
        try configData.write(to: URL(fileURLWithPath: configPath))

        var weightMap: [String: String] = [:]
        for tensor in tensors { weightMap[tensor.name] = shardName }
        let index: [String: Any] = [
            "metadata": ["format": "mlx"],
            "weight_map": weightMap,
        ]
        let indexData = try JSONSerialization.data(
            withJSONObject: index, options: [.sortedKeys])
        let indexPath = (dir as NSString)
            .appendingPathComponent("model.safetensors.index.json")
        try indexData.write(to: URL(fileURLWithPath: indexPath))

        let metadata = try IndexLoader.load(snapshotDir: dir)
        let arch = try ArchInfo.load(configPath: configPath)
        return LFMSnapshot(directory: dir,
                           shardPath: shardPath,
                           metadata: metadata,
                           arch: arch,
                           headers: [parsedHeader])
    }
}

/// Tiny deterministic PRNG. We do not need crypto quality — just stable
/// byte streams across test runs.
struct SplitMix64 {
    private var state: UInt64
    init(seed: UInt64) { self.state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
