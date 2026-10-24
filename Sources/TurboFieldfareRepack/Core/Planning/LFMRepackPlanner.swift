import Foundation
import TurboFieldfareFormat

/// Plans the LFM2.5 MoE resident file and its MoE-only expert layer files.
///
/// The source checkpoint names every layer tensor under `model.layers.N.*`.
/// The installed pack strips that prefix from the layer tensors because the
/// runtime's `LFM2Weights` view resolves `layers.N.*`; the embedding and
/// `embedding_norm` keep their `model.` prefix.
enum LFMRepackPlanner {

    /// One resident slot before its companion tensors are expanded.
    private enum ResidentItem {
        /// `sourceBase` and `installedBase` omit the trailing `.weight`.
        case quantized(sourceBase: String, installedBase: String)
        case plain(source: String, installed: String,
                   expectedDtype: SourceTensor.Dtype)
    }

    /// A resident entry whose source tensors have been resolved.
    private struct ResolvedEntry {
        let name: String
        let dtype: UInt8
        let logicalShape4: [UInt32]
        let quantSpec: QuantSpec?
        let weight: SourceTensor
        let scales: SourceTensor?
        let biases: SourceTensor?
    }

    static func plan(meta: IndexLoader.SourceMetadata,
                     arch: ArchInfo,
                     registry: [String: SourceTensor],
                     outputDir: String,
                     bitsOverrideCount: Int) throws -> RepackPlan {
        guard arch.family == .lfm2Moe else {
            throw RepackError.configurationInvalid(
                detail: "LFM planner requires the lfm2_moe family")
        }
        guard arch.numLayers > 0, arch.numExperts > 0 else {
            throw RepackError.configurationInvalid(
                detail: "LFM planner requires positive layer and expert counts")
        }
        guard let layerTypes = arch.layerTypes,
              layerTypes.count == arch.numLayers else {
            throw RepackError.configurationInvalid(
                detail: "LFM planner requires one layer type per layer")
        }
        guard let moeLayerMask = arch.moeLayerMask,
              moeLayerMask.count == arch.numLayers,
              moeLayerMask.allSatisfy({ $0 == 0 || $0 == 1 }) else {
            throw RepackError.configurationInvalid(
                detail: "LFM planner requires a valid MoE layer mask")
        }

        let items = residentItems(layerTypes: layerTypes,
                                  moeLayerMask: moeLayerMask)
        let resolved = try resolve(items, registry: registry, meta: meta)

        let residentPath = (outputDir as NSString)
            .appendingPathComponent("model_weights.bin")
        let resident = try planResidentFile(path: residentPath, entries: resolved)

        let layersDir = (outputDir as NSString)
            .appendingPathComponent("packed_experts")
        var layerPlans: [LayerFilePlan] = []
        layerPlans.reserveCapacity(moeLayerMask.filter { $0 == 1 }.count)
        for layer in 0..<arch.numLayers where moeLayerMask[layer] == 1 {
            let prefix = "model.layers.\(layer).feed_forward.switch_mlp"
            let path = (layersDir as NSString)
                .appendingPathComponent("layer_\(String(format: "%02d", layer)).bin")
            let plan = try RepackPlanner.planLayerFile(
                path: path, layer: layer,
                gateName: prefix + ".gate_proj.weight",
                upName: prefix + ".up_proj.weight",
                downName: prefix + ".down_proj.weight",
                registry: registry, meta: meta, arch: arch)
            guard plan.expertStride > 0,
                  plan.expertStride % Layout.pageBytes == 0 else {
                throw RepackError.configurationInvalid(
                    detail: "layer \(layer) expert stride \(plan.expertStride) "
                        + "is not page-aligned")
            }
            if let first = layerPlans.first, first.expertStride != plan.expertStride {
                throw RepackError.configurationInvalid(
                    detail: "layer \(layer) expert stride \(plan.expertStride) "
                        + "differs from layer \(first.layerIndex) "
                        + "stride \(first.expertStride)")
            }
            layerPlans.append(plan)
        }

        try rejectUnexpectedTensors(expecting: resolved, layers: layerPlans,
                                    registry: registry)

        return RepackPlan(
            arch: arch,
            baseMode: meta.baseMode,
            baseGroupSize: meta.baseGroupSize,
            bitsOverrideCount: bitsOverrideCount,
            resident: resident,
            layers: layerPlans,
            matchedModelID: SourceFingerprint.modelID(
                forIndexSha256: meta.indexSha256Hex),
            excludedMultimodalTensorNames: [])
    }

    // MARK: - Resident inventory

    /// The fixed resident order from design §9.3: embedding, the final norm,
    /// per-layer norms, convolution tensors, attention tensors, dense FFNs,
    /// then the MoE router and expert bias.
    private static func residentItems(layerTypes: [LayerType],
                                      moeLayerMask: [UInt8]) -> [ResidentItem] {
        var items: [ResidentItem] = [
            .quantized(sourceBase: "model.embed_tokens",
                       installedBase: "model.embed_tokens"),
            .plain(source: "model.embedding_norm.weight",
                   installed: "model.embedding_norm.weight",
                   expectedDtype: .bf16),
        ]

        for layer in layerTypes.indices {
            let source = "model.layers.\(layer)"
            let installed = "layers.\(layer)"
            items.append(.plain(source: source + ".operator_norm.weight",
                                installed: installed + ".operator_norm.weight",
                                expectedDtype: .bf16))
            items.append(.plain(source: source + ".ffn_norm.weight",
                                installed: installed + ".ffn_norm.weight",
                                expectedDtype: .bf16))
        }

        for layer in layerTypes.indices where layerTypes[layer] == .conv {
            let source = "model.layers.\(layer).conv"
            let installed = "layers.\(layer).conv"
            items.append(.quantized(sourceBase: source + ".in_proj",
                                    installedBase: installed + ".in_proj"))
            items.append(.quantized(sourceBase: source + ".out_proj",
                                    installedBase: installed + ".out_proj"))
            items.append(.plain(source: source + ".conv.weight",
                                installed: installed + ".conv.weight",
                                expectedDtype: .bf16))
        }

        for layer in layerTypes.indices where layerTypes[layer] != .conv {
            let source = "model.layers.\(layer).self_attn"
            let installed = "layers.\(layer).self_attn"
            for (sourceName, installedName) in [
                ("q_proj", "q_proj"),
                ("k_proj", "k_proj"),
                ("v_proj", "v_proj"),
                // The checkpoint spells the attention output `out_proj`; the
                // installed runtime resolves `o_proj`.
                ("out_proj", "o_proj"),
            ] {
                items.append(.quantized(
                    sourceBase: source + "." + sourceName,
                    installedBase: installed + "." + installedName))
            }
            items.append(.plain(source: source + ".q_layernorm.weight",
                                installed: installed + ".q_layernorm.weight",
                                expectedDtype: .bf16))
            items.append(.plain(source: source + ".k_layernorm.weight",
                                installed: installed + ".k_layernorm.weight",
                                expectedDtype: .bf16))
        }

        for layer in layerTypes.indices {
            let source = "model.layers.\(layer).feed_forward"
            let installed = "layers.\(layer).feed_forward"
            if moeLayerMask[layer] == 0 {
                for role in ["gate_proj", "up_proj", "down_proj"] {
                    items.append(.quantized(
                        sourceBase: source + "." + role,
                        installedBase: installed + "." + role))
                }
            } else {
                items.append(.plain(source: source + ".gate.weight",
                                    installed: installed + ".gate.weight",
                                    expectedDtype: .fp32))
                items.append(.plain(source: source + ".expert_bias",
                                    installed: installed + ".expert_bias",
                                    expectedDtype: .fp32))
            }
        }
        return items
    }

    private static func resolve(_ items: [ResidentItem],
                                registry: [String: SourceTensor],
                                meta: IndexLoader.SourceMetadata) throws
                                -> [ResolvedEntry] {
        var resolved: [ResolvedEntry] = []
        for item in items {
            switch item {
            case .quantized(let sourceBase, let installedBase):
                let weightName = sourceBase + ".weight"
                guard let weight = registry[weightName] else {
                    throw RepackError.missingTensor(name: weightName)
                }
                guard weight.dtype == .u32 else {
                    throw RepackError.dtypeMismatch(
                        name: weightName,
                        detail: "expected U32 quantized weight, got \(weight.dtype)")
                }
                let scalesName = sourceBase + ".scales"
                let biasesName = sourceBase + ".biases"
                guard let scales = registry[scalesName] else {
                    throw RepackError.missingScalesCompanion(name: weightName)
                }
                guard let biases = registry[biasesName] else {
                    throw RepackError.missingBiasesCompanion(name: weightName)
                }
                guard scales.dtype == .bf16, biases.dtype == .bf16 else {
                    throw RepackError.dtypeMismatch(
                        name: weightName,
                        detail: "expected BF16 scales/biases, got "
                            + "\(scales.dtype)/\(biases.dtype)")
                }
                let spec = IndexLoader.quantSpec(forTensor: weightName, meta: meta)
                let logical = try RepackPlanner.logicalShape(
                    forPackedSource: weight.shape, bits: spec.bits, name: weightName)
                resolved.append(ResolvedEntry(
                    name: installedBase + ".weight",
                    dtype: RepackPlanner.ietnyDtype(.u32),
                    logicalShape4: try logicalShape4(logical, name: weightName),
                    quantSpec: spec,
                    weight: weight,
                    scales: scales,
                    biases: biases))

            case .plain(let source, let installed, let expectedDtype):
                guard let tensor = registry[source] else {
                    throw RepackError.missingTensor(name: source)
                }
                guard tensor.dtype == expectedDtype else {
                    throw RepackError.dtypeMismatch(
                        name: source,
                        detail: "expected \(expectedDtype), got \(tensor.dtype)")
                }
                resolved.append(ResolvedEntry(
                    name: installed,
                    dtype: RepackPlanner.ietnyDtype(tensor.dtype),
                    logicalShape4: try logicalShape4(tensor.shape, name: source),
                    quantSpec: nil,
                    weight: tensor,
                    scales: nil,
                    biases: nil))
            }
        }
        return resolved
    }

    /// Every source tensor must be consumed by the LFM inventory. Anything
    /// else — a stray `lm_head`, a dense layer's `switch_mlp`, a tensor on a
    /// layer type that does not own it — fails closed instead of being
    /// silently dropped.
    private static func rejectUnexpectedTensors(
        expecting resolved: [ResolvedEntry],
        layers: [LayerFilePlan],
        registry: [String: SourceTensor]
    ) throws {
        var expected = Set<String>()
        for entry in resolved {
            expected.insert(entry.weight.name)
            if let scales = entry.scales { expected.insert(scales.name) }
            if let biases = entry.biases { expected.insert(biases.name) }
        }
        for layer in layers {
            for slice in layer.subTensors {
                expected.insert(slice.sourceTensor.name)
            }
        }
        if let unexpected = registry.keys.sorted().first(where: {
            !expected.contains($0)
        }) {
            throw RepackError.unknownTensorPrefix(name: unexpected)
        }
    }

    private static func logicalShape4(_ shape: [UInt64],
                                      name: String) throws -> [UInt32] {
        guard shape.count <= 4 else {
            throw RepackError.shapeMismatch(
                name: name,
                detail: "rank \(shape.count) exceeds the resident index's four dimensions")
        }
        guard shape.allSatisfy({ $0 <= UInt64(UInt32.max) }) else {
            throw RepackError.shapeMismatch(
                name: name,
                detail: "shape \(shape) does not fit the resident index's UInt32 extents")
        }
        return RepackPlanner.padTo4(shape)
    }

    // MARK: - Resident file encoding

    private static func planResidentFile(path: String,
                                         entries: [ResolvedEntry]) throws
                                        -> ResidentFilePlan {
        var stringTable: [UInt8] = []
        var offsets: [UInt32] = []
        offsets.reserveCapacity(entries.count)
        for entry in entries {
            guard stringTable.count <= Int(UInt32.max) else {
                throw RepackError.configurationInvalid(
                    detail: "LFM resident string table exceeds UInt32 offsets")
            }
            offsets.append(UInt32(stringTable.count))
            stringTable.append(contentsOf: entry.name.utf8)
        }

        let headerBytes = UInt64(GTurboBinary.indexHeaderBytes)
        let entryBytes = try checkedMultiply(
            UInt64(entries.count), UInt64(GTurboBinary.indexEntryBytes),
            detail: "LFM resident index entry table")
        let rawIdx = try checkedAdd(
            try checkedAdd(headerBytes, entryBytes,
                           detail: "LFM resident index size"),
            UInt64(stringTable.count),
            detail: "LFM resident index size")
        let indexSize = RepackPlanner.roundUpToPage(rawIdx)

        var cursor = indexSize
        var residentEntries: [ResidentEntry] = []
        residentEntries.reserveCapacity(entries.count)
        for entry in entries {
            let weightOffset = cursor
            let weightSize = entry.weight.sizeBytes
            let scaleOffset = try checkedAdd(
                weightOffset, weightSize,
                detail: "LFM resident scales offset")
            let scaleSize = entry.scales?.sizeBytes ?? 0
            let biasOffset = try checkedAdd(
                scaleOffset, scaleSize,
                detail: "LFM resident biases offset")
            let biasSize = entry.biases?.sizeBytes ?? 0
            cursor = try checkedAdd(
                biasOffset, biasSize,
                detail: "LFM resident payload size")

            residentEntries.append(ResidentEntry(
                name: entry.name,
                dtype: entry.dtype,
                logicalShape4: entry.logicalShape4,
                fileOffset: weightOffset,
                sizeBytes: weightSize,
                scaleOffset: entry.scales == nil ? 0 : scaleOffset,
                scaleSize: scaleSize,
                biasOffset: entry.biases == nil ? 0 : biasOffset,
                biasSize: biasSize,
                quantSpec: entry.quantSpec,
                sourceWeight: entry.weight,
                sourceScales: entry.scales,
                sourceBiases: entry.biases))
        }

        let residentSize = cursor - indexSize
        return ResidentFilePlan(path: path,
                                entries: residentEntries,
                                stringTable: stringTable,
                                stringTableOffsets: offsets,
                                indexSize: indexSize,
                                residentSize: residentSize)
    }

    private static func checkedAdd(_ lhs: UInt64, _ rhs: UInt64,
                                   detail: String) throws -> UInt64 {
        let (value, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow else {
            throw RepackError.configurationInvalid(
                detail: "\(detail) overflows UInt64")
        }
        return value
    }

    private static func checkedMultiply(_ lhs: UInt64, _ rhs: UInt64,
                                        detail: String) throws -> UInt64 {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw RepackError.configurationInvalid(
                detail: "\(detail) overflows UInt64")
        }
        return value
    }
}
