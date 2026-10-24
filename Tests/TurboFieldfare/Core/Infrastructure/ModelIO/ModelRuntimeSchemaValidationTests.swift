import Foundation
import Metal
import Testing

@testable import TurboFieldfare
@testable import TurboFieldfareFormat

@Suite struct ModelRuntimeSchemaValidationTests {
    @Test func rejectsMissingRequiredResidentTensor() throws {
        let model = try loadToy()
        var entries = model.residentIndex.entries
        entries["language_model.model.embed_tokens.weight"] = nil
        let index = ResidentIndex(header: model.residentIndex.header, entries: entries)

        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: index,
                layout: model.packedExpertsLayout,
                manifest: model.manifest,
                config: .gemma4Toy())
        }
    }

    @Test func rejectsMissingRoutedTensorRole() throws {
        let model = try loadToy()
        let layout = replacingExpert(model.packedExpertsLayout, layer: 0, expert: 0) { expert in
            var tensors = expert.subTensors
            tensors["gate"] = nil
            return ExpertEntry(
                expert: expert.expert,
                physicalRank: expert.physicalRank,
                offset: expert.offset,
                size: expert.size,
                subTensors: tensors)
        }

        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: model.residentIndex,
                layout: layout,
                manifest: model.manifest,
                config: .gemma4Toy())
        }
    }

    @Test func rejectsNonuniformRoutedTensorMetadata() throws {
        let model = try loadToy()
        let layout = replacingExpert(model.packedExpertsLayout, layer: 0, expert: 1) { expert in
            var tensors = expert.subTensors
            let gate = tensors["gate"]!
            tensors["gate"] = SubTensorEntry(
                offset: gate.offset + 1,
                size: gate.size,
                dtype: gate.dtype,
                shape: gate.shape,
                bits: gate.bits)
            return ExpertEntry(
                expert: expert.expert,
                physicalRank: expert.physicalRank,
                offset: expert.offset,
                size: expert.size,
                subTensors: tensors)
        }

        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: model.residentIndex,
                layout: layout,
                manifest: model.manifest,
                config: .gemma4Toy())
        }
    }

    @Test func rejectsRoutedTensorRangeOutsideUInt32() throws {
        let model = try loadToy()
        let layout = replacingExpert(model.packedExpertsLayout, layer: 0, expert: 0) { expert in
            var tensors = expert.subTensors
            let gate = tensors["gate"]!
            tensors["gate"] = SubTensorEntry(
                offset: UInt64(UInt32.max),
                size: 2,
                dtype: gate.dtype,
                shape: gate.shape,
                bits: gate.bits)
            return ExpertEntry(
                expert: expert.expert,
                physicalRank: expert.physicalRank,
                offset: expert.offset,
                size: expert.size,
                subTensors: tensors)
        }

        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: model.residentIndex,
                layout: layout,
                manifest: model.manifest,
                config: .gemma4Toy())
        }
    }

    @Test func rejectsMalformedResidentAffineMetadata() throws {
        let model = try loadToy()
        let name = "language_model.model.embed_tokens.weight"
        let entry = try #require(model.residentIndex.entries[name])
        let invalidEntries = [
            copy(entry, dtype: 1),
            copy(entry, shape: (entry.shape.0, entry.shape.1 + 1, 0, 0)),
            copy(entry, sizeBytes: entry.sizeBytes + 1),
            copy(entry, scaleSize: entry.scaleSize + 2),
            copy(entry, fileOffset: entry.fileOffset + 1),
            copy(entry, scaleOffset: entry.scaleOffset + 1),
            copy(entry, biasSize: entry.biasSize + 2),
            copy(entry, biasOffset: entry.biasOffset + 1),
        ]

        for invalid in invalidEntries {
            var entries = model.residentIndex.entries
            entries[name] = invalid
            let index = ResidentIndex(header: model.residentIndex.header, entries: entries)
            #expect(throws: ModelError.self) {
                try Model.validateRuntimeSchema(
                    residentIndex: index,
                    layout: model.packedExpertsLayout,
                    manifest: model.manifest,
                    config: .gemma4Toy())
            }
        }
    }

    @Test func rejectsMalformedRoutedAffineMetadata() throws {
        let model = try loadToy()
        let gate = try #require(model.packedExpertsLayout.layers[0].experts[0].subTensors["gate"])
        let invalidEntries = [
            SubTensorEntry(offset: gate.offset, size: gate.size, dtype: "BF16",
                           shape: gate.shape, bits: gate.bits),
            SubTensorEntry(offset: gate.offset, size: gate.size, dtype: gate.dtype,
                           shape: [gate.shape[0], gate.shape[1] + 1], bits: gate.bits),
            SubTensorEntry(offset: gate.offset, size: gate.size, dtype: gate.dtype,
                           shape: gate.shape, bits: 8),
            SubTensorEntry(offset: gate.offset, size: gate.size + 1, dtype: gate.dtype,
                           shape: gate.shape, bits: gate.bits),
            SubTensorEntry(offset: gate.offset + 1, size: gate.size, dtype: gate.dtype,
                           shape: gate.shape, bits: gate.bits),
        ]

        for invalid in invalidEntries {
            let layout = replacingRoleForAllExperts(
                model.packedExpertsLayout, layer: 0, role: "gate", with: invalid)
            #expect(throws: ModelError.self) {
                try Model.validateRuntimeSchema(
                    residentIndex: model.residentIndex,
                    layout: layout,
                    manifest: model.manifest,
                    config: .gemma4Toy())
            }
        }
    }

    @Test func rejectsExecutableLoadWithoutQuantMetadata() throws {
        let directory = try ModelLoaderTests.writeToySynthetic(includeQuant: false)
        let device = try #require(MTLCreateSystemDefaultDevice())
        #expect(throws: ModelError.self) {
            try Model.load(directoryURL: directory, device: device, expecting: .gemma4Toy())
        }
    }

    @Test func rejectsOverflowingDerivedDimensions() throws {
        let model = try loadToy()
        let toy = ArchConfig.gemma4Toy()
        let config = ArchConfig(
            hiddenSize: toy.hiddenSize,
            intermediateSize: toy.intermediateSize,
            moeIntermediateSize: toy.moeIntermediateSize,
            numHeads: Int.max,
            numKVHeads: toy.numKVHeads,
            numFullKVHeads: toy.numFullKVHeads,
            headDim: 2,
            fullHeadDim: toy.fullHeadDim,
            vocabSize: toy.vocabSize,
            slidingWindow: toy.slidingWindow,
            finalLogitSoftcap: toy.finalLogitSoftcap,
            ropeTheta: toy.ropeTheta,
            fullRopeTheta: toy.fullRopeTheta,
            partialRotaryFactor: toy.partialRotaryFactor,
            numLayers: toy.numLayers,
            numExperts: toy.numExperts,
            topKExperts: toy.topKExperts,
            tieWordEmbeddings: toy.tieWordEmbeddings,
            attentionKEqV: toy.attentionKEqV,
            fullAttentionLayerMask: toy.fullAttentionLayerMask,
            hiddenActivation: toy.hiddenActivation)
        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: model.residentIndex,
                layout: model.packedExpertsLayout,
                manifest: model.manifest,
                config: config)
        }
    }

    @Test func lfmSchemaAcceptsPinnedTensorSet() throws {
        let model = try loadSyntheticLFM()
        #expect(throws: Never.self) {
            try Model.validateRuntimeSchema(
                residentIndex: model.residentIndex,
                layout: model.packedExpertsLayout,
                manifest: model.manifest,
                config: .lfm2_5_24B_A2B)
        }
    }

    @Test func lfmSchemaRejectsMissingConvWeight() throws {
        let model = try loadSyntheticLFM()
        var entries = model.residentIndex.entries
        entries["layers.0.conv.conv.weight"] = nil
        let index = ResidentIndex(header: model.residentIndex.header, entries: entries)
        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: index,
                layout: model.packedExpertsLayout,
                manifest: model.manifest,
                config: .lfm2_5_24B_A2B)
        }
    }

    @Test func lfmSchemaRejectsExpertFileOnDenseLayer() throws {
        let model = try loadSyntheticLFM()
        let layout = addingExpertLayer(to: model.packedExpertsLayout, layer: 0)
        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: model.residentIndex,
                layout: layout,
                manifest: model.manifest,
                config: .lfm2_5_24B_A2B)
        }
    }

    @Test func loadResolvingFamilySelectsLFMNotGemma() throws {
        // An LFM-family manifest carrying Gemma constants must fail at arch
        // validation, which proves the resolver chose the LFM canonical config
        // instead of silently falling back to Gemma.
        let directory = try writeLFMManifestWithGemmaConstants()
        defer { try? FileManager.default.removeItem(at: directory) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        #expect(throws: ModelError.self) {
            _ = try Model.loadResolvingFamily(directoryURL: directory,
                                              device: device,
                                              streamingMode: .pread(slotCount: 8))
        }
        // LFM's baseline is hiddenSize 2048; a Gemma fallback would reject a
        // different field first, so pinning the field proves which config ran.
        do {
            _ = try Model.loadResolvingFamily(directoryURL: directory,
                                              device: device,
                                              streamingMode: .pread(slotCount: 8))
            Issue.record("LFM manifest with Gemma constants must fail arch validation")
        } catch let error as ModelError {
            guard case .archMismatch(let field, let expected, let actual) = error else {
                Issue.record("unexpected error \(error)")
                return
            }
            #expect(field == "hiddenSize")
            #expect(expected == "2048")
            #expect(actual == "2816")
        }
    }

    private func loadToy() throws -> Model {
        let directory = try ModelLoaderTests.writeToySynthetic()
        let device = try #require(MTLCreateSystemDefaultDevice())
        return try Model.load(
            directoryURL: directory,
            device: device,
            expecting: .gemma4Toy())
    }

    private func replacingExpert(
        _ layout: PackedExpertsLayout,
        layer layerIndex: Int,
        expert expertIndex: Int,
        transform: (ExpertEntry) -> ExpertEntry
    ) -> PackedExpertsLayout {
        var layers = layout.layers
        let layer = layers[layerIndex]
        var experts = layer.experts
        experts[expertIndex] = transform(experts[expertIndex])
        layers[layerIndex] = LayerLayout(
            layer: layer.layer,
            file: layer.file,
            experts: experts)
        return PackedExpertsLayout(
            expertStride: layout.expertStride,
            numLayers: layout.numLayers,
            expertsPerLayer: layout.expertsPerLayer,
            layers: layers)
    }

    private func replacingRoleForAllExperts(
        _ layout: PackedExpertsLayout,
        layer layerIndex: Int,
        role: String,
        with replacement: SubTensorEntry
    ) -> PackedExpertsLayout {
        var layers = layout.layers
        let layer = layers[layerIndex]
        let experts = layer.experts.map { expert in
            var tensors = expert.subTensors
            tensors[role] = replacement
            return ExpertEntry(
                expert: expert.expert,
                physicalRank: expert.physicalRank,
                offset: expert.offset,
                size: expert.size,
                subTensors: tensors)
        }
        layers[layerIndex] = LayerLayout(layer: layer.layer, file: layer.file, experts: experts)
        return PackedExpertsLayout(
            expertStride: layout.expertStride,
            numLayers: layout.numLayers,
            expertsPerLayer: layout.expertsPerLayer,
            layers: layers)
    }

    private func copying(
        _ layout: PackedExpertsLayout,
        layers: [LayerLayout]
    ) -> PackedExpertsLayout {
        PackedExpertsLayout(
            expertStride: layout.expertStride,
            numLayers: layout.numLayers,
            expertsPerLayer: layout.expertsPerLayer,
            layers: layers,
            moeLayerMask: layout.moeLayerMask)
    }

    /// Adds a packed-expert file for a layer that the mask marks as dense.
    private func addingExpertLayer(
        to layout: PackedExpertsLayout,
        layer layerIndex: Int
    ) -> PackedExpertsLayout {
        let donorExperts = layout.layers.first?.experts ?? []
        var layers = layout.layers
        layers.insert(
            LayerLayout(layer: layerIndex,
                        file: String(format: "layer_%02d.bin", layerIndex),
                        experts: donorExperts),
            at: 0)
        return copying(layout, layers: layers)
    }

    private func copy(
        _ entry: ResidentIndexEntry,
        dtype: UInt8? = nil,
        fileOffset: UInt64? = nil,
        sizeBytes: UInt64? = nil,
        shape: (UInt32, UInt32, UInt32, UInt32)? = nil,
        scaleOffset: UInt64? = nil,
        scaleSize: UInt64? = nil,
        biasOffset: UInt64? = nil,
        biasSize: UInt64? = nil
    ) -> ResidentIndexEntry {
        ResidentIndexEntry(
            name: entry.name,
            dtype: dtype ?? entry.dtype,
            fileOffset: fileOffset ?? entry.fileOffset,
            sizeBytes: sizeBytes ?? entry.sizeBytes,
            shape: shape ?? entry.shape,
            scaleOffset: scaleOffset ?? entry.scaleOffset,
            scaleSize: scaleSize ?? entry.scaleSize,
            biasOffset: biasOffset ?? entry.biasOffset,
            biasSize: biasSize ?? entry.biasSize)
    }
}

/// Metadata-only LFM fixture: real installed names and pinned shapes, with no
/// 657 MB resident payload and no packed-expert files.
struct SyntheticLFM {
    let residentIndex: ResidentIndex
    let packedExpertsLayout: PackedExpertsLayout
    let manifest: Manifest
}

func loadSyntheticLFM() throws -> SyntheticLFM {
    let config = ArchConfig.lfm2_5_24B_A2B
    let manifest = try ManifestReader.decode(
        data: GTurboManifestCodec.encode(makeSyntheticLFMWire()),
        expecting: config)

    var cursor: UInt64 = 16_384
    var entries: [String: ResidentIndexEntry] = [:]

    func addEntry(name: String,
                  dtype: UInt8,
                  shape: (UInt32, UInt32, UInt32, UInt32),
                  weightBytes: UInt64,
                  auxBytes: UInt64) {
        let weightOffset = cursor
        let scaleOffset = auxBytes == 0 ? 0 : weightOffset + weightBytes
        let biasOffset = auxBytes == 0 ? 0 : scaleOffset + auxBytes
        cursor = auxBytes == 0 ? weightOffset + weightBytes : biasOffset + auxBytes
        entries[name] = ResidentIndexEntry(
            name: name,
            dtype: dtype,
            fileOffset: weightOffset,
            sizeBytes: weightBytes,
            shape: shape,
            scaleOffset: scaleOffset,
            scaleSize: auxBytes,
            biasOffset: biasOffset,
            biasSize: auxBytes)
    }

    func addBF16(_ name: String,
                 shape: (UInt32, UInt32, UInt32, UInt32),
                 count: Int) {
        addEntry(name: name, dtype: 1, shape: shape,
                 weightBytes: UInt64(count) * UInt64(MemoryLayout<UInt16>.size),
                 auxBytes: 0)
    }

    func addFP32(_ name: String,
                 shape: (UInt32, UInt32, UInt32, UInt32),
                 count: Int) {
        addEntry(name: name, dtype: 3, shape: shape,
                 weightBytes: UInt64(count) * UInt64(MemoryLayout<Float>.size),
                 auxBytes: 0)
    }

    func addAffine(_ name: String, rows: Int, columns: Int, bits: Int) throws {
        let sizes = try Quantization.affineSizes(
            rows: rows, columns: columns, bits: bits,
            groupSize: Quantization.groupSize)
        addEntry(name: name, dtype: 0,
                 shape: (UInt32(rows), UInt32(columns), 0, 0),
                 weightBytes: sizes.weightBytes, auxBytes: sizes.auxBytes)
    }

    try addAffine("model.embed_tokens.weight",
                  rows: config.vocabSize, columns: config.hiddenSize, bits: 6)
    addBF16("model.embedding_norm.weight",
            shape: (UInt32(config.hiddenSize), 0, 0, 0),
            count: config.hiddenSize)

    for layer in 0..<config.numLayers {
        let prefix = "layers.\(layer)"
        addBF16("\(prefix).operator_norm.weight",
                shape: (UInt32(config.hiddenSize), 0, 0, 0),
                count: config.hiddenSize)
        addBF16("\(prefix).ffn_norm.weight",
                shape: (UInt32(config.hiddenSize), 0, 0, 0),
                count: config.hiddenSize)

        switch config.layerTypes[layer] {
        case .conv:
            try addAffine("\(prefix).conv.in_proj.weight",
                          rows: 3 * config.hiddenSize,
                          columns: config.hiddenSize, bits: 4)
            try addAffine("\(prefix).conv.out_proj.weight",
                          rows: config.hiddenSize,
                          columns: config.hiddenSize, bits: 4)
            addBF16("\(prefix).conv.conv.weight",
                    shape: (UInt32(config.hiddenSize),
                            UInt32(config.convKernelSize), 1, 0),
                    count: config.hiddenSize * config.convKernelSize)
        case .fullAttention, .slidingAttention:
            let kvDimension = config.numKVHeads * config.headDim
            try addAffine("\(prefix).self_attn.q_proj.weight",
                          rows: config.hiddenSize,
                          columns: config.hiddenSize, bits: 4)
            try addAffine("\(prefix).self_attn.k_proj.weight",
                          rows: kvDimension,
                          columns: config.hiddenSize, bits: 4)
            try addAffine("\(prefix).self_attn.v_proj.weight",
                          rows: kvDimension,
                          columns: config.hiddenSize, bits: 4)
            try addAffine("\(prefix).self_attn.o_proj.weight",
                          rows: config.hiddenSize,
                          columns: config.hiddenSize, bits: 4)
            addBF16("\(prefix).self_attn.q_layernorm.weight",
                    shape: (UInt32(config.headDim), 0, 0, 0),
                    count: config.headDim)
            addBF16("\(prefix).self_attn.k_layernorm.weight",
                    shape: (UInt32(config.headDim), 0, 0, 0),
                    count: config.headDim)
        }

        if layer < config.numDenseLayers {
            try addAffine("\(prefix).feed_forward.gate_proj.weight",
                          rows: config.intermediateSize,
                          columns: config.hiddenSize, bits: 4)
            try addAffine("\(prefix).feed_forward.up_proj.weight",
                          rows: config.intermediateSize,
                          columns: config.hiddenSize, bits: 4)
            try addAffine("\(prefix).feed_forward.down_proj.weight",
                          rows: config.hiddenSize,
                          columns: config.intermediateSize, bits: 4)
        } else {
            addFP32("\(prefix).feed_forward.gate.weight",
                    shape: (UInt32(config.numExperts),
                            UInt32(config.hiddenSize), 0, 0),
                    count: config.numExperts * config.hiddenSize)
            addFP32("\(prefix).feed_forward.expert_bias",
                    shape: (UInt32(config.numExperts), 0, 0, 0),
                    count: config.numExperts)
        }
    }

    let header = ResidentIndexHeader(
        indexSize: 16_384, residentSize: cursor - 16_384,
        entryCount: UInt64(entries.count))
    let residentIndex = ResidentIndex(header: header, entries: entries)

    var expertTensors: [String: SubTensorEntry] = [:]
    var offset: UInt64 = 0
    let routedShapes: [(String, Int, Int)] = [
        ("gate", config.moeIntermediateSize, config.hiddenSize),
        ("up", config.moeIntermediateSize, config.hiddenSize),
        ("down", config.hiddenSize, config.moeIntermediateSize),
    ]
    for (role, rows, columns) in routedShapes {
        let sizes = try Quantization.affineSizes(
            rows: rows, columns: columns, bits: 4,
            groupSize: Quantization.groupSize)
        expertTensors[role] = SubTensorEntry(
            offset: offset, size: sizes.weightBytes, dtype: "U32",
            shape: [UInt32(rows), UInt32(columns)], bits: 4)
        offset += sizes.weightBytes
        let auxShape = [UInt32(rows),
                        UInt32(columns / Quantization.groupSize)]
        for component in ["scales", "biases"] {
            expertTensors["\(role)_\(component)"] = SubTensorEntry(
                offset: offset, size: sizes.auxBytes, dtype: "BF16",
                shape: auxShape, bits: nil)
            offset += sizes.auxBytes
        }
    }
    guard offset == config.admission.expertStrideBytes else {
        throw ModelError.indexCorrupt(
            detail: "synthetic LFM routed stride \(offset) does not match "
                + "\(config.admission.expertStrideBytes)")
    }

    var layers: [LayerLayout] = []
    for layer in 0..<config.numLayers where config.moeLayerMask[layer] == 1 {
        let experts = (0..<config.numExperts).map { expert in
            ExpertEntry(
                expert: expert,
                offset: UInt64(expert) * config.admission.expertStrideBytes,
                size: config.admission.expertStrideBytes,
                subTensors: expertTensors)
        }
        layers.append(LayerLayout(
            layer: layer,
            file: String(format: "layer_%02d.bin", layer),
            experts: experts))
    }
    let layout = PackedExpertsLayout(
        expertStride: config.admission.expertStrideBytes,
        numLayers: config.numLayers,
        expertsPerLayer: config.numExperts,
        layers: layers,
        moeLayerMask: config.moeLayerMask)

    return SyntheticLFM(residentIndex: residentIndex,
                        packedExpertsLayout: layout,
                        manifest: manifest)
}

private func syntheticLFMSlot(_ bits: Int) -> GTurboManifestQuantSlotV1 {
    GTurboManifestQuantSlotV1(
        weightBits: bits, scheme: "affine", scaleType: "bf16",
        biasType: "bf16", groupSize: Quantization.groupSize)
}

private func makeSyntheticLFMWire(hiddenSize: Int = 2048) -> GTurboManifestV1 {
    let config = ArchConfig.lfm2_5_24B_A2B
    let layerTypes: [LayerType] = (0..<config.numLayers).map {
        config.fullAttentionLayerMask[$0] == 1 ? .fullAttention : .conv
    }
    let zeroHash = String(repeating: "0", count: 64)
    return GTurboManifestV1(
        flags: ["streamingPresent": true,
                "turboQuantKV": false,
                "aneSharedExpert": false],
        modelID: "test/lfm-synthetic",
        sourceSnapshotHash: nil,
        arch: GTurboManifestArchV1(
            hiddenSize: hiddenSize,
            ffnIntermediate: config.intermediateSize,
            moeIntermediateSize: config.moeIntermediateSize,
            numHeads: config.numHeads,
            numKVHeads: config.numKVHeads,
            numFullKVHeads: config.numFullKVHeads,
            headDim: config.headDim,
            fullHeadDim: config.fullHeadDim,
            vocabSize: config.vocabSize,
            slidingWindow: config.slidingWindow,
            finalLogitSoftcap: config.finalLogitSoftcap,
            ropeTheta: config.ropeTheta,
            fullRopeTheta: config.fullRopeTheta,
            partialRotaryFactor: config.partialRotaryFactor,
            numLayers: config.numLayers,
            numExperts: config.numExperts,
            topKExperts: config.topKExperts,
            tieWordEmbeddings: config.tieWordEmbeddings,
            attentionKEqV: config.attentionKEqV,
            hiddenActivation: config.hiddenActivation,
            fullAttentionLayerMask: config.fullAttentionLayerMask.map(Int.init),
            family: .lfm2Moe,
            layerTypes: layerTypes,
            moeLayerMask: config.moeLayerMask.map(Int.init),
            attentionScale: config.attentionScale,
            rmsNormUnitOffset: config.rmsNormUnitOffset,
            ropeStyle: config.ropeStyle,
            convKernelSize: config.convKernelSize,
            routerKind: config.routerKind,
            sharedExpert: config.hasSharedExpert,
            embeddingScaledBySqrtHidden: config.embeddingScaledBySqrtHidden,
            numDenseLayers: config.numDenseLayers,
            routerNormalizeTopK: config.routerNormalizeTopK,
            routerBias: config.routerBias,
            routedScalingFactor: config.routedScalingFactor),
        quant: GTurboManifestQuantV1(
            embedding: syntheticLFMSlot(6),
            attention: syntheticLFMSlot(4),
            sharedExpert: syntheticLFMSlot(4),
            routedExpert: syntheticLFMSlot(4)),
        files: [
            "model_weights.bin": GTurboManifestFileV1(
                size: 1024, sha256: zeroHash),
            "packed_experts/layout.json": GTurboManifestFileV1(
                size: 1024, sha256: zeroHash),
        ],
        expertsPerLayer: config.numExperts,
        numLayers: config.numLayers,
        expertStride: config.admission.expertStrideBytes,
        bitWidthOverridesHonored: nil)
}

/// Writes only `manifest.json`: the LFM family with Gemma's hidden size must
/// fail arch validation before any file or size check.
private func writeLFMManifestWithGemmaConstants() throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("gturbo-lfm-family-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
        at: directory, withIntermediateDirectories: true)
    let data = try GTurboManifestCodec.encode(
        makeSyntheticLFMWire(hiddenSize: 2816))
    try data.write(to: directory.appendingPathComponent("manifest.json"))
    return directory
}
