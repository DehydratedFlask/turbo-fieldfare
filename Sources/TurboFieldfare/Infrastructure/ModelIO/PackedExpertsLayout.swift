import Foundation
import TurboFieldfareFormat

struct SubTensorEntry: Sendable, Equatable {
    let offset: UInt64    // relative to the expert blob's start
    let size: UInt64
    let dtype: String     // "U32" | "BF16"
    let shape: [UInt32]
    let bits: Int?        // weight bit width override (4 or 8), if applicable
}

struct ExpertEntry: Sendable {
    /// Logical routed-expert id used by the model/router.
    let expert: Int
    /// Physical rank inside the layer file. Old identity layouts use
    /// `physicalRank == expert`.
    let physicalRank: Int
    /// Absolute byte offset of this expert blob's start inside its layer file.
    let offset: UInt64
    /// Total bytes consumed by this expert blob (== `expertStride`).
    let size: UInt64
    /// Sub-tensors keyed by role (gate / up / down / shared) and component
    /// (raw, `_scales`, `_biases`).
    let subTensors: [String: SubTensorEntry]

    init(expert: Int,
                physicalRank: Int? = nil,
                offset: UInt64,
                size: UInt64,
                subTensors: [String: SubTensorEntry]) {
        self.expert = expert
        self.physicalRank = physicalRank ?? expert
        self.offset = offset
        self.size = size
        self.subTensors = subTensors
    }
}

struct LayerLayout: Sendable {
    let layer: Int
    let file: String          // basename, e.g. "layer_00.bin"
    let experts: [ExpertEntry]
}

struct PackedExpertsLayout: Sendable {
    let expertStride: UInt64
    let numLayers: Int
    let expertsPerLayer: Int
    /// MoE layers only, sorted by layer index. Dense layers have no entry.
    let layers: [LayerLayout]
    /// `1` where the model layer has routed experts, `0` for a dense layer.
    let moeLayerMask: [UInt8]
    private let layerIndex: [Int: Int]

    init(expertStride: UInt64,
         numLayers: Int,
         expertsPerLayer: Int,
         layers: [LayerLayout],
         moeLayerMask: [UInt8]? = nil) {
        self.expertStride = expertStride
        self.numLayers = numLayers
        self.expertsPerLayer = expertsPerLayer
        self.layers = layers
        self.moeLayerMask = moeLayerMask
            ?? [UInt8](repeating: 1, count: numLayers)
        self.layerIndex = layers.enumerated().reduce(into: [:]) { index, pair in
            index[pair.element.layer] = pair.offset
        }
    }

    /// The MoE layer's layout. Dense and out-of-range layers throw.
    func layerLayout(for layer: Int) throws -> LayerLayout {
        guard moeLayerMask.indices.contains(layer), moeLayerMask[layer] == 1,
              let index = layerIndex[layer] else {
            throw ModelError.routedExpertsUnavailable(layer: layer)
        }
        return layers[index]
    }

    /// Resolve `(layer, expert)` -> `ExpertEntry`. O(1).
    func expert(layer: Int, expert: Int) throws -> ExpertEntry {
        try layerLayout(for: layer).experts[expert]
    }
}

enum PackedExpertsLayoutReader {
    static let defaultMaxBytes: UInt64 = 16 * 1024 * 1024

    static func load(directoryURL: URL,
                            maxBytes: UInt64 = defaultMaxBytes) throws -> PackedExpertsLayout {
        try load(directoryURL: directoryURL, manifest: nil, maxBytes: maxBytes)
    }

    package static func load(directoryURL: URL,
                             manifest: Manifest,
                             maxBytes: UInt64 = defaultMaxBytes) throws -> PackedExpertsLayout {
        try load(directoryURL: directoryURL, manifest: Optional(manifest), maxBytes: maxBytes)
    }

    private static func load(directoryURL: URL,
                             manifest: Manifest?,
                             maxBytes: UInt64) throws -> PackedExpertsLayout {
        let directory = try GTurboModelDirectory(rootURL: directoryURL)
        let data = try directory.readMetadata(
            "packed_experts/layout.json", maxBytes: maxBytes)
        return try decode(data: data, manifest: manifest)
    }

    package static func decode(data: Data,
                               manifest: Manifest?) throws -> PackedExpertsLayout {
        let wire: GTurboPackedExpertsLayoutV1
        let mask = manifest?.arch.moeLayerMask
        do {
            wire = try GTurboPackedExpertsLayoutCodec.decode(data, moeLayerMask: mask)
            if let manifest {
                try GTurboV1StructuralValidator.crossValidate(
                    manifestNumLayers: manifest.numLayers,
                    manifestExpertsPerLayer: manifest.expertsPerLayer,
                    manifestExpertStride: manifest.expertStride,
                    manifestFileSizes: manifest.files.mapValues(\.size),
                    layout: wire,
                    moeLayerMask: mask)
            }
        } catch {
            throw ModelError.indexCorrupt(detail: "layout.json: \(error)")
        }
        let layers = wire.layers.sorted { $0.layer < $1.layer }.map { layer -> LayerLayout in
            let experts = layer.experts.enumerated().map { position, expert -> ExpertEntry in
                let logical = expert.expert ?? position
                return ExpertEntry(
                    expert: logical,
                    physicalRank: expert.physicalRank,
                    offset: expert.offset,
                    size: expert.size,
                    subTensors: expert.tensors.mapValues {
                        SubTensorEntry(offset: $0.offset, size: $0.size,
                                       dtype: $0.dtype, shape: $0.shape,
                                       bits: $0.bits)
                    })
            }.sorted { $0.expert < $1.expert }
            return LayerLayout(layer: layer.layer, file: layer.file, experts: experts)
        }
        return PackedExpertsLayout(expertStride: wire.expertStride,
                                   numLayers: wire.numLayers,
                                   expertsPerLayer: wire.expertsPerLayer,
                                   layers: layers,
                                   moeLayerMask: (mask ?? Array(
                                       repeating: 1, count: wire.numLayers)).map { UInt8($0) })
    }

}
