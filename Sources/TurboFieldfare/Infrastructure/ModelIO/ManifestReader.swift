import Foundation
import TurboFieldfareFormat

public struct ManifestFileEntry: Decodable, Equatable, Sendable {
    public let size: UInt64
    public let sha256: String
}

public struct ManifestArch: Decodable, Equatable, Sendable {
    public let hiddenSize: Int
    public let ffnIntermediate: Int
    public let moeIntermediateSize: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let finalLogitSoftcap: Double
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let numLayers: Int
    public let numExperts: Int
    public let topKExperts: Int
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    public let hiddenActivation: String
    public let fullAttentionLayerMask: [Int]
    public let family: ModelFamily?
    public let layerTypes: [LayerType]?
    public let moeLayerMask: [Int]?
    public let attentionScale: Double?
    public let rmsNormUnitOffset: Bool?
    public let ropeStyle: RopeStyle?
    public let convKernelSize: Int?
    public let routerKind: RouterKind?
    public let sharedExpert: Bool?
    public let embeddingScaledBySqrtHidden: Bool?
    public let numDenseLayers: Int?
    public let routerNormalizeTopK: Bool?
    public let routerBias: Bool?
    public let routedScalingFactor: Double?
}

public struct ManifestQuantSlot: Decodable, Equatable, Sendable {
    public let weightBits: Int
    public let scheme: String
    public let scaleType: String
    public let biasType: String
    public let groupSize: Int
}

public struct ManifestQuant: Decodable, Equatable, Sendable {
    public let embedding: ManifestQuantSlot
    public let attention: ManifestQuantSlot
    public let router: ManifestQuantSlot?
    public let sharedExpert: ManifestQuantSlot
    public let routedExpert: ManifestQuantSlot
}

public struct Manifest: Decodable, Equatable, Sendable {
    public let magic: String
    public let versionMajor: Int
    public let versionMinor: Int
    public let flags: [String: Bool]
    public let modelID: String
    public let sourceSnapshotHash: String?
    public let arch: ManifestArch
    public let quant: ManifestQuant?
    public let files: [String: ManifestFileEntry]
    public let expertsPerLayer: Int
    public let numLayers: Int
    public let expertStride: UInt64
}

public enum ManifestReader {
    public static let defaultMaxBytes: UInt64 = 4 * 1024 * 1024

    /// Recognized flag keys. Anything else in `manifest.flags` is an error.
    public static let knownFlags: Set<String> = GTurboFormatV1.knownFlags

    /// Fixed required entries. Packed-layer filenames come from layout.json and
    /// are cross-validated only after that document is decoded.
    public static let requiredFiles: [String] = [
        "model_weights.bin",
        "packed_experts/layout.json",
    ]

    public static func load(directoryURL: URL,
                            expecting: ArchConfig,
                            maxBytes: UInt64 = defaultMaxBytes) throws -> Manifest {
        let directory = try GTurboModelDirectory(rootURL: directoryURL)
        let data: Data
        do {
            data = try directory.readMetadata("manifest.json", maxBytes: maxBytes)
        } catch ModelError.missingFile {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        return try decode(data: data, expecting: expecting)
    }

    /// Read only `arch.family`, without validating or hashing anything else.
    /// Used to pick the baseline before the full load; an absent family means
    /// `.gemma4`.
    public static func peekFamily(directoryURL: URL,
                                  maxBytes: UInt64 = defaultMaxBytes) throws -> ModelFamily {
        let directory = try GTurboModelDirectory(rootURL: directoryURL)
        let data: Data
        do {
            data = try directory.readMetadata("manifest.json", maxBytes: maxBytes)
        } catch ModelError.missingFile {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        do {
            let wire = try GTurboManifestCodec.decodeUnchecked(data)
            guard wire.magic == GTurboFormatV1.magic else {
                throw ModelError.notAGTurboDirectory
            }
            guard wire.versionMajor == GTurboFormatV1.versionMajor,
                  wire.versionMinor >= 0 else {
                throw ModelError.unsupportedVersion(major: wire.versionMajor,
                                                    minor: wire.versionMinor)
            }
            return wire.arch.family ?? .gemma4
        } catch let error as ModelError {
            throw error
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }
    }

    package static func decode(data: Data,
                               expecting: ArchConfig) throws -> Manifest {
        let manifest: Manifest
        do {
            let wire = try GTurboManifestCodec.decodeUnchecked(data)
            guard wire.magic == GTurboFormatV1.magic else {
                throw ModelError.notAGTurboDirectory
            }
            guard wire.versionMajor == GTurboFormatV1.versionMajor,
                  wire.versionMinor >= 0 else {
                throw ModelError.unsupportedVersion(major: wire.versionMajor,
                                                    minor: wire.versionMinor)
            }
            for key in wire.flags.keys where !GTurboFormatV1.knownFlags.contains(key) {
                throw ModelError.unknownFlag(name: key)
            }
            if wire.expertStride % GTurboFormatV1.alignmentBytes != 0 {
                throw ModelError.expertStrideNotPageAligned(
                    stride: wire.expertStride,
                    pageSize: Int(GTurboFormatV1.alignmentBytes))
            }
            try GTurboManifestCodec.validate(wire)
            manifest = Manifest(wire: wire)
        } catch let error as ModelError {
            throw error
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }

        try validate(manifest, against: expecting)
        return manifest
    }

    static func validate(_ m: Manifest,
                         against expected: ArchConfig) throws {
        if m.flags["turboQuantKV"] == true {
            throw ModelError.indexCorrupt(
                detail: "manifest requests removed TurboQuant KV runtime support")
        }
        try validateArch(m.arch, expertStride: m.expertStride, expected: expected)
        if let quant = m.quant {
            if expected.family == .gemma4, quant.router == nil {
                throw ModelError.indexCorrupt(
                    detail: "manifest.quant.router is required by the Gemma runtime")
            }
            try validateQuant(quant, family: expected.family)
        } else if expected.numLayers == ArchConfig.gemma4_26B_A4B.numLayers,
                  expected.hiddenSize == ArchConfig.gemma4_26B_A4B.hiddenSize {
            throw ModelError.indexCorrupt(detail: "manifest.quant is required for the production architecture")
        }
        for f in requiredFiles {
            if m.files[f] == nil { throw ModelError.missingFile(name: f) }
        }
    }

    private static func validateQuant(_ quant: ManifestQuant,
                                      family: ModelFamily) throws {
        let embeddingBits: Set<Int> = family == .lfm2Moe ? [6] : [4]
        var slots: [(String, ManifestQuantSlot, Set<Int>)] = [
            ("embedding", quant.embedding, embeddingBits),
            ("attention", quant.attention, [4]),
            ("sharedExpert", quant.sharedExpert, [4, 8]),
            ("routedExpert", quant.routedExpert, [4]),
        ]
        if let router = quant.router { slots.append(("router", router, [8])) }
        for (name, slot, allowedBits) in slots {
            guard allowedBits.contains(slot.weightBits),
                  slot.scheme.lowercased() == "affine",
                  slot.scaleType.lowercased() == "bf16",
                  slot.biasType.lowercased() == "bf16",
                  slot.groupSize == Quantization.groupSize else {
                throw ModelError.indexCorrupt(detail: "unsupported quantization for \(name)")
            }
        }
    }

    private static func validateArch(_ a: ManifestArch,
                                     expertStride: UInt64,
                                     expected e: ArchConfig) throws {
        func check<T: Equatable & CustomStringConvertible>(
            _ field: String, _ actual: T, _ expected: T) throws {
            if actual != expected {
                throw ModelError.archMismatch(field: field,
                                              expected: "\(expected)",
                                              actual: "\(actual)")
            }
        }
        try check("hiddenSize",          a.hiddenSize,          e.hiddenSize)
        try check("ffnIntermediate",     a.ffnIntermediate,     e.intermediateSize)
        try check("moeIntermediateSize", a.moeIntermediateSize, e.moeIntermediateSize)
        try check("numHeads",            a.numHeads,            e.numHeads)
        try check("numKVHeads",          a.numKVHeads,          e.numKVHeads)
        try check("numFullKVHeads",      a.numFullKVHeads,      e.numFullKVHeads)
        try check("headDim",             a.headDim,             e.headDim)
        try check("fullHeadDim",         a.fullHeadDim,         e.fullHeadDim)
        try check("vocabSize",           a.vocabSize,           e.vocabSize)
        try check("slidingWindow",       a.slidingWindow,       e.slidingWindow)
        try check("finalLogitSoftcap",   a.finalLogitSoftcap,   e.finalLogitSoftcap)
        try check("ropeTheta",           a.ropeTheta,           e.ropeTheta)
        try check("fullRopeTheta",       a.fullRopeTheta,       e.fullRopeTheta)
        try check("partialRotaryFactor", a.partialRotaryFactor, e.partialRotaryFactor)
        try check("numLayers",           a.numLayers,           e.numLayers)
        try check("numExperts",          a.numExperts,          e.numExperts)
        try check("topKExperts",         a.topKExperts,         e.topKExperts)
        try check("tieWordEmbeddings",   a.tieWordEmbeddings,   e.tieWordEmbeddings)
        try check("attentionKEqV",       a.attentionKEqV,       e.attentionKEqV)
        try check("hiddenActivation",    a.hiddenActivation,    e.hiddenActivation)
        let actualMask = a.fullAttentionLayerMask.map { UInt8($0) }
        try check("fullAttentionLayerMask",
                  actualMask.description,
                  e.fullAttentionLayerMask.description)
        // Optional traits: an absent field means the expected baseline value,
        // so pre-LFM manifests keep their meaning.
        try check("family",
                  (a.family ?? e.family).rawValue,
                  e.family.rawValue)
        try check("layerTypes",
                  (a.layerTypes ?? e.layerTypes).map(\.rawValue).description,
                  e.layerTypes.map(\.rawValue).description)
        let expectedMoeMask = e.moeLayerMask.map(Int.init)
        try check("moeLayerMask",
                  (a.moeLayerMask ?? expectedMoeMask).description,
                  expectedMoeMask.description)
        try check("attentionScale",      a.attentionScale ?? e.attentionScale,
                  e.attentionScale)
        try check("rmsNormUnitOffset",   a.rmsNormUnitOffset ?? e.rmsNormUnitOffset,
                  e.rmsNormUnitOffset)
        try check("ropeStyle",
                  (a.ropeStyle ?? e.ropeStyle).rawValue,
                  e.ropeStyle.rawValue)
        try check("convKernelSize",      a.convKernelSize ?? e.convKernelSize,
                  e.convKernelSize)
        try check("routerKind",
                  (a.routerKind ?? e.routerKind).rawValue,
                  e.routerKind.rawValue)
        try check("sharedExpert",        a.sharedExpert ?? e.hasSharedExpert,
                  e.hasSharedExpert)
        try check("embeddingScaledBySqrtHidden",
                  a.embeddingScaledBySqrtHidden ?? e.embeddingScaledBySqrtHidden,
                  e.embeddingScaledBySqrtHidden)
        try check("numDenseLayers",      a.numDenseLayers ?? e.numDenseLayers,
                  e.numDenseLayers)
        try check("routerNormalizeTopK", a.routerNormalizeTopK ?? e.routerNormalizeTopK,
                  e.routerNormalizeTopK)
        try check("routerBias",          a.routerBias ?? e.routerBias,
                  e.routerBias)
        try check("routedScalingFactor", a.routedScalingFactor ?? e.routedScalingFactor,
                  e.routedScalingFactor)
        try check("expertStride",        expertStride,
                  e.admission.expertStrideBytes)
    }
}

private extension ManifestFileEntry {
    init(wire: GTurboManifestFileV1) {
        self.init(size: wire.size, sha256: wire.sha256)
    }
}

private extension ManifestArch {
    init(wire: GTurboManifestArchV1) {
        self.init(hiddenSize: wire.hiddenSize,
                  ffnIntermediate: wire.ffnIntermediate,
                  moeIntermediateSize: wire.moeIntermediateSize,
                  numHeads: wire.numHeads,
                  numKVHeads: wire.numKVHeads,
                  numFullKVHeads: wire.numFullKVHeads,
                  headDim: wire.headDim,
                  fullHeadDim: wire.fullHeadDim,
                  vocabSize: wire.vocabSize,
                  slidingWindow: wire.slidingWindow,
                  finalLogitSoftcap: wire.finalLogitSoftcap,
                  ropeTheta: wire.ropeTheta,
                  fullRopeTheta: wire.fullRopeTheta,
                  partialRotaryFactor: wire.partialRotaryFactor,
                  numLayers: wire.numLayers,
                  numExperts: wire.numExperts,
                  topKExperts: wire.topKExperts,
                  tieWordEmbeddings: wire.tieWordEmbeddings,
                  attentionKEqV: wire.attentionKEqV,
                  hiddenActivation: wire.hiddenActivation,
                  fullAttentionLayerMask: wire.fullAttentionLayerMask,
                  family: wire.family,
                  layerTypes: wire.layerTypes,
                  moeLayerMask: wire.moeLayerMask,
                  attentionScale: wire.attentionScale,
                  rmsNormUnitOffset: wire.rmsNormUnitOffset,
                  ropeStyle: wire.ropeStyle,
                  convKernelSize: wire.convKernelSize,
                  routerKind: wire.routerKind,
                  sharedExpert: wire.sharedExpert,
                  embeddingScaledBySqrtHidden: wire.embeddingScaledBySqrtHidden,
                  numDenseLayers: wire.numDenseLayers,
                  routerNormalizeTopK: wire.routerNormalizeTopK,
                  routerBias: wire.routerBias,
                  routedScalingFactor: wire.routedScalingFactor)
    }
}

private extension ManifestQuantSlot {
    init(wire: GTurboManifestQuantSlotV1) {
        self.init(weightBits: wire.weightBits, scheme: wire.scheme,
                  scaleType: wire.scaleType, biasType: wire.biasType,
                  groupSize: wire.groupSize)
    }
}

private extension ManifestQuant {
    init(wire: GTurboManifestQuantV1) {
        self.init(embedding: ManifestQuantSlot(wire: wire.embedding),
                  attention: ManifestQuantSlot(wire: wire.attention),
                  router: wire.router.map(ManifestQuantSlot.init(wire:)),
                  sharedExpert: ManifestQuantSlot(wire: wire.sharedExpert),
                  routedExpert: ManifestQuantSlot(wire: wire.routedExpert))
    }
}

private extension Manifest {
    init(wire: GTurboManifestV1) {
        self.init(magic: wire.magic,
                  versionMajor: wire.versionMajor,
                  versionMinor: wire.versionMinor,
                  flags: wire.flags,
                  modelID: wire.modelID,
                  sourceSnapshotHash: wire.sourceSnapshotHash,
                  arch: ManifestArch(wire: wire.arch),
                  quant: wire.quant.map(ManifestQuant.init(wire:)),
                  files: wire.files.mapValues(ManifestFileEntry.init(wire:)),
                  expertsPerLayer: wire.expertsPerLayer,
                  numLayers: wire.numLayers,
                  expertStride: wire.expertStride)
    }
}
