import Foundation
import Metal
import TurboFieldfareFormat

/// Family-specific expert paging and cache budget. The loader and the decode
/// service read these instead of assuming the Gemma layout.
public struct ArchAdmissionProfile: Sendable, Equatable {
    public let expertStrideBytes: UInt64
    public let expertCacheLayers: Int
    public let includedExpertCacheSlots: Int
    public let nonKVRuntimeFootprintBytes: UInt64

    public init(expertStrideBytes: UInt64, expertCacheLayers: Int,
                includedExpertCacheSlots: Int, nonKVRuntimeFootprintBytes: UInt64) {
        self.expertStrideBytes = expertStrideBytes
        self.expertCacheLayers = expertCacheLayers
        self.includedExpertCacheSlots = includedExpertCacheSlots
        self.nonKVRuntimeFootprintBytes = nonKVRuntimeFootprintBytes
    }

    /// Gemma 4 26B-A4B: 30 routed layers, 16 expert-cache slots included.
    public static let gemma4 = ArchAdmissionProfile(
        expertStrideBytes: 3_358_720,
        expertCacheLayers: 30,
        includedExpertCacheSlots: 16,
        nonKVRuntimeFootprintBytes: 2 << 30)

    /// LFM2.5 24B-A2B: 38 routed layers (two leading dense layers), no expert
    /// cache slots included.
    public static let lfm2 = ArchAdmissionProfile(
        expertStrideBytes: 5_308_416,
        expertCacheLayers: 38,
        includedExpertCacheSlots: 0,
        nonKVRuntimeFootprintBytes: 2 << 30)
}

/// Compile-time architecture baseline. `manifest.json -> arch` must match this
/// field-by-field at load time; mismatches throw `ModelError.archMismatch`.
public struct ArchConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let intermediateSize: Int          // shared expert FFN (== ffnIntermediate in manifest)
    public let moeIntermediateSize: Int       // per-expert FFN
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
    public let fullAttentionLayerMask: [UInt8]
    public let hiddenActivation: String
    /// The checkpoint's `max_position_embeddings`. The one ceiling every
    /// surface refuses above; not a manifest-compared field, so
    /// `ManifestReader.validateArch` does not read it.
    public let maxPositionEmbeddings: Int

    public let family: ModelFamily
    /// Per-layer mechanism. Derived from `fullAttentionLayerMask` when the
    /// initializer receives `nil`, so Gemma configs need no extra input.
    public let layerTypes: [LayerType]
    /// `1` where the layer has routed experts, `0` for dense layers.
    public let moeLayerMask: [UInt8]
    public let attentionScale: Double
    public let rmsNormUnitOffset: Bool
    public let ropeStyle: RopeStyle
    public let convKernelSize: Int
    public let routerKind: RouterKind
    public let hasSharedExpert: Bool
    public let embeddingScaledBySqrtHidden: Bool
    public let numDenseLayers: Int
    public let routerNormalizeTopK: Bool
    public let routerBias: Bool
    public let routedScalingFactor: Double
    public let admission: ArchAdmissionProfile

    public init(
        hiddenSize: Int,
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
        maxPositionEmbeddings: Int = 262_144,
        family: ModelFamily = .gemma4,
        layerTypes: [LayerType]? = nil,
        moeLayerMask: [UInt8]? = nil,
        attentionScale: Double = 1.0,
        rmsNormUnitOffset: Bool = true,
        ropeStyle: RopeStyle = .gemmaPartial,
        convKernelSize: Int = 0,
        routerKind: RouterKind = .gemmaSoftmaxTopK,
        hasSharedExpert: Bool = true,
        embeddingScaledBySqrtHidden: Bool = true,
        numDenseLayers: Int = 0,
        routerNormalizeTopK: Bool = false,
        routerBias: Bool = false,
        routedScalingFactor: Double = 1.0,
        admission: ArchAdmissionProfile = .gemma4
    ) {
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
        self.layerTypes = layerTypes ?? fullAttentionLayerMask.map {
            $0 == 1 ? .fullAttention : .slidingAttention
        }
        self.moeLayerMask = moeLayerMask ?? [UInt8](repeating: 1, count: numLayers)
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
        self.admission = admission
    }

    /// Canonical Gemma 4 26B-A4B baseline, checked against the installed
    /// model manifest.
    /// `intermediateSize = 2112` is the shared-expert FFN width (3 × moe).
    public static let gemma4_26B_A4B = ArchConfig(
        hiddenSize: 2816,
        intermediateSize: 2112,
        moeIntermediateSize: 704,
        numHeads: 16,
        numKVHeads: 8,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 512,
        vocabSize: 262144,
        slidingWindow: 1024,
        finalLogitSoftcap: 30.0,
        ropeTheta: 10_000.0,
        fullRopeTheta: 1_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 30,
        numExperts: 128,
        topKExperts: 8,
        tieWordEmbeddings: true,
        attentionKEqV: true,
        fullAttentionLayerMask: Self.gemma4LayerMask(),
        hiddenActivation: "gelu_pytorch_tanh",
        maxPositionEmbeddings: 262_144,
        family: .gemma4,
        admission: .gemma4
    )

    private static func gemma4LayerMask() -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: 30)
        for i in stride(from: 5, to: 30, by: 6) { mask[i] = 1 }
        return mask
    }

    /// Pinned LFM2.5 24B-A2B baseline: two leading dense conv layers, then a
    /// conv/full-attention mix with ten full-attention layers in total.
    public static let lfm2_5_24B_A2B = ArchConfig(
        hiddenSize: 2048,
        intermediateSize: 11_776,
        moeIntermediateSize: 1536,
        numHeads: 32,
        numKVHeads: 8,
        numFullKVHeads: 8,
        headDim: 64,
        fullHeadDim: 64,
        vocabSize: 128_000,
        slidingWindow: 0,
        finalLogitSoftcap: 0,
        ropeTheta: 10_000_000,
        fullRopeTheta: 10_000_000,
        partialRotaryFactor: 1.0,
        numLayers: 40,
        numExperts: 64,
        topKExperts: 4,
        tieWordEmbeddings: true,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.lfm2FullAttentionMask(),
        hiddenActivation: "silu",
        maxPositionEmbeddings: 131_072,
        family: .lfm2Moe,
        layerTypes: Self.lfm2LayerTypes(),
        moeLayerMask: Array(repeating: 0, count: 2)
            + Array(repeating: 1, count: 38),
        attentionScale: 0.125,
        rmsNormUnitOffset: false,
        ropeStyle: .neoxFull,
        convKernelSize: 3,
        routerKind: .lfm2SigmoidTopKBias,
        hasSharedExpert: false,
        embeddingScaledBySqrtHidden: false,
        numDenseLayers: 2,
        routerNormalizeTopK: true,
        routerBias: true,
        routedScalingFactor: 1.0,
        admission: .lfm2
    )

    /// Layer indices with full attention. Every other layer is a conv layer.
    private static let lfm2FullAttentionLayers = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38]

    private static func lfm2FullAttentionMask() -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: 40)
        for i in lfm2FullAttentionLayers { mask[i] = 1 }
        return mask
    }

    private static func lfm2LayerTypes() -> [LayerType] {
        (0..<40).map {
            lfm2FullAttentionLayers.contains($0) ? .fullAttention : .conv
        }
    }

    /// The pinned config for a manifest family, or `nil` when this build has no
    /// baseline for it.
    public static func canonical(for family: ModelFamily) -> ArchConfig? {
        switch family {
        case .gemma4:
            return .gemma4_26B_A4B
        case .lfm2Moe:
            return .lfm2_5_24B_A2B
        }
    }
}

/// Failure modes for the validation gates in `Model.load`.
public enum ModelError: Error, CustomStringConvertible, Equatable {
    case partialInstall(path: String)
    case notAGTurboDirectory
    case unsupportedVersion(major: Int, minor: Int)
    case unknownFlag(name: String)
    case archMismatch(field: String, expected: String, actual: String)
    case expertStrideNotPageAligned(stride: UInt64, pageSize: Int)
    case missingFile(name: String)
    case checksumMismatch(file: String)
    case tensorNotFound(name: String)
    case tensorSizeMismatch(name: String, expected: UInt64, actual: UInt64)
    case residentBufferWrapFailed
    case indexCorrupt(detail: String)
    case posixFailed(call: String, errno: Int32)
    case trustedReceiptInvalid(detail: String)
    case contextExceedsModel(requested: Int, maximum: Int)
    case kvAllocationFailed(layer: Int, bytes: Int)
    case routedExpertsUnavailable(layer: Int)

    public var description: String {
        switch self {
        case .partialInstall(let p):
            return "model.gturbo directory at \(p) is missing manifest.json"
        case .notAGTurboDirectory:
            return "manifest.json magic does not equal \"GTURBO\""
        case .unsupportedVersion(let maj, let min):
            return "manifest version \(maj).\(min) is not supported (need 1.x)"
        case .unknownFlag(let n):
            return "manifest.flags contains unknown key \"\(n)\""
        case .archMismatch(let field, let exp, let act):
            return "manifest.arch.\(field) = \(act); expected \(exp)"
        case .expertStrideNotPageAligned(let s, let p):
            return "expertStride \(s) is not a multiple of page size \(p)"
        case .missingFile(let n):
            return "model.gturbo is missing required file \(n)"
        case .checksumMismatch(let f):
            return "SHA-256 of \(f) does not match manifest.files[\(f)].sha256"
        case .tensorNotFound(let n):
            return "no IndexEntry named \(n) in model_weights.bin"
        case .tensorSizeMismatch(let n, let e, let a):
            return "tensor \(n) size \(a) does not match expected \(e)"
        case .residentBufferWrapFailed:
            return "MTLDevice.makeBuffer(bytesNoCopy:...) returned nil"
        case .indexCorrupt(let d):
            return "resident index is corrupt: \(d)"
        case .posixFailed(let c, let e):
            return "\(c) failed with errno \(e)"
        case .trustedReceiptInvalid(let detail):
            return "trusted install receipt invalid: \(detail)"
        case .contextExceedsModel(let requested, let maximum):
            return "maxContext \(requested) exceeds the model's maximum position \(maximum)"
        case .kvAllocationFailed(let layer, let bytes):
            return "KV cache allocation of \(bytes) bytes for layer \(layer) failed"
        case .routedExpertsUnavailable(let layer):
            return "layer \(layer) has no routed experts"
        }
    }
}

/// View into a tensor that lives inside one of the loader's resident or
/// streamed `MTLBuffer`s. No `MTLBuffer` is allocated per tensor — the
/// `buffer` reference is shared across many `TensorView` instances and
/// addressed by byte offsets.
public struct TensorView: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let offset: UInt64
    public let length: UInt64
    public let scaleOffset: UInt64
    public let scaleLength: UInt64
    public let biasOffset: UInt64
    public let biasLength: UInt64
    public let shape: (UInt32, UInt32, UInt32, UInt32)
    /// Dtype byte. 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    public let dtype: UInt8

    public init(buffer: MTLBuffer,
                offset: UInt64, length: UInt64,
                scaleOffset: UInt64, scaleLength: UInt64,
                biasOffset: UInt64, biasLength: UInt64,
                shape: (UInt32, UInt32, UInt32, UInt32),
                dtype: UInt8) {
        self.buffer = buffer
        self.offset = offset
        self.length = length
        self.scaleOffset = scaleOffset
        self.scaleLength = scaleLength
        self.biasOffset = biasOffset
        self.biasLength = biasLength
        self.shape = shape
        self.dtype = dtype
    }
}
