import Foundation
import Metal
import TurboFieldfareFormat

/// LFM2.5 24B-A2B production runner.
///
/// The graph is family-specific and deliberately separate from `RealForwardRunner`:
///
///   embed_int6(token)                                  // no sqrt(H) scale
///   for L in 0..<40:
///     a = standard_rmsnorm(h, operator_norm, eps=1e-5)
///     conv layers:      proj = in_proj(a) -> [B | C | x]
///                       Bx = B * x; conv = 3-tap causal filter
///                       out = out_proj(C * conv)
///     attention layers: Q/K/V 4-bit GEMV
///                       Q/K standard per-head norm + full NeoX RoPE (theta 1e7)
///                       full causal attention (32 Q heads, 8 KV heads, dim 64, scale 0.125)
///                       out = o_proj(attn)
///     h = h + out
///     ffn = standard_rmsnorm(h, ffn_norm, eps=1e-5)
///     layers 0-1: SiLU dense FFN
///     layers 2-39: sigmoid+bias top-4 router over 64 experts, SiLU routed FFN
///     h = h + ffn
///   logits = DequantInt6GEMV(standard_rmsnorm(h, embedding_norm, eps=1e-5),
///                            tied embedding table)
///
/// The short-convolution state is the two preceding `Bx` rows per conv layer.
/// Rewinds inside `convRewindLimit` restore that state from a 130-row ring
/// indexed by absolute position, then rewind the KV cursor.
public final class LFM2ForwardRunner: TextRunner, @unchecked Sendable {
    public let convRewindLimit = 128

    private struct ConvLayerState {
        let weight: MTLBuffer      // [hidden, 3] FP16, oldest -> current
        let state: MTLBuffer       // [2, hidden] FP16, oldest first
        let ring: MTLBuffer        // [convRewindLimit + 2, hidden] FP16
    }

    private struct DenseLayerProjections {
        let gate: SharedExpertProjection
        let up: SharedExpertProjection
        let down: SharedExpertProjection
    }

    private let model: Model
    private let ctx: MetalContext
    private let cfg: ArchConfig
    private let weights: LFM2Weights
    private let kv: KVCacheManager
    private let runtimeConfiguration: RuntimeConfiguration

    public let maxContext: Int

    // Decode kernels.
    private let embed: EmbedLookupInt6
    private let rms: RMSNorm
    private let int4: DequantInt4GEMV
    private let int6: DequantInt6GEMV
    private let attention: Attention
    private let qkNormRoPE: LFM2QKNormRoPE
    private let shortConv: ShortConv
    private let convOps: LFM2ConvOps
    private let dense: LFM2SharedExpert
    private let router: LFM2Router
    private let moe: LFM2MoE

    // Prefill kernels.
    private let prefillRMS: PrefillRMSNorm
    private let prefillQMM: PrefillInt4QMM
    private let prefillAttention: PrefillAttention
    private let prefillRouter: LFM2PrefillRouter
    private let prefillGroupedMoE: LFM2PrefillMoE
    private let prefillMoE: PrefillMoE

    // Decode scratch.
    private let hidden: MTLBuffer
    private let normed: MTLBuffer
    private let qScratch: MTLBuffer
    private let oOut: MTLBuffer
    private let h1: MTLBuffer
    private let h2: MTLBuffer
    private let ffnX: MTLBuffer
    private let denseGateScratch: MTLBuffer
    private let denseUpScratch: MTLBuffer
    private let denseActScratch: MTLBuffer
    private let routerLogits: MTLBuffer
    private let routerIndices: MTLBuffer
    private let routerWeights: MTLBuffer
    private let moeActs: MTLBuffer
    private let zeroResidual: MTLBuffer
    private let convProj: MTLBuffer
    private let convBx: MTLBuffer
    private let convOut: MTLBuffer
    private let convGated: MTLBuffer
    private let convStateOut: MTLBuffer

    // Prefill scratch that is not part of `PrefillChunkScratchBuffers`.
    private let convPrefillProj: MTLBuffer
    private let convPrefillBx: MTLBuffer
    private let convPrefillOut: MTLBuffer

    // Per-layer weights.
    private let convLayers: [ConvLayerState?]
    private let denseLayers: [DenseLayerProjections?]

    private var prefillChunkState = PrefillChunkCommitState()
    private var prefillScratch: PrefillChunkScratchBuffers?

    private static let normEpsilon: Float = 1e-5
    private static let attentionScale: Float = 0.125
    private static let fp16Stride = MemoryLayout<Float16>.stride
    private static let convRingRows = 128 + 2
    private static let prefillChunkCapacity = PrefillRuntimeConfig.maxChunkTokens

    public init(model: Model,
                context: MetalContext,
                maxContext: Int,
                runtimeConfiguration: RuntimeConfiguration = .production) throws {
        let cfg = model.config
        guard cfg.family == .lfm2Moe else {
            throw ModelError.unsupportedFamily(cfg.family.rawValue)
        }
        try Self.validate(config: cfg)
        guard maxContext > 0 else {
            throw PrefillError.chunkedUnsupported("maxContext must be positive")
        }

        self.model = model
        self.ctx = context
        self.cfg = cfg
        self.maxContext = maxContext
        self.runtimeConfiguration = runtimeConfiguration
        self.weights = LFM2Weights(model: model)

        self.kv = try KVCacheManager(
            device: context.device,
            config: cfg,
            maxContext: maxContext,
            fp16RingEnabled: runtimeConfiguration.fp16RingEnabled,
            slidingWindow: cfg.slidingWindow,
            maxPrefillChunkTokens: PrefillRuntimeConfig.maxChunkTokens)

        self.embed = try EmbedLookupInt6(context: context)
        self.rms = try RMSNorm(context: context)
        self.int4 = try DequantInt4GEMV(context: context)
        self.int6 = try DequantInt6GEMV(context: context)
        self.attention = try Attention(context: context, maxQHeads: 32)
        self.qkNormRoPE = try LFM2QKNormRoPE(context: context)
        self.shortConv = try ShortConv(context: context)
        self.convOps = try LFM2ConvOps(context: context)
        self.dense = try LFM2SharedExpert(context: context)
        self.router = try LFM2Router(context: context)
        self.moe = try LFM2MoE(context: context)

        self.prefillRMS = try PrefillRMSNorm(context: context)
        self.prefillQMM = try PrefillInt4QMM(context: context)
        self.prefillAttention = try PrefillAttention(context: context)
        self.prefillRouter = try LFM2PrefillRouter(context: context)
        self.prefillGroupedMoE = try LFM2PrefillMoE(context: context)
        self.prefillMoE = try PrefillMoE(context: context)

        let device = context.device
        let D = cfg.hiddenSize
        let F = cfg.intermediateSize
        let Fmoe = cfg.moeIntermediateSize
        let topK = cfg.topKExperts
        let qDim = cfg.numHeads * cfg.headDim
        let chunk = Self.prefillChunkCapacity

        func shared(_ elements: Int,
                    stride: Int = MemoryLayout<Float16>.stride,
                    label: String) throws -> MTLBuffer {
            try Self.makeBuffer(device: device,
                                length: max(elements, 1) * stride,
                                options: .storageModeShared,
                                label: label)
        }
        func privateBuffer(_ elements: Int,
                           stride: Int = MemoryLayout<Float16>.stride,
                           label: String) throws -> MTLBuffer {
            try Self.makeBuffer(device: device,
                                length: max(elements, 1) * stride,
                                options: .storageModePrivate,
                                label: label)
        }

        self.hidden = try privateBuffer(D, label: "lfm.hidden")
        self.normed = try privateBuffer(D, label: "lfm.normed")
        self.qScratch = try privateBuffer(qDim, label: "lfm.q")
        self.oOut = try privateBuffer(D, label: "lfm.o")
        self.h1 = try privateBuffer(D, label: "lfm.h1")
        self.h2 = try privateBuffer(D, label: "lfm.h2")
        self.ffnX = try privateBuffer(D, label: "lfm.ffnX")
        self.denseGateScratch = try privateBuffer(F, label: "lfm.denseGate")
        self.denseUpScratch = try privateBuffer(F, label: "lfm.denseUp")
        self.denseActScratch = try privateBuffer(F, label: "lfm.denseAct")
        self.routerLogits = try privateBuffer(
            Int(LFM2Router.numExperts),
            stride: MemoryLayout<Float>.stride,
            label: "lfm.routerLogits")
        self.routerIndices = try shared(
            topK,
            stride: MemoryLayout<UInt32>.stride,
            label: "lfm.routerIndices")
        self.routerWeights = try shared(topK, label: "lfm.routerWeights")
        self.moeActs = try privateBuffer(topK * Fmoe, label: "lfm.moeActs")
        self.zeroResidual = try shared(D, label: "lfm.zeroResidual")
        memset(self.zeroResidual.contents(), 0, self.zeroResidual.length)
        self.convProj = try privateBuffer(3 * D, label: "lfm.convProj")
        self.convBx = try privateBuffer(D, label: "lfm.convBx")
        self.convOut = try privateBuffer(D, label: "lfm.convOut")
        self.convGated = try privateBuffer(D, label: "lfm.convGated")
        self.convStateOut = try shared(2 * D, label: "lfm.convStateOut")

        self.convPrefillProj = try privateBuffer(chunk * 3 * D,
                                                 label: "lfm.prefill.convProj")
        self.convPrefillBx = try privateBuffer(chunk * D,
                                               label: "lfm.prefill.convBx")
        self.convPrefillOut = try privateBuffer(chunk * D,
                                                label: "lfm.prefill.convOut")

        // Conv weights are BF16 in the installed pack. Convert once to the
        // FP16 `[channels, 3]` layout `ShortConv` reads; binding the BF16 bytes
        // to that slot is not diagnosed and yields plausible wrong numbers.
        var convLayers: [ConvLayerState?] = Array(repeating: nil, count: cfg.numLayers)
        var denseLayers: [DenseLayerProjections?] = Array(repeating: nil, count: cfg.numLayers)
        let ringRows = Self.convRingRows

        for layer in 0..<cfg.numLayers {
            let layerType = cfg.layerTypes[layer]
            if layerType == .conv {
                let source = try weights.convWeight(layer: layer)
                let tapCount = cfg.hiddenSize * cfg.convKernelSize
                let required = tapCount * MemoryLayout<UInt16>.stride
                guard source.length >= UInt64(required) else {
                    throw ModelError.tensorSizeMismatch(
                        name: "layers.\(layer).conv.conv.weight",
                        expected: UInt64(required),
                        actual: source.length)
                }
                let converted = try shared(tapCount,
                                           label: "lfm.convWeight.L\(layer)")
                let src = source.buffer.contents()
                    .advanced(by: Int(source.offset))
                    .assumingMemoryBound(to: UInt16.self)
                let dst = converted.contents()
                    .assumingMemoryBound(to: Float16.self)
                for index in 0..<tapCount {
                    dst[index] = Float16(Quantization.bf16ToFloat(src[index]))
                }
                let state = try shared(2 * D, label: "lfm.convState.L\(layer)")
                memset(state.contents(), 0, state.length)
                let ring = try shared(ringRows * D, label: "lfm.convRing.L\(layer)")
                memset(ring.contents(), 0, ring.length)
                convLayers[layer] = ConvLayerState(weight: converted,
                                                   state: state,
                                                   ring: ring)
            }

            if layer < cfg.numDenseLayers {
                let gate = try weights.denseGate(layer: layer)
                let up = try weights.denseUp(layer: layer)
                let down = try weights.denseDown(layer: layer)
                denseLayers[layer] = DenseLayerProjections(
                    gate: Self.projection(gate, rows: F, cols: D),
                    up: Self.projection(up, rows: F, cols: D),
                    down: Self.projection(down, rows: D, cols: F))
            }
        }
        self.convLayers = convLayers
        self.denseLayers = denseLayers
    }

    // MARK: - Validation

    private static func validate(config cfg: ArchConfig) throws {
        func require(_ condition: Bool, _ field: String,
                     expected: String, actual: String) throws {
            guard condition else {
                throw ModelError.archMismatch(field: field,
                                              expected: expected,
                                              actual: actual)
            }
        }
        try require(cfg.hiddenSize == Int(LFM2Router.hiddenDimension),
                    "hiddenSize",
                    expected: "\(LFM2Router.hiddenDimension)",
                    actual: "\(cfg.hiddenSize)")
        try require(cfg.numExperts == Int(LFM2Router.numExperts),
                    "numExperts",
                    expected: "\(LFM2Router.numExperts)",
                    actual: "\(cfg.numExperts)")
        try require(cfg.topKExperts == Int(LFM2Router.topK),
                    "topKExperts",
                    expected: "\(LFM2Router.topK)",
                    actual: "\(cfg.topKExperts)")
        try require(cfg.convKernelSize == 3,
                    "convKernelSize",
                    expected: "3",
                    actual: "\(cfg.convKernelSize)")
        try require(cfg.headDim > 0 && cfg.headDim.isMultiple(of: 2),
                    "headDim",
                    expected: "a positive even value",
                    actual: "\(cfg.headDim)")
        try require(cfg.fullHeadDim == cfg.headDim,
                    "fullHeadDim",
                    expected: "\(cfg.headDim)",
                    actual: "\(cfg.fullHeadDim)")
        try require(cfg.numHeads == 32 && cfg.numKVHeads == 8,
                    "numHeads/numKVHeads",
                    expected: "32/8",
                    actual: "\(cfg.numHeads)/\(cfg.numKVHeads)")
        try require(cfg.numFullKVHeads == cfg.numKVHeads,
                    "numFullKVHeads",
                    expected: "\(cfg.numKVHeads)",
                    actual: "\(cfg.numFullKVHeads)")
        try require(!cfg.embeddingScaledBySqrtHidden,
                    "embeddingScaledBySqrtHidden",
                    expected: "false",
                    actual: "true")
        try require(!cfg.rmsNormUnitOffset,
                    "rmsNormUnitOffset",
                    expected: "false",
                    actual: "true")
        try require(!cfg.hasSharedExpert,
                    "hasSharedExpert",
                    expected: "false",
                    actual: "true")
        try require(cfg.routerBias && cfg.routerNormalizeTopK,
                    "routerBias/routerNormalizeTopK",
                    expected: "true/true",
                    actual: "\(cfg.routerBias)/\(cfg.routerNormalizeTopK)")
    }

    private static func projection(_ view: TensorView,
                                   rows: Int,
                                   cols: Int) -> SharedExpertProjection {
        SharedExpertProjection(weights: view.buffer,
                               scales: view.buffer,
                               biases: view.buffer,
                               weightsOffset: Int(view.offset),
                               scalesOffset: Int(view.scaleOffset),
                               biasesOffset: Int(view.biasOffset),
                               rows: UInt32(rows),
                               cols: UInt32(cols))
    }

    private static func makeBuffer(device: MTLDevice,
                                   length: Int,
                                   options: MTLResourceOptions,
                                   label: String) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: max(length, 1),
                                             options: options) else {
            throw ModelError.residentBufferWrapFailed
        }
        buffer.label = label
        return buffer
    }

    // MARK: - TextRunner lifecycle

    public var usesFusedGreedyHead: Bool { false }
    public var lastGreedyToken: UInt32 { 0 }
    public var continuationPosition: Int { kv.position }

    public func reset() {
        kv.reset()
        for layer in convLayers {
            guard let layer else { continue }
            memset(layer.state.contents(), 0, layer.state.length)
            memset(layer.ring.contents(), 0, layer.ring.length)
        }
        prefillChunkState.reset()
    }

    public func prepareForContinuation(expectedPosition: Int) throws {
        guard expectedPosition > 0, kv.position == expectedPosition else {
            throw PrefillError.prefillCursorMismatch(
                "continuation expected position \(expectedPosition), current \(kv.position)")
        }
        prefillChunkState.reset()
    }

    public func rewind(to position: Int) throws {
        guard position > 0,
              position <= kv.position,
              position >= kv.highWaterPosition - convRewindLimit else {
            throw PrefillError.prefillCursorMismatch(
                "cannot rewind from \(kv.position) to \(position); "
                    + "high water \(kv.highWaterPosition), conv ring \(convRewindLimit)")
        }
        let rows = Self.convRingRows
        for layer in convLayers {
            guard let layer else { continue }
            let state = layer.state.contents().assumingMemoryBound(to: Float16.self)
            let ring = layer.ring.contents().assumingMemoryBound(to: Float16.self)
            if position >= 2 {
                let oldest = (position - 2) % rows
                memcpy(state,
                        ring.advanced(by: oldest * cfg.hiddenSize),
                        cfg.hiddenSize * Self.fp16Stride)
            } else {
                memset(state, 0, cfg.hiddenSize * Self.fp16Stride)
            }
            let previous = (position - 1) % rows
            memcpy(state.advanced(by: cfg.hiddenSize),
                   ring.advanced(by: previous * cfg.hiddenSize),
                   cfg.hiddenSize * Self.fp16Stride)
        }
        kv.rewind(to: position)
        prefillChunkState.reset()
    }

    // MARK: - Decode

    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try prefillChunkState.requireClean(operation: "produce")
        guard position == kv.position else {
            throw PrefillError.prefillCursorMismatch(
                "produce cursor \(kv.position) != position \(position)")
        }
        guard position < maxContext else {
            throw PrefillError.prefillCursorMismatch(
                "produce position \(position) exceeds maxContext \(maxContext)")
        }
        try requireLogits(logits)

        let D = UInt32(cfg.hiddenSize)
        let embedding = try weights.embedding()

        try runSync { cb in
            embed.encode(commandBuffer: cb,
                         table: embedding.buffer, tableOffset: Int(embedding.offset),
                         scales: embedding.buffer, scalesOffset: Int(embedding.scaleOffset),
                         biases: embedding.buffer, biasesOffset: Int(embedding.biasOffset),
                         out: hidden,
                         tokenId: UInt32(bitPattern: token),
                         d: D,
                         outScale: 1.0)
        }

        for layerIndex in 0..<cfg.numLayers {
            let layerType = cfg.layerTypes[layerIndex]
            let operatorNorm = try weights.operatorNorm(layer: layerIndex)
            let ffnNorm = try weights.ffnNorm(layer: layerIndex)
            let branchCB = try makeCommandBuffer("lfm decode layer \(layerIndex)")

            rms.encodeStandard(commandBuffer: branchCB,
                               x: hidden,
                               weight: operatorNorm.buffer,
                               weightOffset: Int(operatorNorm.offset),
                               out: normed,
                               d: D,
                               eps: Self.normEpsilon)

            if layerType == .conv {
                try encodeConvDecode(commandBuffer: branchCB,
                                     layer: layerIndex,
                                     position: position)
            } else {
                try encodeAttentionDecode(commandBuffer: branchCB,
                                          layer: layerIndex,
                                          position: position)
            }
            convOps.encodeResidualAdd(commandBuffer: branchCB,
                                      hidden: hidden,
                                      branch: oOut,
                                      count: cfg.hiddenSize)
            rms.encodeStandard(commandBuffer: branchCB,
                               x: hidden,
                               weight: ffnNorm.buffer,
                               weightOffset: Int(ffnNorm.offset),
                               out: ffnX,
                               d: D,
                               eps: Self.normEpsilon)

            if layerIndex < cfg.numDenseLayers {
                guard let projections = denseLayers[layerIndex] else {
                    throw ModelError.tensorNotFound(
                        name: "layers.\(layerIndex).feed_forward.dense")
                }
                try dense.encode(commandBuffer: branchCB,
                                 x: ffnX,
                                 gate: projections.gate,
                                 up: projections.up,
                                 down: projections.down,
                                 y: h1,
                                 scratchGate: denseGateScratch,
                                 scratchUp: denseUpScratch,
                                 scratchAct: denseActScratch)
                convOps.encodeResidualAdd(commandBuffer: branchCB,
                                          hidden: hidden,
                                          branch: h1,
                                          count: cfg.hiddenSize)
                branchCB.commit()
                try waitForCompletion(branchCB)
                continue
            }

            // MoE: the router logits must reach the CPU before the streamed
            // expert fetch can issue, so the layer splits here.
            guard cfg.moeLayerMask[layerIndex] == 1 else {
                throw ModelError.tensorNotFound(
                    name: "layers.\(layerIndex).feed_forward.gate.weight")
            }
            let gateWeight = try weights.routerGate(layer: layerIndex)
            let bias = try weights.expertBias(layer: layerIndex)
            router.encodeLogits(commandBuffer: branchCB,
                                weights: gateWeight.buffer,
                                weightsOffset: Int(gateWeight.offset),
                                x: ffnX,
                                logits: routerLogits)
            router.encodeSelect(commandBuffer: branchCB,
                                logits: routerLogits,
                                bias: bias.buffer,
                                biasOffset: Int(bias.offset),
                                indices: routerIndices,
                                weights: routerWeights,
                                topK: UInt32(cfg.topKExperts),
                                routedScalingFactor: Float(cfg.routedScalingFactor))
            branchCB.commit()
            try waitForCompletion(branchCB)

            let experts = try readRouterExperts()
            let routedOffsets = try model.routedExpertOffsets(layer: layerIndex)
            let blobs = try await model.fetchRoutedExperts(layer: layerIndex,
                                                           experts: experts)
            guard blobs.count == cfg.topKExperts else {
                throw ModelError.indexCorrupt(
                    detail: "layer \(layerIndex) fetched \(blobs.count) experts, expected \(cfg.topKExperts)")
            }
            let routedBuffers = blobs.map(\.buffer)
            let argumentBuffer = moe.makeReusedRoutedArgumentBuffer(
                routedBlobs: routedBuffers,
                topK: UInt32(cfg.topKExperts))

            guard let moeCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            moeCB.label = "lfm decode layer \(layerIndex) moe"
            moe.encodePhase1(commandBuffer: moeCB,
                             routedArgBuffer: argumentBuffer,
                             routedBlobs: routedBuffers,
                             routedOffsets: routedOffsets,
                             x: ffnX,
                             acts: moeActs,
                             d: D,
                             f: UInt32(cfg.moeIntermediateSize),
                             topK: UInt32(cfg.topKExperts))
            moe.encodePhase2(commandBuffer: moeCB,
                             routedArgBuffer: argumentBuffer,
                             routedBlobs: routedBuffers,
                             routedOffsets: routedOffsets,
                             acts: moeActs,
                             routingWeights: routerWeights,
                             residual: zeroResidual,
                             y: h2,
                             d: D,
                             f: UInt32(cfg.moeIntermediateSize),
                             topK: UInt32(cfg.topKExperts))
            convOps.encodeResidualAdd(commandBuffer: moeCB,
                                      hidden: hidden,
                                      branch: h2,
                                      count: cfg.hiddenSize)
            moeCB.commit()
            try withExtendedLifetime((argumentBuffer, routedBuffers)) {
                try waitForCompletion(moeCB)
            }
        }

        // Final standard norm + tied 6-bit lm-head.
        let finalNorm = try weights.embeddingNorm()
        guard let headCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        headCB.label = "lfm decode head"
        rms.encodeStandard(commandBuffer: headCB,
                           x: hidden,
                           weight: finalNorm.buffer,
                           weightOffset: Int(finalNorm.offset),
                           out: normed,
                           d: D,
                           eps: Self.normEpsilon)
        int6.encode(commandBuffer: headCB,
                    weights: embedding.buffer, weightsOffset: Int(embedding.offset),
                    scales: embedding.buffer, scalesOffset: Int(embedding.scaleOffset),
                    biases: embedding.buffer, biasesOffset: Int(embedding.biasOffset),
                    x: normed,
                    y: logits,
                    m: UInt32(cfg.vocabSize),
                    n: D)
        headCB.commit()
        try waitForCompletion(headCB)
        kv.advance()
    }

    private func encodeConvDecode(commandBuffer: MTLCommandBuffer,
                                  layer: Int,
                                  position: Int) throws {
        guard let conv = convLayers[layer] else {
            throw ModelError.tensorNotFound(name: "layers.\(layer).conv")
        }
        let input = try weights.convIn(layer: layer)
        let output = try weights.convOut(layer: layer)
        let D = UInt32(cfg.hiddenSize)

        int4.encode(commandBuffer: commandBuffer,
                    weights: input.buffer, weightsOffset: Int(input.offset),
                    scales: input.buffer, scalesOffset: Int(input.scaleOffset),
                    biases: input.buffer, biasesOffset: Int(input.biasOffset),
                    x: normed,
                    y: convProj,
                    m: UInt32(3 * cfg.hiddenSize),
                    n: D)
        convOps.encodeBxMul(commandBuffer: commandBuffer,
                            proj: convProj,
                            bx: convBx,
                            tokens: 1,
                            channels: cfg.hiddenSize)

        let ringRow = position % Self.convRingRows
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        blit.copy(from: convBx,
                  sourceOffset: 0,
                  to: conv.ring,
                  destinationOffset: ringRow * cfg.hiddenSize * Self.fp16Stride,
                  size: cfg.hiddenSize * Self.fp16Stride)
        blit.endEncoding()

        shortConv.encodeDecode(commandBuffer: commandBuffer,
                               bx: convBx,
                               weight: conv.weight,
                               state: conv.state,
                               out: convOut,
                               channels: D)
        convOps.encodeGateMul(commandBuffer: commandBuffer,
                              proj: convProj,
                              conv: convOut,
                              out: convGated,
                              tokens: 1,
                              channels: cfg.hiddenSize)
        int4.encode(commandBuffer: commandBuffer,
                    weights: output.buffer, weightsOffset: Int(output.offset),
                    scales: output.buffer, scalesOffset: Int(output.scaleOffset),
                    biases: output.buffer, biasesOffset: Int(output.biasOffset),
                    x: convGated,
                    y: oOut,
                    m: D,
                    n: D)
    }

    private func encodeAttentionDecode(commandBuffer: MTLCommandBuffer,
                                       layer: Int,
                                       position: Int) throws {
        guard kv.layerKind(layer) != .conv else {
            throw ModelError.tensorNotFound(name: "layers.\(layer).self_attn")
        }
        let q = try weights.qProj(layer: layer)
        let k = try weights.kProj(layer: layer)
        let v = try weights.vProj(layer: layer)
        let o = try weights.oProj(layer: layer)
        let qNorm = try weights.qLayernorm(layer: layer)
        let kNorm = try weights.kLayernorm(layer: layer)
        let headDim = UInt32(cfg.headDim)
        let qDim = UInt32(cfg.numHeads * cfg.headDim)
        let kvDim = UInt32(cfg.numKVHeads * cfg.headDim)
        let kSlot = kv.kSlot(layer: layer, position: position)
        let vSlot = kv.vSlot(layer: layer, position: position)

        int4.encode(commandBuffer: commandBuffer,
                    weights: q.buffer, weightsOffset: Int(q.offset),
                    scales: q.buffer, scalesOffset: Int(q.scaleOffset),
                    biases: q.buffer, biasesOffset: Int(q.biasOffset),
                    x: normed,
                    y: qScratch,
                    m: qDim,
                    n: UInt32(cfg.hiddenSize))
        int4.encode(commandBuffer: commandBuffer,
                    weights: k.buffer, weightsOffset: Int(k.offset),
                    scales: k.buffer, scalesOffset: Int(k.scaleOffset),
                    biases: k.buffer, biasesOffset: Int(k.biasOffset),
                    x: normed,
                    y: kSlot.buffer, yOffset: kSlot.offset,
                    m: kvDim,
                    n: UInt32(cfg.hiddenSize))
        int4.encode(commandBuffer: commandBuffer,
                    weights: v.buffer, weightsOffset: Int(v.offset),
                    scales: v.buffer, scalesOffset: Int(v.scaleOffset),
                    biases: v.buffer, biasesOffset: Int(v.biasOffset),
                    x: normed,
                    y: vSlot.buffer, yOffset: vSlot.offset,
                    m: kvDim,
                    n: UInt32(cfg.hiddenSize))

        qkNormRoPE.encode(commandBuffer: commandBuffer,
                          q: qScratch,
                          k: kSlot.buffer, kOffset: kSlot.offset,
                          qWeight: qNorm.buffer, qWeightOffset: Int(qNorm.offset),
                          kWeight: kNorm.buffer, kWeightOffset: Int(kNorm.offset),
                          qOut: qScratch,
                          kOut: kSlot.buffer, kOutOffset: kSlot.offset,
                          qHeads: UInt32(cfg.numHeads),
                          kvHeads: UInt32(cfg.numKVHeads),
                          headDim: headDim,
                          position: UInt32(position),
                          theta: Float(cfg.ropeTheta),
                          eps: Self.normEpsilon)

        let seqLen = UInt32(position + 1)
        let keyBuffer = kv.keyBuffer(layer: layer, validTokenCount: position + 1)
        let valueBuffer = kv.valueBuffer(layer: layer, validTokenCount: position + 1)
        if kv.layerKind(layer) == .full {
            attention.encodeFull(commandBuffer: commandBuffer,
                                 q: qScratch,
                                 k: keyBuffer,
                                 v: valueBuffer,
                                 out: h1,
                                 headDim: headDim,
                                 numQHeads: UInt32(cfg.numHeads),
                                 numKVHeads: UInt32(cfg.numKVHeads),
                                 seqLen: seqLen,
                                 scale: Self.attentionScale,
                                 useGroupedVector: false)
        } else {
            let ringCapacity = kv.ringCapacity(layer: layer)
            let activeRing = ringCapacity > 0 && Int(seqLen) > ringCapacity
                ? UInt32(ringCapacity)
                : 0
            attention.encodeSWA(commandBuffer: commandBuffer,
                                q: qScratch,
                                k: keyBuffer,
                                v: valueBuffer,
                                out: h1,
                                headDim: headDim,
                                numQHeads: UInt32(cfg.numHeads),
                                numKVHeads: UInt32(cfg.numKVHeads),
                                seqLen: seqLen,
                                window: UInt32(cfg.slidingWindow),
                                scale: Self.attentionScale,
                                ringCapacity: activeRing)
        }

        int4.encode(commandBuffer: commandBuffer,
                    weights: o.buffer, weightsOffset: Int(o.offset),
                    scales: o.buffer, scalesOffset: Int(o.scaleOffset),
                    biases: o.buffer, biasesOffset: Int(o.biasOffset),
                    x: h1,
                    y: oOut,
                    m: UInt32(cfg.hiddenSize),
                    n: qDim)
    }

    private func readRouterExperts() throws -> [Int] {
        let count = cfg.topKExperts
        let pointer = routerIndices.contents()
            .bindMemory(to: UInt32.self, capacity: count)
        var experts = [Int](repeating: 0, count: count)
        for index in 0..<count {
            experts[index] = min(Int(pointer[index]), cfg.numExperts - 1)
        }
        return experts
    }

    // MARK: - Chunked prefill

    public func prefillChunked(tokens: ArraySlice<Int32>,
                               startPosition: Int,
                               outputMode: PrefillOutputMode,
                               config: PrefillRuntimeConfig,
                               into logits: MTLBuffer,
                               onProgress: (Int) -> Void) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillChunked")
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "prefillChunked requires PrefillRuntimeConfig.mode == .chunked")
        }
        guard startPosition >= 0 else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill startPosition must be non-negative")
        }
        guard startPosition == kv.position else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kv.position) != startPosition \(startPosition)")
        }
        guard tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range starting at \(startPosition) with \(tokens.count) tokens exceeds maxContext \(maxContext)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }
        try requireLogits(logits)

        let scratch = try ensurePrefillScratch(config: config)
        let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count,
                                              startPosition: startPosition,
                                              config: config)
        for (spanIndex, span) in spans.enumerated() {
            let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
            let upper = tokens.index(lower, offsetBy: span.tokenCount)
            try await executePrefillChunk(tokens: tokens[lower..<upper],
                                          startPosition: span.startPosition,
                                          logits: logits,
                                          scratch: scratch,
                                          config: config,
                                          writeFinalHead: spanIndex == spans.count - 1)
            onProgress(span.completedCount)
        }
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }

    private func ensurePrefillScratch(config: PrefillRuntimeConfig) throws
        -> PrefillChunkScratchBuffers {
        let layout = PrefillChunkScratchLayout(config: cfg,
                                               chunkTokens: config.chunkTokens)
        if let scratch = prefillScratch, scratch.layout == layout {
            return scratch
        }
        let scratch = try PrefillChunkScratchBuffers.allocate(device: ctx.device,
                                                              layout: layout)
        prefillScratch = scratch
        return scratch
    }

    private func executePrefillChunk(tokens: ArraySlice<Int32>,
                                     startPosition: Int,
                                     logits: MTLBuffer,
                                     scratch: PrefillChunkScratchBuffers,
                                     config: PrefillRuntimeConfig,
                                     writeFinalHead: Bool) async throws {
        guard !tokens.isEmpty else { return }
        guard startPosition == kv.position else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kv.position) != startPosition \(startPosition)")
        }
        guard startPosition >= 0,
              startPosition + tokens.count <= maxContext else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range [\(startPosition), \(startPosition + tokens.count)) exceeds maxContext \(maxContext)")
        }
        guard tokens.count <= scratch.layout.chunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill token count \(tokens.count) exceeds scratch chunk size \(scratch.layout.chunkTokens)")
        }

        let t = tokens.count
        let D = cfg.hiddenSize
        let embedding = try weights.embedding()

        prefillChunkState.markDirty(startPosition: startPosition, tokenCount: t)

        // int6 embedding lookup, one row at a time. There is no fused 6-bit
        // chunk kernel; the lookup is a row gather and the cost is negligible
        // beside the projections.
        guard let embedCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        embedCB.label = "lfm prefill start=\(startPosition) count=\(t) embed"
        for (row, token) in tokens.enumerated() {
            embed.encode(commandBuffer: embedCB,
                         table: embedding.buffer, tableOffset: Int(embedding.offset),
                         scales: embedding.buffer, scalesOffset: Int(embedding.scaleOffset),
                         biases: embedding.buffer, biasesOffset: Int(embedding.biasOffset),
                         out: scratch.hidden,
                         outOffset: row * D * Self.fp16Stride,
                         tokenId: UInt32(bitPattern: token),
                         d: UInt32(D),
                         outScale: 1.0)
        }
        embedCB.commit()
        try waitForCompletion(embedCB)

        for layer in 0..<cfg.numLayers {
            model.beginOpeningRoutedExpertStreamer(layer: layer)
            let operatorNorm = try weights.operatorNorm(layer: layer)
            let ffnNorm = try weights.ffnNorm(layer: layer)
            guard let cb = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            cb.label = "lfm prefill start=\(startPosition) count=\(t) layer=\(layer)"

            prefillRMS.encodeBF16W(commandBuffer: cb,
                                   x: scratch.hidden,
                                   weight: operatorNorm.buffer,
                                   weightOffset: Int(operatorNorm.offset),
                                   out: scratch.normed,
                                   t: UInt32(t),
                                   d: UInt32(D),
                                   eps: Self.normEpsilon)

            if cfg.layerTypes[layer] == .conv {
                try encodeConvPrefill(commandBuffer: cb,
                                      layer: layer,
                                      startPosition: startPosition,
                                      tokenCount: t,
                                      scratch: scratch)
            } else {
                try encodeAttentionPrefill(commandBuffer: cb,
                                           layer: layer,
                                           startPosition: startPosition,
                                           tokenCount: t,
                                           config: config,
                                           scratch: scratch)
            }
            convOps.encodeResidualAdd(commandBuffer: cb,
                                      hidden: scratch.hidden,
                                      branch: scratch.attentionOutput,
                                      count: t * D)
            prefillRMS.encodeBF16W(commandBuffer: cb,
                                   x: scratch.hidden,
                                   weight: ffnNorm.buffer,
                                   weightOffset: Int(ffnNorm.offset),
                                   out: scratch.routerX,
                                   t: UInt32(t),
                                   d: UInt32(D),
                                   eps: Self.normEpsilon)

            if layer < cfg.numDenseLayers {
                guard let projections = denseLayers[layer] else {
                    throw ModelError.tensorNotFound(
                        name: "layers.\(layer).feed_forward.dense")
                }
                for row in 0..<t {
                    try dense.encode(commandBuffer: cb,
                                     x: scratch.routerX,
                                     xOffset: row * D * Self.fp16Stride,
                                     gate: projections.gate,
                                     up: projections.up,
                                     down: projections.down,
                                     y: scratch.h1,
                                     yOffset: row * D * Self.fp16Stride,
                                     scratchGate: scratch.sharedGateScratch,
                                     scratchUp: scratch.sharedUpScratch,
                                     scratchAct: scratch.sharedActScratch)
                }
                convOps.encodeResidualAdd(commandBuffer: cb,
                                          hidden: scratch.hidden,
                                          branch: scratch.h1,
                                          count: t * D)
                cb.commit()
                try waitForCompletion(cb)
                continue
            }

            guard cfg.moeLayerMask[layer] == 1 else {
                throw ModelError.tensorNotFound(
                    name: "layers.\(layer).feed_forward.gate.weight")
            }
            let gateWeight = try weights.routerGate(layer: layer)
            let bias = try weights.expertBias(layer: layer)
            prefillRouter.encodeBlock(commandBuffer: cb,
                                      weights: gateWeight.buffer,
                                      weightsOffset: Int(gateWeight.offset),
                                      bias: bias.buffer,
                                      biasOffset: Int(bias.offset),
                                      hidden: scratch.routerX,
                                      outIndices: scratch.routeIDs,
                                      outWeights: scratch.routeWeights,
                                      queryCount: UInt32(t),
                                      numExperts: UInt32(cfg.numExperts),
                                      d: UInt32(D),
                                      topK: UInt32(cfg.topKExperts),
                                      hiddenStrideElements: UInt32(D),
                                      routedScalingFactor: Float(cfg.routedScalingFactor))
            cb.commit()
            try waitForCompletion(cb)

            try await encodeMoEPrefill(layer: layer,
                                       tokenCount: t,
                                       scratch: scratch)
        }

        if writeFinalHead {
            let finalNorm = try weights.embeddingNorm()
            guard let headCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            headCB.label = "lfm prefill start=\(startPosition) count=\(t) head"
            rms.encodeStandard(commandBuffer: headCB,
                               x: scratch.hidden,
                               xOffset: (t - 1) * D * Self.fp16Stride,
                               weight: finalNorm.buffer,
                               weightOffset: Int(finalNorm.offset),
                               out: scratch.normed,
                               d: UInt32(D),
                               eps: Self.normEpsilon)
            int6.encode(commandBuffer: headCB,
                        weights: embedding.buffer, weightsOffset: Int(embedding.offset),
                        scales: embedding.buffer, scalesOffset: Int(embedding.scaleOffset),
                        biases: embedding.buffer, biasesOffset: Int(embedding.biasOffset),
                        x: scratch.normed,
                        y: logits,
                        m: UInt32(cfg.vocabSize),
                        n: UInt32(D))
            headCB.commit()
            try waitForCompletion(headCB)
        }

        kv.advance(by: t)
        prefillChunkState.markCommitted()
    }

    private func encodeConvPrefill(commandBuffer: MTLCommandBuffer,
                                   layer: Int,
                                   startPosition: Int,
                                   tokenCount t: Int,
                                   scratch: PrefillChunkScratchBuffers) throws {
        guard let conv = convLayers[layer] else {
            throw ModelError.tensorNotFound(name: "layers.\(layer).conv")
        }
        let input = try weights.convIn(layer: layer)
        let output = try weights.convOut(layer: layer)
        let D = cfg.hiddenSize

        prefillQMM.encode(commandBuffer: commandBuffer,
                          weights: input.buffer, weightsOffset: Int(input.offset),
                          scales: input.buffer, scalesOffset: Int(input.scaleOffset),
                          biases: input.buffer, biasesOffset: Int(input.biasOffset),
                          x: scratch.normed,
                          y: convPrefillProj,
                          t: t,
                          n: 3 * D,
                          k: D)
        convOps.encodeBxMul(commandBuffer: commandBuffer,
                            proj: convPrefillProj,
                            bx: convPrefillBx,
                            tokens: t,
                            channels: D)

        // Record every chunk row, not just the final two: a rewind inside the
        // chunk restores state from the ring.
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        for row in 0..<t {
            let ringRow = (startPosition + row) % Self.convRingRows
            blit.copy(from: convPrefillBx,
                      sourceOffset: row * D * Self.fp16Stride,
                      to: conv.ring,
                      destinationOffset: ringRow * D * Self.fp16Stride,
                      size: D * Self.fp16Stride)
        }
        blit.endEncoding()

        shortConv.encodePrefill(commandBuffer: commandBuffer,
                                bx: convPrefillBx,
                                weight: conv.weight,
                                stateIn: conv.state,
                                out: convPrefillOut,
                                stateOut: convStateOut,
                                tokens: UInt32(t),
                                channels: UInt32(D))
        guard let stateBlit = commandBuffer.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        stateBlit.copy(from: convStateOut,
                       sourceOffset: 0,
                       to: conv.state,
                       destinationOffset: 0,
                       size: 2 * D * Self.fp16Stride)
        stateBlit.endEncoding()

        convOps.encodeGateMul(commandBuffer: commandBuffer,
                              proj: convPrefillProj,
                              conv: convPrefillOut,
                              out: scratch.normed,
                              tokens: t,
                              channels: D)
        prefillQMM.encode(commandBuffer: commandBuffer,
                          weights: output.buffer, weightsOffset: Int(output.offset),
                          scales: output.buffer, scalesOffset: Int(output.scaleOffset),
                          biases: output.buffer, biasesOffset: Int(output.biasOffset),
                          x: scratch.normed,
                          y: scratch.attentionOutput,
                          t: t,
                          n: D,
                          k: D)
    }

    private func encodeAttentionPrefill(commandBuffer: MTLCommandBuffer,
                                        layer: Int,
                                        startPosition: Int,
                                        tokenCount t: Int,
                                        config: PrefillRuntimeConfig,
                                        scratch: PrefillChunkScratchBuffers) throws {
        guard kv.layerKind(layer) != .conv else {
            throw ModelError.tensorNotFound(name: "layers.\(layer).self_attn")
        }
        let q = try weights.qProj(layer: layer)
        let k = try weights.kProj(layer: layer)
        let v = try weights.vProj(layer: layer)
        let o = try weights.oProj(layer: layer)
        let qNorm = try weights.qLayernorm(layer: layer)
        let kNorm = try weights.kLayernorm(layer: layer)
        let D = cfg.hiddenSize
        let headDim = cfg.headDim
        let qDim = cfg.numHeads * headDim
        let kvDim = cfg.numKVHeads * headDim
        let isFull = kv.layerKind(layer) == .full

        prefillQMM.encode(commandBuffer: commandBuffer,
                          weights: q.buffer, weightsOffset: Int(q.offset),
                          scales: q.buffer, scalesOffset: Int(q.scaleOffset),
                          biases: q.buffer, biasesOffset: Int(q.biasOffset),
                          x: scratch.normed,
                          y: scratch.q,
                          t: t, n: qDim, k: D)
        prefillQMM.encode(commandBuffer: commandBuffer,
                          weights: k.buffer, weightsOffset: Int(k.offset),
                          scales: k.buffer, scalesOffset: Int(k.scaleOffset),
                          biases: k.buffer, biasesOffset: Int(k.biasOffset),
                          x: scratch.normed,
                          y: scratch.kStage,
                          t: t, n: kvDim, k: D)
        prefillQMM.encode(commandBuffer: commandBuffer,
                          weights: v.buffer, weightsOffset: Int(v.offset),
                          scales: v.buffer, scalesOffset: Int(v.scaleOffset),
                          biases: v.buffer, biasesOffset: Int(v.biasOffset),
                          x: scratch.normed,
                          y: scratch.vStage,
                          t: t, n: kvDim, k: D)

        // The Q/K norm+RoPE kernel is one position per dispatch. Rows are
        // independent, so in-place Q and K remain race-free.
        for row in 0..<t {
            qkNormRoPE.encode(commandBuffer: commandBuffer,
                              q: scratch.q,
                              qOffset: row * qDim * Self.fp16Stride,
                              k: scratch.kStage,
                              kOffset: row * kvDim * Self.fp16Stride,
                              qWeight: qNorm.buffer, qWeightOffset: Int(qNorm.offset),
                              kWeight: kNorm.buffer, kWeightOffset: Int(kNorm.offset),
                              qOut: scratch.q,
                              qOutOffset: row * qDim * Self.fp16Stride,
                              kOut: scratch.kStage,
                              kOutOffset: row * kvDim * Self.fp16Stride,
                              qHeads: UInt32(cfg.numHeads),
                              kvHeads: UInt32(cfg.numKVHeads),
                              headDim: UInt32(headDim),
                              position: UInt32(startPosition + row),
                              theta: Float(cfg.ropeTheta),
                              eps: Self.normEpsilon)
        }

        try copyPrefillKV(commandBuffer: commandBuffer,
                          layer: layer,
                          startPosition: startPosition,
                          tokenCount: t,
                          keySource: scratch.kStage,
                          valueSource: scratch.vStage,
                          bytesPerToken: kvDim * Self.fp16Stride)

        let params = PrefillAttentionParams(
            startPosition: UInt32(startPosition),
            queryCount: UInt32(t),
            headDim: UInt32(headDim),
            numQHeads: UInt32(cfg.numHeads),
            numKVHeads: UInt32(cfg.numKVHeads),
            kvValidCount: UInt32(startPosition + t),
            slidingWindow: isFull ? UInt32(startPosition + t)
                                  : UInt32(cfg.slidingWindow),
            kvTokenStrideElements: UInt32(kvDim),
            qTokenStrideElements: UInt32(qDim),
            oTokenStrideElements: UInt32(qDim),
            scale: Self.attentionScale,
            bidirectionalBlockStart: 0,
            bidirectionalBlockEnd: 0)
        let keyBuffer = kv.keyBuffer(layer: layer,
                                     validTokenCount: startPosition + t)
        let valueBuffer = kv.valueBuffer(layer: layer,
                                         validTokenCount: startPosition + t)
        let ringCapacity = kv.ringCapacity(layer: layer)
        let activeRing = ringCapacity > 0 && startPosition + t > ringCapacity
            ? UInt32(ringCapacity)
            : 0
        prefillAttention.encodeCausal(commandBuffer: commandBuffer,
                                      q: scratch.q,
                                      k: keyBuffer,
                                      v: valueBuffer,
                                      out: scratch.h1,
                                      params: params,
                                      kvRingCapacity: activeRing,
                                      layerKind: isFull ? .full : .slidingWindow,
                                      path: runtimeConfiguration.prefillAttentionPath)

        prefillQMM.encode(commandBuffer: commandBuffer,
                          weights: o.buffer, weightsOffset: Int(o.offset),
                          scales: o.buffer, scalesOffset: Int(o.scaleOffset),
                          biases: o.buffer, biasesOffset: Int(o.biasOffset),
                          x: scratch.h1,
                          y: scratch.attentionOutput,
                          t: t, n: D, k: qDim)
    }

    private func encodeMoEPrefill(layer: Int,
                                  tokenCount t: Int,
                                  scratch: PrefillChunkScratchBuffers) async throws {
        let D = cfg.hiddenSize
        let Fmoe = cfg.moeIntermediateSize
        let topK = cfg.topKExperts
        let routeCount = t * topK
        let idPointer = scratch.routeIDs.contents()
            .bindMemory(to: UInt32.self, capacity: routeCount)
        let weightPointer = scratch.routeWeights.contents()
            .bindMemory(to: Float16.self, capacity: routeCount)
        var indices = [UInt32](repeating: 0, count: routeCount)
        var routeWeights = [Float16](repeating: 0, count: routeCount)
        for index in 0..<routeCount {
            indices[index] = min(idPointer[index], UInt32(cfg.numExperts - 1))
            routeWeights[index] = weightPointer[index]
        }
        // Only the family-neutral pair packer is reused from `PrefillRouter`;
        // the route selection itself came from `prefill_lfm2_router_block`.
        let pairs = PrefillRouter.makeTokenExpertPairs(indices: indices,
                                                       weights: routeWeights,
                                                       queryCount: t,
                                                       topK: topK)
        let tileExperts = model.routedExpertCacheSlotCount(layer: layer).map {
            max(1, min(PrefillRoutedTileSchedulerConfig().tileExperts, $0))
        } ?? PrefillRoutedTileSchedulerConfig().tileExperts
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: t,
            topK: topK,
            numExperts: cfg.numExperts,
            tileExpertCount: tileExperts,
            expertSortKeys: try model.routedExpertPhysicalOffsets(layer: layer))
        let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(
            device: ctx.device,
            routes: routes)
        let routedOffsets = try model.routedExpertOffsets(layer: layer)
        var tileLifetime = PrefillStreamedTileSlotLifetime()

        for (tileIndex, tile) in routes.tiles.enumerated() {
            let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                model: model,
                layer: layer,
                tileIndex: tileIndex,
                routes: routes)
            try fetch.binding.validateCoversPairs(routes.sortedPairs,
                                                  pairStart: Int(tile.pairStart),
                                                  pairCount: Int(tile.pairCount))
            if !fetch.plannedMissSlots.isEmpty {
                try tileLifetime.begin(tileIndex: tileIndex,
                                       plannedSlots: fetch.plannedMissSlots)
            }
            let argumentBuffer = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                device: ctx.device,
                binding: fetch.binding)
            let params = PrefillGroupedRoutedMoEStreamedParams(
                pairStart: tile.pairStart,
                pairCount: tile.pairCount,
                d: UInt32(D),
                routedIntermediate: UInt32(Fmoe),
                topK: UInt32(topK),
                hiddenStrideElements: UInt32(D),
                binding: fetch.binding,
                offsets: routedOffsets)
            guard let tileCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            tileCB.label = "lfm prefill layer=\(layer) routed tile=\(tileIndex)"
            _ = prefillGroupedMoE.encodeStreamedBatched(
                commandBuffer: tileCB,
                hidden: scratch.routerX,
                sortedPairs: metadata.sortedPairs,
                routePartials: scratch.routePartials,
                gateUpActScratch: scratch.routedGateUpActScratch,
                downScratch: scratch.routedDownScratch,
                argumentBuffer: argumentBuffer,
                binding: fetch.binding,
                params: params,
                pairMicrobatchRows: scratch.layout.routedPairMicrobatchRows)
            tileCB.commit()
            try withExtendedLifetime((fetch, argumentBuffer)) {
                try waitForCompletion(tileCB)
            }
            if !fetch.plannedMissSlots.isEmpty {
                try tileLifetime.complete(tileIndex: tileIndex)
            }
        }

        guard let tailCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        tailCB.label = "lfm prefill layer=\(layer) moe reduce"
        prefillMoE.encodeReduceTokenMajor(commandBuffer: tailCB,
                                          routePartials: scratch.routePartials,
                                          routeWeights: scratch.routeWeights,
                                          h2: scratch.h2,
                                          queryCount: UInt32(t),
                                          topK: UInt32(topK),
                                          d: UInt32(D))
        convOps.encodeResidualAdd(commandBuffer: tailCB,
                                  hidden: scratch.hidden,
                                  branch: scratch.h2,
                                  count: t * D)
        tailCB.commit()
        try withExtendedLifetime(metadata) {
            try waitForCompletion(tailCB)
        }
    }

    private func copyPrefillKV(commandBuffer: MTLCommandBuffer,
                               layer: Int,
                               startPosition: Int,
                               tokenCount: Int,
                               keySource: MTLBuffer,
                               valueSource: MTLBuffer,
                               bytesPerToken: Int) throws {
        guard tokenCount > 0 else { return }
        let capacity = kv.capacity(layer: layer)
        guard capacity > 0 else {
            throw PrefillError.chunkedUnsupported(
                "layer \(layer) has no KV capacity")
        }
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        var copied = 0
        while copied < tokenCount {
            let absolute = startPosition + copied
            let physical = absolute % capacity
            let span = min(tokenCount - copied, capacity - physical)
            let keyRange = kv.kRange(layer: layer, start: absolute, count: span)
            let valueRange = kv.vRange(layer: layer, start: absolute, count: span)
            blit.copy(from: keySource,
                      sourceOffset: copied * bytesPerToken,
                      to: keyRange.buffer,
                      destinationOffset: keyRange.offset,
                      size: span * bytesPerToken)
            blit.copy(from: valueSource,
                      sourceOffset: copied * bytesPerToken,
                      to: valueRange.buffer,
                      destinationOffset: valueRange.offset,
                      size: span * bytesPerToken)
            copied += span
        }
        blit.endEncoding()
    }

    // MARK: - Utilities

    private func requireLogits(_ logits: MTLBuffer) throws {
        let required = cfg.vocabSize * Self.fp16Stride
        guard logits.length >= required else {
            throw PrefillError.chunkedUnsupported(
                "logits buffer holds \(logits.length) bytes; expected \(required)")
        }
    }

    private func makeCommandBuffer(_ label: String) throws -> MTLCommandBuffer {
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        cb.label = label
        return cb
    }

    private func runSync(_ body: (MTLCommandBuffer) -> Void) throws {
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        body(cb)
        cb.commit()
        try waitForCompletion(cb)
    }

    private func waitForCompletion(_ cb: MTLCommandBuffer) throws {
        cb.waitUntilCompleted()
        if cb.status == .error {
            throw ModelError.indexCorrupt(
                detail: "command buffer failed: \(cb.error?.localizedDescription ?? "unknown")")
        }
    }
}
