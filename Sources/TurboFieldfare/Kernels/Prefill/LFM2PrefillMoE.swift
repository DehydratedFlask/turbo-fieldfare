import Metal

/// Gated activation used by the grouped routed MoE prefill phase-1 kernel.
///
/// Gemma 4 (the default) keeps `gelu_pytorch_tanh`; LFM2.5 selects SiLU. The
/// shader carries both behind function constant 50 so one scheduler serves
/// both families without a second kernel.
enum PrefillMoEActivation: Sendable {
    case gelu
    case silu
}

/// LFM2.5 top-4 grouped routed MoE for prefill.
///
/// Thin validation wrapper over `PrefillGroupedRoutedMoE`: the scheduler and
/// argument-buffer layout are shared with Gemma, while LFM2.5 requires top-4
/// selection from 64 experts and the SiLU gate/up activation. Prefill routing
/// itself (sigmoid + bias, normalized unbiased weights) is produced by
/// `PrefillRouter` and handed in as `PrefillMoEGroupedRoutes`.
final class LFM2PrefillMoE {
    static let topK: UInt32 = 4
    static let numExperts: UInt32 = 64

    private let grouped: PrefillGroupedRoutedMoE

    init(context: MetalContext) throws {
        self.grouped = try PrefillGroupedRoutedMoE(
            context: context, activation: .silu)
    }

    func makeStreamedMetadataBuffers(
        device: MTLDevice,
        routes: PrefillMoEGroupedRoutes
    ) throws -> PrefillGroupedRoutedMoEStreamedMetadataBuffers {
        try grouped.makeStreamedMetadataBuffers(device: device, routes: routes)
    }

    func makeStreamedArgumentBuffer(
        device: MTLDevice,
        binding: PrefillStreamedTileBinding
    ) throws -> PrefillStreamedTileArgumentBuffer {
        try grouped.makeStreamedArgumentBuffer(device: device, binding: binding)
    }

    @discardableResult
    func encodeStreamedBatched(commandBuffer: MTLCommandBuffer,
                               hidden: MTLBuffer,
                               hiddenOffset: Int = 0,
                               sortedPairs: MTLBuffer,
                               sortedPairsOffset: Int = 0,
                               routePartials: MTLBuffer,
                               routePartialsOffset: Int = 0,
                               gateUpActScratch: MTLBuffer,
                               gateUpActScratchOffset: Int = 0,
                               downScratch: MTLBuffer,
                               downScratchOffset: Int = 0,
                               argumentBuffer: PrefillStreamedTileArgumentBuffer,
                               binding: PrefillStreamedTileBinding,
                               params: PrefillGroupedRoutedMoEStreamedParams,
                               pairMicrobatchRows: Int = 32) -> Int {
        precondition(params.topK == Self.topK,
                     "LFM2 prefill MoE requires topK == 4, got \(params.topK)")
        precondition(binding.expertIDs.allSatisfy {
            $0 >= 0 && $0 < Int(Self.numExperts)
        }, "LFM2 expert id must be in 0..<64")
        return grouped.encodeStreamedBatched(
            commandBuffer: commandBuffer,
            hidden: hidden, hiddenOffset: hiddenOffset,
            sortedPairs: sortedPairs, sortedPairsOffset: sortedPairsOffset,
            routePartials: routePartials, routePartialsOffset: routePartialsOffset,
            gateUpActScratch: gateUpActScratch,
            gateUpActScratchOffset: gateUpActScratchOffset,
            downScratch: downScratch, downScratchOffset: downScratchOffset,
            argumentBuffer: argumentBuffer,
            binding: binding,
            params: params,
            pairMicrobatchRows: pairMicrobatchRows)
    }
}
