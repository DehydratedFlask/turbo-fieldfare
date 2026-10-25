import Foundation
import TurboFieldfareFormat

/// The one host-memory admission rule for a requested context length.
///
/// Every surface that can pick a context — the app menu, the CLI, the server,
/// the decode service — asks this type whether the host can back it, so a
/// refusal, a hidden menu row and a warning all quote the same arithmetic. The
/// numbers are FP16 KV geometry plus two measured constants; nothing here
/// allocates or touches the model.
public enum ContextAdmission {

    /// Everything resident that does not scale with context: the LM weights,
    /// the expert-cache slots, prefill scratch, the tokenizer and the Metal
    /// library. Measured at 1.75-1.94 GB across PR 156's five rungs and the
    /// M2-8 8K row at 16 cache slots; extra slots are accounted for separately.
    /// Gemma 4's profile carries this same value; the arithmetic below reads
    /// `config.admission.nonKVRuntimeFootprintBytes` so a family can override
    /// it. This constant stays the Gemma 4 baseline and the public name.
    public static let nonKVRuntimeFootprintBytes: UInt64 = 2 << 30

    /// Bytes charged for expert-cache slots beyond the family's included
    /// baseline.
    ///
    /// Gemma 4 includes 16 slots in the measured 2 GiB allowance; LFM2.5
    /// includes none and charges from slot 0. Each routed layer owns its slots
    /// and the streamer rounds each allocation up to the host page size.
    /// Internal rather than public so its arithmetic can be pinned; the public
    /// surface is `projectedFootprintBytes`/`minimumHostMemoryBytes`.
    static func additionalExpertCacheBytes(config: ArchConfig, slots: Int) -> UInt64 {
        let page = UInt64(getpagesize())
        let stride = config.admission.expertStrideBytes
        let slotBytes = ((stride + page - 1) / page) * page
        // Do not discount the measured baseline for smaller caches: the rest
        // of that allowance has not been measured separately.
        return UInt64(max(0, slots - config.admission.includedExpertCacheSlots))
            * UInt64(config.admission.expertCacheLayers) * slotBytes
    }

    /// Headroom left to the OS and to a minimum routed-expert page cache. A
    /// host that satisfied only the projected footprint would run with every
    /// expert read faulting against a cold cache.
    public static let hostReserveBytes: UInt64 = 3 << 30

    public enum Availability: Equatable, Sendable {
        case available
        case needsMemory(minimumHostBytes: UInt64)
        case exceedsModel(maximum: Int)
    }

    /// Rows the production runtime allocates per sliding-window layer.
    ///
    /// `RealForwardRunner` floors its `maxPrefillChunkTokens` at the pooled
    /// image-token count, and `KVCacheManager` sizes the ring at
    /// `slidingWindow + maxPrefillChunkTokens`, so the widest chunk the runtime
    /// may see — not the default text chunk — is what the ring costs. 1,304
    /// today. A family with no sliding window (LFM2.5) has no ring at all.
    public static func productionSlidingRingRows(config: ArchConfig, maxContext: Int) -> Int {
        guard config.slidingWindow > 0 else { return 0 }
        let chunkRows = max(PrefillRuntimeConfig.defaultChunked.chunkTokens,
                            VisionConfig().maximumPooledTokens)
        return min(maxContext, config.slidingWindow + chunkRows)
    }

    /// FP16 K and V bytes the runtime allocates for `maxContext`.
    ///
    /// Summed per layer from `layerTypes`: conv layers contribute nothing,
    /// sliding-window layers are ring-capped, and only the full-attention
    /// layers grow with the context — at 20,480 B per token across Gemma 4's
    /// five of them and across LFM2.5's ten.
    public static func fp16KVBytes(config: ArchConfig, maxContext: Int) -> UInt64 {
        let fp16Bytes = 2
        let keyAndValue = 2
        let slidingRows = productionSlidingRingRows(config: config, maxContext: maxContext)
        let slidingBytesPerRow = config.numKVHeads * config.headDim * keyAndValue * fp16Bytes
        let fullBytesPerRow = config.numFullKVHeads * config.fullHeadDim * keyAndValue * fp16Bytes
        var total: UInt64 = 0
        for layer in 0..<config.numLayers {
            // Mirrors `KVCacheManager`'s layer-kind resolution exactly, so the
            // admission figure can never describe a different set of layers
            // than the one the runtime allocates for.
            let type: LayerType = layer < config.layerTypes.count
                ? config.layerTypes[layer]
                : (config.fullAttentionLayerMask[layer] != 0 ? .fullAttention : .slidingAttention)
            switch type {
            case .conv:
                continue
            case .slidingAttention:
                total += UInt64(slidingRows * slidingBytesPerRow)
            case .fullAttention:
                total += UInt64(maxContext * fullBytesPerRow)
            }
        }
        return total
    }

    /// What the process is expected to occupy at this context.
    public static func projectedFootprintBytes(config: ArchConfig, maxContext: Int, expertCacheSlots: Int = 16) -> UInt64 {
        fp16KVBytes(config: config, maxContext: maxContext)
            + config.admission.nonKVRuntimeFootprintBytes
            + additionalExpertCacheBytes(config: config, slots: expertCacheSlots)
    }

    /// The smallest host memory that admits this context.
    public static func minimumHostMemoryBytes(config: ArchConfig, maxContext: Int, expertCacheSlots: Int = 16) -> UInt64 {
        projectedFootprintBytes(config: config, maxContext: maxContext, expertCacheSlots: expertCacheSlots) + hostReserveBytes
    }

    public static func availability(config: ArchConfig,
                                    maxContext: Int,
                                    hostMemoryBytes: UInt64, expertCacheSlots: Int = 16) -> Availability {
        guard maxContext <= config.maxPositionEmbeddings else {
            return .exceedsModel(maximum: config.maxPositionEmbeddings)
        }
        let minimum = minimumHostMemoryBytes(config: config, maxContext: maxContext, expertCacheSlots: expertCacheSlots)
        return hostMemoryBytes >= minimum ? .available : .needsMemory(minimumHostBytes: minimum)
    }

    public static var hostMemoryBytes: UInt64 { ProcessInfo.processInfo.physicalMemory }

    /// One sentence naming what the context needs and what this Mac has, for a
    /// refusal, a hidden menu row's caption or a CLI warning.
    public static func needDescription(config: ArchConfig,
                                       maxContext: Int,
                                       hostMemoryBytes: UInt64, expertCacheSlots: Int = 16) -> String {
        guard maxContext <= config.maxPositionEmbeddings else {
            return "The model supports at most \(grouped(config.maxPositionEmbeddings)) tokens; "
                + "\(grouped(maxContext)) is beyond its trained positions."
        }
        let need = marketingGigabytesRoundedUp(
            minimumHostMemoryBytes(config: config, maxContext: maxContext, expertCacheSlots: expertCacheSlots))
        let have = marketingGigabytesNearest(hostMemoryBytes)
        return "A \(grouped(maxContext))-token context needs \(need) GB of memory; this Mac has \(have) GB."
    }

    /// Memory sizes Apple actually ships, so the sentence names a machine the
    /// reader can recognize instead of "10.25 GiB".
    private static let machineSizesGB = [8, 16, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512]

    /// The requirement is rounded in GiB because that is the unit
    /// `physicalMemory` reports: an "8 GB Mac" answers 8,589,934,592, so a need
    /// of 8,320,122,880 is met by it and must print as 8 GB, not 16.
    private static func marketingGigabytesRoundedUp(_ bytes: UInt64) -> Int {
        if let size = machineSizesGB.first(where: { UInt64($0) << 30 >= bytes }) { return size }
        return Int((bytes + (1 << 30) - 1) >> 30)
    }

    /// The host is reported in decimal GB, which is how the same machine is
    /// labeled in About This Mac.
    private static func marketingGigabytesNearest(_ bytes: UInt64) -> Int {
        let decimal = Double(bytes) / 1e9
        guard let nearest = machineSizesGB.min(by: {
            abs(Double($0) - decimal) < abs(Double($1) - decimal)
        }) else { return Int(decimal.rounded()) }
        return nearest
    }

    /// Grouped by threes without a locale, so the message reads the same in
    /// every region and in test assertions.
    private static func grouped(_ value: Int) -> String {
        let digits = Array(String(value))
        var out = ""
        for (index, digit) in digits.enumerated() {
            if index > 0, (digits.count - index) % 3 == 0 { out.append(",") }
            out.append(digit)
        }
        return out
    }
}
