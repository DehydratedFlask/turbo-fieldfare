import Foundation
import Darwin
import Metal
import TurboFieldfareFormat

/// Which attention variant a layer runs. Gemma 4 interleaves 25 sliding-window
/// layers with 5 full-attention layers (the latter carry the K=V shared-tensor
/// quirk). LFM2.5 replaces attention with a short convolution on 30 of its 40
/// layers and runs full attention on the other 10. Sourced from
/// `ArchConfig.layerTypes`.
public enum LayerKind: Sendable { case swa, full, conv }

/// A read view the attention kernels bind. `offset` stays 0; ring-enabled SWA
/// layers expose the physical start slot for diagnostics while kernels map
/// logical positions with the supplied ring capacity.
public struct KVView: @unchecked Sendable {
    public let buffer: MTLBuffer
    /// Byte offset of logical position 0. Always 0 under linear storage.
    public let offset: Int
    /// Bytes per token (numKVHeads * headDim * sizeof(FP16)).
    public let stride: Int
    /// Number of valid positions written so far (== `position`). Attention reads
    /// `[0, validTokenCount]` (inclusive of the just-written token).
    public let validTokenCount: Int
    /// Ring start slot. 0 under linear storage; the hook for the wrap-aware path.
    public let startSlot: Int
}

/// Per-layer FP16 K/V storage for the decode loop.
///
/// One K buffer and one V buffer per attention layer, allocated once in `init`
/// — the decode hot path never allocates. Linear storage sizes every attention
/// layer for `maxContext`; FP16 ring storage caps SWA layers to their physical
/// capacity while full-attention layers remain linear. Conv layers
/// (`LayerKind.conv`, LFM2.5's short convolution) carry a fixed-size hidden
/// state instead of K/V, so they allocate nothing and report zero KV bytes.
///
/// Gemma 4 full-attention layers carry the `attention_k_eq_v` quirk: K and V
/// share the `k_proj` weight, so a single 4-bit dequant + GEMV produces the
/// raw projection. But after that the K-slot runs `k_norm` (per-head, with
/// scale) + RoPE while the V-slot runs `v_norm` (per-head, no scale) and
/// skips RoPE — they diverge before entering attention. So the cache buffers
/// must be separate; aliasing them would smash the V values with K's normed,
/// rotated bytes (gemma4-block.md §2.2). We always allocate K and V slots,
/// independent of `attentionKEqV`.
///
/// The K/V projection GEMV writes straight into the slot returned by
/// `kSlot`/`vSlot` (no separate `kv_write` kernel); the runner then norms +
/// optionally RoPE's each slot in place. `advance()` bumps the cursor once
/// both are written.
///
/// 8 GB rule: storage is bounded by per-layer physical capacity, allocated
/// once. `reset()` returns physical pages to the OS via `MADV_DONTNEED` so a
/// finished generation does not keep its KV resident into the next turn.
public final class KVCacheManager {
    public let config: ArchConfig
    public let maxContext: Int
    public let fp16RingEnabled: Bool

    private let kBuffers: [MTLBuffer?]
    private let vBuffers: [MTLBuffer?]
    private let strides:  [Int]         // bytes per token, per layer
    private let kinds:    [LayerKind]
    private let capacityTokens: [Int]
    private let slidingWindowTokens: Int

    public private(set) var position: Int = 0
    public private(set) var highWaterPosition: Int = 0

    private static let fp16Size = 2

    public init(device: MTLDevice,
                config: ArchConfig,
                maxContext: Int,
                fp16RingEnabled: Bool = false,
                slidingWindow: Int? = nil,
                maxPrefillChunkTokens: Int = 128,
                fp16RingCapacityOverride: Int? = nil) throws {
        precondition(maxContext > 0, "maxContext must be positive")
        precondition(maxPrefillChunkTokens > 0, "maxPrefillChunkTokens must be positive")
        // Refused before a single `makeBuffer`: a context past the checkpoint's
        // position ceiling has no valid RoPE positions, and allocating 5+ GiB
        // of KV before saying so is the expensive way to learn it.
        guard maxContext <= config.maxPositionEmbeddings else {
            throw ModelError.contextExceedsModel(requested: maxContext,
                                                 maximum: config.maxPositionEmbeddings)
        }
        self.config = config
        self.maxContext = maxContext
        let ringEnabled = fp16RingEnabled
        self.fp16RingEnabled = ringEnabled

        let swaStride  = config.numKVHeads     * config.headDim     * Self.fp16Size
        let fullStride = config.numFullKVHeads * config.fullHeadDim  * Self.fp16Size
        let swaCapacity = min(maxContext,
                              max(1, fp16RingCapacityOverride
                                  ?? ((slidingWindow ?? config.slidingWindow) + maxPrefillChunkTokens)))
        self.slidingWindowTokens = slidingWindow ?? config.slidingWindow

        var ks: [MTLBuffer?] = []
        var vs: [MTLBuffer?] = []
        var st: [Int] = []
        var kd: [LayerKind] = []
        var caps: [Int] = []
        ks.reserveCapacity(config.numLayers)
        vs.reserveCapacity(config.numLayers)
        st.reserveCapacity(config.numLayers)
        kd.reserveCapacity(config.numLayers)
        caps.reserveCapacity(config.numLayers)

        for layer in 0..<config.numLayers {
            let kind = Self.layerKind(config: config, layer: layer)
            kd.append(kind)
            // Conv layers run a fixed-size short-convolution state held by the
            // runner; they own no rows of K/V, so they allocate nothing.
            guard kind != .conv else {
                ks.append(nil)
                vs.append(nil)
                st.append(0)
                caps.append(0)
                continue
            }
            let stride = kind == .full ? fullStride : swaStride
            let capacity = ringEnabled && kind == .swa ? swaCapacity : maxContext
            let length = capacity * stride

            guard let kBuf = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw ModelError.kvAllocationFailed(layer: layer, bytes: length)
            }
            kBuf.label = "kv.K.layer\(layer)"
            ks.append(kBuf)

            guard let vBuf = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw ModelError.kvAllocationFailed(layer: layer, bytes: length)
            }
            vBuf.label = "kv.V.layer\(layer)"
            vs.append(vBuf)

            st.append(stride)
            caps.append(capacity)
        }

        self.kBuffers = ks
        self.vBuffers = vs
        self.strides  = st
        self.kinds    = kd
        self.capacityTokens = caps
    }

    /// Per-layer mechanism from `LayerType`. A config whose `layerTypes` is
    /// shorter than `numLayers` (never produced by the loader) falls back to
    /// the Gemma mask rather than indexing out of bounds.
    private static func layerKind(config: ArchConfig, layer: Int) -> LayerKind {
        if layer < config.layerTypes.count {
            switch config.layerTypes[layer] {
            case .fullAttention: return .full
            case .slidingAttention: return .swa
            case .conv: return .conv
            }
        }
        return config.fullAttentionLayerMask[layer] != 0 ? .full : .swa
    }

    public func layerKind(_ layer: Int) -> LayerKind { kinds[layer] }

    /// This layer's K buffer, or `nil` for a conv layer. Callers that need to
    /// bind a buffer should use `kSlot`/`kRange`/`keyView`, which require an
    /// attention layer.
    public func keyBufferIfPresent(layer: Int) -> MTLBuffer? { kBuffers[layer] }

    /// This layer's V buffer, or `nil` for a conv layer.
    public func valueBufferIfPresent(layer: Int) -> MTLBuffer? { vBuffers[layer] }

    /// Bytes per token for `layer` (K and V share the same stride). Zero for
    /// conv layers, which have no KV rows.
    public func stride(layer: Int) -> Int { strides[layer] }

    /// Physical token capacity for `layer`. Ring-enabled SWA layers can be
    /// smaller than `maxContext`; full layers and ring-off storage stay linear.
    /// Zero for conv layers.
    public func capacity(layer: Int) -> Int { capacityTokens[layer] }

    public func ringCapacity(layer: Int) -> Int {
        guard fp16RingEnabled, kinds[layer] == .swa else { return 0 }
        return capacityTokens[layer]
    }

    /// Total bytes of the K buffer for `layer`. Zero for conv layers.
    public func bufferLength(layer: Int) -> Int {
        return capacityTokens[layer] * strides[layer]
    }

    /// Write target for this layer's K projection at `position`. A conv layer
    /// has no KV; asking for one is a programming error and fails the
    /// precondition instead of force-unwrapping a missing buffer.
    public func kSlot(layer: Int, position: Int) -> (buffer: MTLBuffer, offset: Int) {
        validateRange(start: position, count: 1)
        let buffer = requireKV(layer: layer, isKey: true)
        return (buffer, physicalSlot(layer: layer, position: position) * strides[layer])
    }

    /// Write target for this layer's V projection at `position`. Always
    /// distinct from `kSlot` — full layers no longer alias K and V (Gemma 4
    /// applies different per-head norms + RoPE to K vs V; gemma4-block.md §2.2).
    public func vSlot(layer: Int, position: Int) -> (buffer: MTLBuffer, offset: Int) {
        validateRange(start: position, count: 1)
        let buffer = requireKV(layer: layer, isKey: false)
        return (buffer, physicalSlot(layer: layer, position: position) * strides[layer])
    }

    public func kRange(layer: Int, start: Int, count: Int) -> (buffer: MTLBuffer, offset: Int, stride: Int) {
        validateRange(start: start, count: count)
        validateContiguousPhysicalRange(layer: layer, start: start, count: count)
        let buffer = requireKV(layer: layer, isKey: true)
        return (buffer, physicalSlot(layer: layer, position: start) * strides[layer], strides[layer])
    }

    public func vRange(layer: Int, start: Int, count: Int) -> (buffer: MTLBuffer, offset: Int, stride: Int) {
        validateRange(start: start, count: count)
        validateContiguousPhysicalRange(layer: layer, start: start, count: count)
        let buffer = requireKV(layer: layer, isKey: false)
        return (buffer, physicalSlot(layer: layer, position: start) * strides[layer], strides[layer])
    }

    public func keyView(layer: Int) -> KVView {
        keyView(layer: layer, validTokenCount: position)
    }

    public func keyView(layer: Int, validTokenCount: Int) -> KVView {
        validateValidTokenCount(validTokenCount)
        return KVView(buffer: requireKV(layer: layer, isKey: true), offset: 0,
                      stride: strides[layer],
                      validTokenCount: validTokenCount, startSlot: ringStartSlot(layer: layer,
                                                                                 validTokenCount: validTokenCount))
    }

    public func valueView(layer: Int) -> KVView {
        valueView(layer: layer, validTokenCount: position)
    }

    func keyBuffer(layer: Int, validTokenCount: Int) -> MTLBuffer {
        keyView(layer: layer, validTokenCount: validTokenCount).buffer
    }

    func valueBuffer(layer: Int, validTokenCount: Int) -> MTLBuffer {
        valueView(layer: layer, validTokenCount: validTokenCount).buffer
    }

    public func valueView(layer: Int, validTokenCount: Int) -> KVView {
        validateValidTokenCount(validTokenCount)
        return KVView(buffer: requireKV(layer: layer, isKey: false), offset: 0,
                      stride: strides[layer],
                      validTokenCount: validTokenCount, startSlot: ringStartSlot(layer: layer,
                                                                                 validTokenCount: validTokenCount))
    }

    /// The one documented place a nil K/V buffer is rejected. Reaching it means
    /// a caller asked a conv layer for attention storage.
    private func requireKV(layer: Int, isKey: Bool) -> MTLBuffer {
        precondition(kinds[layer] != .conv,
                     "layer \(layer) is a conv layer and has no \(isKey ? "K" : "V") cache")
        guard let buffer = isKey ? kBuffers[layer] : vBuffers[layer] else {
            preconditionFailure("layer \(layer) is missing its \(isKey ? "K" : "V") buffer")
        }
        return buffer
    }

    /// Advance the position cursor once the current token's K/V are written
    /// across all layers.
    public func advance() { advance(by: 1) }

    public func advance(by count: Int) {
        precondition(count >= 0, "advance count must be non-negative")
        precondition(position + count <= maxContext, "advance would exceed maxContext")
        position += count
        highWaterPosition = max(highWaterPosition, position)
    }

    /// Maximum safe rewind before a reused SWA ring slot could become visible.
    public var maxRewindTokens: Int {
        guard fp16RingEnabled else { return maxContext }
        var slack = maxContext
        for layer in 0..<config.numLayers where kinds[layer] == .swa {
            let capacity = capacityTokens[layer]
            guard capacity < maxContext else { continue }
            slack = min(slack, max(0, capacity - slidingWindowTokens))
        }
        return slack
    }

    /// Abandon the newest rows while keeping the remaining KV lineage intact.
    public func rewind(to newPosition: Int) {
        precondition(newPosition >= 0, "rewind target must be non-negative")
        precondition(newPosition <= position,
                     "rewind target \(newPosition) is beyond position \(position)")
        precondition(highWaterPosition - newPosition <= maxRewindTokens,
                     "rewind to \(newPosition) exceeds ring slack \(maxRewindTokens)")
        position = newPosition
    }

    /// Drop all cached positions and advise the pages away.
    ///
    /// No buffer zeroing — the attention kernels read only `[0, validTokenCount]`,
    /// and `validTokenCount` is now 0. The `MADV_DONTNEED` is advisory only:
    /// measured on macOS 26 (M5 Pro, 2026-08-28), `phys_footprint` did not move
    /// after advising a fully written 262,144-row cache, and a single attention
    /// bind had already charged each bound buffer in full. Sizing decisions
    /// must therefore treat the KV as resident at its allocated size for as
    /// long as the runner lives; see `ContextAdmission`.
    public func reset() {
        position = 0
        highWaterPosition = 0
        let pageSize = Int(getpagesize())
        var advised = Set<ObjectIdentifier>()
        for layer in 0..<config.numLayers {
            if let k = kBuffers[layer] { advise(k, pageSize: pageSize, seen: &advised) }
            if let v = vBuffers[layer] { advise(v, pageSize: pageSize, seen: &advised) }
        }
    }

    private func validateRange(start: Int, count: Int) {
        precondition(count >= 0, "count must be non-negative")
        precondition(start >= 0, "start must be non-negative")
        precondition(start + count <= maxContext,
                     "range \(start)..<\(start + count) exceeds maxContext \(maxContext)")
    }

    private func validateValidTokenCount(_ count: Int) {
        precondition(count >= 0, "validTokenCount must be non-negative")
        precondition(count <= maxContext,
                     "validTokenCount \(count) exceeds maxContext \(maxContext)")
    }

    private func physicalSlot(layer: Int, position: Int) -> Int {
        position % capacityTokens[layer]
    }

    private func ringStartSlot(layer: Int, validTokenCount: Int) -> Int {
        guard fp16RingEnabled, kinds[layer] == .swa else { return 0 }
        let capacity = capacityTokens[layer]
        guard validTokenCount > capacity else { return 0 }
        return validTokenCount % capacity
    }

    private func validateContiguousPhysicalRange(layer: Int, start: Int, count: Int) {
        guard count > 0, fp16RingEnabled, kinds[layer] == .swa else { return }
        let capacity = capacityTokens[layer]
        let physicalStart = start % capacity
        precondition(physicalStart + count <= capacity,
                     "range \(start)..<\(start + count) wraps FP16 KV ring capacity \(capacity)")
    }

    private func advise(_ buffer: MTLBuffer, pageSize: Int, seen: inout Set<ObjectIdentifier>) {
        let id = ObjectIdentifier(buffer)
        if seen.contains(id) { return }
        seen.insert(id)
        // MTLBuffer allocations are page-aligned; round the length down to a
        // whole number of pages so we never hand madvise a partial tail page.
        let len = (buffer.length / pageSize) * pageSize
        if len > 0 {
            _ = posix_madvise(buffer.contents(), len, POSIX_MADV_DONTNEED)
        }
    }
}
