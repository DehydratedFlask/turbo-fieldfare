import TurboFieldfare

public enum AppContextLengthOption: Int, CaseIterable, Identifiable, Sendable {
    case fourK = 4_096
    case eightK = 8_192
    case sixteenK = 16_384
    case thirtyTwoK = 32_768
    case sixtyFourK = 65_536
    case oneTwentyEightK = 131_072
    case twoFiftySixK = 262_144

    public var id: Int { rawValue }
    public var tokens: Int { rawValue }

    public var shortLabel: String {
        "\(tokens / 1_024)K"
    }

    /// Every figure comes from the active model's architecture; the
    /// no-argument properties keep Gemma as the compatibility default.
    public func fp16KVBytes(config: ArchConfig = .gemma4_26B_A4B) -> UInt64 {
        ContextAdmission.fp16KVBytes(config: config, maxContext: tokens)
    }

    public var fp16KVBytes: UInt64 { fp16KVBytes(config: .gemma4_26B_A4B) }

    public func projectedFootprintBytes(config: ArchConfig = .gemma4_26B_A4B) -> UInt64 {
        ContextAdmission.projectedFootprintBytes(config: config, maxContext: tokens)
    }

    public var projectedFootprintBytes: UInt64 {
        projectedFootprintBytes(config: .gemma4_26B_A4B)
    }

    public func minimumHostMemoryBytes(config: ArchConfig = .gemma4_26B_A4B) -> UInt64 {
        ContextAdmission.minimumHostMemoryBytes(config: config, maxContext: tokens)
    }

    public var minimumHostMemoryBytes: UInt64 {
        minimumHostMemoryBytes(config: .gemma4_26B_A4B)
    }

    public func availability(hostMemoryBytes: UInt64,
                             expertCacheSlots: Int = 16,
                             config: ArchConfig = .gemma4_26B_A4B) -> ContextAdmission.Availability {
        ContextAdmission.availability(config: config,
                                      maxContext: tokens,
                                      hostMemoryBytes: hostMemoryBytes,
                                      expertCacheSlots: expertCacheSlots)
    }

    public func needDescription(hostMemoryBytes: UInt64,
                                expertCacheSlots: Int = 16,
                                config: ArchConfig = .gemma4_26B_A4B) -> String {
        ContextAdmission.needDescription(config: config,
                                         maxContext: tokens,
                                         hostMemoryBytes: hostMemoryBytes,
                                         expertCacheSlots: expertCacheSlots)
    }

    /// The options this host can actually back, in ascending order.
    ///
    /// A 262,144-token KV is charged in full the moment a command buffer binds
    /// it — one full-attention layer over sixteen rows already costs the whole
    /// 1,085 MiB — so admission is decided by the cap, never by how much of it
    /// a conversation happens to use.
    public static func available(on hostMemoryBytes: UInt64,
                                 expertCacheSlots: Int = 16,
                                 config: ArchConfig = .gemma4_26B_A4B) -> [AppContextLengthOption] {
        allCases.filter {
            $0.availability(hostMemoryBytes: hostMemoryBytes,
                            expertCacheSlots: expertCacheSlots,
                            config: config) == .available
        }
    }

    /// The largest context this host can back, for clamping a stored choice it
    /// cannot. A host too small even for the smallest option still gets one:
    /// the loader refuses an unbacked context before it allocates, so the
    /// refusal is where that machine is told, not a menu with no rows in it.
    public static func largestAvailable(on hostMemoryBytes: UInt64,
                                        expertCacheSlots: Int = 16,
                                        config: ArchConfig = .gemma4_26B_A4B) -> AppContextLengthOption {
        available(on: hostMemoryBytes,
                  expertCacheSlots: expertCacheSlots,
                  config: config).last ?? .fourK
    }

    public static let defaultOption: AppContextLengthOption = .eightK

    /// The default row is labeled as such; every other row shows its FP16 KV
    /// allocation relative to the default, computed from `fp16KVBytes` rather
    /// than hardcoded so a default change moves every delta with it.
    public func menuLabel(config: ArchConfig = .gemma4_26B_A4B) -> String {
        guard self != Self.defaultOption else { return "\(shortLabel), Default" }
        let delta = Int64(fp16KVBytes(config: config))
            - Int64(Self.defaultOption.fp16KVBytes(config: config))
        return "\(shortLabel), \(Self.formattedDelta(delta))"
    }

    public var menuLabel: String { menuLabel(config: .gemma4_26B_A4B) }

    private static func formattedDelta(_ bytes: Int64) -> String {
        let sign = bytes < 0 ? "-" : "+"
        let megabytes = Double(abs(bytes)) / 1_000_000
        guard megabytes < 1_000 else {
            return String(format: "%@%.2f GB", sign, megabytes / 1_000)
        }
        let rounded = Int((megabytes / 5).rounded()) * 5
        return "\(sign)\(rounded) MB"
    }
}
