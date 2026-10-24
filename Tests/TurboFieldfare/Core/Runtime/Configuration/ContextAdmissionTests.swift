import Testing
import Foundation
@testable import TurboFieldfare

/// Pins the admission arithmetic every context-picking surface shares. The
/// byte figures are the FP16 KV allocation the runtime actually makes — 25
/// ring-capped sliding layers plus 5 linear full layers — so a change to the
/// ring geometry or the layer mask fails here before it silently moves a menu
/// label, a server refusal and a CLI warning apart.
@Suite struct ContextAdmissionTests {

    @Test(arguments: [24, 32])
    func extraCacheSlotsAreChargedAtTheAdmissionBoundary(slots: Int) {
        let config = ArchConfig.gemma4_26B_A4B
        let baseline = ContextAdmission.minimumHostMemoryBytes(config: config, maxContext: 131_072)
        let page = Int(getpagesize())
        let slotBytes = ((3_358_720 + page - 1) / page) * page
        let expected = baseline + UInt64((slots - 16) * 30 * slotBytes)
        #expect(ContextAdmission.minimumHostMemoryBytes(config: config, maxContext: 131_072,
                                                        expertCacheSlots: slots) == expected)
        #expect(ContextAdmission.availability(config: config, maxContext: 131_072,
                                              hostMemoryBytes: expected - 1, expertCacheSlots: slots)
                == .needsMemory(minimumHostBytes: expected))
        #expect(ContextAdmission.availability(config: config, maxContext: 131_072,
                                              hostMemoryBytes: expected, expertCacheSlots: slots) == .available)
        #expect(ContextAdmission.availability(config: config, maxContext: 131_072,
                                              hostMemoryBytes: 8 << 30, expertCacheSlots: slots)
                == .needsMemory(minimumHostBytes: expected))
    }

    @Test func smallerCacheDoesNotDiscountTheMeasuredRuntimeAllowance() {
        #expect(ContextAdmission.projectedFootprintBytes(config: config, maxContext: 131_072,
                                                         expertCacheSlots: 8)
                == ContextAdmission.projectedFootprintBytes(config: config, maxContext: 131_072))
    }


    private let config = ArchConfig.gemma4_26B_A4B
    private let eightGigabyteHost: UInt64 = 8_589_934_592
    private let sixteenGigabyteHost: UInt64 = 17_179_869_184

    @Test func fp16KVBytesMatchesTheRuntimeAllocationAtEverySize() {
        let expected: [(context: Int, bytes: UInt64)] = [
            (4_096, 350_945_280),
            (8_192, 434_831_360),
            (16_384, 602_603_520),
            (32_768, 938_147_840),
            (65_536, 1_609_236_480),
            (131_072, 2_951_413_760),
            (262_144, 5_635_768_320),
        ]
        for row in expected {
            #expect(ContextAdmission.fp16KVBytes(config: config, maxContext: row.context)
                    == row.bytes,
                    "context \(row.context)")
        }
    }

    /// Only the five full-attention layers grow with the context; the sliding
    /// ring is fixed. 20,480 B per token across those five is the whole slope.
    @Test func onlyFullAttentionLayersGrowWithContext() {
        let small = ContextAdmission.fp16KVBytes(config: config, maxContext: 8_192)
        let large = ContextAdmission.fp16KVBytes(config: config, maxContext: 16_384)
        #expect(large - small == UInt64((16_384 - 8_192) * 20_480))
    }

    @Test func productionSlidingRingRowsMatchTheRunnersChunkFloor() {
        #expect(ContextAdmission.productionSlidingRingRows(config: config, maxContext: 8_192)
                == 1_304)
        // A context below the ring size caps the ring at the context.
        #expect(ContextAdmission.productionSlidingRingRows(config: config, maxContext: 1_024)
                == 1_024)
    }

    @Test func eightGigabyteHostBacks128KButNot256K() {
        #expect(ContextAdmission.availability(config: config,
                                              maxContext: 131_072,
                                              hostMemoryBytes: eightGigabyteHost)
                == .available)
        #expect(ContextAdmission.availability(config: config,
                                              maxContext: 262_144,
                                              hostMemoryBytes: eightGigabyteHost)
                == .needsMemory(minimumHostBytes: 11_004_477_440))
    }

    @Test func sixteenGigabyteHostBacksEverySize() {
        for context in [4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144] {
            #expect(ContextAdmission.availability(config: config,
                                                  maxContext: context,
                                                  hostMemoryBytes: sixteenGigabyteHost)
                    == .available,
                    "context \(context)")
        }
    }

    /// The rule admits at exactly the minimum, not one byte above it: an
    /// off-by-one here would hide 128K from every 8 GB Mac, whose 270 MB of
    /// margin is the whole question step 9 of the plan measures.
    @Test func admissionBoundaryIsInclusive() {
        let minimum = ContextAdmission.minimumHostMemoryBytes(config: config,
                                                              maxContext: 131_072)
        #expect(ContextAdmission.availability(config: config,
                                              maxContext: 131_072,
                                              hostMemoryBytes: minimum - 1)
                == .needsMemory(minimumHostBytes: minimum))
        #expect(ContextAdmission.availability(config: config,
                                              maxContext: 131_072,
                                              hostMemoryBytes: minimum)
                == .available)
    }

    @Test func projectedFootprintIsKVPlusTheMeasuredNonKVRuntime() {
        let projected = ContextAdmission.projectedFootprintBytes(config: config,
                                                                 maxContext: 262_144)
        #expect(projected == 5_635_768_320 + ContextAdmission.nonKVRuntimeFootprintBytes)
        #expect(ContextAdmission.minimumHostMemoryBytes(config: config, maxContext: 262_144)
                == projected + ContextAdmission.hostReserveBytes)
    }

    @Test func needDescriptionNamesBothTheRequirementAndTheHost() {
        let message = ContextAdmission.needDescription(config: config,
                                                       maxContext: 262_144,
                                                       hostMemoryBytes: eightGigabyteHost)
        #expect(message.contains("16 GB"))
        #expect(message.contains("8 GB"))
        #expect(message.contains("262,144"))
    }

    /// 128K needs 8,320,122,880 bytes, which an 8 GB Mac has. Reporting that
    /// as "needs 16 GB" while `availability` calls it available would put the
    /// caption and the menu in direct contradiction.
    @Test func needDescriptionRoundsToASizeThatActuallyBacksTheContext() {
        let message = ContextAdmission.needDescription(config: config,
                                                       maxContext: 131_072,
                                                       hostMemoryBytes: eightGigabyteHost)
        #expect(message.contains("needs 8 GB"))
    }

    @Test func contextCeilingComesFromTheCheckpoint() {
        #expect(config.maxPositionEmbeddings == 262_144)
    }

    // MARK: - LFM2.5 profile

    private let lfm = ArchConfig.lfm2_5_24B_A2B

    @Test func lfmAdmissionMatchesDesignTable() {
        let config = ArchConfig.lfm2_5_24B_A2B
        #expect(ContextAdmission.fp16KVBytes(config: config, maxContext: 8_192) == 167_772_160)
        #expect(ContextAdmission.fp16KVBytes(config: config, maxContext: 131_072) == 2_684_354_560)
        let projected = ContextAdmission.projectedFootprintBytes(
            config: config, maxContext: 8_192, expertCacheSlots: 8)
        // 167,772,160 KV + 2,147,483,648 non-KV baseline
        // + 8 slots × 38 layers × 5,308,416 = 3,929,014,272.
        #expect(projected == 3_929_014_272)
    }

    /// Thirty of LFM's forty layers are convolutions. Only the ten full
    /// attention layers carry K/V, so the whole context slope is 20,480 B per
    /// token — 8 KV heads × 64 head dim × 2 bytes × K and V × 10 layers.
    @Test func lfmConvLayersContributeNoKV() {
        let small = ContextAdmission.fp16KVBytes(config: lfm, maxContext: 8_192)
        let large = ContextAdmission.fp16KVBytes(config: lfm, maxContext: 16_384)
        #expect(large - small == UInt64((16_384 - 8_192) * 20_480))
        #expect(ContextAdmission.fp16KVBytes(config: lfm, maxContext: 8_192)
                == UInt64(10 * 8_192 * 8 * 64 * 2 * 2))
    }

    /// No sliding window means no ring at all. Before this, the arithmetic
    /// charged 30 conv layers a 1,304-row ring they never allocate.
    @Test func lfmHasNoSlidingRingRows() {
        #expect(ContextAdmission.productionSlidingRingRows(config: lfm, maxContext: 8_192) == 0)
        #expect(ContextAdmission.productionSlidingRingRows(config: lfm, maxContext: 131_072) == 0)
    }

    /// LFM includes no expert-cache slots in its baseline, so every slot is
    /// charged: slots × 38 routed layers × 5,308,416 bytes.
    @Test func lfmExpertCacheIsChargedFromSlotZero() {
        let config = ArchConfig.lfm2_5_24B_A2B
        #expect(ContextAdmission.additionalExpertCacheBytes(config: config, slots: 0) == 0)
        #expect(ContextAdmission.additionalExpertCacheBytes(config: config, slots: 8)
                == 8 * 38 * 5_308_416)
        #expect(ContextAdmission.additionalExpertCacheBytes(config: config, slots: 16)
                == 16 * 38 * 5_308_416)
    }

    @Test func lfmProjectedFootprintUsesTheProfileBaseline() {
        let kv = ContextAdmission.fp16KVBytes(config: lfm, maxContext: 131_072)
        let experts = ContextAdmission.additionalExpertCacheBytes(config: lfm, slots: 16)
        #expect(ContextAdmission.projectedFootprintBytes(config: lfm, maxContext: 131_072,
                                                         expertCacheSlots: 16)
                == kv + lfm.admission.nonKVRuntimeFootprintBytes + experts)
    }

    /// Gemma's admission arithmetic is the measured baseline: the same 8K KV
    /// bytes, the same 2 GiB non-KV allowance, and no charge for the 16 slots
    /// its baseline already includes.
    @Test func gemmaAdmissionBytesAreUnchanged() {
        let config = ArchConfig.gemma4_26B_A4B
        #expect(ContextAdmission.fp16KVBytes(config: config, maxContext: 8_192)
                == 434_831_360)
        #expect(ContextAdmission.additionalExpertCacheBytes(config: config, slots: 16) == 0)
        #expect(ContextAdmission.additionalExpertCacheBytes(config: config, slots: 0) == 0)
        #expect(config.admission == .gemma4)
        #expect(config.admission.nonKVRuntimeFootprintBytes
                == ContextAdmission.nonKVRuntimeFootprintBytes)
    }
}
