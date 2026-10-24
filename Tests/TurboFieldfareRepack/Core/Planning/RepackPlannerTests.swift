import Testing
@testable import TurboFieldfareRepackCore

@Suite struct RepackPlannerTests {

    /// LFM's 6-bit tied embedding/lm-head source width packs 2048 logical
    /// columns into 384 UInt32 columns.
    @Test func sixBitPackedWidthMapsBackToLogicalColumns() throws {
        let wide = try RepackPlanner.logicalShape(
            forPackedSource: [128_000, 384], bits: 6)
        #expect(wide == [128_000, 2048])

        let tall = try RepackPlanner.logicalShape(
            forPackedSource: [2048, 384], bits: 6)
        #expect(tall == [2048, 2048])
    }

    @Test func fourBitMappingIsUnchanged() throws {
        for packed in [1, 8, 64, 704] as [UInt64] {
            let mapped = try RepackPlanner.logicalShape(
                forPackedSource: [7, packed], bits: 4)
            #expect(mapped == [7, packed * 8])
        }
    }

    @Test func eightBitMappingIsUnchanged() throws {
        let mapped = try RepackPlanner.logicalShape(
            forPackedSource: [128, 704], bits: 8)
        #expect(mapped == [128, 2816])
    }

    @Test func nonDivisiblePackedWidthThrows() {
        // 383 * 32 = 12256 is not divisible by 6.
        #expect(throws: RepackError.self) {
            _ = try RepackPlanner.logicalShape(forPackedSource: [128_000, 383], bits: 6)
        }
    }

    @Test func overflowingPackedWidthThrows() {
        #expect(throws: RepackError.self) {
            _ = try RepackPlanner.logicalShape(
                forPackedSource: [1, UInt64.max], bits: 6)
        }
    }

    @Test func nonPositiveBitsThrowsInsteadOfTrapping() {
        #expect(throws: RepackError.self) {
            _ = try RepackPlanner.logicalShape(forPackedSource: [1, 8], bits: 0)
        }
    }

    @Test func emptyShapePassesThrough() throws {
        let mapped = try RepackPlanner.logicalShape(forPackedSource: [], bits: 6)
        #expect(mapped.isEmpty)
    }
}
