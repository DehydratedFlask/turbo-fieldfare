import Testing
import Foundation
import TurboFieldfare

@Suite struct QuantizationAffineTests {

    @Test func quantizeDequantizeInt4Affine_roundtrip() {
        var row = [Float]()
        row.reserveCapacity(128)
        for k in 0..<64 { row.append(0.10 + Float(k) * 0.01) }   // 0.10 .. 0.73 (positive-only, bias > 0)
        for k in 0..<64 { row.append(-0.50 - Float(k) * 0.005) } // -0.50 .. -0.815 (negative-only, bias < 0)

        let q = Quantization.quantizeInt4Affine(row)
        #expect(q.packed.count == 64)
        #expect(q.scales.count == 2)
        #expect(q.biases.count == 2)

        let r = Quantization.dequantizeInt4Affine(q, n: 128)
        // 4-bit affine, group-0 scale ≈ (0.73 - 0.10) / 15 ≈ 0.042; group-1 scale
        // ≈ 0.315/15 ≈ 0.021. Worst-case rounding ≤ scale/2 plus BF16 rounding
        // of the scale (rel ≤ 2^-7) — bound under 0.025.
        for i in 0..<128 {
            #expect(abs(r[i] - row[i]) < 0.025,
                    "i=\(i) ref=\(row[i]) got=\(r[i])")
        }
    }

    @Test func quantizeDequantizeInt8Affine_roundtrip() {
        var row = [Float]()
        row.reserveCapacity(64)
        for k in 0..<64 { row.append(-0.3 + Float(k) * 0.02) }   // -0.30 .. +0.96

        let q = Quantization.quantizeInt8Affine(row)
        #expect(q.packed.count == 64)
        #expect(q.scales.count == 1)
        #expect(q.biases.count == 1)

        let r = Quantization.dequantizeInt8Affine(q, n: 64)
        // 8-bit affine, scale ≈ 1.26/255 ≈ 0.005 → rounding ≤ scale/2 ≈ 0.0025;
        // BF16 rounding of scale ≤ 2^-7 → extra ≤ 0.005 * 255/2 * 2^-7 ≈ 0.005.
        // Bound under 0.008 absorbs both contributions.
        for i in 0..<64 {
            #expect(abs(r[i] - row[i]) < 0.008,
                    "i=\(i) ref=\(row[i]) got=\(r[i])")
        }
    }

    @Test func bf16Roundtrip_isLossyButPredictable() {
        let values: [Float] = [0.0, 1.0, -1.0, 0.5, -0.5, 1.0 / 3.0, 12345.0]
        for v in values {
            let bits = Quantization.bf16Bits(v)
            let back = Quantization.bf16ToFloat(bits)
            // BF16 has 7 mantissa bits; rel error ≤ 2^-7 = 7.8e-3.
            let denom = max(abs(v), 1e-6)
            #expect(abs(back - v) / denom < 8e-3,
                    "v=\(v) back=\(back)")
        }
    }

    /// Constant-group sanity: a flat group reproduces exactly.
    @Test func quantizeInt4Affine_constantGroupRoundtripsExact() {
        let row = [Float](repeating: 0.42, count: 64)
        let q = Quantization.quantizeInt4Affine(row)
        let r = Quantization.dequantizeInt4Affine(q, n: 64)
        // Constant-group path stores bias = value, scale = 1; the only error is
        // the BF16 rounding of the bias.
        let bf16Rounded = Quantization.bf16ToFloat(Quantization.bf16Bits(0.42))
        for i in 0..<64 {
            #expect(abs(r[i] - bf16Rounded) < 1e-6,
                    "i=\(i) got=\(r[i]) ref=\(bf16Rounded)")
        }
    }

    // MARK: - Affine size arithmetic

    @Test func sixBitEmbeddingUsesThreeHundredEightyFourPackedColumns() throws {
        let size = try Quantization.affineSizes(rows: 128_000, columns: 2048,
                                                bits: 6, groupSize: 64)
        #expect(size.packedColumns == 384)
        #expect(size.weightBytes == 128_000 * 384 * 4)
        #expect(size.auxBytes == 128_000 * 32 * 2)
    }

    @Test func sixBitPackedColumnCountMustDivideExactly() {
        #expect(throws: Quantization.AffineError.self) {
            _ = try Quantization.affineSizes(rows: 1, columns: 3, bits: 6, groupSize: 64)
        }
    }

    /// 4-bit arithmetic must match the legacy `rows * columns / 2` byte count.
    @Test func fourBitEmbeddingMatchesLegacyByteCount() throws {
        let size = try Quantization.affineSizes(rows: 128_000, columns: 2048,
                                                bits: 4, groupSize: 64)
        #expect(size.packedColumns == 256)
        #expect(size.weightBytes == UInt64(128_000 * 2048 / 2))
        #expect(size.auxBytes == 128_000 * 32 * 2)
    }

    @Test func eightBitSizesUseQuarterWidthColumns() throws {
        let size = try Quantization.affineSizes(rows: 128, columns: 2816,
                                                bits: 8, groupSize: 64)
        #expect(size.packedColumns == 704)
        #expect(size.weightBytes == UInt64(128 * 2816))
        #expect(size.auxBytes == 128 * 44 * 2)
    }

    @Test func affineWeightAlignmentIsTwoFourAndOne() {
        #expect(Quantization.affineWeightAlignment(bits: 4) == 2)
        #expect(Quantization.affineWeightAlignment(bits: 6) == 4)
        // 8-bit payloads stay byte-addressed: existing Gemma packs place the
        // resident 8-bit router payload at offsets that are only 2-byte
        // aligned, so requiring 4 here would reject them.
        #expect(Quantization.affineWeightAlignment(bits: 8) == 1)
    }

    @Test func rejectsUnsupportedAffineBits() {
        #expect(throws: Quantization.AffineError.unsupportedBits(3)) {
            _ = try Quantization.affineSizes(rows: 1, columns: 64, bits: 3, groupSize: 64)
        }
        #expect(throws: Quantization.AffineError.unsupportedBits(16)) {
            _ = try Quantization.affineSizes(rows: 1, columns: 64, bits: 16, groupSize: 64)
        }
    }

    @Test func rejectsNonPositiveGroupSize() {
        #expect(throws: Quantization.AffineError.self) {
            _ = try Quantization.affineSizes(rows: 1, columns: 64, bits: 6, groupSize: 0)
        }
    }

    @Test func rejectsColumnsOutsideTheGroupGrid() {
        #expect(throws: Quantization.AffineError.columnsNotDivisible(columns: 96, groupSize: 64)) {
            _ = try Quantization.affineSizes(rows: 1, columns: 96, bits: 6, groupSize: 64)
        }
    }

    /// A group of 64 always packs 6-bit rows into whole UInt32 columns, but
    /// smaller groups can leave a fractional trailing column.
    @Test func sixBitRequiresIntegralPackedColumns() {
        #expect(throws: Quantization.AffineError.packedColumnsNotIntegral(columns: 4, bits: 6)) {
            _ = try Quantization.affineSizes(rows: 1, columns: 4, bits: 6, groupSize: 4)
        }
    }

    /// Asymmetric range (positive-only with non-zero bias) — the failure mode
    /// the symmetric scheme cannot represent without wasted dynamic range.
    @Test func quantizeInt4Affine_positiveOnlyUsesFullCodebook() {
        let row = (0..<64).map { Float($0) / 63.0 + 1.0 } // 1.0 .. 2.0
        let q = Quantization.quantizeInt4Affine(row)
        var seenLow = false, seenHigh = false
        for b in q.packed {
            let lo = b & 0x0F
            let hi = b >> 4
            if lo == 0 || hi == 0   { seenLow = true }
            if lo == 15 || hi == 15 { seenHigh = true }
        }
        #expect(seenLow,  "affine 4-bit on positive-only range should hit q=0")
        #expect(seenHigh, "affine 4-bit on positive-only range should hit q=15")
    }
}
