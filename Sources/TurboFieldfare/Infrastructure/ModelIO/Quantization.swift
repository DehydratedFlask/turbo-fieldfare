import Foundation

public enum Quantization {

    public static let groupSize: Int = 64

    // MARK: - BF16 helpers
    //
    // BF16 = top 16 bits of FP32 (8-bit exponent, 7-bit mantissa). Stored on
    // disk and on the GPU as the native `bfloat` type; in Swift we carry the
    // raw bits as `UInt16` and convert via this pair. Round-half-to-even on
    // encode matches the IEEE-754 default. No NaN/Inf special-case: fixtures
    // are bounded and finite, and the .gturbo importer copies BF16 bytes
    // through unchanged so we never call the encoder on weight data.

    @inline(__always)
    public static func bf16Bits(_ x: Float) -> UInt16 {
        let bits = x.bitPattern
        let lsb  = (bits >> 16) & 1
        let roundingBias: UInt32 = 0x7FFF &+ lsb
        let rounded = (bits &+ roundingBias) >> 16
        return UInt16(truncatingIfNeeded: rounded)
    }

    @inline(__always)
    public static func bf16ToFloat(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    // MARK: - Affine payload sizes

    /// Byte sizes of one affine-quantized projection whose weights are stored
    /// as packed `UInt32` columns.
    public struct AffineSizes: Sendable, Equatable {
        /// Packed `UInt32` columns per row (`columns * bits / 32`).
        public let packedColumns: Int
        /// Packed weight payload bytes (`rows * packedColumns * 4`).
        public let weightBytes: UInt64
        /// Bytes of each BF16 companion array (`rows * (columns / groupSize) * 2`).
        /// Scales and biases each own one array of this size.
        public let auxBytes: UInt64
    }

    public enum AffineError: Error, Equatable {
        case unsupportedBits(Int)
        case columnsNotDivisible(columns: Int, groupSize: Int)
        case packedColumnsNotIntegral(columns: Int, bits: Int)
    }

    /// Size arithmetic for 4-, 6-, and 8-bit affine payloads.
    ///
    /// `packedColumns` must divide exactly: a packed source width is only a
    /// valid `UInt32` column count when `columns * bits` is a multiple of 32.
    /// A non-positive group size or a negative dimension fails closed as
    /// `columnsNotDivisible`, and multiplicative overflow is reported through
    /// `packedColumnsNotIntegral` — the shape cannot be represented as packed
    /// columns either way.
    public static func affineSizes(rows: Int,
                                   columns: Int,
                                   bits: Int,
                                   groupSize: Int) throws -> AffineSizes {
        guard bits == 4 || bits == 6 || bits == 8 else {
            throw AffineError.unsupportedBits(bits)
        }
        guard groupSize > 0, rows >= 0, columns >= 0,
              columns % groupSize == 0 else {
            throw AffineError.columnsNotDivisible(columns: columns,
                                                  groupSize: groupSize)
        }
        let rowsU = UInt64(rows)
        let (scaledColumns, scaledOverflow) = UInt64(columns)
            .multipliedReportingOverflow(by: UInt64(bits))
        guard !scaledOverflow, scaledColumns % 32 == 0 else {
            throw AffineError.packedColumnsNotIntegral(columns: columns, bits: bits)
        }
        let packedColumns = scaledColumns / 32
        let groups = UInt64(columns / groupSize)

        let (weightElements, weightOverflow) = rowsU
            .multipliedReportingOverflow(by: packedColumns)
        let (weightBytes, weightBytesOverflow) = weightElements
            .multipliedReportingOverflow(by: 4)
        let (auxElements, auxOverflow) = rowsU.multipliedReportingOverflow(by: groups)
        let (auxBytes, auxBytesOverflow) = auxElements
            .multipliedReportingOverflow(by: 2)
        guard !weightOverflow, !weightBytesOverflow,
              !auxOverflow, !auxBytesOverflow,
              let packed = Int(exactly: packedColumns) else {
            throw AffineError.packedColumnsNotIntegral(columns: columns, bits: bits)
        }
        return AffineSizes(packedColumns: packed,
                           weightBytes: weightBytes,
                           auxBytes: auxBytes)
    }

    /// Byte alignment required of a packed affine weight payload. 4-bit
    /// nibbles need 2-byte alignment and 6-bit packed columns are read as
    /// `UInt32`, so they need 4. 8-bit payloads stay byte-addressed (one byte
    /// per logical element), and existing Gemma packs place them at offsets
    /// that are only 2-byte aligned because a 2-byte `layer_scalar` precedes
    /// the next layer — see `docs/SYSTEM_DESIGN.md`. Widths outside 4/6/8
    /// report the strictest value; callers validate widths through
    /// `affineSizes` first.
    public static func affineWeightAlignment(bits: Int) -> Int {
        switch bits {
        case 4:  return MemoryLayout<UInt16>.alignment
        case 6:  return MemoryLayout<UInt32>.alignment
        case 8:  return MemoryLayout<UInt8>.alignment
        default: return MemoryLayout<UInt32>.alignment
        }
    }

    // MARK: - INT4 affine

    /// MLX `affine` 4-bit row. Packed unsigned nibbles, BF16 scale + bias per
    /// group of 64. `scales` and `biases` carry BF16 bit patterns as `UInt16`
    /// so the same buffer can be uploaded to a Metal `device const bfloat*`.
    public struct Int4AffineRow {
        public let packed: [UInt8]   // N / 2 bytes; low nibble = even index, high = odd
        public let scales: [UInt16]  // N / 64 BF16 bits
        public let biases: [UInt16]  // N / 64 BF16 bits

        public init(packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
            self.packed = packed
            self.scales = scales
            self.biases = biases
        }
    }

    /// Affine 4-bit quantize: `q ∈ [0..15]`, `w ≈ q * scale + bias`.
    /// Scale and bias are computed from per-group min/max, then rounded to BF16.
    /// Test-fixture only — the runtime importer never calls this.
    public static func quantizeInt4Affine(_ row: [Float]) -> Int4AffineRow {
        precondition(row.count % groupSize == 0,
                     "row length \(row.count) is not a multiple of \(groupSize)")

        let nGroups = row.count / groupSize
        var packed = [UInt8](repeating: 0, count: row.count / 2)
        var scales = [UInt16](repeating: 0, count: nGroups)
        var biases = [UInt16](repeating: 0, count: nGroups)

        for g in 0..<nGroups {
            var wmin: Float =  .infinity
            var wmax: Float = -.infinity
            for k in 0..<groupSize {
                let w = row[g * groupSize + k]
                if w < wmin { wmin = w }
                if w > wmax { wmax = w }
            }
            // Constant group: scale=1, bias=value preserves exact reconstruction.
            let scaleF: Float
            let biasF:  Float
            if wmax == wmin {
                scaleF = 1
                biasF  = wmin
            } else {
                scaleF = (wmax - wmin) / 15.0
                biasF  = wmin
            }
            // Round through BF16 first, then quantize against the rounded
            // values so the runtime decode (which reads BF16) reproduces the
            // same q the fixture stored.
            let sBits = bf16Bits(scaleF)
            let bBits = bf16Bits(biasF)
            scales[g] = sBits
            biases[g] = bBits
            let scale = bf16ToFloat(sBits)
            let bias  = bf16ToFloat(bBits)
            let invScale = scale == 0 ? Float(0) : 1.0 / scale

            for k in 0..<groupSize {
                let w = row[g * groupSize + k]
                var q = Int(((w - bias) * invScale).rounded())
                q = max(0, min(15, q))
                let nibble = UInt8(q) & 0x0F
                let byteIdx = g * (groupSize / 2) + (k / 2)
                if (k & 1) == 0 {
                    packed[byteIdx] = (packed[byteIdx] & 0xF0) | nibble
                } else {
                    packed[byteIdx] = (packed[byteIdx] & 0x0F) | (nibble << 4)
                }
            }
        }
        return Int4AffineRow(packed: packed, scales: scales, biases: biases)
    }

    public static func dequantizeInt4Affine(_ r: Int4AffineRow, n: Int) -> [Float] {
        precondition(n == r.packed.count * 2)
        var out = [Float](repeating: 0, count: n)
        let nGroups = n / groupSize
        for g in 0..<nGroups {
            let scale = bf16ToFloat(r.scales[g])
            let bias  = bf16ToFloat(r.biases[g])
            for k in 0..<groupSize {
                let byteIdx = g * (groupSize / 2) + (k / 2)
                let b = r.packed[byteIdx]
                let nibble: Int = (k & 1) == 0 ? Int(b & 0x0F) : Int(b >> 4)
                out[g * groupSize + k] = Float(nibble) * scale + bias
            }
        }
        return out
    }

    // MARK: - INT8 affine

    public struct Int8AffineRow {
        public let packed: [UInt8]   // N unsigned bytes
        public let scales: [UInt16]  // N / 64 BF16 bits
        public let biases: [UInt16]  // N / 64 BF16 bits

        public init(packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
            self.packed = packed
            self.scales = scales
            self.biases = biases
        }
    }

    public static func quantizeInt8Affine(_ row: [Float]) -> Int8AffineRow {
        precondition(row.count % groupSize == 0,
                     "row length \(row.count) is not a multiple of \(groupSize)")

        let nGroups = row.count / groupSize
        var packed = [UInt8](repeating: 0, count: row.count)
        var scales = [UInt16](repeating: 0, count: nGroups)
        var biases = [UInt16](repeating: 0, count: nGroups)

        for g in 0..<nGroups {
            var wmin: Float =  .infinity
            var wmax: Float = -.infinity
            for k in 0..<groupSize {
                let w = row[g * groupSize + k]
                if w < wmin { wmin = w }
                if w > wmax { wmax = w }
            }
            let scaleF: Float
            let biasF:  Float
            if wmax == wmin {
                scaleF = 1
                biasF  = wmin
            } else {
                scaleF = (wmax - wmin) / 255.0
                biasF  = wmin
            }
            let sBits = bf16Bits(scaleF)
            let bBits = bf16Bits(biasF)
            scales[g] = sBits
            biases[g] = bBits
            let scale = bf16ToFloat(sBits)
            let bias  = bf16ToFloat(bBits)
            let invScale = scale == 0 ? Float(0) : 1.0 / scale

            for k in 0..<groupSize {
                let w = row[g * groupSize + k]
                var q = Int(((w - bias) * invScale).rounded())
                q = max(0, min(255, q))
                packed[g * groupSize + k] = UInt8(q)
            }
        }
        return Int8AffineRow(packed: packed, scales: scales, biases: biases)
    }

    public static func dequantizeInt8Affine(_ r: Int8AffineRow, n: Int) -> [Float] {
        precondition(n == r.packed.count)
        var out = [Float](repeating: 0, count: n)
        let nGroups = n / groupSize
        for g in 0..<nGroups {
            let scale = bf16ToFloat(r.scales[g])
            let bias  = bf16ToFloat(r.biases[g])
            for k in 0..<groupSize {
                out[g * groupSize + k] = Float(r.packed[g * groupSize + k]) * scale + bias
            }
        }
        return out
    }
}
