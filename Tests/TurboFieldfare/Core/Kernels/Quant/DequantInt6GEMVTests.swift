import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

// MARK: - Independent MLX 6-bit fixtures
//
// `packInt6`/`unpackInt6`/`cpuInt6GEMV`/`cpuDequantRow` (the latter lives here
// and is consumed by `EmbedLookupInt6Tests`) implement the four-values-per-
// three-bytes, least-significant-bits-first layout from design §8.4 from
// scratch. They deliberately call neither the Metal kernels, the Swift
// wrappers, nor `Quantization`, so every kernel assertion is against a second,
// independent implementation. `makeRandomInt6`/`makeFP16`/`int6BF16Bits` are
// shared with `EmbedLookupInt6Tests`.

/// BF16 encode, round-half-to-even. Test-local so references never borrow the
/// production conversion they validate.
func int6BF16Bits(_ x: Float) -> UInt16 {
    let bits = x.bitPattern
    let lsb = (bits >> 16) & 1
    let rounded = (bits &+ (0x7FFF &+ lsb)) >> 16
    return UInt16(truncatingIfNeeded: rounded)
}

/// BF16 decode: the top 16 bits of the FP32 pattern.
func int6BF16Float(_ bits: UInt16) -> Float {
    Float(bitPattern: UInt32(bits) << 16)
}

/// Packed bytes in one row of `columns` 6-bit values.
func int6RowBytes(_ columns: Int) -> Int {
    precondition(columns % 4 == 0, "6-bit packing consumes four values per triple")
    return columns / 4 * 3
}

/// Packs 6-bit values into bytes, four values per three bytes, LSB-first:
/// `b0 = v0 | v1 << 6`, `b1 = v1 >> 2 | v2 << 4`, `b2 = v2 >> 4 | v3 << 2`.
func packInt6(_ values: [UInt8]) -> [UInt8] {
    precondition(values.count % 4 == 0, "6-bit packing consumes four values per triple")
    var out = [UInt8](repeating: 0, count: values.count / 4 * 3)
    for i in stride(from: 0, to: values.count, by: 4) {
        let v0 = UInt32(values[i] & 0x3f)
        let v1 = UInt32(values[i + 1] & 0x3f)
        let v2 = UInt32(values[i + 2] & 0x3f)
        let v3 = UInt32(values[i + 3] & 0x3f)
        let o = i / 4 * 3
        out[o]     = UInt8((v0 | (v1 << 6)) & 0xff)
        out[o + 1] = UInt8(((v1 >> 2) | (v2 << 4)) & 0xff)
        out[o + 2] = UInt8(((v2 >> 4) | (v3 << 2)) & 0xff)
    }
    return out
}

/// Inverse of `packInt6`, written from the same three formulas.
func unpackInt6(_ bytes: [UInt8]) -> [UInt8] {
    precondition(bytes.count % 3 == 0, "6-bit packing consumes four values per triple")
    var out = [UInt8]()
    out.reserveCapacity(bytes.count / 3 * 4)
    for i in stride(from: 0, to: bytes.count, by: 3) {
        let b0 = UInt32(bytes[i])
        let b1 = UInt32(bytes[i + 1])
        let b2 = UInt32(bytes[i + 2])
        out.append(UInt8(b0 & 0x3f))
        out.append(UInt8(((b0 >> 6) & 0x03) | ((b1 & 0x0f) << 2)))
        out.append(UInt8(((b1 >> 4) & 0x0f) | ((b2 & 0x03) << 4)))
        out.append(UInt8((b2 >> 2) & 0x3f))
    }
    return out
}

/// Random 6-bit weights with BF16 per-group scales/biases, row-major, `columns`
/// wide. Magnitudes are small so an FP16 result stays inside a fixed absolute
/// tolerance against the FP32 reference.
func makeRandomInt6(rows: Int, columns: Int, seed: UInt64)
    -> (packed: [UInt8], scales: [UInt16], biases: [UInt16])
{
    precondition(columns % Quantization.groupSize == 0)
    var rng = SeedTree(seed).key("int6-kernel-rows\(rows)-cols\(columns)")
    let groupsPerRow = columns / Quantization.groupSize
    var packed = [UInt8]()
    packed.reserveCapacity(rows * int6RowBytes(columns))
    var scales = [UInt16](repeating: 0, count: rows * groupsPerRow)
    var biases = [UInt16](repeating: 0, count: rows * groupsPerRow)
    for row in 0..<rows {
        let values = (0..<columns).map { _ in UInt8(rng.next() % 64) }
        packed.append(contentsOf: packInt6(values))
        for g in 0..<groupsPerRow {
            scales[row * groupsPerRow + g] = int6BF16Bits(rng.uniform(-0.002, 0.002))
            biases[row * groupsPerRow + g] = int6BF16Bits(rng.uniform(-0.001, 0.001))
        }
    }
    return (packed, scales, biases)
}

/// FP16 vector fixture: one rounding point shared by the GPU binding and the
/// CPU reference so both see identical inputs.
func makeFP16(_ values: [Float]) -> [Float16] {
    values.map { Float16($0) }
}

/// FP32 reference `y = W * x` over packed 6-bit rows: bulk-dequantize a row,
/// then dot in FP32. Different staging and summation order from the kernel.
func cpuInt6GEMV(_ packed: [UInt8], _ scales: [UInt16], _ biases: [UInt16],
                 _ x: [Float16], rows: Int, columns: Int) -> [Float] {
    precondition(x.count == columns)
    let rowBytes = int6RowBytes(columns)
    let groupsPerRow = columns / Quantization.groupSize
    var y = [Float](repeating: 0, count: rows)
    for row in 0..<rows {
        let start = row * rowBytes
        let values = unpackInt6(Array(packed[start..<(start + rowBytes)]))
        var acc: Float = 0
        for i in 0..<columns {
            let g = i / Quantization.groupSize
            let s = int6BF16Float(scales[row * groupsPerRow + g])
            let b = int6BF16Float(biases[row * groupsPerRow + g])
            acc += (Float(values[i]) * s + b) * Float(x[i])
        }
        y[row] = acc
    }
    return y
}

/// FP32 reference for one dequantized embedding row.
func cpuDequantRow(_ packed: [UInt8], _ scales: [UInt16], _ biases: [UInt16],
                   row: Int, columns: Int) -> [Float] {
    let rowBytes = int6RowBytes(columns)
    let groupsPerRow = columns / Quantization.groupSize
    let start = row * rowBytes
    let values = unpackInt6(Array(packed[start..<(start + rowBytes)]))
    var out = [Float](repeating: 0, count: columns)
    for i in 0..<columns {
        let g = i / Quantization.groupSize
        out[i] = Float(values[i]) * int6BF16Float(scales[row * groupsPerRow + g])
               + int6BF16Float(biases[row * groupsPerRow + g])
    }
    return out
}

enum Int6TestError: Error {
    case bufferAllocation
}

/// Runs the kernel and returns the FP16 output widened to FP32. `weightsOffset`
/// pads the packed buffer and binds it at that byte offset: the 6-bit payload is
/// byte-addressed, so odd offsets must work.
func runInt6GEMV(_ ctx: MetalContext,
                 _ packed: [UInt8], _ scales: [UInt16], _ biases: [UInt16],
                 _ x: [Float16],
                 rows: Int, columns: Int,
                 weightsOffset: Int = 0) throws -> [Float] {
    let kernel = try DequantInt6GEMV(context: ctx)
    var padded = [UInt8](repeating: 0, count: packed.count + weightsOffset)
    for i in 0..<packed.count { padded[weightsOffset + i] = packed[i] }
    guard let wBuf = ctx.device.makeBuffer(
            bytes: padded, length: padded.count, options: .storageModeShared),
          let sBuf = ctx.device.makeBuffer(
            bytes: scales, length: scales.count * MemoryLayout<UInt16>.size,
            options: .storageModeShared),
          let bBuf = ctx.device.makeBuffer(
            bytes: biases, length: biases.count * MemoryLayout<UInt16>.size,
            options: .storageModeShared),
          let xBuf = Fp16Buffer.make(ctx.device, halves: x),
          let yBuf = Fp16Buffer.make(ctx.device, count: rows),
          let cmd = ctx.queue.makeCommandBuffer() else {
        throw Int6TestError.bufferAllocation
    }
    kernel.encode(commandBuffer: cmd,
                  weights: wBuf, weightsOffset: weightsOffset,
                  scales: sBuf, biases: bBuf,
                  x: xBuf, y: yBuf,
                  m: UInt32(rows), n: UInt32(columns))
    cmd.commit()
    cmd.waitUntilCompleted()
    try checkCommandBufferError(cmd)
    return Fp16Buffer.read(yBuf, count: rows)
}

/// Compares against the FP32 reference under an FP16-storage-aware bar: the
/// kernel writes FP16, so the allowed absolute error scales with magnitude.
private func expectInt6Matches(_ actual: [Float], _ expected: [Float],
                               _ label: String,
                               tolerance: Float = 1e-2) {
    precondition(actual.count == expected.count, "length mismatch")
    for i in 0..<actual.count {
        let bound = tolerance * max(1, abs(expected[i]))
        #expect(abs(actual[i] - expected[i]) <= bound,
                "\(label)[\(i)]: actual=\(actual[i]) expected=\(expected[i])")
    }
}

/// Compares the Metal 6-bit GEMV against the independent CPU reference.
///
/// The kernel decodes one 3-byte triple (four values) per lane within a 64-wide
/// group and one threadgroup per output row, so both row counts and group counts
/// that leave partial threadgroups and odd group counts are covered.
@Suite struct DequantInt6GEMVTests {

    private static let sharedContext = try! MetalContext()

    /// Brief case: seven output rows of 128 columns.
    @Test func sixBitGemvMatchesCPUReference() throws {
        let ctx = try MetalContext()
        let rows = 7, columns = 128
        let (packed, scales, biases) = makeRandomInt6(rows: rows, columns: columns, seed: 1)
        let x = makeFP16(Array(repeating: Float(0.25), count: columns))
        let out = try runInt6GEMV(ctx, packed, scales, biases, x, rows: rows, columns: columns)
        let expected = cpuInt6GEMV(packed, scales, biases, x, rows: rows, columns: columns)
        for i in 0..<rows { #expect(abs(out[i] - expected[i]) < 0.01) }
    }

    /// The packing convention itself: LSB-first, four values per three bytes.
    @Test func sixBitPackingUsesLeastSignificantBitsFirst() {
        let values: [UInt8] = [1, 2, 3, 4]
        let bytes = packInt6(values)
        #expect(bytes == [0x81, 0x30, 0x10])
        #expect(unpackInt6(bytes) == values)
    }

    /// Every 6-bit value round-trips, including 0x3f whose high bits are carried
    /// across the byte boundary.
    @Test func sixBitPackingRoundTripsAllValues() {
        let values = (0..<256).map { UInt8($0 % 64) }
        #expect(unpackInt6(packInt6(values)) == values)
    }

    /// Rows and columns that are not a multiple of the threadgroup geometry:
    /// a threadgroup is one row of 32 lanes, each lane decoding a 3-byte triple.
    @Test(arguments: [1, 5, 7, 33] as [Int],
                     [64, 192, 320] as [Int])
    func gemv_sweep(rows: Int, columns: Int) throws {
        let seed = UInt64(rows) &* 0x9E37 &+ UInt64(columns)
        let (packed, scales, biases) = makeRandomInt6(rows: rows, columns: columns, seed: seed)
        let x = makeFP16((0..<columns).map { _ in 0.25 })
        let out = try runInt6GEMV(Self.sharedContext, packed, scales, biases, x,
                                  rows: rows, columns: columns)
        let expected = cpuInt6GEMV(packed, scales, biases, x, rows: rows, columns: columns)
        expectInt6Matches(out, expected, "M=\(rows) N=\(columns)")
    }

    /// The 6-bit payload is byte-addressed: a 3-bytes-per-4-values row stride is
    /// only 3-aligned, so odd weight offsets must stay correct.
    @Test(arguments: [1, 3] as [Int])
    func gemv_weightsAtOddByteOffset(weightsOffset: Int) throws {
        let rows = 9, columns = 192
        let (packed, scales, biases) = makeRandomInt6(
            rows: rows, columns: columns, seed: 0x1600 &+ UInt64(weightsOffset))
        let x = makeFP16((0..<columns).map { _ in 0.125 })
        let out = try runInt6GEMV(Self.sharedContext, packed, scales, biases, x,
                                  rows: rows, columns: columns,
                                  weightsOffset: weightsOffset)
        let expected = cpuInt6GEMV(packed, scales, biases, x, rows: rows, columns: columns)
        expectInt6Matches(out, expected, "offset=\(weightsOffset)")
    }

    /// Column mapping probe: with unit scale and zero bias the dot product with a
    /// one-hot vector must return exactly the decoded weight of that column. The
    /// probe set crosses triple boundaries (0..5), a group boundary (63/64) and
    /// the last column.
    @Test func gemv_oneHotRecoversDecodedColumn() throws {
        let rows = 1, columns = 128
        let values = (0..<columns).map { UInt8($0 % 64) }
        let packed = packInt6(values)
        let scales = [UInt16](repeating: int6BF16Bits(1.0), count: columns / 64)
        let biases = [UInt16](repeating: int6BF16Bits(0.0), count: columns / 64)

        for probe in [0, 1, 2, 3, 4, 5, 62, 63, 64, 65, 126, 127] {
            var x = [Float16](repeating: 0, count: columns)
            x[probe] = 1
            let out = try runInt6GEMV(Self.sharedContext, packed, scales, biases, x,
                                      rows: rows, columns: columns)
            #expect(abs(out[0] - Float(probe % 64)) < 1e-3,
                    "column \(probe): got \(out[0]) expected \(probe % 64)")
        }
    }

    /// Multiple rows must not bleed into each other: row `r` of a fixture whose
    /// weights all equal `r` decodes to `r * sum(x)`.
    @Test func gemv_rowsAreIndependent() throws {
        let rows = 7, columns = 64
        // Bias 0, scale 1: each row is a constant field of its own row index.
        let packed = (0..<rows).flatMap { row in
            packInt6([UInt8](repeating: UInt8(row), count: columns))
        }
        let scales = [UInt16](repeating: int6BF16Bits(1.0), count: rows)
        let biases = [UInt16](repeating: int6BF16Bits(0.0), count: rows)
        let x = makeFP16([Float](repeating: 0.25, count: columns))
        let out = try runInt6GEMV(Self.sharedContext, packed, scales, biases, x,
                                  rows: rows, columns: columns)
        for row in 0..<rows {
            #expect(abs(out[row] - Float(row) * 0.25 * Float(columns)) < 1e-2,
                    "row \(row): got \(out[row])")
        }
    }
}
