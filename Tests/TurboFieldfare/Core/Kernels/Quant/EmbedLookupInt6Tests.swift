import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Runs the 6-bit embedding lookup and returns the FP16 output widened to FP32.
/// `tableOffset` binds the packed table at a non-zero byte offset (the layout is
/// byte-addressed).
func runEmbedInt6(_ ctx: MetalContext,
                  _ packed: [UInt8], _ scales: [UInt16], _ biases: [UInt16],
                  token: Int, rows: Int, columns: Int,
                  outScale: Float = 1.0,
                  tableOffset: Int = 0) throws -> [Float] {
    precondition(token >= 0 && token < rows, "token \(token) outside \(rows) rows")
    let kernel = try EmbedLookupInt6(context: ctx)
    var padded = [UInt8](repeating: 0, count: packed.count + tableOffset)
    for i in 0..<packed.count { padded[tableOffset + i] = packed[i] }
    guard let tableBuf = ctx.device.makeBuffer(
            bytes: padded, length: padded.count, options: .storageModeShared),
          let sBuf = ctx.device.makeBuffer(
            bytes: scales, length: scales.count * MemoryLayout<UInt16>.size,
            options: .storageModeShared),
          let bBuf = ctx.device.makeBuffer(
            bytes: biases, length: biases.count * MemoryLayout<UInt16>.size,
            options: .storageModeShared),
          let outBuf = Fp16Buffer.make(ctx.device, count: columns),
          let cmd = ctx.queue.makeCommandBuffer() else {
        throw Int6TestError.bufferAllocation
    }
    kernel.encode(commandBuffer: cmd,
                  table: tableBuf, tableOffset: tableOffset,
                  scales: sBuf, biases: bBuf,
                  out: outBuf,
                  tokenId: UInt32(token), d: UInt32(columns),
                  outScale: outScale)
    cmd.commit()
    cmd.waitUntilCompleted()
    try checkCommandBufferError(cmd)
    return Fp16Buffer.read(outBuf, count: columns)
}

/// Validates the tied 6-bit embedding lookup used by LFM2.5.
@Suite struct EmbedLookupInt6Tests {

    private static let sharedContext = try! MetalContext()

    /// Brief case: the selected row must equal the dequantized reference row.
    @Test func embeddingLookupMatchesDequantizedRow() throws {
        let ctx = try MetalContext()
        let (packed, scales, biases) = makeRandomInt6(rows: 5, columns: 128, seed: 2)
        let out = try runEmbedInt6(ctx, packed, scales, biases,
                                   token: 3, rows: 5, columns: 128)
        let expected = cpuDequantRow(packed, scales, biases, row: 3, columns: 128)
        for i in 0..<128 { #expect(abs(out[i] - expected[i]) < 0.01) }
    }

    /// Hand-computed bytes for the values `[1, 2, 3, 4]` (`0x81, 0x30, 0x10`)
    /// with unit scale and zero bias: the kernel must return them verbatim. This
    /// pins the decode against the packing convention, not against a helper.
    @Test func embeddingLookupDecodesHandPackedTriple() throws {
        let columns = 64
        var table = [UInt8](repeating: 0, count: int6RowBytes(columns))
        table[0] = 0x81
        table[1] = 0x30
        table[2] = 0x10
        let scales = [int6BF16Bits(1.0)]
        let biases = [int6BF16Bits(0.0)]

        let out = try runEmbedInt6(Self.sharedContext, table, scales, biases,
                                   token: 0, rows: 1, columns: columns)
        #expect(out[0] == 1)
        #expect(out[1] == 2)
        #expect(out[2] == 3)
        #expect(out[3] == 4)
        for i in 4..<columns { #expect(out[i] == 0) }
    }

    /// Row selection, the `sqrt(hidden)` post-embedding scale, and a table bound
    /// at an odd byte offset.
    @Test func embedLookup_scaledLastRowAtOddTableOffset() throws {
        let rows = 5, columns = 320
        let (packed, scales, biases) = makeRandomInt6(rows: rows, columns: columns, seed: 0xE6)
        let token = rows - 1
        let outScale = Float(columns).squareRoot()

        let out = try runEmbedInt6(Self.sharedContext, packed, scales, biases,
                                   token: token, rows: rows, columns: columns,
                                   outScale: outScale, tableOffset: 3)
        let raw = cpuDequantRow(packed, scales, biases, row: token, columns: columns)
        for i in 0..<columns {
            let expected = raw[i] * outScale
            #expect(abs(out[i] - expected) < 0.01,
                    "index \(i): actual=\(out[i]) expected=\(expected)")
        }
    }

    /// Each token decodes its own row: two different tokens over the same table
    /// must return different vectors matching their references.
    @Test func embedLookup_distinguishesRows() throws {
        let rows = 5, columns = 128
        let (packed, scales, biases) = makeRandomInt6(rows: rows, columns: columns, seed: 0xE7)

        for token in [0, 2, 4] {
            let out = try runEmbedInt6(Self.sharedContext, packed, scales, biases,
                                       token: token, rows: rows, columns: columns)
            let expected = cpuDequantRow(packed, scales, biases, row: token, columns: columns)
            for i in 0..<columns {
                #expect(abs(out[i] - expected[i]) < 0.01,
                        "token \(token) index \(i)")
            }
        }
    }
}
