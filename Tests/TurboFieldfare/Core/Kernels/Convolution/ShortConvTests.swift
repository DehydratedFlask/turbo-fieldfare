import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// LFM2.5's depthwise causal short convolution, kernel 3, no bias:
///
///   Bx   = B * x
///   conv = w0 * Bx[t-2] + w1 * Bx[t-1] + w2 * Bx[t]
///
/// The cache holds the two previous `Bx` vectors (not conv outputs), oldest
/// first, so `w0` multiplies the oldest row and `w2` the current value. The
/// kernels here own only that two-row state; Task 15's runner owns the rewind
/// ring. Prefill must agree with token-by-token decode because production
/// prefill and decode are separate dispatches over the same arithmetic.

private enum ShortConvTestError: Error {
    case bufferAllocation
}

/// FP16-storage-aware bar: the kernels write FP16, so the allowed error scales
/// with magnitude. The brief's fractional `1e-3` bound is preserved; a plain
/// absolute 1e-3 is below one half-precision ulp near 10 and cannot pass for
/// any conforming kernel.
private func expectShortConvClose(_ actual: Float,
                                  _ expected: Float,
                                  _ label: String) {
    let bound = 1e-3 * max(1, abs(expected))
    #expect(abs(actual - expected) <= bound,
            "\(label): actual=\(actual) expected=\(expected)")
}

/// Deterministic gated values `Bx` in `[-2, 2)`, row-major `[tokens, channels]`.
private func makeBx(tokens: Int, channels: Int, seed: UInt64) -> [Float16] {
    var rng = SeedTree(seed).key("shortconv-tokens\(tokens)-channels\(channels)")
    return (0..<(tokens * channels)).map { _ in Float16(rng.uniform(-2, 2)) }
}

/// Deterministic depthwise taps, layout `[channels, 3]`.
private func makeShortConvWeight(channels: Int, seed: UInt64) -> [Float16] {
    var rng = SeedTree(seed).key("shortconv-weight-channels\(channels)")
    return (0..<(channels * 3)).map { _ in Float16(rng.uniform(-1, 1)) }
}

private func row(_ bx: [Float16], _ t: Int, channels: Int) -> [Float16] {
    Array(bx[(t * channels)..<((t + 1) * channels)])
}

private func zeroState(channels: Int) -> [Float16] {
    [Float16](repeating: 0, count: 2 * channels)
}

/// Allocates `halves` behind `leading` zero halves so the wrapper's byte
/// offsets are exercised.
private func paddedHalfBuffer(_ device: MTLDevice,
                              _ halves: [Float16],
                              leading: Int) -> MTLBuffer? {
    var padded = [Float16](repeating: 0, count: leading + halves.count)
    for i in 0..<halves.count { padded[leading + i] = halves[i] }
    return Fp16Buffer.make(device, halves: padded)
}

private func runShortConvDecode(_ ctx: MetalContext,
                                bx: [Float16],
                                weight: [Float16],
                                state: inout [Float16],
                                channels: UInt32,
                                byteOffset: Int = 0) throws -> [Float] {
    precondition(byteOffset % MemoryLayout<Float16>.size == 0,
                 "byte offset must be half-aligned")
    let kernel = try ShortConv(context: ctx)
    let leading = byteOffset / MemoryLayout<Float16>.size
    let count = Int(channels)
    guard let bxBuf = paddedHalfBuffer(ctx.device, bx, leading: leading),
          let wBuf = paddedHalfBuffer(ctx.device, weight, leading: leading),
          let stateBuf = paddedHalfBuffer(ctx.device, state, leading: leading),
          let outBuf = Fp16Buffer.make(ctx.device, count: leading + count),
          let cmd = ctx.queue.makeCommandBuffer() else {
        throw ShortConvTestError.bufferAllocation
    }
    kernel.encodeDecode(commandBuffer: cmd,
                        bx: bxBuf, bxOffset: byteOffset,
                        weight: wBuf, weightOffset: byteOffset,
                        state: stateBuf, stateOffset: byteOffset,
                        out: outBuf, outOffset: byteOffset,
                        channels: channels)
    cmd.commit()
    cmd.waitUntilCompleted()
    try checkCommandBufferError(cmd)
    let stateRead = Fp16Buffer.readHalf(stateBuf, count: leading + state.count)
    state = Array(stateRead[leading...])
    let outRead = Fp16Buffer.read(outBuf, count: leading + count)
    return Array(outRead[leading...])
}

private func runShortConvPrefill(
    _ ctx: MetalContext,
    _ bx: [Float16],
    _ weight: [Float16],
    initial: [Float16],
    tokens: Int,
    channels: Int,
    byteOffset: Int = 0
) throws -> (out: [Float], stateOut: [Float16], stateInAfter: [Float16]) {
    precondition(byteOffset % MemoryLayout<Float16>.size == 0,
                 "byte offset must be half-aligned")
    let kernel = try ShortConv(context: ctx)
    let leading = byteOffset / MemoryLayout<Float16>.size
    let outCount = tokens * channels
    guard let bxBuf = paddedHalfBuffer(ctx.device, bx, leading: leading),
          let wBuf = paddedHalfBuffer(ctx.device, weight, leading: leading),
          let stateInBuf = paddedHalfBuffer(ctx.device, initial, leading: leading),
          let outBuf = Fp16Buffer.make(ctx.device, count: leading + outCount),
          let stateOutBuf = Fp16Buffer.make(ctx.device, count: leading + 2 * channels),
          let cmd = ctx.queue.makeCommandBuffer() else {
        throw ShortConvTestError.bufferAllocation
    }
    kernel.encodePrefill(commandBuffer: cmd,
                         bx: bxBuf, bxOffset: byteOffset,
                         weight: wBuf, weightOffset: byteOffset,
                         stateIn: stateInBuf, stateInOffset: byteOffset,
                         out: outBuf, outOffset: byteOffset,
                         stateOut: stateOutBuf, stateOutOffset: byteOffset,
                         tokens: UInt32(tokens), channels: UInt32(channels))
    cmd.commit()
    cmd.waitUntilCompleted()
    try checkCommandBufferError(cmd)
    let outRead = Fp16Buffer.read(outBuf, count: leading + outCount)
    let stateOutRead = Fp16Buffer.readHalf(stateOutBuf, count: leading + 2 * channels)
    let stateInRead = Fp16Buffer.readHalf(stateInBuf, count: leading + initial.count)
    return (Array(outRead[leading...]),
            Array(stateOutRead[leading...]),
            Array(stateInRead[leading...]))
}

@Suite struct ShortConvTests {

    private static let sharedContext = try! MetalContext()

    /// Brief case. `w0` reads the oldest state row, `w1` the previous row, `w2`
    /// the current gated value; the state then shifts by one position.
    @Test func decodeUsesTwoPreviousGatedValues() throws {
        let ctx = Self.sharedContext
        let channels = 4
        let weight: [Float16] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        var state: [Float16] = [1, 2, 3, 4, 5, 6, 7, 8]  // two rows
        let bx: [Float16] = [9, 10, 11, 12]
        let oldState = state
        let out = try runShortConvDecode(ctx, bx: bx, weight: weight,
                                         state: &state, channels: UInt32(channels))
        for c in 0..<channels {
            let expected = Float(weight[c * 3]) * Float(oldState[c])
                + Float(weight[c * 3 + 1]) * Float(oldState[channels + c])
                + Float(weight[c * 3 + 2]) * Float(bx[c])
            expectShortConvClose(out[c], expected, "channel \(c)")
        }
        #expect(state == [5, 6, 7, 8, 9, 10, 11, 12])
    }

    /// Each tap maps to exactly one slot: a one-hot weight exposes the state row
    /// (or current value) that slot must read.
    @Test func decodeTapOrderIsOldestToCurrent() throws {
        let ctx = Self.sharedContext
        let channels = 3
        let state: [Float16] = [11, 12, 13, 21, 22, 23]
        let bx: [Float16] = [31, 32, 33]
        let expectedRows: [[Float16]] = [Array(state[0..<channels]),
                                         Array(state[channels..<(2 * channels)]),
                                         bx]
        for tap in 0..<3 {
            var weight = [Float16](repeating: 0, count: channels * 3)
            for c in 0..<channels { weight[c * 3 + tap] = 1 }
            var mutableState = state
            let out = try runShortConvDecode(ctx, bx: bx, weight: weight,
                                             state: &mutableState,
                                             channels: UInt32(channels))
            for c in 0..<channels {
                #expect(out[c] == Float(expectedRows[tap][c]),
                        "tap \(tap) channel \(c)")
            }
        }
    }

    @Test func prefillMatchesTokenByTokenDecode() throws {
        let ctx = Self.sharedContext
        let channels = 8
        let tokens = 7
        let bx = makeBx(tokens: tokens, channels: channels, seed: 4)
        let weight = makeShortConvWeight(channels: channels, seed: 5)
        let initial = zeroState(channels: channels)
        let (chunked, stateAfterChunk, _) = try runShortConvPrefill(
            ctx, bx, weight, initial: initial,
            tokens: tokens, channels: channels)
        var state = initial
        var perToken: [Float] = []
        for t in 0..<tokens {
            perToken += try runShortConvDecode(ctx, bx: row(bx, t, channels: channels),
                                               weight: weight, state: &state,
                                               channels: UInt32(channels))
        }
        #expect(chunked.count == perToken.count)
        for i in 0..<perToken.count {
            #expect(abs(chunked[i] - perToken[i]) < 2e-3,
                    "index \(i): prefill=\(chunked[i]) decode=\(perToken[i])")
        }
        #expect(state == stateAfterChunk)
    }

    /// Chunk lengths 1 and 2 are the boundary cases (left context comes wholly
    /// or partly from the incoming state); longer chunks must not read past the
    /// chunk start either. Every length is compared against decode.
    @Test(arguments: [1, 2, 3, 5, 7, 16] as [Int], [5, 8] as [Int])
    func prefillChunkSizesMatchTokenByTokenDecode(tokens: Int, channels: Int) throws {
        let ctx = Self.sharedContext
        let bx = makeBx(tokens: tokens, channels: channels, seed: UInt64(0x10 &+ tokens))
        let weight = makeShortConvWeight(channels: channels, seed: UInt64(0x20 &+ channels))
        var rng = SeedTree(UInt64(0x30 &+ tokens &+ channels)).key("shortconv-initial")
        let initial = (0..<(2 * channels)).map { _ in Float16(rng.uniform(-1, 1)) }

        let (chunked, stateAfterChunk, stateInAfter) = try runShortConvPrefill(
            ctx, bx, weight, initial: initial, tokens: tokens, channels: channels)
        var state = initial
        var perToken: [Float] = []
        for t in 0..<tokens {
            perToken += try runShortConvDecode(ctx, bx: row(bx, t, channels: channels),
                                               weight: weight, state: &state,
                                               channels: UInt32(channels))
        }
        for i in 0..<perToken.count {
            #expect(abs(chunked[i] - perToken[i]) < 2e-3,
                    "tokens=\(tokens) channels=\(channels) index \(i)")
        }
        #expect(state == stateAfterChunk, "final state, tokens=\(tokens)")
        #expect(stateInAfter == initial, "stateIn must not be mutated")
    }

    /// `stateOut` carries the last two `Bx` rows of the chunk, oldest first.
    /// For a one-token chunk that means `[stateIn[1], bx[0]]`; nothing may read
    /// before the chunk or the incoming state.
    @Test(arguments: [1, 2, 3, 7] as [Int])
    func prefillStateOutHoldsLastTwoGatedRows(tokens: Int) throws {
        let ctx = Self.sharedContext
        let channels = 6
        let bx = makeBx(tokens: tokens, channels: channels, seed: UInt64(0x40 &+ tokens))
        let weight = makeShortConvWeight(channels: channels, seed: 0x50)
        let initial = makeBx(tokens: 2, channels: channels, seed: 0x60)

        let (_, stateOut, _) = try runShortConvPrefill(
            ctx, bx, weight, initial: initial, tokens: tokens, channels: channels)
        let expectedOldest = tokens >= 2
            ? row(bx, tokens - 2, channels: channels)
            : row(initial, 1, channels: channels)
        let expectedNewest = row(bx, tokens - 1, channels: channels)
        #expect(stateOut == expectedOldest + expectedNewest,
                "tokens=\(tokens) stateOut=\(stateOut)")
    }

    /// The wrapper's byte offsets must bind every buffer, not just the base.
    @Test func offsetsBindAllBuffers() throws {
        let ctx = Self.sharedContext
        let tokens = 5
        let channels = 7
        let bx = makeBx(tokens: tokens, channels: channels, seed: 0x71)
        let weight = makeShortConvWeight(channels: channels, seed: 0x72)
        let initial = makeBx(tokens: 2, channels: channels, seed: 0x73)
        let offset = 8 * MemoryLayout<Float16>.size

        let plain = try runShortConvPrefill(ctx, bx, weight, initial: initial,
                                            tokens: tokens, channels: channels)
        let shifted = try runShortConvPrefill(ctx, bx, weight, initial: initial,
                                              tokens: tokens, channels: channels,
                                              byteOffset: offset)
        #expect(plain.out == shifted.out)
        #expect(plain.stateOut == shifted.stateOut)

        var plainState = initial
        var shiftedState = initial
        let plainDecode = try runShortConvDecode(ctx, bx: row(bx, 0, channels: channels),
                                                 weight: weight, state: &plainState,
                                                 channels: UInt32(channels))
        let shiftedDecode = try runShortConvDecode(ctx, bx: row(bx, 0, channels: channels),
                                                   weight: weight, state: &shiftedState,
                                                   channels: UInt32(channels),
                                                   byteOffset: offset)
        #expect(plainDecode == shiftedDecode)
        #expect(plainState == shiftedState)
    }
}
