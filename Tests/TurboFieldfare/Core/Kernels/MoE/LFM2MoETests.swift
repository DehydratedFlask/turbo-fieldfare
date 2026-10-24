import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// LFM2.5 MoE: top-4 of 64 experts, SiLU-gated routed FFNs, and a weighted
/// residual combine. Fixtures are 8 x 8 / 8 x 6 logical blocks embedded in an
/// affine group-sized block (see `LFM2SharedExpertTests` for the rationale).
@Suite struct LFM2MoETests {

    private static let groupWidth = Quantization.groupSize
    private static let topK = 4

    private struct Expert {
        let gate: [[Float]]
        let up: [[Float]]
        let down: [[Float]]
    }

    @Test func top4MoEWeightsSumAndMatchCPU() throws {
        let (experts, x) = makeSmallMoE(experts: 8, hidden: 8, intermediate: 6, seed: 9)
        let indices: [UInt32] = [1, 4, 6, 7]
        let weights: [Float16] = [0.1, 0.2, 0.3, 0.4]
        let y = try runLFM2MoE(experts, indices, weights, x)
        let expected = cpuWeightedExperts(experts, indices, weights, x)
        for i in 0..<8 { #expect(abs(y[i] - expected[i]) < 0.02, "index \(i)") }
    }

    /// The routed activation is SiLU, not the Gelu used by Gemma's MoE. A
    /// one-hot gate/up/down isolates `silu(x[i])` per routed output.
    @Test func routedActivationIsSiluNotGelu() throws {
        let hidden = 8
        let intermediate = 6
        var gate = [[Float]](repeating: [Float](repeating: 0, count: hidden),
                             count: intermediate)
        var up = gate
        var down = [[Float]](repeating: [Float](repeating: 0, count: intermediate),
                             count: hidden)
        for i in 0..<intermediate {
            gate[i][i] = 1
            up[i][i] = 1
            down[i][i] = 1
        }
        let expert = Expert(gate: gate, up: up, down: down)
        let experts = [expert, expert, expert, expert]
        let x = (0..<hidden).map { Float(Double($0) / 8) }
        let y = try runLFM2MoE(experts, [0, 1, 2, 3],
                               [Float16](repeating: 0.25, count: 4), x)
        for i in 1..<intermediate {
            let xv = Float(Float16(x[i]))
            let silu = xv / (1 + exp(-xv))
            let gelu = 0.5 * xv * (1 + tanh(0.7978845608028654
                * (xv + 0.044715 * xv * xv * xv)))
            // Four identical experts with weight 0.25 sum to one routed copy;
            // `up` is the identity, so the value is silu(x) * x.
            #expect(abs(y[i] - silu * xv) < 0.02, "index \(i)")
            if i >= 2 {
                #expect(abs(y[i] - gelu * xv) > 1e-3, "gelu would match index \(i)")
            }
        }
    }

    /// The prefill scheduler is shared with Gemma; `LFM2PrefillMoE` must select
    /// its SiLU activation variant. One token routed to four experts pins the
    /// activation and the per-pair reduction. Fixtures use O(1) hidden values
    /// and O(0.1) weights so routed partials are O(0.1): at small magnitudes
    /// SiLU and GELU round to the same FP16 value and the test would be
    /// vacuous. `d = 64` and `f = 128` also make a D/F swap observable.
    @Test func prefillRoutedMoEUsesSiluAndMatchesCPU() throws {
        let d = 64
        let f = 128
        let topK = 4
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            [
                PrefillGroupedRoutedMoETests.pair(token: 0, expert: 2, rank: 0),
                PrefillGroupedRoutedMoETests.pair(token: 0, expert: 3, rank: 1),
                PrefillGroupedRoutedMoETests.pair(token: 0, expert: 5, rank: 2),
                PrefillGroupedRoutedMoETests.pair(token: 0, expert: 7, rank: 3),
            ],
            queryCount: 1,
            topK: topK,
            numExperts: 16,
            tileExpertCount: 16)
        let pool = Self.makePrefillExpertPool(numExperts: 16, d: d, f: f)
        var rng = SplitMix64(seed: 0x9F3A_11E2)
        let hidden: [Float16] = (0..<d).map { _ in
            Float16(rng.uniform(-1, 1))
        }
        let siluExpected = Self.cpuPrefillRoutePartials(
            routes: routes, hidden: hidden, hiddenStride: d,
            pool: pool, topK: topK, d: d, f: f, activation: .silu)
        let geluExpected = Self.cpuPrefillRoutePartials(
            routes: routes, hidden: hidden, hiddenStride: d,
            pool: pool, topK: topK, d: d, f: f, activation: .gelu)

        let context = try MetalContext()
        let kernel = try LFM2PrefillMoE(context: context)
        let hiddenBuffer = try #require(Fp16Buffer.make(
            context.device, halves: hidden))
        let pairBuffer = try #require(context.device.makeBuffer(
            bytes: routes.sortedPairs,
            length: routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
            options: .storageModeShared))
        let outputBuffer = try #require(Fp16Buffer.make(
            context.device, count: routes.queryCount * topK * d))
        let activationScratch = try #require(context.device.makeBuffer(
            length: 3 * topK * f * MemoryLayout<Float16>.stride,
            options: .storageModePrivate))
        let downScratch = try #require(context.device.makeBuffer(
            length: topK * d * MemoryLayout<Float16>.stride,
            options: .storageModePrivate))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        let expertIDs = try PrefillStreamedTileBinding.expertIDs(
            forTile: 0, routes: routes)
        let binding = try PrefillStreamedTileBinding(
            expertIDs: expertIDs,
            views: try PrefillGroupedRoutedMoETests.streamedViewsWithNonzeroOffsets(
                device: context.device, pool: pool, expertIDs: expertIDs))
        let params = PrefillGroupedRoutedMoEStreamedParams(
            pairStart: 0,
            pairCount: UInt32(routes.sortedPairs.count),
            d: UInt32(d),
            routedIntermediate: UInt32(f),
            topK: UInt32(topK),
            hiddenStrideElements: UInt32(d),
            binding: binding,
            offsets: pool.offsets)
        let argumentBuffer = try kernel.makeStreamedArgumentBuffer(
            device: context.device, binding: binding)
        kernel.encodeStreamedBatched(
            commandBuffer: commandBuffer,
            hidden: hiddenBuffer,
            sortedPairs: pairBuffer,
            routePartials: outputBuffer,
            gateUpActScratch: activationScratch,
            downScratch: downScratch,
            argumentBuffer: argumentBuffer,
            binding: binding,
            params: params)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        let actual = Fp16Buffer.readHalf(
            outputBuffer, count: routes.queryCount * topK * d)
        let siluError = Self.maxAbsoluteError(actual, siluExpected)
        let geluError = Self.maxAbsoluteError(actual, geluExpected)
        let expectedMagnitude = siluExpected.reduce(Float(0)) {
            max($0, abs(Float($1)))
        }
        #expect(expectedMagnitude > 0.02,
                "fixture below FP16 resolution: max=\(expectedMagnitude)")
        #expect(siluError < 0.0015, "maxAbsoluteError=\(siluError)")
        // A GELU kernel (or an empty output) must not satisfy the SiLU check.
        #expect(geluError > 0.0015,
                "GELU reference unexpectedly matched: error=\(geluError)")
    }

    // MARK: - Fixtures

    private func makeSmallMoE(experts: Int,
                              hidden: Int,
                              intermediate: Int,
                              seed: UInt64) -> ([Expert], [Float]) {
        var rng = SplitMix64(seed: seed)
        func matrix(_ rows: Int, _ columns: Int) -> [[Float]] {
            (0..<rows).map { _ in
                (0..<columns).map { _ in rng.uniform(-0.4, 0.4) }
            }
        }
        let list = (0..<experts).map { _ in
            Expert(gate: matrix(intermediate, hidden),
                   up: matrix(intermediate, hidden),
                   down: matrix(hidden, intermediate))
        }
        let x = (0..<hidden).map { _ in rng.uniform(-0.5, 0.5) }
        return (list, x)
    }

    // MARK: - Production path

    private func runLFM2MoE(_ experts: [Expert],
                            _ indices: [UInt32],
                            _ weights: [Float16],
                            _ x: [Float]) throws -> [Float] {
        precondition(indices.count == Self.topK)
        precondition(weights.count == Self.topK)
        let width = Self.groupWidth
        let blobs = indices.map { Self.makeBlob(experts[Int($0)]) }

        let context = try MetalContext()
        let kernel = try LFM2MoE(context: context)
        let routedBuffers = try blobs.map { blob in
            try #require(context.device.makeBuffer(
                bytes: blob.bytes,
                length: blob.bytes.count,
                options: .storageModeShared))
        }
        let xBuffer = try #require(Fp16Buffer.make(
            context.device, halves: Self.paddedHalves(x)))
        let actsBuffer = try #require(Fp16Buffer.make(
            context.device, count: Self.topK * width))
        let routingBuffer = try #require(Fp16Buffer.make(
            context.device, halves: weights))
        let residualBuffer = try #require(Fp16Buffer.make(
            context.device, count: width))
        let outputBuffer = try #require(Fp16Buffer.make(
            context.device, count: width))
        let argumentBuffer = try #require(kernel.makeRoutedArgumentBuffer(
            routedBlobs: routedBuffers, topK: UInt32(Self.topK)))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        kernel.encodePhase1(
            commandBuffer: commandBuffer,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            x: xBuffer,
            acts: actsBuffer,
            d: UInt32(width),
            f: UInt32(width),
            topK: UInt32(Self.topK))
        kernel.encodePhase2(
            commandBuffer: commandBuffer,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            acts: actsBuffer,
            routingWeights: routingBuffer,
            residual: residualBuffer,
            y: outputBuffer,
            d: UInt32(width),
            f: UInt32(width),
            topK: UInt32(Self.topK))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        return Array(Fp16Buffer.read(outputBuffer, count: width).prefix(x.count))
    }

    // MARK: - CPU reference

    private func cpuWeightedExperts(_ experts: [Expert],
                                    _ indices: [UInt32],
                                    _ weights: [Float16],
                                    _ x: [Float]) -> [Float] {
        var y = [Float](repeating: 0, count: x.count)
        for slot in indices.indices {
            let output = Self.cpuSiluFFN(experts[Int(indices[slot])],
                                         x: x.map { Float16($0) })
            for i in 0..<x.count {
                y[i] += Float(weights[slot]) * output[i]
            }
        }
        return y
    }

    private static func cpuSiluFFN(_ expert: Expert, x: [Float16]) -> [Float] {
        let width = groupWidth
        let xPadded = paddedVector(x)
        let gate = DequantInt4GemvRef.apply(
            weightRows: paddedRows(expert.gate), x: xPadded, n: width)
        let up = DequantInt4GemvRef.apply(
            weightRows: paddedRows(expert.up), x: xPadded, n: width)
        let acts = zip(gate, up).map { gateValue, upValue in
            Float(Float16(silu(gateValue) * upValue))
        }
        let output = DequantInt4GemvRef.apply(
            weightRows: paddedRows(expert.down), x: acts, n: width)
        return Array(output.prefix(x.count))
    }

    private enum PrefillActivation {
        case silu
        case gelu
    }

    private static func cpuPrefillRoutePartials(
        routes: PrefillMoEGroupedRoutes,
        hidden: [Float16],
        hiddenStride: Int,
        pool: PrefillGroupedRoutedMoETests.SyntheticExpertPool,
        topK: Int,
        d: Int,
        f: Int,
        activation: PrefillActivation
    ) -> [Float16] {
        var out = [Float16](repeating: -99, count: routes.queryCount * topK * d)
        for pair in routes.sortedPairs {
            let expertBase = Int(pair.expert) * pool.stride
            let xBase = Int(pair.token) * hiddenStride
            let x = (0..<d).map { Float(hidden[xBase + $0]) }
            var acts = [Float16](repeating: 0, count: f)
            for row in 0..<f {
                let gate = PrefillGroupedRoutedMoETests.cpuInt4Dot(
                    bytes: pool.bytes, base: expertBase,
                    wOff: Int(pool.offsets.gateWOff),
                    sOff: Int(pool.offsets.gateSOff),
                    bOff: Int(pool.offsets.gateBOff),
                    row: row, n: d, x: x)
                let up = PrefillGroupedRoutedMoETests.cpuInt4Dot(
                    bytes: pool.bytes, base: expertBase,
                    wOff: Int(pool.offsets.upWOff),
                    sOff: Int(pool.offsets.upSOff),
                    bOff: Int(pool.offsets.upBOff),
                    row: row, n: d, x: x)
                switch activation {
                case .silu: acts[row] = Float16(silu(gate) * up)
                case .gelu: acts[row] = Float16(gelu(gate) * up)
                }
            }
            let actFloat = acts.map { Float($0) }
            let outBase = (Int(pair.token) * topK + Int(pair.rank)) * d
            for row in 0..<d {
                let value = PrefillGroupedRoutedMoETests.cpuInt4Dot(
                    bytes: pool.bytes, base: expertBase,
                    wOff: Int(pool.offsets.downWOff),
                    sOff: Int(pool.offsets.downSOff),
                    bOff: Int(pool.offsets.downBOff),
                    row: row, n: f, x: actFloat)
                out[outBase + row] = Float16(value)
            }
        }
        return out
    }

    /// Local prefill expert pool with O(0.1) affine weights. Built through the
    /// shared quantized-block helper so the on-disk layout matches production
    /// expert blobs while keeping every routed partial above FP16 resolution.
    private static func makePrefillExpertPool(
        numExperts: Int,
        d: Int,
        f: Int
    ) -> PrefillGroupedRoutedMoETests.SyntheticExpertPool {
        var allBytes: [UInt8] = []
        var offsets: MoEExpertOffsets?
        var stride = 0
        for expert in 0..<numExperts {
            var bytes: [UInt8] = []
            func rows(_ count: Int, _ columns: Int, _ role: Int) -> [[Float]] {
                (0..<count).map { row in
                    (0..<columns).map { column in
                        let sign: Float =
                            (row + column + role + expert) % 2 == 0 ? 1 : -1
                        let wobble = Float(
                            (row * 3 + column * 5 + role + expert) % 7) * 0.004
                        return sign * (0.10 + wobble)
                    }
                }
            }
            func append(_ projectionRows: [[Float]])
                -> (w: UInt32, s: UInt32, b: UInt32) {
                let w = UInt32(bytes.count)
                PrefillGroupedRoutedMoETests.appendProjection(
                    rows: projectionRows, to: &bytes, component: .packed)
                let s = UInt32(bytes.count)
                PrefillGroupedRoutedMoETests.appendProjection(
                    rows: projectionRows, to: &bytes, component: .scales)
                let b = UInt32(bytes.count)
                PrefillGroupedRoutedMoETests.appendProjection(
                    rows: projectionRows, to: &bytes, component: .biases)
                return (w, s, b)
            }

            let gate = append(rows(f, d, 0))
            let up = append(rows(f, d, 1))
            let down = append(rows(d, f, 2))
            let current = MoEExpertOffsets(
                gateWOff: gate.w, gateSOff: gate.s, gateBOff: gate.b,
                upWOff: up.w, upSOff: up.s, upBOff: up.b,
                downWOff: down.w, downSOff: down.s, downBOff: down.b)
            if offsets == nil {
                offsets = current
                stride = bytes.count
            } else {
                #expect(stride == bytes.count)
                #expect(offsets!.gateWOff == current.gateWOff)
                #expect(offsets!.downBOff == current.downBOff)
            }
            allBytes.append(contentsOf: bytes)
        }
        return PrefillGroupedRoutedMoETests.SyntheticExpertPool(
            bytes: allBytes, offsets: offsets!, stride: stride)
    }

    private static func maxAbsoluteError(_ actual: [Float16],
                                         _ expected: [Float16]) -> Float {
        zip(actual, expected).reduce(Float(0)) {
            max($0, abs(Float($1.0) - Float($1.1)))
        }
    }

    // MARK: - Padding / blob helpers

    private static func silu(_ x: Float) -> Float {
        x / (1 + exp(-x))
    }

    private static func gelu(_ x: Float) -> Float {
        let cube = x * x * x
        var inner = 0.7978845608028654 * (x + 0.044715 * cube)
        inner = min(max(inner, -20), 20)
        return 0.5 * x * (1 + tanh(inner))
    }

    private static func paddedHalves(_ x: [Float]) -> [Float16] {
        precondition(x.count <= groupWidth)
        var values = [Float16](repeating: 0, count: groupWidth)
        for i in x.indices { values[i] = Float16(x[i]) }
        return values
    }

    private static func paddedVector(_ x: [Float16]) -> [Float] {
        precondition(x.count <= groupWidth)
        var values = [Float](repeating: 0, count: groupWidth)
        for i in x.indices { values[i] = Float(x[i]) }
        return values
    }

    private static func paddedRows(_ rows: [[Float]]) -> [Quantization.Int4AffineRow] {
        let columns = rows.first?.count ?? 0
        precondition(rows.count <= groupWidth && columns <= groupWidth)
        return (0..<groupWidth).map { row in
            var values = [Float](repeating: 0, count: groupWidth)
            if row < rows.count {
                for column in 0..<columns { values[column] = rows[row][column] }
            }
            return Quantization.quantizeInt4Affine(values)
        }
    }

    private static func makeBlob(_ expert: Expert)
        -> (bytes: [UInt8], offsets: MoEExpertOffsets) {
        var bytes = [UInt8]()
        func append(_ values: [UInt8]) { bytes.append(contentsOf: values) }
        func append(_ values: [UInt16]) {
            for value in values {
                bytes.append(UInt8(truncatingIfNeeded: value))
                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
            }
        }
        enum Component { case packed, scales, biases }
        func appendComponent(_ rows: [Quantization.Int4AffineRow],
                             _ component: Component) {
            switch component {
            case .packed:
                for row in rows { append(row.packed) }
            case .scales:
                for row in rows { append(row.scales) }
            case .biases:
                for row in rows { append(row.biases) }
            }
        }
        func appendProjection(_ rows: [Quantization.Int4AffineRow])
            -> (w: UInt32, s: UInt32, b: UInt32) {
            let w = UInt32(bytes.count)
            appendComponent(rows, .packed)
            let s = UInt32(bytes.count)
            appendComponent(rows, .scales)
            let b = UInt32(bytes.count)
            appendComponent(rows, .biases)
            return (w, s, b)
        }

        let gate = appendProjection(paddedRows(expert.gate))
        let up = appendProjection(paddedRows(expert.up))
        let down = appendProjection(paddedRows(expert.down))
        let gateW = gate.w, gateS = gate.s, gateB = gate.b
        let upW = up.w, upS = up.s, upB = up.b
        let downW = down.w, downS = down.s, downB = down.b

        return (bytes,
                MoEExpertOffsets(
                    gateWOff: gateW, gateSOff: gateS, gateBOff: gateB,
                    upWOff: upW, upSOff: upS, upBOff: upB,
                    downWOff: downW, downSOff: downS, downBOff: downB))
    }
}
