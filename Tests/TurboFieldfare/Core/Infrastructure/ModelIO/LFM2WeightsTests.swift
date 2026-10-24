import Metal
import Testing

@testable import TurboFieldfare

@Suite struct LFM2WeightsTests {
    @Test func lfmWeightsExposeEveryNamedTensor() throws {
        let weights = try makeWeights()
        #expect(try weights.operatorNorm(layer: 0).shape.0 == 2048)
        #expect(try weights.convIn(layer: 0).shape.0 == 6144)
        #expect(try weights.qProj(layer: 2).shape.0 == 2048)
        #expect(try weights.routerGate(layer: 2).dtype == 3)
        #expect(throws: ModelError.self) { _ = try weights.convIn(layer: 2) }
    }

    @Test func lfmWeightsResolveEachLayerType() throws {
        let weights = try makeWeights()

        #expect(try weights.embedding().shape.0 == 128_000)
        #expect(try weights.embeddingNorm().shape.0 == 2048)
        #expect(try weights.ffnNorm(layer: 0).shape.0 == 2048)

        #expect(try weights.convOut(layer: 0).shape.0 == 2048)
        #expect(try weights.convWeight(layer: 0).shape
                == (2048, 3, 1, 0))

        #expect(try weights.kProj(layer: 2).shape.0 == 512)
        #expect(try weights.vProj(layer: 2).shape.0 == 512)
        #expect(try weights.oProj(layer: 2).shape.0 == 2048)
        #expect(try weights.qLayernorm(layer: 2).shape.0 == 64)
        #expect(try weights.kLayernorm(layer: 2).shape.0 == 64)

        #expect(try weights.denseGate(layer: 0).shape.0 == 11_776)
        #expect(try weights.denseUp(layer: 1).shape.0 == 11_776)
        #expect(try weights.denseDown(layer: 1).shape.0 == 2048)

        #expect(try weights.expertBias(layer: 2).shape.0 == 64)
    }

    @Test func lfmWeightsRejectTensorsTheLayerTypeDoesNotHave() throws {
        let weights = try makeWeights()
        #expect(throws: ModelError.self) { _ = try weights.convIn(layer: 2) }
        #expect(throws: ModelError.self) { _ = try weights.qProj(layer: 0) }
        #expect(throws: ModelError.self) { _ = try weights.denseGate(layer: 2) }
        #expect(throws: ModelError.self) { _ = try weights.routerGate(layer: 0) }
        #expect(throws: ModelError.self) { _ = try weights.expertBias(layer: 1) }
        #expect(throws: ModelError.self) { _ = try weights.operatorNorm(layer: 40) }
    }

    private func makeWeights() throws -> LFM2Weights {
        let model = try loadSyntheticLFM()
        return LFM2Weights(lookup: try SyntheticLFMLookup(model: model),
                           config: .lfm2_5_24B_A2B)
    }
}

/// Tiny `MTLBuffer`-backed lookup: metadata comes from the synthetic resident
/// index, so no real tensor payload is allocated.
private struct SyntheticLFMLookup: LFM2TensorLookup {
    private let entries: [String: ResidentIndexEntry]
    private let buffer: MTLBuffer

    init(model: SyntheticLFM) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalError.noDevice
        }
        guard let buffer = device.makeBuffer(length: 16,
                                             options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        self.entries = model.residentIndex.entries
        self.buffer = buffer
    }

    func tensor(named name: String) throws -> TensorView {
        guard let entry = entries[name] else {
            throw ModelError.tensorNotFound(name: name)
        }
        return TensorView(
            buffer: buffer,
            offset: 0,
            length: entry.sizeBytes,
            scaleOffset: 0, scaleLength: entry.scaleSize,
            biasOffset: 0, biasLength: entry.biasSize,
            shape: entry.shape,
            dtype: entry.dtype)
    }
}
