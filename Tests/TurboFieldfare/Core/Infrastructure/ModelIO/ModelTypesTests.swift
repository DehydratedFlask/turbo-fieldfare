import Testing
import Foundation
@testable import TurboFieldfare

@Suite struct ModelTypesTests {

    @Test func archConfigGemma4BaselineMatchesDocs() {
        let a = ArchConfig.gemma4_26B_A4B
        #expect(a.hiddenSize == 2816)
        #expect(a.intermediateSize == 2112)
        #expect(a.moeIntermediateSize == 704)
        #expect(a.numLayers == 30)
        #expect(a.numExperts == 128)
        #expect(a.topKExperts == 8)
        #expect(a.vocabSize == 262144)
        #expect(a.tieWordEmbeddings == true)
        #expect(a.finalLogitSoftcap == 30.0)
        #expect(a.fullAttentionLayerMask.count == 30)
        let fullCount = a.fullAttentionLayerMask.reduce(0) { $0 + Int($1) }
        #expect(fullCount == 5, "Gemma 4 has 5 full-attention layers, got \(fullCount)")
        // Mask flags layers 5, 11, 17, 23, 29.
        for L in [5, 11, 17, 23, 29] {
            #expect(a.fullAttentionLayerMask[L] == 1, "layer \(L) should be full-attention")
        }
    }

    @Test func lfmArchConfigMatchesPinnedCheckpoint() {
        let c = ArchConfig.lfm2_5_24B_A2B
        #expect(c.family == .lfm2Moe)
        #expect(c.hiddenSize == 2048 && c.intermediateSize == 11_776)
        #expect(c.moeIntermediateSize == 1536 && c.numLayers == 40)
        #expect(c.numHeads == 32 && c.numKVHeads == 8 && c.headDim == 64)
        #expect(c.numFullKVHeads == 8 && c.fullHeadDim == 64)
        #expect(c.fullAttentionLayerMask.count == 40
            && c.fullAttentionLayerMask.reduce(0) { $0 + Int($1) } == 10)
        #expect(c.numExperts == 64 && c.topKExperts == 4)
        #expect(c.vocabSize == 128_000 && c.maxPositionEmbeddings == 131_072)
        #expect(c.slidingWindow == 0 && c.finalLogitSoftcap == 0)
        #expect(c.attentionScale == 0.125 && c.convKernelSize == 3)
        #expect(c.rmsNormUnitOffset == false && c.embeddingScaledBySqrtHidden == false)
        #expect(c.routerKind == .lfm2SigmoidTopKBias)
        #expect(c.layerTypes.filter { $0 == .conv }.count == 30)
        #expect(c.layerTypes.filter { $0 == .fullAttention }.count == 10)
        #expect(Array(c.moeLayerMask.prefix(2)) == [0, 0] && c.moeLayerMask.dropFirst(2).allSatisfy { $0 == 1 })
        #expect(c.admission.expertStrideBytes == 5_308_416)
        #expect(c.admission.expertCacheLayers == 38)
        #expect(c.admission.includedExpertCacheSlots == 0)
    }

    @Test func gemmaArchConfigDefaultsAreUnchanged() {
        let c = ArchConfig.gemma4_26B_A4B
        #expect(c.family == .gemma4)
        #expect(c.layerTypes[5] == .fullAttention)
        #expect(c.moeLayerMask.allSatisfy { $0 == 1 })
        #expect(c.admission.expertStrideBytes == 3_358_720)
        #expect(c.admission.includedExpertCacheSlots == 16)
    }

    @Test func modelErrorDescriptionsContainKeyFacts() {
        let e1 = ModelError.archMismatch(field: "hiddenSize", expected: "2816", actual: "4096")
        #expect(e1.description.contains("2816") && e1.description.contains("4096"))
        let e2 = ModelError.unsupportedVersion(major: 2, minor: 0)
        #expect(e2.description.contains("2"))
        let e3 = ModelError.checksumMismatch(file: "model_weights.bin")
        #expect(e3.description.contains("model_weights.bin"))
    }
}
