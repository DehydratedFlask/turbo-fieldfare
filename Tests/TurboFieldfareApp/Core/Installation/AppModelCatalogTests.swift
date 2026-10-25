import Testing
import TurboFieldfareFormat
@testable import TurboFieldfareAppCore

@Suite struct AppModelCatalogTests {
    @Test func catalogHasGemmaDefaultAndLFM() {
        #expect(AppModelCatalog.default.id == AppModelCatalog.gemma4.id)
        #expect(AppModelCatalog.gemma4.family == .gemma4)
        #expect(AppModelCatalog.gemma4.defaultFilename == "gemma4.gturbo")
        #expect(AppModelCatalog.gemma4.defaultContext == 16_384)
        #expect(AppModelCatalog.gemma4.defaultExpertCacheSlots == 16)
        #expect(AppModelCatalog.gemma4.supportsVision)

        #expect(AppModelCatalog.lfm2_5.family == .lfm2Moe)
        #expect(AppModelCatalog.lfm2_5.defaultFilename == "lfm2.5-24b-a2b.gturbo")
        #expect(AppModelCatalog.lfm2_5.defaultContext == 8_192)
        #expect(AppModelCatalog.lfm2_5.defaultExpertCacheSlots == 8)
        #expect(!AppModelCatalog.lfm2_5.supportsVision)
        #expect(AppModelCatalog.descriptor(id: nil).id == AppModelCatalog.gemma4.id)
        #expect(AppModelCatalog.descriptor(id: AppModelCatalog.lfm2_5.id).family == .lfm2Moe)
    }

    @Test func catalogDescriptorCarriesPinnedLFMValues() {
        let lfm = AppModelCatalog.lfm2_5
        #expect(lfm.repoID == "LiquidAI/LFM2.5-24B-A2B-MLX-4bit")
        #expect(lfm.revision == "667408d494bde191c2cb9dd4f4549827816b9275")
        #expect(lfm.sourceIndexSHA256 == "7887b03265aef7824c8e5b9a51667e486f889749240d8c59a95f3cd899e86e4a")
        #expect(lfm.approximateDownloadBytes == 13_567_188_992)
    }
}
