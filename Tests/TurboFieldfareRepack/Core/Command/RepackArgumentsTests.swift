import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

@Suite
struct RepackArgumentsTests {
    @Test func sourceSelectorDefaultsToGemmaAndAcceptsLFM() throws {
        #expect(try RepackArguments.parse(["--output", "/tmp/x.gturbo"]).modelSource == "gemma4")
        #expect(try RepackArguments.parse(["--output", "/tmp/x.gturbo",
                                           "--model-source", "lfm2_moe"]).modelSource
            == "lfm2_moe")
    }

    @Test func sourceSelectorRejectsAnUnknownSource() {
        #expect(throws: RepackArgumentError.self) {
            _ = try RepackArguments.parse(["--output", "/tmp/x.gturbo",
                                           "--model-source", "llama"])
        }
    }

    @Test func sourceSelectorRequiresAValue() {
        #expect(throws: RepackArgumentError.self) {
            _ = try RepackArguments.parse(["--output", "/tmp/x.gturbo",
                                           "--model-source"])
        }
    }

    @Test func catalogPinsBothSources() {
        #expect(ModelSourceCatalog.gemma4.family == .gemma4)
        #expect(ModelSourceCatalog.lfm2_5_24B_A2B.family == .lfm2Moe)
        #expect(ModelSourceCatalog.lfm2_5_24B_A2B.sourceIndexSHA256
            == "7887b03265aef7824c8e5b9a51667e486f889749240d8c59a95f3cd899e86e4a")
        #expect(ModelSourceCatalog.lfm2_5_24B_A2B.defaultFilename == "lfm2.5-24b-a2b.gturbo")
        #expect(ModelSourceCatalog.lfm2_5_24B_A2B.defaultExpertCacheSlots == 8)
        #expect(ModelSourceCatalog.lfm2_5_24B_A2B.supportsVision == false)
    }

    @Test func catalogPinsTheLFMSourceIdentity() {
        let lfm = ModelSourceCatalog.lfm2_5_24B_A2B
        #expect(lfm.id == "lfm2_moe")
        #expect(lfm.repoID == "LiquidAI/LFM2.5-24B-A2B-MLX-4bit")
        #expect(lfm.revision == "667408d494bde191c2cb9dd4f4549827816b9275")
        #expect(lfm.approximateDownloadBytes == 13_567_188_992)
        #expect(lfm.installedBytes == 13_567_188_992)
        #expect(lfm.reserveBytes == 1_073_741_824)
        #expect(lfm.defaultContext == 8_192)
    }

    @Test func catalogPreservesTheGemmaSourceIdentity() {
        let gemma = ModelSourceCatalog.gemma4
        #expect(gemma.id == "gemma4")
        #expect(gemma.repoID == "mlx-community/gemma-4-26b-a4b-it-4bit")
        #expect(gemma.revision == "0d77464eeb233a2da68ebf9d7dc4edaac7db956d")
        #expect(gemma.sourceIndexSHA256
            == "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13")
        #expect(gemma.approximateDownloadBytes == 14_620_479_420)
        #expect(gemma.installedBytes == 14_291_921_884)
        #expect(gemma.reserveBytes == 1_073_741_824)
        #expect(gemma.defaultFilename == "gemma4.gturbo")
        #expect(gemma.defaultContext == 16_384)
        #expect(gemma.defaultExpertCacheSlots == 16)
        #expect(gemma.supportsVision == true)
    }

    @Test func descriptorLookupUsesTheSelectorIDs() {
        #expect(ModelSourceCatalog.all.count == 2)
        #expect(ModelSourceCatalog.descriptor(id: "gemma4") == ModelSourceCatalog.gemma4)
        #expect(ModelSourceCatalog.descriptor(id: "lfm2_moe")
            == ModelSourceCatalog.lfm2_5_24B_A2B)
        #expect(ModelSourceCatalog.descriptor(id: "llama") == nil)
    }

    @Test func installOptionsFollowTheSelectedDescriptor() {
        let output = URL(fileURLWithPath: "/tmp/x.gturbo")
        let lfm = SupportedModelSource.installOptions(
            for: ModelSourceCatalog.lfm2_5_24B_A2B,
            outputDirectory: output,
            overwrite: true,
            token: "token",
            resume: true)
        #expect(lfm.repoID == ModelSourceCatalog.lfm2_5_24B_A2B.repoID)
        #expect(lfm.revision == ModelSourceCatalog.lfm2_5_24B_A2B.revision)
        #expect(lfm.minFreeReserveBytes == ModelSourceCatalog.lfm2_5_24B_A2B.reserveBytes)
        #expect(lfm.overwrite && lfm.resume)
        #expect(lfm.requireKnownSource)

        // The no-descriptor overload keeps the historical Gemma behavior.
        let gemma = SupportedModelSource.installOptions(
            outputDirectory: output,
            overwrite: false,
            token: nil)
        #expect(gemma.repoID == ModelSourceCatalog.gemma4.repoID)
        #expect(gemma.revision == ModelSourceCatalog.gemma4.revision)
        #expect(gemma.minFreeReserveBytes == ModelSourceCatalog.gemma4.reserveBytes)
    }

    @Test func everyExistingModeStillParses() throws {
        #expect(try RepackArguments.parse(["--output", "/tmp/x.gturbo", "--overwrite",
                                           "--resume"]).resume)
        #expect(try RepackArguments.parse(["--discard-partial",
                                           "--output", "/tmp/x.gturbo"]).discardPartial)
        #expect(try RepackArguments.parse(["--verify-install",
                                           "--input-gturbo", "/tmp/x.gturbo"]).verifyInstall)
        #expect(try RepackArguments.parse(["--vision-output", "/tmp/x.vision.gturbo",
                                           "--text-model", "/tmp/x.gturbo"]).visionOutput
            == "/tmp/x.vision.gturbo")
    }
}
