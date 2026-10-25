import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

@Suite("Server family dispatch")
struct ServerFamilyTests {
    @Test func lfmArgumentsDefaultToLFMContextAndSlots() throws {
        let args = try ServerArguments.parse(
            ["--model", "/tmp/lfm.gturbo"],
            hostMemoryBytes: 64 << 30,
            manifestFamily: .lfm2Moe)
        #expect(args.family == .lfm2Moe)
        #expect(args.modelID == "lfm2.5-24b-a2b")
        #expect(args.maxContext == 8_192)
        #expect(args.expertCacheSlots == 8)
        let config = try args.resolvedRuntimeConfiguration()
        #expect(config.prefillPolicy == .chunked)
    }

    @Test func gemmaArgumentsKeepTheirDefaults() throws {
        let args = try ServerArguments.parse(
            ["--model", "/tmp/gemma.gturbo"],
            hostMemoryBytes: 64 << 30,
            manifestFamily: .gemma4)
        #expect(args.family == .gemma4)
        #expect(args.modelID == "gemma-4-26b-a4b-it")
        #expect(args.maxContext == 16_384)
        #expect(args.expertCacheSlots == 16)
    }

    @Test func explicitFlagsStillOverrideFamilyDefaults() throws {
        let args = try ServerArguments.parse(
            ["--model", "/tmp/lfm.gturbo", "--model-id", "custom",
             "--max-context", "16384", "--expert-cache-slots", "24"],
            hostMemoryBytes: 64 << 30,
            manifestFamily: .lfm2Moe)
        #expect(args.modelID == "custom")
        #expect(args.maxContext == 16_384)
        #expect(args.expertCacheSlots == 24)
    }

    @Test func lfmRejectsVisionPackFlag() {
        #expect(throws: ServerArgumentError.self) {
            _ = try ServerArguments.parse(
                ["--model", "/tmp/lfm.gturbo", "--vision-pack", "/tmp/v.gturbo"],
                hostMemoryBytes: 64 << 30,
                manifestFamily: .lfm2Moe)
        }
    }

    @Test func lfmImageRequestIsRejectedWithAnUnsupportedCode() {
        let error = ServerModelSession.imageRejection(family: .lfm2Moe, hasImages: true)
        guard case .invalid(_, _, let code)? = error else {
            Issue.record("expected invalid"); return
        }
        #expect(code == "unsupported_image")
        #expect(ServerModelSession.imageRejection(family: .gemma4, hasImages: true) == nil)
        #expect(ServerModelSession.imageRejection(family: .lfm2Moe, hasImages: false) == nil)
    }
}
