import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfare

@Suite("LFM tokenizer")
struct LFM2TokenizerTests {
    @Test func rejectsMissingIdentity() async throws {
        let directory = try makeLFMTokenizerDirectory(includeIdentity: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        await #expect(throws: LFM2TokenizerError.self) {
            _ = try await LFM2Tokenizer.load(forModelDirectory: directory)
        }
    }

    @Test(.enabled(if: InstalledTokenizerPacks.gemmaDirectory != nil
                    && InstalledTokenizerPacks.lfmDirectory != nil))
    func factorySelectsByFamily() async throws {
        let gemma = try #require(InstalledTokenizerPacks.gemmaDirectory)
        let lfm = try #require(InstalledTokenizerPacks.lfmDirectory)
        #expect(try await TokenizerFactory.load(
            family: .gemma4, forModelDirectory: gemma).family == .gemma4)
        #expect(try await TokenizerFactory.load(
            family: .lfm2Moe, forModelDirectory: lfm).family == .lfm2Moe)
    }
}

private func makeLFMTokenizerDirectory(includeIdentity: Bool) throws -> URL {
    let manager = FileManager.default
    let directory = manager.temporaryDirectory
        .appendingPathComponent("lfm-tokenizer-\(UUID().uuidString)", isDirectory: true)
    let sidecar = directory.appendingPathComponent("tokenizer", isDirectory: true)
    try manager.createDirectory(at: sidecar, withIntermediateDirectories: true)
    if includeIdentity {
        let identity = TokenizerIdentity(
            family: .lfm2Moe,
            chatTemplateIdentity: "lfm2.5-24b-a2b-chatml-v1",
            chatTemplateSHA256: "00",
            tokenizerSHA256: "00",
            tokenizerConfigSHA256: "00")
        try JSONEncoder().encode(identity)
            .write(to: sidecar.appendingPathComponent(TokenizerIdentity.fileName))
    }
    return directory
}

enum InstalledTokenizerPacks {
    static var gemmaDirectory: URL? { modelDirectory(named: "gemma4.gturbo") }
    static var lfmDirectory: URL? { modelDirectory(named: "lfm2.5-24b-a2b.gturbo") }

    private static func modelDirectory(named name: String) -> URL? {
        var directory = URL(fileURLWithPath: "\(#filePath)").deletingLastPathComponent()
        while directory.path != "/" {
            let candidate = directory.appendingPathComponent("scratch")
                .appendingPathComponent(name)
            let tokenizer = candidate.appendingPathComponent("tokenizer/tokenizer.json")
            if FileManager.default.fileExists(atPath: tokenizer.path) {
                return candidate
            }
            directory = directory.deletingLastPathComponent()
        }
        return nil
    }
}
