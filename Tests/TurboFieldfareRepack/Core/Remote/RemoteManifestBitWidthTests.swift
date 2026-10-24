import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

@Suite struct RemoteManifestBitWidthTests {

    @Test func lfmPlanInfersSixBitEmbeddingAndFourBitDenseFFN() throws {
        let dir = temporaryRoot("lfm")
        defer { try? FileManager.default.removeItem(atPath: dir) }
        let snapshot = try SyntheticSnapshot.buildLFM(at: dir)
        let plan = try RepackPlanner.plan(
            meta: snapshot.metadata, arch: snapshot.arch,
            shardHeaders: snapshot.headers,
            outputDir: (dir as NSString).appendingPathComponent("out"))

        let bits = RemoteStreamingRepacker.inferBitWidths(plan: plan)
        #expect(bits.embedding == 6)
        #expect(bits.attention == 4)
        #expect(bits.sharedExpert == 4)
        #expect(bits.routedExpert == 4)

        // The inferred widths must produce a manifest the LFM gate accepts.
        let data = try GTurboJSON.encodeManifest(
            plan: plan,
            modelID: "LiquidAI/LFM2.5-24B-A2B-MLX-4bit",
            sourceSnapshotHash: "sha256:test",
            files: [], expertsPerLayer: 64, numLayers: 40,
            expertStride: 5_308_416, bitWidths: bits)
        let wire = try GTurboManifestCodec.decode(data)
        #expect(wire.quant?.embedding.weightBits == 6)
        #expect(wire.quant?.sharedExpert.weightBits == 4)
        #expect(wire.quant?.router == nil)
    }

    @Test func gemmaPlanKeepsHistoricalBitWidths() throws {
        let dir = temporaryRoot("gemma")
        defer { try? FileManager.default.removeItem(atPath: dir) }
        let snapshot = try SyntheticSnapshot.build(at: dir)
        let metadata = try IndexLoader.load(snapshotDir: dir)
        let arch = try ArchInfo.load(
            configPath: (dir as NSString).appendingPathComponent("config.json"))
        let header = try Safetensors.parseHeader(path: snapshot.shardPath)
        let plan = try RepackPlanner.plan(
            meta: metadata, arch: arch, shardHeaders: [header],
            outputDir: (dir as NSString).appendingPathComponent("out"))

        let bits = RemoteStreamingRepacker.inferBitWidths(plan: plan)
        #expect(bits.embedding == 4)
        #expect(bits.attention == 4)
        #expect(bits.router == 8)
        #expect(bits.sharedExpert == 8)
        #expect(bits.routedExpert == 4)
    }

    private func temporaryRoot(_ tag: String) -> String {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent(
                "turbofieldfare-lfm-bits-\(tag)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            atPath: path, withIntermediateDirectories: true)
        return path
    }
}
