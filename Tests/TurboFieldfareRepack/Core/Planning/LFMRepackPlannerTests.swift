import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

@Suite struct LFMRepackPlannerTests {

    @Test func lfmPlanPutsEmbeddingDenseFFNAndRouterInResident() throws {
        let plan = try lfmPlan()
        let names = Set(plan.resident.entries.map(\.name))
        #expect(names.contains("model.embed_tokens.weight"))
        let embedding = try #require(
            plan.resident.entries.first { $0.name == "model.embed_tokens.weight" })
        #expect(embedding.scaleSize > 0 && embedding.biasSize > 0)
        #expect(embedding.logicalShape4[0] == 128_000
            && embedding.logicalShape4[1] == 2_048)
        #expect(embedding.quantSpec?.bits == 6)
        #expect(names.contains("layers.0.feed_forward.gate_proj.weight"))
        #expect(names.contains("layers.0.feed_forward.up_proj.weight"))
        #expect(names.contains("layers.0.feed_forward.down_proj.weight"))
        #expect(names.contains("layers.2.feed_forward.gate.weight"))
        #expect(names.contains("layers.2.feed_forward.expert_bias"))
        #expect(!names.contains("lm_head.weight"))
    }

    @Test func lfmPlanEmitsThirtyEightExpertFilesAtThePinnedStride() throws {
        let plan = try lfmPlan()
        #expect(plan.layers.map(\.layerIndex) == Array(2..<40))
        #expect(plan.layers.allSatisfy { $0.expertsPerLayer == 64 })
        #expect(plan.layers.allSatisfy { $0.expertStride == 5_308_416 })
        #expect(plan.layers.allSatisfy { $0.expertStride % 16_384 == 0 })
        #expect(plan.layers.allSatisfy { $0.subTensors.count == 9 })
    }

    @Test func lfmPlanClassifiesF32RouterAndRenamedAttentionOutput() throws {
        let plan = try lfmPlan()
        let router = try #require(
            plan.resident.entries.first {
                $0.name == "layers.2.feed_forward.gate.weight"
            })
        #expect(router.dtype == 3)
        #expect(router.quantSpec == nil)
        #expect(router.logicalShape4.prefix(2) == [64, 2_048])
        let bias = try #require(
            plan.resident.entries.first {
                $0.name == "layers.2.feed_forward.expert_bias"
            })
        #expect(bias.dtype == 3)
        #expect(bias.quantSpec == nil)

        let names = Set(plan.resident.entries.map(\.name))
        #expect(names.contains("layers.2.self_attn.q_proj.weight"))
        #expect(names.contains("layers.2.self_attn.k_proj.weight"))
        #expect(names.contains("layers.2.self_attn.v_proj.weight"))
        #expect(names.contains("layers.2.self_attn.o_proj.weight"))
        #expect(!names.contains("layers.2.self_attn.out_proj.weight"))
        #expect(names.contains("layers.2.self_attn.q_layernorm.weight"))
        #expect(names.contains("layers.2.self_attn.k_layernorm.weight"))

        #expect(names.contains("layers.0.conv.in_proj.weight"))
        #expect(names.contains("layers.0.conv.out_proj.weight"))
        #expect(names.contains("layers.0.conv.conv.weight"))
        #expect(names.contains("model.embedding_norm.weight"))
        #expect(names.contains("layers.0.operator_norm.weight"))
        #expect(names.contains("layers.0.ffn_norm.weight"))
    }

    @Test func lfmPlanHasNoDenseFFNOrExpertFileForDenseLayers() throws {
        let plan = try lfmPlan()
        let names = Set(plan.resident.entries.map(\.name))
        #expect(!names.contains("layers.0.feed_forward.gate.weight"))
        #expect(!names.contains("layers.0.feed_forward.expert_bias"))
        #expect(!names.contains("layers.1.feed_forward.gate.weight"))
        #expect(!names.contains("layers.1.feed_forward.expert_bias"))
        #expect(plan.layers.allSatisfy { (2..<40).contains($0.layerIndex) })
    }

    @Test func lfmManifestWritesFamilyTraitsAndAbsentRouterSlot() throws {
        let plan = try lfmPlan()
        let data = try GTurboJSON.encodeManifest(
            plan: plan, modelID: "LiquidAI/LFM2.5-24B-A2B-MLX-4bit",
            sourceSnapshotHash: "sha256:test", files: [],
            expertsPerLayer: 64, numLayers: 40,
            expertStride: 5_308_416,
            bitWidths: .init(embedding: 6, attention: 4, router: 8,
                             sharedExpert: 4, routedExpert: 4))
        let wire = try GTurboManifestCodec.decode(data)
        #expect(wire.arch.family == .lfm2Moe)
        #expect(wire.arch.layerTypes?.count == 40)
        #expect(wire.arch.moeLayerMask?.prefix(2) == [0, 0])
        #expect(wire.arch.moeLayerMask?.suffix(38).allSatisfy { $0 == 1 } == true)
        #expect(wire.arch.convKernelSize == 3)
        #expect(wire.arch.routerKind == .lfm2SigmoidTopKBias)
        #expect(wire.arch.sharedExpert == false)
        #expect(wire.arch.embeddingScaledBySqrtHidden == false)
        #expect(wire.arch.numDenseLayers == 2)
        #expect(wire.arch.routerNormalizeTopK == true)
        #expect(wire.arch.routerBias == true)
        #expect(wire.quant?.router == nil)
        #expect(wire.quant?.embedding.weightBits == 6)
        #expect(wire.quant?.attention.weightBits == 4)
        #expect(wire.quant?.sharedExpert.weightBits == 4)
        #expect(wire.quant?.routedExpert.weightBits == 4)
    }

    @Test func lfmLayoutHasNoDenseLayerEntries() throws {
        let plan = try lfmPlan()
        let mask = plan.arch.moeLayerMask?.map { Int($0) }
        let data = try GTurboJSON.encodeLayout(
            plan: plan, expertStride: 5_308_416)
        let layout = try GTurboPackedExpertsLayoutCodec.decode(
            data, moeLayerMask: mask)
        #expect(layout.layers.map(\.layer) == Array(2..<40))
        #expect(layout.numLayers == 40)
        #expect(layout.expertsPerLayer == 64)
        #expect(layout.expertStride == 5_308_416)
    }

    @Test func lfmLayoutValidatorUsesTheManifestMoELayerMask() throws {
        let plan = try lfmPlan()
        let data = try GTurboJSON.encodeLayout(
            plan: plan, expertStride: 5_308_416)
        let dir = temporaryRoot("layout")
        defer { try? FileManager.default.removeItem(atPath: dir) }
        let path = (dir as NSString).appendingPathComponent("layout.json")
        try data.write(to: URL(fileURLWithPath: path))
        try GTurboLayoutValidator.validate(path: path, plan: plan)
        // The same 38-entry layout must fail the all-ones default used by a
        // mask-unaware reader.
        #expect(throws: GTurboFormatError.self) {
            _ = try GTurboPackedExpertsLayoutCodec.decode(data)
        }
    }

    @Test func lfmPlanPinsInstalledByteTotal() throws {
        let plan = try lfmPlan()
        let total = plan.resident.totalSize
            + plan.layers.reduce(UInt64(0)) { $0 + $1.fileSize }
        #expect(plan.resident.entries.count == 314)
        #expect(plan.resident.indexSize == 32_768)
        #expect(plan.resident.residentSize == 657_121_280)
        #expect(total == 13_567_221_760)
    }

    @Test func lfmPlanRejectsAnUnexpectedTensor() throws {
        let fixture = try lfmSnapshot()
        defer { try? FileManager.default.removeItem(atPath: fixture.directory) }
        let lmHead = SourceTensor(
            name: "lm_head.weight", shardPath: fixture.shardPath,
            dtype: .u32, shape: [512, 8], absoluteOffset: 0, sizeBytes: 16_384)
        let header = Safetensors.Header(
            path: fixture.shardPath,
            payloadBaseOffset: fixture.headers[0].payloadBaseOffset,
            tensors: fixture.headers[0].tensors + [lmHead])
        #expect(throws: RepackError.self) {
            _ = try RepackPlanner.plan(
                meta: fixture.metadata, arch: fixture.arch,
                shardHeaders: [header], outputDir: fixture.outputDir)
        }
    }

    @Test func lfmPlanRejectsAMissingExpertTensor() throws {
        let fixture = try lfmSnapshot()
        defer { try? FileManager.default.removeItem(atPath: fixture.directory) }
        let missing = "model.layers.2.feed_forward.switch_mlp.down_proj.scales"
        let header = Safetensors.Header(
            path: fixture.shardPath,
            payloadBaseOffset: fixture.headers[0].payloadBaseOffset,
            tensors: fixture.headers[0].tensors.filter { $0.name != missing })
        #expect(throws: RepackError.self) {
            _ = try RepackPlanner.plan(
                meta: fixture.metadata, arch: fixture.arch,
                shardHeaders: [header], outputDir: fixture.outputDir)
        }
    }

    // MARK: - Fixtures

    private struct Fixture {
        let directory: String
        let outputDir: String
        let shardPath: String
        let metadata: IndexLoader.SourceMetadata
        let arch: ArchInfo
        let headers: [Safetensors.Header]
    }

    private func temporaryRoot(_ tag: String) -> String {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent(
                "turbofieldfare-lfm-plan-\(tag)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            atPath: path, withIntermediateDirectories: true)
        return path
    }

    private func lfmSnapshot() throws -> Fixture {
        let directory = temporaryRoot("snapshot")
        let snapshot = try SyntheticSnapshot.buildLFM(at: directory)
        return Fixture(
            directory: directory,
            outputDir: (directory as NSString).appendingPathComponent("out"),
            shardPath: snapshot.shardPath,
            metadata: snapshot.metadata,
            arch: snapshot.arch,
            headers: snapshot.headers)
    }

    private func lfmPlan() throws -> RepackPlan {
        let fixture = try lfmSnapshot()
        defer { try? FileManager.default.removeItem(atPath: fixture.directory) }
        return try RepackPlanner.plan(
            meta: fixture.metadata, arch: fixture.arch,
            shardHeaders: fixture.headers, outputDir: fixture.outputDir)
    }
}
