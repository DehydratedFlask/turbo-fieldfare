import Testing
import Foundation
import Darwin
@testable import TurboFieldfare

@Suite struct PackedExpertsLayoutTests {

    @Test func rejectsLayoutFIFOWithoutBlocking() throws {
        let dir = try Self.writeToyLayout()
        defer { try? FileManager.default.removeItem(at: dir) }
        let layout = dir.appendingPathComponent("packed_experts/layout.json")
        try FileManager.default.removeItem(at: layout)
        #expect(mkfifo(layout.path, 0o600) == 0)

        #expect(throws: ModelError.self) {
            try PackedExpertsLayoutReader.load(directoryURL: dir)
        }
    }

    /// Hand-write a tiny layout.json with one layer, two experts, two
    /// sub-tensors each. Returns the directory URL.
    static func writeToyLayout() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-layout-test-\(UUID().uuidString)")
        let exp = dir.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: exp, withIntermediateDirectories: true)

        let root: [String: Any] = [
            "expertStride": 16384,
            "numLayers": 1,
            "expertsPerLayer": 2,
            "layers": [
                [
                    "layer": 0,
                    "file": "layer_00.bin",
                    "experts": [
                        [
                            "expert": 0,
                            "offset": 0,
                            "size": 16384,
                            "tensors": [
                                "gate": [
                                    "offset": 0,
                                    "size": 4096,
                                    "dtype": "U32",
                                    "shape": [64, 64],
                                    "bits": 4
                                ],
                                "gate_scales": [
                                    "offset": 4096,
                                    "size": 256,
                                    "dtype": "BF16",
                                    "shape": [64, 1]
                                ],
                            ],
                        ],
                        [
                            "expert": 1,
                            "physicalRank": 1,
                            "offset": 16384,
                            "size": 16384,
                            "tensors": [
                                "gate": [
                                    "offset": 0,
                                    "size": 4096,
                                    "dtype": "U32",
                                    "shape": [64, 64],
                                    "bits": 4
                                ],
                                "gate_scales": [
                                    "offset": 4096,
                                    "size": 256,
                                    "dtype": "BF16",
                                    "shape": [64, 1]
                                ],
                            ],
                        ],
                    ],
                ],
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
        try data.write(to: exp.appendingPathComponent("layout.json"))
        return dir
    }

    @Test func decodesToyLayout() throws {
        let dir = try Self.writeToyLayout()
        defer { try? FileManager.default.removeItem(at: dir) }
        let layout = try PackedExpertsLayoutReader.load(directoryURL: dir)
        #expect(layout.expertStride == 16384)
        #expect(layout.numLayers == 1)
        #expect(layout.expertsPerLayer == 2)
        #expect(layout.layers.count == 1)
        let exp0 = try layout.expert(layer: 0, expert: 0)
        #expect(exp0.offset == 0)
        #expect(exp0.size == 16384)
        let gate = try #require(exp0.subTensors["gate"])
        #expect(gate.offset == 0)
        #expect(exp0.subTensors["gate_scales"]?.offset == 4096)
        let exp1 = try layout.expert(layer: 0, expert: 1)
        #expect(exp1.offset == 16384)
        #expect(exp1.expert == 1)
    }

    @Test func layoutWithDenseLayersRejectsDenseLayerLookup() throws {
        let layout = try decodeLayout(makeLFMLayout(moeLayers: [2, 3], numLayers: 4))
        #expect(layout.layers.map(\.layer) == [2, 3])
        #expect(throws: ModelError.routedExpertsUnavailable(layer: 0)) {
            _ = try layout.expert(layer: 0, expert: 0)
        }
        #expect(throws: ModelError.routedExpertsUnavailable(layer: 1)) {
            _ = try layout.layerLayout(for: 1)
        }
        #expect(throws: ModelError.routedExpertsUnavailable(layer: 4)) {
            _ = try layout.layerLayout(for: 4)
        }
        #expect(try layout.expert(layer: 2, expert: 0).expert == 0)
    }

    @Test func missingLayoutJsonThrowsMissingFile() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-no-layout-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("packed_experts"),
            withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try PackedExpertsLayoutReader.load(directoryURL: dir)
        } throws: { error in
            if case ModelError.missingFile = error { return true }
            return false
        }
    }

    @Test func oversizedLayoutRejectsBeforeDecode() throws {
        let dir = try Self.writeToyLayout()
        defer { try? FileManager.default.removeItem(at: dir) }
        let layoutURL = dir
            .appendingPathComponent("packed_experts")
            .appendingPathComponent("layout.json")
        try Data(repeating: 0x20, count: 64).write(to: layoutURL)

        #expect {
            _ = try PackedExpertsLayoutReader.load(directoryURL: dir,
                                                   maxBytes: 16)
        } throws: { error in
            if case ModelError.indexCorrupt(let detail) = error {
                return detail.contains("metadata cap")
            }
            return false
        }
    }
}

// MARK: - Dense-layer fixtures

private struct LayoutFixture {
    let data: Data
    let moeLayerMask: [Int]
    let numLayers: Int
    let expertsPerLayer: Int
    let expertStride: UInt64
}

/// A layout.json whose `layers` array holds only `moeLayers`, matching the
/// LFM2.5 shape where layers 0-1 are dense.
private func makeLFMLayout(moeLayers: [Int], numLayers: Int) throws -> LayoutFixture {
    let expertStride: UInt64 = 16384
    let expertsPerLayer = 1
    let tensor: [String: Any] = [
        "offset": 0, "size": 4096, "dtype": "U32", "shape": [64, 64], "bits": 4,
    ]
    let scaleTensor: [String: Any] = [
        "offset": 4096, "size": 256, "dtype": "BF16", "shape": [64, 1],
    ]
    let layers: [[String: Any]] = moeLayers.map { layer in
        [
            "layer": layer,
            "file": String(format: "layer_%02d.bin", layer),
            "experts": [
                [
                    "expert": 0,
                    "physicalRank": 0,
                    "offset": 0,
                    "size": expertStride,
                    "tensors": ["gate": tensor, "gate_scales": scaleTensor],
                ],
            ],
        ]
    }
    let root: [String: Any] = [
        "expertStride": expertStride,
        "numLayers": numLayers,
        "expertsPerLayer": expertsPerLayer,
        "layers": layers,
    ]
    let mask = (0..<numLayers).map { moeLayers.contains($0) ? 1 : 0 }
    return LayoutFixture(
        data: try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys]),
        moeLayerMask: mask,
        numLayers: numLayers,
        expertsPerLayer: expertsPerLayer,
        expertStride: expertStride)
}

private func decodeLayout(_ fixture: LayoutFixture) throws -> PackedExpertsLayout {
    let layerFiles = (0..<fixture.numLayers)
        .filter { fixture.moeLayerMask[$0] == 1 }
        .map { "packed_experts/" + String(format: "layer_%02d.bin", $0) }
    let fileSize = UInt64(fixture.expertsPerLayer) * fixture.expertStride
    let files = Dictionary(uniqueKeysWithValues: layerFiles.map {
        ($0, ManifestFileEntry(size: fileSize, sha256: ""))
    })
    let manifest = Manifest(
        magic: "GTURBO", versionMajor: 1, versionMinor: 0,
        flags: [:], modelID: "fixture/lfm", sourceSnapshotHash: nil,
        arch: ManifestArch(
            hiddenSize: 64, ffnIntermediate: 128, moeIntermediateSize: 32,
            numHeads: 4, numKVHeads: 2, numFullKVHeads: 2,
            headDim: 16, fullHeadDim: 16, vocabSize: 1024,
            slidingWindow: 0, finalLogitSoftcap: 0,
            ropeTheta: 10_000, fullRopeTheta: 10_000,
            partialRotaryFactor: 1,
            numLayers: fixture.numLayers, numExperts: 2, topKExperts: 1,
            tieWordEmbeddings: true, attentionKEqV: false,
            hiddenActivation: "silu",
            fullAttentionLayerMask: Array(repeating: 0, count: fixture.numLayers),
            family: .lfm2Moe, layerTypes: nil,
            moeLayerMask: fixture.moeLayerMask,
            attentionScale: nil, rmsNormUnitOffset: nil, ropeStyle: nil,
            convKernelSize: nil, routerKind: nil, sharedExpert: nil,
            embeddingScaledBySqrtHidden: nil, numDenseLayers: nil,
            routerNormalizeTopK: nil, routerBias: nil,
            routedScalingFactor: nil),
        quant: nil,
        files: files,
        expertsPerLayer: fixture.expertsPerLayer,
        numLayers: fixture.numLayers,
        expertStride: fixture.expertStride)
    return try PackedExpertsLayoutReader.decode(data: fixture.data, manifest: manifest)
}
