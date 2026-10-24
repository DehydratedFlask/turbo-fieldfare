import Testing
import Foundation
import Darwin
@testable import TurboFieldfare
@testable import TurboFieldfareFormat

@Suite struct ManifestReaderTests {

    @Test func rejectsManifestFIFOWithoutBlocking() throws {
        let (dir, toy) = try Self.writeToyManifest()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = dir.appendingPathComponent("manifest.json")
        try FileManager.default.removeItem(at: manifest)
        #expect(mkfifo(manifest.path, 0o600) == 0)

        #expect(throws: ModelError.self) {
            try ManifestReader.load(directoryURL: dir, expecting: toy)
        }
    }

    /// Build a manifest dictionary for a 2-layer toy ArchConfig and write it
    /// into a temp directory. Returns the directory URL and the toy config.
    static func writeToyManifest(_ overrides: [String: Any] = [:],
                                 flags: [String: Bool] = ["streamingPresent": true,
                                                          "turboQuantKV": false,
                                                          "aneSharedExpert": false],
                                 archOverrides: [String: Any] = [:],
                                 filesOverride: [String: [String: Any]]? = nil,
                                 config: ArchConfig = .gemma4Toy()) throws
                                 -> (URL, ArchConfig) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-manifest-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("packed_experts"),
            withIntermediateDirectories: true)

        let toy = config
        var archDict: [String: Any] = [
            "hiddenSize": toy.hiddenSize,
            "ffnIntermediate": toy.intermediateSize,
            "moeIntermediateSize": toy.moeIntermediateSize,
            "numHeads": toy.numHeads,
            "numKVHeads": toy.numKVHeads,
            "numFullKVHeads": toy.numFullKVHeads,
            "headDim": toy.headDim,
            "fullHeadDim": toy.fullHeadDim,
            "vocabSize": toy.vocabSize,
            "slidingWindow": toy.slidingWindow,
            "finalLogitSoftcap": toy.finalLogitSoftcap,
            "ropeTheta": toy.ropeTheta,
            "fullRopeTheta": toy.fullRopeTheta,
            "partialRotaryFactor": toy.partialRotaryFactor,
            "numLayers": toy.numLayers,
            "numExperts": toy.numExperts,
            "topKExperts": toy.topKExperts,
            "tieWordEmbeddings": toy.tieWordEmbeddings,
            "attentionKEqV": toy.attentionKEqV,
            "hiddenActivation": toy.hiddenActivation,
            "fullAttentionLayerMask": toy.fullAttentionLayerMask.map { Int($0) },
        ]
        for (k, v) in archOverrides { archDict[k] = v }

        var files: [String: [String: Any]]
        if let f = filesOverride {
            files = f
        } else {
            files = [
                "model_weights.bin": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
                "packed_experts/layout.json": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            ]
            for L in 0..<toy.numLayers {
                files["packed_experts/layer_\(L).bin"] = ["size": 16384, "sha256": String(repeating: "0", count: 64)]
            }
        }

        var root: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": flags,
            "modelID": "toy",
            "arch": archDict,
            "files": files,
            "expertsPerLayer": toy.numExperts,
            "numLayers": toy.numLayers,
            "expertStride": Int(toy.admission.expertStrideBytes),
        ]
        for (k, v) in overrides { root[k] = v }

        let data = try JSONSerialization.data(withJSONObject: root,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        try data.write(to: dir.appendingPathComponent("manifest.json"))
        return (dir, toy)
    }

    static func quant(sharedExpertBits: Int = 4,
                      routerBits: Int = 8) -> [String: Any] {
        func slot(_ bits: Int) -> [String: Any] {
            [
                "weightBits": bits,
                "scheme": "affine",
                "scaleType": "bf16",
                "biasType": "bf16",
                "groupSize": Quantization.groupSize,
            ]
        }
        return [
            "embedding": slot(4),
            "attention": slot(4),
            "router": slot(routerBits),
            "sharedExpert": slot(sharedExpertBits),
            "routedExpert": slot(4),
        ]
    }

    @Test func loadsValidManifest() throws {
        let (dir, toy) = try Self.writeToyManifest()
        defer { try? FileManager.default.removeItem(at: dir) }
        let m = try ManifestReader.load(directoryURL: dir, expecting: toy)
        #expect(m.magic == "GTURBO")
        #expect(m.numLayers == toy.numLayers)
        #expect(m.expertStride == 16384)
    }

    @Test func missingManifestThrowsPartialInstall() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: .gemma4Toy())
        } throws: { error in
            if case ModelError.partialInstall = error { return true }
            return false
        }
    }

    @Test func oversizedManifestRejectsBeforeDecode() throws {
        let (dir, toy) = try Self.writeToyManifest()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifestURL = dir.appendingPathComponent("manifest.json")
        try Data(repeating: 0x20, count: 64).write(to: manifestURL)

        #expect {
            _ = try ManifestReader.load(directoryURL: dir,
                                        expecting: toy,
                                        maxBytes: 16)
        } throws: { error in
            if case ModelError.indexCorrupt(let detail) = error {
                return detail.contains("metadata cap")
            }
            return false
        }
    }

    @Test func wrongMagicThrowsNotAGTurboDirectory() throws {
        let (dir, toy) = try Self.writeToyManifest(["magic": "NOT_GTURBO"])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: ModelError.notAGTurboDirectory) {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        }
    }

    @Test func versionTwoThrowsUnsupportedVersion() throws {
        let (dir, toy) = try Self.writeToyManifest(["versionMajor": 2])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.unsupportedVersion(let maj, _) = error { return maj == 2 }
            return false
        }
    }

    @Test func unknownFlagThrowsUnknownFlag() throws {
        let (dir, toy) = try Self.writeToyManifest(flags: ["streamingPresent": true,
                                                           "newFangledOption": true])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.unknownFlag(let n) = error { return n == "newFangledOption" }
            return false
        }
    }

    @Test func removedTurboQuantFlagIsRejected() throws {
        let (dir, toy) = try Self.writeToyManifest(flags: ["streamingPresent": true,
                                                           "turboQuantKV": true,
                                                           "aneSharedExpert": false])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("removed TurboQuant KV")
        }
    }

    @Test func productionManifestRequiresQuantMetadata() throws {
        let (dir, config) = try Self.writeToyManifest(config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("manifest.quant is required")
        }
    }

    @Test func productionManifestAcceptsInt4SharedExpert() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(sharedExpertBits: 4)],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        #expect(manifest.quant?.sharedExpert.weightBits == 4)
    }

    @Test func productionManifestAcceptsHistoricalInt8SharedExpert() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(sharedExpertBits: 8)],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        #expect(manifest.quant?.sharedExpert.weightBits == 8)
    }

    @Test func productionManifestRejectsUnsupportedQuantMetadata() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(sharedExpertBits: 3, routerBits: 4)],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("unsupported quantization")
        }
    }

    @Test func archMismatchThrowsArchMismatch() throws {
        let (dir, toy) = try Self.writeToyManifest(archOverrides: ["hiddenSize": 4096])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case let ModelError.archMismatch(field, _, _) = error else { return false }
            return field == "hiddenSize"
        }
    }

    @Test func nonPageAlignedExpertStrideThrows() throws {
        let (dir, toy) = try Self.writeToyManifest(["expertStride": 1024])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.expertStrideNotPageAligned = error { return true }
            return false
        }
    }

    @Test func defersPackedLayerFilenamesToLayoutValidation() throws {
        let files: [String: [String: Any]] = [
            "model_weights.bin": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layout.json": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/experts_a.bin": ["size": 16384, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/experts_b.bin": ["size": 16384, "sha256": String(repeating: "0", count: 64)],
        ]
        let (dir, toy) = try Self.writeToyManifest(filesOverride: files)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: toy)
        #expect(manifest.files["packed_experts/experts_a.bin"] != nil)
    }

    @Test func acceptsZeroPaddedLayerFilenames() throws {
        // Writer emits packed_experts/layer_%02d.bin; loader should accept either form.
        let files: [String: [String: Any]] = [
            "model_weights.bin": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layout.json": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layer_00.bin": ["size": 16384, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layer_01.bin": ["size": 16384, "sha256": String(repeating: "0", count: 64)],
        ]
        let (dir, toy) = try Self.writeToyManifest(filesOverride: files)
        defer { try? FileManager.default.removeItem(at: dir) }
        let m = try ManifestReader.load(directoryURL: dir, expecting: toy)
        #expect(m.numLayers == toy.numLayers)
    }

    @Test func peekFamilyDefaultsToGemmaWhenAbsent() throws {
        let dir = try writeManifest(makeGemmaWireWithoutFamily())
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try ManifestReader.peekFamily(directoryURL: dir) == .gemma4)
    }

    @Test func lfmManifestRejectsGemmaConstants() throws {
        let data = try GTurboManifestCodec.encode(makeLFMWireWithGemmaHiddenSize())
        #expect(throws: ModelError.self) {
            try ManifestReader.decode(data: data, expecting: .lfm2_5_24B_A2B)
        }
    }

    @Test func expertStrideMustMatchFamilyProfile() throws {
        let data = try GTurboManifestCodec.encode(makeLFMWire(expertStride: 3_358_720))
        #expect(throws: ModelError.self) {
            try ManifestReader.decode(data: data, expecting: .lfm2_5_24B_A2B)
        }
    }
}

private let zeroSHA = String(repeating: "0", count: 64)

private func wireQuantSlot(_ bits: Int) -> GTurboManifestQuantSlotV1 {
    GTurboManifestQuantSlotV1(weightBits: bits, scheme: "affine",
                              scaleType: "bf16", biasType: "bf16",
                              groupSize: Quantization.groupSize)
}

private func wireFiles() -> [String: GTurboManifestFileV1] {
    [
        "model_weights.bin": GTurboManifestFileV1(size: 1024, sha256: zeroSHA),
        "packed_experts/layout.json": GTurboManifestFileV1(size: 1024, sha256: zeroSHA),
    ]
}

/// Encode a wire manifest into a fresh temp directory so directory-reading
/// paths (`peekFamily`, `load`) can be exercised.
private func writeManifest(_ wire: GTurboManifestV1) throws -> URL {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("gturbo-wire-manifest-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    try GTurboManifestCodec.encode(wire)
        .write(to: dir.appendingPathComponent("manifest.json"))
    return dir
}

private let lfmFullAttentionLayers = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38]

/// A manifest with no `arch.family`, i.e. the Gemma-defaulting shape every
/// pre-LFM checkpoint has.
private func makeGemmaWireWithoutFamily() -> GTurboManifestV1 {
    GTurboManifestV1(
        flags: ["streamingPresent": true, "turboQuantKV": false,
                "aneSharedExpert": false],
        modelID: "test/gemma",
        sourceSnapshotHash: nil,
        arch: GTurboManifestArchV1(
            hiddenSize: 64, ffnIntermediate: 128, moeIntermediateSize: 32,
            numHeads: 4, numKVHeads: 2, numFullKVHeads: 1,
            headDim: 16, fullHeadDim: 32, vocabSize: 1024,
            slidingWindow: 128, finalLogitSoftcap: 30,
            ropeTheta: 10_000, fullRopeTheta: 1_000_000,
            partialRotaryFactor: 0.25, numLayers: 2, numExperts: 8,
            topKExperts: 2, tieWordEmbeddings: true, attentionKEqV: true,
            hiddenActivation: "gelu_pytorch_tanh",
            fullAttentionLayerMask: [0, 1]),
        quant: GTurboManifestQuantV1(
            embedding: wireQuantSlot(4), attention: wireQuantSlot(4),
            router: wireQuantSlot(8), sharedExpert: wireQuantSlot(4),
            routedExpert: wireQuantSlot(4)),
        files: wireFiles(),
        expertsPerLayer: 8, numLayers: 2, expertStride: 16_384,
        bitWidthOverridesHonored: nil)
}

/// LFM2.5-shaped wire matching `ArchConfig.lfm2_5_24B_A2B`, with the two
/// overridable fields the mismatch tests need.
private func makeLFMWire(expertStride: UInt64 = 5_308_416,
                         hiddenSize: Int = 2048) -> GTurboManifestV1 {
    let layerTypes: [LayerType] = (0..<40).map {
        lfmFullAttentionLayers.contains($0) ? .fullAttention : .conv
    }
    return GTurboManifestV1(
        flags: ["streamingPresent": true, "turboQuantKV": false,
                "aneSharedExpert": false],
        modelID: "test/lfm",
        sourceSnapshotHash: nil,
        arch: GTurboManifestArchV1(
            hiddenSize: hiddenSize, ffnIntermediate: 11_776,
            moeIntermediateSize: 1536,
            numHeads: 32, numKVHeads: 8, numFullKVHeads: 8,
            headDim: 64, fullHeadDim: 64, vocabSize: 128_000,
            slidingWindow: 0, finalLogitSoftcap: 0,
            ropeTheta: 10_000_000, fullRopeTheta: 10_000_000,
            partialRotaryFactor: 1, numLayers: 40, numExperts: 64,
            topKExperts: 4, tieWordEmbeddings: true, attentionKEqV: false,
            hiddenActivation: "silu",
            fullAttentionLayerMask: layerTypes.map {
                $0 == .fullAttention ? 1 : 0
            },
            family: .lfm2Moe,
            layerTypes: layerTypes,
            moeLayerMask: Array(repeating: 0, count: 2)
                + Array(repeating: 1, count: 38),
            attentionScale: 0.125,
            rmsNormUnitOffset: false,
            ropeStyle: .neoxFull,
            convKernelSize: 3,
            routerKind: .lfm2SigmoidTopKBias,
            sharedExpert: false,
            embeddingScaledBySqrtHidden: false,
            numDenseLayers: 2,
            routerNormalizeTopK: true,
            routerBias: true,
            routedScalingFactor: 1),
        quant: GTurboManifestQuantV1(
            embedding: wireQuantSlot(6), attention: wireQuantSlot(4),
            sharedExpert: wireQuantSlot(4), routedExpert: wireQuantSlot(4)),
        files: wireFiles(),
        expertsPerLayer: 64, numLayers: 40, expertStride: expertStride,
        bitWidthOverridesHonored: nil)
}

private func makeLFMWireWithGemmaHiddenSize() -> GTurboManifestV1 {
    makeLFMWire(hiddenSize: 2816)
}

extension ArchConfig {
    /// Tiny baseline used across the loader tests. 2 layers (both full), hidden 64,
    /// vocab 1024, 8 experts. Numbers are intentionally toy.
    static func gemma4Toy() -> ArchConfig {
        ArchConfig(
            hiddenSize: 64,
            intermediateSize: 256,
            moeIntermediateSize: 128,
            numHeads: 4,
            numKVHeads: 2,
            numFullKVHeads: 1,
            headDim: 16,
            fullHeadDim: 32,
            vocabSize: 1024,
            slidingWindow: 256,
            finalLogitSoftcap: 30.0,
            ropeTheta: 10_000.0,
            fullRopeTheta: 1_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 2,
            numExperts: 8,
            topKExperts: 2,
            tieWordEmbeddings: true,
            attentionKEqV: true,
            fullAttentionLayerMask: [0, 1],
            hiddenActivation: "gelu_pytorch_tanh",
            // Toy manifests use a 16 KB expert stride, so the admission
            // profile must match it rather than the production Gemma value.
            admission: ArchAdmissionProfile(
                expertStrideBytes: 16_384,
                expertCacheLayers: 2,
                includedExpertCacheSlots: 0,
                nonKVRuntimeFootprintBytes: 0)
        )
    }
}
