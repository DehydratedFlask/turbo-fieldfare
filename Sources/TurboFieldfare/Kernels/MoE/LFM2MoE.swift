import Foundation
import Metal

/// LFM2.5 top-4 routed-expert decode path.
///
/// Mirrors `MoE`'s phase API and argument-buffer layout, but streams exactly
/// four experts and applies SiLU gating. The installed LFM2.5-8B-A1B pack has
/// 64 experts with hidden 2048 and routed intermediate 1536; the wrapper stays
/// dimension-generic so the same code serves prefill-sized fixtures.
final class LFM2MoE {
    static let maxStreamedExperts = 4
    static let topK: UInt32 = 4

    private let phase1PSO: MTLComputePipelineState
    private let phase2PSO: MTLComputePipelineState
    private let routedArgEncoder: MTLArgumentEncoder
    private let reusableRoutedArgBuffer: MTLBuffer

    init(context: MetalContext) throws {
        self.phase1PSO = try context.pipeline("lfm2_moe_phase1_top4")
        self.phase2PSO = try context.pipeline("lfm2_moe_phase2_top4")

        guard let phase1Function = context.library.makeFunction(
                name: "lfm2_moe_phase1_top4") else {
            throw MetalError.missingFunction("lfm2_moe_phase1_top4")
        }
        self.routedArgEncoder = phase1Function.makeArgumentEncoder(bufferIndex: 0)
        guard let reusable = context.device.makeBuffer(
            length: routedArgEncoder.encodedLength,
            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.reusableRoutedArgBuffer = reusable
    }

    func makeRoutedArgumentBuffer(routedBlobs: [MTLBuffer],
                                  topK: UInt32) -> MTLBuffer? {
        validate(routedBlobs: routedBlobs, topK: topK)
        guard let buffer = routedBlobs.first?.device.makeBuffer(
            length: routedArgEncoder.encodedLength,
            options: .storageModeShared) else {
            return nil
        }
        encodeRoutedArgumentBuffer(buffer, routedBlobs: routedBlobs)
        return buffer
    }

    func makeReusedRoutedArgumentBuffer(routedBlobs: [MTLBuffer],
                                        topK: UInt32) -> MTLBuffer {
        validate(routedBlobs: routedBlobs, topK: topK)
        encodeRoutedArgumentBuffer(reusableRoutedArgBuffer, routedBlobs: routedBlobs)
        return reusableRoutedArgBuffer
    }

    /// `acts[slot * f + i] = silu(gate_e(x)[i]) * up_e(x)[i]` for the four
    /// streamed slots.
    func encodePhase1(commandBuffer: MTLCommandBuffer,
                      routedArgBuffer: MTLBuffer,
                      routedBlobs: [MTLBuffer],
                      routedOffsets: MoEExpertOffsets,
                      x: MTLBuffer,
                      acts: MTLBuffer,
                      d: UInt32,
                      f: UInt32,
                      topK: UInt32) {
        validate(routedBlobs: routedBlobs, topK: topK)
        var dimension = d
        var intermediate = f
        var expertCount = topK
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(phase1PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for buffer in routedBlobs { encoder.useResource(buffer, usage: .read) }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(topK * f) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// `y[d] = residual[d] + sum_slot routingWeights[slot] * down_e(acts[slot])`.
    func encodePhase2(commandBuffer: MTLCommandBuffer,
                      routedArgBuffer: MTLBuffer,
                      routedBlobs: [MTLBuffer],
                      routedOffsets: MoEExpertOffsets,
                      acts: MTLBuffer,
                      routingWeights: MTLBuffer,
                      residual: MTLBuffer,
                      y: MTLBuffer,
                      d: UInt32,
                      f: UInt32,
                      topK: UInt32) {
        validate(routedBlobs: routedBlobs, topK: topK)
        var dimension = d
        var intermediate = f
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(phase2PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for buffer in routedBlobs { encoder.useResource(buffer, usage: .read) }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(acts, offset: 0, index: 2)
        encoder.setBuffer(routingWeights, offset: 0, index: 3)
        encoder.setBuffer(residual, offset: 0, index: 4)
        encoder.setBuffer(y, offset: 0, index: 5)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(d), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * Self.maxStreamedExperts,
                                           height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func validate(routedBlobs: [MTLBuffer], topK: UInt32) {
        precondition(topK == Self.topK, "LFM2 MoE streams exactly 4 experts")
        precondition(routedBlobs.count == Int(topK),
                     "expected \(topK) routed blobs, got \(routedBlobs.count)")
    }

    private func encodeRoutedArgumentBuffer(_ buffer: MTLBuffer,
                                            routedBlobs: [MTLBuffer]) {
        routedArgEncoder.setArgumentBuffer(buffer, offset: 0)
        for (index, blob) in routedBlobs.enumerated() {
            routedArgEncoder.setBuffer(blob, offset: 0, index: index)
        }
    }
}
