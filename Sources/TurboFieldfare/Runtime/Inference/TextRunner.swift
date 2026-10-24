/// Reports whether a runner's final projection can emit a greedy token itself
/// instead of leaving sampled logits in the caller's buffer.
public protocol FusedGreedyHeadReporting: AnyObject, Sendable {
    var usesFusedGreedyHead: Bool { get }
    var lastGreedyToken: UInt32 { get }
}

/// Family-neutral decode, prefill, and continuation surface. Every supported
/// model family's production runner conforms to this protocol so generation
/// code never needs to know which architecture is loaded.
public protocol TextRunner: LogitProducer, ChunkedPrefillRunner,
                            ContinuableLogitProducer, ContextWindowReporting,
                            FusedGreedyHeadReporting, Sendable {
    func rewind(to position: Int) throws
}
