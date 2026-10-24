import TurboFieldfareFormat

/// Which production runner a model family selects. The routing table is split
/// out so it can be tested without loading a model checkpoint.
enum RunnerFactoryPlan: Equatable, Sendable {
    case gemma4
    case lfm2Moe
}

/// Builds the production text runner for a loaded model.
public enum RunnerFactory {
    /// The runner family a manifest family selects. Unknown families fail
    /// closed instead of trapping.
    static func plan(for family: ModelFamily) throws -> RunnerFactoryPlan {
        switch family {
        case .gemma4:
            return .gemma4
        case .lfm2Moe:
            return .lfm2Moe
        @unknown default:
            throw ModelError.unsupportedFamily(family.rawValue)
        }
    }

    /// Returns the runner that matches `model.config.family`. Unknown and
    /// not-yet-implemented families fail closed instead of trapping.
    public static func make(model: Model,
                            context: MetalContext,
                            maxContext: Int,
                            runtimeConfiguration: RuntimeConfiguration = .production)
        throws -> any TextRunner {
        switch try plan(for: model.config.family) {
        case .gemma4:
            return try RealForwardRunner(model: model,
                                         context: context,
                                         maxContext: maxContext,
                                         runtimeConfiguration: runtimeConfiguration)
        case .lfm2Moe:
            return try LFM2ForwardRunner(model: model,
                                         context: context,
                                         maxContext: maxContext,
                                         runtimeConfiguration: runtimeConfiguration)
        }
    }
}
