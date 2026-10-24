/// Builds the production text runner for a loaded model.
public enum RunnerFactory {
    /// Returns the runner that matches `model.config.family`. Unknown and
    /// not-yet-implemented families fail closed instead of trapping.
    public static func make(model: Model,
                            context: MetalContext,
                            maxContext: Int,
                            runtimeConfiguration: RuntimeConfiguration = .production)
        throws -> any TextRunner {
        switch model.config.family {
        case .gemma4:
            return try RealForwardRunner(model: model,
                                         context: context,
                                         maxContext: maxContext,
                                         runtimeConfiguration: runtimeConfiguration)
        case .lfm2Moe:
            throw ModelError.unsupportedFamily("lfm2_moe runner is not available")
        @unknown default:
            throw ModelError.unsupportedFamily(model.config.family.rawValue)
        }
    }
}
