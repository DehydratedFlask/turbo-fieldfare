import Foundation
import Metal
import Testing

@testable import TurboFieldfare
@testable import TurboFieldfareFormat

@Suite struct RunnerFactoryTests {
    @Test func factoryReturnsGemmaRunnerForGemmaFamily() throws {
        let model = try loadSyntheticGemma()
        let runner = try RunnerFactory.make(model: model,
                                            context: try MetalContext(),
                                            maxContext: 4_096)
        #expect(runner is RealForwardRunner)
        #expect(runner.maxContext == 4_096)
    }

    /// Loads the tiny Gemma model the loader tests synthesize. No real model
    /// checkpoint is touched.
    private func loadSyntheticGemma() throws -> Model {
        let directory = try ModelLoaderTests.writeToySynthetic()
        let device = try #require(MTLCreateSystemDefaultDevice())
        return try Model.load(directoryURL: directory,
                              device: device,
                              expecting: .gemma4Toy())
    }
}
