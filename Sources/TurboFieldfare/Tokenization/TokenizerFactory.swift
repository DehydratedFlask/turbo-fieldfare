import Foundation
import TurboFieldfareFormat

/// Loads the tokenizer that matches an installed pack's manifest family.
public enum TokenizerFactory {
    public static func load(family: ModelFamily,
                            forModelDirectory modelDirectory: URL) async throws -> any ModelTokenizer {
        switch family {
        case .gemma4:
            return try await GFTokenizer.load(forModelDirectory: modelDirectory)
        case .lfm2Moe:
            return try await LFM2Tokenizer.load(forModelDirectory: modelDirectory)
        }
    }
}
