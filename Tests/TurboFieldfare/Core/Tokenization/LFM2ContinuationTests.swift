import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfare

@Suite("LFM continuation planning")
struct LFM2ContinuationTests {
    private let user = ChatMessage(role: .user, content: "hi")
    private let assistant = ChatMessage(role: .assistant, content: "ok")
    private let tool = ChatMessage(role: .tool, content: "result")

    @Test func reusesOnlyAtMessageBoundary() {
        let cached = [user, assistant]
        let incoming = cached + [tool]
        let cachedIDs: [Int32] = [10, 11, 12]
        let full = cachedIDs + [124_900, 20, 21]

        let plan = LFM2ContinuationPlanner.plan(
            cachedMessages: cached,
            cachedTokenIDs: cachedIDs,
            incomingMessages: incoming,
            fullTokens: full)

        guard case .reuse(let prefix, let suffix) = plan else {
            Issue.record("expected reuse"); return
        }
        #expect(prefix == cachedIDs)
        #expect(suffix == [124_900, 20, 21])
    }

    @Test func reusesAtAnImStartBoundaryToo() {
        let cachedIDs: [Int32] = [10]
        let full = cachedIDs + [124_899, 20]
        let plan = LFM2ContinuationPlanner.plan(
            cachedMessages: [user],
            cachedTokenIDs: cachedIDs,
            incomingMessages: [user, assistant],
            fullTokens: full)
        guard case .reuse(_, let suffix) = plan else {
            Issue.record("expected reuse"); return
        }
        #expect(suffix == [124_899, 20])
    }

    @Test func fallsBackToReplayWhenTemplateDiffers() {
        let full: [Int32] = [1, 2, 3]
        let plan = LFM2ContinuationPlanner.plan(
            cachedMessages: [user],
            cachedTokenIDs: [],
            incomingMessages: [.init(role: .system, content: "new"), user],
            fullTokens: full)
        #expect(plan == .replayAll(full))
    }

    @Test func fallsBackWhenTheTokenPrefixDiverges() {
        let full: [Int32] = [9, 9, 9, 124_900]
        let plan = LFM2ContinuationPlanner.plan(
            cachedMessages: [user],
            cachedTokenIDs: [1, 2],
            incomingMessages: [user, assistant],
            fullTokens: full)
        #expect(plan == .replayAll(full))
    }

    @Test func fallsBackWhenTheSuffixIsNotABoundary() {
        let cachedIDs: [Int32] = [1, 2]
        let full = cachedIDs + [42, 124_900]
        let plan = LFM2ContinuationPlanner.plan(
            cachedMessages: [user],
            cachedTokenIDs: cachedIDs,
            incomingMessages: [user, assistant],
            fullTokens: full)
        #expect(plan == .replayAll(full))
    }

    @Test func fallsBackWhenCachedMessagesAreNotAPrefix() {
        let full: [Int32] = [1, 2, 124_900]
        let plan = LFM2ContinuationPlanner.plan(
            cachedMessages: [assistant],
            cachedTokenIDs: [1, 2],
            incomingMessages: [user, assistant],
            fullTokens: full)
        #expect(plan == .replayAll(full))
    }

    @Test(.enabled(if: InstalledTokenizerPacks.lfmDirectory != nil))
    func continuationReusesOnlyAtMessageBoundaryWithTheInstalledTokenizer() async throws {
        let directory = try #require(InstalledTokenizerPacks.lfmDirectory)
        let tokenizer = try await LFM2Tokenizer.load(forModelDirectory: directory)
        let cached = [user, assistant]
        let incoming = cached + [tool]
        let cachedIDs = try tokenizer.applyChatTemplate(
            messages: cached, tools: nil, addGenerationPrompt: false)
        let plan = try tokenizer.continuationPlan(
            cachedMessages: cached,
            cachedTokenIDs: cachedIDs,
            incomingMessages: incoming,
            tools: nil)
        guard case .reuse(let prefix, let suffix) = plan else {
            Issue.record("expected reuse"); return
        }
        #expect(prefix == cachedIDs)
        #expect(!suffix.isEmpty)
        #expect(tokenizer.decode(suffix, skipSpecialTokens: true).contains("result"))
    }
}
