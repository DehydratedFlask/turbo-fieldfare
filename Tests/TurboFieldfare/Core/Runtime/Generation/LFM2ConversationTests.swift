import Foundation
import Testing
@testable import TurboFieldfare

@Suite("LFM conversation")
struct LFM2ConversationTests {
    @Test func rejectsImagesBeforeTouchingTheModel() {
        #expect(throws: LFM2ConversationError.imagesUnsupported) {
            try LFM2Conversation.requireTextOnly(
                parts: [.image], images: [URL(fileURLWithPath: "/tmp/x.png")])
        }
        #expect(throws: LFM2ConversationError.imagesUnsupported) {
            try LFM2Conversation.requireTextOnly(parts: [.image], images: [])
        }
        #expect(throws: Never.self) {
            try LFM2Conversation.requireTextOnly(parts: [.text("hi")], images: [])
        }
    }
}
