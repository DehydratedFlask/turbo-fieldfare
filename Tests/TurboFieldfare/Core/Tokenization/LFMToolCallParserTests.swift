import Foundation
import Testing
@testable import TurboFieldfare

@Suite("LFM tool-call parser")
struct LFMToolCallParserTests {
    @Test func parsesMultipleCallsWithPythonAndJSONValues() throws {
        let calls = try LFMToolCallParser().parse(
            "[weather(city='Paris', metric=True, limit=3, note=None), "
            + "search(query=\"x\", opts={\"deep\": false})]",
            allowedTools: ["weather", "search"], id: "call_1")
        #expect(calls.count == 2)
        #expect(calls[0].name == "weather")
        #expect(calls[0].arguments == .object([
            "city": .string("Paris"), "metric": .bool(true),
            "limit": .integer(3), "note": .null]))
        #expect(calls[0].id == "call_1")
        #expect(calls[1].arguments == .object([
            "query": .string("x"), "opts": .object(["deep": .bool(false)])]))
    }

    @Test func parsesEscapedSingleQuotedString() throws {
        let calls = try LFMToolCallParser().parse(
            "[echo(text='a\\'b\\\\c\\nd')]", allowedTools: ["echo"], id: "call_1")
        #expect(calls[0].arguments == .object(["text": .string("a'b\\c\nd")]))
    }

    @Test func acceptsEmptyArgumentsAndOuterListOmitted() throws {
        #expect(try LFMToolCallParser().parse("ping()", allowedTools: ["ping"], id: "x")[0].arguments == .object([:]))
        #expect(try LFMToolCallParser().parse("[a(), b()]", allowedTools: ["a", "b"], id: "x").count == 2)
    }

    @Test func malformedInputReturnsNoPartialCall() {
        #expect(throws: LFMToolCallParserError.self) {
            _ = try LFMToolCallParser().parse("[a(x=1), b(]", allowedTools: ["a", "b"], id: "x")
        }
        #expect(throws: LFMToolCallParserError.unknownTool("c")) {
            _ = try LFMToolCallParser().parse("[c()]", allowedTools: ["a"], id: "x")
        }
        #expect(throws: LFMToolCallParserError.oversized) {
            _ = try LFMToolCallParser().parse(String(repeating: "x", count: 256 * 1024 + 1),
                                              allowedTools: ["a"], id: "x")
        }
        // A body that fails only on its trailing call must not leak the
        // successfully parsed leading call.
        let partial = try? LFMToolCallParser().parse("[a(x=1), b(y=]", allowedTools: ["a", "b"], id: "x")
        #expect(partial == nil)
    }

    @Test func parsesNestedContainersWithCommasAndQuotes() throws {
        let calls = try LFMToolCallParser().parse(
            #"[f(payload={'a': [1, "x,y", {'b': 'c\'d'}]})]"#,
            allowedTools: ["f"], id: "call_1")
        #expect(calls[0].arguments == .object([
            "payload": .object([
                "a": .array([
                    .integer(1),
                    .string("x,y"),
                    .object(["b": .string("c'd")]),
                ]),
            ]),
        ]))
    }

    @Test func parsesEscapesInDoubleQuotedStrings() throws {
        let calls = try LFMToolCallParser().parse(
            #"[e(note="a\nb\tc\u00e9")]"#,
            allowedTools: ["e"], id: "x")
        #expect(calls[0].arguments == .object(["note": .string("a\nb\tc\u{e9}")]))
    }

    @Test func parsesNumbersAndBooleans() throws {
        let calls = try LFMToolCallParser().parse(
            "[n(a=-7, b=1.5, c=2.5e3, d=False, e=null, f=true)]",
            allowedTools: ["n"], id: "x")
        #expect(calls[0].arguments == .object([
            "a": .integer(-7),
            "b": .number(1.5),
            "c": .number(2500),
            "d": .bool(false),
            "e": .null,
            "f": .bool(true),
        ]))
    }

    @Test(arguments: [
        "[a(x=1) trailing]",
        "[a(x=1)] extra",
        "a(x=1),",
        "[]",
    ])
    func rejectsTrailingGarbageAndEmptyBodies(body: String) {
        #expect(throws: LFMToolCallParserError.self) {
            _ = try LFMToolCallParser().parse(body, allowedTools: ["a"], id: "x")
        }
    }

    @Test(arguments: [
        "[a(x=1, x=2)]",
        "[a(1)]",
        "[a(x)]",
        "[a(x=)]",
        "[a(x='unterminated)]",
        "[a(x=1,, y=2)]",
        "[a(x=1),, b()]",
    ])
    func rejectsStructurallyInvalidArguments(body: String) {
        #expect(throws: LFMToolCallParserError.self) {
            _ = try LFMToolCallParser().parse(body, allowedTools: ["a", "b"], id: "x")
        }
    }

    @Test func enforcesToolNameGrammar() throws {
        let calls = try LFMToolCallParser().parse(
            "[a.b_c-d.e()]", allowedTools: ["a.b_c-d.e"], id: "x")
        #expect(calls[0].name == "a.b_c-d.e")
        #expect(throws: LFMToolCallParserError.self) {
            _ = try LFMToolCallParser().parse("[1a()]", allowedTools: ["1a"], id: "x")
        }
        #expect(throws: LFMToolCallParserError.self) {
            _ = try LFMToolCallParser().parse("[a b()]", allowedTools: ["a"], id: "x")
        }
    }

    @Test func toleratesWhitespaceAndNewlines() throws {
        let calls = try LFMToolCallParser().parse(
            "[\n  a(\n    x = 1\n  ),\n  b()\n]\n",
            allowedTools: ["a", "b"], id: "x")
        #expect(calls.count == 2)
        #expect(calls[0].arguments == .object(["x": .integer(1)]))
    }

    @Test func preservesQuoteCharactersAcrossStringStyles() throws {
        let calls = try LFMToolCallParser().parse(
            #"[q(s='he said "hi"', t="it's fine")]"#,
            allowedTools: ["q"], id: "x")
        #expect(calls[0].arguments == .object([
            "s": .string(#"he said "hi""#),
            "t": .string("it's fine"),
        ]))
        #expect(throws: LFMToolCallParserError.self) {
            _ = try LFMToolCallParser().parse(#"[q(s='bad \q escape')]"#, allowedTools: ["q"], id: "x")
        }
    }

    @Test func rejectsTooManyCalls() {
        let body = "[" + Array(repeating: "a()", count: 17).joined(separator: ",") + "]"
        #expect(throws: LFMToolCallParserError.tooManyCalls) {
            _ = try LFMToolCallParser().parse(body, allowedTools: ["a"], id: "x")
        }
    }

    @Test func rejectsTooManyArguments() {
        let pairs = (0...64).map { "k\($0)=\($0)" }.joined(separator: ",")
        #expect(throws: LFMToolCallParserError.tooManyArguments) {
            _ = try LFMToolCallParser().parse("a(\(pairs))", allowedTools: ["a"], id: "x")
        }
    }
}
