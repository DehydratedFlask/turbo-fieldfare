import TurboFieldfare

/// Validation for tool schemas handed to the LFM ChatML template.
///
/// The pinned template serializes the schema with `tojson`, so the OpenAI
/// shape is already what the model sees. This only bounds the structure: an
/// object schema, at most 16 levels deep and 128 named properties in total.
enum LFMToolSchema {
    static let maximumDepth = 16
    static let maximumProperties = 128

    static func validated(_ schema: JSONValue, toolName: String) throws -> JSONValue {
        guard case .object(let root) = schema else {
            throw invalid("tool \(toolName) parameters must be an object schema")
        }
        if let type = root["type"] {
            guard type == .string("object") else {
                throw invalid("tool \(toolName) parameters must have type object")
            }
        }
        var properties = 0
        try walk(schema, depth: 0, properties: &properties, toolName: toolName)
        return schema
    }

    private static func walk(_ value: JSONValue,
                             depth: Int,
                             properties: inout Int,
                             toolName: String) throws {
        guard depth <= maximumDepth else {
            throw invalid("tool \(toolName) schema is nested more than \(maximumDepth) levels")
        }
        switch value {
        case .object(let object):
            for (key, child) in object {
                if key == "properties" {
                    guard case .object(let definitions) = child else {
                        throw invalid("tool \(toolName) schema properties must be an object")
                    }
                    properties += definitions.count
                    guard properties <= maximumProperties else {
                        throw invalid(
                            "tool \(toolName) schema declares more than \(maximumProperties) properties")
                    }
                    for (_, definition) in definitions {
                        try walk(definition, depth: depth + 1,
                                 properties: &properties, toolName: toolName)
                    }
                } else {
                    try walk(child, depth: depth + 1,
                             properties: &properties, toolName: toolName)
                }
            }
        case .array(let items):
            for item in items {
                try walk(item, depth: depth + 1,
                         properties: &properties, toolName: toolName)
            }
        default:
            break
        }
    }

    private static func invalid(_ message: String) -> ServerRequestError {
        .invalid(message: message, param: "tools", code: "invalid_tool_schema")
    }
}
