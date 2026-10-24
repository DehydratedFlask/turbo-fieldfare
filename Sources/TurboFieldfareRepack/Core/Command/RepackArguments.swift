import Foundation

/// Parsed repacker command line. `parse` receives the arguments after the
/// executable name.
public struct RepackArguments: Sendable, Equatable {
    public var output: String?
    public var overwrite = false
    public var resume = false
    public var discardPartial = false
    public var verifyInstall = false
    public var inputGTurbo: String?
    public var visionOutput: String?
    public var textModel: String?
    public var verifyVisionInstall = false
    public var activateVisionInstall = false
    public var removeVisionInstall = false
    /// Selector id from `ModelSourceCatalog`. Defaults to Gemma so existing
    /// install commands keep working without a new flag.
    public var modelSource = ModelSourceCatalog.gemma4.id

    public init() {}

    public static func parse(_ values: [String]) throws -> RepackArguments {
        var parsed = RepackArguments()
        var index = 0
        while index < values.count {
            let flag = values[index]
            switch flag {
            case "--help":
                throw RepackArgumentError.help
            case "--overwrite":
                parsed.overwrite = true
                index += 1
            case "--resume":
                parsed.resume = true
                index += 1
            case "--discard-partial":
                parsed.discardPartial = true
                index += 1
            case "--verify-install":
                parsed.verifyInstall = true
                index += 1
            case "--verify-vision-install":
                parsed.verifyVisionInstall = true
                index += 1
            case "--activate-vision-install":
                parsed.activateVisionInstall = true
                index += 1
            case "--remove-vision-install":
                parsed.removeVisionInstall = true
                index += 1
            case "--vision-output":
                guard index + 1 < values.count else {
                    throw RepackArgumentError.missingValue(flag)
                }
                parsed.visionOutput = values[index + 1]
                index += 2
            case "--text-model":
                guard index + 1 < values.count else {
                    throw RepackArgumentError.missingValue(flag)
                }
                parsed.textModel = values[index + 1]
                index += 2
            case "--model-source":
                guard index + 1 < values.count else {
                    throw RepackArgumentError.missingValue(flag)
                }
                let source = values[index + 1]
                guard ModelSourceCatalog.descriptor(id: source) != nil else {
                    throw RepackArgumentError.unknownModelSource(source)
                }
                parsed.modelSource = source
                index += 2
            case "--output", "--input-gturbo":
                guard index + 1 < values.count else {
                    throw RepackArgumentError.missingValue(flag)
                }
                if flag == "--output" {
                    parsed.output = values[index + 1]
                } else {
                    parsed.inputGTurbo = values[index + 1]
                }
                index += 2
            default:
                throw RepackArgumentError.unknown(flag)
            }
        }

        let visionModes = [parsed.verifyVisionInstall,
                           parsed.activateVisionInstall,
                           parsed.removeVisionInstall].filter { $0 }.count
        guard visionModes <= 1 else {
            throw RepackArgumentError.invalidMode(
                "vision install modes are mutually exclusive")
        }
        if visionModes == 1 || parsed.visionOutput != nil {
            guard parsed.visionOutput != nil else {
                throw RepackArgumentError.missingRequired("--vision-output")
            }
            guard parsed.output == nil, parsed.inputGTurbo == nil,
                  !parsed.verifyInstall else {
                throw RepackArgumentError.invalidMode(
                    "vision install operations do not accept text install arguments")
            }
            // Discard runs first below, so accepting it alongside another mode
            // would silently perform the discard and exit 0 without ever doing
            // what was asked.
            guard !(parsed.discardPartial && visionModes == 1) else {
                throw RepackArgumentError.invalidMode(
                    "--discard-partial is mutually exclusive with the other vision "
                        + "install operations")
            }
            if parsed.removeVisionInstall || parsed.discardPartial {
                guard parsed.textModel == nil, !parsed.overwrite, !parsed.resume else {
                    throw RepackArgumentError.invalidMode(
                        "this vision operation only accepts --vision-output")
                }
            } else if parsed.verifyVisionInstall || parsed.activateVisionInstall {
                guard parsed.textModel != nil else {
                    throw RepackArgumentError.missingRequired("--text-model")
                }
                // Neither reads a download, so a transfer flag here is a
                // request this mode cannot honour rather than a no-op.
                guard !parsed.overwrite, !parsed.resume else {
                    throw RepackArgumentError.invalidMode(
                        "this vision operation only accepts --vision-output and "
                            + "--text-model")
                }
            } else {
                guard parsed.textModel != nil else {
                    throw RepackArgumentError.missingRequired("--text-model")
                }
            }
            return parsed
        }
        guard parsed.textModel == nil else {
            throw RepackArgumentError.invalidMode("--text-model requires --vision-output")
        }
        guard !(parsed.resume && parsed.discardPartial) else {
            throw RepackArgumentError.invalidMode(
                "--resume and --discard-partial are mutually exclusive")
        }
        if parsed.discardPartial {
            guard parsed.output != nil else {
                throw RepackArgumentError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil, !parsed.overwrite, !parsed.verifyInstall else {
                throw RepackArgumentError.invalidMode("--discard-partial only accepts --output")
            }
            return parsed
        }
        if parsed.verifyInstall {
            guard parsed.inputGTurbo != nil else {
                throw RepackArgumentError.missingRequired("--input-gturbo")
            }
            guard parsed.output == nil, !parsed.overwrite, !parsed.resume else {
                throw RepackArgumentError.invalidMode("verification accepts only --input-gturbo")
            }
        } else {
            guard parsed.output != nil else {
                throw RepackArgumentError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil else {
                throw RepackArgumentError.invalidMode(
                    "--input-gturbo requires --verify-install")
            }
        }
        return parsed
    }
}

public enum RepackArgumentError: Error, CustomStringConvertible, Equatable {
    case help
    case unknown(String)
    case missingValue(String)
    case missingRequired(String)
    case invalidMode(String)
    case unknownModelSource(String)

    public var description: String {
        switch self {
        case .help: return "help"
        case .unknown(let flag): return "unknown argument: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .missingRequired(let flag): return "missing required argument: \(flag)"
        case .invalidMode(let message): return message
        case .unknownModelSource(let source):
            return "unknown model source: \(source)"
        }
    }
}
