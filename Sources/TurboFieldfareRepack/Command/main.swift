import Foundation
import TurboFieldfareRepackCore

private let usage = """
Usage:
  TurboFieldfareRepack --output <model.gturbo>
                       [--model-source <gemma4|lfm2_moe>] [--overwrite] [--resume]
  TurboFieldfareRepack --discard-partial --output <model.gturbo>
  TurboFieldfareRepack --verify-install --input-gturbo <model.gturbo>
  TurboFieldfareRepack --vision-output <model.vision.gturbo>
                       --text-model <model.gturbo> [--overwrite] [--resume]
  TurboFieldfareRepack --verify-vision-install
                       --vision-output <model.vision.gturbo>
                       --text-model <model.gturbo>
  TurboFieldfareRepack --activate-vision-install
                       --vision-output <model.vision.gturbo>
                       --text-model <model.gturbo>
  TurboFieldfareRepack --remove-vision-install
                       --vision-output <model.vision.gturbo>
  TurboFieldfareRepack --discard-partial
                       --vision-output <model.vision.gturbo>
  TurboFieldfareRepack --help

The installer streams a supported checkpoint from Hugging Face and repackages
it without materializing the source checkpoint on disk. --model-source selects
the pinned source and defaults to gemma4. Set HF_TOKEN only if Hugging Face
requests authentication. A cancelled or interrupted download can be continued
with --resume or removed with --discard-partial.

The optional image companion pack installs beside an existing text model and
is bound to it. Without the pack the text runtime is unchanged; image input is
simply unavailable.
"""

private func printError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func runVisionInstall(_ arguments: RepackArguments) async -> Int32? {
    guard let visionOutput = arguments.visionOutput else { return nil }

    if arguments.discardPartial {
        do {
            try RemoteVisionPackInstaller.discardPartial(outputDirectory: visionOutput)
            print("Discarded saved image-pack download for \(visionOutput)")
            return 0
        } catch {
            printError("discard failed: \(error)")
            return 1
        }
    }

    if arguments.removeVisionInstall {
        do {
            try RemoteVisionPackInstaller.removeInstalled(outputDirectory: visionOutput)
            print("Removed image pack \(visionOutput)")
            return 0
        } catch {
            printError("remove-vision-install failed: \(error)")
            return 1
        }
    }

    guard let textModel = arguments.textModel else { return 2 }

    // Refuse an LFM text model before the installer takes a lock or creates
    // any output path: no LFM checkpoint ships an image tower, so a companion
    // pack could never be bound to it.
    do {
        try RemoteVisionPackInstaller.ensureGemmaTextModel(
            at: URL(fileURLWithPath: textModel, isDirectory: true),
            outputDirectory: URL(fileURLWithPath: visionOutput, isDirectory: true))
    } catch {
        printError("vision install failed: \(error)")
        return 1
    }

    if arguments.verifyVisionInstall {
        do {
            let verification = try VisionPackVerifier.verify(
                directory: URL(fileURLWithPath: visionOutput, isDirectory: true),
                installedDirectory: URL(fileURLWithPath: visionOutput, isDirectory: true),
                textModelDirectory: URL(fileURLWithPath: textModel, isDirectory: true),
                verifyWeights: true)
            print("Verified image pack \(visionOutput)")
            print("Bound to text model \(textModel)")
            print("Text manifest sha256 \(verification.compatibleTextManifestSha256)")
            return 0
        } catch {
            printError("verify-vision-install failed: \(error)")
            return 1
        }
    }

    if arguments.activateVisionInstall {
        do {
            try RemoteVisionPackInstaller.activatePrepared(
                outputDirectory: visionOutput,
                textModelDirectory: textModel,
                repoID: SupportedModelSource.repoID,
                requestedRevision: SupportedModelSource.revision)
            print("Activated image pack \(visionOutput)")
            return 0
        } catch {
            printError("activate-vision-install failed: \(error)")
            return 1
        }
    }

    let options = RemoteVisionPackInstallOptions(
        repoID: SupportedModelSource.repoID,
        revision: SupportedModelSource.revision,
        textModelDirectory: textModel,
        outputDirectory: visionOutput,
        token: ProcessInfo.processInfo.environment["HF_TOKEN"],
        overwrite: arguments.overwrite,
        resume: arguments.resume)
    do {
        let progress = InstallProgressReporter()
        try await RemoteVisionPackInstaller(options: options).run(
            progress: { progress($0) })
        print("Installed image pack \(visionOutput)")
        print("Text model: \(textModel)")
        return 0
    } catch {
        printError("vision install failed: \(error)")
        return 1
    }
}

private func run(_ values: [String]) async -> Int32 {
    let arguments: RepackArguments
    do {
        arguments = try RepackArguments.parse(Array(values.dropFirst()))
    } catch RepackArgumentError.help {
        print(usage)
        return 0
    } catch {
        printError("error: \(error)\n\n\(usage)")
        return 2
    }

    if let code = await runVisionInstall(arguments) {
        return code
    }

    if arguments.discardPartial, let output = arguments.output {
        do {
            try RemoteStreamingRepacker.discardPartial(outputDirectory: output)
            print("Discarded saved download for \(output)")
            return 0
        } catch {
            printError("discard failed: \(error)")
            return 1
        }
    }

    if arguments.verifyInstall, let input = arguments.inputGTurbo {
        do {
            let result = try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputGTurbo: input))
            print("Verified \(result.fileCount) files (\(result.bytesVerified) bytes)")
            print("Receipt: \(result.receiptPath)")
            return 0
        } catch {
            printError("verification failed: \(error)")
            return 1
        }
    }

    guard let output = arguments.output else { return 2 }
    guard let descriptor = ModelSourceCatalog.descriptor(id: arguments.modelSource) else {
        printError("error: unknown model source: \(arguments.modelSource)")
        return 2
    }
    let options = SupportedModelSource.installOptions(
        for: descriptor,
        outputDirectory: URL(fileURLWithPath: output),
        overwrite: arguments.overwrite,
        token: ProcessInfo.processInfo.environment["HF_TOKEN"],
        resume: arguments.resume)
    do {
        let progress = InstallProgressReporter()
        let result = try await RemoteStreamingRepacker(options: options).run(
            progress: { progress($0) })
        print("Installed \(descriptor.displayName)")
        print("Source revision: \(result.resolvedCommit)")
        print("Model: \(result.outputDir)")
        return 0
    } catch {
        printError("install failed: \(error)")
        return 1
    }
}

exit(await run(CommandLine.arguments))
