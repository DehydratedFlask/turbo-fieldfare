> **Fork:** This is [DehydratedFlask/turbo-fieldfare](https://github.com/DehydratedFlask/turbo-fieldfare), a fork of [drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare). It adds LFM2.5 24B-A2B as a second selectable model. Gemma 4 remains the default.

<p align="center">
  <img src="docs/assets/turbofieldfare-logo-rounded.png" alt="TurboFieldfare logo: a fieldfare inside a segmented cache ring" width="280">
</p>

<h1 align="center">TurboFieldfare</h1>

<p align="center">
  <strong>Gemma 4 26B-A4B in about 2 GB — now with LFM2.5 24B-A2B</strong><br>
  A custom Swift + Metal runtime for bounded-memory inference on Apple Silicon.
</p>

<p align="center">
  <img alt="Swift 6.2" src="https://img.shields.io/badge/Swift-6.2-F05138?logo=swift&logoColor=white">
  <img alt="Metal 4" src="https://img.shields.io/badge/Metal-4-5E5CE6">
  <img alt="macOS 26 or later" src="https://img.shields.io/badge/macOS-26%2B-000000?logo=apple&logoColor=white">
  <a href="LICENSE"><img alt="Apache 2.0 license" src="https://img.shields.io/badge/License-Apache%202.0-2ea44f"></a>
</p>

<p align="center">
  <a href="#choose-a-model">Choose a model</a> ·
  <a href="#try-it">Quick start</a> ·
  <a href="docs/OPENAI_SERVER.md">Local server</a> ·
  <a href="docs/BENCHMARKS.md">Benchmarks</a> ·
  <a href="docs/COMMUNITY_BENCHMARKS.md">Contribute results</a> ·
  <a href="docs/SYSTEM_DESIGN.md">How it works</a> ·
  <a href="docs/OPTIMIZATION_JOURNEY.md">Experiments</a> ·
  <a href="docs/IMPLEMENTATION_REFERENCES.md">References</a>
</p>

![TurboFieldfare Mac app generating text with Gemma 4 26B-A4B](docs/assets/turbofieldfare-app.webp)

*The Mac app running Gemma 4. The install picker also offers LFM2.5 24B-A2B.*

Memory got expensive. So I gave a 26-billion-parameter model a ~2 GB budget.

TurboFieldfare runs the instruction-tuned
**[Gemma 4 26B-A4B](https://ai.google.dev/gemma/docs/core/model_card_4)**
without loading the entire 14.3 GB model into memory. It keeps the shared
1.35 GB core and FP16 KV cache in memory, then streams only the experts needed
for each token from SSD. This is what lets the model run on Macs with 8 GB of
RAM.

The runtime, streaming installer, CLI, and native Mac app are written in Swift
and Metal. TurboFieldfare is a purpose-built runtime rather than a wrapper
around MLX or llama.cpp. The curated
[experiment record](docs/experiments/EXPERIMENT_INVENTORY.md) summarizes 103
measured results across kernels, caching, I/O, prefill, and decode.

This fork keeps that runtime and adds **LFM2.5 24B-A2B** as a selectable
second architecture. Gemma 4 stays the default. Both models install as separate
`.gturbo` directories, and the installed manifest's `arch.family` selects the
tokenizer, runner, chat framing, and memory profile. LFM2.5 is text-only, and
this checkout publishes no LFM2.5 speed or memory benchmark.

## Choose a model

| | **Gemma 4 26B-A4B IT** (default) | **LFM2.5 24B-A2B** (added in this fork) |
| --- | --- | --- |
| Best for | Optional vision and the published Gemma benchmarks | Text-only ChatML chat with hidden reasoning and server-side tools |
| Install source | `--model-source gemma4` | `--model-source lfm2_moe` |
| Installed size | About 14.3 GB | About 13.6 GB |
| Vision | Optional 1.1 GB companion pack on M2 or newer | None; image requests are refused |
| App and CLI context | 8,192 tokens | 8,192 tokens |
| Server context and slots | 16,384 tokens; 16 slots | 8,192 tokens; 8 slots |
| Decode speed | 5.1-6.3 tok/s on an 8 GB M2; 31-35 tok/s on a 24 GB M5 Pro | Not yet measured |
| Weights memory | About 2 GB with a 4K KV cache | Bounded expert streaming; not yet measured |

Both models install as separate `.gturbo` directories and can sit side by
side. Gemma 4 remains the default for new and existing users. A path is never
guessed from its filename; the manifest decides the family. LFM2.5 has no
vision tower.

## Try it

```bash
git clone https://github.com/DehydratedFlask/turbo-fieldfare.git
cd turbo-fieldfare
swift build -c release
.build/release/TurboFieldfareMac
```

On the first run, Swift Package Manager downloads and builds the Swift packages
required by the tokenizer. The complete release build includes the foreground
Mac app and its sibling decode-service executable.

When the app opens, select **Gemma 4** or **LFM2.5 24B-A2B** in the model
picker, then choose the matching **Download** button and let TurboFieldfare
fetch and repack the pinned checkpoint (about 15 GB for Gemma 4, about 13.6 GB
for LFM2.5). Once it is ready, choose **Load Model**, type your prompt, and
press **Generate**.

The two models install side by side; only one runs at a time. LFM2.5 has no
image pack.

Or install from the CLI:

```bash
# Gemma 4 (default)
swift run -c release TurboFieldfareRepack \
  --model-source gemma4 \
  --output scratch/gemma4.gturbo

# LFM2.5 24B-A2B
swift run -c release TurboFieldfareRepack \
  --model-source lfm2_moe \
  --output scratch/lfm2.5-24b-a2b.gturbo
```

## At a glance

| Metric | Gemma 4 26B-A4B IT | LFM2.5 24B-A2B |
| --- | --- | --- |
| Model | 26B total, about 3.88B active per token | 24B total, A2B active |
| Weights | MLX affine 4-bit, group 64; 8-bit router; 4-bit shared and routed experts | 4-bit resident and routed expert weights; 6-bit tied embedding and lm-head; F32 router |
| Memory | About 2 GB with a 4K KV cache | Bounded expert streaming; no published measurement |
| Storage | About 14.3 GB text, plus about 1.1 GB optional image pack | About 13.6 GB |
| Hardware | Apple Silicon Mac; 8 GB validated | Apple Silicon Mac; 8 GB hosts are admitted, but no published LFM benchmark |
| Platform | macOS 26, Metal 4, Swift 6.2 | macOS 26, Metal 4, Swift 6.2 |
| Decode | [5.1-6.3 tok/s](docs/BENCHMARKS.md#m2-measured-decode) on an 8 GB M2; [31-35 tok/s](docs/BENCHMARKS.md#m5-measured-decode) on a 24 GB M5 Pro | Not yet measured |
| Vision | Optional pack on M2 or newer | None |

The measured Gemma result is a reference point, not a performance ceiling.
Prompt length, generated length, page-cache state, and hardware all affect
throughput. See
[community benchmark results](docs/COMMUNITY_BENCHMARKS.md#community-results)
from other Macs, or follow the
[community benchmark guide](docs/COMMUNITY_BENCHMARKS.md) to add your own.
LFM2.5 has no published result yet; the community guide includes commands for
it.

## Using TurboFieldfare

TurboFieldfare provides a native Mac app, a command-line interface, and an
experimental loopback OpenAI-compatible server. They use the same `.gturbo`
model directory, but only one model-owning product should run at a time.

The Swift package exposes six products:

| Product | Purpose |
| --- | --- |
| `TurboFieldfare` | Swift library containing the runtime and Metal kernels |
| `TurboFieldfareMac` | Native Mac app for installation and generation |
| `TurboFieldfareDecodeService` | One-shot local model and Metal owner used by the Mac app |
| `TurboFieldfareCLI` | Command-line instruction chat and raw completion |
| `TurboFieldfareServer` | Loopback OpenAI-compatible Chat Completions server |
| `TurboFieldfareRepack` | Streaming model installer and install verifier |

### Supported models

| Model | Family | Install source | Server context | Server slots | Vision |
| --- | --- | --- | --- | --- | --- |
| Gemma 4 26B-A4B IT 4-bit (default) | `gemma4` | `--model-source gemma4` | 16,384 | 16 | Yes |
| LFM2.5 24B-A2B 4-bit | `lfm2_moe` | `--model-source lfm2_moe` | 8,192 | 8 | No |

Both models install as separate `.gturbo` directories and can sit side by
side. The installed manifest's `arch.family` selects the tokenizer, runner,
chat framing, and memory profile; a path is never guessed from its filename.
Gemma 4 remains the default for new and existing users. The app and CLI use
8,192 tokens for both models; the table's context and slot figures are the
server/descriptor defaults. See
[LFM2.5 24B-A2B](docs/LFM2_5.md) for the architecture, tensor map, and tool
grammar.

#### LFM2.5 notes

- Text-only: there is no vision pack, and `--image` with an LFM model fails
  before a tokenizer or Metal device is loaded.
- ChatML with a reasoning channel: generation starts in `<think>`;
  `</think>` ends it. Reasoning is not returned or saved, and the app shows a
  "Thinking…" indicator.
- Defaults: 8,192-token context and 8 expert-cache slots. The checkpoint
  supports up to 131,072 positions, but the app hides options above the model
  ceiling.
- Server tool calls are supported. The app and CLI do not expose a tool loop.
- No LFM2.5 speed or memory benchmark is published in this checkout. The
  [community benchmark guide](docs/COMMUNITY_BENCHMARKS.md) includes commands
  for reporting one.

### Requirements

- An Apple Silicon Mac; Gemma 4 is validated on an 8 GB M2 MacBook Air. The
  runtime admits LFM2.5 on 8 GB hosts, but no LFM benchmark is published here.
- macOS 26 with Metal 4
- Apple Command Line Tools or Xcode with Swift 6.2+ and the macOS 26 SDK or newer
- Enough free storage for the model installation (about 14.3 GB for Gemma 4,
  about 13.6 GB for LFM2.5)
- An internet connection for the first model install

The package is arm64-only. Older macOS and Metal versions are not supported.

### Prompting the model

The Mac app treats what you type as an instruction and handles the selected
model's chat formatting automatically: Gemma's chat template or LFM2.5's
ChatML template. For LFM2.5, reasoning stays hidden. Just describe the task
and include any context the model needs.

Generation defaults to temperature `0.2`, Top-K `64`, and Top-P `0.95`. Set
temperature to `0` for deterministic greedy output. The model can still repeat
itself or give incorrect answers, so check important results.

The app and CLI support user and model messages plus optional system guidance;
they do not expose or execute tools. The loopback server accepts function-tool
declarations and returns model-produced tool calls for the client to authorize
and execute. Audio and video are not supported.

### Images

For Gemma 4, images are supported through a vision tower, which installs as a
companion pack beside the text model. Install it once and the app, CLI, and
server all accept images for Gemma 4. Without it they tell you image support is
unavailable, and the text runtime is untouched. The image tower requires an M2
or newer Apple Silicon Mac; text-only inference remains available on M1.
LFM2.5 has no vision tower; image requests are refused rather than answered
without the image.

[System design](docs/SYSTEM_DESIGN.md#images) covers how the tower runs and
what it costs on an 8 GB machine.

### Mac app

Clone the repository, then run the app from its root:

```bash
swift build -c release
.build/release/TurboFieldfareMac
```

Build the complete package so the app and its sibling decode service are both
available. New installs from this checkout default to a Gemma 4 path
(`scratch/gemma4.gturbo`); the app's model picker also offers LFM2.5. The app
remembers a location you choose.

#### Build a Mac app

From the repository root, run:

```bash
Scripts/build_app.sh
```

This creates `dist/TurboFieldfare.app`, which you can double-click or move to
Applications. Add `--install` to also copy it to `~/Applications`.
The bundle includes the decode service and resources; model weights stay in
their selected folder. The script uses ad-hoc signing, so no Apple developer
account is required.

#### Install the model

Select **Gemma 4** or **LFM2.5 24B-A2B** on the install screen, then choose
the matching **Download** button to install, or **Change…** to pick a
destination. The two install side by side, and switching unloads the current
model and keeps each model's own conversation history. LFM2.5 has no image
pack; selecting it hides **Image Support** and applies the 8,192-token,
8-slot defaults.

To reuse a model, choose **Use Existing Model…** and select its `.gturbo`
folder or parent folder. If several models are found, pick one.
Keep the optional image pack beside its text model: `gemma4.gturbo` and
`gemma4.vision.gturbo`.

The installer never materializes the full source checkpoint. It streams the
required byte ranges from the pinned Hugging Face revision and repacks them
directly into the `.gturbo` layout as they arrive. This avoids a second full
checkpoint on disk and keeps scratch memory bounded.

The first installation transfers about 15 GB for Gemma 4 or about 13.6 GB for
LFM2.5 through bounded Hugging Face range requests. Network speed and Hugging
Face response times vary, so it can take a while. The completed `.gturbo`
installation occupies about 14.3 GB for Gemma 4 or about 13.6 GB for LFM2.5 and
is accepted only after its manifest and file hashes have been validated.
Installation does not load the model into memory.

#### Load and generate

After installation:

1. Choose **Load Model**.
2. Enter a prompt in the composer.
3. Choose **Generate**, or press <kbd>Command</kbd>+<kbd>Return</kbd>. Use **Settings > Send Message With** to choose Return or Command-Return.
4. Send another message to continue the conversation, or choose **New Chat** to start over.
5. Use the stop button or <kbd>Escape</kbd> to end generation early.

The status bar shows generation progress, decode speed, and memory use. Use the
right pane to configure sampling, context length, expert-cache slots, and
runtime options. See [Runtime controls](docs/RUNTIME_CONTROLS.md) for details
and defaults.

Use **Settings → Text Size** to enlarge conversation and message-editor text
from 100% to 200%. Changes apply immediately and are remembered across launches.

#### Conversation history

The app saves chats locally so you can browse and continue them from the
sidebar. Every launch opens an empty chat. Only one conversation's model context
(KV cache) stays in memory.
Browsing other chats keeps that cache intact; continuing another chat replaces
it. **New Chat** starts fresh.

Settings and chats live in `~/Library/Application Support/TurboFieldfare/`
(`mac-app-settings.json` and `conversations/`). A standalone app also uses
this folder for new model downloads by default. Moving the app does not move
these files or change a saved model location.

### Command-line interface

The CLI uses an existing `.gturbo` installation. For a model installed through
the Mac app, pass its location with `--model`. To install from the command line:

```bash
swift run -c release TurboFieldfareRepack \
  --output scratch/gemma4.gturbo \
  --overwrite
```

Continue a cancelled or interrupted download:

```bash
swift run -c release TurboFieldfareRepack \
  --output scratch/gemma4.gturbo \
  --overwrite \
  --resume
```

Remove saved download state:

```bash
swift run -c release TurboFieldfareRepack \
  --discard-partial \
  --output scratch/gemma4.gturbo
```

The runtime accepts only a completed `.gturbo` directory with a final
`manifest.json`.

Install LFM2.5 beside Gemma instead of replacing it:

```bash
swift run -c release TurboFieldfareRepack \
  --model-source lfm2_moe \
  --output scratch/lfm2.5-24b-a2b.gturbo
```

Run it through the same CLI:

```bash
swift run -c release TurboFieldfareCLI \
  --model scratch/lfm2.5-24b-a2b.gturbo \
  --chat-prompt "Explain the difference between a cache and a buffer." \
  --max-new 256
```

The footer names the manifest model ID and family. LFM has no image support;
`--image` with an LFM model exits before loading a tokenizer or a Metal device.

Verify an existing installation without loading the model:

```bash
swift run -c release TurboFieldfareRepack \
  --verify-install \
  --input-gturbo scratch/gemma4.gturbo
```

#### Install image support

The Gemma 4 companion pack installs beside the text model:

```bash
swift run -c release TurboFieldfareRepack \
  --vision-output scratch/gemma4.vision.gturbo \
  --text-model scratch/gemma4.gturbo
```

The pack adds about 1.1 GB. Verify it with `--verify-vision-install`, remove an
installed one with `--remove-vision-install`, and drop a cancelled download with
`--discard-partial --vision-output <dir>`. A cancelled transfer can also be
continued with `--resume`. The Mac app installs the same pack from its
**Image Support** section.

#### Send an image with Gemma 4

```bash
swift run -c release TurboFieldfareCLI \
  --model scratch/gemma4.gturbo \
  --chat-prompt "What is in this picture?" \
  --image photo.jpg
```

`--image` is repeatable and requires `--chat-prompt`; it cannot be combined
with `--prompt` or `--messages-file`. Without the companion pack the run stops
and says image support is unavailable.

A multi-turn conversation carries its images inside the messages file instead,
as `image_file` parts in any user message; `--image` covers the single-turn
case only. For Gemma 4, either route needs at least 16 expert-cache slots,
because an image prompt always prefills chunked. LFM2.5 rejects image input
before that point.

#### Instruction chat

Put chat messages in a JSON array and pass it with `--messages-file`:

```json
[
  {"role": "user", "content": "Explain why chunked prefill reduces time to first token while keeping memory bounded."}
]
```

```bash
swift run -c release TurboFieldfareCLI \
  --model scratch/gemma4.gturbo \
  --messages-file messages.json
```

This formats messages in the same way as the Mac app. The CLI response limit
is set with `--max-new`, which defaults to 1,024 tokens. The Mac app can
generate until the selected context window is full.

Common generation options include `--max-context`, `--temperature`, `--top-k`,
`--top-p`, `--repetition-penalty`, `--seed`, and repeatable `--stop` strings.
Runtime options include `--expert-cache-slots`, `--expert-cache-policy`,
`--prefill`, `--prefill-chunk-tokens`, and `--rdadvise`; omitted options use
the [production defaults](docs/RUNTIME_CONTROLS.md). Run the following command
for the complete option list:

```bash
swift run -c release TurboFieldfareCLI --help
```

Generated text goes to standard output. Timing statistics go to standard error;
add `--quiet` to suppress that footer in scripts.

### Local OpenAI-compatible server

Build the server and point it at an installed model:

```bash
swift build -c release --product TurboFieldfareServer
.build/release/TurboFieldfareServer \
  --model scratch/gemma4.gturbo
```

Pass an LFM2.5 `.gturbo` path instead to serve that model.

It listens on `http://127.0.0.1:8080/v1` and supports Chat Completions,
streaming, function tools, and single-prefix prompt reuse. The client must
authorize and run every tool call. Keep the server on loopback; it has no
remote authentication or TLS.

The server reads the installed family. Gemma 4 defaults to a 16,384-token
context and 16 slots; LFM2.5 defaults to 8,192 tokens and 8 slots. LFM2.5 tool
calls use its ChatML grammar.

See [Local server](docs/OPENAI_SERVER.md) for a test request, Python and
OpenCode setup, prompt reuse, tool handling, and the supported API subset.

## Test and contribute

Run the public test suite serially:

```bash
Scripts/test.sh
```

Before starting a model run, close memory-heavy apps and check
`memory_pressure -Q`. If it reports little free memory, postpone the run. Run
only one TurboFieldfare app, decode service, CLI, server, test, or other
local-model process at a time.

To contribute a comparable performance result, follow the
[community benchmark guide](docs/COMMUNITY_BENCHMARKS.md).

## How the inference engine works

For Gemma 4, at each transformer layer, Metal computes attention and the
router from resident weights. The CPU uses the router's top-8 expert IDs to
plan against the layer's 16-slot LFU cache, then fills misses with bounded
parallel `pread` calls into Metal-visible buffers. Metal computes the resident
shared-expert branch while those reads run, then combines the shared and routed
outputs.

LFM2.5 uses top-4 routing, an 8-slot default cache, dense MLPs in its first two
layers, and short-convolution operators in 30 of its 40 layers. The same
bounded pread and expert-cache machinery serves both architectures; the
installed manifest selects the runner and its kernels.

Prompt prefill uses chunks of up to 128 tokens so one fetched expert can serve
multiple rows. Generation repeats the routed layer loop one token at a time.
The installer applies the same bounded-memory rule: it repacks remote ranges
directly into `.gturbo` without staging a full shard or tensor.

For a video overview of TurboFieldfare, see Better Stack's
[Local AI On Apple Silicon uses 7X Less RAM](https://youtu.be/vHhephsP6vU).

For a visual introduction to the model architecture, see Maarten Grootendorst's
[A Visual Guide to Gemma 4](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4).

[System design](docs/SYSTEM_DESIGN.md) explains the `.gturbo` layout, memory
ownership, prefill, router handoff, `cb1`/`io`/`cb2` phases, Metal kernels, and
correctness invariants.

## Status and scope

TurboFieldfare currently includes:

- Remote streaming repack into the `.gturbo` model format
- Instruction-tuned Gemma 4 26B-A4B with verified chat formatting
- LFM2.5 24B-A2B with short-convolution/full-attention layers, dense and MoE
  blocks, 6-bit tied embedding/lm-head, and ChatML tool calls
- Family dispatch from the installed manifest (`gemma4` or `lfm2_moe`) through
  the tokenizer, runner, CLI, server, app, and decode service
- Gemma 4's 4-bit MLX affine embedding, attention, shared-expert, and
  routed-expert weights, with an 8-bit router
- Custom Metal kernels for quantized GEMV, attention, MoE, normalization,
  RoPE, sampling, and production fusions
- SSD-backed routed-expert streaming with a bounded expert cache
- Chunked prefill for one-shot prompts and new conversational turns, followed
  by token-by-token generation
- Gemma 4's FP16 KV storage with bounded circular storage for 25 sliding-window
  layers and linear storage for 5 full-attention layers
- LFM2.5's 10 full-attention layers and 30 short-convolution layers, with FP16
  KV and a 128-token convolution rewind span
- Exact split-K/V decode attention with distinct normalized K and V paths
- A Swift library, streaming installer, command-line interface, loopback
  OpenAI-compatible server, and native SwiftUI/AppKit Mac app with a sibling
  local decode service
- Optional image input from a separately installed companion pack: the vision
  tower runs on bounded scratch, image rows attend in both directions inside
  the sliding-window layers, and routed experts are released while the tower
  runs

This fork adds LFM2.5 24B-A2B to the original Gemma 4 26B-A4B work. Gemma 4
remains the default. Current scope is text input from both pinned checkpoints
on Apple Silicon Macs with at least 8 GB of RAM, plus image input for Gemma 4
on M2 or newer Macs. LFM2.5 has no vision tower, so image input is refused for
an LFM model. Audio and video are out of scope.

### Future work

- Build iPhone and iPad apps, then measure inference speed and memory use on
  mobile hardware.
- Benchmark more Apple Silicon Macs, especially the base 16 GB M4 Mac mini and
  other 8 GB models.
- Measure LFM2.5 decode speed and memory on 8 GB and larger Macs, then publish
  the commands and results alongside the Gemma 4 numbers.

## Experiments and technical documentation

The [experiments that shaped TurboFieldfare](docs/OPTIMIZATION_JOURNEY.md)
explain the largest wins, the plausible ideas that failed, and the early
results that reversed under stronger validation. The detailed
[experiment record](docs/experiments/EXPERIMENT_INVENTORY.md) keeps all 103
audited entries as optional evidence.

Useful entry points:

- [Local OpenAI-compatible server](docs/OPENAI_SERVER.md)
- [System design](docs/SYSTEM_DESIGN.md)
- [LFM2.5 24B-A2B architecture and terms](docs/LFM2_5.md)
- [Benchmarks](docs/BENCHMARKS.md)
- [The experiments that shaped TurboFieldfare](docs/OPTIMIZATION_JOURNEY.md)
- [Experiment inventory and summaries](docs/experiments/EXPERIMENT_INVENTORY.md)
- [Implementation references](docs/IMPLEMENTATION_REFERENCES.md)

## About this fork

This fork is maintained by [@DehydratedFlask](https://github.com/DehydratedFlask).
It adds LFM2.5 24B-A2B support to Andrey Mikhaylov's TurboFieldfare. The
original project, Gemma 4 work, and design notes are at
[drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare).

## License and model terms

TurboFieldfare's source and documentation are licensed under the
[Apache License 2.0](LICENSE).

Model weights are not included. The installer downloads Gemma 4 and LFM2.5
from their pinned Hugging Face checkpoints, and the weights remain governed by
their source terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for
the Gemma 4, LFM2.5, and Swift package license review.

TurboFieldfare is an independent research project. It is not affiliated with,
sponsored by, or endorsed by Google.

*The afterword below is from the upstream author, Andrey Mikhaylov.*

## Afterword and the project name

Thanks for checking out this project!

My name is Andrey Mikhaylov. You can find me on
[LinkedIn](https://www.linkedin.com/in/andrey-mikhaylov-ios-dev/).
I am the author of TurboFieldfare and an iOS and Metal engineer. Most of my
work is with images, video, and on-device AI.

I dedicate this project to my wife, Sasha, the most supportive person I know.
She stands by me even through the hardest times. She loves wildlife, goes
birdwatching, and volunteers with our local birding community. Because of her,
I have also grown closer to birds and nature.

TurboFieldfare is named after the fieldfare, a member of the thrush family and
my favourite bird. It is not the most noticeable or brightly coloured bird, but
it definitely has a character and unique features of its own. I think the same
is true of this project: it may not be the most practical, but I built it with
my favourite tools, especially Metal, in my favourite field, on-device ML
inference. It definitely has its own character and unique features.

Next time you are outside, touch the grass and listen to the birds. Sometimes
it is the most beautiful thing you can do. And if you can, support your local
wildlife community. They do important work.

Thank you!
