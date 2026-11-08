# LFM2.5 fork landing page and README redesign

## Intent

Publish the LFM2.5 support work as the `main` branch of
`DehydratedFlask/turbo-fieldfare`, a fork of `drumih/turbo-fieldfare`, and
redesign the README so a visitor can immediately choose between Gemma 4 and
LFM2.5. Gemma 4 stays the default; LFM2.5 becomes a first-class second option.

## Goals

- Make the fork's `main` the verified upstream 0.9.3 + LFM2.5 build.
- Identify the repository as a fork without hiding the upstream project.
- Present Gemma 4 and LFM2.5 as two supported model choices.
- Give each model a clear install-and-run path from the top of the README.
- Keep the existing Gemma-first documentation where it is still accurate.
- Update the GitHub description and topics to match the fork.
- Preserve upstream attribution, license, and the original afterword.

## Non-goals

- Merging or restoring Qwen 3.6 support.
- Changing runtime code, model defaults, or the `.gturbo` format.
- Publishing new LFM2.5 speed or memory benchmarks.
- Opening a pull request against upstream.
- Rewriting the upstream deep-dive docs beyond LFM-specific corrections.

## Constraints

- LFM2.5 has no published decode-speed or weights-footprint measurement in
  this checkout. The README must say that plainly instead of reusing Gemma's
  ~2 GB and 5.1-6.3 tok/s figures.
- Gemma 4 remains the default for new and existing users.
- LFM2.5 is text-only. Image input must be described as refused, not degraded.
- Only the `DehydratedFlask` fork is pushed. `origin` (NeelM0906) and
  `upstream` (drumih) are read-only for this work.
- The existing green test state is inherited; this change makes no code edits.

## Approaches

### A. Two-model fork landing page (chosen)

Add a compact fork banner, rewrite the header and intro to name both models,
and put a "Choose a model" decision table immediately after the intro. Quick
start gets the fork clone URL and one install path per model. The rest of the
README keeps its Gemma-first detail, with LFM differences called out where
they matter.

Trade-off: the README gets a little longer at the top, but the fork's addition
is obvious and neither model is misrepresented.

### B. LFM-first edition

Retitle the fork around LFM2.5 and make Gemma the alternative.

Rejected: it makes an unmeasured model the headline, changes the project's
identity more than the change warrants, and could confuse visitors looking for
the original project.

### C. Minimal patch

Keep the current structure, add a fork note and a two-row model table, update
the clone URL.

Rejected: fast, but it leaves LFM buried under a Gemma-only narrative and does
not deliver the requested redesign.

## Design

### Git and publishing

1. Add a `fork` remote for `DehydratedFlask/turbo-fieldfare`.
2. Create local `main` from the fork's current `main` (upstream 0.4).
3. Fast-forward `main` to `lfm25-support` (upstream 0.9.3 + 35 LFM commits).
4. Commit the README redesign and this design record on `main`.
5. Push `main` to `fork/main` only after the README diff is approved.
6. Update the fork description and add topics for Gemma 4, LFM2.5, Swift,
   Metal, and Apple Silicon.

### README structure

- **Fork banner** above the logo: names the fork owner, the upstream project,
  and the added architecture.
- **Header:** keep the TurboFieldfare name; replace the single-model tagline
  with one that covers the two-model fork without claiming LFM's speed.
- **Intro:** explain the original Gemma 4 result, then state that this fork
  adds LFM2.5 24B-A2B as a selectable second architecture.
- **Choose a model:** a compact table comparing:
  - role: default vs added option
  - install source: `gemma4` vs `lfm2_moe`
  - storage: about 14.3 GB vs about 13.6 GB
  - vision: optional pack vs none, images refused
  - context and slots: 8,192 app/CLI, 16,384 server, 16 slots for Gemma;
    8,192 and 8 slots for LFM
  - measured speed: Gemma's existing M2/M5 figures; LFM "not yet measured"
- **Quick start:** fork clone URL; build once; install Gemma or LFM2.5;
  open the app and pick the model. Mention that the two installs sit side by
  side and that only one model runs at a time.
- **At a glance:** split into Gemma 4 and LFM2.5 rows, keeping Gemma's
  measured numbers and marking LFM's memory/speed as unmeasured.
- **Supported models / LFM notes:** keep the manifest-driven family dispatch
  explanation and link to `docs/LFM2_5.md`. Add the LFM caveats: text-only,
  thinking channel hidden, 8K/8-slot defaults, server tool calls supported,
  no app tool loop, no performance claim.
- **App, CLI, and server sections:** add short LFM-specific notes where the
  current text assumes Gemma: model picker, `--model-source lfm2_moe`,
  `--image` refusal, and server defaults.
- **Status and scope:** state that the fork adds LFM2.5, that Gemma remains
  the default, and that LFM is text-only.
- **Attribution:** keep the upstream author's afterword and license section;
  add a brief fork-maintainer note near the top or before the afterword.

### Verification

- `git diff --check` for whitespace errors.
- Audit every relative README link and anchor touched by the redesign.
- Confirm the pushed fork `main` SHA equals local `main`.
- Confirm the fork description and topics match the new positioning.
- No package test run is required because no code or package metadata changes.

## What is left to the builder

- Choose the exact tagline and table wording, keeping every number sourced
  from the checkout.
- Decide how much of the existing Gemma-first prose to keep verbatim.
- Run the final link and remote-SHA checks, then report the pushed commit.
