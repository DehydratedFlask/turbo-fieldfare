# LFM2.5 Fork Landing Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish `lfm25-support` as the `main` branch of `DehydratedFlask/turbo-fieldfare` and redesign the README so Gemma 4 and LFM2.5 are equally legible model choices.

**Architecture:** Fast-forward the fork's stale `main` to the already-green LFM branch, then edit only documentation. The README becomes a two-model landing page: fork identity first, model choice second, model-specific details after. No runtime or package changes.

**Tech Stack:** Git, GitHub CLI, Markdown.

**Spec:** `docs/superpowers/specs/2026-10-25-fork-readme-lfm-design.md`

## Global Constraints

- LFM2.5 has no published speed or weights-memory measurement; never reuse Gemma's `~2 GB` or `5.1-6.3 tok/s` figures for it.
- Gemma 4 remains the default for new and existing users.
- LFM2.5 is text-only; image requests are refused, not silently ignored.
- Push only to `DehydratedFlask/turbo-fieldfare`. Never push to `origin` (NeelM0906) or `upstream` (drumih).
- Do not change runtime code, package metadata, model defaults, or `.gturbo` behavior.
- Preserve the Apache-2.0 license, upstream attribution, and the upstream author's afterword.
- The existing green test state is inherited; no package test run is required for documentation-only changes.

## Review Focus

- A visitor arriving for Gemma must immediately see that this is a fork and that Gemma remains the default.
- A visitor choosing LFM must see "text-only", "8K/8 slots", and "not yet measured" before installing.
- A visitor following a relative link or anchor must land on an existing section.
- A visitor reading the license section must not be told that `THIRD_PARTY_NOTICES.md` covers a model it does not cover.
- The published fork `main` must match local `main`, with `origin` and `upstream` untouched.

---

### Task 1: Prepare the fork's `main` branch

**Files:** None (Git refs only).

**Interfaces:**
- Consumes: `lfm25-support` at `fb67024`; fork `main` at `8648274`.
- Produces: local `main` at `fb67024`, a `fork` remote, and a verified fast-forward base.

- [ ] **Step 1: Confirm the worktree is clean**

Run: `git status --short`
Expected: no output.

- [ ] **Step 2: Add the fork remote if it is missing**

Run: `git remote get-url fork 2>/dev/null || git remote add fork https://github.com/DehydratedFlask/turbo-fieldfare.git`

Run: `git remote -v`
Expected: `fork` fetches and pushes `https://github.com/DehydratedFlask/turbo-fieldfare.git`; `origin` and `upstream` are unchanged.

- [ ] **Step 3: Fetch the fork and create local `main`**

Run: `git fetch fork main`
Run: `git switch -c main --track fork/main`
Expected: `main` is created at the fetched fork tip (`8648274`).

- [ ] **Step 4: Fast-forward `main` to the LFM branch**

Run: `git merge --ff-only lfm25-support`
Expected: "Fast-forward" and `main` moves to `fb67024`.

- [ ] **Step 5: Verify the fast-forward**

Run: `git rev-parse main lfm25-support`
Expected: both lines print `fb6702494a66fa54fd7b148ad829a7a6b9c8f2b0`.

Run: `git merge-base --is-ancestor upstream/main main; echo $?`
Expected: `0`.

- [ ] **Step 6: Do not push yet**

Stop after the local verification. The README must be committed before `main` is published.

---

### Task 2: Redesign the top of the README

**Files:**
- Modify: `README.md:1-82`
- Add: `docs/superpowers/specs/2026-10-25-fork-readme-lfm-design.md` (already written)
- Add: `docs/superpowers/plans/2026-10-27-lfm-fork-readme.md` (this plan)

**Interfaces:**
- Consumes: model facts in `docs/LFM2_5.md` and `README.md:100-125`.
- Produces: fork banner, `## Choose a model`, rewritten `## Try it`, and a two-model `## At a glance`.

- [ ] **Step 1: Add the fork banner and rewrite the header**

Place a blockquote above the logo:

```markdown
> **Fork:** This is [DehydratedFlask/turbo-fieldfare](https://github.com/DehydratedFlask/turbo-fieldfare),
> a fork of [drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare).
> It adds LFM2.5 24B-A2B as a second selectable model. Gemma 4 remains the default.
```

Keep the logo, `<h1>TurboFieldfare</h1>`, badges, and screenshot.

Replace the two-line tagline with:

```html
<strong>Gemma 4 26B-A4B in about 2 GB — plus LFM2.5 24B-A2B</strong><br>
A custom Swift + Metal runtime for bounded-memory inference on Apple Silicon.
```

Add a screenshot caption below the image:

```markdown
*The Mac app running Gemma 4. The install picker also offers LFM2.5 24B-A2B.*
```

- [ ] **Step 2: Rewrite the intro**

Keep the original Gemma story ("Memory got expensive..."), then add one paragraph:

```markdown
This fork keeps that runtime and adds **LFM2.5 24B-A2B** as a selectable second
architecture. Gemma 4 stays the default. Both models install as separate
`.gturbo` directories, and the installed manifest's `arch.family` selects the
tokenizer, runner, chat framing, and memory profile. LFM2.5 is text-only, and
this checkout publishes no LFM2.5 speed or memory benchmark.
```

- [ ] **Step 3: Add `## Choose a model` after the intro**

Use this table verbatim; every number must come from the checkout:

```markdown
## Choose a model

| | **Gemma 4 26B-A4B IT** (default) | **LFM2.5 24B-A2B** (added in this fork) |
| --- | --- | --- |
| Best for | Optional vision and the most measured path | Text-only ChatML chat with hidden reasoning and server-side tools |
| Install source | `--model-source gemma4` | `--model-source lfm2_moe` |
| Installed size | About 14.3 GB | About 13.6 GB |
| Vision | Optional 1.1 GB companion pack on M2 or newer | None; image requests are refused |
| App and CLI context | 8,192 tokens | 8,192 tokens |
| Server context and slots | 16,384 tokens; 16 slots | 8,192 tokens; 8 slots |
| Decode speed | 5.1-6.3 tok/s on an 8 GB M2; 31-35 tok/s on a 24 GB M5 Pro | Not yet measured |
| Weights memory | About 2 GB plus the KV cache | Bounded expert streaming; not yet measured |

Both models install as separate `.gturbo` directories and can sit side by
side. Gemma 4 remains the default for new and existing users. A path is never
guessed from its filename; the manifest decides the family. LFM2.5 has no
vision tower.
```

- [ ] **Step 4: Rewrite `## Try it` for the fork**

Replace the upstream clone URL with the fork's URL:

```bash
git clone https://github.com/DehydratedFlask/turbo-fieldfare.git
cd turbo-fieldfare
swift build -c release
.build/release/TurboFieldfareMac
```

After the existing "When the app opens..." paragraph, add:

````markdown
The install screen offers **Gemma 4** and **LFM2.5 24B-A2B**. They install
side by side; only one model runs at a time. LFM2.5 has no image pack.

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
````

- [ ] **Step 5: Replace `## At a glance` with a two-model table**

```markdown
## At a glance

| Metric | Gemma 4 26B-A4B IT | LFM2.5 24B-A2B |
| --- | --- | --- |
| Model | 26B total, about 3.88B active per token | 24B total, A2B active |
| Weights | MLX affine 4-bit, group 64; 8-bit router; 4-bit shared and routed experts | 4-bit resident and routed expert weights; 6-bit tied embedding and lm-head; F32 router |
| Memory | About 2 GB of weights and 4K KV cache | Bounded expert streaming; no published measurement |
| Storage | About 14.3 GB text, plus about 1.1 GB optional image pack | About 13.6 GB |
| Hardware | Apple Silicon Mac; 8 GB validated | Apple Silicon Mac; 8 GB hosts are admitted, but no published LFM benchmark |
| Platform | macOS 26, Metal 4, Swift 6.2 | macOS 26, Metal 4, Swift 6.2 |
| Decode | 5.1-6.3 tok/s on an 8 GB M2; 31-35 tok/s on a 24 GB M5 Pro | Not yet measured |
| Vision | Optional pack on M2 or newer | None |
```

Keep the existing "measured result is a reference point" paragraph, and add:
"LFM2.5 has no published result yet; the community guide includes commands for
it."

- [ ] **Step 6: Check the new top-level structure**

Run: `grep -n '^## ' README.md | head -12`
Expected order includes `## Choose a model`, `## Try it`, `## At a glance`, `## Using TurboFieldfare`.

- [ ] **Step 7: Commit the top half and design record**

```bash
git add README.md docs/superpowers/specs/2026-10-25-fork-readme-lfm-design.md docs/superpowers/plans/2026-10-27-lfm-fork-readme.md
git commit -m "docs: present the fork as a two-model landing page"
```

---

### Task 3: Add LFM-specific notes and model terms

**Files:**
- Modify: `README.md` (body sections from `## Using TurboFieldfare` onward)
- Modify: `THIRD_PARTY_NOTICES.md` (add the LFM2.5 model weights section)

**Interfaces:**
- Consumes: Task 2's anchors and the LFM facts in `docs/LFM2_5.md`.
- Produces: accurate LFM caveats, a fork attribution section, and a notices file that covers both models.

- [ ] **Step 1: Add LFM notes under `### Supported models`**

Keep the existing six-row product table and the existing supported-models
table. After the paragraph that links to `docs/LFM2_5.md`, add:

```markdown
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
```

- [ ] **Step 2: Update `### Requirements`**

Keep the existing bullets. Change the storage bullet to mention both models:

```markdown
- Enough free storage for the model installation (about 14.3 GB for Gemma 4,
  about 13.6 GB for LFM2.5)
```

Change the hardware bullet to:

```markdown
- An Apple Silicon Mac; Gemma 4 is validated on an 8 GB M2 MacBook Air. The
  runtime admits LFM2.5 on 8 GB hosts, but no LFM benchmark is published here.
```

- [ ] **Step 3: Update `### Prompting the model`**

Change "handles Gemma's chat formatting automatically" to:

```markdown
The Mac app treats what you type as an instruction and handles the selected
model's chat formatting automatically: Gemma's chat template or LFM2.5's
ChatML template. For LFM2.5, reasoning stays hidden.
```

Leave the sampling defaults and the tool-boundary paragraph intact.

- [ ] **Step 4: Update `### Images`**

Add one sentence at the end of the first paragraph:

```markdown
LFM2.5 has no vision tower; image requests are refused rather than answered
without the image.
```

- [ ] **Step 5: Update the Mac app install paragraph**

After the sentence about the install screen's model picker, add:

```markdown
Selecting LFM2.5 hides **Image Support** and applies the 8,192-token,
8-slot defaults.
```

- [ ] **Step 6: Update the local server section**

After the existing server paragraph, add:

```markdown
The server reads the installed family. Gemma 4 defaults to a 16,384-token
context and 16 slots; LFM2.5 defaults to 8,192 tokens and 8 slots. LFM2.5
tool calls use its ChatML grammar.
```

- [ ] **Step 7: Update `## Status and scope` and `### Future work`**

In the status bullet list, keep the existing LFM bullet. Replace the scope
paragraph with:

```markdown
This fork adds LFM2.5 24B-A2B to the original Gemma 4 26B-A4B work. Gemma 4
remains the default. Current scope is text input from both pinned checkpoints
on Apple Silicon Macs with at least 8 GB of RAM, plus image input for Gemma 4
on M2 or newer Macs. LFM2.5 has no vision tower, so image input is refused for
an LFM model. Audio and video are out of scope.
```

Add this future-work bullet:

```markdown
- Measure LFM2.5 decode speed and memory on 8 GB and larger Macs, then publish
  the commands and results alongside the Gemma 4 numbers.
```

- [ ] **Step 8: Add `## About this fork` before the license section**

```markdown
## About this fork

This fork is maintained by [@DehydratedFlask](https://github.com/DehydratedFlask).
It adds LFM2.5 24B-A2B support to Andrey Mikhaylov's TurboFieldfare. The
original project, Gemma 4 work, and design notes are at
[drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare).
```

- [ ] **Step 9: Update the license section and preserve the afterword**

Change the notices sentence to cover both models:

```markdown
Model weights are not included. The installer downloads Gemma 4 and LFM2.5
from their pinned Hugging Face checkpoints, and the weights remain governed by
their source terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for
the Gemma 4, LFM2.5, and Swift package license review.
```

Immediately before `## Afterword and the project name`, add:

```markdown
*The afterword below is from the upstream author, Andrey Mikhaylov.*
```

Do not rewrite the afterword itself.

- [ ] **Step 10: Add the LFM2.5 section to `THIRD_PARTY_NOTICES.md`**

Insert it after the Gemma 4 model-weights section and before
`## Swift package graph`:

```markdown
## LFM2.5 model weights

The fork's LFM2.5 installer downloads the pinned revision
`667408d494bde191c2cb9dd4f4549827816b9275` of
[`LiquidAI/LFM2.5-24B-A2B-MLX-4bit`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit).
Its model card declares the `lfm1.0` license. The weights are not included in
this repository and remain governed by that license.
```

Update the opening sentence to:

```markdown
This file records the dependency review performed on 2026-07-15 and the
LFM2.5 model terms added on 2026-10-27.
```

- [ ] **Step 11: Check the content constraints**

Run: `grep -n 'not yet measured\|no published LFM\|Not yet measured' README.md`
Expected: at least three matches covering the model table, the at-a-glance table, and the community note.

Run: `grep -nE 'LFM.*(tok/s|~2 GB|2 GB of)' README.md`
Expected: no matches.

Run: `grep -n 'Gemma 4 remains the default' README.md`
Expected: at least two matches (intro and Choose a model).

Run: `grep -n '^## About this fork\|^## Afterword' README.md`
Expected: both headings appear, with the afterword last.

- [ ] **Step 12: Commit the LFM-specific documentation**

```bash
git add README.md THIRD_PARTY_NOTICES.md
git commit -m "docs: document LFM2.5 behavior, scope, and model terms"
```

---

### Task 4: Verify, publish, and update GitHub metadata

**Files:** None.

**Interfaces:**
- Consumes: local `main` with Tasks 2 and 3 committed.
- Produces: published `fork/main`, verified remote SHA, and updated repository description/topics.

- [ ] **Step 1: Check whitespace and the diff**

Run: `git diff --check lfm25-support..main`
Expected: no output.

Run: `git diff --stat lfm25-support..main`
Expected: `README.md`, `THIRD_PARTY_NOTICES.md`, and the two `docs/superpowers` files; no source or package files.

- [ ] **Step 2: Audit relative links in the README**

Run:

```bash
python3 - <<'PY'
import pathlib, re
root = pathlib.Path('.')
text = (root / 'README.md').read_text()
missing = []
for target in re.findall(r'\]\(([^)]+)\)', text):
    if target.startswith(('http://', 'https://', '#', 'mailto:')):
        continue
    path = target.split('#', 1)[0]
    if path and not (root / path).exists():
        missing.append(target)
print('README relative links OK' if not missing else 'MISSING: ' + ', '.join(missing))
PY
```

Expected: `README relative links OK`.

- [ ] **Step 3: Check the new anchors**

Run: `grep -n '^## Choose a model$\|^## Try it$\|^## At a glance$\|^## About this fork$' README.md`
Expected: all four headings present.

Run: `grep -n '#choose-a-model' README.md`
Expected: at least one navigation link.

- [ ] **Step 4: Confirm the branch is still ahead and clean**

Run: `git status --short`
Expected: no output.

Run: `git log --oneline -4`
Expected: the two documentation commits on top of `fb67024`.

- [ ] **Step 5: Fetch the fork once more and confirm it has not moved**

Run: `git fetch fork main`
Run: `git merge-base --is-ancestor fork/main main; echo $?`
Expected: `0`. If it is not `0`, stop and report that the fork advanced; do not force-push.

- [ ] **Step 6: Push `main` to the fork**

Run: `git push fork main:main`
Expected: the fork's `main` updates from `8648274` to the local documentation commit.

- [ ] **Step 7: Verify the remote SHA**

Run: `git rev-parse main`
Run: `git ls-remote fork refs/heads/main`
Expected: the SHA in the `ls-remote` output matches `git rev-parse main`.

- [ ] **Step 8: Verify the other remotes were untouched**

Run: `git ls-remote origin refs/heads/main`
Expected: `f8abc4422e33a8808d5a5c1032a0e97ed5aa5118`.

Run: `git ls-remote upstream refs/heads/main`
Expected: `2e9ea50e8c35a65d9170e3bff9d1063be8931ec0`.

- [ ] **Step 9: Update the fork description and topics**

Run:

```bash
gh repo edit DehydratedFlask/turbo-fieldfare \
  --description "A fork of drumih/turbo-fieldfare adding LFM2.5 24B-A2B alongside Gemma 4 26B-A4B." \
  --add-topic swift \
  --add-topic metal \
  --add-topic apple-silicon \
  --add-topic llm \
  --add-topic gemma \
  --add-topic lfm25 \
  --add-topic on-device-ai
```

- [ ] **Step 10: Verify the GitHub metadata**

Run: `gh repo view DehydratedFlask/turbo-fieldfare --json description,repositoryTopics`
Expected: the new description and all seven topics are present; no Qwen wording remains.

- [ ] **Step 11: Report**

Report the local commit SHA, the pushed `fork/main` SHA, the README/notice diffstat, the link-check result, and the GitHub description/topics. State that `origin` and `upstream` were not pushed.

---

## Self-Review

**Spec coverage:** The fork banner, two-model chooser, separate quick starts, LFM caveats, notices, attribution, GitHub metadata, and fork-only push are each covered by Tasks 2-4. The design's "no new LFM benchmark" constraint is checked in Task 3 Step 11.

**Step scan:** Each step names a concrete edit or command with an expected result. No TBD or "handle edge cases" lines.

**Type consistency:** Branch names, remote names, SHAs, anchors, and model facts match across tasks. The README headings created in Task 2 are the anchors checked in Task 4.

**Review Focus:** Each of the five failure modes has a check: fork/default visibility (Task 2 Step 6 and Task 3 Step 11), LFM caveats (Task 3 Step 11), link integrity (Task 4 Steps 2-3), notices accuracy (Task 3 Step 10), and remote safety (Task 4 Steps 5-8).

**Proportion:** The plan is documentation and Git operations only; it does not restate the README's full prose.
