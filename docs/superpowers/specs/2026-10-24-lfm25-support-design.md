# LFM2.5-24B-A2B support on a clean upstream base

Status: approved

Date: 2026-10-24

Branch: `lfm25-support`, created from `upstream/main` at `0.9.3`
(`2e9ea50`). The prior Qwen 3.6 work is preserved only on the local branch
`backup/qwen36-support-2026-10-24` (`027e76e`) until this work passes
validation.

## 1. Intent

The user asked for three things, in order:

1. Delete the Qwen 3.6 support and the Qwen 3.6 model files.
2. Update the base TurboFieldfare code before adding a new model.
3. Add support for `LiquidAI/LFM2.5-24B-A2B` with full parity with the Qwen
   support that was removed.

The base update is done: the working tree is a clean checkout of upstream
`0.9.3`, not a descendant of the Qwen branch. This document describes the
LFM2.5 work on that base.

"Full parity" means the same product surfaces Qwen had, not just a model that
can be made to run:

- installer/repacker, including resumable streaming install and source
  fingerprinting;
- CLI;
- loopback OpenAI-compatible server, including chat completions, streaming,
  thinking separation, and native tool calls;
- Mac app, including install, model selection, generation controls, history,
  HUD, and model-aware UI;
- decode service, CLI/app/server tests, and documentation.

The app must keep Gemma 4 as the default. LFM is a second, explicitly selected
model family. It is not a hard-coded replacement in the style of the Qwen
branch.

## 2. Goals

- Install the pinned LFM2.5 MLX 4-bit checkpoint from Hugging Face into a
  verified `.gturbo` directory.
- Run that directory through the existing CLI, server, app, and decode
  service.
- Preserve the checkpoint's actual numerics: 4-bit affine weights, 6-bit
  tied embedding/lm-head, float32 routers and expert bias, BF16 conv weights
  and norms.
- Keep Gemma 4 behavior, file layout, defaults, and performance unchanged.
- Make model family a property of the installed manifest, resolved at load
  time, not a compile-time global.
- Remove every Qwen 3.6 artifact from the working tree and disk.
- Leave the repository with model-free tests that pass without any model and
  model-gated tests that exercise LFM when it is installed.

## 3. Non-goals

- Optimizing LFM performance beyond correctness and reasonable memory use.
  No performance ceiling is claimed.
- Adding vision to LFM. The pinned LFM repo has no vision tower. Image input
  must fail closed for LFM.
- Supporting arbitrary Hugging Face architectures. Only the pinned LFM
  checkpoint and the existing Gemma checkpoint are supported.
- Requantizing the 6-bit embedding to 4-bit to avoid writing 6-bit kernels.
  That would change the checkpoint's numerics and is explicitly rejected.
- Refactoring the Gemma decode graph into a generic transformer engine.
  The LFM graph is different enough that a separate runner is safer.
- Changing Gemma defaults, the Gemma model, or the Gemma vision pack.
- Reusing the Qwen branch's code or its hard-coded selected-model design.

## 4. Anti-goals

- Do not silently answer a prompt after dropping an image. If LFM receives
  image content, refuse it.
- Do not accept a manifest whose family and runtime constants disagree.
- Do not let a dense layer appear as a routed-expert layer, or a conv layer
  appear as an attention layer, at runtime.
- Do not make the app install LFM for users who never select it, and do not
  download the LFM checkpoint during tests.
- Do not delete the Gemma model, Gemma app data, or Gemma vision pack as part
  of Qwen cleanup.

## 5. Approaches considered

### A. Continue the Qwen branch and replace Qwen with LFM

Rejected. The Qwen branch is 23 commits and ~7.3k lines diverged from 0.9.3,
including a hard-coded selected model, Qwen-specific tensor names, and
Qwen-specific tokenizer behavior. The user explicitly asked for a base update
first, and the Qwen branch does not contain one. Reusing it would carry
unrelated assumptions and make regressions hard to attribute.

### B. Add LFM branches to `RealForwardRunner`

Rejected. `RealForwardRunner` is a 1,909-line Gemma-specific graph: Gemma
norm placement, sandwich norms, shared expert, K=V quirk, SWA/full
interleaving, partial RoPE, logit softcap, fused Gemma epilogues. Branching
inside it would make every Gemma optimization conditional and put the
existing production path at risk.

### C. Family-aware format and product layer, dedicated LFM runner (chosen)

The manifest gains a `family` discriminator and a small set of optional
architecture traits. Loading resolves a family-specific `ArchConfig`.
A `RunnerFactory` returns either the existing Gemma runner or a new
`LFM2ForwardRunner`. Product code holds the common runner protocol.
Tokenizers are selected by family. The app uses a model catalog for
installation and a persisted selection, with Gemma as the default.

This is more up-front work than B, but it keeps Gemma's hot path untouched
and gives LFM a testable, independently shaped graph.

### D. Convert the LFM checkpoint into Gemma-shaped tensors at repack time

Rejected. The LFM block algebra, conv/attention interleaving, router
semantics, dense layers, and 6-bit embedding cannot be expressed faithfully
by the Gemma graph. A conversion would either lose numerics or become a
second model implementation hidden inside the repacker.

## 6. Architecture overview

```
manifest.json arch.family
        |
        v
  ModelFamily / ArchConfig  ----  ModelLoader ----> Model
        |                                            |
        |                                            +-- resident blob (mmap)
        |                                            +-- packed_experts/ (streamed)
        v
  RunnerFactory
     |                 |
     v                 v
 Gemma runner      LFM2ForwardRunner
 (RealForwardRunner)  (conv + attention + dense/MoE)
     |
     v
 any TextRunner  <---- CLI / server / app / decode service

tokenizer/ sidecar
        |
        v
  TokenizerFactory ----> GemmaTokenizer / LFM2Tokenizer
        |
        v
  chat rendering, thinking split, tool parser
```

The family comes from the installed manifest. The app's catalog is a
user-facing choice and an installer description, not the source of truth for
runtime behavior. A user can point at any supported `.gturbo` directory and
the runtime will run the family it finds, after validating it.

## 7. Format and manifest changes

### 7.1 Optional arch fields

`GTurboManifestArchV1`, the runtime `ManifestArch`, `RepackArchInfo`, and
`ArchConfig` gain optional fields. All are absent in existing Gemma manifests
and default to current Gemma behavior. New manifests write them explicitly.

| Field | Type | Gemma default | LFM value |
|---|---|---|---|
| `family` | String | `gemma4` | `lfm2_moe` |
| `layerTypes` | `[String]?` | derived from `fullAttentionLayerMask` | 30 × `conv`, 10 × `full_attention` |
| `moeLayerMask` | `[Int]?` | all 1 | 0 for layers 0–1, 1 for 2–39 |
| `attentionScale` | Double? | 1.0 | 0.125 |
| `rmsNormUnitOffset` | Bool? | true | false |
| `ropeStyle` | String? | `gemma_partial` | `neox_full` |
| `convKernelSize` | Int? | 0 | 3 |
| `routerKind` | String? | `gemma_softmax_topk` | `lfm2_sigmoid_topk_bias` |
| `sharedExpert` | Bool? | true | false |
| `embeddingScaledBySqrtHidden` | Bool? | true | false |
| `numDenseLayers` | Int? | 0 | 2 |
| `routerNormalizeTopK` | Bool? | false | true |
| `routerBias` | Bool? | false | true |
| `routedScalingFactor` | Double? | 1.0 | 1.0 |

Existing fields keep their meaning:

- `hiddenActivation`: `gelu_pytorch_tanh` for Gemma, `silu` for LFM.
- `slidingWindow`: LFM writes `0`. Validation allows `0` only when
  `layerTypes` contains no `sliding_attention` entry. Gemma still requires a
  positive value.
- `fullAttentionLayerMask`: LFM writes 1 only for the 10 full-attention
  layers. Conv layers write 0. Runtime code must consult `layerTypes`, not
  infer "attention layer" from this mask.
- `tieWordEmbeddings`: true for LFM.
- `attentionKEqV`: false for LFM.
- `finalLogitSoftcap`: 0 for LFM.
- `partialRotaryFactor`: 1.0 for LFM.
- `numExperts`: 64, `topKExperts`: 4.
- `headDim` and `fullHeadDim`: 64.
- `numKVHeads` and `numFullKVHeads`: 8.

`ArchConfig` gains `family` and the trait fields it needs, plus static
constants:

- `.gemma4_26B_A4B` (unchanged field values, family `.gemma4`);
- `.lfm2_5_24B_A2B` (family `.lfm2Moe`).

`ManifestReader.validateArch` compares field by field as today, with family
defaulting to `.gemma4` when absent. A manifest that says `lfm2_moe` but
carries Gemma constants is rejected.

### 7.2 Quantization slots

`GTurboManifestQuantV1.router` becomes optional. Gemma manifests keep a
router slot. LFM has a float32 router and no affine router slot.

`quant.embedding.weightBits` is 6 for LFM. `quant.sharedExpert` describes
the dense FFN weights for LFM (4-bit). `quant.attention` describes the conv
and attention projections (4-bit). `quant.routedExpert` describes the MoE
experts (4-bit). LFM's quant block therefore looks like:

```
embedding:    { bits: 6, group_size: 64, mode: affine }
attention:    { bits: 4, group_size: 64, mode: affine }
sharedExpert: { bits: 4, group_size: 64, mode: affine }
routedExpert: { bits: 4, group_size: 64, mode: affine }
router:       absent
```

`GTurboManifestCodec.validate` and the runtime `ManifestQuant` decoder accept
the absent router slot. Gemma accessors that require it fail with a clear
"manifest is not Gemma" error rather than force-unwrapping.

### 7.3 Resident index and 6-bit metadata

The resident index continues to store packed U32 tensors with a logical shape.
For 6-bit embedding the packed second dimension is `2048 * 6 / 32 = 384`.
`affineSizes` is extended to accept bit widths 4, 6, and 8 and computes:

```
packedColumns = columns * bits / 32
weightBytes   = rows * packedColumns * 4
scaleBytes    = rows * (columns / groupSize) * 2
biasBytes     = scaleBytes
```

The 6-bit payload is the MLX affine layout: four 6-bit values per three
bytes, least-significant bits first. Alignment is 4 bytes for U32 payloads.
The existing 4-bit layout and Gemma manifests are unchanged.

Runtime schema validation gains an LFM branch instead of forcing every
family through the Gemma tensor list. The LFM branch checks:

- `model.embed_tokens.weight` (6-bit, 128000 × 2048) plus scales/biases;
- `model.embedding_norm.weight` (BF16, 2048);
- per layer:
  - `operator_norm.weight`, `ffn_norm.weight` (BF16, 2048);
  - conv layers: `conv.in_proj.weight` (6144 × 2048, 4-bit),
    `conv.out_proj.weight` (2048 × 2048, 4-bit),
    `conv.conv.weight` (BF16, 2048 × 3);
  - attention layers: `self_attn.q_proj.weight` (2048 × 2048, 4-bit),
    `self_attn.k_proj.weight` (512 × 2048, 4-bit),
    `self_attn.v_proj.weight` (512 × 2048, 4-bit),
    `self_attn.o_proj.weight` (2048 × 2048, 4-bit),
    `self_attn.q_layernorm.weight`, `self_attn.k_layernorm.weight`
    (BF16, 64);
  - dense layers 0–1: `feed_forward.{gate,up,down}_proj.weight`
    (4-bit, 11776 × 2048, 11776 × 2048, 2048 × 11776);
  - MoE layers 2–39: `feed_forward.gate.weight` (F32, 64 × 2048),
    `feed_forward.expert_bias` (F32, 64);
- routed experts: `gate`, `up` (1536 × 2048, 4-bit), `down`
  (2048 × 1536, 4-bit), each with scales/biases.

There is no `lm_head` tensor. The tied embedding is used for logits.

### 7.4 Expert layout with dense layers

`packed_experts/layout.json` currently assumes one layer entry per model
layer and one uniform `expertsPerLayer`. LFM has two dense layers with no
routed experts.

The manifest writes `moeLayerMask`. The layout contains entries only for MoE
layers. `numLayers` remains 40, `expertsPerLayer` remains 64. Validation
requires:

- every layout layer index is in range and unique;
- every layout layer has `moeLayerMask[layer] == 1`;
- every MoE layer has exactly one layout entry;
- every layer has exactly `expertsPerLayer` experts;
- dense layers have no layout entry.

`PackedExpertsLayout` changes from positional `layers[layer]` lookup to a
dictionary or a 40-element array of optionals. `Model`'s streamer array uses
the same shape. A call to `routedExpert(layer:)` for a dense layer fails
closed.

`expertStride` for LFM is `5,308,416` bytes:

```
gate 1536*2048*4/8 = 1,572,864
up   1536*2048*4/8 = 1,572,864
down 2048*1536*4/8 = 1,572,864
scales/biases       = 6 * 1536*32*2 = 589,824
total               = 5,308,416  (324 × 16 KiB)
```

The existing 16 KiB stride alignment holds exactly.

### 7.5 Versioning

No format version bump. Every new field is optional and absent Gemma
manifests decode and validate exactly as before. An older reader that
encounters an LFM manifest will either ignore `family` and fail the
architecture comparison, or fail the runtime schema check. Both are
fail-closed. The `family` field is the discriminator for the new schema
branch.

## 8. LFM2.5 reference facts

These are verified from the pinned checkpoint config, index, shard headers,
`tokenizer.json`, `chat_template.jinja`, `generation_config.json`, and the
HF/MLX `lfm2_moe` reference implementations.

### 8.1 Architecture

- `model_type`: `lfm2_moe`; architecture class `Lfm2MoeForCausalLM`.
- 40 layers. Layer types: `conv` for all layers except indices
  `2, 6, 10, 14, 18, 22, 26, 30, 34, 38`, which are `full_attention`.
- Hidden size 2048.
- Dense FFN intermediate 11,776 for layers 0–1.
- MoE intermediate 1,536 for layers 2–39.
- 64 experts, top-4 per token.
- 32 query heads, 8 KV heads, head dimension 64 (4:1 GQA).
- Vocab 128,000; max positions 131,072.
- RoPE theta 10,000,000; full rotary; NeoX pairing; no scaling.
- RMSNorm epsilon 1e-5; standard `weight * x / rms(x)` (no `1 + weight`).
- SiLU/SwiGLU activation.
- Short convolution: depthwise, kernel 3, no bias, `conv_L_cache = 3`
  (two previous gated values of state).
- Attention scale `1 / sqrt(64) = 0.125`.
- No sliding window.
- No logit softcap.
- Tied word embeddings.
- No `lm_head` tensor; no `model.norm`; final norm is
  `model.embedding_norm`.
- `use_expert_bias = true`, `norm_topk_prob = true`,
  `routed_scaling_factor = 1.0`.

### 8.2 Block algebra

```
residual = h
h = operator_norm(h)
h = operator(h)          # conv or full attention
h = residual + h

residual = h
h = ffn_norm(h)
h = ffn(h)               # dense MLP or MoE
h = residual + h
```

Conv:

```
BCx = in_proj(x)          # [3 * hidden]
B, C, x = split(BCx)
Bx = B * x
conv = w0 * Bx[t-2] + w1 * Bx[t-1] + w2 * Bx[t]
y = C * conv
out = out_proj(y)
```

The conv cache stores the last two `Bx` vectors, not conv outputs.

Attention:

```
q = q_proj(x); k = k_proj(x); v = v_proj(x)
q = per_head_rmsnorm(q, q_layernorm, eps)
k = per_head_rmsnorm(k, k_layernorm, eps)
q, k = neox_rope(q, k, position, theta)
out = attention(q, k, v, scale=0.125, causal=True)
out = o_proj(out)
```

There is no value norm. Q and K norms are per head over 64 values.

Dense MLP:

```
down(silu(gate(x)) * up(x))
```

MoE:

```
logits = gate(x)                    # F32 [64]
p = sigmoid(logits)
selection = p + expert_bias         # bias only for selection
top4 = argmax4(selection)           # ties by lower expert index
weights = p[top4]
weights = weights / (sum(weights) + 1e-6)
weights = weights * routed_scaling_factor
y = sum_i weights[i] * expert_i(x)
```

No shared expert. The two dense layers are ordinary MLPs, not routed
experts.

### 8.3 Tensor mapping

| HF tensor | Shape | Type | Placement |
|---|---|---|---|
| `model.embed_tokens.weight` | 128000 × 2048 | U32 6-bit | resident |
| `model.embed_tokens.scales` | 128000 × 32 | BF16 | resident |
| `model.embed_tokens.biases` | 128000 × 32 | BF16 | resident |
| `model.embedding_norm.weight` | 2048 | BF16 | resident |
| `layers.N.operator_norm.weight` | 2048 | BF16 | resident |
| `layers.N.ffn_norm.weight` | 2048 | BF16 | resident |
| `layers.N.conv.in_proj.weight` | 6144 × 2048 | U32 4-bit | resident |
| `layers.N.conv.out_proj.weight` | 2048 × 2048 | U32 4-bit | resident |
| `layers.N.conv.conv.weight` | 2048 × 3 × 1 | BF16 | resident |
| `layers.N.self_attn.q_proj.weight` | 2048 × 2048 | U32 4-bit | resident |
| `layers.N.self_attn.k_proj.weight` | 512 × 2048 | U32 4-bit | resident |
| `layers.N.self_attn.v_proj.weight` | 512 × 2048 | U32 4-bit | resident |
| `layers.N.self_attn.o_proj.weight` | 2048 × 2048 | U32 4-bit | resident |
| `layers.N.self_attn.q_layernorm.weight` | 64 | BF16 | resident |
| `layers.N.self_attn.k_layernorm.weight` | 64 | BF16 | resident |
| `layers.N.feed_forward.gate_proj.weight` | 11776 × 2048 | U32 4-bit | resident (N < 2) |
| `layers.N.feed_forward.up_proj.weight` | 11776 × 2048 | U32 4-bit | resident (N < 2) |
| `layers.N.feed_forward.down_proj.weight` | 2048 × 11776 | U32 4-bit | resident (N < 2) |
| `layers.N.feed_forward.gate.weight` | 64 × 2048 | F32 | resident (N ≥ 2) |
| `layers.N.feed_forward.expert_bias` | 64 | F32 | resident (N ≥ 2) |
| `layers.N.feed_forward.switch_mlp.gate_proj.weight` | 64 × 1536 × 2048 | U32 4-bit | per-expert stream |
| `layers.N.feed_forward.switch_mlp.up_proj.weight` | 64 × 1536 × 2048 | U32 4-bit | per-expert stream |
| `layers.N.feed_forward.switch_mlp.down_proj.weight` | 64 × 2048 × 1536 | U32 4-bit | per-expert stream |

Each quantized tensor has `scales` and `biases` of shape
`[rows, columns / 64]` in BF16.

### 8.4 6-bit affine packing

MLX packs four 6-bit values into three bytes, least-significant bits first:

```
v0 = b0 & 0x3f
v1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0f) << 2)
v2 = ((b1 >> 4) & 0x0f) | ((b2 & 0x03) << 4)
v3 = (b2 >> 2) & 0x3f
```

Dequantization is `value * scale + bias`, with one scale/bias pair per group
of 64 input columns. A U32 tensor of shape `[rows, columns]` stores
`columns * bits / 32` words per row.

### 8.5 Tokenizer and chat

Added tokens:

| ID | Text | Special |
|---|---|---|
| 124893 | `<|pad|>` | yes |
| 124894 | `<|startoftext|>` | yes |
| 124895 | `<|endoftext|>` | yes |
| 124896 | `<|fim_pre|>` | yes |
| 124897 | `<|fim_mid|>` | yes |
| 124898 | `<|fim_suf|>` | yes |
| 124899 | `<|im_start|>` | yes |
| 124900 | `<|im_end|>` | yes |
| 124901 | `<think>` | no |
| 124902 | `</think>` | no |
| 124903 | `<|tool_list_start|>` | yes |
| 124904 | `<|tool_list_end|>` | yes |
| 124905 | `<|tool_call_start|>` | no |
| 124906 | `<|tool_call_end|>` | no |
| 124907 | `<image>` | yes |

BOS is 124894. EOS is 124900 (`<|im_end|>`); the generation config lists
only 124900. Pad is 124893. The tokenizer decoder is `ByteLevel` (BPE),
not the Gemma `Replace/ByteFallback/Fuse` chain.

The chat template is ChatML-like:

```
<|startoftext|><|im_start|>system
...<|im_end|>
<|im_start|>user
...<|im_end|>
<|im_start|>assistant
<think>
```

- A system prompt is emitted only if the first message is a system message
  with content, or if tools are supplied (tools are appended to the system
  prompt as `List of tools: [json, ...]`).
- Historical assistant thinking is dropped unless
  `preserve_thinking` is set or the assistant turn is after the last user
  turn. The app will not store reasoning as visible history, so this is
  naturally handled.
- Tool calls use
  `<|tool_call_start|>[func(arg='x', n=1)]<|tool_call_end|>`.
  Arguments are Python-like: strings single-quoted with `\\`, `\'`, `\n`,
  `\r` escapes; mappings/arrays as JSON via `tojson`; numbers and booleans
  plain.
- The generation prompt ends with `<think>` (token 124901), so generation
  starts inside the reasoning channel. The model emits `</think>` (124902)
  before visible answer text. Both markers are non-special tokens.

## 9. Repacker and installer

### 9.1 Source catalog

`SupportedModelSource` becomes a catalog of pinned descriptors. Each
descriptor carries display name, repo ID, revision, source index SHA-256,
approximate download bytes, installed bytes, reserve bytes, family, default
output filename, default context, and vision support.

Gemma entry: unchanged values.

LFM entry:

```
repoID: LiquidAI/LFM2.5-24B-A2B-MLX-4bit
revision: 667408d494bde191c2cb9dd4f4549827816b9275
sourceIndexSHA256: 7887b03265aef7824c8e5b9a51667e486f889749240d8c59a95f3cd899e86e4a
approximateDownloadBytes: 13,567,188,992
output: lfm2.5-24b-a2b.gturbo
family: lfm2_moe
vision: false
```

`SourceFingerprint.knownFingerprints` adds the LFM index SHA. The installer
still verifies the index before reading any shard and still streams ranges
without staging the full checkpoint.

The CLI gains a model-source selector, defaulting to Gemma. The app's
catalog exposes both entries. Existing Gemma install commands keep working
without a new flag.

### 9.2 Config parsing

`ArchInfo.load` currently requires `text_config`. It gains family detection:

- if `text_config` exists, use the Gemma path unchanged;
- else if `model_type == "lfm2_moe"`, parse the flat LFM config;
- else throw a clear `unsupportedModelFamily` error.

LFM parsing details:

- `headDim = hidden_size / num_attention_heads` (64);
- `fullHeadDim = headDim`; `numKVHeads = numFullKVHeads =
  num_key_value_heads`;
- `slidingWindow = 0`;
- `finalLogitSoftcap = 0`;
- `ropeTheta = fullRopeTheta = rope_parameters.rope_theta ?? rope_theta`;
- `partialRotaryFactor = 1.0`;
- `tieWordEmbeddings = tie_word_embeddings`;
- `attentionKEqV = false`;
- `hiddenActivation = "silu"`;
- `layerTypes` from `layer_types`;
- `moeLayerMask` from `num_dense_layers`;
- `convKernelSize = conv_L_cache`;
- `routerKind = "lfm2_sigmoid_topk_bias"`;
- `routerNormalizeTopK = norm_topk_prob`;
- `routerBias = use_expert_bias`;
- `routedScalingFactor`;
- `numDenseLayers`.

`IndexLoader` continues to parse `quantization_config` with base bits 4,
group size 64, affine mode, and per-tensor overrides. It must accept the
`model.embed_tokens` 6-bit override and reject any 6-bit override on a
non-embedding tensor in this version.

### 9.3 LFM repack plan

A family-specific planner runs after the shared source validation. It emits:

Resident entries, in a stable order:

1. embedding table, scales, biases (6-bit);
2. `model.embedding_norm.weight`;
3. per layer: `operator_norm`, `ffn_norm`;
4. conv layers: `in_proj`, `out_proj`, `conv.conv.weight`;
5. attention layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`,
   `q_layernorm`, `k_layernorm`;
6. dense layers 0–1: `gate_proj`, `up_proj`, `down_proj`;
7. MoE layers 2–39: router `gate.weight` (F32), `expert_bias` (F32).

Per-layer expert files for MoE layers 2–39, each containing 64 experts of
gate/up/down plus scales/biases. Dense layers have no expert file.

The plan records `moeLayerMask`, `layerTypes`, family, quant slots, and the
physical expert stride. The manifest's `arch` block is written from the same
values.

Estimated sizes:

- resident blob: ~0.62 GB;
- routed experts: 38 × 64 × 5,308,416 ≈ 12.91 GB;
- installed total: ~13.53 GB.

This is recorded in the descriptor, not hard-coded in the app.

### 9.4 Tokenizer sidecar

The repacker copies into `tokenizer/`:

- `tokenizer.json` (17 MB);
- `tokenizer_config.json`;
- `chat_template.jinja`;
- `generation_config.json`;
- `identity.json` (family, chat-template identity, and SHA-256 hashes of the
  template and tokenizer files).

The manifest's file hashes cover them. The runtime loads them from the
sidecar, never from the network, when a `.gturbo` is installed.

### 9.5 Vision pack

`--vision-output` with an LFM text model fails before writing any output with
`visionUnsupportedForFamily`. The Gemma vision installer continues to work
only with a Gemma text manifest. The app hides vision controls for LFM.

### 9.6 App installation catalog

`AppModelInstallDescriptor` gains `family`, `defaultFilename`,
`defaultContext`, `defaultExpertCacheSlots`, and `supportsVision`. The app
ships two descriptors:

- Gemma 4 26B-A4B IT 4-bit (default; vision);
- LFM2.5-24B-A2B MLX 4-bit (selectable; no vision).

The existing single install state becomes keyed by descriptor ID. Only one
install runs at a time. The install view shows the selected descriptor's
name, size, progress, and resume state. A completed install can be "used"
without reinstalling.

## 10. Runtime

### 10.1 Family resolution and loading

`ModelFamily` is a `String` enum: `gemma4`, `lfm2Moe`.

`ManifestReader.peekFamily(directoryURL:)` reads only `manifest.json` and
returns the family (default `gemma4`). It does not verify hashes.

`Model.load` gains an overload that resolves the expected `ArchConfig` from
the manifest family. The existing `expecting:` parameter remains for
compatibility and for tests that pin Gemma.

`Model` keeps the resident index and exposes a generic `tensor(named:)`
returning a `TensorView` with the packed shape and bit width. Gemma's typed
accessors stay as they are. LFM gets a `LFM2Weights` view built from the
generic accessor, with names matching the HF checkpoint.

`Model.validateRuntimeSchema` dispatches on family. Gemma's checks are
unchanged. LFM's checks are listed in §7.3.

### 10.2 Common runner protocol

Make the existing internal runner protocols usable by all surfaces:

```
public protocol TextRunner: AnyObject, Sendable {
    var maxContext: Int { get }
    var continuationPosition: Int { get }
    var lastGreedyToken: UInt32? { get }
    var diagnostics: RunnerDiagnostics { get }

    func reset()
    func rewind(to position: Int) throws
    func prepareForContinuation(expectedPosition: Int) throws
    func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws
    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult
}
```

`RealForwardRunner` conforms. `LFM2ForwardRunner` conforms. A
`RunnerFactory` chooses based on `model.config.family`. The Gemma
multimodal path keeps a concrete `RealForwardRunner` and is only invoked
when the family is Gemma and a valid vision pack is installed.

`RawCompletion`'s fused-Gemma fast path becomes an optional protocol hook,
so LFM can use the generic sampler without a cast to `RealForwardRunner`.
The app, server, CLI, and decode service hold `any TextRunner`.

### 10.3 LFM2 forward graph

Decode, one token at a time:

1. 6-bit embedding lookup; no `sqrt(hidden)` scale.
2. For each layer in order:
   - standard RMSNorm (`operator_norm`);
   - conv layer or attention layer;
   - residual add;
   - standard RMSNorm (`ffn_norm`);
   - dense MLP or MoE;
   - residual add.
3. `embedding_norm`, then 6-bit tied lm-head GEMV; no softcap.
4. FP16 logits.

Conv layer:

- `in_proj` 4-bit GEMV, split into B/C/x, `Bx = B * x`;
- depthwise causal conv with kernel 3 and the two cached `Bx` values;
- `C * conv`, `out_proj` 4-bit GEMV;
- push the new `Bx` into the layer's conv state and history ring.

Attention layer:

- `q`, `k`, `v` 4-bit GEMVs;
- per-head standard RMSNorm on Q and K over 64 values, eps 1e-5;
- full NeoX RoPE at the absolute position, theta 1e7;
- write K/V into the layer's KV slot;
- full causal GQA attention with 32 Q heads, 8 KV heads, head dim 64,
  scale 0.125;
- `o_proj` 4-bit GEMV.

Dense layers 0–1:

- reuse the 4-bit shared-expert projection path, but with SiLU instead of
  GELU and intermediate 11,776.

MoE layers 2–39:

- router logits: float32 `gate.weight [64, 2048]` GEMV into 64 floats;
- selection: sigmoid, add `expert_bias`, top-4 with lower-index tie-break,
  gather unbiased sigmoid values, normalize by `sum + 1e-6`, multiply
  `routed_scaling_factor`;
- stream the 4 selected experts, compute SwiGLU with SiLU, weighted sum;
- add to the residual.

No shared-expert branch and no Gemma sandwich norms.

Prefill processes bounded chunks (default 128 tokens):

- embedding lookup for the chunk;
- standard RMSNorm;
- conv prefill computes B/C/x for all chunk positions, `Bx`, causal
  depthwise conv with left context from the conv state, `C` multiply,
  `out_proj`; the state advances to the last two `Bx` values;
- attention prefill uses the existing generic prefill attention path with
  head dim 64 and 32 Q heads; K/V are written for every position;
- dense FFN and MoE use prefilled 4-bit QMM and grouped expert paths;
- final row uses the same final norm and 6-bit lm-head.

Prefill and decode must agree numerically within the existing FP16
tolerance. Chunk size must not change logits beyond that tolerance.

### 10.4 KV and conv state

`LayerKind` gains `.conv`. `KVCacheManager` allocates K/V buffers only for
attention layers. Conv layers have no K/V allocation and no KV views.
`ContextAdmission` counts only attention layers.

Conv state per conv layer:

- current two `Bx` rows: `2 × 2048` FP16;
- a history ring of `convRewindLimit + 2` rows for rewind and replay.

`LFM2ForwardRunner` owns a family-specific `convRewindLimit = 128`. The
history is therefore 30 × 130 × 2048 × 2 ≈ 16 MB. The runner enforces the
limit itself:

- `KVCacheManager.maxRewindTokens` is **not** used for LFM. With no SWA
  layers it returns `maxContext`, which would both permit a rewind the conv
  state cannot restore and size the ring at ~16 GB if used naively.
- `LFM2ForwardRunner.rewind(to:)` throws
  `PrefillError.prefillCursorMismatch` when
  `target < kv.highWaterPosition - convRewindLimit`, before calling
  `kv.rewind(to:)`. The KV cache's own precondition is not relaxed.
- A rewind within the limit restores the conv ring position and the two
  current `Bx` rows, then delegates to `kv.rewind(to:)`.
- `reset()` and `prepareForContinuation()` reset/validate both KV and conv
  state.

The app already replays recorded tokens when continuing a saved chat. Deep
rewinds do not need to be supported beyond the ring; the replay path goes
through a fresh reset and prefill. A test must show that a 128-token rewind
works and a 129-token rewind throws without mutating either cache.

### 10.5 Kernels

New kernels:

- `embed_lookup_int6`: 6-bit affine embedding row lookup, no output scale.
- `dequant_int6_gemv`: 6-bit affine GEMV with the MLX packing formula.
- `rmsnorm_standard_fp16`: standard `weight * x / rms(x)` (the existing
  Gemma kernel applies `1 + weight`).
- `lfm2_qk_norm_rope`: per-head standard RMSNorm on Q/K, full NeoX RoPE,
  no value norm.
- `lfm2_shortconv_decode`: depthwise kernel 3 over Bx with cached state,
  `C` multiply.
- `lfm2_shortconv_prefill`: chunked causal depthwise conv, state update.
- `router_gemv_f32`: float32 router GEMV for 64 × 2048.
- `router_select_lfm2_top4`: sigmoid, bias, top-4, normalize, scale.
- `silu_mul_fp16`: SiLU-gated multiply for dense FFN decode.
- `prefill_silu_mul_fp16`: prefill variant.

Reused kernels and required changes:

- 4-bit GEMV and prefill QMM for all 4-bit projections;
- `SharedExpertInt4` / `PrefillSharedExpert` with a SiLU activation option;
- `Attention` gains an init-time `maxQHeads` (default 16). Gemma constructs
  it with 16 and is byte-for-byte unchanged; LFM constructs it with 32 and
  passes scale 0.125 to `encodeFull`. There is no global constant change, so
  Gemma's split-KV scratch and pipeline selection are untouched.
- `PrefillAttention` generic path with head dim 64, 32 Q heads, 8 KV heads;
- MoE decode/prefill kernels parameterized for top-4 and 64 experts;
- `PrefillRMSNorm` with a standard-norm option;
- `PrefillRoPE` full NeoX path.

No fused 6-bit lm-head is required for the first version. The generic
6-bit GEMV produces FP16 logits; sampling stays in `RawCompletion`. A fused
6-bit head can be added later if measurements justify it.

### 10.6 Memory and context admission

LFM FP16 KV:

```
10 full-attention layers
× 2 (K, V)
× 8 KV heads
× 64 head dim
× 2 bytes
= 20,480 bytes/token
```

At 131,072 tokens: 2.68 GB.

Conv state plus rewind ring: ~16 MB.

Resident weights: ~0.62 GB. Expert cache: 5,308,416 bytes per expert slot;
at 8 slots × 38 MoE layers ≈ 1.61 GB; at 16 slots ≈ 3.23 GB.

`ContextAdmission` changes:

- The admission profile is family-specific. `ArchConfig` carries
  `expertStrideBytes`, `expertCacheLayers`, `includedExpertCacheSlots`, and
  `nonKVRuntimeFootprintBytes`. The manifest's top-level `expertStride` must
  equal `ArchConfig.expertStrideBytes`; `ManifestReader` validates that.
- `additionalExpertCacheBytes` is
  `max(0, slots - includedExpertCacheSlots) * expertCacheLayers *
  alignedExpertStride`. It no longer hard-codes `slots - 16`, a Gemma
  stride, or `config.numLayers`.
- Gemma's profile is byte-identical to today: 2 GiB baseline,
  3,358,720-byte stride, 16 included slots, 30 layers.
- LFM's profile uses the 2 GiB baseline initially, a 5,308,416-byte stride,
  0 included slots, and 38 expert layers. The baseline is a conservative
  starting constant, not a measurement. It must be measured on the 16 GB
  validation host before release and raised if the measured working set is
  higher.
- KV bytes count only attention layers; conv layers contribute zero.
- `productionSlidingRingRows` returns 0 when there is no sliding window.
- `minimumHostMemoryBytes` is used by the app, CLI, server, and decode
  service exactly as today.

Projected LFM footprints, before the 3 GiB host reserve:

| Context | Slots | KV | Expert cache | 2 GiB baseline | Total |
|---|---|---|---|---|---|
| 8,192 | 8 | 0.16 GB | 1.61 GB | 2.00 GB | 3.77 GB |
| 8,192 | 16 | 0.16 GB | 3.23 GB | 2.00 GB | 5.39 GB |
| 131,072 | 8 | 2.68 GB | 1.61 GB | 2.00 GB | 6.29 GB |
| 131,072 | 16 | 2.68 GB | 3.23 GB | 2.00 GB | 7.91 GB |

These are admission arithmetic, not measurements. The 16 GB host can admit
all of them on paper; validation must confirm with `memory_pressure -Q` and
an actual run. If the LFM working set is higher, raise the LFM baseline
rather than discounting the expert cache.

The app's LFM default expert-cache slots are 8. The control range stays
1–16. The default context for LFM is 8,192; the app offers larger values
only when admission passes on the host.

### 10.7 Numerical traps to guard

- 6-bit packing order: least-significant bits first, four values per three
  bytes.
- Router bias is selection-only; the weights come from unbiased sigmoid.
- Normalization uses `sum + 1e-6`, not `max`.
- Conv cache stores gated `Bx`, not conv outputs.
- Conv and attention layers both use `operator_norm`; the FFN norm is
  separate.
- No `sqrt(hidden)` embedding scale.
- No value norm, no logit softcap, no sliding window.
- Dense layers 0–1 are not routed experts.
- Tied embedding drives both input and logits; there is no lm-head tensor.

## 11. Tokenizer, chat, thinking, and tools

### 11.1 Tokenizer abstraction

Introduce a common `ModelTokenizer` protocol with:

- family, bos/eos/pad IDs, stop token IDs, structural marker IDs, vocab size;
- `encode(_:addBOS:)`, `decode(_:skipSpecialTokens:)`;
- `applyChatTemplate(messages:tools:addGenerationPrompt:)` returning token
  IDs;
- a streaming detokenizer factory;
- tool-result continuation/replay support.

`GFTokenizer` remains the Gemma implementation, wrapped as `GemmaTokenizer`.
`LFM2Tokenizer` is new.

`LFM2Tokenizer`:

- loads `tokenizer.json` from the installed sidecar;
- requires a ByteLevel BPE decoder;
- validates the expected added-token IDs above and fails closed on mismatch;
- encodes with BOS 124894 only when explicitly requested; template output
  already contains BOS, so callers must not add a second one;
- decodes with a byte-level streaming detokenizer that buffers incomplete
  UTF-8 sequences and preserves text for the non-special think/tool markers;
- strips special tokens when requested, but never strips 124901/124902/
  124905/124906 before the structured decoder sees them;
- loads `tokenizer/identity.json`, written by the repacker, containing
  `family`, `chatTemplateIdentity`, `chatTemplateSHA256`,
  `tokenizerSHA256`, and `tokenizerConfigSHA256`. The runtime verifies the
  hashes and uses the identity for conversation keys. `LFM2Tokenizer`'s
  constant identity is `lfm2.5-24b-a2b-chatml-v1`. Gemma has no identity
  file and keeps its existing constant identity exactly, so existing Gemma
  history still loads.

### 11.2 Chat rendering

Use the installed `chat_template.jinja` for LFM rather than hand-building
the string, so template revisions are honored. The renderer:

- passes messages with role system/user/assistant/tool and content;
- passes historical assistant `tool_calls` with `function.arguments` as a
  mapping (the template rejects JSON-encoded strings);
- passes tool definitions in the OpenAI `{"type":"function","function":...}`
  shape that the server receives and mlx_lm passes;
- sets `preserve_thinking = false` (default), so prior reasoning is not
  replayed;
- adds the generation prompt, which ends with `<think>`.

A pure string-rendering helper is used in tests so exact ChatML framing can
be asserted without a 17 MB tokenizer fixture.

#### Tool-result continuation and replay

The tokenizer protocol returns a `ContinuationPlan` for a tool loop:

```
enum ContinuationPlan {
    case reuse(prefixTokens: [Int32], suffixTokens: [Int32])
    case replayAll([Int32])
}
```

Gemma keeps its existing `encodeToolResultContinuation` boundary logic. LFM
uses a generic, correctness-first algorithm:

1. Render `cachedMessages + [assistant]` with
   `addGenerationPrompt: false` to `prefixTokens`.
2. Render `incomingMessages` with `addGenerationPrompt: true` to
   `fullTokens`.
3. Compute the longest common prefix of the two token arrays.
4. Return `.reuse` only when `prefixTokens` is a prefix of `fullTokens` and
   the next full token is a message boundary (`<|im_start|>` for LFM).
5. Otherwise return `.replayAll(fullTokens)`.

The caller uses `.reuse` only when the runner's continuation position equals
`prefixTokens.count`; it then calls `prepareForContinuation` and prefills
the suffix. For `.replayAll`, it resets and prefills the full conversation.
Boundary detection is therefore an optimization, never a correctness
dependency. Saved-history replay always takes the `.replayAll` path.

LFM renders tool results as `<|im_start|>tool\n...<|im_end|>\n`.
`tool_call_id` is accepted in the message model but not rendered by the
pinned template. Tests cover both a reusable tool-result suffix and a
forced full replay.

### 11.3 Streaming thinking separation

The LFM decoder starts in `reasoning` because the generation prompt already
opened `<think>`.

Rules:

- text before 124902 is reasoning; it is not emitted as visible `content`;
- 124902 switches to `visible`; the marker itself is not emitted;
- a duplicate 124901 while already reasoning is ignored;
- a tool-call start marker is recognized in either state; text between the
  tool-call markers is buffered for the parser;
- if the model emits a tool call before `</think>`, the tool call is still
  returned and the turn ends as a tool-call turn;
- EOS before `</think>` produces no visible answer; the server returns an
  empty-content finish rather than leaking partial reasoning;
- reasoning text is capped (64 KB) and discarded; it is never stored as
  chat history.

The runtime decoder emits lightweight reasoning-state events so the app can
show a "Thinking…" indicator. It does not expose the reasoning text in the
first version. This matches Qwen parity (reasoning suppressed) while
avoiding a silent UI during long reasoning turns.

### 11.4 LFM tool-call parser

`LFMToolCallParser` parses the body between 124905 and 124906:

```
[func(a='x', b=1), other({...})]
```

Grammar:

- an optional outer `[ ... ]` list;
- zero or more calls, comma-separated;
- function name `[A-Za-z_][A-Za-z0-9_.-]*`;
- empty `()` or `key=value` pairs separated by top-level commas;
- values:
  - single-quoted strings with `\\`, `\'`, `\n`, `\r` unescaping;
  - JSON objects/arrays, numbers, booleans, `null` via the existing
    `JSONValue` parser;
- trailing whitespace allowed;
- maximum 256 KB body, 16 calls, 64 arguments per call;
- every function name must be in the allowed tool set;
- malformed input fails the turn; no partial tool call is returned.

The parser is pure Swift and heavily unit-tested. It returns the existing
`ParsedToolCall` type with `JSONValue` arguments.

### 11.5 Server tools

- Gemma keeps `GemmaToolSchema.adapted`.
- LFM passes the OpenAI tool definitions through, with basic size and shape
  validation. No Gemma-specific schema rewriting.
- Historical tool calls and tool results are rendered by the LFM template.
  `tool_call_id` is accepted but not rendered by this template.
- Tool-call streaming uses the same OpenAI delta shape as Gemma.
- `finish_reason` is `tool_calls` when a call is parsed, `stop` on EOS,
  `length` on max tokens.
- Image content for LFM is rejected with a clear 400 before generation.

## 12. Product surfaces

### 12.1 CLI

- `--model <path>` stays required.
- `TurboFieldfareRepack` gains a source selector; default Gemma.
- `TurboFieldfareCLI` peeks the manifest family, loads the matching
  tokenizer and runner, and renders the family's chat prompt.
- `--max-new`, sampling flags, context admission, footer, and exit codes
  keep their current shape.
- The footer names the model ID and family.
- Image flags with LFM fail closed.

### 12.2 Server

- `ServerArguments` resolves family from the manifest instead of assuming
  Gemma.
- `ServerInference` builds the runner through `RunnerFactory` and the
  tokenizer through `TokenizerFactory`.
- `/v1/models` reports the manifest model ID.
- Chat completions, streaming, tool calls, and prompt reuse work for LFM.
- `/v1/images` or image-bearing chat requests for LFM return a clear
  unsupported error.
- The server stays loopback-only.

### 12.3 Mac app and decode service

Catalog and selection:

- Two catalog entries, Gemma default.
- `MacAppSettings` persists `selectedModelID`; absent means Gemma.
- The default model path is derived from the selected descriptor's
  filename. `TURBO_FIELDFARE_MODEL_PATH` still pins a path and disables
  location changes.
- The install view lets the user pick Gemma or LFM, shows per-model install
  state, and never runs two installs at once.
- The existing-model picker probes any `.gturbo` directory, shows its
  family and display name, and selects it.
- Switching models unloads the current runner and loads the new one. It is
  blocked while a generation is active, with the existing banner style.
- History is already keyed by model directory, so each model keeps its own
  conversations.
- `ConversationIdentity.forModelDirectory` peeks the manifest family.
  Gemma keeps the existing `GFTokenizer.chatTemplateIdentity` string exactly,
  so old Gemma conversations still load. LFM uses the identity from
  `tokenizer/identity.json`. A missing identity file, a family mismatch, or
  a hash mismatch fails closed instead of replaying tokens under a different
  template.
- `AppModelInstallationProbe` loads with the `ArchConfig` resolved from
  `peekFamily`, not a hard-coded Gemma constant.

Context and controls:

- `AppContextLengthOption` becomes family-aware and computes options from
  the active `ArchConfig`, host memory, and expert-cache slots.
- The current context is clamped to the new model's options when switching.
- Expert-cache slot defaults come from the descriptor (Gemma 16, LFM 8).
- The HUD shows the family and model display name.
- Vision controls and the vision install section are hidden for LFM.
- Images already attached are cleared before switching to LFM; an image
  request while LFM is active is refused.

Thinking and tools:

- The output pane shows a "Thinking…" indicator while the LFM decoder is in
  reasoning, then streams visible answer text.
- Tool calls surface through the existing tool-call UI.
- Reasoning is not stored in history.

Decode service:

- Resolves family from the manifest.
- Uses the same runner factory and context admission.
- Reports the same metrics; no Gemma-specific assumptions in load
  admission.

### 12.4 Documentation

Update:

- `README.md` (supported models, install commands, model selection);
- `docs/SYSTEM_DESIGN.md` (family dispatch, LFM graph, memory);
- `docs/OPENAI_SERVER.md` (LFM model ID, tools, thinking);
- `docs/RUNTIME_CONTROLS.md` (family-specific context/cache defaults);
- `docs/COMMUNITY_BENCHMARKS.md` (LFM commands, no performance claims);
- `docs/IMPLEMENTATION_REFERENCES.md` (LFM references);
- a new `docs/LFM2_5.md` with the architecture, tensor map, chat/tool rules,
  and known limitations.

## 13. Qwen removal and cleanup

The working tree already has no Qwen source because it was reset to
upstream. The remaining artifacts are:

- `scratch/qwen36.gturbo` (~18 GB);
- `scratch/qwen36.gturbo.install.lock`;
- `scratch/qwen36.vision.gturbo.install.lock`;
- `TurboFieldfare Qwen.app/` (untracked);
- any Qwen launcher script;
- Qwen-specific app data/settings/history under the user's application
  support directory.

Cleanup steps:

1. Audit for Qwen references in the working tree, scripts, docs, and app
   data. Only Qwen-specific paths are deleted.
2. Delete the Qwen model directory and lock files.
3. Delete the Qwen app bundle and launcher.
4. Delete Qwen app settings/history after identifying the bundle ID or data
   directory; do not touch Gemma data.
5. Keep the local `backup/qwen36-support-2026-10-24` branch until LFM passes
   validation. Then delete it, since the user asked for Qwen support to be
   removed and it is the last copy.
6. Verify `git status` is clean and no Qwen references remain.

The deletion of the 18 GB model happens during implementation, after the
LFM repacker is ready enough that disk pressure is not a risk. It does not
depend on LFM correctness.

## 14. Testing

### 14.1 Model-free suites

Run through `Scripts/test.sh`.

Format and repacker:

- optional arch fields round-trip; Gemma manifests unchanged;
- LFM manifest with `family`, `layerTypes`, `moeLayerMask`, optional router
  slot, and 6-bit embedding;
- layer-type/mask validation: conv layers cannot be treated as attention,
  dense layers cannot have expert files;
- expert stride arithmetic and 16 KiB alignment;
- synthetic LFM index/config planning: resident entries, 38 expert files,
  64 experts each, 6-bit embedding, dense layers 0–1 resident;
- 6-bit quant metadata and packed-column arithmetic;
- vision repack rejection for LFM.

Tokenizer/chat/thinking/tools:

- pure LFM ChatML renderer: system, user, assistant, tool definitions,
  generation prompt ending in `<think>`;
- tool-history rendering with argument mappings;
- thinking state machine: start in reasoning, `</think>` transition, tool
  call during reasoning, EOS before close;
- tool parser: strings, escapes, JSON objects/arrays, numbers, booleans,
  null, multiple calls, empty args, malformed inputs, unknown tools,
  size/count limits;
- streaming detokenizer with synthetic byte tokens and split UTF-8.

Server and app:

- server model ID/family resolution, LFM chat validation, tool schema
  pass-through, image rejection, streaming tool deltas;
- app settings migration (absent selection → Gemma), catalog defaults,
  context option computation, model switch guards, vision visibility;
- decode-service load admission with LFM geometry.

Kernels (device tests, no model):

- 6-bit dequant/GEMV vs CPU reference for random packed data;
- 6-bit embedding lookup vs CPU dequant;
- standard vs Gemma RMSNorm;
- Q/K norm + full NeoX RoPE vs CPU;
- conv decode and prefill vs CPU; chunk equivalence; state update;
- router sigmoid/bias/top-4/normalize vs CPU, including tie-break;
- attention with 32 Q heads, 8 KV heads, head dim 64, scale 0.125;
- SiLU-gated dense FFN vs CPU for small synthetic shapes.

Model-gated suites skip when `scratch/lfm2.5-24b-a2b.gturbo` and its
tokenizer sidecar are absent:

- install/verify the LFM pack;
- CLI chat prompt produces non-empty, coherent output and stops on EOS;
- prefill vs token-by-token decode logits equivalence for a fixed prompt;
- chunk-size invariance (1, 7, 128);
- rewind and replay equivalence for KV and conv state;
- tool call with a simple weather tool yields a valid `ParsedToolCall`;
- server chat completion, streaming, tool loop, and image rejection;
- app launch, model selection, generation, thinking indicator, stop, new
  chat, history.

### 14.2 Gemma regression

- Existing model-free suite unchanged.
- If `scratch/gemma4.gturbo` is present, run a short Gemma CLI prompt and
  compare the footer shape and output sanity to the pre-change baseline.
- Do not delete, reinstall, or modify the Gemma model.

### 14.3 Validation gates

1. `Scripts/test.sh` green.
2. `swift build -c release` green.
3. LFM install completes, manifest verifies, all file hashes pass.
4. LFM CLI output is coherent; no NaN/Inf; EOS observed.
5. Prefill/decode/chunk/rewind equivalence tests pass.
6. Server OpenAI smoke passes, including a tool loop.
7. App smoke passes on the host with LFM selected.
8. Gemma smoke passes unchanged.
9. Qwen artifacts and backup branch are gone; `git status` clean.

## 15. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| 6-bit packing or scale grouping wrong | Garbage logits | CPU reference tests from the MLX formula; embedding/lm-head checks on a known row; end-to-end sanity |
| Conv state/rewind corruption | Wrong continuation output | Ring history, replay equivalence tests, fail closed on deep rewind |
| Router bias/normalization wrong | Wrong expert selection, subtle quality loss | CPU reference, top-4 tie-break tests, compare with reference implementation |
| Dense layers packed as experts | Missing/wrong FFN | `moeLayerMask` validation, planner tests, fail-closed runtime lookup |
| Manifest changes break Gemma | Production regression | Optional fields with Gemma defaults; full Gemma suite; Gemma graph unchanged and shared kernel init defaults preserve Gemma's 16-head path |
| LFM memory exceeds 16 GB host | OOM or swap | Admission math, 8 default slots, 8K default context, measured memory pressure before runs |
| App model switching complexity | UI bugs, wrong tokenizer/runner | Family from manifest, catalog tests, block switching during generation, per-model history |
| ByteLevel detokenizer corrupts UTF-8 | Broken streaming text | Byte buffering tests with split multibyte sequences |
| Tool-call parser rejects valid calls | Tool loop fails | Template-derived fixtures, permissive grammar for JSON/Python values, size limits |
| Server image path silently drops images | Violates fail-closed rule | Explicit family check before generation; test |
| Performance is disappointing | User expectations | Report measured tok/s and memory honestly; no ceilings; optional fused 6-bit head later |

## 16. Rollout

1. **Format and repacker**: optional arch fields, quant router optional,
   6-bit index support, LFM config parsing, planner, expert layout with
   dense layers, source catalog, tokenizer sidecar.
2. **Runtime and CLI**: family resolution, LFM weights view, 6-bit kernels,
   conv/attention/dense/MoE graph, KV/conv state, runner factory, CLI.
   First end-to-end LFM generation.
3. **Tokenizer and tools**: LFM tokenizer, chat template, thinking split,
   tool parser, replay/continuation.
4. **Server**: family-aware arguments, tokenizer/runner, tools, image
   rejection, streaming.
5. **App and decode service**: catalog, install/select, context/cache
   options, thinking indicator, vision hiding, history.
6. **Docs, tests, cleanup, validation**: full suites, model-gated runs,
   remove Qwen artifacts, delete backup branch.

Each phase lands with its tests. No phase claims completion without the
relevant suite output.

## 17. Decisions and defaults

These are the defaults this design proposes. They can be changed before
implementation.

1. **Reasoning display**: show a "Thinking…" indicator, not the reasoning
   text. This matches Qwen parity (reasoning suppressed) without a silent
   UI. The reasoning text is discarded and never stored.
2. **Qwen backup branch**: keep `backup/qwen36-support-2026-10-24` only
   until LFM validation, then delete it. All user-visible Qwen artifacts and
   the model are deleted during implementation.
3. **Default model**: Gemma 4 remains the default for new and existing
   users. LFM is opt-in.
4. **Model selection**: both models can be installed side by side; switching
   unloads the current one and uses per-model history.
5. **LFM sampling defaults**: the app keeps its current global defaults
   (temperature 0.2, Top-K 64, Top-P 0.95) for both families; the LFM model
   card's 0.1/50/1.05 recommendation is documented and can be set manually.
   The CLI keeps its existing sampling defaults.
6. **LFM context default**: 8,192 tokens, with larger options offered only
   when context admission passes. Expert-cache slots default to 8.
7. **LFM performance**: correctness first. No fused 6-bit lm-head, no
   speculative decoding, no new profiling in this work.

## 18. Builder-check resolutions and remaining implementation details

A fresh builder-check review raised five design-level questions. All five
are resolved in this document:

1. **Non-KV admission and expert stride**: `ArchConfig` carries a
   family-specific admission profile (`expertStrideBytes`,
   `expertCacheLayers`, `includedExpertCacheSlots`,
   `nonKVRuntimeFootprintBytes`). LFM charges its expert cache from slot 0;
   Gemma keeps its byte-identical 16-slot-inclusive 2 GiB profile. The
   manifest's top-level `expertStride` is validated against the profile.
   See §10.6.
2. **Rewind limit**: `LFM2ForwardRunner` owns `convRewindLimit = 128` and
   enforces it before delegating to `kv.rewind`. `KVCacheManager`'s
   `maxRewindTokens` is not used for LFM because it returns `maxContext`
   when there are no SWA layers. See §10.4.
3. **32-head attention**: `Attention` takes `maxQHeads` at init. Gemma
   constructs 16, LFM constructs 32. There is no global constant change and
   no Gemma scratch or pipeline change. See §10.5.
4. **Tool-result continuation**: LFM uses a generic render-prefix/render-full
   longest-common-prefix plan, with a mandatory full-replay fallback. Gemma
   keeps its existing boundary logic. See §11.2.
5. **Conversation/template identity**: LFM uses
   `tokenizer/identity.json` with a constant identity and hashes of the
   installed template/tokenizer files. Gemma keeps its existing constant
   identity so old history loads. See §11.1 and §12.3.

The following are implementation choices, not design blockers:

- Exact split between `Model` generic tensor access and the LFM weights
  view.
- Whether the MoE decode kernel is parameterized or duplicated for top-4.
- The exact shape of the app's per-descriptor install state and catalog
  migration.
- The test-fixture strategy for the LFM tokenizer (pure renderer plus
  model-gated tokenizer tests versus a trimmed fixture).
- Whether the conv prefill kernel updates the ring directly or returns the
  last two rows for the caller to store. Either is acceptable if the
  prefill/decode and rewind equivalence tests pass.
