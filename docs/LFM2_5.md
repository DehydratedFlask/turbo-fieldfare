# LFM2.5 24B-A2B

TurboFieldfare can install and run `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` beside
the Gemma 4 model. Gemma 4 remains the default. LFM is selected explicitly by
model path, by the app's model picker, or by the server/CLI reading the
installed manifest.

## Pinned checkpoint

| Field | Value |
| --- | --- |
| Repository | `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` |
| Revision | `667408d494bde191c2cb9dd4f4549827816b9275` |
| Source index SHA-256 | `7887b03265aef7824c8e5b9a51667e486f889749240d8c59a95f3cd899e86e4a` |
| Download | about 13.57 GB |
| Installed `.gturbo` | about 13.57 GB |
| Source shards | 3 safetensors shards |

The repacker streams the pinned ranges and verifies the source index before it
reads a shard. It never stages the whole Hugging Face checkpoint.

## Install

```bash
swift run -c release TurboFieldfareRepack \
  --model-source lfm2_moe \
  --output scratch/lfm2.5-24b-a2b.gturbo
```

Resume an interrupted install with the same command plus `--resume`. Discard
the partial ranges with `--discard-partial --output scratch/lfm2.5-24b-a2b.gturbo`.

The Mac app exposes the same source in its install picker. Gemma 4 stays the
default; choosing LFM changes only the selected model.

## Architecture

- 40 layers. 30 are short-convolution layers; 10 are full-attention layers at
  indices 2, 6, 10, 14, 18, 22, 26, 30, 34, and 38.
- Layers 0–1 are dense MLPs. Layers 2–39 are mixture-of-experts layers.
- Hidden size 2,048. Dense intermediate 11,776. MoE intermediate 1,536.
- 64 experts per MoE layer, top-4 selected per token.
- 32 query heads, 8 KV heads, head dimension 64 (4:1 GQA).
- Vocabulary 128,000; maximum positions 131,072.
- Full NeoX RoPE, theta 10,000,000. No sliding window. No logit softcap.
- Standard RMSNorm (`weight * x / rms(x)`, epsilon 1e-5), not Gemma's
  `1 + weight` norm.
- SiLU/SwiGLU activation.
- Short convolution: depthwise, kernel 3, no bias. The cache stores the last
  two gated `Bx` vectors, not conv outputs.
- Tied 6-bit word embedding and lm-head. There is no `lm_head` tensor and no
  separate final norm; `model.embedding_norm` is the final norm.

```text
residual = h
h = operator_norm(h)
h = operator(h)          # short conv or full attention
h = residual + h

residual = h
h = ffn_norm(h)
h = ffn(h)               # dense MLP or MoE
h = residual + h
```

## Tensor map

The repacker accepts the checkpoint's raw `model.*` names:

| Tensor | Shape | Storage |
| --- | --- | --- |
| `model.embed_tokens.weight` | 128000 × 2048 | resident, 6-bit affine |
| `model.embed_tokens.scales` / `.biases` | 128000 × 32 | resident, BF16 |
| `model.embedding_norm.weight` | 2048 | resident, BF16 |
| `layers.N.operator_norm.weight` | 2048 | resident, BF16 |
| `layers.N.ffn_norm.weight` | 2048 | resident, BF16 |
| `layers.N.conv.in_proj.weight` | 6144 × 2048 | resident, 4-bit |
| `layers.N.conv.out_proj.weight` | 2048 × 2048 | resident, 4-bit |
| `layers.N.conv.conv.weight` | 2048 × 3 × 1 | resident, BF16 |
| `layers.N.self_attn.q_proj.weight` | 2048 × 2048 | resident, 4-bit |
| `layers.N.self_attn.k_proj.weight` | 512 × 2048 | resident, 4-bit |
| `layers.N.self_attn.v_proj.weight` | 512 × 2048 | resident, 4-bit |
| `layers.N.self_attn.out_proj.weight` | 2048 × 2048 | resident, 4-bit (installed as `o_proj`) |
| `layers.N.self_attn.q_layernorm.weight` | 64 | resident, BF16 |
| `layers.N.self_attn.k_layernorm.weight` | 64 | resident, BF16 |
| `layers.N.feed_forward.gate_proj.weight` | 11776 × 2048 | resident, 4-bit (N < 2) |
| `layers.N.feed_forward.up_proj.weight` | 11776 × 2048 | resident, 4-bit (N < 2) |
| `layers.N.feed_forward.down_proj.weight` | 2048 × 11776 | resident, 4-bit (N < 2) |
| `layers.N.feed_forward.gate.weight` | 64 × 2048 | resident, F32 (N ≥ 2) |
| `layers.N.feed_forward.expert_bias` | 64 | resident, F32 (N ≥ 2) |
| `layers.N.feed_forward.switch_mlp.gate_proj.weight` | 64 × 1536 × 2048 | per-expert stream, 4-bit |
| `layers.N.feed_forward.switch_mlp.up_proj.weight` | 64 × 1536 × 2048 | per-expert stream, 4-bit |
| `layers.N.feed_forward.switch_mlp.down_proj.weight` | 64 × 2048 × 1536 | per-expert stream, 4-bit |

Each quantized tensor carries BF16 `scales` and `biases` of shape
`[rows, columns / 64]`. The 6-bit payload is MLX's least-significant-bits-first
packing, four values into three bytes. The resident expert stride is 5,308,416
bytes. The attention output projection is spelled `out_proj` in the checkpoint
and installed as `o_proj`; the repacker translates that one name.

## Chat template

The installed `tokenizer/identity.json` names the template
`lfm2.5-24b-a2b-chatml-v1` and hashes the template and tokenizer files. A
mismatch fails load rather than rendering a different prompt.

- BOS is 124,894 (`<|startoftext|>`); EOS is 124,900 (`<|im_end|>`); pad is
  124,893.
- The generation prompt ends with `<|im_start|>assistant\n<think>`, so decode
  starts in the reasoning channel.
- `</think>` (124,902) ends reasoning. Text before it is discarded and never
  returned as visible content or stored as history.
- Tool calls are `<|tool_call_start|>[func(arg='x', n=1)]<|tool_call_end|>`.
  Arguments are Python literals: single-quoted strings with `\\`, `\'`, `\n`,
  and `\r`; `True`/`False`/`None`; JSON objects and arrays for nested values.
- Tool definitions pass through unchanged; the template serializes them with
  `tojson`.

## Defaults

- Context: 8,192 tokens.
- Expert-cache slots: 8.
- Gemma 4 remains the default model for new and existing users.
- FP16 KV is about 20,480 bytes per token (10 full-attention layers × 2 × 8
  KV heads × 64 head dimension × 2 bytes). At 131,072 tokens it is about
  2.68 GB.
- The conversation can rewind up to 128 tokens, the span the short-conv ring
  retains. A deeper rewind fails closed; the app replays a saved chat through a
  fresh prefill instead.

## No vision

The pinned LFM checkpoint has no vision tower. Image input is refused with an
unsupported-image error in the CLI, server, app, and decode service. It is
never accepted and answered as if the image were absent.

## Known limitations

- The Mac app does not expose a tool loop for LFM. Tool calls are supported by
  the server and by the tokenizer/parser layer.
- Server prompt reuse uses the rendered-prefix fast path; a miss full-renders
  and prefills. There is no Gemma-style bridge encoder for LFM.
- Reasoning text is not shown or saved. The app may show a "Thinking…"
  indicator.
- No LFM performance claim is made until it is measured on the host. The
  runtime is tuned for correctness and reasonable memory use first.
