# Qwen3.5 and Qwen3.5-MoE, as an extension

Status: design. Nothing below is implemented yet.

## Why bother

Not for the parameter count. For the memory.

Qwen3.5 interleaves two kinds of layer. One in four is ordinary softmax
attention with a KV cache; the other three are **linear attention** carrying a
fixed-size recurrent state. The tensor names say exactly which mechanism:
`A_log`, `dt_bias`, `conv1d.weight`, `in_proj_a`, `in_proj_b`, `in_proj_qkv`,
`in_proj_z`, `norm`, `out_proj`. That is Gated DeltaNet with a Mamba-style
short causal convolution.

For a decoder that emits one token at a time -- which is the only thing this
kernel ever does -- a linear-attention layer needs no cache at all. It needs a
state matrix of `key_head_dim x value_head_dim` per head, and that matrix is
the same size at token 32 and at token 32,768.

    Qwen3-0.6B    28 dense layers        7.0 GiB of KV at 32k (f32)
    Qwen3.5-0.8B  6 full + 18 linear     786 MiB of KV at 32k (f32)
                                         + 18 MiB of fixed state, context-independent

The KV problem that Phase 2 of the main plan works around by splitting the
cache per layer and quantising it to int8 mostly stops existing. That is the
argument for doing this.

## What is actually different

Measured from `config.json` on Qwen/Qwen3.5-0.8B and Qwen/Qwen3.5-35B-A3B,
and from the safetensors index.

| | Qwen3 (works today) | Qwen3.5 | Qwen3.5-MoE |
|---|---|---|---|
| layers | all dense attention | 6 full + 18 linear | 10 full + 30 linear |
| hidden | 1024 | 1024 | 2048 |
| heads q/kv | 16 / 8 | 8 / 2 | 16 / 2 |
| head_dim | 128 | 256 | 256 |
| RoPE | full width | **25% of head** | 25% of head |
| RoPE theta | 1e6 | 1e7 | 1e7 |
| attn output gate | no | **yes** | yes |
| FFN | dense SwiGLU | dense SwiGLU | **256 experts, top-8, + shared** |
| vocab | 151,936 | 248,320 | 248,320 |
| vision tower | no | yes (skipped) | yes (skipped) |
| MTP head | no | yes (skipped) | yes (skipped) |

Five separate changes, any one of which produces fluent nonsense on its own.
This is the failure mode the project keeps meeting: a wrong architecture does
not throw, it degrades.

## Scope

**In:** the text model. Both dense and MoE variants. Linear attention in its
recurrent (single-token) form only.

**Out, deliberately:**

* The vision tower (`model.visual.*`). No image path exists anywhere in this
  kernel and adding one is a separate project.
* The MTP head (`mtp.*`). It speculates; the kernel has no speculative decode
  loop to speculate for.
* Chunked/parallel linear attention. The training-time formulation processes a
  whole sequence at once. Decoding needs only the recurrence, which is far
  simpler, and prompt ingestion can run the recurrence per token.
* mRoPE sections. With text-only input the three position streams are equal,
  so mRoPE reduces to ordinary RoPE. **This must be verified against the
  oracle, not assumed.**

## Design

### Nothing replaces anything

`GLADOSM3` v3 keeps loading exactly as it does. A new `arch` field selects the
forward pass:

```rust
pub enum Arch {
    Qwen3Dense,      // what exists; v2 and v3 files
    Qwen35Hybrid,    // v4
    Qwen35MoeHybrid, // v4, moe fields non-zero
}
```

Header goes to v4 with: `arch`, `full_attention_interval`, `partial_rotary_factor`,
`linear_key_head_dim`, `linear_value_head_dim`, `linear_num_key_heads`,
`linear_num_value_heads`, `linear_conv_kernel_dim`, `num_experts`,
`num_experts_per_tok`, `moe_intermediate_size`, `shared_expert_intermediate_size`.
A v4 reader must still accept v2 and v3, as v3 already accepts v2.

### The layer schedule is data, not arithmetic

`layer_types` in the config is an explicit list. `full_attention_interval: 4`
happens to describe it today, but deriving the schedule from the interval means
a checkpoint that breaks the pattern loads and runs wrong. The header carries a
bitmap, one bit per layer, written from the list.

### State, not cache, for linear layers

```rust
struct LinearState {
    /// [heads][key_dim][value_dim], f32. Size independent of context.
    s: Vec<f32>,
    /// Last `conv_kernel_dim` inputs per channel, a ring.
    conv: Vec<f32>,
    conv_at: usize,
}
```

Kept f32. The KV cache is int8 because it is enormous and read once per token;
this state is small and read *and written* every step, so quantisation error
compounds through the recurrence rather than averaging out.

### MoE

Top-8 of 256 experts plus a shared expert, `moe_intermediate_size: 512`. Only
the selected experts are read, so bytes-per-token is roughly
`8 x 3 x hidden x 512` rather than the full expert bank. That is what makes a
35B model conceivable here at all: the active path is small even though the
file is not.

Expert weights stay in the mapped blob and are addressed by arithmetic, like
every other tensor.

## The oracle problem, and it is the hard part

The venv has numpy and tokenizers. No torch, no transformers.

For Qwen3 that was survivable: the architecture is widely understood, and the
RoPE convention could be settled by reading generated text for a known fact.
Gated DeltaNet has no such tell. If the recurrence is implemented wrong in both
`reference.py` and `model.rs` in the same way, they agree, the output is
fluent, and nothing catches it.

So the first milestone is not code. It is a **golden fixture**: install torch
and transformers, run the reference model on a fixed prompt, and save hidden
states per layer plus final logits to an `.npz`. Everything after is diffed
against that file. Roughly 2.5 GB of install against 53 GB free.

Without the fixture this port is unverifiable, and an unverifiable port of an
architecture whose failure mode is fluent nonsense is worse than none.

## Phases

0. **Fixture.** torch + transformers in the venv, golden per-layer activations
   and logits for Qwen3.5-0.8B on a fixed prompt. Nothing else proceeds first.
1. **reference.py** grows the hybrid forward: partial RoPE, attention output
   gate, the Gated DeltaNet recurrence, and the MoE router. Diffed layer by
   layer against the fixture until it matches.
2. **convert.py** reads the new config, strips the `model.language_model.`
   prefix, skips `model.visual.*` and `mtp.*`, and writes v4.
3. **model.rs** ports the verified reference. Dense first, MoE second.
4. Measure: tokens/sec and memory against Qwen3-0.6B at 512 and at 32k, which
   is where the whole argument is supposed to pay off.

Phase 0 and 1 are where the risk is. Phase 3 is transcription of something
already known correct, which is the position this project always tries to
reach before writing kernel code.
