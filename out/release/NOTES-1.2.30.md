**The kernel is unchanged.** `git diff v1.2.29 HEAD -- src/` is empty: this
image differs from 1.2.29 by its version string and nothing else. If you are
running 1.2.29 there is no reason to update.

The release exists because the deliverable is install media, and only a tag
produces it. **This is the first release to ship a 2B model.**

## The 2B distill, at two context lengths

Qwen3.5-2B, Apache-2.0, converted to the kernel's v4 layout. A hybrid: three
layers in four are linear attention and the fourth is ordinary softmax, which
is why its context can be long without the KV cache growing the way a dense
model's would.

| | |
|---|---|
| Parameters | 1,666,299,904 |
| Weights | 1,852,736 KiB, int8 |
| Vocabulary | 248,320 rows, 248,070 covered by the tokenizer |
| Layers | 24, of which 6 full-attention and 18 linear |

Measured on a real boot, not derived:

| context | resident state | kernel heap in use |
|---|---|---|
| 8,192 | 100,966 KiB | 130 MiB of 1,341 |
| 32,768 | 341,350 KiB | 347 MiB of 1,341 |

Both were booted from the ISOs this release publishes. **The 32k image was very
nearly not built**, because the arithmetic said it could not fit and the
arithmetic was wrong in two places: `HEAP_LADDER`'s 320 MiB is the first
*contiguous* region rather than the whole heap, and the kernel holds the KV
cache int8 where the converter reports it f32.

The weights are read whole into a pool before `ExitBootServices`, so a machine
running the 2B needs room for 1.89 GB of model plus the heap. Tested with 4 GB.

## Which image

- **`qwen3-0.6b`** — the small one, 576 MB. Fastest per token, lowest memory.
- **`q35-2b-8192`** — the 2B at 8k context. The general choice.
- **`q35-2b-32768`** — the 2B at 32k. Same weights, more room, 210 MiB more
  resident state.

The two 2B images are the same model. They differ in four bytes of header,
which is what tells the kernel how large a KV cache to build.

## The ISOs are built in CI now

They used to be a manual build on one machine, because the payloads are
hundreds of megabytes of weights that exist nowhere in the repository. They
still do not live there: they live in pinned releases, and the build fetches
them and checks them against digests that *are* in the repository before
assembling anything.

That check is the point. A truncated download otherwise produces an ISO that
builds without complaint, boots without complaint, and cannot load the model,
because nothing else in the build knows how long the weights should be.

## Files

| | |
|---|---|
| `glados-1.2.30.efi` | the kernel image, and what the in-OS updater installs |
| `glados-1.2.30.efi.sig` | its detached GLADOSIG signature |
| `manifest` | the signed manifest the updater reads |
| `glados-1.2.30-qwen3-0.6b.iso` | install image, 0.6B, 576 MB |
| `glados-1.2.30-q35-2b-8192.iso` | install image, 2B at 8k, 1.81 GB |
| `glados-1.2.30-q35-2b-32768.iso` | install image, 2B at 32k, 1.81 GB |

## Still true from 1.2.29

Everything in that release's notes about the adapter fix stands, including the
part that matters most: if you have been running `godel` trials on a build
older than 1.2.29, those verdicts were recorded against an adapter that could
not learn, and `godel forget` is how you walk the grid again.

## Tested under emulation

Every figure here is from QEMU with the Windows hypervisor accelerator,
including the boots of both 2B images. The boot selftests pass with no
failures. **None of it has run on real hardware.**
