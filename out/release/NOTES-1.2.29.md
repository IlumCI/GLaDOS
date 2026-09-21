A correctness release. One real bug, one new diagnostic, two report-only verbs.

## Fixed: the adapter could not learn, and never said so

`Dora::new` started **both** low-rank factors at zero. Two zero matrices
multiply to a zero product and, worse, to a zero *gradient* — so the low-rank
branch of every QDoRA adapter this system has ever trained could not move.
What was left was the per-row magnitude, which rescales a row and does nothing
else.

Nothing failed. `train adapter` ran, reported a falling loss (the magnitude was
still training), wrote a file, and the judges compared the result against the
baseline and rejected it.

**If you have been running trials, read this part.** Those rejections were made
against an adapter that was structurally incapable of learning, so they say
nothing about the grid point they were testing. After updating:

```
godel forget
```

That clears the tried-markers and walks the search grid again. J4's cost
figures stand — the shapes did not change, only what the shapes could learn.

`diag adapterinit` is the check that would have caught this, and it ships here
too: it differences the adapter's own gradient and asserts the low-rank branch
is not identically zero on a freshly constructed one.

## New: `gpu`

```
gpu
```

Reports what is on the PCI bus, whether every bridge forwards a bus that
answers, and — when the part is NVIDIA's — maps BAR0 and decodes
`NV_PMC_BOOT_0` into a chip id. From ring 0, with no vendor driver anywhere in
the path.

It refuses to name a chip it cannot identify rather than guessing. Under QEMU
that is exactly what it does: the emulated VGA answers `0x001c222c`, which is
not an NVIDIA boot register, and the decoder declines.

**If you run this on real hardware, the output is useful.** The binding
constraint on that work is that there is one machine here to test on and no
second sample. Every install that runs `gpu` and reports back is a data point
on whether a from-scratch, non-Unix OS can see and address a discrete GPU.

## New: `study` and `abstract`

Both report-only in this release. They measure and print, and change nothing.

- `abstract` enumerates the subtrees of every program under `/ai/tools`,
  canonicalises them so structurally identical code with different names
  collides, and ranks what repeats by how much naming it would save.
- `study` and `study seq` measure what learning one field costs the fields
  already learned.

## Tested under emulation, and not on hardware

Every check here ran under QEMU with the Windows hypervisor accelerator. The
boot selftests pass with no failures, and `diag` re-runs the suites by name.

**None of it has run on real hardware.** That matters most for `gpu`, which has
never seen a GPU: its refusal path is exercised and its success path is not.

## Files

| | |
|---|---|
| `glados-1.2.29.efi` | the kernel image, and what the in-OS updater installs |
| `glados-1.2.29.efi.sig` | its detached GLADOSIG signature |
| `manifest` | the signed manifest the updater reads |
| `glados-1.2.29.iso` | a full install image: kernel, model, tokenizer, roots. 576 MB |

The ISO for this release was built by hand. From the next release CI builds
them, from a payload pinned outside the repository and verified against
recorded digests before the image is assembled.

## Updating in place

On a machine that already trusts the signing key:

```
update check
update fetch
update stage <first eight characters of the digest>
```

then reboot. The image is replaced by the *next* boot, not the running one.
Installing fresh is the ISO.
