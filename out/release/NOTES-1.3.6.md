# GLaDOS 1.3.6 — it mines, and it repairs itself

A hundred and thirty-nine commits since 1.3.5, and two of them are the release.
The kernel mines several coins at once against a pool that exists and is
deployed. And on the day it first booted on real hardware it died of a
thermometer, so it now survives its own selftests, diagnoses what broke, and
fixes it without being asked.

## The two measurements that matter

**It booted on the laptop it was written for, and the subsystem that killed it
now reports instead of halting:**

```
glados> power
  vendor intel    hypervisor no
  dts yes  package yes  hwp yes  aperf yes  turbo yes
  66 C, 34 below the limit of 100
  base 2700 MHz, running at 4094 MHz over 20 ms
  governor balanced
  hwp present but not enabled, so its range cannot be read yet
```

That last line is the whole release in one sentence. CPUID saying HWP exists is
not permission to read its capabilities — the register is gated behind
`IA32_PM_ENABLE`, this laptop supports HWP and boots with it off, and the very
first read took a `#GP` that stopped the machine before the shell existed. No
storage, no namespace, no model, for a temperature nothing needs. **It is the
first bug in this tree that only bare metal could find.**

**And it repairs itself, judged rather than assumed:**

```
[selftest] power general protection fault -- this subsystem is unavailable
[repair] 1 subsystem(s) to try
  power          repaired by 'skip-hwp', and the check now passes
[boot] 1 subsystem(s) did not survive their own selftest, 0 still broken:
  power          general protection fault  (skip-hwp)
                 in dev::power::gf63_shape +0x16
```

The judge is the check that failed. Apply a repair, re-run the selftest, and
passing is the verdict — which is the rule `godel` already imposes on anything
this machine adopts, arriving somewhere it can be applied in a boot.

## What arrived

**The miner.** Block headers, targets and the midstate, pinned by a real block
whose hash is public. SHA-256d, yespower, NeoScrypt and BLAKE2s in ring 0,
every one against its own upstream vectors rather than a fixture this project
chose. Slices that genuinely run in parallel across cores — the first `unpin`
callers outside a selftest, and the audit that permission required. A coin per
slice, so four coins are worked at once on four algorithms.

**A pool, and it has met the actual internet.** It speaks Stratum V1 to a real
upstream, rebuilds the coinbase, and serves assembled headers downstream over
one connection carrying several coins — a field Stratum does not have, which is
the entire reason the protocol exists. Shares are validated by recomputing
them. PPLNS accounting on a ledger that survives a restart, difficulty per
miner per coin, and every bound driven past its limit until three of them
turned out to be wrong. It runs as a service.

**Mining stands down for the model.** On by default, and the default is the
argument: this kernel's reason for existing is the model in it, and a miner
quietly taking a third of the machine's arithmetic away from inference has
inverted that. Measured, not assumed — four slices cost about a third.

**The machine survives a subsystem that does not.** A boot check declares what
it is worth: vital halts with a reason, optional records the failure and boots
on without it. Guards nest properly, a guard that could not guard says so
rather than reporting a pass, and a panic is caught only inside a selftest
window. Fault reports name the function now instead of an offset, from a
13,817-symbol table built out of the linker map.

**Repairs survive a reboot**, in a file on the boot volume applied before
anything can fault — and a repair that prevents boot **withdraws itself** on the
next attempt, in one reboot, with no operator and no recovery media.

**The guest says which hypervisor it is on**, and what that hypervisor has
configured wrongly. Both VirtualBox and VMware default to SATA and there is no
AHCI driver here, so a guest on defaults boots to a shell with nothing behind
it. It now says so in those words.

## Three things this release learned by being wrong

**QEMU reports itself as VMware.** CPUID leaf `0x40000000` read
`56 4d 77 61 72 65 56 4d 77 61 72 65` under QEMU — `VMwareVMware`, exactly.
That is `vmware-cpuid-freq`, on by default, borrowing VMware's convention for
the timing leaf. Naming a hypervisor from CPUID would have told every QEMU user
they were on VMware, in writing. Named from PCI vendor ids instead.

**`LoadedImage` was missing a field.** The UEFI spec puts `FilePath` between
`DeviceHandle` and `Reserved`, and it simply was not in the struct — so every
field below it read one early. `image_base` was reading `LoadOptions` and came
back null, which sent boot down a fallback that worked, and `image_size` was
reading `ImageBase`, which made the "is this rip inside the image" bound about
two gigabytes wide. That check exists to stop a wild jump being reported as a
plausible RVA and at that width could not have stopped one.

**A fault report can name the wrong function.** The symbol table holds public
symbols, so anything inlined has no entry and the search returns whatever
precedes it. A deliberate fault in a small `dev::power` helper was reported as
`doom::play::dispatch +0x14a2` — five kilobytes into an unrelated function in
an unrelated subsystem. Average spacing is about a hundred bytes, so an offset
in the thousands means the real function is absent rather than enormous.

## What is not true yet

**No coin has been mined.** The pool has served real jobs and shares have been
verified, but no share has gone up to a chain, and at this hashrate none will.

**The payout loop does not close, and the arithmetic is the reason.** At an
800-miner 36-hour event the pot is $84 and a share is $0.105. Every on-chain
route costs more than the pot: $195 to claim a transfer, $456 to claim and
swap. The granularity is the cost, not the route. So "mine and get paid in
$GLADOS" is not a claim this release makes.

**Nothing has been booted in VirtualBox or VMware.** The device identification
and the setup advice are asserted against synthetic device lists — nine claims
covering both vendors — and derived from what the kernel requires. Not from a
run anybody did.

**The miner uses four threads of sixteen**, and the limit is the task table
rather than the hardware: `MAX_TASKS` is 24, every core adopts an idle task,
and slots are never reclaimed. A machine with more cores has *fewer* slices
available, which is exactly backwards.

**The yespower figures are a floor.** 125 H/s under emulation against upstream's
~1000 on bare metal, and the 8x gap is emulation and the reference
implementation together, neither separated from the other.

## The gate

```
48 passed, 0 failed, 0 not run
```

`diag all` under QEMU with WHPX at `-smp 4`, on the image this release ships.

One flake worth recording rather than hiding: the timer selftest failed once in
three runs of an unchanged binary, reporting `50 ticks in 94 ms of TSC time`
where the other two read 488 ms and 496 ms and passed. That is the host
descheduling the guest mid-measurement — the single-sample error `video bench`
was rewritten to stop making — and the timer check still takes one sample.
