# 1.3.2

Twenty-three commits since 1.3.1, and `git diff v1.3.1 HEAD -- src/` is 13,642
insertions across 34 files. Four things happened, and one of them was not
planned when the release before this one went out.

## DOOM went from drawing walls to being playable

1.3.1 shipped a renderer. This ships a game. On FreeDoom E1M1 the monsters
notice you, walk to you, shoot you, and kill you.

    E1M1      53 monsters, 126 damage in 10 s walking forward, PLAYER DIED
    fixture   1 monster, 1 awake, nearest 42 from 283, 27 damage in 6 s

What landed: visplanes, so a floor no wall claimed still gets filled. Doors,
lifts, floors, switches, lights and the exit. An inventory, with pickups that
can refuse (a medikit at full health stays on the floor) and doors that want a
key. Hitscan shooting with a chain reaction through barrels. A weapon with its
own state machine. Monsters that look, chase and shoot. And things that carry a
position and a velocity, which is the substrate projectiles will need.

**`src/doom/info.rs` is generated now, and the hand-written table it replaced
was the bug.** That table carried 44 doomednums on the argument that copying
`mobjinfo` would be copying a game's content. The argument was sound and the
conclusion was wrong: a doomednum means what id decided it means, so a partial
table is not a smaller version of the right answer, it is a level with 49 things
missing from it. `tools/doominfo.py` emits 967 states, 137 kinds and 118
doomednums from an upstream checkout, the arrangement `rtlconv.py` already had
with the wireless tables.

    before   179 drawn, 49 of a doomednum the table does not carry,
             52 monsters with no rotation-0 lump, 12 invisible
    after    280 drawn, 12 start(s), 0 of unknown kind, 0 with no picture

The 12 are the four player starts and eight deathmatch starts, which are
positions the map format defines with no `mobjinfo` row behind them.

**The rate of fire was predicted wrong and measured right.** The obvious reading
is that a weapon fires once per attack chain, so a pistol's 19 tics would be 1.8
shots a second. Holding the trigger for three seconds fired eight shots, not
five. `A_ReFire` runs on entry to its state and restarts the chain there, so
that state never spends its own tics while the trigger is down. The real cycle
is 14 tics, which is 2.5 shots a second, which is what DOOM's pistol does.
`weapon::held_cycle` computes it and a claim asserts the 14, so a table that
changed would fail there rather than quietly changing how the game plays.

**The random table is DOOM's**, which is what makes a run repeatable: two
identical scripts spent 17 draws each and reported the same damage to the byte.
That counter is also why a coincidence did not become a claim. Three runs each
reported exactly 27 damage taken, which looked like a cap. It was not: 6 seconds
gives 27, 15 gives 57, 30 gives 114 and kills the player.

## It runs Linux binaries

Not planned for this release. An unmodified x86-64 ELF loads off the boot
volume, runs at ring 0, and its `syscall` instructions trap into the kernel.

    [linux] 185 byte(s), 1 segment(s), 185 byte span at 0x17f1000, entry 0x17f1078
    hello from ring 0
        1 write        0x1 0x17f10a7 0x12 -> 18
      231 exit_group   0x5 0x17f10a7 0x12 -> 0
      exited 5 after 2 syscall(s)

`syscall` is not the ring-3 door. It loads RIP from `LSTAR` and CS from
`STAR[47:32]` whatever the current privilege level, so a guest already at CPL 0
traps exactly as one at CPL 3 would. `STAR[47:32]` is this kernel's `KERNEL_CS`
and the processor derives SS as that plus eight, which is already `KERNEL_DS`.
A same-ring trap therefore costs nothing to set up.

Answered: `write`, `brk`, `mmap`, `munmap`, `mprotect`, `arch_prctl`.
Everything else is recorded and refused with `-ENOSYS`, which is the instrument
rather than a gap. A binary that dies on call 47 has said which call to write
next.

**`arch_prctl(ARCH_SET_GS)` has to refuse, and that is the most specific thing
this work has found.** `cpu::percpu` points GS at each core's own block and
`gs:[0]` is how the allocator discovers which core it is billing. There is no
privilege boundary here to stop a guest overwriting it, so a guest setting GS
would leave the next kernel allocation reading its thread-local storage as a
per-core structure. `diag census` passes on a boot where a guest has already set
FS, which is how that restore is checked rather than assumed.

**Every guest pointer is bounds-checked.** A guest at ring 0 shares an address
space with the kernel, so a pointer it passes is a pointer at anything at all.
`Space` records every range the loader handed out and `reachable` is asked
before any syscall reads or writes through a guest address. This cannot stop a
guest dereferencing a bad pointer itself, and nothing at CPL 0 can. It stops the
kernel doing it on the guest's behalf.

Two calls needed that and neither had it. `write` built a slice straight from
`rsi`, and `arch_prctl(ARCH_GET_FS)` wrote eight bytes wherever `rsi` pointed.
`sys_write` also swallowed bytes and reported success: `for chunk in
core::str::from_utf8(bytes)` iterates a `Result`, so the body ran zero times on
the error arm and a guest writing Latin-1 printed nothing while getting the full
length back.

`tools/mkelf.py` builds five fixtures by hand rather than by compiling, because
a negative has to differ from the positive in exactly one field and no compiler
will emit a dynamically linked binary otherwise identical to a static one.

## Page rights are enforced, and they were not before

Two bits decided whether a permission meant anything and neither was on.

**`CR0.WP`.** Without it a write from ring 0 ignores the R/W bit entirely, and
every instruction here runs at ring 0. A page marked read-only without it is a
page marked read-only in a comment.

**`EFER.NXE`**, gated on `CPUID.80000001H:EDX[20]` for the reason `dev::power`
gates its MSRs. Boot prints `page rights  wp=1  nx=1`.

Neither changed anything the day it landed, which is the point: everything was
mapped writable and nothing had ever set bit 63.

The identity map is built from 2 MiB pages, which is why per-page rights were
not free. `split_large` replaces one with 512 entries over the same bytes with
the same flags, cacheability included, so uncached device memory survives.

**One of the ten paging claims faults on purpose.** It makes a heap page
read-only, writes to it inside `cpu::recover::guard`, and requires the fault to
arrive and the write not to land. Enforcement is the one property that cannot be
asserted by reading the tables back.

So `mprotect` stopped being a lie. It was unimplemented on purpose, because
answering 0 would have claimed an enforcement that did not exist while refusing
stops any real allocator. `PROT_NONE` clears the present bit now, so a guard
page genuinely guards.

## The self-modification loop got much harder to fool

A loop that may rewrite its own bar at any moment converges on an evaluator that
says yes to everything, and the ledger recording that would be true.

**Red Queen epochs** freeze the criterion inside an epoch of five trials, one
per other axis. `is_boundary` is a pure function of ledger length, so where the
loop stands is re-derivable from the record.

**The judge may move its own bar**, within `[0.5, 12.0]`, off a declared
six-point grid. Coarse deliberately: a continuum lets the bar creep down by a
hundredth a night, every step honestly certified, and the sum of them is exactly
the drift this axis exists to catch. `judge_verdict` asks three questions of a
cross-evaluation matrix. Sane, moves, and honest, where honest means a held-out
anchor neither bar can see agrees with the direction. All seven outcomes are
asserted at boot with no model and no NVMe.

The price is the test-slice budget, and charging it there is the point. Every
other axis reads the anchor after a variant has already won, as confirmation.
Here the anchor is the evidence, so three reads and a criterion change cannot be
grounded at all.

**MAP-Elites**, twelve cells over four rank bands and three repair behaviours,
so a rank-4 adapter that repairs a different set of decisions than the rank-32
one survives on its own terms. **Bayesian surprise** replaces round-robin: an
axis that always says yes and one that always says no are equally predictable,
and the one near 50% is where the information is.

`godel storm` is the whole apparatus in one command:

    3 trained, 0 descendant(s) of the incumbent, 3 chimera(s) bred
    archive: 2 cell(s) lit; best validation 58%
    tribunal: the best was rejected -- net repair below the floor

Rejected, and nothing quietly kept.

The ledger could not previously say which axis a line came from, because five
judges wrote lines in the same shape. Certificates carry an `axis=` now, and the
`cell=` a variant lands in, derived at render time from columns already there.

## Tooling

`scripts/run.ps1` had three faults that between them meant it could not boot at
all: it projected the 604 MB deploy `esp/`, used a `fat:32:` volume OVMF cannot
read, and never reset NVRAM. Each presented as "the kernel did not boot".

`tools/drive.py` can record while a command is still running, and stopped
trusting the monitor prompt for synchronisation. It now takes `-accel whpx -cpu
max` as a matter of routine, which is roughly 160x faster than TCG on this
workload.

## Still missing, stated plainly

Sector specials, teleporters and crushers in DOOM. Seven of nine weapons fire
nothing, because a projectile needs momentum that only arrived at the end of
this cycle. No monster throws anything for the same reason.

Wireless. The SMP audit, so every task is still pinned to core 0 on purpose. No
GPU driver. No hardware entropy. A fatal fault on the laptop still prints one
line, because painting from inside an interrupt gate raises a #GP here and the
laptop has no UART.

## Verification

    35 passed, 0 failed

Every `diag` suite, on one boot, under `-accel whpx -cpu max`. Two of them are
new this release: `linux` at 60 claims and `paging` at 10. `doom` is 103.

`cargo build --release --locked` is clean, which is worth stating because the
lock file is what broke CI on 1.3.1.

181 files, 114,973 lines. `src/doom/` is 13,296 of them and `src/linux/` is
1,557.
