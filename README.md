# sanctum

A from-scratch, non-Unix, ring-0 operating system for one specific laptop:
an **MSI Thin GF63 12UC**, board **MS-16R8**.

Written in the spirit of TempleOS. Single address space, identity-mapped,
everything at CPL 0, no syscalls, no user/kernel split. Not a Linux
distribution, not a fork of anything — the only code here that this project did
not write is the Rust `core` library.

## State

| Milestone | What it does | Status |
|---|---|---|
| M0 | Toolchain, QEMU+OVMF loop, ESP staging | done |
| M1 | UEFI entry, GOP framebuffer, ExitBootServices, own PML4 | done |
| M2 | Framebuffer console, GDT+TSS, IDT with fault reporting | done |
| M3 | Physical frame allocator, kernel heap | done |
| M4 | ACPI, LAPIC/IOAPIC, APIC timer, i8042 keyboard, tasking, shell | done |
| M5 | Tokenizer, parser, single-pass JIT, REPL | not started |
| M6 | NVMe driver, own filesystem | not started |
| M7 | 640×480 16-colour aesthetic layer, hypertext documents | not started |

All of the above is verified running under QEMU, not merely compiling. The
shell accepts `help`, `mem`, `uptime`, `acpi`, `tasks`, `video`, `echo`,
`clear` and `fault`.

### Verified numbers

- Heap returns to **exactly 0 bytes used** after a 256-element `Vec` and a
  `String` are dropped, so `alloc` and `dealloc` are exact inverses.
- APIC timer measured at **63,067,500 Hz** against the PIT — QEMU's 1.009 GHz
  bus clock over the divide-by-16 — giving 50 ticks in half a second at 100 Hz.
- `cpus 4` counted correctly under `-smp 4`, exercising the MADT entry walk.
- Two tasks round-robin fairly at **819 and 820 resumes**, with the shell
  responsive while a never-yielding task burns 10.5 M iterations.

## Build and run

```powershell
.\scripts\run.ps1                # debug build, boot in QEMU
.\scripts\run.ps1 -Release
.\scripts\run.ps1 -Gdb           # pause for gdb on :1234
.\scripts\run.ps1 -TraceFaults   # log every exception -- finds triple faults
```

Bare metal, once an ESP exists on the USB SSD:

```powershell
.\scripts\deploy.ps1 -EspDrive S:
```

Then reboot and hold **F11**. `deploy.ps1` writes exactly one file and creates
no partitions. The internal Windows NVMe is never touched.

## Design

**No bootloader stage.** UEFI already delivers long mode, CPL 0 and an identity
map, so this UEFI application *is* the kernel. That removes ELF loading,
relocation, and a handoff ABI — three entire categories of bug, for free.

**Ring 0, single address space, identity-mapped.** No TSS `RSP0` juggling, no
SYSCALL/SYSRET setup, no per-process page tables, no TLB shootdown design. A
context switch is GPRs plus `RSP`, because there is no privilege change.

**Page 0 is deliberately not mapped.** The first 2 MiB is mapped at 4 KiB
granularity purely so that entry 0 can stay absent. Costs one page table, turns
every null dereference into a `#PF` the reporter can name.

**Faults report to the framebuffer.** The GF63 has no serial port. On real
hardware, pixels are the only channel by which a diagnostic reaches a human, so
the exception handlers draw. Under QEMU everything is also mirrored to COM1.

## Hardware notes

Surveyed from the running Windows install, not assumed.

| | |
|---|---|
| CPU | i7-12650H, Alder Lake, 6 P-core + 4 E-core. **BSP only** — SMP across asymmetric cores is its own project |
| Firmware | UEFI, **Secure Boot off**, so unsigned `BOOTX64.EFI` loads directly |
| Boot mode | No CSM assumed: no real mode, no INT 10h, no VGA text mode, no `0xB8000` |
| Display | Intel UHD via GOP. The RTX 3050 is mux-less and irrelevant — the iGPU owns the panel |
| Keyboard | **i8042** (`ACPI\MSI0007`) — ports `0x60`/`0x64`. Most 2022 laptops route the internal keyboard over USB-HID, which would mean an xHCI + HID stack before you could type one character. This one does not |
| Storage | Boots from a USB SSD. UEFI's own drivers load us *before* `ExitBootServices`, so no USB driver is needed to boot. Persistence in M6 targets the internal NVMe (~600 lines) rather than USB Mass Storage (several times that) |

## The boot disk is counterfeit

The USB SSD advertises **976.56 GB** and actually holds **14.67 GB**. It was
sold with a generic `VendorCo ProductCode` bridge, and the controller accepts
writes past the real end of flash, discards them, and returns garbage on read.

This was not obvious. It presented as a series of unrelated Windows failures —
`diskpart` refusing to format, then `format.com` reporting *"Invalid media or
Track 0 bad - disk unusable"* — because every failing write happened to be at
the 966 GB mark, where the original shrink-based layout put the ESP.
`scripts/probe-media.ps1` settled it by writing a self-identifying pattern and
reading it back; `scripts/find-capacity.ps1` binary-searched the boundary in
15 probes.

Consequences baked into `scripts/build-layout.ps1`:

- **MBR, not GPT.** GPT keeps a backup header and partition array in the last
  sectors of the *reported* size — 976 GB, inside flash that does not exist.
  The disk would be permanently missing half its partition table. MBR keeps
  everything in LBA 0.
- **ESP first, at 1 MB**, in the most thoroughly verified region.
- Everything above the layout is left unpartitioned so Windows never offers
  the fictional capacity to anything.

Do not put anything you care about on this device. It is acceptable as a boot
target only because the entire payload is a 77 KB file reproducible from git.

## Two bugs worth remembering

**`extern "C"` on this target is Microsoft x64, not System V.**
`x86_64-unknown-uefi` is a Windows-ABI target. The context switch assembly read
its arguments from `rdi`/`rsi`, but the compiler was passing them in
`rcx`/`rdx`. The garbage in `rdi` became a stack pointer that happened to land
in the framebuffer aperture, so the task's saved frame was written into video
memory and `ret` popped a black pixel into `rip`. The context switch is now
pinned to `extern "sysv64"` explicitly rather than inheriting a target default.

**Do not take the max over every UEFI memory descriptor.** OVMF describes
`Reserved` space to `0x100_0000_0000` — a clean 1 TiB. Using that as a map
limit exceeds what one PDPT covers, so the identity map silently failed to
build and we fell back to firmware page tables, which map page 0 — which in
turn made the null-dereference self-test pass without faulting. A masked
failure disabling the test meant to catch it. `max_ram_address` now uses an
allowlist of genuinely-RAM types plus a hard clamp.

## Known gaps

- **Exception handlers do not dump general-purpose registers.** The
  `x86-interrupt` ABI hands us the hardware frame, but the compiler's prologue
  has already clobbered the GPRs. Capturing them needs a naked assembly stub per
  vector. `RIP` plus `CR2` diagnoses most early faults, so this is deliberately
  deferred rather than faked.
- **The framebuffer is mapped write-back, not write-combining.** Correct, just
  slower than it could be. Needs PAT setup.
- **No NX.** Every mapping is `PRESENT | WRITABLE`. Requires `EFER.NXE` plus
  splitting text from data.
- `BootServicesCode`/`Data` are not reclaimed. They are genuinely free after
  exit, but our current stack is probably sitting in one of them, so the early
  frame allocator restricts itself to `Conventional` memory.

## Layout

```
src/
  main.rs        UEFI entry and boot sequence
  uefi.rs        hand-written firmware bindings -- field order IS the ABI
  sync.rs        Racy<T>: single-core interior mutability. Grep target for SMP
  serial.rs      COM1, for QEMU only
  cpu/
    gdt.rs       GDT + TSS, IST stacks for #DF and #PF
    idt.rs       IDT and the fault reporter
    port.rs      port I/O
  gfx/
    mod.rs       linear framebuffer
    font.rs      8x8 bitmap font, drawn at 2x
    console.rs   scrolling console over a RAM shadow buffer
  mem/
    frame.rs     bump frame allocator over the UEFI memory map
    paging.rs    identity map construction
```
