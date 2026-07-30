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
| M0 | Toolchain, QEMU+OVMF loop, ESP staging | done (QEMU install pending) |
| M1 | UEFI entry, GOP framebuffer, ExitBootServices, own PML4 | done |
| M2 | Framebuffer console, GDT+TSS, IDT with fault reporting | done |
| M3 | Physical frame allocator, kernel heap | frame allocator done; heap pending |
| M4 | ACPI, LAPIC/IOAPIC, APIC timer, i8042 keyboard, tasking | not started |
| M5 | Tokenizer, parser, single-pass JIT, REPL | not started |
| M6 | NVMe driver, own filesystem | not started |
| M7 | 640×480 16-colour aesthetic layer, hypertext documents | not started |

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
