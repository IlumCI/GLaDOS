//! Running a program this kernel did not compile.
//!
//! Everything else in this tree was written here and signed by the key
//! compiled in. A Linux binary is neither, and that single fact sets the shape
//! of the whole module: nothing read out of the file is trusted, every offset
//! is checked against the file's own length, and a refusal names the field it
//! refused.
//!
//! ### What stage 0 is, and what it deliberately is not
//!
//! It is a **measuring instrument.** The expensive unknown in porting the
//! Linux ABI is not the loader -- that is a weekend -- it is that Linux has no
//! specification you can test against, so a subtly wrong `mmap` flag surfaces
//! as a crash in a different subsystem an hour later. Before committing to a
//! privilege model, an isolation mechanism or a syscall subset, the useful
//! thing to own is a *trace*: which calls does a real binary actually make, in
//! what order, with what arguments.
//!
//! So this loads a static binary at ring 0 with no isolation whatsoever, traps
//! its syscalls, and writes down what it asked for. A guest that faults takes
//! the kernel with it, and that is the intended behaviour rather than a gap:
//! at this stage a fault is the measurement. `gvisor` needed 237 of Linux's
//! ~350 calls to run containers, and the point of stage 0 is to find out which
//! ones this machine's first target actually reaches.
//!
//! The isolation question -- ring 3 in one address space, `PKS` domains,
//! `SFI`, or `VT-x` with the guest still at CPL 0 -- is real and is answered
//! later, from data this produces rather than from first principles.

pub mod elf;
pub mod load;
pub mod syscall;

/// What `diag linux` asks of everything here.
pub fn checks() -> alloc::vec::Vec<(&'static str, bool)> {
    let mut out = elf::checks();
    out.extend(syscall::checks());
    out.extend(load::checks());
    out
}
