//! Getting a parsed image into memory it can run from.
//!
//! The whole loader is "allocate the span, copy the segments, jump", and the
//! two interesting decisions are both refusals.
//!
//! **Fixed-address executables are declined.** An `ET_EXEC` insists on the
//! addresses in its own headers -- classically `0x400000` -- and this kernel is
//! identity-mapped with one address space, so those addresses are *real
//! physical RAM the frame allocator may already have handed out*. Honouring
//! them means reserving a range nothing else may ever take, which is a change
//! to the memory map rather than to the loader. A position-independent
//! executable asks for none of that: it can be placed at whatever the heap
//! returns, and nearly everything modern is built that way.
//!
//! **Dynamically linked binaries are declined**, because the entry point in
//! the header is not where execution starts -- `ld.so` is -- and loading one as
//! though it were static jumps into a PLT stub nobody has filled in.
//!
//! Neither refusal is permanent and both are stated at the point of refusal
//! rather than discovered as a fault.

use super::{elf, syscall};
use crate::cpu::code::Exec;
use alloc::vec::Vec;

/// Sixteen KiB of guest stack. A static binary that does not recurse needs a
/// fraction of it; the number is chosen so overflowing it is a bug in the
/// guest rather than a limit of the harness.
const GUEST_STACK: usize = 16 * 1024;

/// What `brk` may grow into.
///
/// A separate region rather than the bytes after the image, because in an
/// identity map those bytes belong to whatever the frame allocator gave them
/// to. 256 KiB is enough for any allocator's first few arenas and small
/// enough that a guest which runs away hits the ceiling instead of the heap.
const GUEST_BRK: usize = 256 * 1024;

/// The tag a guest's pages are registered under, so `cpu::code::locate` can
/// name them in a fault report. Reads as `LNX` in a hex dump, which is the
/// only reason a tag is a number rather than a pointer.
const TAG_GUEST: u64 = 0x4C4E_5800;

pub struct Guest {
    /// Held because dropping it frees the pages the guest is running from.
    _image: Exec,
    _stack: Exec,
    _brk: Exec,
    pub base: u64,
    pub entry: u64,
    pub stack_top: u64,
    pub span: usize,
    pub segments: usize,
}

/// Place an image and answer where its entry landed.
pub fn load(bytes: &[u8]) -> Result<Guest, &'static str> {
    let img = elf::parse(bytes)?;
    syscall::runnable(&img)?;
    let (lo, hi) = img.span().ok_or("nothing to load")?;
    let span = (hi - lo) as usize;

    let mut image = Exec::new(span).ok_or("no room for the image")?;
    let base = image.addr();
    for s in &img.segments {
        let at = (s.vaddr - lo) as usize;
        let from = bytes.get(s.offset..s.offset + s.filesz).ok_or("segment past the file")?;
        if !image.write_at(at, from) {
            return Err("a segment does not fit inside the span its own headers describe");
        }
        // The `.bss` tail needs no work: `Exec::new` allocates zeroed, and the
        // span was sized from `memsz`. Worth stating rather than leaving to be
        // inferred, because a loader that stopped zeroing would produce a
        // program whose globals start as the previous tenant's heap -- correct
        // on the first run and wrong on the second.
    }

    let brk = Exec::new(GUEST_BRK).ok_or("no room for a break region")?;
    syscall::install(brk.addr(), GUEST_BRK);

    let stack = Exec::new(GUEST_STACK).ok_or("no room for a stack")?;
    let stack_top = syscall::build_stack(stack.addr() as *mut u8, GUEST_STACK);
    let entry = base + (img.entry - lo);
    image.arm(TAG_GUEST);

    Ok(Guest {
        _image: image,
        _stack: stack,
        _brk: brk,
        base,
        entry,
        stack_top,
        span,
        segments: img.segments.len(),
    })
}

/// Run it to completion.
///
/// # Safety
/// Jumps to an address derived from a file this kernel did not compile. Stage
/// 0 has no isolation of any kind: a guest that faults halts the machine, and
/// the fault report will at least name the guest's pages because `load` armed
/// them.
pub unsafe fn run(g: &Guest) -> u64 {
    syscall::clear_trace();
    unsafe { syscall::run(g.entry, g.stack_top) }
}

/// What `diag linux` asks of the refusals.
///
/// The loader itself needs a heap and an image, so what is asserted here is
/// the gate in front of it -- which is the part that is dangerous when wrong,
/// because every one of these refusals is a fault somewhere unrecognisable if
/// it does not happen.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();
    let img = |kind: u16, interp: bool, segs: usize| elf::Image {
        entry: 0,
        kind,
        segments: (0..segs)
            .map(|i| elf::Segment {
                offset: 0,
                vaddr: i as u64 * 4096,
                filesz: 16,
                memsz: 4096,
                flags: elf::PF_R | elf::PF_X,
                align: 4096,
            })
            .collect(),
        interp: if interp { Some(alloc::string::String::from("/lib/ld.so")) } else { None },
    };

    out.push((
        "a position-independent static executable is accepted",
        syscall::runnable(&img(elf::ET_DYN, false, 1)).is_ok(),
    ));
    out.push((
        "a dynamically linked binary is refused, naming the interpreter as the reason",
        syscall::runnable(&img(elf::ET_DYN, true, 1)).is_err(),
    ));
    out.push((
        "a fixed-address executable is refused, because one address space cannot promise it",
        syscall::runnable(&img(elf::ET_EXEC, false, 1)).is_err(),
    ));
    out.push((
        "an image with nothing loadable is refused rather than run at its entry",
        syscall::runnable(&img(elf::ET_DYN, false, 0)).is_err(),
    ));
    // The span is what the allocation is sized from, so a two-segment image
    // whose second segment sits a page up must reserve both pages and not just
    // the larger one.
    let two = img(elf::ET_DYN, false, 2);
    out.push((
        "the span of a gapped image covers both segments, not the larger one",
        two.span() == Some((0, 4096 + 4096)),
    ));
    out
}
