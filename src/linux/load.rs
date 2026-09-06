//! Getting a parsed image into memory it can run from.
//!
//! The whole loader is "allocate the span, copy the segments, jump", and the
//! two interesting decisions are both refusals.
//!
//! **Fixed-address executables are placed where they insist, when the machine
//! can promise the range.** An `ET_EXEC` demands the addresses in its own
//! headers -- classically `0x400000` -- and this kernel is identity-mapped with
//! one address space, so those addresses are real physical RAM. That was
//! refused for a long time under a reason that was true of what the kernel
//! *knew* rather than of the machine: nothing here could answer "does anything
//! own four megabytes at four megabytes", so declining was the only honest
//! answer available.
//!
//! `mem::fixed` answers it now, from the firmware's map less what boot took,
//! and the refusal became a measurement. It matters more than it sounds:
//! nearly every prebuilt static binary in the world is non-PIE, busybox's own
//! included, so this was not a corner case but most of the software this
//! loader exists to run.
//!
//! The claim is exclusive and released at teardown. One at a time, which is
//! the same one-guest assumption the syscall stack and `SPACE` already rest
//! on -- and unlike them it is a real constraint rather than a simplification,
//! because two binaries both insisting on `0x400000` cannot coexist in one
//! address space however clever the loader is.
//!
//! **A dynamically linked binary gets its interpreter placed beside it.**
//! This was the second refusal, and the reason given was correct: the entry in
//! the header is not where execution starts, `ld.so` is, and loading one as
//! though it were static jumps into a PLT stub nobody has filled in. The
//! answer is not to implement linking, it is to load the linker -- `PT_INTERP`
//! names a path, the path is a file in the namespace, and a second image at a
//! second base is the whole of it.
//!
//! Three numbers have to be right and all three are silent when wrong.
//! `AT_ENTRY` is the *program's* entry, not the interpreter's, or `ld.so`
//! relocates everything correctly and jumps back into itself. `AT_BASE` is
//! where the interpreter landed, and it is the only way an unrelocated
//! `ET_DYN` can find its own `_DYNAMIC`. And the address jumped to is the
//! interpreter's, which is the one thing that is obviously wrong when wrong.
//!
//! What is still missing before a real `ld.so` gets anywhere: `mmap` with
//! `MAP_FIXED` and file-backed mappings, which is how it lays out everything
//! it loads afterwards. Refused today, and the next rung.

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

/// The largest image this will place.
///
/// A file is free to claim a segment at address 0 and another at 2^40, and the
/// span between them is what the allocation is sized from. `Exec::new` would
/// fail on it anyway, but failing with "no room for the image" describes the
/// machine when the truth is about the file.
const MAX_SPAN: usize = 64 * 1024 * 1024;

/// Where a guest's image came from, because the two are freed differently.
///
/// A PIE lives in an `Exec` taken from the heap and goes back when it drops. A
/// fixed image lives at an address the heap never owned, so what has to happen
/// is a `mem::fixed::release`. Keeping both in one enum means `Guest`'s drop
/// cannot forget which -- the alternative was two `Option`s and an invariant
/// that exactly one is set, which is the same thing written so it can be got
/// wrong.
enum Image {
    Placed(Exec),
    Fixed { at: u64, len: usize },
}

impl Drop for Image {
    fn drop(&mut self) {
        if let Image::Fixed { at, len } = *self {
            // Put the rights back before the range leaves this guest's hands,
            // for the reason `syscall::teardown` gives about mappings: a page
            // left user-accessible or read-only is a page the next tenant
            // inherits, and the symptom lands somewhere else entirely.
            crate::mem::paging::protect(at, len, crate::mem::paging::Perm::RWX);
            crate::mem::fixed::release(at);
        }
    }
}

pub struct Guest {
    /// Every range handed to this guest, which is what bounds-checks its
    /// pointers once it is running.
    regions: syscall::Regions,
    /// Held because dropping it frees the pages the guest is running from.
    _image: Image,
    /// The interpreter's pages, held for exactly the same reason.
    _interp: Option<Image>,
    _stack: Exec,
    _brk: Exec,
    pub base: u64,
    /// Where execution starts, which is the interpreter's entry when there is
    /// one and the program's otherwise.
    pub entry: u64,
    /// The interpreter, when there is one: what it was called, where it went,
    /// and where its own entry landed. Reported rather than kept private,
    /// because "which ld.so did it find" is the first question when a
    /// dynamically linked program does nothing.
    pub interp: Option<(alloc::string::String, u64, u64)>,
    pub stack_top: u64,
    pub span: usize,
    pub segments: usize,
    /// What it was invoked as, kept for `/proc/self/cmdline` and for
    /// `/proc/self/exe`, neither of which is answerable once the stack the
    /// kernel built has been handed over.
    pub argv: Vec<alloc::string::String>,
}

/// One image in memory: what it said about itself and where it went.
struct Placed {
    img: elf::Image,
    hold: Image,
    base: u64,
    lo: u64,
    span: usize,
}

/// Put one image in memory. Shared by the program and its interpreter, because
/// the two differ in nothing a loader cares about: both are ELF, both may be
/// `ET_DYN` or `ET_EXEC`, and both are copied segment by segment relative to
/// their own lowest address.
fn place(bytes: &[u8]) -> Result<Placed, &'static str> {
    let img = elf::parse(bytes)?;
    syscall::runnable(&img)?;
    let (lo, hi) = img.span().ok_or("nothing to load")?;
    let span = hi.checked_sub(lo).ok_or("the segments span a range that runs backwards")? as usize;
    if span > MAX_SPAN {
        return Err("the image claims a span larger than this will place");
    }

    // A PIE goes wherever the heap has room; a fixed image goes where its
    // headers insist or nowhere at all. `lo` is that address, and it is the
    // same number the segments are already written relative to, so the copy
    // loop below is identical for both.
    let (mut placed, base) = if img.relocatable() {
        let e = Exec::new(span).ok_or("no room for the image")?;
        let at = e.addr();
        (Some(e), at)
    } else {
        crate::mem::fixed::claim(lo, span)?;
        // Zeroed here because `Exec::new` zeroes and the `.bss` tail depends on
        // it. This range is whatever the last tenant left, and a program whose
        // globals start as somebody else's memory is one that works once.
        unsafe { core::ptr::write_bytes(lo as *mut u8, 0, span) };
        (None, lo)
    };
    for s in &img.segments {
        let at = s.vaddr.checked_sub(lo).ok_or("a segment sits below the image's own base")? as usize;
        let end = s.offset.checked_add(s.filesz).ok_or("a segment's file range overflows")?;
        let from = bytes.get(s.offset..end).ok_or("segment past the file")?;
        let fits = at.checked_add(from.len()).is_some_and(|e| e <= span);
        match placed.as_mut() {
            Some(e) => {
                if !e.write_at(at, from) {
                    return Err("a segment does not fit inside the span its own headers describe");
                }
            }
            None => {
                if !fits {
                    return Err("a segment does not fit inside the span its own headers describe");
                }
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        from.as_ptr(),
                        (base + at as u64) as *mut u8,
                        from.len(),
                    )
                };
            }
        }
        // The `.bss` tail needs no work: `Exec::new` allocates zeroed, and the
        // span was sized from `memsz`. Worth stating rather than leaving to be
        // inferred, because a loader that stopped zeroing would produce a
        // program whose globals start as the previous tenant's heap -- correct
        // on the first run and wrong on the second.
    }

    // The entry has to land inside what was actually placed. A file is free to
    // name one outside its own segments, and jumping there would leave the
    // fault reporter naming a range this loader never armed.
    let off = img.entry.checked_sub(lo).ok_or("the entry point sits below the image")?;
    if off as usize >= span {
        return Err("the entry point is outside every segment the file loads");
    }

    let hold = match placed {
        Some(mut e) => {
            e.arm(TAG_GUEST);
            Image::Placed(e)
        }
        None => Image::Fixed { at: base, len: span },
    };
    Ok(Placed { img, hold, base, lo, span })
}

/// The interpreter paths the world actually uses, and which libc each is.
///
/// **Not a list of what is supported.** Nothing in this loader is
/// libc-specific and it has never looked at these strings: `PT_INTERP` names a
/// path, the path is read out of the namespace, and whatever is there is
/// loaded. This is a list of what to *check for*, so an operator can be told
/// what is installed rather than guessing a path and finding out from a
/// refusal.
///
/// Both may be present at once and they do not interact, because a libc is
/// userspace and the choice was made by whoever built the binary. What cannot
/// happen is two of them in one process: two mallocs each certain it owns the
/// break, two thread-local layouts, two `errno`s. So "use whichever is faster"
/// is a per-program question and never a per-call one.
pub const INTERPRETERS: &[(&str, &str)] = &[
    ("/lib/ld-musl-x86_64.so.1", "musl"),
    ("/lib64/ld-linux-x86-64.so.2", "glibc"),
    // Some distributions put the same object here and symlink the other way
    // round, and a binary built on one of those names this path in its header.
    ("/lib/ld-linux-x86-64.so.2", "glibc, in /lib"),
];

/// Which of them this machine actually has.
pub fn installed() -> Vec<(&'static str, &'static str, bool)> {
    INTERPRETERS
        .iter()
        .map(|&(path, what)| (path, what, crate::sysbox::blob_len(path).is_some()))
        .collect()
}

/// What interpreter a file asks for, without loading anything.
///
/// Exists for the *refusal* rather than for the loading. "The interpreter this
/// binary names is not in the namespace" is true and useless without the name
/// in it, and the error type is a `&'static str` that cannot carry one.
pub fn wants(bytes: &[u8]) -> Option<alloc::string::String> {
    elf::parse(bytes).ok().and_then(|i| i.interp)
}

/// Place a program, its interpreter if it wants one, and build its stack.
pub fn load(bytes: &[u8], args: &[&str]) -> Result<Guest, &'static str> {
    let prog = place(bytes)?;
    let (img, base, lo, span) = (&prog.img, prog.base, prog.lo, prog.span);

    // The interpreter, if the program named one. Read through `sysbox` rather
    // than through `fs::resolve`, because there is no guest yet to have a
    // working directory: `PT_INTERP` is always absolute, and a relative one
    // would be a file resolved against nothing.
    let interp = match img.interp.as_deref() {
        None => None,
        Some(path) => {
            let bytes = crate::sysbox::read_blob(path)
                .ok_or("the interpreter this binary names is not in the namespace")?;
            let p = place(&bytes)?;
            // An interpreter that itself wants an interpreter is refused
            // rather than followed. Nothing real does it, the recursion has no
            // natural bound, and a loader that chased it would run out of
            // stack in ring 0, which is a triple fault and not a message.
            if p.img.dynamic() {
                return Err("the interpreter names an interpreter of its own");
            }
            Some((alloc::string::String::from(path), p))
        }
    };

    let brk = Exec::new(GUEST_BRK).ok_or("no room for a break region")?;
    let stack = Exec::new(GUEST_STACK).ok_or("no room for a stack")?;
    let prog_entry = base + (img.entry - lo);

    // A fixed image is not registered with `cpu::code`, which addresses heap
    // ranges by tag and offset. Saying so rather than leaving a fault report
    // to name the wrong thing: an rip inside one reports as being in neither
    // the kernel image nor generated code, which is exactly true.

    // `AT_PHDR` is a runtime address, so it is the segment that contains the
    // header table plus the offset into it. `base + phoff` is the same number
    // only when the first segment starts at file offset zero, which is true of
    // every fixture here and is not a property of the format -- and a header
    // pointer that is wrong is worse than one that is absent, because
    // `dl_iterate_phdr` walks it either way.
    let table = img.phnum.checked_mul(img.phentsize).and_then(|n| n.checked_add(img.phoff));
    let phdr = table.and_then(|end| {
        img.segments
            .iter()
            .find(|s| img.phoff >= s.offset && end <= s.offset + s.filesz)
            .map(|s| base + (s.vaddr - lo) + (img.phoff - s.offset) as u64)
    });
    let stack_top = syscall::build_stack(
        stack.addr() as *mut u8,
        GUEST_STACK,
        args,
        syscall::Aux {
            phdr: phdr.unwrap_or(0),
            phent: img.phentsize as u64,
            phnum: phdr.map_or(0, |_| img.phnum as u64),
            // The program's entry even when the interpreter is what runs. This
            // is the field `ld.so` jumps to when it has finished, so swapping
            // the two gives a linker that relocates everything and then
            // re-enters itself.
            entry: prog_entry,
            base: interp.as_ref().map_or(0, |(_, p)| p.base),
        },
    )
    .ok_or("the stack is too small for the arguments")?;

    // Execution starts at the interpreter when there is one. Everything above
    // this line treats the two images identically and this is the one place
    // that does not, which is why it is a single expression rather than
    // threaded through `place`.
    let entry = interp.as_ref().map_or(prog_entry, |(_, p)| p.base + (p.img.entry - p.lo));
    let segments = img.segments.len() + interp.as_ref().map_or(0, |(_, p)| p.img.segments.len());
    let (interp_named, interp_hold, interp_region) = match interp {
        None => (None, None, None),
        Some((path, p)) => (
            Some((path, p.base, p.base + (p.img.entry - p.lo))),
            Some(p.hold),
            Some(syscall::Region { at: p.base, len: p.span }),
        ),
    };

    Ok(Guest {
        argv: args.iter().map(|a| alloc::string::String::from(*a)).collect(),
        regions: syscall::Regions {
            image: syscall::Region { at: base, len: span },
            stack: syscall::Region { at: stack.addr(), len: GUEST_STACK },
            brk: syscall::Region { at: brk.addr(), len: GUEST_BRK },
            interp: interp_region,
        },
        _image: prog.hold,
        _interp: interp_hold,
        _stack: stack,
        _brk: brk,
        base,
        entry,
        interp: interp_named,
        stack_top,
        span,
        segments,
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
    // Open exactly the guest's own three regions to ring 3 and nothing else.
    // Every other page in the machine keeps a clear U bit, which is what makes
    // the guest unable to reach the kernel rather than merely discouraged from
    // trying.
    // The interpreter's pages are opened on exactly the same terms as the
    // program's. Forgetting it is not a subtle failure: `ld.so` takes a
    // protection violation on its first instruction, which reads as a bad
    // entry address rather than as a missing U bit.
    for r in [Some(g.regions.image), Some(g.regions.stack), Some(g.regions.brk), g.regions.interp]
        .into_iter()
        .flatten()
    {
        if !crate::mem::paging::protect(r.at, r.len, crate::mem::paging::Perm::USER_RWX) {
            return 0;
        }
    }
    // Installed here rather than in `load`, so a guest that was loaded and
    // never run leaves nothing naming memory its `Guest` has since freed.
    syscall::install(g.regions);
    // After `install`, which clears the space these names describe.
    let argv: Vec<&str> = g.argv.iter().map(|s| s.as_str()).collect();
    syscall::name_guest(
        &argv,
        argv.first().copied().unwrap_or(""),
        g.interp.as_ref().map(|(p, _, _)| p.as_str()),
    );
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
        phoff: 64,
        phentsize: 56,
        phnum: segs,
        interp: if interp { Some(alloc::string::String::from("/lib/ld.so")) } else { None },
    };

    out.push((
        "a position-independent static executable is accepted",
        syscall::runnable(&img(elf::ET_DYN, false, 1)).is_ok(),
    ));
    out.push((
        "a dynamically linked binary is no longer refused on sight either",
        syscall::runnable(&img(elf::ET_DYN, true, 1)).is_ok(),
    ));
    out.push((
        "and it is still recognised as wanting one, which is what decides the entry",
        img(elf::ET_DYN, true, 1).dynamic() && !img(elf::ET_DYN, false, 1).dynamic(),
    ));
    out.push((
        "a fixed-address executable is no longer refused on sight, since the map can answer",
        syscall::runnable(&img(elf::ET_EXEC, false, 1)).is_ok(),
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

    // Two properties of the interpreter table, both of which would be silent.
    // A duplicate path reports one file twice and reads as two libcs
    // installed; a relative one resolves against a working directory that does
    // not exist yet, since `PT_INTERP` is read before there is a guest.
    out.push((
        "no interpreter path is listed twice, so two rows cannot mean one file",
        INTERPRETERS.iter().enumerate().all(|(i, (p, _))| {
            INTERPRETERS.iter().skip(i + 1).all(|(q, _)| p != q)
        }),
    ));
    out.push((
        "and every one of them is absolute, there being nothing to resolve against",
        INTERPRETERS.iter().all(|(p, _)| p.starts_with('/')),
    ));
    out
}
