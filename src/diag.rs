//! The self-tests, as something you can run and see the result of.
//!
//! Every one of these already ran at boot and printed `ok` or `FAIL` into a
//! log that twenty more sections then scrolled away. That is the right shape
//! for a boot -- and the wrong shape for the question an operator actually
//! asks, which is "is this machine still correct *now*", usually after
//! changing something.
//!
//! So the suites get a registry, a remembered verdict, and a window. The
//! detail still goes to the terminal, because the detail is a log and a log
//! is text; what the window holds is the part a log is bad at -- which suites
//! exist, which have been run, and which of them is currently unhappy.
//!
//! **The verdict is remembered, never inferred.** A suite that has not been
//! run reads as unknown rather than as passing. That distinction is the whole
//! value of the thing: a green board that is green because nobody looked is
//! worse than no board.

use core::sync::atomic::{AtomicU8, Ordering};

/// What a suite last answered.
#[derive(Clone, Copy, PartialEq)]
pub enum Verdict {
    /// Not run since boot. Not the same as passing.
    Unknown,
    Pass,
    Fail,
}

/// A checkable claim about the machine, and how to check it.
///
/// `run` returns the same bool the boot sequence prints, so there is one
/// implementation of every check rather than a boot copy and a window copy
/// that drift.
pub struct Suite {
    pub name: &'static str,
    pub about: &'static str,
    pub run: fn() -> bool,
}

pub const SUITES: &[Suite] = &[
    Suite {
        name: "crypto",
        about: "published vectors for every primitive",
        run: crate::crypto::selftest,
    },
    Suite {
        name: "rng",
        about: "the generator, and that it refuses when starved",
        run: crate::rng::selftest,
    },
    Suite {
        name: "json",
        about: "the parser, against its own edge cases",
        run: crate::json::selftest,
    },
    Suite {
        name: "aiksi",
        about: "the language, and the capability gate",
        run: crate::aiksi::selftest,
    },
    Suite {
        name: "sysbox",
        about: "Merkle addressing and the namespace",
        run: crate::sysbox::selftest,
    },
    Suite {
        name: "smp",
        about: "a split matvec equals a whole one",
        run: crate::smp::selftest,
    },
    Suite {
        name: "update",
        about: "the update signature verifier refuses what it should",
        run: crate::update::selftest,
    },
    Suite {
        name: "model",
        about: "the forward pass and its geometry",
        run: crate::ai::selftest,
    },
    Suite {
        name: "wgate",
        about: "a write cannot leave the region that was claimed",
        run: crate::dev::nvme::gate_selftest,
    },
    Suite {
        name: "skill",
        about: "each judge can actually veto something",
        run: crate::ai::skill::selftest,
    },
    Suite {
        name: "desk",
        about: "one painter at a time, which no screenshot would show",
        run: crate::gfx::desk::selftest,
    },
    Suite {
        name: "recover",
        about: "a fault inside a program does not stop the machine",
        run: crate::cpu::recover::selftest,
    },
    Suite {
        name: "census",
        about: "which task the memory went to",
        run: crate::mem::census::selftest,
    },
    Suite {
        name: "migrate",
        about: "a task carried onto another core and back",
        run: crate::task::migration_selftest,
    },
    Suite {
        name: "mt",
        about: "a real lock, and a heap several cores allocate from at once",
        run: crate::smp::selftest_mt,
    },
    Suite {
        name: "power",
        about: "the gate in front of every model-specific register",
        run: crate::dev::power::selftest,
    },
    Suite {
        name: "fmt",
        about: "what a file is, and reading it as that",
        run: crate::fmt::selftest,
    },
    Suite {
        name: "differ",
        about: "two ways of running one program, required to agree exactly",
        run: crate::aiksi::differ::selftest,
    },
    Suite {
        name: "code",
        about: "running machine code from the heap, and naming a fault in it",
        run: crate::cpu::code::selftest,
    },
    Suite {
        name: "text",
        about: "the glyph table, and decoding UTF-8 one byte at a time",
        run: crate::gfx::text_selftest,
    },
];

/// One slot per suite. Indexed by position in `SUITES`, which is a constant,
/// so the table cannot get out of step with the list.
static RESULTS: [AtomicU8; 20] = [
    AtomicU8::new(0),
    AtomicU8::new(0),
    AtomicU8::new(0),
    AtomicU8::new(0),
    AtomicU8::new(0),
    AtomicU8::new(0),
    AtomicU8::new(0),
    AtomicU8::new(0),
    AtomicU8::new(0),
    AtomicU8::new(0),
    AtomicU8::new(0),
    AtomicU8::new(0),
    AtomicU8::new(0),
    AtomicU8::new(0),
    AtomicU8::new(0),
    AtomicU8::new(0),
    AtomicU8::new(0),
    AtomicU8::new(0),
    AtomicU8::new(0),
    AtomicU8::new(0),
];

/// Checked here rather than trusted: a suite added to `SUITES` without a slot
/// would silently never record a verdict.
const _: () = assert!(SUITES.len() == 20);

pub fn verdict(i: usize) -> Verdict {
    match RESULTS.get(i).map(|r| r.load(Ordering::Relaxed)) {
        Some(1) => Verdict::Pass,
        Some(2) => Verdict::Fail,
        _ => Verdict::Unknown,
    }
}

/// Run one suite and record what it said.
pub fn run_one(i: usize) -> Option<bool> {
    let s = SUITES.get(i)?;
    let ok = (s.run)();
    RESULTS[i].store(if ok { 1 } else { 2 }, Ordering::Relaxed);
    Some(ok)
}

pub fn find(name: &str) -> Option<usize> {
    SUITES.iter().position(|s| s.name == name)
}

/// Pass, fail, and not-yet-run counts.
pub fn tally() -> (usize, usize, usize) {
    let (mut p, mut f, mut u) = (0, 0, 0);
    for i in 0..SUITES.len() {
        match verdict(i) {
            Verdict::Pass => p += 1,
            Verdict::Fail => f += 1,
            Verdict::Unknown => u += 1,
        }
    }
    (p, f, u)
}
