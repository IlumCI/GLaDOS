//! Which Linux programs the model is allowed to name.
//!
//! The model's whole action surface is `sysbox::APPLETS`, and the decoding
//! grammar is built from it, so an applet that does not exist is *unreachable*
//! rather than refused afterwards. That property has to survive reaching a
//! guest, and an applet taking a path does not survive it: `constrain.rs`
//! makes invalid output unreachable, and "any path in the namespace" is not a
//! set anything can enumerate. `agent::run` already paid for that -- the model
//! had to spell `/ai/tools/learned-3f2a91c4.ai&xi` exactly, and a skill it
//! could not spell was a skill it could not use, until `skill_choices` made the
//! set a closed table. This is that table for guests.
//!
//! **The directory is outside the guest-writable jail, and that is the point
//! rather than tidiness.** `fs.rs` lets a guest write inside `/tmp` and
//! refuses everywhere else, so a table built by scanning `/tmp` would let a
//! program install the next program the model is able to invoke -- the model's
//! vocabulary writable by the things it runs, which is an escalation from
//! guest to kernel action surface with no gate in front of it. Nothing at ring
//! 3 can write `/linux/bin`, so only the operator puts a name in here.
//!
//! Scanned rather than declared, which is the opposite of
//! `load::INTERPRETERS`. An interpreter path is dictated by some binary's
//! `PT_INTERP` and the useful list is of paths to *check for*; a program is
//! whatever somebody installed, so the only honest source is the machine.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Where an installed guest program lives.
///
/// Not `/bin`: `sysbox::tree` is not a filesystem and the top level is the
/// machine's own vocabulary (`/ai`, `/lib`, `/tmp`), so a Linux-only directory
/// says which world it belongs to. `/lib` and `/lib64` already hold the
/// interpreters for the same reason -- a binary's own header names those paths
/// and nothing here chose them.
pub const DIR: &str = "/linux/bin";

/// Whether `name` is a name rather than a path.
///
/// Checked because the resolution below joins it onto `DIR`, and a name
/// carrying a separator is the one input that would reach outside the
/// directory the table is defined by. `resolve` refuses `..` outright, so this
/// is belt and braces -- but the braces are what keeps the grammar's promise:
/// the model is offered a closed set, and anything not in that set must be
/// refused by the *dispatch* as well, or the table is documentation rather
/// than a leash.
fn is_name(name: &str) -> bool {
    !name.is_empty()
        && !name.contains('/')
        && name != "."
        && name != ".."
        // And it may not contain the grammar's own terminator, which is what
        // keeps the closed set *decodable* rather than merely closed.
        // `constrain::Grammar::new` appends `TERMINATOR` to every alternative,
        // and that is the whole reason a short name cannot shadow a longer one
        // it prefixes: `finished()` is an exact match on what has been
        // produced, and the decode loop breaks the moment it hits, so without
        // the terminator `sh` would commit and `shuf` would be unreachable --
        // the pre-existing `snap`/`snaps` pair is safe for exactly this
        // reason. A name carrying a newline puts the terminator *inside* an
        // alternative and hands that property back. Nothing is likely to be
        // called this; the point is that the set offered to a grammar has to
        // be checked against the grammar's own delimiter rather than against
        // what a filename usually looks like.
        && !name.as_bytes().contains(&crate::ai::constrain::TERMINATOR)
        // Anything else unprintable is refused on the same argument one step
        // out: a name is rendered into a prompt and matched back out of a
        // decode, and a control character survives neither trip legibly.
        && !name.chars().any(|c| c.is_control())
}

/// Every program installed, by name, in the order the namespace lists them.
///
/// Directory entries are kept sorted, so this is deterministic -- which the
/// grammar needs: the model's choice is an index into this list and a set that
/// reordered between the render and the decode would run the wrong program.
///
/// **It does not promise a name will load.** Deciding that means reading each
/// blob's ELF header, and an open file here *is* its whole contents
/// (`fs.rs` says so and gives the reason), so validating a 2 MB busybox on
/// every prompt render costs a 2 MB copy per program per decode. Offering a
/// name that turns out not to be an ELF costs one refusal from `load`, which
/// already words it better than anything here could. Cheap and occasionally
/// wrong beats expensive and always right on a path walked per decode.
pub fn installed() -> Vec<String> {
    if !crate::sysbox::is_dir(DIR) {
        return Vec::new();
    }
    crate::sysbox::children(DIR)
        .into_iter()
        .filter(|n| is_name(n))
        .filter(|n| {
            // A directory under here is not a program. `blob_len` answers for
            // a blob and nothing else, so it settles both questions at once
            // without reading any bytes.
            let mut p = String::from(DIR);
            p.push('/');
            p.push_str(n);
            crate::sysbox::blob_len(&p).is_some_and(|l| l > 0)
        })
        .collect()
}

/// Where an installed program's bytes are, or `None` if it is not installed.
///
/// The one resolution, so the grammar and the dispatch cannot disagree about
/// what a name means. A name absent from `installed()` answers `None` here
/// too, which is what makes the table a gate rather than a hint.
pub fn path_of(name: &str) -> Option<String> {
    if !is_name(name) {
        return None;
    }
    let mut p = String::from(DIR);
    p.push('/');
    p.push_str(name);
    crate::sysbox::blob_len(&p).is_some_and(|l| l > 0).then_some(p)
}

pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();

    // The gate, stated as arithmetic rather than as a scan: a name that could
    // reach outside `DIR` is refused before anything is joined onto it. These
    // are the inputs that would turn "a closed set of programs" into "any path
    // the model can spell", so they are claims and not a comment.
    out.push(("a program name may not be a path", path_of("../../ai/about").is_none()));
    out.push(("nor climb with a bare ..", path_of("..").is_none()));
    out.push(("nor name a directory", path_of(".").is_none()));
    out.push(("nor be empty", path_of("").is_none()));
    out.push((
        "nor carry a separator at all",
        path_of("bin/sh").is_none() && path_of("/tmp/x").is_none(),
    ));

    // An uninstalled name answers nothing, which is the whole of the leash:
    // the dispatch asks this and refuses, so the closed set holds even if a
    // decode somehow produced a name outside it.
    out.push((
        "a name nobody installed resolves to nothing",
        path_of("nothing-is-installed-under-this-name").is_none(),
    ));

    // Every name offered resolves, and resolves under `DIR`. Vacuous on a
    // machine with no programs installed, and that is why the count is
    // reported beside it rather than asserted -- a suite that passed because
    // the table was empty would be the "claim whose subject does not exist"
    // failure this tree already records.
    let names = installed();
    out.push((
        "every offered program resolves under the program directory",
        names.iter().all(|n| match path_of(n) {
            Some(p) => p.starts_with(DIR) && crate::sysbox::blob_len(&p).is_some(),
            None => false,
        }),
    ));
    out.push((
        "the offered set is free of paths",
        names.iter().all(|n| is_name(n)),
    ));

    // The decodability of the set, which is a different question from its
    // closedness and is the one a reader will not think to ask. Every name
    // becomes a grammar alternative with `TERMINATOR` appended, and
    // `Cursor::finished` is an exact match that the decode loop breaks on --
    // so if one alternative were a prefix of another the longer program would
    // be unreachable however well it was installed, which is the "a skill it
    // could not spell was a skill it could not use" failure arriving on
    // programs. The terminator is what prevents it, so a name may not contain
    // one, and the pair that would expose it is asserted rather than trusted.
    out.push((
        "a name may not carry the grammar's terminator",
        !is_name("sh\nls") && path_of("sh\nls").is_none(),
    ));
    out.push((
        "nor any other control character",
        !is_name("busy\tbox") && !is_name("bell\u{7}"),
    ));
    out.push((
        "so no offered name can shadow a longer one it prefixes",
        names.iter().all(|a| {
            names.iter().all(|b| {
                if a == b {
                    return true;
                }
                // What the grammar actually compares: the terminated forms.
                let (mut ta, mut tb) = (a.clone(), b.clone());
                ta.push(crate::ai::constrain::TERMINATOR as char);
                tb.push(crate::ai::constrain::TERMINATOR as char);
                !tb.starts_with(&ta)
            })
        }),
    ));

    // The directory is outside what a guest may write. This is the property
    // the module exists for, and it is checked against `fs.rs`'s own jail
    // rather than restated: if that jail ever widens, this fails here instead
    // of becoming an escalation nobody noticed.
    out.push((
        "a guest cannot write the program directory",
        !super::fs::writable(DIR) && !super::fs::writable(&(DIR.to_string() + "/sh")),
    ));

    out
}
