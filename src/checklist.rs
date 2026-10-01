//! The bare-metal bring-up list, which records its own results.
//!
//! **`design/gf63.md` is this in prose and it has one flaw: it cannot say what
//! already happened.** Testing on the GF63 means rebooting into it, which takes
//! the editor, the browser and every note away -- so the list has to be *on the
//! machine*, and it has to answer rather than only ask. One command prints what
//! to do and what has been done, which is the difference between a plan and a
//! transcript.
//!
//! **Every status is derived, never recorded.** A row asks the kernel a question
//! it can already answer -- has anything faulted, did every core join, has a
//! share been accepted -- so nothing is written, nothing can go stale against the
//! machine it describes, and a row cannot claim a pass for work that was undone
//! afterwards. The one exception is the radio, which has to be read before it can
//! be reported, and `dev::iwx` caches that read for exactly this.
//!
//! Sized for one screenshot on purpose. On the GF63 there is no serial port and
//! the framebuffer is the whole diagnostic, so a list that scrolled would be a
//! list whose first half nobody can photograph.

use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED, WHITE, YELLOW};
use crate::kprintln;
use alloc::string::String;

pub enum Status {
    /// Done, with the figure that says so.
    Ok(String),
    /// Reached and wrong. The most valuable row on the page.
    Failed(String),
    /// Not attempted. Distinguished from a failure, because a plan that read
    /// "not run" as "broken" would make a fresh machine look ruined -- the
    /// mistake `diag`'s own bare listing makes when its tally reads 0 passed.
    Todo,
    /// Cannot be answered here, and says why rather than failing. A hypervisor
    /// hides the hybrid split and models no radio, so under QEMU those rows are
    /// unanswerable rather than bad.
    NotHere(&'static str),
}

pub struct Item {
    pub what: &'static str,
    /// What the operator types. `(automatic)` where the kernel answers by itself.
    pub how: &'static str,
    pub probe: fn() -> Status,
}

fn ok(s: alloc::string::String) -> Status {
    Status::Ok(s)
}

// --- the probes --------------------------------------------------------------

/// Is the miner pointed at an address that could ever be paid?
///
/// **First of the mining rows, because it is the only one whose failure costs
/// money rather than time.** Everything below it answers "does the machine
/// hash"; this answers "does the hashing go anywhere", and the two are
/// independent -- a machine can pass every other row on this page and be mining
/// to a string with one character wrong in it.
///
/// **Asked of the live config, not of `MINER.TXT`.** The file cannot be re-read
/// here at all: `uefi::read_file` wants boot services and they are gone by the
/// time there is a shell. That turns out to be the better subject anyway --
/// `client::CONFIG` is the name the miner is actually mining under, so this row
/// covers one typed at the shell as well as one that arrived from the volume,
/// and it cannot describe a file the running miner is not using.
///
/// Derived, and through `addr::judge` rather than a second opinion about it, so
/// this row and the line `boot::apply` printed cannot disagree.
fn payout_checks() -> Status {
    use crate::mine::addr::{judge, Payout};
    let user = match crate::mine::client::CONFIG.lock_irq().as_ref() {
        Some(c) => c.user.clone(),
        // Not a failure: a desktop image configures no miner and is not supposed
        // to, so the row says there is nothing pointed anywhere.
        None => return Status::NotHere("no pool configured, so nothing is being paid"),
    };
    match judge(&user) {
        Payout::Checked(k) => ok(alloc::format!("{} checksum holds", k.as_str())),
        // `NotHere` and not a pass: the address is well formed and this machine
        // genuinely cannot check where it pays, which is the same category as a
        // hypervisor hiding the hybrid split. The one row on the page a person
        // has to finish with their own eyes.
        Payout::Unchecked(_) => Status::NotHere("no checksum in it -- compare it against your wallet"),
        Payout::Broken(k) => Status::Failed(alloc::format!(
            "{} checksum FAILS -- this would pay nobody",
            k.as_str()
        )),
        Payout::Name => ok(String::from("a worker name; the pool's roster decides")),
    }
}


fn booted() -> Status {
    ok(alloc::format!("{}", crate::VERSION))
}

/// Nothing faulted during boot. The row a kernel panic would answer.
fn no_fault() -> Status {
    let broke = crate::boot_report::count();
    let still = crate::boot_report::outstanding();
    if broke == 0 {
        ok(String::from("boot report empty"))
    } else if still == 0 {
        Status::Failed(alloc::format!("{} broke, all repaired", broke))
    } else {
        Status::Failed(alloc::format!("{} broke, {} still broken", broke, still))
    }
}

fn nothing_vital() -> Status {
    if crate::boot_report::any_vital() {
        Status::Failed(String::from("a vital subsystem is down"))
    } else {
        ok(String::from("none vital"))
    }
}

/// Faults the machine *recovered* from, which a passing `diag all` hides.
///
/// `diag paging` faults on purpose, so a non-zero count is not itself wrong --
/// what matters is whether it is the number those deliberate ones account for.
/// Reported rather than judged, for that reason.
fn faults_taken() -> Status {
    let n = crate::cpu::recover::caught();
    if n == 0 {
        ok(String::from("none caught"))
    } else {
        ok(alloc::format!("{} caught and recovered", n))
    }
}

fn no_repair() -> Status {
    let mut n = 0;
    let mut first = "";
    for (sub, act) in crate::repair::in_force() {
        if n == 0 {
            first = act;
            let _ = sub;
        }
        n += 1;
    }
    if n == 0 {
        ok(String::from("none applied"))
    } else {
        // Not a failure: a repair that holds a subsystem up is the machine
        // working. It is a row because an operator is owed the fact.
        Status::Failed(alloc::format!("{} applied, first '{}'", n, first))
    }
}

fn suites() -> Status {
    let (pass, fail, notrun) = crate::diag::tally();
    if pass == 0 && fail == 0 {
        Status::Todo
    } else if fail > 0 {
        Status::Failed(alloc::format!("{} failed of {}", fail, pass + fail))
    } else if notrun > 0 {
        Status::Failed(alloc::format!("{} passed, {} not run", pass, notrun))
    } else {
        ok(alloc::format!("{} of {}", pass, pass))
    }
}

fn cores() -> Status {
    let n = crate::smp::online();
    let joined = crate::task::joined_mask().count_ones() as usize;
    if joined == n {
        ok(alloc::format!("{} of {} scheduling", joined, n))
    } else {
        Status::Failed(alloc::format!("{} of {} joined", joined, n))
    }
}

fn hybrid() -> Status {
    match crate::smp::performance_cores() {
        None => Status::NotHere("no hypervisor reports leaf 0x1A, correctly"),
        Some(m) => {
            let p = m.count_ones() as usize;
            ok(alloc::format!("{} fast, {} slow", p, crate::smp::online().saturating_sub(p)))
        }
    }
}

fn radio_present() -> Status {
    match crate::dev::iwx::seen() {
        None => Status::Todo,
        // **Not a failure under emulation**, which this had as one. No hypervisor
        // models an Intel wireless part, so "none on the bus" is the only answer
        // QEMU can give and reading it as broken makes every headless run of this
        // page show two failures that mean nothing. The module's own rule is that
        // a row a hypervisor cannot answer is `n/a` with the reason, the way the
        // hybrid split already is -- and on the GF63, where there is no
        // hypervisor, the same absence is a genuine failure and still reads as
        // one.
        Some(0) if crate::dev::power::virtualised() => {
            Status::NotHere("no hypervisor models an Intel wireless part")
        }
        Some(0) => Status::Failed(String::from("no Intel wireless function on the bus")),
        Some(n) => ok(alloc::format!("{} function(s)", n)),
    }
}

fn radio_answers() -> Status {
    match crate::dev::iwx::last_rev() {
        None => Status::Todo,
        Some(Err(r)) => Status::Failed(String::from(r.why())),
        Some(Ok(rev)) => ok(alloc::format!(
            "{} step {}, family {}",
            rev.mac.name(),
            rev.step,
            rev.mac.family().map(|f| f.name()).unwrap_or("unknown")
        )),
    }
}

fn radio_up() -> Status {
    match crate::dev::iwx::last_power_up() {
        None => Status::Todo,
        Some(Err(f)) => Status::Failed(f.why()),
        Some(Ok(())) => ok(String::from("semaphore, reset and clock")),
    }
}

/// Can the firmware file on this machine be turned into what the part boots from?
///
/// **The one radio row that is answerable without the radio**, which is why it
/// sits between the power-up and anything that needs the part to reply: it is
/// parsing a file and laying out memory, so a failure here is a firmware file
/// that is absent or wrong and nothing to do with the hardware. Establishing it
/// before the trip means a bare-metal failure afterwards is about the part.
///
/// Answered from a recorded build rather than by building one, because building
/// allocates about a meganite and a half of DMA regions and a status row must not
/// do that every time the page is printed.
fn firmware_ready() -> Status {
    match crate::dev::iwx::last_ctxt() {
        None => Status::Todo,
        Some(Err(e)) => Status::Failed(String::from(e)),
        Some(Ok((secs, bytes))) => ok(alloc::format!("{} section(s), {} B staged", secs, bytes)),
    }
}

/// Did the part come up and say so?
///
/// **The row the bare-metal trip exists for on the radio side.** Everything above
/// it is host-side arithmetic that passes under emulation; this one cannot pass
/// anywhere but on the laptop, because it needs a part that fetches a megabyte and
/// a half of its own microcode and then answers.
///
/// A failure here is the most valuable line on the page: it names which of the
/// four things went wrong -- the power-up, the build, the kick, or the wait -- and
/// the wait distinguishes "never raised its bit", "raised an error bit", "said
/// nothing after saying it was alive" and "said it was not ok".
fn radio_alive() -> Status {
    match crate::dev::iwx::last_alive() {
        None => Status::Todo,
        Some(Err(why)) => Status::Failed(why),
        Some(Ok(a)) => ok(a.say()),
    }
}

/// Did the part answer a question?
///
/// **The first row that needs the part to talk back rather than merely start.**
/// Everything above it is the host handing bytes to firmware; this is a command in
/// a ring, a doorbell, and an answer through the receive path -- so it is the row
/// that says the two directions both work.
///
/// Shows the address, because that is the one figure from this whole sequence a
/// person can check against a label on the machine.
fn radio_nvm() -> Status {
    match crate::dev::iwx::last_nvm() {
        None => Status::Todo,
        Some(Err(why)) => Status::Failed(why),
        Some(Ok(n)) => ok(alloc::format!(
            "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}, {} chan",
            n.mac[0], n.mac[1], n.mac[2], n.mac[3], n.mac[4], n.mac[5], n.valid_channels
        )),
    }
}

/// The rate, which is the figure the trip exists to bring back.
///
/// Reads the foreground bench first and the slice counter second, because they
/// are different measurements and the bench is the one with a rate in it: a
/// slice count says mining happened, and `H/s` says how fast.
fn hashes() -> Status {
    if let Some((hs, kib)) = crate::mine::client::last_bench() {
        return ok(alloc::format!("{} H/s, {} KiB set", hs, kib));
    }
    let h = crate::mine::client::HASHES.load(core::sync::atomic::Ordering::Relaxed);
    if h == 0 {
        Status::Todo
    } else {
        ok(alloc::format!("{} hash(es) by slices", h))
    }
}

fn share_found() -> Status {
    let f = crate::mine::client::FOUND.load(core::sync::atomic::Ordering::Relaxed);
    if f == 0 {
        Status::Todo
    } else {
        ok(alloc::format!("{} found", f))
    }
}

fn share_accepted() -> Status {
    let a = crate::mine::client::ACCEPTED.load(core::sync::atomic::Ordering::Relaxed);
    if a == 0 {
        Status::Todo
    } else {
        ok(alloc::format!("{} accepted", a))
    }
}

/// The list, in the order it should be worked through.
///
/// Boot health first, because a machine that faulted on the way up makes every
/// figure after it suspect -- and because those rows need nothing typed, so a
/// bare `checklist` on a fresh boot already answers half the page.
pub const ITEMS: &[Item] = &[
    Item { what: "the machine boots and answers", how: "(automatic)", probe: booted },
    Item { what: "no subsystem faulted on the way up", how: "(automatic)", probe: no_fault },
    Item { what: "nothing vital is broken", how: "(automatic)", probe: nothing_vital },
    Item { what: "faults taken and recovered", how: "(automatic)", probe: faults_taken },
    Item { what: "no repair is propping it up", how: "repair", probe: no_repair },
    Item { what: "every suite passes", how: "diag all", probe: suites },
    Item { what: "every core joined the scheduler", how: "smp", probe: cores },
    Item { what: "the hybrid core split was read", how: "smp", probe: hybrid },
    Item { what: "the radio is on the bus", how: "iwx", probe: radio_present },
    Item { what: "its revision reads", how: "iwx probe", probe: radio_answers },
    Item { what: "it resets and its clock starts", how: "iwx up", probe: radio_up },
    Item { what: "its firmware builds a boot descriptor", how: "iwx ctxt <fw>", probe: firmware_ready },
    Item { what: "the firmware boots and says it is alive", how: "iwx boot <fw>", probe: radio_alive },
    Item { what: "it answers with its address and bands", how: "(same command)", probe: radio_nvm },
    Item { what: "the payout address checks out", how: "(automatic)", probe: payout_checks },
    Item { what: "yescrypt hashes", how: "mine algo yescrypt / mine bench 8000", probe: hashes },
    Item { what: "a share is found", how: "mine coin 0 t yescrypt", probe: share_found },
    Item { what: "a pool accepts one", how: "mine pool <host> / mine on", probe: share_accepted },
];

/// Print the list with every status answered now.
pub fn show() {
    let mut done = 0usize;
    let mut bad = 0usize;
    let mut todo = 0usize;
    let mut na = 0usize;
    let mut rows: alloc::vec::Vec<(usize, &Item, Status)> = alloc::vec::Vec::new();
    for (i, it) in ITEMS.iter().enumerate() {
        let s = (it.probe)();
        match s {
            Status::Ok(_) => done += 1,
            Status::Failed(_) => bad += 1,
            Status::Todo => todo += 1,
            Status::NotHere(_) => na += 1,
        }
        rows.push((i + 1, it, s));
    }

    console::set_color(YELLOW);
    kprintln!(
        "[checklist] bare-metal bring-up, {} -- {} ok, {} FAILED, {} to run, {} n/a",
        crate::VERSION, done, bad, todo, na
    );
    console::set_color(LTGRAY);
    kprintln!("  #  what                                type this                      result");
    for (n, it, s) in rows {
        let (colour, tag, detail) = match &s {
            Status::Ok(d) => (LTGREEN, "ok  ", d.as_str()),
            Status::Failed(d) => (LTRED, "FAIL", d.as_str()),
            Status::Todo => (WHITE, "--  ", "not run"),
            Status::NotHere(w) => (LTGRAY, "n/a ", *w),
        };
        console::set_color(colour);
        kprintln!(
            "  {:<2} {:<35} {:<29} {} {}",
            n,
            crate::gfx::theme::head_chars(it.what, 35),
            crate::gfx::theme::head_chars(it.how, 29),
            tag,
            crate::gfx::theme::head_chars(detail, 44)
        );
        // **A failure's detail is never truncated, and the column is why this is
        // a second line rather than a wider one.** The page is sized for one
        // screenshot, so widening the result column for the rare long string
        // costs every row; cutting the string costs only the rows that failed,
        // which are the ones whose text is worth the most. Driven: the payout row
        // read "this would pay nob".
        //
        // Failures only. A long `n/a` reason is a fact about the emulator and
        // reads fine truncated; a long `FAIL` is the sentence somebody has to act
        // on.
        if matches!(s, Status::Failed(_)) && crate::gfx::theme::text_w_of(detail) > 44 {
            kprintln!("     {}", detail);
        }
    }
    console::set_color(LTGRAY);
    // **The panic surface, on the same page.** A fault that halted the machine
    // leaves nothing here to read -- the recovery for that is the boot report on
    // the next boot, and the ESP health flag behind it. What this line can say is
    // whether this boot took one at all.
    kprintln!(
        "  faults {} caught, {} subsystem(s) in the boot report, store {}",
        crate::cpu::recover::caught(),
        crate::boot_report::count(),
        if crate::store::mounted() { "mounted" } else { "absent" }
    );
    if bad > 0 {
        console::set_color(LTRED);
        kprintln!("  a FAILED row is the useful one: read it before running anything below it");
        console::set_color(LTGRAY);
    }
}
