//! Repairs a machine may try on itself, and the judge that says whether one worked.
//!
//! The Windows Troubleshooter's bargain, which was a good one: a fixed set of
//! deterministic actions, one chosen, applied, and then **checked**. It never
//! ran during POST either -- it booted, looked at what was broken, fixed it,
//! and made the fix stick for next time.
//!
//! ### The judge is the check that failed
//!
//! This is the whole reason a repair is allowed to happen unattended. `godel`'s
//! rule is that nothing is adopted without a judge, and for a repair there is
//! an obvious one: apply it and **re-run the selftest that faulted**. Passing
//! is the verdict. That is why `boot_report::Failure` carries a `fn() -> bool`
//! rather than the machine merely remembering that something went wrong -- a
//! failure you cannot re-run is a failure you cannot repair.
//!
//! **And passing means running *and agreeing*.** `judge` asked only whether
//! the check finished, for as long as `section` took a `fn()` and threw every
//! subsystem's verdict away. An action that left a subsystem alive and
//! answering wrongly would have been adopted, marked as the repair that
//! worked, and written to the boot volume for every boot after -- the worst
//! outcome this table can produce, because from every other vantage point it
//! looks exactly like a fix.
//!
//! ### The table is an allowlist and starts small
//!
//! Two actions today. That is not a claim that the machine can fix anything;
//! it is the honest size of the set of knobs that exist. Each new one is a
//! deliberate decision to expose a switch, argued at the switch, and the list
//! being visible in one place is the point -- the same reason `eval::BUILTINS`
//! is an allowlist rather than a denylist.
//!
//! Most useful repairs turn something off or down. So did the Troubleshooter's.
//!
//! ### The model chooses the order, and the judge decides
//!
//! `author::choose` picks which repair to try, under a grammar built from the
//! rows offered for *this* subsystem, so the answer is an index into a list
//! this kernel built and anything else is unreachable rather than merely
//! unlikely. Nothing the model emits is executed, parsed as a command, or used
//! as an argument.
//!
//! **And it does not have to be re-derivable**, which is the one place this
//! departs from `godel` and is worth stating plainly because it looks like a
//! lapse. There a verdict is a certificate somebody may want to refute months
//! later, so the search has to be a function of the record rather than of a
//! coin. Here the verdict is a *live re-run of the check that failed*: a badly
//! chosen repair costs one apply-and-revert and is then refused by the same
//! judge that refuses everything else. So `choose` is left sampling at 0.7,
//! where temperature zero is a fixed point a small model can wedge against --
//! `author.rs` records what that cost.
//!
//! What the model buys is **order**, on a table where the first row is a guess
//! and the right one may be third. What it cannot buy is a wrong repair
//! surviving, because it never judges.
//!
//! The fixed rule is still there and still the fallback: no model, an engine
//! somebody else is holding, or three decodes that will not commit, and the
//! loop walks the table in order exactly as it did. That ordering was built
//! first on purpose -- the apply, judge and revert loop was proven somewhere a
//! decode could not be blamed for before a decode was allowed near it.
//!
//! ### What the chooser actually does, measured
//!
//! Six boots under QEMU with a fault injected into `power` that only
//! `skip-hwp` fixes, three with the table in its own order and three with the
//! offered list reversed:
//!
//!     [retry, skip-hwp]     retry     retry     retry
//!     [skip-hwp, retry]     skip-hwp  retry     retry
//!
//! **It picked `retry` five times in six, wherever `retry` sat.** Reversing the
//! list changed the answer once, which is what rules out the obvious
//! explanation: this is not a model taking whatever is listed first, it is a
//! model preferring a *name*. And the name it prefers is the one that cannot
//! fix this fault, so the pick was wrong five times out of six.
//!
//! The mechanism is worth knowing before anybody adds a row. `retry` is one
//! common English token; `skip-hwp` is several uncommon pieces. Under a
//! constrained grammar the cheapest first-token path wins, so **an action's
//! name carries probability mass that has nothing to do with what the action
//! does**. Naming a row well is not cosmetic here.
//!
//! On this evidence the decode buys nothing on this table: table order also
//! tries `retry` first, so six boots of choosing produced what the fixed rule
//! produces, for the price of a prefill. What it did *not* do is any harm --
//! the machine was repaired on all six, because the judge caught the bad pick
//! and the loop moved on. That is this header's argument arriving as a
//! measurement rather than as a claim.
//!
//! It is left on all the same, and the reason is a caveat rather than
//! optimism: this was measured on SmolLM2-135M, which is the checkpoint that
//! fits under QEMU and not the one the machine runs. Concluding anything about
//! the 0.6B from it would be exactly the small-sample extrapolation this
//! project's notes warn about. `repair model off` is the switch, and the
//! finding above is what somebody should try to reproduce on real hardware
//! before trusting it either way.

use crate::boot_report::Failure;
use alloc::format;
use alloc::string::String;

/// One condition on a failure record.
///
/// **This is the Troubleshooter's actual mechanism**, and it is the thing the
/// model was measured ignoring: a fault carries a vector and a symbolicated
/// site, and those two say far more about which knob is wrong than any amount
/// of reasoning about names. `dev::power::hwp_range +0x13` is not a hint, it is
/// the answer.
pub enum Clue {
    /// The fault `recover` named, spelled exactly as `describe()` spells it,
    /// which a claim checks rather than trusting.
    Fault(&'static str),
    /// The symbolicated site contains this. Matched against the function name
    /// rather than the subsystem, because `offered_for` already covers the
    /// subsystem and this is for telling two faults in one subsystem apart.
    SiteContains(&'static str),
}

pub struct Action {
    /// What the action is called, and **what the chooser mostly decides on**.
    ///
    /// Measured rather than assumed -- see the note at the top of this file. A
    /// name that tokenises into common English is chosen over one that does
    /// not, independent of what either row does, so this field is part of the
    /// prompt's probability mass and not only a label.
    pub name: &'static str,
    /// One line, for a person reading `repair`.
    ///
    /// **Deliberately not in the prompt**, and that was measured rather than
    /// decided. The first version glossed every row -- `skip-hwp (stop reading
    /// the hardware-managed performance registers)` -- and the decode came back
    /// `no choice among 2 after 0 step(s)` three times running: zero steps
    /// means the model never entered an alternative at all, it laid out
    /// whitespace until the idle allowance ran out. The prompts that work in
    /// this tree are one short sentence ending in a question, which is what
    /// `voter` asks and what `author::choose`'s own note about `prompt_for`
    /// describes. A 135M model reads a paragraph of parentheses as an
    /// invitation to write more prose.
    pub about: &'static str,
    /// Subsystems this is offered for. Empty means any.
    ///
    /// **Narrow on purpose.** An action that pokes a specific register has no
    /// business being offered for a fault in the filesystem, and a chooser
    /// that could pick it would be one bad decode away from a second fault.
    pub offered_for: &'static [&'static str],
    /// Apply it. `false` means it declined, and the loop moves on.
    pub apply: fn() -> bool,
    /// Put it back. Called when the judge did not pass.
    pub revert: fn(),
    /// What this repair is *for*, as conditions on the failure.
    ///
    /// Every clue must hold. An action with no clues is a **fallback**: it is
    /// still offered, and it is tried after anything whose clues matched.
    /// `retry` is the example and the reason the distinction exists -- retrying
    /// can only ever help a transient fault, so trying it first on a
    /// deterministic one is a guaranteed wasted attempt, which is exactly what
    /// plain table order did.
    ///
    /// A clue that does not match does not remove an action from the list. It
    /// sinks it. The judge still decides, so the ordering is allowed to be a
    /// guess -- being wrong costs an apply and a revert, never a bad repair.
    pub when: &'static [Clue],
    /// Whether writing this down for the next boot means anything.
    ///
    /// **`retry` is the reason this field exists.** It applies nothing, so
    /// persisting it asks the next boot to run a check that boot runs anyway --
    /// a line in a capped file that can never change an outcome, and one that
    /// would push a real repair out of the eighth slot. An action worth
    /// recording is one that leaves the machine in a different state than it
    /// would otherwise be in.
    pub persist: bool,
}

fn retry_apply() -> bool {
    // Nothing to do: the judge re-runs the check, which is the whole action.
    // Worth a row rather than a special case, because "it did not happen the
    // second time" is a real outcome and deserves to be recorded as the repair
    // that worked.
    true
}
fn nothing() {}

fn skip_hwp_apply() -> bool {
    crate::dev::power::skip_hwp(true);
    true
}
fn skip_hwp_revert() {
    crate::dev::power::skip_hwp(false);
}

pub static ACTIONS: &[Action] = &[
    Action {
        name: "retry",
        about: "run the check again, in case the fault was transient",
        offered_for: &[],
        // No clues: the fallback, and last by construction.
        when: &[],
        apply: retry_apply,
        revert: nothing,
        persist: false,
    },
    Action {
        name: "skip-hwp",
        about: "stop reading the hardware-managed performance registers",
        offered_for: &["power"],
        // **The GF63's own signature.** That machine took a `#GP` at
        // `dev::power::hwp_range +0x13`, reading `IA32_HWP_CAPABILITIES`
        // behind a clear `IA32_PM_ENABLE`. Both halves are load-bearing: a
        // *page* fault in `power` is some other bug entirely and this knob
        // would not touch it, and a `#GP` somewhere outside `dev::power` is
        // not an MSR gate problem.
        when: &[
            Clue::Fault("general protection fault"),
            Clue::SiteContains("dev::power"),
        ],
        apply: skip_hwp_apply,
        revert: skip_hwp_revert,
        persist: true,
    },
];

/// What is applied right now, taken from the table rather than from whoever
/// asked for it.
///
/// **Nothing a file says is stored or executed.** `apply_named` resolves two
/// words to a row and keeps the row's own `&'static str`, which is the same
/// bargain `author::choose` makes when the model picks one: the chooser names a
/// row, the kernel owns what the row does. A hand-edited `REPAIRS.TXT` can
/// therefore ask for a repair that does not exist, and gets nothing.
static IN_FORCE: crate::sync::Racy<[Option<(&'static str, &'static str)>; 8]> =
    crate::sync::Racy::new([None; 8]);

/// Adopted this boot and not yet written down. Separate from `IN_FORCE`
/// because a repair read back off the disk is already recorded, and appending
/// it again every boot is how a capped file fills up with one entry.
static ADOPTED: crate::sync::Racy<[Option<(&'static str, &'static str)>; 8]> =
    crate::sync::Racy::new([None; 8]);

fn note(slots: &crate::sync::Racy<[Option<(&'static str, &'static str)>; 8]>, e: (&'static str, &'static str)) {
    let a = unsafe { slots.get() };
    for s in a.iter_mut() {
        if *s == Some(e) {
            return;
        }
        if s.is_none() {
            *s = Some(e);
            return;
        }
    }
}

/// Every repair currently applied, whether it came off the disk or was decided
/// a moment ago.
pub fn in_force() -> impl Iterator<Item = (&'static str, &'static str)> {
    unsafe { IN_FORCE.get() }.iter().flatten().copied()
}

/// Apply a repair named by two words, refusing anything the table would not
/// have offered.
///
/// **The `offered_for` check is not decoration here.** At boot this is called
/// with strings read off a FAT partition, which anything that can mount that
/// partition can edit -- so without it, a text file could aim a power register
/// knob at the filesystem. The rule that binds a chooser binds a file.
/// Whether `apply_named` would accept this pair, without applying anything.
///
/// Split out so the recording end can ask the applying end rather than
/// reimplementing its rules -- which is exactly how they came to disagree.
pub fn would_apply(subsystem: &str, action: &str) -> bool {
    ACTIONS
        .iter()
        .find(|a| a.name == action)
        .is_some_and(|a| a.offered_for.is_empty() || a.offered_for.contains(&subsystem))
}

pub fn apply_named(subsystem: &str, action: &str) -> Option<(&'static str, &'static str)> {
    let a = ACTIONS.iter().find(|a| a.name == action)?;
    if !a.offered_for.is_empty() && !a.offered_for.contains(&subsystem) {
        return None;
    }
    // Resolved to the row's own strings, so nothing read off the disk outlives
    // this function. A universal action carries "any" rather than the name it
    // was asked about, because that is the truth about what is applied.
    let sub = a.offered_for.iter().find(|s| **s == subsystem).copied().unwrap_or("any");
    if !(a.apply)() {
        return None;
    }
    note(&IN_FORCE, (sub, a.name));
    Some((sub, a.name))
}

/// Whether every clue an action carries holds for this failure.
///
/// An action with no clues answers `false` here and is a fallback rather than a
/// match -- `rank` is what knows the difference, so that "matched nothing" and
/// "asked for nothing" stay separate facts.
pub fn matches(f: &Failure, a: &Action) -> bool {
    !a.when.is_empty()
        && a.when.iter().all(|c| match c {
            Clue::Fault(name) => f.why == *name,
            Clue::SiteContains(part) => crate::boot_report::site_of(f.rip)
                .map(|s| s.contains(part))
                .unwrap_or(false),
        })
}

/// The order to try repairs in, decided entirely by the failure record.
///
/// **Pure, so every case is assertable at boot with no model, no disk and
/// nothing injected** -- the discipline `update::decide` and
/// `repairs::decide` already follow, and the reason this replaced a decode
/// rather than sitting beside one.
///
/// Three groups, and nothing is ever dropped:
///
/// 1. actions whose clues all hold, in table order
/// 2. actions asking for nothing, which are the fallbacks
/// 3. actions that asked for something and did not get it
///
/// The third group is kept rather than discarded because a clue is evidence
/// about what is *likely*, not a proof about what is possible, and the judge is
/// what actually decides. Discarding would turn a wrong guess about a signature
/// into a repair the machine can no longer reach.
pub fn rank(f: &Failure) -> alloc::vec::Vec<&'static Action> {
    let mut matched = alloc::vec::Vec::new();
    let mut fallback = alloc::vec::Vec::new();
    let mut rest = alloc::vec::Vec::new();
    for a in offered(f.name) {
        if matches(f, a) {
            matched.push(a);
        } else if a.when.is_empty() {
            fallback.push(a);
        } else {
            rest.push(a);
        }
    }
    matched.extend(fallback);
    matched.extend(rest);
    matched
}

/// Which actions are offered for a subsystem, in table order.
pub fn offered(subsystem: &str) -> impl Iterator<Item = &'static Action> + '_ {
    ACTIONS
        .iter()
        .filter(move |a| a.offered_for.is_empty() || a.offered_for.contains(&subsystem))
}

/// Re-run the check that failed, and answer whether it survives now.
///
/// Guarded, because the whole reason it is here is that it faulted once. The
/// panic window is opened for the same reason it is open during the boot
/// selftests: an `assert!` is how most of these fail.
///
/// **It asks two things and used to ask one.** `matches!(.., Caught::Ran)`
/// answers whether the check *finished*, which is the question a fault poses
/// and not the question a repair does: an action that leaves a subsystem alive
/// and answering wrongly would have been adopted, marked as the repair that
/// worked, and written to the boot volume for every boot afterwards. So the
/// check has to run **and** agree, and `false` without a fault is refused the
/// same way a fault is.
fn judge(f: &Failure) -> bool {
    // Saved and restored rather than closed, because this is callable from
    // inside a selftest -- and a judge that closed the window on its way out
    // would silently take panic recovery away from every check after it.
    let was = crate::cpu::recover::in_selftest();
    crate::cpu::recover::selftest_window(true);
    let mut agreed = false;
    let ran = matches!(
        crate::cpu::recover::guarded(|| agreed = (f.retry)()),
        crate::cpu::recover::Caught::Ran
    );
    crate::cpu::recover::selftest_window(was);
    ran && agreed
}

/// What happened to one subsystem.
pub enum Outcome {
    /// An action was applied and the check then passed. The name is the action.
    Repaired(&'static str),
    /// Everything offered was tried and the check still fails.
    Unrepaired,
}

/// How many repairs are tried on one subsystem before the machine gives up.
///
/// Three, and the bound is about the *shape* of the evidence rather than about
/// the cost. A subsystem that has refused three different repairs does not have
/// a fourth-repair problem; something upstream of all of them is wrong, and
/// going on would fill the boot log with reverts.
const MAX_ATTEMPTS: usize = 3;

/// Whether the model is consulted at all. Off falls back to table order.
static USE_MODEL: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

pub fn use_model(on: bool) {
    USE_MODEL.store(on, core::sync::atomic::Ordering::Relaxed);
}
pub fn model_in_use() -> bool {
    USE_MODEL.load(core::sync::atomic::Ordering::Relaxed)
}

/// Three decodes before falling back, for `voter::pick`'s reason.
///
/// A constrained decode that will not commit is a small model failing to
/// commit and never a verdict -- treating one as a verdict is exactly how an
/// unlucky decode once abandoned a whole night's composition. Bounded at three
/// because a model that will not commit three times is not unlucky, and an
/// unbounded retry on a path that runs at boot is a machine that never reaches
/// the shell.
const DECODE_TRIES: usize = 3;

/// What the model is told. The operator sees the same words.
///
/// It carries the *site* as well as the subsystem, because those answer
/// different questions -- which check failed, and what actually broke -- and a
/// chooser shown only the first is being asked to repair a name. It also
/// carries what has already been tried and failed, so a second round is a
/// different question rather than the same one asked again.
fn prompt_for(f: &Failure, remaining: &[&'static Action], tried: &[&'static str]) -> String {
    let mut p = format!("The {} selftest failed with {}", f.name, f.why);
    if let Some(site) = crate::boot_report::site_of(f.rip) {
        p.push(' ');
        p.push_str(&site);
    }
    p.push('.');
    if !tried.is_empty() {
        p.push_str(&format!(" {} did not help.", tried.join(" and ")));
    }
    p.push_str(&format!(
        " Repairs: {}. Which is most likely to fix it?",
        remaining.iter().map(|a| a.name).collect::<alloc::vec::Vec<_>>().join(", ")
    ));
    p
}

/// Which of the remaining repairs to try next.
///
/// Answers an index into `remaining`, and **zero is the fixed rule**: the first
/// row the table offers. Every path that cannot produce a considered answer
/// lands there rather than on an error, because a machine with a broken
/// subsystem and no model still deserves its repair attempted.
fn choose_next(
    f: &Failure,
    remaining: &[&'static Action],
    tried: &[&'static str],
) -> (usize, &'static str) {
    // **Who chose is recorded, not inferred.** A model that agrees with table
    // order and a fallback to table order produce the same sequence of
    // attempts, so a transcript that does not say which happened cannot answer
    // the only question worth asking of this feature -- and a boot where the
    // engine was busy would read as a boot where the model picked the first
    // row. Three reasons to land on index zero and they mean different things.
    if remaining.len() == 1 {
        return (0, "forced");
    }
    if !model_in_use() {
        // Zero is the ranking's own answer, which is a decision about this
        // fault rather than a position in a table.
        return (0, "rule");
    }
    if !crate::ai::engine_ready() {
        return (0, "no model");
    }
    let names: alloc::vec::Vec<&str> = remaining.iter().map(|a| a.name).collect();
    let prompt = prompt_for(f, remaining, tried);
    for _ in 0..DECODE_TRIES {
        if let Some(i) = crate::ai::author::choose(&prompt, &names) {
            return (i, "model");
        }
    }
    // Said rather than silent. A boot that fell back to table order and a boot
    // where the model happened to agree with table order are indistinguishable
    // from the outcome, and only one of them is worth looking into.
    crate::serial_println!(
        "[repair] the model would not choose among {} for {}; taking table order",
        remaining.len(),
        f.name
    );
    (0, "undecided")
}

/// One line per attempt, in `/ai/repair/log`.
///
/// Two reasons, and the second is the durable one. It is the transcript worth
/// printing when nothing worked -- what was tried and what happened, rather
/// than a register dump. And it is the corpus a `Probe` could later be fitted
/// on to replace the decode with the routing layer's answer, which is not worth
/// fitting until the corpus exists, because with zero examples it has nothing
/// to beat a grammar decode with.
///
/// Written to the namespace, so it is memory until something snapshots. That is
/// the right level: a repair *decision* is on the ESP because it has to survive
/// a reboot to mean anything, and the reasoning behind it is worth keeping and
/// not worth a disk write.
fn note_attempt(f: &Failure, action: &str, by: &str, outcome: &str) {
    let line = format!(
        "{}\t{}\t{}\t{}\t{}\t{}\n",
        f.name,
        f.why,
        crate::boot_report::site_of(f.rip).unwrap_or_else(|| String::from("-")),
        action,
        by,
        outcome
    );
    let path = "/ai/repair/log";
    let mut all = crate::sysbox::read_blob(path)
        .and_then(|b| String::from_utf8(b).ok())
        .unwrap_or_default();
    // Bounded, because this runs every boot on a machine whose fault may be
    // permanent, and a log nothing trims is a log that eventually is the heap.
    // Oldest lines go first, since the interesting end of a repair history is
    // the recent one.
    while all.len() + line.len() > LOG_BYTES {
        match all.find('\n') {
            Some(i) => all = all.split_off(i + 1),
            None => all.clear(),
        }
    }
    all.push_str(&line);
    crate::sysbox::write_text(path, &all);
}

const LOG_BYTES: usize = 8192;

/// Try to repair one failed subsystem.
///
/// **A failed repair is reverted before the next is tried**, so the machine
/// never accumulates a pile of changes that did not help. Whatever is left
/// standing at the end is exactly the one that worked, or nothing.
pub fn attempt(f: &Failure) -> Outcome {
    let mut remaining: alloc::vec::Vec<&'static Action> = rank(f);
    let mut tried: alloc::vec::Vec<&'static str> = alloc::vec::Vec::new();

    for _ in 0..MAX_ATTEMPTS {
        if remaining.is_empty() {
            break;
        }
        // The index is into `remaining` and came from a grammar built from
        // `remaining`, so it is in range by construction. Clamped anyway, at
        // the one line where a chooser's answer becomes a table index -- the
        // construction is `author::choose`'s to keep and this is the place that
        // would be wrong if it ever stopped keeping it.
        let (i, by) = choose_next(f, &remaining, &tried);
        let a = remaining.remove(i.min(remaining.len() - 1));

        if !(a.apply)() {
            note_attempt(f, a.name, by, "declined to apply");
            tried.push(a.name);
            continue;
        }
        if judge(f) {
            note_attempt(f, a.name, by, "the check passes");
            return Outcome::Repaired(a.name);
        }
        (a.revert)();
        note_attempt(f, a.name, by, "the check still fails");
        tried.push(a.name);
    }
    Outcome::Unrepaired
}

/// Try every failed subsystem, and say what happened.
///
/// Table order rather than failure order, because the table is written in
/// dependency order -- there is no point repairing something that reads
/// storage before storage itself.
pub fn attempt_all() {
    use crate::kprintln;
    if crate::boot_report::count() == 0 {
        return;
    }
    crate::gfx::console::set_color(crate::gfx::console::LTGRAY);
    kprintln!("\n[repair] {} subsystem(s) to try", crate::boot_report::count());
    for f in crate::boot_report::failures() {
        match attempt(&f) {
            Outcome::Repaired(action) => {
                crate::gfx::console::set_color(crate::gfx::console::LTGREEN);
                kprintln!("  {:<14} repaired by '{}', and the check now passes", f.name, action);
                crate::boot_report::mark_repaired(f.name, action);
                note(&IN_FORCE, (f.name, action));
                // Only what was decided *here* is queued to be written down.
                // One read back off the disk is already recorded, and appending
                // it every boot is how an eight-entry file fills with one
                // entry.
                // Only actions that change something are worth writing down,
                // and only ones this boot decided for itself -- see `persist`
                // and `applied_from_disk`.
                let worth_keeping = ACTIONS
                    .iter()
                    .find(|a| a.name == action)
                    .is_some_and(|a| a.persist);
                if worth_keeping && !applied_from_disk(f.name, action) {
                    note(&ADOPTED, (f.name, action));
                }
            }
            Outcome::Unrepaired => {
                crate::gfx::console::set_color(crate::gfx::console::LTRED);
                kprintln!("  {:<14} nothing offered fixed it; still unavailable", f.name);
            }
        }
    }
    crate::gfx::console::set_color(crate::gfx::console::LTGRAY);
}

/// Build a `Failure` for a check that is not a real subsystem's.
///
/// The name is deliberately not one in `ACTIONS`, so the only action offered
/// for it is the universal one -- which is what makes the judge claims below
/// about the judge rather than about `skip-hwp`.
fn synthetic(retry: fn() -> bool) -> Failure {
    Failure {
        name: "nothing-by-this-name",
        need: crate::boot_report::Need::Optional,
        why: "synthetic",
        rip: 0,
        retry,
        repaired_by: None,
    }
}

fn faults() -> bool {
    unsafe { core::ptr::read_volatile(0x0 as *const u64) };
    true
}
fn passes() -> bool {
    true
}
/// Runs to the end and answers no. The third outcome, and the one the judge
/// could not see at all until `section` learned to carry a verdict: a
/// subsystem that is alive and wrong rather than gone.
fn wrong() -> bool {
    false
}

/// An address inside a function whose symbol contains `part`, or 0.
///
/// Found through the same table the report reads rather than written down, so a
/// claim built on it is about this image instead of about a number somebody
/// once measured.
fn site_rip(part: &str) -> u64 {
    if part.is_empty() {
        return 0;
    }
    let base = crate::cpu::idt::IMAGE_BASE.load(core::sync::atomic::Ordering::Relaxed);
    crate::cpu::code::find_symbol(part).map(|rva| base + rva).unwrap_or(0)
}

pub fn selftest() -> bool {
    let mut ok = true;
    fn claim(ok: &mut bool, good: bool, what: &str) {
        crate::kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
        *ok &= good;
    }

    // ---- the table ---------------------------------------------------------
    //
    // Read out of the table rather than written down here. A claim naming
    // `power` and `skip-hwp` in its own text asserts what the table said on the
    // day it was written, so renaming a row would leave the suite passing while
    // testing something that no longer exists.
    claim(
        &mut ok,
        {
            let mut uniq = true;
            for (i, a) in ACTIONS.iter().enumerate() {
                if ACTIONS[..i].iter().any(|b| b.name == a.name) {
                    uniq = false;
                }
            }
            uniq && ACTIONS.iter().all(|a| !a.name.is_empty() && !a.about.is_empty())
        },
        "every action has a distinct name and something to say about itself",
    );

    claim(
        &mut ok,
        ACTIONS
            .iter()
            .filter(|a| a.offered_for.is_empty())
            .all(|a| offered("nothing-by-this-name").any(|b| b.name == a.name)),
        "an action listing no subsystems is offered for every subsystem",
    );

    // **The misattribution claim.** A narrow action must reach its own
    // subsystem and nothing else, because a chooser that could pick a power
    // register knob for a graphics fault is one decision away from a second
    // fault in a subsystem nobody was repairing. Both halves are derived from
    // the row's own list, so the claim keeps meaning this as the table grows.
    let mut narrow_reaches = true;
    let mut narrow_stays = true;
    for a in ACTIONS.iter().filter(|a| !a.offered_for.is_empty()) {
        for sub in a.offered_for {
            if !offered(sub).any(|b| b.name == a.name) {
                narrow_reaches = false;
            }
        }
        if offered("nothing-by-this-name").any(|b| b.name == a.name) {
            narrow_stays = false;
        }
    }
    claim(&mut ok, narrow_reaches, "a narrow action is offered for each subsystem it names");
    claim(&mut ok, narrow_stays, "and for no other, however the table grows");

    // ---- the judge ---------------------------------------------------------
    //
    // Judged against checks that are nothing to do with any subsystem, so what
    // is measured is the judge and never a particular repair. A judge that
    // answered yes to everything would make every failure look repaired by
    // whatever the table happened to offer first.
    claim(
        &mut ok,
        judge(&synthetic(passes)),
        "a check that runs to the end is judged as passing",
    );
    claim(
        &mut ok,
        !judge(&synthetic(faults)),
        "and one that still faults is not",
    );
    // **The claim the rest of this loop was built without.** For as long as
    // `judge` asked only `Caught::Ran`, a repair that left a subsystem alive
    // and answering wrongly was adopted, recorded as the repair that worked,
    // and persisted to the boot volume -- indistinguishable from one that
    // fixed the bug. Running to the end is not the same as being right.
    claim(
        &mut ok,
        !judge(&synthetic(wrong)),
        "and one that runs to the end and answers no is not either",
    );

    // **A judge with nothing to judge is the quiet way this whole loop stops
    // working.** `recheck_persisted` needs `check_for(name)`, and a check that
    // did not fit the roll answers `None` -- so a repair for that subsystem
    // lives on the boot volume forever with nothing able to ask whether its
    // bug has since been fixed. Nothing else can see that, so it is a claim.
    //
    // `checks_noted() > 0` is the other half and is not decoration: on an
    // empty roll nothing was dropped either, so the count alone is a claim
    // that cannot fail.
    claim(
        &mut ok,
        crate::boot_report::checks_dropped() == 0 && crate::boot_report::checks_noted() > 0,
        "every boot check registered fits the roll, so every one can be re-judged",
    );

    // The window is the panic handler's gate, so a judge that left it open
    // would make a panic anywhere afterwards recoverable, and one that left it
    // shut would take recovery away from the checks that follow.
    let before = crate::cpu::recover::in_selftest();
    let _ = judge(&synthetic(faults));
    claim(
        &mut ok,
        crate::cpu::recover::in_selftest() == before,
        "and judging leaves the selftest window exactly as it found it",
    );

    // **The `retry` rule, stated as a property rather than as a row.** An
    // action that applies nothing cannot change what the next boot does, so
    // persisting it spends a slot in a capped file on a line with no effect.
    claim(
        &mut ok,
        ACTIONS.iter().any(|a| a.persist) && ACTIONS.iter().any(|a| !a.persist),
        "the table has both kinds, so the distinction is being exercised",
    );
    claim(
        &mut ok,
        {
            let before = crate::dev::power::hwp_skipped();
            let unchanged = ACTIONS.iter().filter(|a| !a.persist).all(|a| {
                (a.apply)();
                let same = crate::dev::power::hwp_skipped() == before;
                (a.revert)();
                same
            });
            unchanged
        },
        "and an action that does not persist is one that changes no knob",
    );

    // A file may name anything, so `apply_named` is the gate rather than the
    // caller. Both refusals are the misattribution claim again, arriving by the
    // route an edited text file would take.
    claim(
        &mut ok,
        apply_named("power", "no-such-action").is_none(),
        "a repair this kernel does not have is refused however it was asked for",
    );
    claim(
        &mut ok,
        {
            let narrow = ACTIONS.iter().find(|a| !a.offered_for.is_empty());
            match narrow {
                Some(a) => apply_named("nothing-by-this-name", a.name).is_none(),
                None => true,
            }
        },
        "and a narrow one aimed at a subsystem it was never offered for",
    );

    // ---- the ranking, which is the thing that actually decides order -------
    //
    // Pure over the failure record, so all of this runs with no model, no disk
    // and nothing injected. That is the whole reason it replaced a decode: a
    // repair loop that needs the model cannot repair the model, and a boot
    // where the checkpoint will not load is exactly when a repair matters most.
    //
    // Every clue is read out of the table. A claim spelling "general protection
    // fault" in its own text would assert what the table said the day it was
    // written, and worse, would not notice `recover::describe` renaming a
    // vector underneath it.
    let signature = ACTIONS.iter().find(|a| !a.when.is_empty());
    match signature {
        None => claim(&mut ok, false, "some action carries a signature to match on"),
        Some(a) => {
            let sub = a.offered_for.first().copied().unwrap_or("any");

            // Built from the row's own clues, so it is by construction the
            // failure this action is for.
            let mut why = "";
            let mut site_part = "";
            for c in a.when {
                match c {
                    Clue::Fault(n) => why = n,
                    Clue::SiteContains(part) => site_part = part,
                }
            }
            // A rip inside the function the clue names, found through the same
            // symbol table the report uses rather than written down.
            let hit = crate::boot_report::Failure {
                name: sub,
                need: crate::boot_report::Need::Optional,
                why,
                rip: site_rip(site_part),
                retry: passes,
                repaired_by: None,
            };

            claim(
                &mut ok,
                matches(&hit, a),
                "an action's own signature matches the failure it describes",
            );
            claim(
                &mut ok,
                rank(&hit).first().map(|r| r.name) == Some(a.name),
                "and that puts it first, ahead of anything asking for nothing",
            );

            // **The claim this whole redesign exists for.** The same subsystem,
            // a different fault: a register knob must not be reached for
            // because something else in `power` went wrong. Same shape as the
            // `offered_for` claims, one level finer.
            let other = crate::boot_report::Failure {
                why: if why == "page fault" { "invalid opcode" } else { "page fault" },
                ..hit
            };
            claim(
                &mut ok,
                !matches(&other, a),
                "a different fault in the same subsystem does not match that signature",
            );
            claim(
                &mut ok,
                rank(&other).first().map(|r| r.name) != Some(a.name),
                "and is not offered that repair first",
            );

            // A site somewhere else entirely, with the right vector. Half a
            // signature is not a signature.
            let elsewhere = crate::boot_report::Failure { rip: 0, ..hit };
            claim(
                &mut ok,
                !matches(&elsewhere, a),
                "nor does the right fault at a site the signature does not name",
            );

            // `retry` is the case that made ordering worth fixing: it can only
            // help a transient fault, so trying it first on a deterministic one
            // is a guaranteed wasted attempt -- which is what plain table order
            // did on every boot.
            claim(
                &mut ok,
                {
                    let order = rank(&hit);
                    let clueless = order.iter().position(|r| r.when.is_empty());
                    let matched = order.iter().position(|r| matches(&hit, r));
                    match (clueless, matched) {
                        (Some(c), Some(m)) => m < c,
                        _ => true,
                    }
                },
                "an action asking for nothing is tried after one whose clues held",
            );
        }
    }

    // Nothing may be dropped. A clue is evidence about what is likely and never
    // a proof about what is possible, so a wrong guess about a signature must
    // cost an attempt's ordering and never a repair the machine can reach.
    claim(
        &mut ok,
        {
            let f = crate::boot_report::Failure {
                name: ACTIONS
                    .iter()
                    .find(|a| !a.offered_for.is_empty())
                    .and_then(|a| a.offered_for.first().copied())
                    .unwrap_or("any"),
                need: crate::boot_report::Need::Optional,
                why: "nothing that matches any clue",
                rip: 0,
                retry: passes,
                repaired_by: None,
            };
            let ranked = rank(&f);
            let offers = offered(f.name).count();
            ranked.len() == offers
                && offered(f.name).all(|a| ranked.iter().any(|r| r.name == a.name))
        },
        "ranking reorders what is offered and never drops any of it",
    );

    // The clues name vectors by the string `recover` prints, so a rename there
    // would silently stop every signature matching -- with no error, and the
    // machine merely repairing itself worse.
    claim(
        &mut ok,
        ACTIONS.iter().flat_map(|a| a.when.iter()).all(|c| match c {
            Clue::Fault(name) => crate::cpu::recover::names().contains(name),
            Clue::SiteContains(_) => true,
        }),
        "every fault a clue names is one `recover` can actually report",
    );

    // ---- what the chooser is told -----------------------------------------
    //
    // **The misattribution claim, on the other side of the loop.** `offered`
    // stops a power knob being tried on a graphics fault; these stop the model
    // being *told* about a fault other than the one that happened. Both halves
    // matter and they fail differently: the first would apply the wrong repair,
    // the second would apply a defensible repair to a description of somebody
    // else's problem.
    //
    // Every string here is derived from the failure or from the table. A claim
    // that looked for the word "power" would assert what the table said on the
    // day it was written.
    let victim = ACTIONS
        .iter()
        .find(|a| !a.offered_for.is_empty())
        .and_then(|a| a.offered_for.first().copied())
        .unwrap_or("any");
    let hurt = crate::boot_report::Failure {
        name: victim,
        need: crate::boot_report::Need::Optional,
        why: "a fault by some name",
        // A real address in this image, so the site resolves to a real symbol
        // rather than to a number somebody wrote down.
        rip: faults as usize as u64,
        retry: faults,
        repaired_by: None,
    };
    let offers: alloc::vec::Vec<&'static Action> = offered(victim).collect();
    let p = prompt_for(&hurt, &offers, &[]);

    claim(
        &mut ok,
        p.contains(victim) && p.contains(hurt.why),
        "the prompt names the subsystem that failed and the fault it took",
    );
    claim(
        &mut ok,
        match crate::boot_report::site_of(hurt.rip) {
            Some(site) => p.contains(&site),
            // No symbol table in this build is a fact about the build, not a
            // failure of the prompt.
            None => true,
        },
        "and where the fault actually was, not only which check was blamed",
    );

    // The one that would catch a battery fault being described as a graphics
    // one. Every other subsystem the table knows about must be absent.
    claim(
        &mut ok,
        ACTIONS
            .iter()
            .flat_map(|a| a.offered_for.iter())
            .all(|s| *s == victim || !p.contains(*s)),
        "and no subsystem other than the one that broke is mentioned at all",
    );
    claim(
        &mut ok,
        ACTIONS
            .iter()
            .all(|a| p.contains(a.name) == offers.iter().any(|o| o.name == a.name)),
        "the prompt offers exactly the repairs this subsystem is offered, and no others",
    );

    // A second round is a different question. Without this the model is asked
    // the same thing twice and has every reason to answer it the same way.
    let again = prompt_for(&hurt, &offers, &["some-earlier-try"]);
    claim(
        &mut ok,
        again.contains("some-earlier-try") && again != p,
        "and a second round says what already failed",
    );

    // ---- falling back ------------------------------------------------------
    //
    // The fixed rule is not a legacy path, it is what runs on every machine
    // with no model -- which under QEMU is most of them.
    let was = model_in_use();
    use_model(false);
    let fixed = choose_next(&hurt, &offers, &[]);
    use_model(was);
    claim(
        &mut ok,
        fixed == (0, "rule"),
        "with the model off the chooser takes the ranking's answer, and says so",
    );
    // Three ways to land on index zero, and a transcript that could not tell
    // them apart would report a busy engine as a model that picked the first
    // row. The names are distinct because that is the whole use of them.
    claim(
        &mut ok,
        {
            let one: alloc::vec::Vec<&'static Action> = offers.iter().take(1).copied().collect();
            choose_next(&hurt, &one, &[]) == (0, "forced")
        },
        "and a single offered repair is recorded as forced rather than chosen",
    );

    // ---- the transcript ----------------------------------------------------
    //
    // A machine whose fault is permanent boots and writes these forever, so a
    // log nothing trims is a log that is eventually the heap.
    for _ in 0..400 {
        note_attempt(&hurt, "some-action", "table", "the check still fails");
    }
    let grown = crate::sysbox::read_blob("/ai/repair/log").map(|b| b.len()).unwrap_or(0);
    claim(
        &mut ok,
        grown > 0 && grown <= LOG_BYTES,
        "the attempt log records what was tried and stays bounded doing it",
    );

    // ---- apply and revert --------------------------------------------------
    //
    // Found by name: an index would keep passing while testing whichever row
    // had moved into that slot.
    match ACTIONS.iter().find(|a| a.name == "skip-hwp") {
        Some(a) => {
            let before = crate::dev::power::hwp_skipped();
            (a.apply)();
            let during = crate::dev::power::hwp_skipped();
            (a.revert)();
            let after = crate::dev::power::hwp_skipped();
            claim(&mut ok, !before && during, "applying a repair changes the thing it names");
            claim(&mut ok, after == before, "and reverting it puts the value back");
        }
        None => claim(&mut ok, false, "the action this suite reverts is still in the table"),
    }

    // A failed repair must leave nothing behind. `attempt` on a check that
    // cannot be fixed tries everything offered and reverts each one, so the
    // machine after it is the machine before it.
    let before = crate::dev::power::hwp_skipped();
    let outcome = attempt(&synthetic(faults));
    claim(
        &mut ok,
        matches!(outcome, Outcome::Unrepaired) && crate::dev::power::hwp_skipped() == before,
        "a repair that did not work leaves the machine as it was",
    );

    ok
}

/// Repairs that came off the boot volume this boot, so an adoption that merely
/// re-derives one is not written down a second time.
static FROM_DISK: crate::sync::Racy<[Option<(&'static str, &'static str)>; 8]> =
    crate::sync::Racy::new([None; 8]);

pub fn note_from_disk(subsystem: &'static str, action: &'static str) {
    note(&FROM_DISK, (subsystem, action));
}

fn applied_from_disk(subsystem: &str, action: &str) -> bool {
    unsafe { FROM_DISK.get() }
        .iter()
        .flatten()
        .any(|(s, a)| *a == action && (*s == subsystem || *s == "any"))
}

/// Report any repair whose subsystem now passes without it.
///
/// **The rule this closes: a repair never silently replaces a fix.** A
/// workaround adopted for a bug somebody has since actually fixed would
/// otherwise live on the boot volume forever, and from every other vantage
/// point a subsystem held up by a repair looks exactly like one that is simply
/// working.
///
/// So the repair is taken away, the check is run again, and it is put back
/// whatever the answer. Put back rather than left off deliberately: this
/// reports, and withdrawing a repair the machine has been relying on is the
/// operator's decision, not a side effect of looking.
///
/// A subsystem that failed this boot is skipped -- it is still broken, so it
/// has nothing to say about whether its repair is still needed.
pub fn recheck_persisted() {
    use crate::kprintln;
    for (sub, act) in in_force() {
        if sub == "any" || crate::boot_report::failed(sub) {
            continue;
        }
        let Some(a) = ACTIONS.iter().find(|a| a.name == act) else {
            continue;
        };
        let Some(check) = crate::boot_report::check_for(sub) else {
            continue;
        };

        (a.revert)();
        let passes = judge(&crate::boot_report::Failure {
            name: sub,
            need: crate::boot_report::Need::Optional,
            why: "re-checked without its repair",
            rip: 0,
            retry: check,
            repaired_by: None,
        });
        // Unconditionally, including the path where the check faulted -- the
        // fault was caught, and a machine left with its repair off because
        // looking went wrong is worse than one that never looked.
        (a.apply)();

        if passes {
            crate::gfx::console::set_color(crate::gfx::console::LTGREEN);
            kprintln!(
                "[repair] {} passes without '{}' now, so the repair may have outlived its bug",
                sub,
                act
            );
            kprintln!("         `repair clear` forgets it; it stays applied until then");
            crate::gfx::console::set_color(crate::gfx::console::LTGRAY);
        }
    }
}

/// Put every applied repair back, and stop recording.
///
/// Answers how many were reverted. The boot volume is deliberately untouched:
/// undoing a repair for this boot and forgetting it forever are different
/// decisions, and an operator investigating whether a repair is still needed
/// wants the first without committing to the second.
pub fn revert_all() -> usize {
    let mut n = 0;
    for (_, act) in in_force() {
        if let Some(a) = ACTIONS.iter().find(|a| a.name == act) {
            (a.revert)();
            n += 1;
        }
    }
    *unsafe { IN_FORCE.get() } = [None; 8];
    *unsafe { ADOPTED.get() } = [None; 8];
    n
}

/// Write down what was adopted this boot, now that there is a disk to write to.
///
/// Deliberately not part of `attempt_all`, which runs before NVMe comes up:
/// the loop has to decide early so the boot summary is about the machine as it
/// now is, and the write has to happen late because there is nothing to write
/// to until the controller answers. Splitting them is cheaper than moving
/// either.
///
/// Failure here is reported and is not a failure of the repair. A machine with
/// no ESP -- a live ISO, or QEMU's synthetic FAT16 -- still gets the repair for
/// this boot and rediscovers it on the next one, which is the whole loop
/// working slightly harder rather than not working.
pub fn persist_adopted() {
    use crate::kprintln;
    let queued: alloc::vec::Vec<_> = unsafe { ADOPTED.get() }.iter().flatten().copied().collect();
    if queued.is_empty() {
        return;
    }
    for (sub, act) in queued {
        match crate::update::repairs::record(sub, act) {
            Ok(line) => {
                crate::gfx::console::set_color(crate::gfx::console::LTGREEN);
                kprintln!("[repair] {}", line);
            }
            Err(e) => {
                crate::gfx::console::set_color(crate::gfx::console::LTGRAY);
                kprintln!("[repair] '{}' for {} holds for this boot only: {}", act, sub, e);
            }
        }
    }
    crate::gfx::console::set_color(crate::gfx::console::LTGRAY);
}
