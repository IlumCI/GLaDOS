//! Learning from what the operator does, rather than from what the model said.
//!
//! The routing corpus grows by `teach`, which is a person typing a label. So a
//! machine that has been used for a month routes exactly as it did on its first
//! boot, and every correction the operator made by hand -- asking for one thing,
//! watching the wrong applet run, then running the right one themselves -- was
//! thrown away at the moment it was most informative.
//!
//! ### Why corrections, and emphatically not successes
//!
//! The obvious design is to harvest steps that worked. This tree has already
//! measured that and it does nothing, which is worth stating before somebody
//! re-proposes it: `work.rs`'s role adapters were trained on harvested
//! transcripts and reported `fixed 0 broke 0` on 23 examples, base 97.3% and
//! 100% held out. The reason is structural rather than a shortage of data --
//! **a harvested label is the base model's own argmax.** `choose` decodes at
//! temperature zero, so the action in a successful transcript is whatever the
//! classifier already ranks first, and training on it asks the model to
//! reproduce what produced it. Filtering on "it ran" does not escape that,
//! because running records that the applet accepted its arguments and never
//! that it was the right applet.
//!
//! A correction is the opposite: the label comes from the operator's hands.
//! The model ranked `find` first, a person ranked `ls` first, and the
//! disagreement is the whole of the signal. It is the one label in this machine
//! that the model did not generate.
//!
//! ### What counts as a correction
//!
//! `judge` is the whole rule and it is a pure function, in the shape
//! `update::decide` and `repair::rank` already use here, so every branch is
//! asserted at boot with no model, no engine and no disk. Four refusals, each
//! for its own reason:
//!
//! - **No goal live.** A shell command with no preceding request is a person
//!   using their computer, not correcting anything.
//! - **The operator agreed.** If they ran what the agent chose, the label is
//!   the model's argmax again and we are back to the role-adapter result.
//! - **Not an applet.** `fit`, `diag`, `godel` and the rest are operator verbs
//!   with no row in the table, so they cannot be a routing label -- the grammar
//!   could never emit one.
//! - **The goal does not read like a request.** One word, or a line that is
//!   itself a command, is not a task description and would teach the router to
//!   key on something no future goal will look like.
//!
//! ### What it deliberately does not do
//!
//! It does not refit. Appending changes `/ai/train`, the cached router is keyed
//! by that directory's hash, and `harness::ensure_router` refits when the hash
//! moves -- so the next decision pays 1,115 ms once and the loop closes with no
//! verb to type. Nothing here needs to know that; it is a property of the cache.
//!
//! It does not write the held-out slice. `vocab::splits` takes its boundaries
//! from recorded positions and "anything past the recorded length trains", so an
//! append is a training row by construction and cannot move the test set. That
//! is the same rule `teach` on a live system already relies on.
//!
//! And it is **capped per boot** and prints every row it writes. A learner that
//! silently rewrites the thing every judge in this tree measures against is a
//! machine whose evidence nobody can reconstruct; the alpha budget is scoped to
//! a corpus hash for exactly that reason, and a corpus that moves twenty times
//! a session would refill it twenty times.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// How many corrections one boot may write.
///
/// Not a storage bound -- a row is a few dozen bytes. It bounds how far the
/// corpus can move between two chances to look at it, because every judged
/// comparison in `godel` is scoped to a corpus hash and a corpus that moves on
/// every command is one nothing can be compared across.
pub const PER_BOOT: usize = 8;

/// The shortest goal worth learning from, in words.
const MIN_WORDS: usize = 3;

/// What `judge` decided, so a caller can say why nothing was learned.
#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Write this row: the applet, and the task it should have been chosen for.
    Learn(String, String),
    /// Nothing was asked, so nothing was corrected.
    NoGoal,
    /// The operator ran what the agent chose. See the module doc.
    Agreed,
    /// Not a routing label at all.
    NotAnApplet,
    /// Not shaped like a request.
    NotARequest,
    /// Already known, so writing it again would only weight it.
    Known,
    /// This boot has written as many as it may.
    Spent,
}

/// Does this operator command correct the agent's last choice?
///
/// Pure, so every branch is a claim. `goal` is the last request the operator
/// made, `chose` is what the agent did about it, `verb` is what the operator
/// then typed, and `known` answers whether the corpus already holds this pair.
pub fn judge(
    goal: Option<&str>,
    chose: Option<&str>,
    verb: &str,
    is_applet: bool,
    known: bool,
    written: usize,
) -> Verdict {
    let Some(goal) = goal else { return Verdict::NoGoal };
    if goal.trim().is_empty() {
        return Verdict::NoGoal;
    }
    if !is_applet {
        return Verdict::NotAnApplet;
    }
    // Agreement first, because it is the one that would quietly poison the
    // corpus with the model's own output rather than merely be useless.
    if chose == Some(verb) {
        return Verdict::Agreed;
    }
    // A request is several words of prose. One word is a label, and a line
    // carrying a path or a slash is somebody typing a command rather than
    // describing a task -- either would teach the router to key on a shape no
    // future goal has.
    if goal.split_whitespace().count() < MIN_WORDS {
        return Verdict::NotARequest;
    }
    if written >= PER_BOOT {
        return Verdict::Spent;
    }
    if known {
        return Verdict::Known;
    }
    Verdict::Learn(verb.to_string(), goal.trim().to_string())
}

/// The last request, the agent's answer to it, and what this boot has learned.
struct State {
    goal: Option<String>,
    chose: Option<String>,
    written: usize,
    /// What was learned, for the `learn` verb to show. Bounded by `PER_BOOT`.
    rows: Vec<(String, String)>,
    on: bool,
}

static STATE: crate::sync::Spin<Option<State>> = crate::sync::Spin::new(None);

fn with<T>(f: impl FnOnce(&mut State) -> T) -> T {
    let mut g = STATE.lock_irq();
    if g.is_none() {
        *g = Some(State { goal: None, chose: None, written: 0, rows: Vec::new(), on: true });
    }
    f(g.as_mut().unwrap())
}

/// Switched on, or not. Off means notice nothing and write nothing.
pub fn enabled() -> bool {
    with(|s| s.on)
}

pub fn set_enabled(on: bool) {
    with(|s| s.on = on);
}

/// The operator asked for something. Clears the previous answer with it.
pub fn goal_given(goal: &str) {
    with(|s| {
        if !s.on {
            return;
        }
        s.goal = Some(goal.to_string());
        s.chose = None;
    });
}

/// The agent acted on the live goal.
///
/// Only the *first* choice of an episode is kept. A later step is a step taken
/// in a world the earlier ones changed, so the operator's correction is about
/// where the episode went first -- and keeping the last would label the goal
/// with whatever it happened to end on.
pub fn agent_chose(applet: &str) {
    with(|s| {
        if s.on && s.chose.is_none() {
            s.chose = Some(applet.to_string());
        }
    });
}

/// What was learned this boot.
pub fn rows() -> Vec<(String, String)> {
    with(|s| s.rows.clone())
}

/// The live request and the agent's answer to it, for the `learn` verb.
pub fn pending() -> (Option<String>, Option<String>) {
    with(|s| (s.goal.clone(), s.chose.clone()))
}

/// Forget the live request, so the next command cannot be read as a correction.
pub fn clear() {
    with(|s| {
        s.goal = None;
        s.chose = None;
    });
}

/// The operator typed something. Learn from it if it corrects the agent.
///
/// Answers the verdict so a caller can print it. The row is written here rather
/// than by the caller, because the corpus append and the per-boot count have to
/// move together -- two callers keeping their own count is the
/// `axis_counts`-in-its-own-file objection.
pub fn operator_ran(verb: &str) -> Verdict {
    if !enabled() {
        return Verdict::NoGoal;
    }
    let (goal, chose) = pending();
    let is_applet = crate::sysbox::is_applet(verb);
    // Asked before the lock is taken again: `known` walks the corpus, which
    // reads the namespace, and holding this module's lock across that is a lock
    // order nothing else here uses.
    let known = goal
        .as_deref()
        .map(|g| already(verb, g))
        .unwrap_or(false);
    let written = with(|s| s.written);
    let v = judge(goal.as_deref(), chose.as_deref(), verb, is_applet, known, written);
    if let Verdict::Learn(applet, task) = &v {
        if crate::ai::vocab::record(applet, task) {
            with(|s| {
                s.written += 1;
                s.rows.push((applet.clone(), task.clone()));
                // The goal is spent: a second command after one correction is
                // the operator carrying on, not correcting again.
                s.goal = None;
                s.chose = None;
            });
        }
    }
    v
}

/// Does the corpus already pair this applet with this task?
fn already(applet: &str, task: &str) -> bool {
    crate::ai::vocab::examples()
        .iter()
        .any(|e| e.applet == applet && e.task == task)
}

pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();

    // The four refusals, each for its own reason, so a failure names which.
    out.push((
        "with nothing asked, a command is somebody using their computer",
        judge(None, None, "ls", true, false, 0) == Verdict::NoGoal,
    ));
    out.push((
        "an operator verb that is not an applet cannot be a routing label",
        judge(Some("show me the files in /ai"), Some("find"), "diag", false, false, 0)
            == Verdict::NotAnApplet,
    ));
    // The one that matters: this is the role-adapter result, and learning from
    // agreement is how it was earned.
    out.push((
        "agreement teaches nothing, because the label would be the model's own",
        judge(Some("show me the files in /ai"), Some("ls"), "ls", true, false, 0)
            == Verdict::Agreed,
    ));
    out.push((
        "a one-word goal is a label rather than a request",
        judge(Some("files"), Some("find"), "ls", true, false, 0) == Verdict::NotARequest,
    ));
    out.push((
        "and a pair the corpus already holds is not learned twice",
        judge(Some("show me the files in /ai"), Some("find"), "ls", true, true, 0)
            == Verdict::Known,
    ));
    out.push((
        "past the per-boot cap it stops, so the corpus cannot move all session",
        judge(Some("show me the files in /ai"), Some("find"), "ls", true, false, PER_BOOT)
            == Verdict::Spent,
    ));

    // And the one case that learns, which is the whole point.
    out.push((
        "a disagreement after a request is learned, with the operator's verb as the label",
        judge(Some("show me the files in /ai"), Some("find"), "ls", true, false, 0)
            == Verdict::Learn(String::from("ls"), String::from("show me the files in /ai")),
    ));
    // A correction with no agent answer at all is still a correction: the
    // episode may have ended without dispatching anything, and "it did nothing
    // and I did this" is as informative as "it did the wrong thing".
    out.push((
        "a request the agent answered with nothing is still corrected by a command",
        matches!(
            judge(Some("show me the files in /ai"), None, "ls", true, false, 0),
            Verdict::Learn(_, _)
        ),
    ));
    // The cap is checked before the corpus is consulted, so a spent boot does
    // not walk every example to answer a question it has already decided.
    out.push((
        "the cap outranks everything after it",
        judge(Some("show me the files in /ai"), Some("find"), "ls", true, true, PER_BOOT)
            == Verdict::Spent,
    ));
    out.push((
        "an empty goal is no goal, not a short one",
        judge(Some("   "), Some("find"), "ls", true, false, 0) == Verdict::NoGoal,
    ));
    out
}
