//! The agent loop: goal -> constrained applet choice -> validated arguments
//! -> executed tool -> observation fed back -> repeat.
//!
//! The harness module deliberately stops short of this: it proves that the
//! grammar admits exactly the applet names and that permission cannot leak,
//! against models too small to follow an instruction. What was missing is the
//! iteration -- feed the result back and go round again. This module adds
//! that edge, under three constraints inherited from the harness:
//!
//!   * The action space is compiled into a grammar from the live applet
//!     table, plus one pseudo-applet, `done`, which ends the episode. An
//!     invalid choice is unreachable rather than refused.
//!   * Under `Trust::ReadOnly` the mutating applets are absent from the
//!     grammar entirely. Permission and output validity stay the same piece
//!     of code.
//!   * Nothing is parsed on the way in. The decode ends on a real name; only
//!     the *arguments* are free text, and they are shape-checked against the
//!     applet's declared usage before dispatch. A bad argument becomes an
//!     observation the model can react to, never a fault.
//!
//! What this deliberately is not yet: asynchronous. `run` executes on the
//! calling task with a step budget, which keeps a scripted episode
//! deterministic -- there is no second task whose interleaving with the shell
//! has to be reasoned about while the loop itself is still being judged.

use super::deliberate;
use super::harness::{self, Trust};
use super::{sample, with_engine};
use crate::gfx::console::{self, LTCYAN, LTGRAY, LTGREEN, LTRED, WHITE, YELLOW};
use crate::kprintln;
use crate::sync::Racy;
use crate::sysbox;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};

/// The pseudo-applet that ends an episode. It travels in the same grammar as
/// the real names, so stopping is a choice the model makes under the same
/// constraints as acting -- not a special token smuggled past the sampler.
pub const DONE: &str = "done";

/// Longest observation fed back per step. The transcript stays bounded no
/// matter how chatty an applet is; `tree` on a deep namespace would eat the
/// context window otherwise.
const OBS_CLIP: usize = 240;

/// Longest argument string accepted from free generation.
const ARGS_CLIP_BYTES: usize = 96;

/// Tokens the model may spend on arguments before being cut off.
const ARGS_TOKEN_BUDGET: usize = 16;

/// Forked candidates explored when the reflex tier is not confident enough.
/// Three forks at 0.9 temperature cover the neighbourhood of the pulse
/// answer; the probe ranks, so the budget buys selection rather than noise.
const DELIBERATE_FORKS: usize = 3;

// --- shared episode log ---------------------------------------------------
//
// The console transcript is the serial channel's view. The desktop window
// needs the same stream without owning the console, so every event is
// appended here as well -- a ring, because an episode that never ends must
// not grow memory without bound.

static LOG: Racy<Vec<String>> = Racy::new(Vec::new());
const LOG_CAP: usize = 240;

/// The last episode's successful actions, in order -- the ratchet's raw
/// material. A trajectory that worked is a procedure; `agent learn` compiles
/// it into a replayable skill so the next episode can run it instead of
/// re-deriving it. This is procedural memory written by the agent itself.
static LAST_TRAJECTORY: Racy<Vec<String>> = Racy::new(Vec::new());
static LAST_GOAL: Racy<Option<String>> = Racy::new(None);

fn elog(line: String) {
    unsafe {
        let log = LOG.get();
        log.push(line);
        let excess = log.len().saturating_sub(LOG_CAP);
        log.drain(..excess);
    }
}

/// Snapshot for a drawing window. A copy, because the agent task appends
/// while the desktop draws; the line count is small enough not to care.
pub fn log_snapshot() -> Vec<String> {
    unsafe { LOG.get().clone() }
}

// --- abort ----------------------------------------------------------------

static ABORT: AtomicBool = AtomicBool::new(false);

/// Ask the running episode to stop at its next boundary (per step, and
/// between argument tokens). Also cancels a queued-but-not-started episode,
/// which is what `agent stop` mostly means in practice -- the queue returns
/// to the shell faster than anyone can type.
pub fn request_abort() -> &'static str {
    ABORT.store(true, Ordering::Release);
    if take_request().is_some() {
        crate::shell::reprompt();
        return "queued episode cancelled";
    }
    if episode_busy() {
        "stopping after the current step"
    } else {
        ABORT.store(false, Ordering::Release);
        "(no episode is running)"
    }
}

fn aborted() -> bool {
    ABORT.load(Ordering::Acquire)
}

// --- the request queue ----------------------------------------------------
//
// One queued episode at a time, mirroring how `think` queues one prompt.
// The task itself is spawned once at boot and lives forever, exactly like
// the mind; ownership of the engine follows the task id, which mod.rs
// records at spawn before the task can possibly run.

struct EpisodeReq {
    goal: String,
    trust: Trust,
    steps: usize,
}

static REQUEST: Racy<Option<EpisodeReq>> = Racy::new(None);

/// True while an episode is executing (as opposed to merely queued). mod.rs
/// owns the flag; the queue here only needs to know the difference for
/// `request_abort`'s message.
pub(crate) fn set_busy(on: bool) {
    BUSY.store(on, Ordering::Release);
}

static BUSY: AtomicBool = AtomicBool::new(false);

pub(crate) fn episode_busy() -> bool {
    BUSY.load(Ordering::Acquire)
}

/// Queue an episode. False when one is already pending or running -- the
/// caller turns that into a refusal, since two episodes would fight over
/// both the engine and the namespace cursor.
pub fn queue_episode(goal: &str, trust: Trust, steps: usize) -> bool {
    crate::cpu::without_interrupts(|| unsafe {
        if REQUEST.get().is_some() || episode_busy() {
            return false;
        }
        *REQUEST.get() = Some(EpisodeReq {
            goal: String::from(goal),
            trust,
            steps,
        });
        true
    })
}

fn take_request() -> Option<EpisodeReq> {
    crate::cpu::without_interrupts(|| unsafe { REQUEST.get().take() })
}

/// The resident agent task. Spawned once; never returns.
pub fn agent_task() {
    loop {
        let Some(req) = take_request() else {
            crate::task::yield_now();
            continue;
        };
        set_busy(true);
        run(&req.goal, req.trust, req.steps);
        ABORT.store(false, Ordering::Release);
        set_busy(false);
        crate::shell::reprompt();
    }
}

struct Step {
    /// Rendered action, e.g. `ls /sys` -- what the transcript and the next
    /// prompt both show.
    action: String,
    ok: bool,
    observation: String,
}

/// What the operator (or a previous episode) has put in /ai/agent and
/// /ai/tools, injected into every step prompt. This is the self-modification
/// surface made ordinary: the loop reads its own policy the way it reads the
/// applet table, and editing the files is just `write`.
struct EpisodeCtx {
    policy: String,
    notes: String,
    skills: Vec<(String, String)>,
}

const POLICY_CLIP: usize = 240;
const NOTES_CLIP: usize = 160;
const SKILLS_MAX: usize = 6;
const SKILL_DESC_CLIP: usize = 48;

impl EpisodeCtx {
    fn load() -> Self {
        let read = |p: &str, cap: usize| {
            crate::sysbox::read_blob(p)
                .map(|b| String::from_utf8_lossy(&b).into_owned())
                .map(|t| clip(t.trim(), cap))
                .unwrap_or_default()
        };
        let mut skills = crate::sysbox::skills();
        skills.truncate(SKILLS_MAX);
        for (_, d) in skills.iter_mut() {
            *d = clip(d, SKILL_DESC_CLIP);
        }
        Self {
            policy: read("/ai/agent/policy", POLICY_CLIP),
            notes: read("/ai/agent/notes", NOTES_CLIP),
            skills,
        }
    }
}

fn clip(s: &str, max: usize) -> String {
    if s.len() <= max {
        return String::from(s);
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = String::from(&s[..end]);
    out.push_str("...");
    out
}

/// The names the grammar is built from live in `harness::admitted`; the only
/// thing this module adds is the sentinel below.

/// What the model sees each step. Kept short on purpose: history is the last
/// few steps with clipped observations, because a prompt that grows with the
/// episode would spend the context window retelling it and leave nothing for
/// thinking.
fn prompt_for(goal: &str, steps: &[Step], ctx: &EpisodeCtx, names: &[&str]) -> String {
    let mut s = String::from("Goal: ");
    s.push_str(goal);
    s.push('\n');
    if !ctx.policy.is_empty() {
        s.push_str("Policy: ");
        s.push_str(&ctx.policy);
        s.push('\n');
    }
    if !ctx.notes.is_empty() {
        s.push_str("Notes: ");
        s.push_str(&ctx.notes);
        s.push('\n');
    }
    if !ctx.skills.is_empty() {
        s.push_str("Skills:");
        for (name, desc) in &ctx.skills {
            s.push_str("\n- /ai/tools/");
            s.push_str(name);
            s.push_str(" -- ");
            s.push_str(desc);
        }
        s.push('\n');
    }
    if steps.is_empty() {
        s.push_str("(nothing done yet)\n");
    } else {
        for st in steps.iter().rev().take(3).rev() {
            s.push_str("- ");
            s.push_str(&st.action);
            s.push_str(" -> ");
            let mark = if st.ok { "" } else { "(rejected) " };
            s.push_str(mark);
            s.push_str(&clip(&st.observation, 100));
            s.push('\n');
        }
    }
    s.push_str("Tools:");
    for (i, n) in names.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push(' ');
        s.push_str(n);
    }
    s.push_str(". Next tool:");
    s
}

/// One decision, through the deliberation controller. The tiers live in
/// `deliberate::decide` -- reflex, pulse, fork-and-rank -- and this wrapper
/// owns what surrounds a choice: the step prompt, argument generation, and
/// the state contract that argument generation depends on.
///
/// Arguments are free text, greedy, continuing the cache the chosen branch
/// left behind -- a routed choice with required arguments gets a two-line
/// prompt instead, since a reflex has no branch context to continue.
fn propose(goal: &str, steps: &[Step], ctx: &EpisodeCtx, trust: Trust) -> Option<(String, String)> {
    let names = harness::admitted(trust);
    let prompt_of = || {
        let mut all = names.clone();
        all.push(DONE);
        prompt_for(goal, steps, ctx, &all)
    };
        let _alphabet_guard = 0; // alphabet is captured by deliberate's own with_alphabet

    let decision = deliberate::decide(goal, &prompt_of, trust, DELIBERATE_FORKS)?;
    let name = decision.applet;
    console::set_color(LTGRAY);
    kprintln!("     (tier: {})", decision.tier.label());
    console::set_color(WHITE);

    if name == DONE {
        return Some((name, String::new()));
    }

    // Reflex choices carry no branch context; everything else left the
    // engine positioned right after the chosen name's tokens.
    let from_context = decision.tier != deliberate::Tier::Reflex;
    if !from_context && sysbox::check_args(&name, "").is_ok() {
        return Some((name, String::new()));
    }

    let args = harness::with_alphabet(|_alphabet| {
        with_engine(|e| {
            let mut raw: Vec<u8> = Vec::new();
            let eos = e.tok.eos();
            let limit = e.model.cfg.seq_len;
            let mut pos = e.pos;
            if !from_context {
                // Targeted prompt: the routed name is known, only its
                // arguments are wanted.
                let p = format!("Task: {}\n{} ", goal, name);
                let tokens = e.tok.encode(&p, true, false);
                pos = e.model.prefill(&mut e.state, &tokens, pos);
            }
            for _ in 0..ARGS_TOKEN_BUDGET {
                if pos >= limit || raw.len() >= ARGS_CLIP_BYTES {
                    break;
                }
                let vocab = e.tok.vocab_size();
                let all: Vec<u32> = (0..vocab as u32).collect();
                let next =
                    sample::sample_among(&e.state.logits, &all, 0.0, 0.0, &mut e.rng)?;
                if next == eos {
                    break;
                }
                let piece = e.tok.token_bytes(next).to_vec();
                let nl = piece.iter().position(|&b| b == b'\n');
                raw.extend_from_slice(&piece[..nl.unwrap_or(piece.len())]);
                let done = nl.is_some() || raw.len() >= ARGS_CLIP_BYTES;
                e.model.forward(&mut e.state, next, pos);
                pos += 1;
                if done {
                    break;
                }
            }
            harness::invalidate_conversation(e);
            Some(String::from_utf8_lossy(&raw).into_owned())
        })
    });
    let args = match args {
        // Same three layers as every with_alphabet/with_engine call: the
        // closure's Option plus one wrapper each. The innermost is what the
        // greedy argument walk produced.
        Some(Some(Some(a))) => a,
        _ => String::new(),
    };
    Some((name, args))
}

/// Run one episode to completion: bounded steps, live printing, transcript
/// written to /ai/episodes/. Executes on the resident agent task -- the
/// shell returned to its caller the moment this episode was queued.
pub fn run(goal: &str, trust: Trust, max_steps: usize) {
    console::set_color(YELLOW);
    kprintln!("[agent]");
    console::set_color(LTGRAY);
    kprintln!(
        "  goal: {}   trust: {}   budget: {} steps",
        goal,
        match trust {
            Trust::ReadOnly => "read-only",
            Trust::Full => "full",
        },
        max_steps
    );
    elog(format!("[agent] goal: {}", goal));

    if with_engine(|_| ()).is_none() {
        console::set_color(LTRED);
        kprintln!("  no model loaded");
        console::set_color(LTGRAY);
        elog(String::from("no model loaded"));
        return;
    }

    // Policy, notes and skills are loaded once per episode. Logged so an
    // injection is visible in the transcript rather than assumed -- a prompt
    // nobody can see is a prompt nobody can debug.
    let ctx = EpisodeCtx::load();
    if !ctx.policy.is_empty() {
        console::set_color(LTCYAN);
        kprintln!("  policy: {}", ctx.policy);
        console::set_color(LTGRAY);
        elog(format!("policy loaded ({} chars)", ctx.policy.len()));
    }
    if !ctx.notes.is_empty() {
        console::set_color(LTCYAN);
        kprintln!("  notes: {}", ctx.notes);
        console::set_color(LTGRAY);
        elog(format!("notes loaded ({} chars)", ctx.notes.len()));
    }
    if !ctx.skills.is_empty() {
        console::set_color(LTCYAN);
        kprintln!("  skills: {} available (run <path>)", ctx.skills.len());
        console::set_color(LTGRAY);
        elog(format!("skills listed ({})", ctx.skills.len()));
    }

    let (outcome, steps) = episode(goal, trust, max_steps, None, false);

    console::set_color(YELLOW);
    kprintln!("  [agent] {} after {} step(s)", outcome, steps.len());
    console::set_color(LTGRAY);
    elog(format!("-- {} after {} step(s) --", outcome, steps.len()));

    // Record the trajectory for `agent learn`: every step that executed
    // cleanly, in order. A failed step is not procedure, it is a note.
    unsafe {
        let t = LAST_TRAJECTORY.get();
        t.clear();
        for s in steps.iter().filter(|s| s.ok) {
            t.push(s.action.clone());
        }
        *LAST_GOAL.get() = Some(String::from(goal));
    }

    // Transcript into the namespace. Content-addressed like everything else:
    // an episode can be hashed, diffed against its siblings, and snapshotted
    // without any of it being special-cased.
    let report = render(goal, &outcome, &steps);
    let idx = sysbox::children("/ai/episodes").len() + 1;
    let path = format!("/ai/episodes/{:04}.txt", idx);
    if sysbox::write_text(&path, &report) {
        kprintln!("  transcript at {}", path);
        elog(format!("transcript at {}", path));
    }
    crate::gfx::desk::draw();
}

/// The loop itself, factored out of `run` so it can be driven two ways:
/// actions sampled by the model, or actions read from a script. A scripted
/// episode exercises every mechanical organ -- admission, argument
/// validation, dispatch, observation capture, budgets, abort -- with no
/// forward passes at all, which is what makes it a boot selftest rather
/// than a twenty-minute QEMU vigil.
///
/// `quiet` suppresses per-step printing and the transcript: the selftest
/// asserts on returned values and should not pollute /ai/episodes on every
/// boot.
fn episode(
    goal: &str,
    trust: Trust,
    max_steps: usize,
    script: Option<&[String]>,
    quiet: bool,
) -> (String, Vec<Step>) {
    let _ = goal;
    let ctx = EpisodeCtx::load();
    let mut steps: Vec<Step> = Vec::new();
    let mut outcome = "step budget reached";

    for i in 0..max_steps {
        // Checked at every step boundary and between argument tokens; an
        // abort lands within one step of the request, never mid-applet.
        if aborted() {
            outcome = "aborted by operator";
            break;
        }

        // Two sources of action. The sampled path can only name what the
        // grammar admits; the scripted path bypasses the sampler, so the
        // admission rule is applied by hand below -- same list, same rule,
        // or a script could reach what a model never could.
        let picked = match script {
            Some(lines) => lines.get(i).map(|line| {
                let (a, r) = match line.split_once(' ') {
                    Some((a, r)) => (a.trim(), r.trim()),
                    None => (line.trim(), ""),
                };
                (String::from(a), String::from(r))
            }),
            None => propose(goal, &steps, &ctx, trust),
        };
        let Some((name, args)) = picked else {
            outcome = if script.is_some() { "script exhausted" } else { "decode did not settle" };
            break;
        };

        if name == DONE {
            outcome = "model called done";
            if !quiet {
                console::set_color(LTGREEN);
                kprintln!("  {}. done", i + 1);
                console::set_color(LTGRAY);
                elog(format!("{}. done", i + 1));
            }
            break;
        }

        // Shape-check before dispatch. Rejection is an observation, not an
        // error: the model gets to read why and choose differently.
        let admitted = script.is_none()
            || harness::admitted(trust).iter().any(|n| *n == name);
        let checked = if !admitted {
            Err(format!(
                "'{}' is not reachable at this trust level",
                name
            ))
        } else {
            sysbox::check_args(&name, &args)
        };
        let (ok, observation) = match checked {
            Err(why) => (false, format!("invalid arguments: {}", why)),
            Ok(()) => {
                console::begin_capture();
                let ran = sysbox::dispatch(&name, &args);
                let mut obs = console::end_capture().unwrap_or_default();
                if !ran && obs.is_empty() {
                    obs = String::from("(applet did not run)");
                }
                (true, obs)
            }
        };

        let action = if args.is_empty() {
            String::from(name)
        } else {
            format!("{} {}", name, args)
        };
        if !quiet {
            console::set_color(if ok { LTCYAN } else { LTRED });
            kprintln!("  {}. {}", i + 1, action);
            console::set_color(LTGRAY);
            elog(format!("{}. {}{}", i + 1, if ok { "" } else { "[rejected] " }, action));
            for line in observation.lines().take(4) {
                kprintln!("     | {}", line);
            }
            for line in clip(observation.lines().take(4).collect::<Vec<_>>().join(" / ").as_str(), 110).lines() {
                elog(format!("   | {}", line));
            }
            // The window repaints through the diffed present, so refreshing
            // per step costs only what actually changed.
            crate::gfx::desk::draw();
        }

        steps.push(Step {
            action,
            ok,
            observation: clip(&observation, OBS_CLIP),
        });
    }

    (String::from(outcome), steps)
}

/// The loop's mechanical properties, checked at every boot with no model in
/// the way. Each line exists because its absence would be a silent failure:
/// a read-only applet really runs and its output lands in the observation;
/// a mutating applet is unreachable under ReadOnly trust even when named
/// outright by a script; bad arguments are rejected as observations rather
/// than dispatched; and `done` ends the episode inside the budget.
pub fn selftest() -> bool {
    let script = alloc::vec![
        String::from("ls /sys"),
        String::from("rm /sys/readme"),
        String::from("cat"),
        String::from("done"),
    ];
    let (outcome, steps) = episode("boot selftest", Trust::ReadOnly, 8, Some(&script), true);

    let mut ok = true;
    let mut check = |what: &str, pass: bool| {
        console::set_color(if pass { LTGREEN } else { LTRED });
        kprintln!("  {}  {}", if pass { "ok  " } else { "FAIL" }, what);
        console::set_color(LTGRAY);
        ok &= pass;
    };

    check("read-only applet ran, output captured as the observation", {
        steps.first().map(|s| s.ok && s.observation.contains("readme")).unwrap_or(false)
    });
    check("mutating applet named outright is still unreachable", {
        steps.get(1).map(|s| !s.ok && s.observation.contains("not reachable")).unwrap_or(false)
    });
    check("bad arguments rejected before dispatch", {
        steps.get(2).map(|s| !s.ok && s.observation.contains("invalid arguments")).unwrap_or(false)
    });
    check("done ends the episode", {
        outcome == "model called done" && steps.len() == 3
    });

    ok
}

/// Compile the last successful trajectory into a replayable skill.
///
/// The ratchet, mechanically: an episode that worked becomes a lang program
/// whose lines call the same applets with the same arguments, via the
/// `applet` builtin. Running it reproduces the procedure -- and because it
/// is a program, the next run can read its composed output, branch on it,
/// or hand it to another tool. Returns the skill's path, or the reason
/// there is nothing to learn from.
pub fn learn(name: Option<&str>) -> Result<String, &'static str> {
    let (goal, actions) = unsafe {
        let g = LAST_GOAL.get().clone();
        let a = LAST_TRAJECTORY.get().clone();
        (g, a)
    };
    if actions.is_empty() {
        return Err("no successful trajectory to learn -- run an episode first");
    }
    let mut program = String::from("// replay: ");
    program.push_str(goal.as_deref().unwrap_or("(unnamed goal)"));
    program.push_str("\n// compiled by 'agent learn' from a successful episode\n");
    for a in &actions {
        program.push_str(&format!("println(applet(\"{}\"))\n", a.replace('"', "\\\"")));
    }

    let idx = crate::sysbox::children("/ai/tools")
        .iter()
        .filter(|n| n.starts_with("replay-"))
        .count()
        + 1;
    let skill = match name {
        Some(n) => format!("/ai/tools/{}.l", n.trim().replace(".l", "")),
        None => format!("/ai/tools/replay-{:04}.l", idx),
    };
    if crate::sysbox::write_text(&skill, &program) {
        Ok(skill)
    } else {
        Err("could not write the skill")
    }
}

fn render(goal: &str, outcome: &str, steps: &[Step]) -> String {
    let mut s = format!("goal: {}\noutcome: {}\nsteps: {}\n\n", goal, outcome, steps.len());
    for (i, st) in steps.iter().enumerate() {
        s.push_str(&format!("[{}] {}\n", i + 1, st.action));
        s.push_str(&format!("    ok: {}\n", st.ok));
        for line in st.observation.lines() {
            s.push_str(&format!("    | {}\n", line));
        }
    }
    s
}



