//! Judging a skill the machine wrote for itself.
//!
//! `agent learn` compiles a successful episode into an Aiksi program under
//! `/ai/tools` -- a replay of the applets that worked. Writing it *was*
//! adopting it: the file appeared, `run` would execute it, and nothing had
//! asked whether it was any good. Every other change the machine makes to
//! itself passes judges and lands in the lineage; this one did not.
//!
//! ### What can honestly be judged, and what cannot
//!
//! A replay skill takes no arguments and dispatches a fixed sequence. So
//! "does it work on a task it has not seen" is not a question it can answer --
//! it has no place to put a different task. A judge for that would be a judge
//! that cannot fail on the skills this system actually produces, which is
//! worse than not having one, so it is absent and this paragraph is why.
//!
//! What is left is admission, not improvement, and the four here are all
//! falsifiable:
//!
//!   J1  it is a program        -- lexes and parses, and has a body
//!   J2  it runs sandboxed      -- completes under `Caps::Sandbox` and the
//!                                 step budget, without error
//!   J3  it is deterministic    -- two runs touch the same objects and answer
//!                                 the same thing
//!   J4  it is cheap            -- well under the budget, not scraping it
//!
//! J3 is the one that earns its place. Everything downstream -- re-judging a
//! skill later, comparing two, believing a ledger line from last week --
//! rests on a skill behaving the same way twice, and a program that reads
//! `ticks()` or `rtc_now()` does not. It is the same argument `voter` makes
//! for building a fresh interpreter per vote, applied one level up.
//!
//! ### Why the sandbox is the judge rather than the gate
//!
//! `cmd_run` already refuses operator powers to an untrusted skill. J2 is not
//! that check again: it asks whether the skill *works* under those
//! restrictions. A replay compiled from an episode the operator drove can
//! easily depend on a mutating applet, which a sandbox refuses -- so the skill
//! is real, parses, and is useless to anyone but the operator who made it.
//! Adopting that would put a program in the machine's toolkit that fails every
//! time it is reached.

use crate::store::sha256;
use crate::sysbox;
use alloc::string::String;
use alloc::vec::Vec;

/// Where candidates are kept, by content address. `/ai/tools` holds the ones
/// that were adopted, which is the difference between a proposal and a tool.
pub const ROOT: &str = "/ai/skills";

/// Steps one judged run may take. Matches the ceiling `cmd_run` gives an
/// untrusted skill, because that is the budget it will actually have.
const BUDGET: u64 = 5_000_000;

/// What a skill may spend and still be worth adopting.
///
/// A fifth of the ceiling, for the reason the core judge uses a quarter of
/// its own: the budget stops a runaway, and this is the far lower bar a thing
/// has to clear to be worth having. A skill that needs most of its allowance
/// on a good day has none left for a bad one.
const STEP_CEILING: u64 = BUDGET / 5;

pub struct Verdict {
    pub steps: u64,
    /// Objects the run touched, as `Shadow` reports them.
    pub touched: usize,
    pub j1: bool,
    pub j1_why: &'static str,
    pub j2: bool,
    pub j2_why: &'static str,
    pub j3: bool,
    pub j3_why: &'static str,
    pub j4: bool,
}

impl Verdict {
    pub fn passed(&self) -> bool {
        self.j1 && self.j2 && self.j3 && self.j4
    }
}

fn path_of(h: &[u8; 32]) -> String {
    let mut p = String::from(ROOT);
    p.push('/');
    p.push_str(&crate::ai::voter::hex(h));
    p
}

/// Store a candidate and answer its address. Storing is not adopting.
pub fn store(src: &str) -> [u8; 32] {
    let h = sha256::hash(src.as_bytes());
    let p = path_of(&h);
    if sysbox::read_blob(&p).is_none() {
        sysbox::write_text(&p, src);
    }
    h
}

pub fn source(h: &[u8; 32]) -> Option<String> {
    sysbox::read_blob(&path_of(h)).map(|b| String::from_utf8_lossy(&b).into_owned())
}

/// Where a candidate is allowed to write while it is being judged.
///
/// Its own address, so two candidates under judgement cannot reach each
/// other's scratch and a judged run cannot leave anything an adopted run
/// would later read as its own.
fn jail(h: &[u8; 32]) -> String {
    let mut p = String::from("/ai/tools/scratch/judging/");
    p.push_str(&crate::ai::voter::hex(h)[..16]);
    p
}

/// Everything one run of a candidate is judged on.
///
/// A struct rather than a tuple because `said` is the fourth field and a
/// four-tuple destructured at two call sites is where a judge starts
/// comparing the wrong thing against the wrong thing.
struct Run {
    out: Result<String, String>,
    steps: u64,
    touched: usize,
    /// What it printed. **The half J3 could not see.**
    said: String,
}

/// Run a candidate once, under the restrictions it will really have, and
/// report what it did.
///
/// Inside a shadow on both runs, so a candidate that writes cannot leave the
/// namespace changed by having been *considered*. That is not politeness: the
/// second run has to start from the same tree as the first or J3 measures the
/// order of the runs rather than the program.
fn once(h: &[u8; 32], src: &str) -> Run {
    let mut r = Run {
        out: Err(String::from("no namespace")),
        steps: 0,
        touched: 0,
        said: String::new(),
    };
    if let Some(sh) = sysbox::shadow(|| {
        let mut it = crate::aiksi::Interp::sandboxed(&jail(h)).with_step_budget(BUDGET);
        // The capture is a stack, so this nests inside a caller that is
        // already capturing -- which it will be, since `skill judge` is
        // reachable from an episode whose observation is a capture.
        // `differ` closed the identical hole and leans on the same property.
        //
        // It spans the whole call rather than only the success path: a
        // program that printed three lines and then failed printed three
        // lines, and a judge that took the console only when the run
        // succeeded would compare an empty string against an empty string
        // for every candidate that errors.
        crate::gfx::console::begin_capture();
        r.out = match crate::aiksi::eval_line(&mut it, src) {
            Ok(v) => Ok(v.render()),
            Err(e) => Err(e),
        };
        r.said = crate::gfx::console::end_capture().unwrap_or_default();
        r.steps = it.steps();
    }) {
        r.touched = sh.changes as usize;
        sh.discard();
    }
    r
}

/// Judge a stored candidate.
pub fn bench(h: &[u8; 32]) -> Verdict {
    let mut v = Verdict {
        steps: 0,
        touched: 0,
        j1: false,
        j1_why: "no such skill",
        j2: false,
        j2_why: "not run",
        j3: false,
        j3_why: "not run",
        j4: false,
    };
    let Some(src) = source(h) else { return v };

    // --- J1: is it a program? -------------------------------------------
    match crate::aiksi::lex::lex(&src).and_then(crate::aiksi::parse::parse) {
        Err(_) => {
            v.j1_why = "it does not parse";
            return v;
        }
        Ok(prog) if prog.is_empty() => {
            v.j1_why = "it parses to nothing";
            return v;
        }
        Ok(_) => {
            v.j1 = true;
            v.j1_why = "a program";
        }
    }

    // --- J2: does it run under the powers it will have? ------------------
    let first = once(h, &src);
    v.steps = first.steps;
    v.touched = first.touched;
    match &first.out {
        Err(_) => {
            // Deliberately not repeated as "the sandbox refused it". The
            // interpreter answers the same way for a program that reached for
            // the network and one that indexed past the end of a list, and
            // guessing which would be inventing a distinction the error does
            // not carry.
            v.j2_why = "it fails under the powers an unadopted skill has";
            return v;
        }
        Ok(_) => {
            v.j2 = true;
            v.j2_why = "runs sandboxed";
        }
    }

    // --- J3: does it do the same thing twice? ---------------------------
    //
    // **What this compares.** Two runs must agree on the value the program
    // answered, the steps it took, the objects it touched, and what it
    // printed.
    //
    // The last of those was missing, and it was the one that mattered most
    // here. This said "closing it needs a capturing console, which does not
    // exist" -- and `gfx::console::begin_capture` had existed the whole time,
    // used by `applet`, by the agent's observations and by `differ`, which
    // closed the identical hole in its own comparison. The gap was not a
    // missing mechanism, it was a claim about the tree that had gone stale.
    //
    // It is load-bearing rather than thorough: **a replay skill is a sequence
    // of `println(applet(..))`, and `println` answers nil however the applets
    // behaved**, so a replay whose output varies agreed on value, steps and
    // touched objects and passed this judge. That is the shape of every skill
    // `agent learn` writes. The claim in `selftest` was itself written wrongly
    // for the same reason and passed until the program was changed to answer
    // the clock rather than print it; both versions are claims now.
    let second = once(h, &src);
    v.j3 = first.out == second.out
        && first.steps == second.steps
        && first.touched == second.touched
        && first.said == second.said;
    v.j3_why = if v.j3 {
        "the same twice"
    } else {
        "it does not repeat -- a clock or a counter is in it"
    };

    // --- J4: can it be afforded on a bad day? ---------------------------
    v.j4 = first.steps <= STEP_CEILING;

    v
}

/// Where an adopted skill goes, so `run` can find it by a name a person
/// would type.
///
/// Named from the candidate's address rather than from anything the program
/// says about itself, because a program can say anything. The operator can
/// rename it afterwards; that changes its hash and so its identity, which is
/// correct -- a renamed skill has not been judged.
pub fn adopted_path(h: &[u8; 32]) -> String {
    let mut p = String::from("/ai/tools/learned-");
    p.push_str(&crate::ai::voter::hex(h)[..8]);
    p.push_str(".ai&xi");
    p
}

/// The mechanism, without an episode.
///
/// Every claim here is about a program written inline, because the skills
/// this judges come from a model and a corpus and neither is present when
/// the boot tests run. What is checkable is that each judge can actually
/// fail, which is the property a judge that never vetoes quietly loses.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };

    if sysbox::hash_of("/").is_none() {
        kprintln!("  --   skills need a namespace to be judged (`diag skill`)");
        return ok;
    }

    let good = store("println(1 + 1)\n");
    let v = bench(&good);
    claim("a plain program passes every judge", v.passed());

    let broken = store("fn (\n");
    let v = bench(&broken);
    claim("one that does not parse is refused by J1", !v.j1 && !v.passed());

    // `peek8` is an operator power, so the sandbox refuses the call and the
    // program fails. That is the shape of a replay compiled from an episode
    // the operator drove, which is the case this judge exists for.
    let reaching = store("println(peek8(1048576))\n");
    let v = bench(&reaching);
    claim("one that needs operator powers is refused by J2", v.j1 && !v.j2);

    // A clock makes two runs disagree, and everything downstream assumes they
    // would not.
    //
    // The program is `ticks()` and not `println(ticks())`, which is what this
    // claim was first written as and which passed J3 wrongly: printing is a
    // side effect and `println` answers nil, so both runs returned the same
    // value while the console showed 4412 and 4415. See the note on J3's
    // blind spot at `bench`.
    // `tsc()` and not `ticks()`, which was the second wrong version of this
    // claim: the tick counter runs at 100 Hz and two runs a few microseconds
    // apart land inside the same tick, so it answered identically and J3 was
    // right to say so. Worth knowing beyond the test -- a skill whose only
    // nondeterminism is a coarse clock will pass J3 on a fast machine and
    // diverge later, which is the limit of judging repeatability by repeating.
    let restless = store("tsc()\n");
    let v = bench(&restless);
    claim("one that will not repeat is refused by J3", v.j1 && v.j2 && !v.j3);

    // **The first wrong version of that claim, kept as a claim.** This program
    // answers nil twice, spends the same steps twice and touches nothing
    // twice, so every field J3 compared before the console was among them is
    // identical -- and it printed a different number each run. It passed.
    //
    // It is not a corner case: it is exactly the shape of a replay compiled
    // from an episode, `println` around an applet call, which is what
    // `agent learn` writes and what this whole judge exists to admit or
    // refuse. Nothing else in the suite can fail if the capture is dropped.
    let noisy = store("println(tsc())\n");
    let v = bench(&noisy);
    claim(
        "and one whose only difference is what it printed is refused too",
        v.j1 && v.j2 && !v.j3,
    );

    // A judge that cannot veto is not a judge: prove J4 has a real ceiling by
    // spending past it *and finishing*.
    //
    // The size matters and the first version got it wrong in an instructive
    // way. Two million iterations do not test J4 at all -- they exhaust the
    // interpreter's budget, the run errors, and J2 refuses it first. J4 is
    // about a program that completes and is still too expensive to keep, so
    // the loop has to land between the ceiling and the budget.
    let greedy = store("i = 0\nwhile (i < 400000) { i = i + 1 }\ni\n");
    let v = bench(&greedy);
    claim("one that is expensive but finishes is refused by J4", v.j1 && v.j2 && !v.j4);
    if v.j4 || !v.j2 {
        kprintln!("       (it spent {} against a ceiling of {})", v.steps, STEP_CEILING);
    }

    // What `run` can be handed, which is what makes an adopted skill reachable
    // at all. Every entry must be a full path to a program, because the
    // grammar built from this list is the only thing the model may spell.
    let choices = crate::ai::agent::skill_choices();
    claim(
        "every runnable choice is a full path to a program",
        !choices.is_empty()
            && choices
                .iter()
                .all(|c| c.starts_with("/ai/tools/") && c.ends_with(".ai&xi")),
    );

    for h in [good, broken, reaching, restless, noisy, greedy] {
        sysbox::detach(&path_of(&h));
    }
    ok
}
