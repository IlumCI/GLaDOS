//! What the router actually did, kept so somebody can look at it.
//!
//! **Every one of these numbers was already computed and then thrown away.**
//! `harness::route` builds a score for every applet class out of one hidden
//! state and keeps the argmax; `route_verdict` asks two more cores and keeps
//! the winner. The scores, the reachable set, the individual votes and the
//! rule that resolved them all fell on the floor at the end of the call, so
//! the only thing anybody could see about a routing decision was its answer.
//!
//! A ring of the last few decisions costs a few kilobytes and turns that into
//! something a window can draw: what was asked, what the grammar left
//! reachable, how the candidates scored, who voted for what, and which rule
//! broke the tie.
//!
//! ### Why the names are copied in
//!
//! A decision is recorded on whichever task asked -- the shell, the agent, the
//! resident mind -- and read by the compositor, which must never block. Class
//! indices would mean the drawing code calling `with_engine` to turn them into
//! names, and `with_engine` refuses while another task holds the model, so the
//! panel would go blank exactly while the machine was busy being interesting.
//! Copying twenty-odd short strings once per decision is the cheaper half of
//! that trade by a wide margin.
//!
//! ### `Spin` and not `Racy`
//!
//! Two tasks genuinely touch this: the one routing and the one painting. That
//! is the condition `Racy` says it is not for. A decision is recorded a few
//! times a minute and read once a frame, so the lock is uncontended in
//! practice and correct in principle.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Decisions kept. Small on purpose: this is a window somebody glances at,
/// and the interesting one is nearly always the last.
pub const KEEP: usize = 10;

/// Candidates kept per decision. The head is twenty-odd classes wide and a
/// panel can show six bars legibly, so the tail is dropped at record time
/// rather than carried around to be discarded at draw time.
pub const TOPN: usize = 6;

/// One routing decision, whole.
#[derive(Clone)]
pub struct Decision {
    /// What was asked.
    pub task: String,
    /// The best few reachable candidates, highest score first.
    pub cand: Vec<Cand>,
    /// How many classes the trust gate and the grammar left reachable, out of
    /// how many exist. The gap is constrained decoding doing its work: a
    /// forbidden applet is not considered, so it cannot be returned however
    /// high it scored.
    pub allowed: usize,
    pub total: usize,
    /// The three voters, by class index, and what the rule made of them.
    pub probe: usize,
    pub lexical: usize,
    pub character: usize,
    pub winner: usize,
    /// How many of the three landed on the winner. Unanimity measured 90.3%
    /// correct against 50% when split, which is why this is the number the
    /// panel puts in the largest type.
    pub agreement: usize,
    pub rule: &'static str,
    /// False when only `route` ran, so the cores were never asked. An honest
    /// state rather than a silent one-vote decision.
    pub settled: bool,
    /// Uptime when it happened.
    pub at_s: f32,
}

#[derive(Clone)]
pub struct Cand {
    pub class: usize,
    pub name: String,
    pub score: f32,
}

impl Decision {
    /// The winner's name, if it is among the candidates kept.
    pub fn winner_name(&self) -> &str {
        self.cand
            .iter()
            .find(|c| c.class == self.winner)
            .map(|c| c.name.as_str())
            .unwrap_or("?")
    }
}

static RING: crate::sync::Spin<Vec<Decision>> = crate::sync::Spin::new(Vec::new());

/// Record what the probe saw. Called from `route`, where the scores exist.
///
/// `cand` arrives already filtered to what the trust gate admits and already
/// sorted, because the caller is the only place that knows both.
pub fn begin(task: &str, cand: Vec<Cand>, allowed: usize, total: usize) {
    let probe = cand.first().map(|c| c.class).unwrap_or(0);
    let d = Decision {
        task: task.to_string(),
        cand,
        allowed,
        total,
        probe,
        lexical: probe,
        character: probe,
        winner: probe,
        agreement: 1,
        rule: "probe only",
        settled: false,
        at_s: crate::dev::lapic::ticks() as f32 / crate::TIMER_HZ as f32,
    };
    let mut r = RING.lock_irq();
    r.push(d);
    let n = r.len();
    if n > KEEP {
        r.drain(0..n - KEEP);
    }
}

/// Complete the last record with what the council said.
///
/// Separate from `begin` because the scores and the votes are computed in
/// different functions, and recomputing either to get them in one place would
/// cost a forward pass per decision -- `feature` is a prefill, which is the
/// expensive half of routing.
pub fn settle(
    probe: usize,
    lexical: usize,
    character: usize,
    winner: usize,
    agreement: usize,
    rule: &'static str,
) {
    let mut r = RING.lock_irq();
    let Some(d) = r.last_mut() else { return };
    // Only ever amends the decision this call's own `route` just recorded. If
    // something else got in between, the record is left as the one-vote answer
    // it honestly was rather than being given somebody else's votes.
    if d.settled {
        return;
    }
    d.probe = probe;
    d.lexical = lexical;
    d.character = character;
    d.winner = winner;
    d.agreement = agreement;
    d.rule = rule;
    d.settled = true;
}

/// The decisions, oldest first.
pub fn recent() -> Vec<Decision> {
    RING.lock_irq().clone()
}

/// How the council has been getting on: counts of 3, 2 and 1 agreeing.
///
/// Over the ring rather than all time, because what a panel is for is the
/// recent past and a counter since boot would stop moving visibly after a
/// minute of use.
pub fn agreement_census() -> [usize; 3] {
    let mut out = [0usize; 3];
    for d in RING.lock_irq().iter() {
        if !d.settled {
            continue;
        }
        let i = d.agreement.clamp(1, 3) - 1;
        out[i] += 1;
    }
    out
}

pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };

    let before = recent().len();
    let cand = alloc::vec![
        Cand { class: 3, name: "ls".to_string(), score: 2.0 },
        Cand { class: 7, name: "tree".to_string(), score: 1.0 },
    ];
    begin("list the files", cand, 14, 23);
    let d = recent();
    let last = d.last().expect("just pushed");
    claim("a decision is recorded with its task", last.task == "list the files");
    // The probe's answer is the head of the sorted list, so an empty or
    // unsorted candidate list would silently make the wrong class the probe's.
    claim("and the probe's answer is the best candidate", last.probe == 3);
    claim("and it is unsettled until the cores have spoken", !last.settled);
    claim("and it carries what was reachable", last.allowed == 14 && last.total == 23);

    settle(3, 7, 3, 3, 2, "majority");
    let d = recent();
    let last = d.last().expect("still there");
    claim("settling records the votes", last.lexical == 7 && last.character == 3);
    claim("and the agreement", last.agreement == 2 && last.settled);
    // A second settle must not overwrite: it would belong to a decision this
    // one never made.
    settle(0, 0, 0, 0, 3, "probe only");
    let d = recent();
    claim("and a second settle is refused", d.last().expect("there").agreement == 2);

    // The ring is what bounds the memory, so it is worth one claim.
    for _ in 0..KEEP + 4 {
        begin("filler", Vec::new(), 1, 1);
    }
    claim("the ring never grows past its cap", recent().len() == KEEP);
    let _ = before;
    ok
}
