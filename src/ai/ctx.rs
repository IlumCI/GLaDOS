//! Contexts: one set of weights, several minds' worth of working memory.
//!
//! There used to be one `Engine` and one holder, which made the model a
//! single-occupancy room: an agent episode held it for its whole length, the
//! nightly `godel` trial for twenty seconds in one call, and `ask` during
//! either answered nothing at all. What is genuinely exclusive was never the
//! model. Weights are read-only during a forward pass. What two callers
//! cannot share is a **`State`** -- the KV cache, `pos`, `last_token`, the
//! logits -- and an attached adapter, which `godel` swaps in and out while it
//! judges.
//!
//! So each context is a whole `Engine` over **shared weights**
//! (`Model::share`, an `Arc` per buffer) with its own state and its own
//! adapter. Three of them, a closed table in the `repair::ACTIONS` idiom:
//!
//! - `Chat` -- the shell and the resident mind. They share one cache on
//!   purpose, because an interjection is a turn in the same conversation.
//! - `Agent` -- the agent task's episodes and authoring runs.
//! - `Judge` -- unattended trials, so a candidate adapter never reaches the
//!   conversation while it is being judged.
//!
//! Each context has its own holder. The UB argument the single holder made --
//! two `&mut Engine` at once -- still holds per context and is still enforced
//! by a claim per call; the KV-corruption argument is simply gone, because
//! nobody else writes your cache.
//!
//! **What must agree across contexts is published, not shared.** The router
//! (head, probe, council) and the adopted adapter are the machine's learned
//! state, and `fit` in the shell has to reach the agent's next episode.
//! Sharing them by reference would put a `&mut` in one context beside a `&`
//! in another, which is the UB this module exists to remove. So each context
//! keeps its own copy, a change is detected by a counter at the end of the
//! outermost borrow (`Engine::router_epoch`, `Model::adapter_epoch`), and the
//! changed half is copied into `CANON`; a context that starts a borrow behind
//! `CANON` copies it back. Per half, so a `fit` in `Chat` and an adoption in
//! `Judge` landing in one minute cannot clobber each other. Both copies are
//! small or rare: the router is kilobytes, and an adapter moves when somebody
//! adopts one.
//!
//! A missed epoch bump is a stale router in another context, never undefined
//! behaviour -- the failure is the one the selftest looks for, rather than one
//! nothing can see.

use super::{adapter, council, probe, vocab, Engine};
use crate::sync::Spin;
use core::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ctx {
    Chat = 0,
    Agent = 1,
    Judge = 2,
}

pub const COUNT: usize = 3;

impl Ctx {
    pub const ALL: [Ctx; COUNT] = [Ctx::Chat, Ctx::Agent, Ctx::Judge];

    pub fn name(self) -> &'static str {
        match self {
            Ctx::Chat => "chat",
            Ctx::Agent => "agent",
            Ctx::Judge => "judge",
        }
    }

    fn from_u8(v: u8) -> Ctx {
        match v {
            1 => Ctx::Agent,
            2 => Ctx::Judge,
            _ => Ctx::Chat,
        }
    }
}

/// Which context each task borrows, by task id. Zero is `Chat`, so a task
/// nobody has assigned behaves exactly as every task did before contexts
/// existed. Sized well past `task::MAX_TASKS`; an id beyond it is `Chat`.
const SLOTS: usize = 64;
static TASK_CTX: [AtomicU8; SLOTS] = [const { AtomicU8::new(0) }; SLOTS];

/// The context the running task borrows.
pub fn current() -> Ctx {
    let t = crate::task::current();
    if t < SLOTS {
        Ctx::from_u8(TASK_CTX[t].load(Ordering::Acquire))
    } else {
        Ctx::Chat
    }
}

/// Assign the running task a context for the rest of its life. The agent
/// task does this once, at the top of its loop.
pub fn adopt(c: Ctx) {
    let t = crate::task::current();
    if t < SLOTS {
        TASK_CTX[t].store(c as u8, Ordering::Release);
    }
}

/// Borrow another context for the length of a scope, and put the old one
/// back on the way out -- by `Drop`, so an early return cannot leave the
/// initiative task judging its own conversation.
pub struct Scoped {
    was: Ctx,
}

impl Drop for Scoped {
    fn drop(&mut self) {
        adopt(self.was);
    }
}

pub fn scoped(c: Ctx) -> Scoped {
    let was = current();
    adopt(c);
    Scoped { was }
}

/// What every context agrees on, and the generation of each half.
struct Canon {
    router_gen: u64,
    head: Option<vocab::Head>,
    probe: Option<probe::Probe>,
    council: Option<council::Council>,
    adapter_gen: u64,
    adapters: Option<adapter::Adapters>,
}

static CANON: Spin<Canon> = Spin::new(Canon {
    router_gen: 0,
    head: None,
    probe: None,
    council: None,
    adapter_gen: 0,
    adapters: None,
});

/// Per-engine record of what it last agreed with.
#[derive(Clone, Copy, Default)]
pub struct Seen {
    router_gen: u64,
    router_epoch: u64,
    adapter_gen: u64,
    adapter_epoch: u64,
}

/// Bring an engine up to date before a borrow.
pub fn sync_in(e: &mut Engine) {
    let c = CANON.lock_irq();
    if e.seen.router_gen != c.router_gen {
        if let Some(h) = c.head.as_ref() {
            e.head = h.clone();
        }
        e.probe = c.probe.clone();
        e.council = c.council.clone();
        e.seen.router_gen = c.router_gen;
        e.seen.router_epoch = e.router_epoch;
    }
    if e.seen.adapter_gen != c.adapter_gen {
        e.model.adapters = c.adapters.clone();
        e.model.adapter_epoch += 1;
        e.seen.adapter_gen = c.adapter_gen;
        e.seen.adapter_epoch = e.model.adapter_epoch;
    }
}

/// Publish whatever this engine changed during a borrow.
pub fn publish(e: &mut Engine) {
    let router = e.router_epoch != e.seen.router_epoch;
    let adapters = e.model.adapter_epoch != e.seen.adapter_epoch;
    if !router && !adapters {
        return;
    }
    let mut c = CANON.lock_irq();
    if router {
        c.router_gen += 1;
        c.head = Some(e.head.clone());
        c.probe = e.probe.clone();
        c.council = e.council.clone();
        e.seen.router_gen = c.router_gen;
        e.seen.router_epoch = e.router_epoch;
    }
    if adapters {
        c.adapter_gen += 1;
        c.adapters = e.model.adapters.clone();
        e.seen.adapter_gen = c.adapter_gen;
        e.seen.adapter_epoch = e.model.adapter_epoch;
    }
}

/// A fresh engine for a context nobody has used yet, over the base weights
/// and whatever `CANON` currently says.
///
/// `base` is the adapter-free view `ai::init` keeps; `tok` is the one
/// tokenizer, leaked at load. Nothing here borrows the `Chat` engine, which
/// may be held by another task at this exact moment.
pub fn spawn(base: &super::model::Model, tok: &'static super::tokenizer::Tokenizer, head: vocab::Head) -> Engine {
    let cfg = base.cfg;
    let mut e = Engine {
        model: base.share(),
        tok,
        state: super::model::State::new(&cfg),
        rng: super::sample::Rng::new(crate::time::rdtsc()),
        head,
        probe: None,
        council: None,
        pos: 0,
        last_token: super::tokenizer::BOS,
        router_epoch: 0,
        seen: Seen { router_gen: u64::MAX, adapter_gen: u64::MAX, ..Seen::default() },
    };
    sync_in(&mut e);
    e
}

/// `diag ctx`. Needs a loaded model for every claim about logits; without one
/// it checks the bookkeeping that does not, and says so.
pub fn selftest() -> bool {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED};
    use alloc::string::String;
    use alloc::vec::Vec;
    let mut rows: Vec<(String, bool)> = Vec::new();
    let mut claim = |what: &str, ok: bool| rows.push((String::from(what), ok));

    // Assignment and scoping, which need no model.
    let was = current();
    {
        let _j = scoped(Ctx::Judge);
        claim("a scope moves the task into its context", current() == Ctx::Judge);
        {
            let _a = scoped(Ctx::Agent);
            claim("scopes nest", current() == Ctx::Agent);
        }
        claim("and unwind in order", current() == Ctx::Judge);
    }
    claim("leaving the scope puts the old context back", current() == was);

    if !super::engine_ready() {
        claim("(no model loaded -- the logit claims did not run)", true);
    } else {
        // The same tokens through two contexts' engines: shared weights, two
        // caches, and therefore exactly the same logits. Bit for bit, because
        // nothing about the arithmetic differs and a tolerance would hide a
        // context reading the other's cache.
        let toks = [9707usize, 1576, 29817, 3303];
        fn run(e: &mut Engine, toks: &[usize]) -> Vec<f32> {
            let mut st = super::model::State::scratch(&e.model.cfg, toks.len() + 1);
            for (i, &t) in toks.iter().enumerate() {
                e.model.forward(&mut st, t, i);
            }
            st.logits.clone()
        }
        let chat = super::with_engine(|e| run(e, &toks));
        let judge = {
            let _j = scoped(Ctx::Judge);
            super::with_engine(|e| run(e, &toks))
        };
        claim(
            "two contexts over one set of weights give identical logits",
            chat.is_some() && chat == judge,
        );

        // Independent holders: with Chat held by somebody else, Agent still
        // lends. A fake id nothing can be mid-call under, as `engine_selftest`
        // does.
        let other = crate::task::current().wrapping_add(1_000_000);
        super::holder_of(Ctx::Chat).store(other, core::sync::atomic::Ordering::Release);
        let chat_refused = super::with_engine(|_| ()).is_none();
        let agent_lent = {
            let _a = scoped(Ctx::Agent);
            super::with_engine(|_| ()).is_some()
        };
        super::holder_of(Ctx::Chat).store(super::NOBODY, core::sync::atomic::Ordering::Release);
        claim("a held context refuses", chat_refused);
        claim("while another context still lends", agent_lent);

        // A trial's adapter stays in the trial. Attach and perturb in Judge,
        // read Chat from inside that borrow, put Judge's adapter back.
        let leak_free = {
            let _j = scoped(Ctx::Judge);
            super::with_engine(|e| {
                let cfg = e.model.cfg;
                let before = e.model.detach_adapters();
                let mut moved_here = false;
                if e.model.attach_adapters(super::adapter::Adapters::full(&cfg, 8, 16.0)).is_ok() {
                    if let Some(cls) = e.model.adapters.as_mut().and_then(|a| a.cls.as_mut()) {
                        if cls.r >= 1 && cls.s.len() > 5 {
                            cls.a[3] = 0.1;
                            cls.b[5 * cls.r] = 0.25;
                            cls.s[5] = 2.0;
                        }
                    }
                    moved_here = Some(run(e, &toks)) != chat;
                }
                let chat_now = {
                    let _c = scoped(Ctx::Chat);
                    super::with_engine(|c| run(c, &toks))
                };
                e.model.detach_adapters();
                if let Some(b) = before {
                    let _ = e.model.attach_adapters_unseeded(b);
                }
                moved_here && chat_now == chat
            })
            .unwrap_or(false)
        };
        claim("an adapter on trial in one context never reaches another", leak_free);
        // And the conversation reads what it read before the trial, once the
        // trial's own restore has been published.
        let after = super::with_engine(|e| run(e, &toks));
        claim("after the trial puts its adapter back, chat is unchanged", after == chat);
    }

    for (what, good) in super::sched::checks() {
        rows.push((String::from(what), good));
    }

    let mut ok = true;
    for (what, good) in &rows {
        console::set_color(if *good { LTGREEN } else { LTRED });
        crate::kprintln!("  {}  {}", if *good { "ok  " } else { "FAIL" }, what);
        console::set_color(LTGRAY);
        ok &= *good;
    }
    crate::kprintln!("    {} claim(s)", rows.len());
    ok
}
