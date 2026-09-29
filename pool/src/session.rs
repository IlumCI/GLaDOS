//! What the pool says to one miner, with no socket anywhere in it.
//!
//! **This was the body of `server::handle`, and it moved for a second transport.**
//! Every rule a miner is held to -- one worker per connection, the roster check,
//! the rate limits, the validation budget, the answer to a malformed submit, the
//! bad-share cutoff, VarDiff retargeting -- was written inline around `write_all`
//! on a `TcpStream`. A pool served from a Cloudflare Durable Object speaks
//! WebSocket instead, and the choice was to write those rules a second time in
//! JavaScript or to take the socket out of them. Two implementations of the rules
//! that decide who gets paid is the thing this repository refuses everywhere else,
//! so the socket came out: a `Conn` is handed one line and answers what to send
//! back, what to log, and whether to hang up. `server.rs` writes that to a TCP
//! stream; the Worker writes it to a WebSocket; neither decides anything.
//!
//! Framing stays with the transport. A byte stream needs `take_line` and its
//! `MAX_LINE` refusal; a WebSocket message already has boundaries. What arrives
//! here is one line, already cut.

use std::sync::Mutex;

use crate::budget::Budget;
use crate::clock;
use crate::json::Json;
use crate::mine::proto;
use crate::pool::{Pool, Verdict};
use crate::vardiff::VarDiff;

/// ### Why an unauthenticated pool needs these three numbers
///
/// Validating a share means computing it, and a yespower validation is 7.5 ms.
/// `submit` does that *before* anything has established the sender is a real
/// miner -- it cannot do otherwise, since computing the hash is how you find
/// out. So a stranger who connects, takes the job pushed at them, and submits
/// garbage nonces buys 7.5 ms of this machine's CPU per message, and one such
/// connection saturates a core at about 133 a second.
///
/// That is not a subtle hole and it is not solved by being a small target. It
/// is bounded here instead, at the connection, before the mutex is taken.
///
/// **None of this stops a distributed attack**, and on a home connection that
/// is the case that matters most: the upstream link saturates long before this
/// daemon does, and nothing running on the server can prevent that. What these
/// limits buy is that a *single* stranger cannot take the machine down by
/// accident or on a whim.
///
/// A correct miner produces zero bad shares, so thirty-two is not a tolerance
/// -- it is room for one genuine bug or one bad build before the connection is
/// dropped.
pub const MAX_BAD: u32 = 32;
/// A miner submitting to four coins at one share every ten seconds sends 0.4 a
/// second. Twenty is fifty times that, so it never touches a real miner, and it
/// caps one connection at 150 ms of validation per second -- fifteen percent of
/// a core even if every message is garbage.
pub const MAX_SUBMITS_PER_SEC: u32 = 20;
/// A `glados.work` costs one job build rather than a validation, so the bound
/// is about the job ring rather than about the CPU: `KEEP_JOBS` is 64, and a
/// connection allowed to ask freely would evict every job every other miner is
/// working. Four a second is well above what any real device needs -- an RTX
/// 3050 on sha256d spends a nonce space every eight and a half seconds, so a
/// card sixty times faster than that one is still inside this.
pub const MAX_WORK_PER_SEC: u32 = 4;

/// The pool-wide validation budget, shared by every connection.
///
/// **A `Mutex` and not an atomic**, because admitting costs a refill against
/// the clock, a lookup and a subtraction, and doing that as three atomics
/// makes an interleaving where two connections both see enough budget and both
/// spend it -- which is the bound failing exactly when it is under pressure.
/// It is held for a few hundred nanoseconds and never across a validation.
///
/// **Host-only.** `Budget` measures with `std::time::Instant`, which panics on
/// wasm32, and a Durable Object has no use for it anyway: Cloudflare bounds the
/// CPU of each message itself. So the Worker never calls `set_cpu_percent`, this
/// stays `None` there, and no `Instant` is ever read on that target.
static BUDGET: Mutex<Option<Budget>> = Mutex::new(None);

/// Set the share of one core the pool may spend validating. Zero is no limit.
pub fn set_cpu_percent(percent: f64) {
    *BUDGET.lock().unwrap() = Some(Budget::new(percent));
}

/// What the budget has measured, for the operator's report.
pub fn budget_report() -> Vec<(String, f64, f64)> {
    BUDGET.lock().unwrap().as_ref().map(|b| b.report()).unwrap_or_default()
}

pub fn budget_denied() -> u64 {
    BUDGET.lock().unwrap().as_ref().map(|b| b.denied()).unwrap_or(0)
}

/// What handling one line produced.
#[derive(Default)]
pub struct Out {
    /// Messages for the miner, each a complete line ending in `\n`.
    pub send: Vec<String>,
    /// Lines for the operator's log.
    pub log: Vec<String>,
    /// Hang up after sending.
    pub close: bool,
}

impl Out {
    fn say(&mut self, msg: String) {
        self.send.push(msg);
    }
    fn note(&mut self, line: String) {
        self.log.push(line);
    }
    fn hang_up(mut self) -> Out {
        self.close = true;
        self
    }
}

/// One miner's connection, as the pool sees it.
pub struct Conn {
    peer: String,
    worker: String,
    greeted: bool,
    bad: u32,
    deferred: u32,
    malformed: u32,
    submits_this_second: u32,
    works_this_second: u32,
    window_ms: u64,
    // One retargeter per coin, not one per connection. A single machine works
    // several coins on different algorithms and its rate on them differs by
    // three orders of magnitude -- 248,884 H/s of sha256d beside 342 H/s of
    // yespower, measured on the kernel in one run -- so a shared difficulty
    // would be wrong for at least one of them by a factor of a thousand.
    vd: Vec<VarDiff>,
}

impl Conn {
    pub fn new(peer: &str) -> Conn {
        Conn {
            peer: String::from(peer),
            worker: String::from("(unauthenticated)"),
            greeted: false,
            bad: 0,
            deferred: 0,
            malformed: 0,
            submits_this_second: 0,
            works_this_second: 0,
            window_ms: clock::now_ms(),
            vd: Vec::new(),
        }
    }

    pub fn worker(&self) -> &str {
        &self.worker
    }

    pub fn greeted(&self) -> bool {
        self.greeted
    }

    /// Nothing has arrived for a job period.
    ///
    /// **This is the only thing that re-issues work**, and it was wrong the other
    /// way first. The code once re-issued on every wake, arguing that "a miner
    /// that talks constantly works one header for as long as it keeps talking",
    /// which is not a real problem -- the period sets the cadence and a talkative
    /// miner is still on the clock.
    ///
    /// What it *was* is a bug the abuse check found. Every message re-issued every
    /// slot, so a miner sending shares generated two jobs per share; thirty-two
    /// shares against two coins pushed sixty-four jobs through a sixty-four entry
    /// ring and evicted the job being worked on. The miner's own submissions were
    /// then answered `stale`. **A busy miner starved itself**, and the busier it
    /// was the worse it got.
    pub fn on_idle(&mut self, pool: &Mutex<Pool>) -> Out {
        let mut out = Out::default();
        if self.greeted {
            // The idle path is what finds a miner set *too hard*. It will never
            // reach a share window, so without a clock the pool would wait
            // forever to discover it asked too much.
            for (slot, v) in self.vd.iter_mut().enumerate() {
                if let Some(b) = v.on_idle() {
                    out.note(format!("[pool] {} slot {slot} eased to {b} bits", self.worker));
                    pool.lock().unwrap().remember_bits(&self.worker, slot, b);
                }
            }
            issue_all(pool, &self.vd, &mut out);
        }
        out
    }

    /// One line from the miner, already framed.
    pub fn on_line(&mut self, pool: &Mutex<Pool>, line: &str) -> Out {
        let mut out = Out::default();
        let Some(v) = Json::parse(line.trim()) else {
            return out;
        };
        let Some(method) = v.get("method").and_then(|m| m.as_str()) else {
            return out;
        };
        let id = v.get("id").and_then(|i| i.as_i64()).unwrap_or(0) as u64;
        let params = v.get("params");

        match method {
            "glados.hello" => {
                let Some(h) = params.and_then(proto::parse_hello) else {
                    return out;
                };
                // Refused rather than negotiated. Two ends that disagree about the
                // wire have nothing to talk about, and a negotiation is a code path
                // that only runs against versions nobody has.
                if h.v != proto::VERSION {
                    out.say(format!(
                        "{{\"id\":{id},\"result\":null,\"error\":\"protocol v{} wanted, v{} offered\"}}\n",
                        proto::VERSION,
                        h.v
                    ));
                    return out.hang_up();
                }
                // **One connection is one worker.** A second `hello` under a
                // different name was accepted and simply overwrote the first,
                // which made a single socket a way to be as many miners as you
                // like: measured at nineteen names a second against the
                // per-connection submit cap, each one a permanent record in the
                // tally, and each costing the pool *no validation at all* -- a
                // submit naming a job the pool does not hold answers `stale`
                // before it hashes, so the budget never sees any of it. Two
                // hundred and fifty-six connections is around forty-nine hundred
                // a second.
                //
                // The tally cap bounds the damage and this removes the cheap path
                // to it. Refused rather than dropped, because unlike a flood this
                // is one message and the sender may genuinely be a confused
                // client.
                //
                // Re-greeting under the *same* name is allowed and is a no-op that
                // re-issues work, since a client retrying its handshake after a
                // timeout is doing something reasonable and there is nothing to
                // change.
                if self.greeted && h.worker != self.worker {
                    out.note(format!(
                        "[pool] {} tried to become {} having greeted as {}",
                        self.peer, h.worker, self.worker
                    ));
                    out.say(format!(
                        "{{\"id\":{id},\"result\":null,\"error\":\"this connection already greeted as {}; one worker per connection\"}}\n",
                        self.worker
                    ));
                    return out;
                }
                // **Can this name be paid?** Asked here because the alternative
                // is finding out after the event, when `distribute.py` prints a
                // name with real work behind it and no address to send it to.
                // That is unrecoverable for the miner and for the operator both;
                // this is a connection error the miner reads while they still have
                // their config open.
                //
                // Only `No` is actionable. `Unknown` means no roster has ever
                // loaded, and refusing on that would turn a missing file into an
                // outage -- see `roster`'s note on failing open.
                if crate::roster::check(&h.worker) == crate::roster::Payability::No {
                    if crate::roster::require() {
                        out.note(format!("[pool] {} refused: {} has no payout address", self.peer, h.worker));
                        out.say(format!(
                            "{{\"id\":{id},\"result\":null,\"error\":\"the worker name '{}' has no payout address, so its shares could not be paid. Register it, or mine under your 0x address as the worker name. {}\"}}\n",
                            h.worker,
                            crate::roster::where_to_register()
                        ));
                        return out;
                    }
                    // Not refusing, but the operator should see it: this is
                    // somebody about to do work nobody can pay for.
                    out.note(format!("[pool] {} warning: {} is not in the roster", self.peer, h.worker));
                }
                self.worker = h.worker.clone();
                out.note(format!("[pool] {} hello  worker={} agent={}", self.peer, h.worker, h.agent));
                let slots = pool.lock().unwrap().slots();
                // `resume_bits` and not `start_bits`: VarDiff cannot converge on a
                // connection shorter than its own sixty second idle window, so
                // without this a miner that reconnects often restarts at the
                // operator's guess every time and never retargets at all. See
                // `Pool::converged` for the soak that found it.
                self.vd = {
                    let p = pool.lock().unwrap();
                    (0..slots as usize)
                        .map(|i| VarDiff::new(p.resume_bits(&self.worker, i)))
                        .collect()
                };
                let w = proto::Welcome {
                    v: proto::VERSION,
                    slots,
                    session: self.peer.clone(),
                };
                out.say(proto::encode_welcome(id, &w));
                self.greeted = true;
                issue_all(pool, &self.vd, &mut out);
            }
            "glados.submit" => {
                // Before the greeting there is no worker to attribute a share to,
                // so this is only ever an attempt to skip the handshake and spend
                // CPU.
                if !self.greeted {
                    return out.hang_up();
                }
                // **Answered, where the other two refusals are not, and the
                // difference is who is on the other end.** The rate limit and the
                // budget drop silently because the peer is already sending more
                // than the pool can serve and a reply is a second thing to send
                // them. A submit that will not parse is a *client bug* on a
                // connection that greeted correctly and may be sending one message
                // every ten seconds.
                //
                // Silence there cost this project a whole soak run. A test miner
                // hashed correctly for forty minutes and was credited nothing,
                // because it spelled the nonce as a JSON number where
                // `parse_share` wants big-endian hex -- and the pool dropped every
                // one of them without a word, so the miner's log said "connected,
                // working" and the pool's said nothing at all. One line here is the
                // difference between that and a fix in ten seconds.
                //
                // Bounded the same way everything else here is: a peer that sends
                // malformed submits forever gets four replies and then silence, so
                // this cannot become the flood it exists to make visible.
                let Some(sh) = params.and_then(proto::parse_share) else {
                    self.malformed += 1;
                    if self.malformed <= 4 {
                        out.note(format!("[pool] {} sent a submit that will not parse", self.worker));
                        out.say(format!(
                            "{{\"id\":{id},\"result\":null,\"error\":\"submit needs job (string) and nonce (8 hex digits, big-endian)\"}}\n"
                        ));
                    }
                    return out;
                };

                // The rate check comes before the validation and before the mutex,
                // which is the whole point: past this line the cost is 7.5 ms of
                // somebody else's machine.
                self.tick_window();
                self.submits_this_second += 1;
                if self.submits_this_second > MAX_SUBMITS_PER_SEC {
                    // Dropped rather than answered. A reply is a second thing to
                    // send to somebody already sending too much, and a miner at
                    // this rate has a bug an error message will not fix.
                    return out;
                }

                // **The total bound, which the per-connection one is not.**
                // `MAX_SUBMITS_PER_SEC` caps this connection; nothing capped two
                // hundred and fifty-six of them, and the arithmetic that made the
                // per-connection figure look safe was never multiplied by
                // `MAX_CONNECTIONS`.
                //
                // The algorithm is looked up before validating because that is
                // what the cost depends on -- a sha256d share is 2.4 us on the
                // machine this was deployed to and a yespower one is 19,000.
                let algo_name = {
                    let p = pool.lock().unwrap();
                    p.algo_of_job(&sh.job)
                };
                let name = algo_name.unwrap_or_else(|| String::from("?"));
                let admitted = {
                    let mut g = BUDGET.lock().unwrap();
                    match g.as_mut() {
                        None => true,
                        Some(b) => b.admit(&name),
                    }
                };
                if !admitted {
                    // Dropped rather than answered, like the rate limit above and
                    // for the same reason: a reply is a second thing to send to
                    // somebody the pool is already struggling to serve.
                    //
                    // **Quietened against its own counter and not against `bad`.**
                    // It was `bad < 4`, and a deferred share is never validated, so
                    // `bad` never moves on this path and the guard never engaged: a
                    // twenty-second flood at 1% of a core wrote 785 log lines, one
                    // per deferral, unbounded.
                    self.deferred += 1;
                    if self.deferred <= 4 {
                        out.note(format!("[pool] {} share deferred: validation budget spent", self.worker));
                        if self.deferred == 4 {
                            out.note(format!("[pool] {} is over the validation budget; quietening the log", self.worker));
                        }
                    }
                    return out;
                }

                let began = clock::now_us();
                let verdict = pool.lock().unwrap().submit(&self.worker, &sh);
                // Timed here rather than estimated anywhere, because assuming this
                // number is precisely what went wrong: the 7.5 ms in the comment
                // above was measured on a different machine and the real one is
                // 19 ms.
                {
                    let took = clock::now_us().saturating_sub(began) as f64;
                    let mut g = BUDGET.lock().unwrap();
                    if let Some(b) = g.as_mut() {
                        b.record(&name, took);
                    }
                }
                // Accepted shares are the record and are always logged. A refusal
                // is logged for the first few and then counted, because at twenty a
                // second one line each fills a journal -- and on a borrowed
                // machine, filling somebody's disk is a worse way to fail than
                // dropping a connection.
                if verdict == Verdict::Accepted || self.bad < 4 {
                    out.note(format!(
                        "[pool] {} share job={} nonce={:08x} -> {}",
                        self.worker,
                        sh.job,
                        sh.nonce,
                        verdict.name()
                    ));
                }
                let ok = verdict == Verdict::Accepted;
                out.say(format!(
                    "{{\"id\":{id},\"result\":{{\"ok\":{ok},\"verdict\":\"{}\"}},\"error\":null}}\n",
                    verdict.name()
                ));

                // Retarget on accepted shares only. A stale share is work against a
                // job that aged out and says nothing about the rate now, and
                // counting rejected ones would let a miner talk its own difficulty
                // upward by sending noise.
                if verdict == Verdict::Accepted {
                    let slot = {
                        let p = pool.lock().unwrap();
                        p.slot_of_job(&sh.job)
                    };
                    if let Some(slot) = slot {
                        if let Some(v) = self.vd.get_mut(slot) {
                            if let Some(b) = v.on_share() {
                                out.note(format!("[pool] {} slot {slot} retargeted to {b} bits", self.worker));
                                pool.lock().unwrap().remember_bits(&self.worker, slot, b);
                                // Sent straight away rather than at the next job
                                // period: a miner that just proved it is fast
                                // should not spend another thirty seconds flooding
                                // at the old difficulty.
                                issue_all(pool, &self.vd, &mut out);
                            }
                        }
                    }
                }

                // Only `Bad` counts. `Stale` is a job that aged out and is nobody's
                // fault, and `Duplicate` is a retry -- treating either as abuse
                // would disconnect honest miners on a slow link, which is the
                // failure this limit must not have.
                if verdict == Verdict::Bad {
                    self.bad += 1;
                    if self.bad == 4 {
                        out.note(format!("[pool] {} is sending bad shares; quietening the log", self.worker));
                    }
                    if self.bad > MAX_BAD {
                        out.note(format!("[pool] {} dropped after {} bad shares", self.worker, self.bad));
                        return out.hang_up();
                    }
                }
            }
            "glados.work" => {
                // **A job is a finite search and a fast device finishes it.** The
                // nonce is four bytes at a fixed offset, so a job carries 2^32
                // hashes and nothing more; at half a gigahash a second that is
                // eight and a half seconds against a thirty-second re-issue. Before
                // this existed the miner wrapped and rescanned the same space, and
                // the pool saw 35 duplicates against 23 accepted shares, every
                // repeated nonce arriving exactly six times.
                //
                // One slot, not all of them. `make_job` advances the extranonce2
                // for an upstream coin and the job counter for a local one, so the
                // answer is a genuinely different search either way -- and
                // re-issuing every slot on every request is the exact shape of the
                // churn bug the abuse test found.
                if !self.greeted {
                    return out.hang_up();
                }
                self.tick_window();
                self.works_this_second += 1;
                if self.works_this_second > MAX_WORK_PER_SEC {
                    return out;
                }
                let Some(slot) = params.and_then(proto::parse_work) else {
                    return out;
                };
                let job = {
                    let mut p = pool.lock().unwrap();
                    if slot as usize >= p.slots() as usize {
                        None
                    } else {
                        let bits = self
                            .vd
                            .get(slot as usize)
                            .map(|v| v.bits())
                            .unwrap_or_else(|| p.start_bits(slot as usize));
                        p.make_job(slot, bits)
                    }
                };
                // Silence when there is nothing to give. An upstream coin with no
                // template yet yields no job, and inventing one would put a miner
                // on a search that can never pay.
                if let Some(j) = job {
                    out.say(proto::encode_job(&j));
                }
            }
            // Unknown methods are ignored rather than refused. A miner newer than
            // this pool may greet with something extra, and closing on it would
            // make every forward-compatible addition a breaking change.
            _ => {}
        }
        out
    }

    /// Restart the one-second rate window when it has run out.
    fn tick_window(&mut self) {
        let now = clock::now_ms();
        if now.saturating_sub(self.window_ms) >= 1000 {
            self.window_ms = now;
            self.submits_this_second = 0;
            self.works_this_second = 0;
        }
    }
}

/// Fresh work on every slot, at each slot's own difficulty.
fn issue_all(pool: &Mutex<Pool>, vd: &[VarDiff], out: &mut Out) {
    let jobs: Vec<proto::Job> = {
        let mut p = pool.lock().unwrap();
        (0..p.slots())
            .filter_map(|s| {
                // A connection that has not greeted yet has no retargeters, so
                // fall back to the coin's configured start rather than refusing:
                // the alternative is a miner that greets and is told nothing until
                // the first timeout.
                let bits = vd
                    .get(s as usize)
                    .map(|v| v.bits())
                    .unwrap_or_else(|| p.start_bits(s as usize));
                p.make_job(s, bits)
            })
            .collect()
    };
    for j in jobs {
        out.say(proto::encode_job(&j));
    }
}
