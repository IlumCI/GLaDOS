//! The Stratum V1 client, host-side.
//!
//! `design/mining.md` predicted this file before it existed: "the Stratum V1
//! client already in `src/mine/` is not wasted -- it moves to the pool as its
//! *upstream* client, host-side, where it is ordinary code." That turned out to
//! be true in the strongest sense available, because it is not a move at all.
//! `mine::stratum` is the kernel's own file, included by `#[path]`, and this
//! module is the socket around it.
//!
//! So the messages the kernel sends to somebody else's pool and the messages
//! this sends are built by one function each. There is no second implementation
//! to drift.
//!
//! ### One thread per coin, and no shared connection
//!
//! Each upstream is a different pool with its own extranonce, its own
//! difficulty and its own idea of what a job is. Multiplexing them would mean
//! inventing a routing layer to undo something nobody asked for; a thread that
//! blocks in `read` costs a stack and nothing else.
//!
//! ### No socket in the protocol
//!
//! `Upstream` is the conversation with no I/O in it: bytes in, lines out, and a
//! tick for the things that happen on a clock. The native daemon drives it from
//! a thread with a `TcpStream`; the Cloudflare pool drives the *same* state
//! machine from `connect()` in a Durable Object, compiled to WebAssembly. That
//! is `session.rs`'s bargain for the miner-facing side, made again here: two
//! transports, one set of rules, and nothing on either side of the boundary is
//! allowed to decide anything.
//!
//! ### What this is not
//!
//! It is a **proxy** and not a full pool: there is no node, no block template,
//! no coinbase of ours. Upstream decides what is mined and pays its own
//! address, and every share good enough goes straight back to it. That is the
//! B1 sequencing in `design/mining.md`'s plan -- share accounting and the entry
//! gate get to work against an upstream that already produces correct work,
//! and running nodes is a decision deferred until volume justifies it.

use std::sync::Mutex;

use crate::clock;
use crate::json::Json;
use crate::mine::stratum::{self, Message};
use crate::mine::u256::{target_for, U256};
use crate::pool::{Pool, Source, Work};

/// How long to wait for the subscribe and authorize replies.
const HANDSHAKE_MS: u64 = 15_000;
/// How long a connection may go silent before it is assumed dead.
///
/// A pool sends a job every so often even when nothing changes, so silence for
/// this long is a connection that is open at the socket layer and finished in
/// every way that matters -- the half-dead case, which is invisible to `read`
/// because `read` is simply blocked.
const SILENCE_MS: u64 = 300_000;
/// Longer than this and a line is a desynchronised stream, not a message.
const MAX_BUF: usize = 131_072;

/// What one step of the conversation produced.
#[derive(Default)]
pub struct UpOut {
    /// Lines for upstream, each already ending in a newline.
    pub send: Vec<String>,
    pub log: Vec<String>,
    /// Work was installed, so connected miners should be handed fresh jobs.
    pub work: bool,
    /// The connection is finished, and why. The driver closes and reconnects.
    pub close: Option<String>,
}

impl UpOut {
    fn fail(mut self, why: &str) -> UpOut {
        self.close = Some(String::from(why));
        self
    }
}

/// One upstream connection's state, from subscribe to the last submit.
pub struct Upstream {
    pub slot: usize,
    pub label: String,
    pub host: String,
    pub port: u16,
    user: String,
    pass: String,
    buf: Vec<u8>,
    e1: Vec<u8>,
    e2_size: usize,
    got_sub: bool,
    authorized: bool,
    up_target: U256,
    pending: Option<stratum::Job>,
    opened_ms: u64,
    heard_ms: u64,
    next_id: u64,
}

impl Upstream {
    /// The upstream for `slot`, if that coin has one.
    pub fn for_slot(pool: &Pool, slot: usize) -> Option<Upstream> {
        let c = pool.coins.get(slot)?;
        let Source::Upstream { host, port, user, pass } = &c.source else {
            return None;
        };
        Some(Upstream {
            slot,
            label: c.label.clone(),
            host: host.clone(),
            port: *port,
            user: user.clone(),
            pass: pass.clone(),
            buf: Vec::new(),
            e1: Vec::new(),
            e2_size: 4,
            got_sub: false,
            authorized: false,
            up_target: target_for(1, 0).unwrap_or(U256::ZERO),
            pending: None,
            opened_ms: 0,
            heard_ms: 0,
            next_id: 1000,
        })
    }

    pub fn live(&self) -> bool {
        self.got_sub && self.authorized
    }

    /// A fresh connection is open: forget the last one and subscribe.
    ///
    /// `stratum::subscribe` and `authorize` are the kernel's own, so what goes
    /// on this wire is byte for byte what the kernel would send.
    pub fn open(&mut self) -> UpOut {
        self.buf.clear();
        self.e1.clear();
        self.got_sub = false;
        self.authorized = false;
        self.pending = None;
        self.opened_ms = clock::now_ms();
        self.heard_ms = self.opened_ms;
        let mut out = UpOut::default();
        out.log.push(format!("[up {}] connected to {}:{}", self.label, self.host, self.port));
        out.send.push(stratum::subscribe(1));
        out
    }

    /// Bytes from upstream, in whatever pieces the transport delivered them.
    pub fn on_bytes(&mut self, pool: &Mutex<Pool>, bytes: &[u8]) -> UpOut {
        let mut out = UpOut::default();
        self.buf.extend_from_slice(bytes);
        if !bytes.is_empty() {
            self.heard_ms = clock::now_ms();
        }
        loop {
            let line = match stratum::take_line(&mut self.buf) {
                Ok(Some(l)) => l,
                Ok(None) => break,
                Err(_) => return out.fail("upstream sent a line too long to be a message"),
            };
            match stratum::classify(&line) {
                Ok(Message::Response { id, ok, body }) => match id {
                    1 => {
                        let Some((e1, size)) = stratum::subscribe_result(&body) else {
                            return out.fail("the subscribe reply could not be read");
                        };
                        self.e1 = e1;
                        self.e2_size = size;
                        self.got_sub = true;
                        out.send.push(stratum::authorize(2, &self.user, &self.pass));
                    }
                    2 => {
                        if !ok {
                            // Named rather than retried quietly. A refused worker is
                            // a wrong address or a wrong password, and a backoff loop
                            // against that is a machine reconnecting forever for a
                            // reason nobody is being told.
                            return out.fail("upstream refused this worker -- check the address and password");
                        }
                        self.authorized = true;
                        out.log.push(format!(
                            "[up {}] subscribed and authorized, extranonce1 {} bytes, extranonce2 {}",
                            self.label,
                            self.e1.len(),
                            self.e2_size
                        ));
                        // `set_difficulty` and the first `notify` routinely arrive
                        // *before* the authorize reply, so a job may be waiting.
                        if let Some(job) = self.pending.take() {
                            self.install(pool, job, &mut out);
                        }
                    }
                    // A submit's answer. Logged either way: a pool that rejects
                    // everything and a pool that is not there look identical from
                    // a share counter alone.
                    _ => out.log.push(format!(
                        "[up {}] submit {id} {}",
                        self.label,
                        if ok { "accepted" } else { "REJECTED" }
                    )),
                },
                Ok(Message::Notify { method, params }) => {
                    let mut job = None;
                    take_notification(&method, &params, &mut self.up_target, &mut job);
                    if let Some(j) = job {
                        if self.live() {
                            self.install(pool, j, &mut out);
                        } else {
                            self.pending = Some(j);
                        }
                    }
                }
                Err(_) => {}
            }
        }
        if self.buf.len() > MAX_BUF {
            return out.fail("upstream sent a line too long to be a message");
        }
        out
    }

    /// Time passed, or a miner just submitted: forward what qualifies, and
    /// notice a handshake that stalled or a connection that went silent.
    pub fn on_tick(&mut self, pool: &Mutex<Pool>) -> UpOut {
        let mut out = UpOut::default();
        let now = clock::now_ms();
        if !self.live() {
            if now.saturating_sub(self.opened_ms) > HANDSHAKE_MS {
                return out.fail("upstream did not finish the handshake in time");
            }
            return out;
        }
        if now.saturating_sub(self.heard_ms) > SILENCE_MS {
            return out.fail("upstream went silent; reconnecting");
        }
        let forwards = pool.lock().unwrap().take_forwards_for(self.slot);
        for f in forwards {
            self.next_id += 1;
            out.log.push(format!("[up {}] forwarding a share for job {}", self.label, f.job_id));
            out.send.push(stratum::submit(
                self.next_id,
                &self.user,
                &f.job_id,
                &f.extranonce2,
                &f.ntime_be,
                &f.nonce_be,
            ));
        }
        out
    }

    fn install(&self, pool: &Mutex<Pool>, job: stratum::Job, out: &mut UpOut) {
        let t = self.up_target.to_be_bytes();
        out.log.push(format!(
            "[up {}] job {}, {} merkle level(s), target {:02x}{:02x}{:02x}{:02x}..",
            self.label,
            job.id,
            job.branch.len(),
            t[0],
            t[1],
            t[2],
            t[3]
        ));
        let w = Work {
            job_id: job.id,
            prev_wire: job.prev_wire,
            coinb1: job.coinb1,
            coinb2: job.coinb2,
            branch: job.branch,
            version: job.version,
            ntime: job.ntime,
            nbits: job.nbits,
            extranonce1: self.e1.clone(),
            extranonce2_size: self.e2_size,
            up_target: self.up_target,
        };
        pool.lock().unwrap().set_work(self.slot, w);
        out.work = true;
    }
}

/// Start a client thread for every coin that has an upstream. Native only.
#[cfg(not(target_arch = "wasm32"))]
pub fn start_all(pool: std::sync::Arc<Mutex<Pool>>) {
    use std::time::Duration;
    let n = pool.lock().unwrap().coins.len();
    for slot in 0..n {
        let Some(mut up) = Upstream::for_slot(&pool.lock().unwrap(), slot) else {
            continue;
        };
        let pool = std::sync::Arc::clone(&pool);
        std::thread::spawn(move || {
            let (min, max) = (Duration::from_secs(2), Duration::from_secs(60));
            let mut backoff = min;
            loop {
                match drive(&pool, &mut up) {
                    Ok(()) => {
                        println!("[up {}] upstream closed the connection", up.label);
                        backoff = min;
                    }
                    Err(e) => println!("[up {}] {e}", up.label),
                }
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(max);
            }
        });
    }
}

/// One connection, driven over a `TcpStream` until it ends.
#[cfg(not(target_arch = "wasm32"))]
fn drive(pool: &Mutex<Pool>, up: &mut Upstream) -> Result<(), String> {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::time::Duration;

    println!("[up {}] connecting to {}:{}", up.label, up.host, up.port);
    let mut sock = TcpStream::connect((up.host.as_str(), up.port)).map_err(|e| format!("connect: {e}"))?;
    // Short, so the loop gets a turn to forward shares and to notice silence.
    // The read timing out is the ordinary case here rather than an error.
    sock.set_read_timeout(Some(Duration::from_millis(250))).map_err(|e| format!("{e}"))?;

    let emit = |sock: &mut TcpStream, out: UpOut| -> Result<(), String> {
        for l in &out.log {
            println!("{l}");
        }
        for m in &out.send {
            sock.write_all(m.as_bytes()).map_err(|e| format!("write: {e}"))?;
        }
        match out.close {
            Some(why) => Err(why),
            None => Ok(()),
        }
    };
    let o = up.open();
    emit(&mut sock, o)?;
    let mut chunk = [0u8; 8192];
    loop {
        let n = match sock.read(&mut chunk) {
            Ok(0) => return Ok(()),
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock || e.kind() == std::io::ErrorKind::TimedOut => 0,
            Err(e) => return Err(format!("read: {e}")),
        };
        let o = up.on_bytes(pool, &chunk[..n]);
        emit(&mut sock, o)?;
        let o = up.on_tick(pool);
        emit(&mut sock, o)?;
    }
}

fn take_notification(
    method: &str,
    params: &Json,
    up_target: &mut U256,
    job: &mut Option<stratum::Job>,
) {
    match method {
        "mining.set_difficulty" => {
            // Through `stratum::decimal` and never `as_i64`, which is the trap
            // that module exists to document: altcoin pools routinely send a
            // fractional difficulty, `as_i64` reads `0.001` as `0`, and a
            // target built from zero accepts everything.
            let Some(text) = params.idx(0).and_then(raw_number) else {
                return;
            };
            let Some((m, scale)) = stratum::decimal(&text) else {
                return;
            };
            if let Some(t) = target_for(m, scale) {
                *up_target = t;
            }
        }
        "mining.notify" => {
            *job = stratum::parse_job(params);
        }
        // `client.reconnect` is refused here exactly as the kernel refuses it:
        // following a redirect to an address no human typed is not a thing to
        // do quietly, and the reconnect loop will return to the configured host.
        _ => {}
    }
}

/// The literal token of a JSON number, which is what `decimal` wants.
///
/// `Json::Num` keeps the source text precisely so a fractional difficulty
/// survives being read, and a pool that sends its difficulty as a *string* is
/// common enough to accept as well.
fn raw_number(j: &Json) -> Option<String> {
    match j {
        Json::Num(t) => Some(t.clone()),
        Json::Str(t) => Some(t.clone()),
        _ => None,
    }
}
