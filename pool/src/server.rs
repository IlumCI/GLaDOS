//! The listener, and nothing else.
//!
//! Every decision lives in `pool.rs`; this file moves bytes. The split is the
//! one `update::decide` and `code::locate` already have here, and it is what
//! lets `Pool::submit` be tested without a socket.
//!
//! ### The framing is the kernel's own
//!
//! `stratum::take_line` splits the stream, which means the pool and the miner
//! agree about what a frame is because they are running the same function --
//! including `MAX_LINE`, and including the refusal of a line longer than it.
//! A pool that framed differently from its miners would produce a connection
//! that works until somebody sends a large job and then desynchronises, which
//! is the failure mode hardest to attribute.
//!
//! ### A thread per connection, and why that is enough
//!
//! A mining connection is idle almost all of the time: a job every thirty
//! seconds and a share every few. The interesting concurrency is in the
//! miners, not here. An async runtime would be a dependency, and this
//! repository's rule about those is stated in the kernel's own manifest --
//! so a thread that sleeps in `read` is both simpler and the right shape until
//! there are enough connections for the stacks to matter.
//!
//! **What a connection actually costs, measured rather than assumed.** Three
//! hundred were opened at once against the deployed binary on the host it runs
//! on, held for twenty seconds and sampled: 258 threads, 260 descriptors and
//! 8,184 KiB resident against a 904 KiB idle baseline. So a connection is one
//! thread, one descriptor and **28 KiB**, and the full ceiling is 7.3 MiB --
//! which is what "until the stacks matter" was worth as a number and had never
//! been. Everything came back to 2 threads, 4 descriptors and 908 KiB
//! afterwards, so the ceiling does not leak.
//!
//! A first attempt at that measurement reported `Threads: 2` throughout, which
//! is not a pool serving 256 connections -- it is a sampler that opened and
//! closed inside its own sampling interval. The hold is the measurement.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::mine::stratum::take_line;
use crate::pool::Pool;
use crate::session::{Conn, Out};

/// How often a connection is given fresh work.
///
/// A job carries a timestamp, so re-issuing is also what stops a miner
/// searching one header forever. Short enough that a stale job never
/// accumulates much wasted work; long enough that it is not the reason a
/// yespower slice never finishes a batch.
const JOB_PERIOD: Duration = Duration::from_secs(30);

// `MAX_BAD`, `MAX_SUBMITS_PER_SEC` and `MAX_WORK_PER_SEC` live in `session.rs`
// now, with the reasons for each: they are rules about miners, and every
// transport has to hold miners to the same ones.

/// A ceiling on threads as much as on miners, and deliberately below the
/// `TasksMax=512` in the systemd unit so the daemon refuses before the service
/// manager kills it. A refusal is a log line; being killed is an outage.
///
/// **The default, not the limit.** It was a `const`, which was right while the
/// only question was whether a stranger could exhaust the box. It stops being
/// right the moment somebody plans an event: 256 is below the plausible peak of
/// a launch drawing several hundred people in timezone waves, and a ceiling
/// that turns away the fourth wave is indistinguishable from an outage to the
/// people in it. `--max-connections` moves it, and the cost is stated rather
/// than guessed -- one thread, one descriptor and 28 KiB each, measured.
const DEFAULT_MAX_CONNECTIONS: usize = 256;

static MAX_CONNECTIONS: AtomicUsize = AtomicUsize::new(DEFAULT_MAX_CONNECTIONS);

/// Set the connection ceiling. Answers what it actually stored.
///
/// **Refused above the descriptor limit rather than accepted and then failing
/// per connection.** Each connection is a descriptor and so are the listener,
/// the ledger and the three standard ones, so a ceiling above `RLIMIT_NOFILE`
/// is a promise the process cannot keep -- and the way it fails is `accept`
/// returning `EMFILE` in the middle of an event, which reads as the network
/// breaking rather than as a setting being wrong.
pub fn set_max_connections(n: usize) -> usize {
    let n = n.max(1);
    let room = fd_limit().saturating_sub(16);
    let n = if room > 0 && n > room {
        println!("[pool] --max-connections {n} is above the descriptor limit; using {room}");
        room
    } else {
        n
    };
    MAX_CONNECTIONS.store(n, Ordering::Relaxed);
    n
}

/// The soft `RLIMIT_NOFILE`, or zero when it cannot be read.
///
/// Read from `/proc` rather than through `libc`, because this crate has no
/// dependencies and adding one to learn a number is the wrong trade. Zero means
/// "do not know", and the caller treats not knowing as no constraint rather
/// than as zero descriptors -- guessing low here would cap a healthy pool for
/// no reason.
fn fd_limit() -> usize {
    let Ok(text) = std::fs::read_to_string("/proc/self/limits") else {
        return 0;
    };
    for line in text.lines() {
        if !line.starts_with("Max open files") {
            continue;
        }
        // "Max open files   1024   524288   files"
        if let Some(soft) = line.split_whitespace().nth(3) {
            return soft.parse().unwrap_or(0);
        }
    }
    0
}

static LIVE_CONNECTIONS: AtomicUsize = AtomicUsize::new(0);

/// Decrements the connection count however the thread leaves -- returned,
/// errored, or panicked. A count that only decremented on the happy path would
/// drift upward until the pool refused everybody, which reads exactly like
/// being under attack.
struct ConnectionSlot;

impl Drop for ConnectionSlot {
    fn drop(&mut self) {
        LIVE_CONNECTIONS.fetch_sub(1, Ordering::Relaxed);
    }
}

// The validation budget lives in `session.rs`, beside the submit path that
// spends it. Re-exported so `main.rs` asks the same names it always has.
pub use crate::session::{budget_denied, budget_report, set_cpu_percent};

pub fn serve(addr: &str, pool: Arc<Mutex<Pool>>) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    // Printed rather than assumed: binding is the step that fails when another
    // copy is already running, and `stratumstub.py` records what it cost to
    // have that be silent.
    println!("[pool] listening on {}", listener.local_addr()?);
    {
        let p = pool.lock().unwrap();
        for (i, c) in p.coins.iter().enumerate() {
            println!(
                "[pool] slot {i}  {}  {}  ({})",
                c.label,
                c.algo.detail(),
                c.source.name()
            );
        }
    }
    for stream in listener.incoming() {
        let stream = stream?;
        // Counted before the thread exists. Spawning first and checking inside
        // would make the ceiling a suggestion: what is being bounded is the
        // thread and its stack, and by then it is already allocated.
        let ceiling = MAX_CONNECTIONS.load(Ordering::Relaxed);
        if LIVE_CONNECTIONS.fetch_add(1, Ordering::Relaxed) >= ceiling {
            LIVE_CONNECTIONS.fetch_sub(1, Ordering::Relaxed);
            println!("[pool] refused a connection: {ceiling} already open");
            continue;
        }
        let pool = Arc::clone(&pool);
        std::thread::spawn(move || {
            let _slot = ConnectionSlot;
            let peer = stream
                .peer_addr()
                .map(|a| a.to_string())
                .unwrap_or_else(|_| String::from("?"));
            if let Err(e) = handle(stream, pool, &peer) {
                println!("[pool] {peer} closed: {e}");
            }
        });
    }
    Ok(())
}

fn handle(mut stream: TcpStream, pool: Arc<Mutex<Pool>>, peer: &str) -> std::io::Result<()> {
    stream.set_read_timeout(Some(JOB_PERIOD))?;
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 4096];
    // Everything a miner is held to lives in here. This loop only moves bytes
    // between it and the socket, which is what lets a WebSocket move them too.
    let mut conn = Conn::new(peer);

    loop {
        let n = match stream.read(&mut chunk) {
            Ok(0) => return Ok(()),
            Ok(n) => n,
            // The read timeout is the job period, and a quiet period is when a
            // connection is given fresh work -- see `Conn::on_idle` for why that
            // is the *only* time, and the bug that re-issuing more often was.
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                let out = conn.on_idle(&pool);
                if deliver(&mut stream, out)? {
                    return Ok(());
                }
                continue;
            }
            Err(e) => return Err(e),
        };
        buf.extend_from_slice(&chunk[..n]);

        loop {
            let line = match take_line(&mut buf) {
                Ok(Some(l)) => l,
                Ok(None) => break,
                // A frame longer than `MAX_LINE` is not recoverable: the stream
                // is desynchronised and everything after it is a guess.
                Err(_) => return Ok(()),
            };
            let out = conn.on_line(&pool, &line);
            if deliver(&mut stream, out)? {
                return Ok(());
            }
        }
    }
}

/// Write what the core said, log what it noted. Answers whether to hang up.
fn deliver(stream: &mut TcpStream, out: Out) -> std::io::Result<bool> {
    for line in &out.log {
        println!("{line}");
    }
    for msg in &out.send {
        stream.write_all(msg.as_bytes())?;
    }
    Ok(out.close)
}
