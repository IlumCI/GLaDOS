//! `--algo pearl`: the GPU mining Pearl through the CUDA helper `cuda/pearl.cu`.
//!
//! The helper does a whole round on the device and prints a proof for every
//! tile under the share bound; this side holds the connection to the pool's
//! Pearl shard (`/mine/pearl`), hands the helper each job, and submits what it
//! finds. Unlike the nonce coins there is nothing to schedule -- one job, one
//! device, rounds back to back -- so this is a loop of its own rather than a
//! `Backend`.
//!
//!     pool -> {"method":"glados.pearl.job","params":{"job","header","bits",...}}
//!     helper <- job <id> <header> <bits>      helper <- round
//!     helper -> share <id> <proof>            helper -> done <tiles> <ms>
//!     pool <- {"id":n,"method":"glados.pearl.submit","params":{"job","proof"}}
//!
//! Rates are multiply-accumulates a second, which is what Pearl calls a hash.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

use glados_pool::json::Json;
use glados_pool::mine::proto;
use glados_pool::mine::stratum::take_line;

use crate::{rate, Link, PARK};

const MAC_PER_TILE: u64 = 16 * 16 * 2048;

pub struct Helper {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    pub what: String,
    job: Option<(String, String, i64)>,
    /// A round asked for and not yet read back. Kept across sessions: a round
    /// in flight when the pool goes is read by the next session, so the pipe
    /// never gets a round ahead of its reader.
    busy: bool,
}

impl Helper {
    pub fn start(path: &str) -> Result<Helper, String> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| format!("{path}: {e}"))?;
        let stdin = child.stdin.take().ok_or("no stdin")?;
        let mut stdout = BufReader::new(child.stdout.take().ok_or("no stdout")?);
        let mut first = String::new();
        stdout.read_line(&mut first).map_err(|e| format!("{path}: {e}"))?;
        let first = first.trim().to_string();
        let Some(what) = first.strip_prefix("ready ") else {
            return Err(format!("{path} said {first:?} rather than ready"));
        };
        Ok(Helper { child, stdin, stdout, what: what.to_string(), job: None, busy: false })
    }

    fn say(&mut self, line: &str) -> Result<(), String> {
        self.stdin.write_all(line.as_bytes()).and_then(|_| self.stdin.flush()).map_err(|e| format!("pearl helper: {e}"))
    }

    /// Install a job, if it is not the one already installed.
    pub fn job(&mut self, id: &str, header: &str, bits: i64) -> Result<(), String> {
        let j = (id.to_string(), header.to_string(), bits);
        if self.job.as_ref() == Some(&j) {
            return Ok(());
        }
        self.say(&format!("job {id} {header} {bits}\n"))?;
        self.job = Some(j);
        Ok(())
    }

    /// Whether the helper process is still running: a GPU reset or a crash
    /// ends it, and the miner starts another rather than retrying a dead pipe.
    pub fn alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    pub fn has_job(&self) -> bool {
        self.job.is_some()
    }

    /// Start a round, unless one is already running. The GPU works on it while
    /// this side reads the pool; `finish` collects it.
    pub fn begin(&mut self) -> Result<(), String> {
        if !self.busy {
            self.say("round\n")?;
            self.busy = true;
        }
        Ok(())
    }

    pub fn busy(&self) -> bool {
        self.busy
    }

    /// The round started: the shares found (job, proof), the tiles searched, the milliseconds it took.
    pub fn finish(&mut self) -> Result<(Vec<(String, String)>, u64, f64), String> {
        let mut shares = Vec::new();
        loop {
            let mut l = String::new();
            if self.stdout.read_line(&mut l).map_err(|e| format!("pearl helper: {e}"))? == 0 {
                return Err(String::from("the pearl helper exited"));
            }
            let mut w = l.split_whitespace();
            match w.next() {
                Some("share") => {
                    if let (Some(j), Some(p)) = (w.next(), w.next()) {
                        shares.push((j.to_string(), p.to_string()));
                    }
                }
                Some("done") => {
                    self.busy = false;
                    let tiles = w.next().and_then(|x| x.parse().ok()).unwrap_or(0);
                    let ms = w.next().and_then(|x| x.parse().ok()).unwrap_or(0.0);
                    return Ok((shares, tiles, ms));
                }
                Some("error") => return Err(format!("pearl helper: {}", l.trim())),
                _ => {}
            }
        }
    }
}

impl Drop for Helper {
    fn drop(&mut self) {
        let _ = self.stdin.write_all(b"quit\n");
        let _ = self.child.wait();
    }
}

/// A share bound in bits: a JSON number written as a plain integer. `as_i64`
/// reads 3.5 as 3 and 1e2 as 1, which for a bound is a different job.
fn whole(v: &Json) -> Option<i64> {
    match v {
        Json::Num(t) if !t.is_empty() && t.len() <= 3 && t.bytes().all(|c| c.is_ascii_digit()) => t.parse().ok(),
        _ => None,
    }
}

/// One session with the pool's Pearl shard. Returns when the connection ends.
pub fn run(pool: &str, worker: &str, reward: Option<&str>, duty: f64, h: &mut Helper) -> Result<(), String> {
    let mut sock = Link::open(pool)?;
    let agent = concat!("glados-miner/", env!("CARGO_PKG_VERSION"));
    let hello = match reward {
        Some(r) => proto::encode_hello_reward(1, worker, agent, r),
        None => proto::encode_hello(1, worker, agent),
    };
    sock.send(&hello).map_err(|e| format!("hello: {e}"))?;
    println!("[pearl] {worker} -> {pool}, paid in {}, on {}", reward.unwrap_or("the pool's default"), h.what);

    let (mut buf, mut chunk) = (Vec::new(), [0u8; 16384]);
    let (mut next_id, mut accepted, mut rejected, mut found) = (2u64, 0u64, 0u64, 0u64);
    let mut pending: Vec<u64> = Vec::new();
    let (mut macs, mut since) = (0u64, Instant::now());
    loop {
        // The next round goes to the GPU first; reading the pool -- a short
        // timed wait -- then overlaps it instead of idling the device. A job
        // that arrives meanwhile reaches the helper after this round, whose
        // shares name the job they were found on.
        if h.has_job() {
            h.begin()?;
        }
        if !sock.recv(&mut buf, &mut chunk)? {
            return Err(String::from("the pool closed the connection"));
        }
        // An overlong line ends the session: left in the buffer it would grow
        // with every read and stop every later line from being seen.
        while let Some(line) = take_line(&mut buf).map_err(|_| String::from("the pool sent an overlong line"))? {
            let Some(v) = Json::parse(line.trim()) else { continue };
            if v.get("method").and_then(|m| m.as_str()) == Some("glados.maintenance") {
                crate::maintenance_notice(&v);
                continue;
            }
            if v.get("method").and_then(|m| m.as_str()) == Some("glados.pearl.job") {
                let Some(p) = v.get("params") else { continue };
                let (Some(id), Some(header), Some(bits)) = (
                    p.get("job").and_then(|x| x.as_str()),
                    p.get("header").and_then(|x| x.as_str()),
                    p.get("bits").and_then(whole),
                ) else {
                    continue;
                };
                // The id goes to the helper as one word of a line and back to the
                // pool inside a JSON string, so it is held to characters that are
                // safe in both: no space, newline, quote or backslash.
                if id.is_empty() || id.len() > 128 || !id.bytes().all(|c| c.is_ascii_graphic() && c != b'"' && c != b'\\') {
                    eprintln!("[pearl] a job id this miner will not pass on; ignored");
                    continue;
                }
                if header.len() != 152 || !header.bytes().all(|c| c.is_ascii_hexdigit()) || !(0..256).contains(&bits) {
                    eprintln!("[pearl] a job this miner cannot read; ignored");
                    continue;
                }
                h.job(id, header, bits)?;
            } else if v.get("id").and_then(|x| x.as_i64()) == Some(1) {
                if let Some(e) = v.get("error").filter(|e| !e.is_null()) {
                    return Err(format!("the pool refused this miner: {}", e.as_str().unwrap_or("(see the pool)")));
                }
                println!("[pearl] welcomed");
            } else if let Some(id) = v.get("id").and_then(|x| x.as_i64()) {
                if let Some(i) = pending.iter().position(|&p| p as i64 == id) {
                    pending.swap_remove(i);
                    let r = v.get("result");
                    let ok = r.and_then(|r| r.get("ok")).and_then(|x| x.as_bool()).unwrap_or(false);
                    let verdict = r.and_then(|r| r.get("verdict")).and_then(|x| x.as_str()).unwrap_or("?");
                    if ok { accepted += 1 } else { rejected += 1 }
                    let why = r.and_then(|r| r.get("why")).and_then(|x| x.as_str()).map(|w| format!(": {w}")).unwrap_or_default();
                    println!("[pearl] share {verdict}{why}");
                }
            }
        }

        if !h.busy() {
            std::thread::sleep(PARK);
            continue;
        }
        let (shares, tiles, ms) = h.finish()?;
        macs += tiles * MAC_PER_TILE;
        for (job, proof) in shares {
            found += 1;
            let id = next_id;
            next_id += 1;
            pending.push(id);
            if pending.len() > 64 {
                pending.remove(0);
            }
            sock.send(&format!("{{\"id\":{id},\"method\":\"glados.pearl.submit\",\"params\":{{\"job\":\"{job}\",\"proof\":\"{proof}\"}}}}\n"))
                .map_err(|e| format!("submit: {e}"))?;
        }
        if duty < 1.0 {
            std::thread::sleep(Duration::from_secs_f64(ms / 1000.0 * (1.0 - duty) / duty));
        }
        if since.elapsed() >= Duration::from_secs(10) {
            let rate_now = macs as f64 / since.elapsed().as_secs_f64();
            println!("[pearl] {} ({} a round), {found} found, {accepted} accepted, {rejected} rejected", rate(rate_now), format!("{ms:.1} ms"));
            if sock.is_edge() {
                sock.send(&format!("{{\"method\":\"glados.rate\",\"params\":{{\"hs\":{}}}}}\n", rate_now.round() as u64))
                    .map_err(|e| format!("rate: {e}"))?;
            }
            macs = 0;
            since = Instant::now();
        }
    }
}
