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
        Ok(Helper { child, stdin, stdout, what: what.to_string(), job: None })
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

    pub fn has_job(&self) -> bool {
        self.job.is_some()
    }

    /// One round: the shares found (job, proof), the tiles searched, the milliseconds it took.
    pub fn round(&mut self) -> Result<(Vec<(String, String)>, u64, f64), String> {
        self.say("round\n")?;
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
        if !sock.recv(&mut buf, &mut chunk)? {
            return Err(String::from("the pool closed the connection"));
        }
        while let Ok(Some(line)) = take_line(&mut buf) {
            let Some(v) = Json::parse(line.trim()) else { continue };
            if v.get("method").and_then(|m| m.as_str()) == Some("glados.pearl.job") {
                let Some(p) = v.get("params") else { continue };
                let (Some(id), Some(header), Some(bits)) = (
                    p.get("job").and_then(|x| x.as_str()),
                    p.get("header").and_then(|x| x.as_str()),
                    p.get("bits").and_then(|x| x.as_i64()),
                ) else {
                    continue;
                };
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

        if !h.has_job() {
            std::thread::sleep(PARK);
            continue;
        }
        let (shares, tiles, ms) = h.round()?;
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
