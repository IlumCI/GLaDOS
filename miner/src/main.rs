//! A miner that speaks the pool's protocol and hashes on whatever is fastest.
//!
//! ```text
//! glados-miner --pool HOST:PORT --worker NAME [--xpu PATH] [--cpu]
//! ```
//!
//! The point of this program is that the pool has never had a second kind of
//! device on it. `design/pool.md`'s whole Part B claim is that the pool is
//! device-agnostic -- VarDiff per (connection, coin), a ledger denominated in
//! work rather than share count, the algorithm on the wire -- and all of that
//! was built and tested against one miner: the kernel, at a few hundred
//! kilohashes. A GPU at two-thirds of a gigahash is three and a half orders of
//! magnitude away, which is the honest test of every one of those decisions.
//!
//! ### The protocol is not implemented here
//!
//! It comes from `glados_pool::mine::proto`, which is the kernel's own
//! `src/mine/proto.rs` reached by `#[path]`. So the kernel, the pool and this
//! miner encode a share with one function. A protocol is where a second
//! implementation costs most and shows least -- a field read differently at
//! each end is a share rejected for a reason neither side can name -- and
//! `tools/poolclient.py` is the deliberate exception that exists to catch
//! exactly that, having already caught one byte-order bug by being separate.
//!
//! ### The GPU is a subprocess, not a linked library
//!
//! `cuda/xpu.cu` is built by `nvcc` and driven over a pipe. Linking it would
//! mean a `build.rs` that shells out to `nvcc`, a static library, and the CUDA
//! runtime dragged into this crate's link line -- a build with three ways to
//! silently produce yesterday's binary, on a project that has been bitten by
//! exactly that three times in one session. A pipe has one.
//!
//! It also means this program builds and runs with no CUDA toolkit at all,
//! which is what `--cpu` is for and what lets CI exercise everything except
//! the kernel launch itself.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

use glados_pool::json::Json;
use glados_pool::mine::algo::{Algo, Hasher};
use glados_pool::mine::hash::below_target;
use glados_pool::mine::proto;
use glados_pool::mine::stratum::{hex, take_line};
use glados_pool::mine::u256::U256;

/// How many nonces to ask for in one go.
///
/// **Measured, because the first guess cost a third of the card.** A scan is
/// synchronous -- the GPU sits idle while the reply crosses the pipe and this
/// program decides what to do -- so too small a batch spends its life in that
/// gap, and `design/xpu.md` records the other half: a kernel that runs for
/// tens of milliseconds never leaves the idle clock at all.
///
/// On the RTX 3050, through the pool, with the pool on the same host:
///
///     32M    292.7 MH/s
///     128M   377.7 MH/s
///     512M   424.4 MH/s
///
/// 128M is the default rather than 512M because the loop only reads the socket
/// between scans, so the batch is also how late a new job can be noticed --
/// about a third of a second here against one and a quarter. Work done on a
/// job the chain has moved past is worth nothing, and the trade stops being
/// worth it well before the last twelve percent.
///
/// None of these reach the 0.645 GH/s that file measured, and the reason is in
/// it: that figure needs an idle host, and here the pool is running beside the
/// miner on the same machine.
const GPU_BATCH: u32 = 128_000_000;
/// The same in the terms a single host core can manage.
const CPU_BATCH: u32 = 500_000;

/// What actually computes hashes.
trait Backend {
    fn name(&self) -> &str;
    /// Install a job. `false` if this backend cannot compute that algorithm.
    fn set_job(&mut self, algo: &Algo, header: &[u8; 80], target: &U256) -> bool;
    /// Scan `count` nonces from `base`. Answers a nonce **in protocol order**.
    fn scan(&mut self, base: u32, count: u32) -> Option<u32>;
    fn batch(&self) -> u32;
}

// ------------------------------------------------------------------ the GPU

struct Xpu {
    batch: u32,
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Xpu {
    fn start(path: &str) -> Result<Xpu, String> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| format!("{path}: {e}"))?;
        let stdin = child.stdin.take().ok_or("no stdin")?;
        let mut stdout = BufReader::new(child.stdout.take().ok_or("no stdout")?);
        let mut greeting = String::new();
        stdout
            .read_line(&mut greeting)
            .map_err(|e| format!("{path}: {e}"))?;
        if greeting.trim() != "ready" {
            return Err(format!("{path} said {:?} rather than ready", greeting.trim()));
        }
        Ok(Xpu { batch: GPU_BATCH, child, stdin, stdout })
    }

    fn ask(&mut self, line: &str) -> Result<String, String> {
        self.stdin
            .write_all(line.as_bytes())
            .map_err(|e| format!("write: {e}"))?;
        self.stdin.flush().map_err(|e| format!("flush: {e}"))?;
        let mut reply = String::new();
        self.stdout
            .read_line(&mut reply)
            .map_err(|e| format!("read: {e}"))?;
        Ok(reply.trim().to_string())
    }
}

impl Drop for Xpu {
    fn drop(&mut self) {
        let _ = self.stdin.write_all(b"quit\n");
        let _ = self.child.wait();
    }
}

impl Backend for Xpu {
    fn name(&self) -> &str {
        "gpu"
    }

    fn set_job(&mut self, algo: &Algo, header: &[u8; 80], target: &U256) -> bool {
        // The device is asked rather than told. `xpu.cu` knows which algorithms
        // it has and refuses the rest by name, so a third one arriving there
        // needs no change here -- and a device that quietly accepted work it
        // cannot compute would produce wrong shares at half a gigahash a
        // second, which `algo.cuh` records as the worst available outcome.
        let line = format!(
            "job {} {} {}\n",
            algo.name(),
            hex(header),
            hex(&target.to_be_bytes())
        );
        match self.ask(&line) {
            Ok(r) if r == "ok" => true,
            Ok(r) => {
                eprintln!("[gpu] refused a job: {r}");
                false
            }
            Err(e) => {
                eprintln!("[gpu] {e}");
                false
            }
        }
    }

    fn scan(&mut self, base: u32, count: u32) -> Option<u32> {
        // **Protocol nonces on the wire, both ways.** SHA-256 reads its message
        // big-endian and BLAKE2s little-endian, so the word each kernel indexes
        // by is a different view of the same four header bytes. That swap lives
        // in `xpu.cu` beside the algorithm needing it, not here: a caller that
        // had to know which algorithms swap is one that gets it wrong for the
        // third, and getting it wrong is not an error but a share the pool
        // rejects with nothing to say why.
        //
        // Pinned by the strongest vector there is: given block 125552's header
        // and its own target, the device answers `9546a142`, which is that
        // block's nonce as everybody quotes it.
        let line = format!("scan {:08x} {}\n", base, count);
        match self.ask(&line) {
            Ok(r) if r.starts_with("found ") => u32::from_str_radix(r[6..].trim(), 16).ok(),
            Ok(r) if r.starts_with("none") => None,
            Ok(r) => {
                eprintln!("[gpu] {r}");
                None
            }
            Err(e) => {
                eprintln!("[gpu] {e}");
                None
            }
        }
    }

    fn batch(&self) -> u32 {
        self.batch
    }
}

// ------------------------------------------------------------------ the CPU

/// One host core, using the kernel's own `Hasher`.
///
/// Not a fallback so much as the control: it computes every algorithm the
/// kernel does, so a job the GPU refuses is still worked, and a disagreement
/// between the two is visible on one machine.
struct Cpu {
    hasher: Option<Hasher>,
    header: [u8; 80],
    target: U256,
}

impl Backend for Cpu {
    fn name(&self) -> &str {
        "cpu"
    }

    fn set_job(&mut self, algo: &Algo, header: &[u8; 80], target: &U256) -> bool {
        match Hasher::new(algo, header) {
            Some(h) => {
                self.hasher = Some(h);
                self.header = *header;
                self.target = *target;
                true
            }
            None => false,
        }
    }

    fn scan(&mut self, base: u32, count: u32) -> Option<u32> {
        let h = self.hasher.as_mut()?;
        for i in 0..count {
            let n = base.wrapping_add(i);
            if below_target(&h.hash(&self.header, n), &self.target) {
                return Some(n);
            }
        }
        None
    }

    fn batch(&self) -> u32 {
        CPU_BATCH
    }
}

// ----------------------------------------------------------------- the loop

/// How long one visit to a slot should take.
///
/// **The unit of allocation is time, not hashes**, and that is the whole of
/// why several coins can share one device. A yespower hash is roughly a
/// thousand times the work of a sha256d hash, so a scheduler counting hashes
/// hands the slow algorithm the machine -- it looks permanently behind however
/// long it has run. Counting the clock is the only comparison that means the
/// same thing on both.
///
/// A quarter of a second is also, and not by coincidence, what the GPU batch
/// measurement landed on: 128M nonces at half a gigahash. So on a single-coin
/// pool this scheduler asks for about the batch the benchmark chose, and on a
/// multi-coin one it is also how late a new job can be noticed.
const SLICE: Duration = Duration::from_millis(250);

/// The first scan of an algorithm this device has not timed yet.
///
/// Small in absolute terms rather than a fraction of the batch, because the
/// two ends of the range are four orders of magnitude apart: 4096 nonces is
/// microseconds on the GPU and about twelve seconds of yespower on one core,
/// and any larger figure makes the *probe* the thing that blocks. The measured
/// rate then takes it to a full slice within two scans.
const PROBE: u32 = 4096;

/// How long a read may block. Small, because it is subtracted from the device.
const POLL: Duration = Duration::from_millis(2);

/// How long to wait when there is nothing this device can work.
///
/// A separate number from `POLL` and not a longer timeout on the socket,
/// because the two answer different questions: `POLL` is how much of a working
/// device is given up to check for news, and this is how often a *stopped* one
/// asks whether that is still true. Sleeping rather than spinning matters here
/// -- a miner with every slot refused would otherwise take a core to do
/// nothing, which on a machine also running the pool is the worst way to be
/// idle.
const PARK: Duration = Duration::from_millis(50);

/// Everything a job's nonce space holds. The nonce is four bytes at a fixed
/// offset, so this is all the work one job can ever carry.
const SPACE: u64 = 1u64 << 32;

struct Job {
    slot: u32,
    id: String,
    coin: String,
    algo: Algo,
    header: [u8; 80],
    target: U256,
    echo: proto::Echo,
}

/// One of the pool's coins, as this device sees it.
struct SlotState {
    job: Option<Job>,
    /// This device cannot compute the slot's algorithm. Cleared when the
    /// algorithm changes, and not when the job does.
    refused: bool,
    /// Where the search has reached in this job's space, in `[0, SPACE]`.
    /// A `u64` rather than the `u32` it is written into: the whole point is
    /// that reaching the end must be representable instead of wrapping.
    cursor: u64,
    /// Device time this slot has been given. What the share-out divides.
    spent: Duration,
    /// How much of the device this slot is meant to get, relative to the
    /// others. Mechanism only -- nothing here decides what a coin is worth.
    weight: u32,
    /// A `glados.work` is outstanding for this slot.
    asked: bool,
    /// The last job id announced, so switching between slots is not narrated.
    announced: String,
    hashes: u64,
    found: u64,
}

impl SlotState {
    fn new(weight: u32) -> SlotState {
        SlotState {
            job: None,
            refused: false,
            cursor: 0,
            spent: Duration::ZERO,
            weight,
            asked: false,
            announced: String::new(),
            hashes: 0,
            found: 0,
        }
    }

    /// Nonces left above the cursor. Zero means this job is finished, which is
    /// a thing that happens several times a minute on a fast device.
    fn left(&self) -> u64 {
        SPACE - self.cursor
    }

    fn workable(&self) -> bool {
        self.job.is_some() && !self.refused && self.left() > 0
    }
}

/// Measured hashes a second, per algorithm.
///
/// Per algorithm and not per slot, because it is a property of the device and
/// the arithmetic: two slots on sha256d run at the same rate, and carrying the
/// figure across means a new job costs no probe.
struct Rates(Vec<(String, f64)>);

impl Rates {
    fn get(&self, algo: &str) -> Option<f64> {
        self.0.iter().find(|(a, _)| a == algo).map(|(_, r)| *r)
    }

    fn set(&mut self, algo: &str, rate: f64) {
        match self.0.iter_mut().find(|(a, _)| a == algo) {
            Some(e) => e.1 = rate,
            None => self.0.push((String::from(algo), rate)),
        }
    }
}

/// **Share the device out by time, weighted.**
///
/// The loop took the first workable slot, which on a multi-coin pool is a
/// device that works slot zero and nothing else: a three-coin run put every
/// one of five hundred megahashes on `btc` and never touched the blake2s coin
/// beside it. Least-virtual-time is the smallest scheduler that fixes it and
/// stays deterministic -- the slot furthest behind its share goes next, ties by
/// slot number, so two runs against one pool make the same choices.
///
/// **Weights are the whole of the policy interface and nothing here sets
/// them.** Even by default; profitability when there are numbers to put in
/// them, which is `mine ev`'s job and not this loop's. Keeping the two apart is
/// the point -- a scheduler that also decided what a coin was worth would have
/// to be rewritten every time the answer changed.
///
/// **What the weights should be, when something does set them.** Soria, Moya
/// and Mohazab (*Finance Research Letters* 53:103610) model mining as a Tullock
/// contest and their first-order condition is the ordinary one: buy hash until
/// marginal revenue equals marginal cost. Where the scarce thing is
/// device-seconds rather than hashes, that says give the next second to the
/// coin with the highest marginal revenue per second and equalise marginal
/// revenue across the coins being run -- which is `payrate.py`'s measured
/// dollars per day, and needs no learner, because the reward is quoted rather
/// than unknown.
///
/// Their own individual-rationality constraint is worth writing down beside it,
/// since it is about this machine specifically. Mining happens only while
/// profit is positive, and at a corner the optimum is zero hash: in their
/// asymmetric simulation the high-cost miner "chooses the hash value of zero at
/// the end". This laptop earns $0.07 a day against $0.36 of electricity, so the
/// model's answer for the whole device is `h* = 0`. A profitability-weighted
/// `choose` is therefore solving a subproblem whose outer problem has a corner
/// solution, and that is a fact about the subproblem rather than an objection
/// to solving it.
///
/// Pure, and separate from the loop for that reason: what it does is
/// arithmetic over a table, and arithmetic is the half that can be checked
/// without a pool, a socket or a GPU.
fn choose(slots: &[SlotState]) -> Option<usize> {
    slots
        .iter()
        .enumerate()
        .filter(|(_, s)| s.workable())
        .min_by_key(|(i, s)| (s.spent.as_nanos() / s.weight as u128, *i))
        .map(|(i, _)| i)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pool = String::from("127.0.0.1:3334");
    let mut worker = String::from("glados-miner");
    let mut xpu_path = String::from("../cuda/xpu.exe");
    let mut force_cpu = false;
    let mut batch: Option<u32> = None;
    let mut weights: Vec<(String, u32)> = Vec::new();

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--pool" => pool = it.next().cloned().unwrap_or(pool),
            "--worker" => worker = it.next().cloned().unwrap_or(worker),
            "--xpu" => xpu_path = it.next().cloned().unwrap_or(xpu_path),
            "--cpu" => force_cpu = true,
            "--batch" => batch = it.next().and_then(|v| v.parse().ok()),
            // Repeatable, `COIN:N`, matched on the coin label the pool sends.
            // By label rather than by slot number because a slot is the pool's
            // own index into a table an operator may reorder, and a share-out
            // that silently moved to a different coin when they did would be
            // the worst available failure: it changes what is mined and reports
            // nothing.
            "--weight" => {
                let Some(v) = it.next() else { continue };
                let Some((coin, n)) = v.split_once(':') else {
                    eprintln!("--weight wants COIN:N, got {v}");
                    std::process::exit(2);
                };
                match n.parse::<u32>() {
                    Ok(w) if w > 0 => weights.push((coin.to_string(), w)),
                    _ => {
                        eprintln!("--weight {v}: N must be a positive integer");
                        std::process::exit(2);
                    }
                }
            }
            "-h" | "--help" => {
                println!(
                    "glados-miner --pool HOST:PORT --worker NAME [--xpu PATH] [--cpu] \
                     [--batch N] [--weight COIN:N]..."
                );
                return;
            }
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
    }

    let mut backend: Box<dyn Backend> = if force_cpu {
        Box::new(Cpu { hasher: None, header: [0u8; 80], target: U256::ZERO })
    } else {
        match Xpu::start(&xpu_path) {
            Ok(mut x) => {
                if let Some(b) = batch {
                    x.batch = b;
                }
                println!("[miner] gpu ready via {xpu_path}, up to {} nonces a scan", x.batch);
                Box::new(x)
            }
            Err(e) => {
                // Said out loud and then carried on. A miner that silently fell
                // back to one core would report a rate three orders of
                // magnitude below what was expected and look like a broken GPU
                // rather than an absent one.
                eprintln!("[miner] no gpu ({e}); hashing on this core instead");
                Box::new(Cpu { hasher: None, header: [0u8; 80], target: U256::ZERO })
            }
        }
    };

    // **Reconnect, as the kernel's miner does.** A pool restarts -- a Cloudflare
    // Durable Object is reset every time its code is deployed -- and a miner that
    // exits on the first dropped connection is a machine that stops earning until
    // somebody notices. Backoff doubles to a minute, and resets once a session has
    // lasted long enough to count as a working one.
    let mut backoff = std::time::Duration::from_secs(2);
    loop {
        let began = std::time::Instant::now();
        match run(&pool, &worker, backend.as_mut(), &weights) {
            Ok(()) => eprintln!("[miner] disconnected; reconnecting in {}s", backoff.as_secs()),
            Err(e) => eprintln!("[miner] {e}; retrying in {}s", backoff.as_secs()),
        }
        if began.elapsed() > std::time::Duration::from_secs(60) {
            backoff = std::time::Duration::from_secs(2);
        }
        std::thread::sleep(backoff);
        backoff = (backoff * 2).min(std::time::Duration::from_secs(60));
    }
}

fn run(
    pool: &str,
    worker: &str,
    backend: &mut dyn Backend,
    weights: &[(String, u32)],
) -> Result<(), String> {
    let mut sock = TcpStream::connect(pool).map_err(|e| format!("{pool}: {e}"))?;
    // **The read timeout is device time, and it was a fifth of the card.**
    // The loop reads the socket once per scan, so a fifty-millisecond timeout
    // against a two-hundred-and-fifty-millisecond slice is a sixth of the
    // machine spent waiting for a pool that usually has nothing to say -- and
    // it showed as exactly that: two coins measured at 37% and 36% of the
    // wall clock with nothing accounting for the rest.
    //
    // Two milliseconds instead, with `park` below covering the case the long
    // timeout was really there for. Writes are unaffected; only reads carry
    // this.
    sock.set_read_timeout(Some(POLL))
        .map_err(|e| format!("{e}"))?;
    sock.write_all(
        proto::encode_hello(1, worker, concat!("glados-miner/", env!("CARGO_PKG_VERSION")))
            .as_bytes(),
    )
    .map_err(|e| format!("hello: {e}"))?;
    println!("[miner] {worker} -> {pool}, hashing on the {}", backend.name());

    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 16384];
    // Indexed by the pool's slot number, so a device that can work one coin
    // and not another keeps the one it can.
    let mut slots: Vec<SlotState> = Vec::new();
    let mut rates = Rates(Vec::new());
    let mut installed = String::new();
    let mut accepted = 0u64;
    let mut duplicate = 0u64;
    let mut since = Instant::now();
    let mut next_id = 2u64;

    loop {
        match sock.read(&mut chunk) {
            Ok(0) => return Err(String::from("the pool closed the connection")),
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut => {}
            Err(e) => return Err(format!("read: {e}")),
        }

        while let Ok(Some(line)) = take_line(&mut buf) {
            let Some(v) = Json::parse(line.trim()) else { continue };
            if v.get("method").and_then(|m| m.as_str()) == Some("glados.job") {
                let Some(p) = v.get("params").and_then(proto::parse_job) else {
                    continue;
                };
                // The proof is checked when one is offered, exactly as the
                // kernel does it -- `proves` is the same function. A pool that
                // misrepresents what it pays should not get this device's
                // hashes any more than it gets the kernel's.
                if let Some(pr) = &p.proof {
                    if !proto::proves(pr, &p.header) {
                        eprintln!("[miner] a job's proof does not match its header; refused");
                        continue;
                    }
                }
                let n = p.slot as usize;
                let j = Job {
                    slot: p.slot,
                    id: p.job,
                    coin: p.coin,
                    algo: p.algo,
                    header: p.header,
                    target: p.target,
                    echo: p.echo,
                };
                while slots.len() <= n {
                    slots.push(SlotState::new(1));
                }
                let st = &mut slots[n];
                // The weight is matched on the label, which is not known until
                // the first job for a slot arrives.
                if let Some((_, w)) = weights.iter().find(|(c, _)| *c == j.coin) {
                    st.weight = *w;
                }
                // A slot that was refused before is retried when its *algorithm*
                // changes, and not otherwise. Retrying every job would mean a
                // pipe round trip per refusal on a pool serving coins this
                // device cannot work, which on a busy table is most of them.
                if st.job.as_ref().map(|o| o.algo != j.algo).unwrap_or(true) {
                    st.refused = false;
                }
                if st.announced != j.id {
                    println!(
                        "[miner] job {} on {} ({}), slot {}, weight {}",
                        j.id,
                        j.coin,
                        j.algo.name(),
                        j.slot,
                        st.weight
                    );
                    st.announced = j.id.clone();
                }
                st.job = Some(j);
                st.cursor = 0;
                st.asked = false;
            } else if let Some(r) = v.get("result") {
                if r.get("slots").is_some() {
                    println!("[miner] welcomed");
                } else if let Some(verdict) = r.get("verdict").and_then(|x| x.as_str()) {
                    if r.get("ok").and_then(|x| x.as_bool()).unwrap_or(false) {
                        accepted += 1;
                    }
                    if verdict == "duplicate" {
                        duplicate += 1;
                    }
                    println!("[miner] share {verdict}");
                }
            }
        }

        // **A job is a finite search, and this device finishes one every few
        // seconds.** The nonce is four bytes, so a job carries 2^32 hashes and
        // no more -- eight and a half seconds at half a gigahash, against a
        // pool that re-issues every thirty. The cursor used to be a `u32` that
        // wrapped, so the miner rescanned the space it had already searched and
        // resubmitted what it found there: measured at 35 duplicates against 23
        // accepted, every repeated nonce arriving exactly six times, which is
        // about the wrap count for a 128M batch over a run of that length.
        //
        // So the end of the space is reachable rather than wrapping, and
        // reaching it is something to *say* rather than to work through again.
        for i in 0..slots.len() {
            let st = &mut slots[i];
            if st.job.is_some() && !st.refused && st.left() == 0 && !st.asked {
                sock.write_all(proto::encode_work(next_id, i as u32).as_bytes())
                    .map_err(|e| format!("work: {e}"))?;
                next_id += 1;
                st.asked = true;
            }
        }

        // **Share the device out by time, weighted.**
        //
        // This took the first workable slot, which on a multi-coin pool is a
        // device that works slot zero and nothing else: a three-coin run put
        // every one of five hundred megahashes on `btc` and never touched the
        // blake2s coin beside it. Least-virtual-time is the smallest scheduler
        // that fixes it and stays deterministic -- the slot furthest behind its
        // share goes next, ties by slot number, so two runs against one pool
        // make the same choices.
        //
        // Weights are the whole of the policy interface and nothing here sets
        // them. Even by default; profitability when there are numbers to put in
        // them, which is `mine ev`'s job and not this loop's.
        let Some(slot) = choose(&slots) else {
            std::thread::sleep(PARK);
            continue;
        };

        {
            let st = &slots[slot];
            let j = st.job.as_ref().unwrap();
            if installed != j.id {
                if !backend.set_job(&j.algo, &j.header, &j.target) {
                    // Not fatal, and the ordinary case for a single-algorithm
                    // device on a multi-algorithm pool. The slot is marked and
                    // the loop moves to whatever else is on offer.
                    if !st.refused {
                        println!(
                            "[miner] the {} cannot compute {} for {}; leaving that coin alone",
                            backend.name(),
                            j.algo.name(),
                            j.coin
                        );
                    }
                    slots[slot].refused = true;
                    continue;
                }
                installed = j.id.clone();
            }
        }

        // How many nonces buy about one slice. An algorithm this device has not
        // timed gets the probe, and the measurement below takes it to a full
        // slice within two scans.
        let st = &slots[slot];
        let j = st.job.as_ref().unwrap();
        let algo = j.algo.name().to_string();
        let want = match rates.get(&algo) {
            Some(r) => ((r * SLICE.as_secs_f64()) as u64).clamp(1, backend.batch() as u64),
            None => PROBE as u64,
        };
        // Never past the end of the job. This is the line the duplicates came
        // through.
        let count = want.min(st.left()) as u32;
        let base = st.cursor as u32;

        let t0 = Instant::now();
        let hit = backend.scan(base, count);
        let dt = t0.elapsed();

        // Below a millisecond the pipe round trip is most of what was measured,
        // so the figure would describe this process rather than the device. The
        // probe grows geometrically through those instead.
        if dt.as_secs_f64() > 0.001 {
            rates.set(&algo, count as f64 / dt.as_secs_f64());
        } else {
            rates.set(&algo, count as f64 * 8.0 / SLICE.as_secs_f64());
        }

        let st = &mut slots[slot];
        st.spent += dt;
        st.hashes += count as u64;
        st.cursor += count as u64;

        if let Some(n) = hit {
            st.found += 1;
            let j = st.job.as_ref().unwrap();
            let share = proto::Share {
                job: j.id.clone(),
                nonce: n,
                echo: j.echo.clone(),
            };
            sock.write_all(proto::encode_submit(next_id, &share).as_bytes())
                .map_err(|e| format!("submit: {e}"))?;
            next_id += 1;
            // Resume above it rather than past the batch: the device answers
            // the *lowest* qualifying nonce in a range, so the rest of that
            // range may hold more and has not really been searched. Rewinding
            // the cursor is therefore correct and is not what produced the
            // duplicates -- wrapping at the top of the space was.
            st.cursor = n as u64 + 1;
        }

        if since.elapsed() >= Duration::from_secs(10) {
            let secs = since.elapsed().as_secs_f64();
            let total: u64 = slots.iter().map(|s| s.hashes).sum();
            println!(
                "[miner] {total} hashes in {secs:.1}s, {accepted} accepted, {duplicate} duplicate"
            );
            // Per slot, because there is no meaningful aggregate rate across
            // algorithms -- adding a yespower hash to a sha256d hash is adding
            // two numbers a thousand times apart and calling the sum a speed.
            for (i, s) in slots.iter().enumerate() {
                let Some(j) = s.job.as_ref() else { continue };
                if s.spent.is_zero() {
                    continue;
                }
                println!(
                    "[miner]   slot {i} {} ({}) w{}: {:.3} MH/s over {} hashes, {:.0}% of the device, {} found",
                    j.coin,
                    j.algo.name(),
                    s.weight,
                    s.hashes as f64 / s.spent.as_secs_f64() / 1e6,
                    s.hashes,
                    s.spent.as_secs_f64() / secs * 100.0,
                    s.found
                );
            }
            for s in slots.iter_mut() {
                s.hashes = 0;
                s.found = 0;
                // The share-out is over a window rather than over all time, or
                // a coin added an hour in would monopolise the device until it
                // had caught up on an hour it did not exist for.
                s.spent = Duration::ZERO;
            }
            since = Instant::now();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A slot with a job on it, which is all `choose` reads.
    fn slot(weight: u32) -> SlotState {
        let mut s = SlotState::new(weight);
        s.job = Some(Job {
            slot: 0,
            id: String::from("x"),
            coin: String::from("c"),
            algo: Algo::Sha256d,
            header: [0u8; 80],
            target: U256::ZERO,
            echo: Vec::new(),
        });
        s
    }

    /// Run the scheduler for `rounds` visits of `each`, and answer how much
    /// device time every slot ended up with. This is the loop's own shape with
    /// the hashing taken out.
    fn share_out(slots: &mut Vec<SlotState>, rounds: usize, each: Duration) -> Vec<u128> {
        for _ in 0..rounds {
            let i = choose(slots).expect("something to work");
            slots[i].spent += each;
        }
        slots.iter().map(|s| s.spent.as_nanos()).collect()
    }

    #[test]
    fn even_weights_split_the_device_evenly() {
        let mut s = vec![slot(1), slot(1)];
        let got = share_out(&mut s, 100, Duration::from_millis(250));
        assert_eq!(got[0], got[1]);
    }

    #[test]
    fn a_weight_of_three_takes_three_times_the_device() {
        let mut s = vec![slot(3), slot(1)];
        let got = share_out(&mut s, 400, Duration::from_millis(250));
        // Exact rather than approximate: every visit is the same length here,
        // so the ratio is arithmetic and a tolerance would hide an off-by-one
        // that a real run would show as a coin quietly getting less than it
        // was promised.
        assert_eq!(got[0], got[1] * 3);
    }

    /// Slices of wildly different lengths are the case this exists for: a
    /// yespower scan is a thousand times a sha256d scan in hashes and about
    /// the same in seconds, and only the seconds are comparable.
    #[test]
    fn unequal_slice_lengths_still_converge_on_the_weights() {
        let mut s = vec![slot(1), slot(1)];
        let mut lengths = [7u64, 300, 41, 900, 13].iter().cycle();
        for _ in 0..500 {
            let i = choose(&s).unwrap();
            let ms = *lengths.next().unwrap();
            s[i].spent += Duration::from_millis(ms);
        }
        let a = s[0].spent.as_secs_f64();
        let b = s[1].spent.as_secs_f64();
        // One slice of drift is all that is available: the scheduler cannot
        // know how long the next one will be, so it can only ever be behind by
        // the one it just handed out.
        assert!((a - b).abs() < 1.0, "{a} against {b}");
    }

    #[test]
    fn a_refused_slot_is_never_chosen() {
        let mut s = vec![slot(1), slot(1)];
        s[0].refused = true;
        assert_eq!(choose(&s), Some(1));
        s[1].refused = true;
        assert_eq!(choose(&s), None);
    }

    /// **The nonce space of a job is finite and reaching the end is
    /// representable.** This is the duplicate bug as a claim: the cursor was a
    /// `u32` that wrapped, so a device fast enough to finish a job rescanned
    /// it from zero and resubmitted what it found. Measured before the fix at
    /// 35 duplicates against 23 accepted.
    #[test]
    fn a_spent_job_is_finished_rather_than_wrapped() {
        let mut s = vec![slot(1)];
        assert_eq!(s[0].left(), SPACE);
        s[0].cursor = SPACE - 1;
        assert_eq!(s[0].left(), 1);
        assert!(s[0].workable());
        s[0].cursor = SPACE;
        assert_eq!(s[0].left(), 0);
        assert!(!s[0].workable());
        assert_eq!(choose(&s), None);
        // And the cursor still fits the four bytes it is written into, which
        // is what makes the last scan of a job a legal one rather than an
        // empty one at zero.
        assert_eq!((SPACE - 1) as u32, u32::MAX);
    }

    /// A slot with no job is not a slot with nothing left to do -- the two look
    /// identical to a scheduler reading only the cursor, and only one of them
    /// should ask the pool for work.
    #[test]
    fn a_slot_with_no_job_is_not_workable() {
        let s = vec![SlotState::new(1)];
        assert!(!s[0].workable());
        assert_eq!(choose(&s), None);
    }
}
