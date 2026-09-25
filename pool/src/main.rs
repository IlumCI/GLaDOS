//! `glados-pool`, the thing GLaDOS mines against.
//!
//! ```text
//! glados-pool [--listen ADDR] [--ledger PATH] [COIN ...]
//! glados-pool --selftest
//! glados-pool --bench
//! ```
//!
//! A coin is one token, `label:algo:bits[@host:port,user[,pass]]`.
//!
//! The algorithm is parameters and not a name -- BitZeny, Yenten and Koto all
//! run yespower and all run it differently, so a preset table would be this
//! program asserting numbers about somebody else's chain. `bits` is how many
//! leading zero bits a *share* must have, which is a difficulty said in a way
//! that has no float in it, and it is deliberately not upstream's difficulty:
//! ours decides what a miner is credited for, upstream's decides what is worth
//! forwarding.
//!
//! Without the `@` part the pool builds its own headers and there is no chain
//! behind the coin. With it, work comes from a real Stratum V1 pool.
//!
//! ```text
//! testnet:sha256d:20
//! yenten:yespower-10-2048-32-59656e74656e:16
//! bitzeny:yespower-10-2048-8:16@stratum.example.com:3333,waLLet.rig1,x
//! ```

use std::sync::{Arc, Mutex};

use glados_pool::mine::algo::Algo;
use glados_pool::mine::stratum::unhex;
use glados_pool::pool::{target_with_leading_zeros, Coin, Pool, Source};
use glados_pool::server;

fn parse_coin(spec: &str) -> Result<Coin, String> {
    // Split the upstream off first. `@` rather than another `:`, because a
    // `host:port` already contains one and a positional parser would have to
    // count colons from the right to tell them apart -- which breaks the first
    // time somebody omits the port.
    let (spec, source) = match spec.split_once('@') {
        None => (spec, Source::Local),
        Some((left, up)) => {
            let mut f = up.split(',');
            let hostport = f.next().unwrap_or("");
            let user = f.next().unwrap_or("");
            // Most pools ignore the password entirely and the convention is a
            // single `x`. Defaulted rather than required, since demanding a
            // field nobody reads is how a config gets copied wrong.
            let pass = f.next().unwrap_or("x");
            if user.is_empty() {
                return Err(format!("'{up}' has no worker name after the host"));
            }
            let (host, port) = match hostport.rsplit_once(':') {
                Some((h, p)) => match p.parse::<u16>() {
                    Ok(n) => (h.to_string(), n),
                    Err(_) => return Err(format!("'{p}' is not a port")),
                },
                None => return Err(format!("'{hostport}' needs a :port")),
            };
            (
                left,
                Source::Upstream {
                    host,
                    port,
                    user: String::from(user),
                    pass: String::from(pass),
                },
            )
        }
    };

    let mut it = spec.split(':');
    let label = it.next().unwrap_or("").trim();
    let algo_s = it.next().unwrap_or("");
    let bits_s = it.next().unwrap_or("");
    // A fourth field, optional, naming the traded asset. Positional and last
    // so every configuration written before it still parses -- and defaulting
    // to the label, so a pool whose coins are named as `prices.py` names them
    // needs nothing at all.
    let asset = it.next().unwrap_or(label).trim().to_string();
    if label.is_empty() || algo_s.is_empty() || bits_s.is_empty() {
        return Err(format!("'{spec}' is not label:algo:bits"));
    }
    let bits: u32 = bits_s
        .parse()
        .map_err(|_| format!("'{bits_s}' is not a number of bits"))?;
    // 256 leading zero bits is a target of zero, which nothing ever meets. A
    // pool configured that way accepts no share ever and looks exactly like a
    // pool with a broken hash, so it is refused at the argument instead.
    if bits >= 256 {
        return Err(format!("{bits} leading bits is a target nothing can meet"));
    }

    // One parser, in `record`, because a share record has to spell the
    // algorithm in exactly this form -- and two parsers for one format is what
    // `differ.rs` exists to catch elsewhere in this tree.
    let algo = glados_pool::record::parse_algo(algo_s)?;

    Ok(Coin {
        label: String::from(label),
        asset,
        algo,
        share_bits: bits,
        share_target: target_with_leading_zeros(bits),
        // Filled from upstream's `set_difficulty` when there is an upstream,
        // and `None` otherwise -- rather than a plausible constant, because an
        // expected value derived from an invented difficulty is worse than one
        // that refuses to print.
        network_target: None,
        source,
        work: None,
        e2: 0,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut roster_path: Option<String> = None;
    let mut roster_url = String::new();
    let mut require_roster = false;
    if args.iter().any(|a| a == "--selftest") {
        // The roster first, and separately, because it needs no socket: a
        // suite that binds a port before checking pure functions fails for the
        // wrong reason on a machine where the port is busy.
        let (rp, rf) = glados_pool::roster::checks();
        println!("[roster] {rp} passed, {rf} failed");
        // The share record, for the same reason and in the same place: pure
        // functions before anything binds a port. These are what the
        // verification job trusts before it trusts a log, and the one that earns
        // its place is the algorithm spec round-trip -- `Algo::name()` drops
        // yespower's N and r, so a record written from it would verify against a
        // different hash and every share would fail for a reason nothing reports.
        let mut recf = 0;
        let recs = glados_pool::record::checks();
        for (good, what) in &recs {
            println!("{}  {}", if *good { "ok  " } else { "FAIL" }, what);
            if !*good {
                recf += 1;
            }
        }
        println!("[record] {} passed, {} failed", recs.len() - recf, recf);
        std::process::exit(if selftest() && rf == 0 && recf == 0 { 0 } else { 1 });
    }
    if args.iter().any(|a| a == "--bench") {
        bench();
        return;
    }
    // **The verifier, which is the half a pool operator cannot be trusted with.**
    // It takes a share log -- the one `--sharelog` writes -- and recomputes every
    // hash in it with the kernel's own code, reached by `#[path]`. Nothing about
    // it needs the pool: no socket, no job table, no state. That is what lets it
    // run somewhere the operator does not control, which is the entire point.
    //
    // Exits non-zero on the first disagreement so a CI job fails rather than
    // printing a wall of text somebody has to read.
    if let Some(i) = args.iter().position(|a| a == "--verify") {
        let path = match args.get(i + 1) {
            Some(p) => p.clone(),
            None => {
                eprintln!("--verify wants a share log written by --sharelog");
                std::process::exit(2);
            }
        };
        std::process::exit(verify_log(&path));
    }

    let mut listen = String::from("0.0.0.0:3334");
    let mut lie = false;
    let mut ledger: Option<String> = None;
    let mut sharelog: Option<std::path::PathBuf> = None;
    let mut prices: Option<String> = None;
    let mut window: Option<u64> = None;
    let mut max_conns: Option<usize> = None;
    let mut share_secs: Option<u64> = None;
    let mut cpu_percent: f64 = 50.0;
    let mut coins: Vec<Coin> = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--listen" => match it.next() {
                Some(v) => listen = v.clone(),
                None => {
                    eprintln!("--listen wants an address");
                    std::process::exit(2);
                }
            },
            // Sends a proof that does not match the header it accompanies.
            //
            // A pool that can lie on purpose is how the miner's refusal gets
            // watched rather than assumed -- the same reason `diag paging`
            // faults deliberately and `fault code` jumps into a bad address.
            // A check nobody has seen refuse is a check written in a comment.
            "--bad-proof" => lie = true,
            "--ledger" => match it.next() {
                Some(v) => ledger = Some(v.clone()),
                None => {
                    eprintln!("--ledger wants a path");
                    std::process::exit(2);
                }
            },
            // Percent of **one core**, not of the machine. Zero is no limit.
            "--cpu-percent" => match it.next().and_then(|v| v.parse::<f64>().ok()) {
                Some(n) if n.is_finite() && (0.0..=10_000.0).contains(&n) => cpu_percent = n,
                _ => {
                    eprintln!("--cpu-percent wants 0..10000 (percent of one core, fractional ok; 0 is no limit)");
                    std::process::exit(2);
                }
            },
            // Both of these exist because a planned event is a different
            // problem from a stranger poking at the box, and the numbers that
            // were right for the second are wrong for the first.
            "--max-connections" => match it.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(v) => max_conns = Some(v),
                None => {
                    eprintln!("--max-connections wants a count (default 256)");
                    std::process::exit(2);
                }
            },
            "--share-seconds" => match it.next().and_then(|v| v.parse::<u64>().ok()) {
                Some(v) => share_secs = Some(v),
                None => {
                    eprintln!("--share-seconds wants 2..600, how often a miner should find a share (default 10)");
                    std::process::exit(2);
                }
            },
            "--window" => match it.next().and_then(|v| v.parse::<u64>().ok()) {
                Some(w) if w > 0 => window = Some(w),
                _ => {
                    eprintln!("--window wants a positive amount of work, e.g. 4294967296");
                    std::process::exit(2);
                }
            },
            "--sharelog" => match it.next() {
                Some(v) => sharelog = Some(std::path::PathBuf::from(v)),
                None => {
                    eprintln!("--sharelog wants a path to append recomputable share records to");
                    std::process::exit(2);
                }
            },
            "--roster" => match it.next() {
                Some(v) => roster_path = Some(v.clone()),
                None => {
                    eprintln!("--roster wants a path to the worker mapping, refreshed out of band");
                    std::process::exit(2);
                }
            },
            "--roster-url" => match it.next() {
                Some(v) => roster_url = v.clone(),
                None => {
                    eprintln!("--roster-url wants the page a refused miner should be sent to");
                    std::process::exit(2);
                }
            },
            "--require-roster" => require_roster = true,
            "--prices" => match it.next() {
                Some(v) => prices = Some(v.clone()),
                None => {
                    eprintln!("--prices wants a path");
                    std::process::exit(2);
                }
            },
            "-h" | "--help" => {
                println!("glados-pool [--listen ADDR] [--ledger PATH] [COIN ...]");
                println!();
                println!("  COIN is label:algo:bits[:asset][@host:port,user[,pass]]");
                println!("  --prices PATH reads what tools/prices.py wrote and says");
                println!("            whether each coin can be sold at all");
                println!("  --cpu-percent N caps *total* validation at N% of one core.");
                println!("  --max-connections N is the ceiling on concurrent miners (default 256).");
                println!("     Each costs one thread, one descriptor and about 28 KiB. Refused");
                println!("     above the process descriptor limit rather than failing later.");
                println!("  --share-seconds N is how often a miner should find a share (default 10).");
                println!("     Offered validation is connections x coins / N, so this is the");
                println!("     largest lever on what an expensive algorithm costs at scale.");
                println!("            Per-connection limits never bounded the sum: 256");
                println!("            connections at 20 submits a second of yespower is");
                println!("            ninety-seven cores. Default 50. 0 is no limit.");
                println!("  --window N is the payout window, in work. A share credits");
                println!("            2^bits, so the default 2^32 is one difficulty-1");
                println!("            share's worth. PPLNS pays a worker its slice of");
                println!("            the last N, which is what a block found now was");
                println!("            actually produced by.");
                println!("  --bad-proof deliberately corrupts every proof, to watch a miner refuse");
                println!("  without @, the pool builds its own headers and there is no chain");
                println!("            --selftest");
                println!();
                println!("--ledger writes the share log as canonical JSON, for publishing.");
                println!("--roster PATH checks worker names against a mapping file at greeting.");
                println!("--roster-url URL is where a refused miner is told to register.");
                println!("--require-roster refuses an unregistered name instead of warning.");
                return;
            }
            other => match parse_coin(other) {
                Ok(c) => coins.push(c),
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(2);
                }
            },
        }
    }
    if coins.is_empty() {
        // A default so the thing runs, and a loud one so nobody mistakes it for
        // a configured pool.
        eprintln!("[pool] no coins given, serving one sha256d coin at 20 bits");
        coins.push(parse_coin("local:sha256d:20").unwrap());
    }

    // **Said before the listener opens, because the point is to be read before
    // the machine is pointed at something.** `design/mining.md` spent a
    // document ranking yespower first and one run of `tools/prices.py`
    // answered it: every coin on that list was last traded between four months
    // and seven years ago. A line at startup is that finding arriving in time.
    //
    // It never refuses to start. A pool serving a coin nobody trades is a
    // decision an operator is allowed to make -- a testnet, a chain they
    // believe in, a coin whose market has not opened yet -- and a daemon that
    // would not run without a fresh price file would be one more thing to go
    // wrong at three in the morning on somebody else's server.
    let market = match &prices {
        None => None,
        Some(path) => match std::fs::read_to_string(path) {
            Ok(text) => match glados_pool::market::parse(&text) {
                Ok(m) => Some(m),
                Err(e) => {
                    eprintln!("[pool] {path} could not be read ({e}); no coin will be priced");
                    None
                }
            },
            Err(e) => {
                eprintln!("[pool] {path}: {e}; no coin will be priced");
                None
            }
        },
    };
    if let Some(m) = &market {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let age = m.age_hours(now);
        // The age of the *file*, which is a different thing from the age of a
        // price in it. A file fetched a week ago holds prices that were fresh
        // when it was written and says so on every row, so both have to be
        // shown or a reader cannot tell which is stale.
        println!("[pool] prices from {} ({:.1} h old)", prices.as_deref().unwrap_or(""), age);
    }
    for c in &coins {
        println!("[pool] {}", glados_pool::market::verdict(market.as_ref(), &c.label, &c.asset));
    }

    let mut built = Pool::new(coins);
    if let Some(p) = &sharelog {
        println!("[pool] recording every accepted share to {}", p.display());
        println!("       'glados-pool --verify {}' recomputes them all", p.display());
        built.sharelog = Some(p.clone());
    }
    if let Some(w) = window {
        built.set_window(w);
    }
    let pool = Arc::new(Mutex::new(built));

    // Read the log back before anything can add to it. A pool that began at
    // zero after every restart would lose a miner's whole record to a reboot
    // or a crash-restart, and on a machine that is not ours both of those are
    // ordinary rather than exceptional.
    if let Some(path) = &ledger {
        match std::fs::read_to_string(path) {
            // **The lock is taken once and released before anything else
            // touches it.** Written first as `match pool.lock()...load_ledger()`
            // with a second `pool.lock()` inside an arm, which compiles and
            // then deadlocks on every restart that has a ledger to read: a
            // scrutinee's temporary lives to the end of the `match`, and a
            // `std::sync::Mutex` is not reentrant.
            Ok(text) => {
                let outcome = {
                    let mut p = pool.lock().unwrap();
                    let r = p.load_ledger(&text);
                    // **PPLNS has no round boundary, so there is no safe
                    // moment to change the window.** The restored shares were
                    // accumulated under whatever the file says, the command
                    // line now says something else, and nothing rescales the
                    // stored values -- so every outstanding miner's pending
                    // reward has just moved. Rosenfeld's remedy is to scale
                    // each share by the ratio; this does not do that, and the
                    // one thing worse than not doing it is not saying so.
                    let moved = p
                        .loaded_window_work()
                        .filter(|was| *was != p.window_work())
                        .map(|was| (was, p.window_work()));
                    (r, moved)
                };
                match outcome {
                    (Ok(n), moved) => {
                        println!("[pool] resumed {n} row(s) from {path}");
                        if let Some((was, now)) = moved {
                            eprintln!("[pool] the window changed across this restart: {was} -> {now} work.");
                            eprintln!("[pool] the restored shares were credited under the old one and are not rescaled, so every pending payout has moved. Intended, or a config that got copied wrong?");
                        }
                    }
                // Loud, and it carries on with an empty tally rather than
                // refusing to start. A pool that will not run because its
                // history is unreadable helps nobody; one that starts quietly
                // and silently forgets is what has to be avoided.
                    (Err(e), _) => {
                        eprintln!("[pool] {path} could not be read ({e}); starting from zero")
                    }
                }
            }
            // Absent is the ordinary first run and is not an error.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                println!("[pool] no ledger at {path} yet; starting from zero")
            }
            Err(e) => eprintln!("[pool] {path}: {e}; starting from zero"),
        }
    }
    // One client thread per coin that has an upstream, started before the
    // listener so a miner connecting immediately is more likely to find work
    // already in hand rather than a coin that answers no job.
    if lie {
        eprintln!("[pool] --bad-proof: every proof will be deliberately wrong");
        glados_pool::pool::lie_about_proofs();
    }
    // Before the listener, so the bound exists from the first connection
    // rather than from whenever a share happens to arrive.
    if let Some(n) = max_conns {
        let got = server::set_max_connections(n);
        println!("[pool] connection ceiling {got} (one thread, one descriptor and ~28 KiB each)");
    }
    if let Some(v) = share_secs {
        let got = glados_pool::vardiff::set_target_secs(v);
        // Printed with what it implies, because the number that matters is not
        // the cadence but the share rate it produces against the ceiling.
        println!("[pool] retargeting for one share per {got}s per coin");
    }
    server::set_cpu_percent(cpu_percent);
    if cpu_percent <= 0.0 {
        eprintln!("[pool] no validation budget: this pool will use whatever the machine has");
    } else {
        println!("[pool] validation budget {cpu_percent}% of one core");
    }

    glados_pool::upstream::start_all(Arc::clone(&pool));
    let reporter = Arc::clone(&pool);
    // The share log is the whole product of a non-custodial pool: Layer 1
    // never holds a miner's coins, so there is no wallet to audit and this
    // record plus whatever checks against it is all a miner has. It goes to
    // stdout always, and to a file when asked.
    //
    // **A file rather than an endpoint, deliberately.** The intended home is a
    // static site whose history is a commit chain, and a commit chain cannot
    // be quietly rewritten where a live endpoint can. For a pool whose only
    // asset is being checkable, tamper-evidence beats freshness.
    // ---- the roster, if one was named ------------------------------------
    //
    // Loaded once here so a bad path is a complaint at startup rather than a
    // surprise at the first greeting, then re-read on a timer. The interval is
    // a minute because the thing it tracks is somebody registering a name and
    // then starting their rig, and a miner will wait a minute.
    if let Some(path) = roster_path.clone() {
        *glados_pool::roster::WHERE.lock().unwrap() = roster_url.clone();
        glados_pool::roster::REQUIRE
            .store(require_roster, std::sync::atomic::Ordering::Relaxed);
        let mut r = glados_pool::roster::Roster::new(
            Some(path.clone()),
            std::time::Duration::from_secs(60),
        );
        if let Some(note) = r.refresh() {
            println!("{note}");
        }
        if !r.ready() {
            // Said plainly at startup, because the consequence is invisible
            // afterwards: with no roster every name is `Unknown`, everything
            // is admitted, and `--require-roster` enforces nothing at all.
            println!("[pool] no roster loaded, so every worker name is admitted");
        }
        *glados_pool::roster::ROSTER.lock().unwrap() = Some(r);

        std::thread::spawn(move || loop {
            std::thread::sleep(std::time::Duration::from_secs(30));
            if let Some(r) = glados_pool::roster::ROSTER.lock().unwrap().as_mut() {
                if let Some(note) = r.refresh() {
                    println!("{note}");
                }
            }
        });
    } else if require_roster {
        // A flag that cannot do anything is worse than a missing one: the
        // operator believes names are being checked.
        eprintln!("--require-roster needs --roster; nothing would be checked");
        std::process::exit(2);
    }

    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_secs(60));
        let p = reporter.lock().unwrap();
        for (w, c, t) in p.ledger() {
            // Work first, because it is the number a payout comes from and
            // the share count is only how you tell a miner is alive.
            println!(
                "[ledger] {w}  {c}  {} work, {} accepted, {} stale, {} bad, {} dup",
                t.work, t.accepted, t.stale, t.bad, t.duplicate
            );
        }
        // What the budget has actually measured on *this* machine, and what
        // it therefore allows. The whole reason it exists is that the figure
        // in `server.rs`'s comment was measured somewhere else and was out by
        // 2.5x, so a pool that did not print its own would repeat that.
        let denied = server::budget_denied();
        for (algo, us, per_s) in server::budget_report() {
            if per_s.is_finite() {
                println!(
                    "[budget] {algo}  {us:.1} us a share, so {per_s:.0} a second within the cap"
                );
            } else {
                println!("[budget] {algo}  {us:.1} us a share, and no cap on how many");
            }
        }
        if denied > 0 {
            println!("[budget] {denied} share(s) deferred so far; raise --cpu-percent to admit more");
        }
        // Both maps are keyed by a worker name and a worker name is whatever a
        // stranger types, so both are capped -- and a cap that bites without
        // saying so is how an operator ends up looking for a miner's record in
        // the wrong place. Printed only when it is actually biting.
        if p.untallied() > 0 {
            println!(
                "[pool] {} verdict(s) went unrecorded: {} of {} tally slots are held by workers with credited work",
                p.untallied(),
                p.tally_len(),
                glados_pool::pool::MAX_TALLIES
            );
        }

        // **What each worker is actually owed, which the tally does not say.**
        // The tally is all-time; a payout comes from the window. Printing only
        // the first is how an operator ends up paying against the wrong number
        // -- the coins in a block found now were produced by recent hashing,
        // not by somebody who left last year.
        for c in p.coin_labels() {
            let Some(rows) = p.payouts(&c) else { continue };
            let Some(w) = p.window(&c) else { continue };
            println!(
                "[payout] {c}  window {} of {} work over {} share(s)",
                w.total(),
                p.window_work(),
                w.len()
            );
            // **The window said in the only unit that means anything**, which
            // needs the network target and therefore an upstream. Rosenfeld:
            // variance goes as `pB^2/N` and mean time to payment as `pN/2`, so
            // their product is fixed however `N` is chosen -- the window is a
            // dial between paying smoothly and paying soon, not an optimum, and
            // an operator cannot pick a point on a dial with no markings.
            //
            // Printed as a ratio to one block's expected work because that is
            // the number with the surprise in it: a window far below a block is
            // a window that pays out the last few shares, whatever it was meant
            // to be, and the arithmetic says so where the raw figure looks
            // perfectly large.
            if let Some(blocks) = p.window_in_blocks(&c) {
                if blocks >= 0.01 {
                    println!(
                        "[payout]   the window is {blocks:.2} block(s) of work; mean wait to be paid for a share is about {:.2} block(s)",
                        blocks / 2.0
                    );
                } else {
                    println!(
                        "[payout]   the window is 1/{:.0} of one block's work -- far too small to be a payout window on this chain; it pays the most recent shares and nothing else",
                        1.0 / blocks.max(f64::MIN_POSITIVE)
                    );
                }
            }
            for (name, work, share) in rows {
                println!("[payout]   {name}  {work} work  {:.4}%", share * 100.0);
            }
        }
        if let Some(path) = &ledger {
            let at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let doc = p.ledger_json(1, at);
            // Written through a temporary and renamed, because a publisher may
            // be reading this file at any moment and half a document is worse
            // than a stale one -- it parses as far as the truncation and then
            // does not.
            let tmp = format!("{path}.new");
            let wrote = std::fs::write(&tmp, doc.as_bytes())
                .and_then(|_| std::fs::rename(&tmp, path));
            if let Err(e) = wrote {
                eprintln!("[pool] could not write {path}: {e}");
            }
        }
    });

    if let Err(e) = server::serve(&listen, pool) {
        eprintln!("[pool] {listen}: {e}");
        std::process::exit(1);
    }
}

/// What one share costs to check, which is what sizing a host comes down to.
///
/// **The pool's work is per share and the miner's is per hash**, and the two
/// differ by the difficulty: a miner grinds several million nonces to find one
/// share and the pool hashes exactly once to agree. So the load here does not
/// scale with anybody's hashrate, only with how often shares arrive -- which is
/// a number the operator sets, by choosing the share target.
///
/// Best of nine, like `video bench` and `core bench`, and for the reason those
/// record: a single sample on a shared machine measures the host's scheduler.
fn bench() {
    use glados_pool::mine::algo::{Algo, Hasher};
    use std::time::Instant;

    let algos = [
        ("sha256d", Algo::Sha256d),
        ("blake2s", Algo::Blake2s),
        (
            "yespower 2 MiB",
            Algo::Yespower { v10: true, n: 2048, r: 8, pers: None },
        ),
        (
            "yespower 8 MiB",
            Algo::Yespower { v10: true, n: 2048, r: 32, pers: None },
        ),
    ];
    let header: [u8; 80] = core::array::from_fn(|i| (i as u32 * 3) as u8);

    println!("what one share costs to validate, best of nine");
    println!();
    println!("  algorithm        per share   working set   shares/s on one core");
    for (name, algo) in algos.iter() {
        let Some(h) = Hasher::new(algo, &header) else {
            continue;
        };
        let foot = h.footprint();
        drop(h);

        let mut best = u128::MAX;
        for _ in 0..9 {
            // A fresh `Hasher` every time, because that is what validating a
            // share actually does: the pool holds no per-miner scratch, so the
            // allocation is part of the cost rather than something amortised
            // away by a benchmark that reuses one.
            let t = Instant::now();
            let reps = if foot > 0 { 20 } else { 2000 };
            for n in 0..reps {
                let mut hh = Hasher::new(algo, &header).unwrap();
                core::hint::black_box(hh.hash(&header, n));
            }
            let per = t.elapsed().as_nanos() / reps as u128;
            best = best.min(per);
        }
        let per_s = if best > 0 { 1_000_000_000 / best } else { 0 };
        println!(
            "  {name:<15}  {:>7} us   {:>7} KiB   {:>10}",
            best as f64 / 1000.0,
            foot / 1024,
            per_s
        );
    }
    println!();
    println!("A miner grinds millions of nonces per share; the pool hashes once.");
    println!("So this scales with the share *rate*, which the operator sets by");
    println!("choosing the target -- not with anybody's hashrate.");
}

/// A stranger cannot spend this machine's CPU without limit.
///
/// Checked rather than asserted, because a limit nobody has watched fire is a
/// limit written in a comment -- the objection `diag paging` makes about page
/// rights, which faults on purpose for exactly this reason.
///
/// The garbage is a *valid* submit for a *real* job with a nonce that does not
/// meet the target, which is the expensive case: the pool has to compute the
/// hash to find out, so every one of these costs it a full validation. A
/// malformed message would be rejected by the parser for free and would prove
/// nothing.
fn abuse_check(addr: std::net::SocketAddr) -> bool {
    use glados_pool::json::Json;
    use glados_pool::mine::proto;
    use glados_pool::mine::stratum::take_line;
    use std::io::{Read, Write};

    let Ok(mut sock) = std::net::TcpStream::connect(addr) else {
        println!("FAIL  could not open a second connection");
        return false;
    };
    sock.set_read_timeout(Some(std::time::Duration::from_secs(15)))
        .unwrap();
    if sock
        .write_all(proto::encode_hello(1, "abuse.rig", "abuse-check").as_bytes())
        .is_err()
    {
        println!("FAIL  could not greet on the second connection");
        return false;
    }

    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 4096];
    let mut job: Option<proto::Job> = None;
    for _ in 0..40 {
        let Ok(n) = sock.read(&mut chunk) else { break };
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        while let Ok(Some(line)) = take_line(&mut buf) {
            let Some(v) = Json::parse(line.trim()) else { continue };
            if v.get("method").and_then(|m| m.as_str()) == Some("glados.job") {
                if let Some(j) = v.get("params").and_then(proto::parse_job) {
                    job = Some(j);
                }
            }
        }
        if job.is_some() {
            break;
        }
    }
    let Some(j) = job else {
        println!("FAIL  the second connection was never given a job");
        return false;
    };

    // Nonce zero against a twelve-bit target is a solution about one time in
    // four thousand, so a few dozen of these are bad shares with near
    // certainty -- and the one-in-4096 case is an *accepted* share, which
    // simply does not count toward the limit and costs an extra round.
    for i in 0..200u32 {
        let sh = proto::Share { job: j.job.clone(), nonce: i, echo: vec![] };
        if sock
            .write_all(proto::encode_submit(2, &sh).as_bytes())
            .is_err()
        {
            println!("ok    a stranger sending garbage was cut off after {i} shares");
            return true;
        }
        // Under the per-second cap, so the drop that ends this is the bad-share
        // limit rather than the rate limit. Testing both at once would leave it
        // unclear which one fired.
        std::thread::sleep(std::time::Duration::from_millis(80));

        // A closed connection shows as a read of zero as readily as a failed
        // write, and which one appears depends on timing rather than on
        // behaviour.
        let mut probe = [0u8; 1024];
        match sock.read(&mut probe) {
            Ok(0) => {
                println!("ok    a stranger sending garbage was cut off after {i} shares");
                return true;
            }
            Ok(_) => {}
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut => {}
            Err(_) => {
                println!("ok    a stranger sending garbage was cut off after {i} shares");
                return true;
            }
        }
    }
    println!("FAIL  200 bad shares and the connection is still open");
    false
}

/// One connection is one worker, and this is the only place that is checked
/// without a Python harness and a server somebody has to remember to start.
///
/// It matters more than it looks. A second `hello` under a different name used
/// to overwrite the first, so a single socket could register worker names at
/// the per-connection message rate -- nineteen a second, measured -- and each
/// one bought a permanent record in the tally while costing **no validation at
/// all**, because a submit naming a job the pool does not hold is answered
/// before anything is hashed. The pool-wide validation budget, the bound that
/// exists precisely to stop a stranger spending this machine, never saw any of
/// it.
///
/// So the claim has two halves and the second is the one a naive fix breaks:
/// the rename is refused, **and the connection survives it**. Dropping the
/// connection would be a denial of service handed to anybody who can send one
/// malformed handshake.
fn rename_check(addr: std::net::SocketAddr) -> bool {
    use glados_pool::json::Json;
    use glados_pool::mine::proto;
    use glados_pool::mine::stratum::take_line;
    use std::io::{Read, Write};

    let Ok(mut sock) = std::net::TcpStream::connect(addr) else {
        println!("FAIL  could not open a third connection");
        return false;
    };
    sock.set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .unwrap();
    if sock
        .write_all(proto::encode_hello(1, "rename.first", "rename-check").as_bytes())
        .is_err()
        || sock
            .write_all(proto::encode_hello(2, "rename.second", "rename-check").as_bytes())
            .is_err()
    {
        println!("FAIL  could not greet twice");
        return false;
    }

    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 4096];
    let mut refused = false;
    let mut welcomes = 0usize;
    for _ in 0..40 {
        let Ok(n) = sock.read(&mut chunk) else { break };
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        while let Ok(Some(line)) = take_line(&mut buf) {
            let Some(v) = Json::parse(line.trim()) else { continue };
            if let Some(r) = v.get("result") {
                if proto::parse_welcome(r).is_some() {
                    welcomes += 1;
                }
            }
            if let Some(e) = v.get("error").and_then(|x| x.as_str()) {
                if e.contains("one worker per connection") {
                    refused = true;
                }
            }
        }
        if refused {
            break;
        }
    }

    if welcomes > 1 {
        println!("FAIL  a second worker name was welcomed on one connection");
        return false;
    }
    if !refused {
        println!("FAIL  a rename was neither refused nor answered");
        return false;
    }

    // Still usable afterwards, which is the half a refusal that dropped the
    // connection would fail. Asking for work is the cheapest proof.
    if sock
        .write_all(proto::encode_work(3, 0).as_bytes())
        .is_err()
    {
        println!("FAIL  the connection was dropped over a refused rename");
        return false;
    }
    for _ in 0..40 {
        let Ok(n) = sock.read(&mut chunk) else { break };
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        while let Ok(Some(line)) = take_line(&mut buf) {
            let Some(v) = Json::parse(line.trim()) else { continue };
            if v.get("method").and_then(|m| m.as_str()) == Some("glados.job") {
                println!("ok    a rename was refused and the connection kept working");
                return true;
            }
        }
    }
    println!("FAIL  no work after a refused rename");
    false
}

/// The whole path, in one process, with no kernel and no network beyond
/// loopback: listen, greet, take a job, mine it, submit, be believed.
///
/// It exists because every other check here is of a piece in isolation. The
/// interesting failures in a pool are between the pieces -- a job encoded one
/// way and decoded another, a nonce byte order that differs across the wire, a
/// share validated against the wrong header -- and none of those shows up in a
/// codec round-trip, because a round-trip uses one encoder against its own
/// decoder rather than against a socket and a second process's view of a job.
/// Recompute every share in a log. Answers a process exit code.
///
/// The whole of what a verification server does, and it is deliberately this
/// small: read a line, build the hasher the line names, hash the header with the
/// nonce, compare against the target. There is no pool here and no network, so
/// the same binary runs on a laptop, a runner, or a miner's machine checking the
/// operator's arithmetic.
fn verify_log(path: &str) -> i32 {
    use glados_pool::record::Record;
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            return 2;
        }
    };
    let mut ok = 0u64;
    let mut bad = 0u64;
    let mut broken = 0u64;
    for (n, line) in text.lines().enumerate() {
        match Record::parse(line) {
            Ok(None) => continue,
            Err(e) => {
                println!("line {}: not a record -- {}", n + 1, e);
                broken += 1;
            }
            Ok(Some(r)) => match r.check() {
                None => {
                    // The record names parameters no hasher will build. That is
                    // the operator having written it wrongly, not a miner having
                    // cheated, and the two must not be counted together.
                    println!("line {}: {} is not a hashable set of parameters",
                             n + 1, glados_pool::record::algo_spec(&r.algo));
                    broken += 1;
                }
                Some((met, digest)) => {
                    if met {
                        ok += 1;
                    } else {
                        // Printed in full, because this is the only interesting
                        // line in the file: a share the operator credited that
                        // does not meet the target they published.
                        print!("line {}: {} does NOT meet its target, digest ",
                               n + 1, r.worker);
                        for b in digest.iter().rev() {
                            print!("{b:02x}");
                        }
                        println!();
                        bad += 1;
                    }
                }
            },
        }
    }
    println!("{ok} share(s) recomputed and met their target, {bad} did not, {broken} unreadable");
    if bad == 0 && broken == 0 {
        println!("every share in {path} is arithmetic anybody can repeat");
        0
    } else {
        1
    }
}

fn selftest() -> bool {
    use glados_pool::json::Json;
    use glados_pool::mine::algo::Hasher;
    use glados_pool::mine::hash::below_target;
    use glados_pool::mine::proto;
    use glados_pool::mine::stratum::take_line;
    use std::io::{Read, Write};

    // Port 0, so the operating system picks one. A fixed port is how two runs
    // of a test collide, which `stratumstub.py` records at length.
    let listener = match std::net::TcpListener::bind("127.0.0.1:0") {
        Ok(l) => l,
        Err(e) => {
            println!("FAIL  could not bind loopback: {e}");
            return false;
        }
    };
    let addr = listener.local_addr().unwrap();
    drop(listener);

    // Two coins on two different algorithms, because one coin proves the
    // plumbing and this protocol exists for the other thing. A pool that
    // served both from one connection but validated both under one algorithm
    // would pass a single-coin test perfectly.
    let coins = vec![
        parse_coin("selftest-a:sha256d:12").unwrap(),
        parse_coin("selftest-b:blake2s:12").unwrap(),
    ];
    let pool = Arc::new(Mutex::new(Pool::new(coins)));
    let served = Arc::clone(&pool);
    let bind = addr.to_string();
    std::thread::spawn(move || {
        let _ = server::serve(&bind, served);
    });

    let mut sock = None;
    for _ in 0..50 {
        if let Ok(s) = std::net::TcpStream::connect(addr) {
            sock = Some(s);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(40));
    }
    let Some(mut sock) = sock else {
        println!("FAIL  the listener never came up");
        return false;
    };
    sock.set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .unwrap();

    if sock
        .write_all(proto::encode_hello(1, "selftest.rig", "glados-pool/selftest").as_bytes())
        .is_err()
    {
        println!("FAIL  could not greet");
        return false;
    }

    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 4096];
    let mut welcomed = false;
    let mut pending: Vec<proto::Job> = Vec::new();
    let mut sent: Vec<String> = Vec::new();
    let mut accepted = 0usize;
    let mut algos: Vec<String> = Vec::new();
    let want = 2usize;

    for _ in 0..200 {
        let n = match sock.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => {
                println!("FAIL  read: {e}");
                return false;
            }
        };
        buf.extend_from_slice(&chunk[..n]);
        while let Ok(Some(line)) = take_line(&mut buf) {
            let Some(v) = Json::parse(line.trim()) else {
                continue;
            };
            if let Some(r) = v.get("result") {
                if let Some(w) = proto::parse_welcome(r) {
                    if w.v != proto::VERSION {
                        println!("FAIL  welcomed at v{} against v{}", w.v, proto::VERSION);
                        return false;
                    }
                    welcomed = true;
                    continue;
                }
                // The submit's answer. This is the claim: the pool
                // independently recomputed the hash and agreed.
                if !sent.is_empty() {
                    let ok = r.get("ok").and_then(|x| x.as_bool()).unwrap_or(false);
                    let verdict = r.get("verdict").and_then(|x| x.as_str()).unwrap_or("?");
                    if !ok {
                        println!("FAIL  the pool refused a share it could reproduce: {verdict}");
                        return false;
                    }
                    accepted += 1;
                    if accepted == want {
                        println!(
                            "ok    greeted, mined {want} coins ({}), and every share was accepted",
                            algos.join(" + ")
                        );
                        // The second half. A pool that accepts good shares and
                        // never refuses a stranger is half checked, and on a
                        // machine somebody lent us it is the wrong half.
                        return rename_check(addr) && abuse_check(addr);
                    }
                }
            }
            if v.get("method").and_then(|m| m.as_str()) == Some("glados.job") {
                if let Some(j) = v.get("params").and_then(proto::parse_job) {
                    // One share per slot. The server re-issues on every wake,
                    // so without this the first coin would be mined forever
                    // and the second never reached.
                    if !sent.iter().any(|s| s == &j.job)
                        && !pending.iter().any(|p| p.slot == j.slot)
                        && !algos.iter().any(|a| a == j.algo.name())
                    {
                        pending.push(j);
                    }
                }
            }
        }

        if !welcomed {
            continue;
        }
        while let Some(j) = pending.pop() {
            let Some(mut h) = Hasher::new(&j.algo, &j.header) else {
                println!("FAIL  the pool issued parameters its own algorithm refuses");
                return false;
            };
            let mut found = None;
            for nonce in 0..5_000_000u32 {
                if below_target(&h.hash(&j.header, nonce), &j.target) {
                    found = Some(nonce);
                    break;
                }
            }
            let Some(nonce) = found else {
                println!("FAIL  no {} share inside five million nonces", j.algo.name());
                return false;
            };
            let sh = proto::Share {
                job: j.job.clone(),
                nonce,
                echo: j.echo.clone(),
            };
            if sock
                .write_all(proto::encode_submit(2, &sh).as_bytes())
                .is_err()
            {
                println!("FAIL  could not submit");
                return false;
            }
            sent.push(j.job.clone());
            algos.push(String::from(j.algo.name()));
        }
    }

    println!(
        "FAIL  the exchange never completed (welcomed={welcomed} sent={} accepted={accepted})",
        sent.len()
    );
    false
}
