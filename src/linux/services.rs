//! Ring-3 services: programs the machine starts by itself and keeps running.
//!
//! **What is not the agent goes to ring 3**, and a thing that has to be
//! running for the machine to be useful -- a network manager, a sound
//! server, a game's dedicated server -- needs somebody to start it at boot
//! and start it again when it dies. That is this file and nothing more: a
//! list, a supervisor, and a verb.
//!
//! `/etc/services`, one per line, `#` for a comment:
//!
//! ```text
//! netd /linux/bin/netd
//! ```
//!
//! Each runs as a background job (`linux::start_job`), so it has the same
//! leash as anything the operator starts: ring 3, its own page tables, its
//! own session. A service that exits is restarted after a pause that doubles
//! each time it dies quickly -- 1, 2, 4 ... 60 seconds -- and resets once it
//! has stayed up for a minute, so a program that cannot start does not spend
//! the machine starting it.
//!
//! Supervised from the shell's idle loop beside DHCP and wifi, because
//! starting a job reads its program from storage and the clock task must not.

use crate::sync::Racy;
use alloc::string::String;
use alloc::vec::Vec;

pub const LIST: &str = "/etc/services";

struct Service {
    name: String,
    path: String,
    args: Vec<String>,
    job: Option<u32>,
    started: u64,
    backoff_s: u64,
    next_at: u64,
    restarts: u32,
    last: String,
}

static SERVICES: Racy<Vec<Service>> = Racy::new(Vec::new());
/// Loaded once on the first supervisor turn, which is the first moment the
/// namespace is certainly up; `stop` holds it off until `load` again.
static ON: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
static STOPPED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

fn now_s() -> u64 {
    crate::dev::lapic::ticks() / crate::TIMER_HZ as u64
}

/// Parse a list. Pure, for the claims.
pub fn parse(text: &str) -> Vec<(String, String, Vec<String>)> {
    text.lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
        .filter_map(|l| {
            let mut w = l.split_whitespace();
            let name = w.next()?;
            let path = w.next()?;
            Some((String::from(name), String::from(path), w.map(String::from).collect()))
        })
        .collect()
}

/// Read the list and start supervising it. Called once the namespace is up;
/// again to pick up an edited list.
pub fn load() {
    let text = crate::sysbox::read_blob(LIST).map(|b| String::from_utf8_lossy(&b).into_owned()).unwrap_or_default();
    let s = unsafe { SERVICES.get() };
    let mut next = Vec::new();
    for (name, path, args) in parse(&text) {
        // A service already running under the same command keeps its job.
        let old = s.iter().position(|o| o.name == name && o.path == path && o.args == args).map(|i| s.remove(i));
        next.push(old.unwrap_or(Service {
            name,
            path,
            args,
            job: None,
            started: 0,
            backoff_s: 1,
            next_at: 0,
            restarts: 0,
            last: String::new(),
        }));
    }
    for gone in s.drain(..) {
        if let Some(j) = gone.job {
            crate::job::kill(j);
        }
    }
    *s = next;
    ON.store(true, core::sync::atomic::Ordering::Release);
    STOPPED.store(false, core::sync::atomic::Ordering::Release);
}

/// One supervisor turn.
pub fn tick() {
    if STOPPED.load(core::sync::atomic::Ordering::Acquire) {
        return;
    }
    if !ON.load(core::sync::atomic::Ordering::Acquire) {
        load();
    }
    let s = unsafe { SERVICES.get() };
    if s.is_empty() {
        return;
    }
    let rows = crate::job::list();
    let now = now_s();
    for sv in s.iter_mut() {
        if let Some(j) = sv.job {
            match rows.iter().find(|r| r.id == j) {
                Some(r) if r.state != "done" => continue,
                Some(r) => sv.last = r.summary.clone(),
                None => {}
            }
            // Ended. Quickly means the pause doubles; after a good run it
            // starts over at one second.
            sv.job = None;
            if now.saturating_sub(sv.started) >= 60 {
                sv.backoff_s = 1;
            } else {
                sv.backoff_s = (sv.backoff_s * 2).min(60);
            }
            sv.next_at = now + sv.backoff_s;
            crate::kprintln!("\n  [services] {} ended ({}); again in {} s", sv.name, sv.last, sv.backoff_s);
        }
        if now < sv.next_at {
            continue;
        }
        let args: Vec<&str> = sv.args.iter().map(|a| a.as_str()).collect();
        match super::start_job(&sv.path, &args) {
            Ok(id) => {
                sv.job = Some(id);
                sv.started = now;
                if sv.next_at != 0 {
                    sv.restarts += 1;
                }
            }
            Err(why) => {
                sv.last = String::from(why);
                sv.backoff_s = (sv.backoff_s * 2).min(60);
                sv.next_at = now + sv.backoff_s;
            }
        }
    }
}

pub fn report() {
    let s = unsafe { SERVICES.get() };
    if s.is_empty() {
        crate::kprintln!("  no services -- {} lists them, one 'name path [args]' per line", LIST);
        return;
    }
    for sv in s.iter() {
        let state = match sv.job {
            Some(j) => alloc::format!("running as job {}", j),
            None => alloc::format!("waiting, next try in {} s", sv.next_at.saturating_sub(now_s())),
        };
        crate::kprintln!("  {:<12} {:<28} {}  restarts {}", sv.name, sv.path, state, sv.restarts);
        if !sv.last.is_empty() {
            crate::kprintln!("               last: {}", sv.last);
        }
    }
}

/// Stop supervising, and end every service.
pub fn stop() {
    STOPPED.store(true, core::sync::atomic::Ordering::Release);
    for sv in unsafe { SERVICES.get() }.iter_mut() {
        if let Some(j) = sv.job.take() {
            crate::job::kill(j);
        }
    }
}

pub fn checks() -> Vec<(&'static str, bool)> {
    let p = parse("# a comment\nnetd /linux/bin/netd -v  # trailing\n\nlonely\nsnd /bin/x a b\n");
    alloc::vec![
        ("a list skips comments, blanks and a line with no program", p.len() == 2),
        ("and keeps a service's arguments", p[0].0 == "netd" && p[0].2 == ["-v"] && p[1].2 == ["a", "b"]),
    ]
}
