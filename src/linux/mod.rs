//! Running a program this kernel did not compile.
//!
//! Everything else in this tree was written here and signed by the key
//! compiled in. A Linux binary is neither, and that single fact sets the shape
//! of the whole module: nothing read out of the file is trusted, every offset
//! is checked against the file's own length, and a refusal names the field it
//! refused.
//!
//! ### What stage 0 is, and what it deliberately is not
//!
//! It is a **measuring instrument.** The expensive unknown in porting the
//! Linux ABI is not the loader -- that is a weekend -- it is that Linux has no
//! specification you can test against, so a subtly wrong `mmap` flag surfaces
//! as a crash in a different subsystem an hour later. Before committing to a
//! privilege model, an isolation mechanism or a syscall subset, the useful
//! thing to own is a *trace*: which calls does a real binary actually make, in
//! what order, with what arguments.
//!
//! So this loads a static binary at ring 0 with no isolation whatsoever, traps
//! its syscalls, and writes down what it asked for. A guest that faults takes
//! the kernel with it, and that is the intended behaviour rather than a gap:
//! at this stage a fault is the measurement. `gvisor` needed 237 of Linux's
//! ~350 calls to run containers, and the point of stage 0 is to find out which
//! ones this machine's first target actually reaches.
//!
//! The isolation question -- ring 3 in one address space, `PKS` domains,
//! `SFI`, or `VT-x` with the guest still at CPL 0 -- is real and is answered
//! later, from data this produces rather than from first principles.

pub mod elf;
pub mod fs;
pub mod load;
pub mod dev;
pub mod input;
pub mod thread;
pub mod fork;
pub mod signal;
pub mod unix;
pub mod poll;
pub mod epoll;
pub mod proc;
pub mod program;
pub mod session;
pub mod i386;
pub mod disk;
pub mod ns;
pub mod link;
pub mod timer;
pub mod pty;
pub mod syscall;
pub mod prof;
pub mod snd;
pub mod drm;
pub mod netctl;
pub mod services;

/// What `diag linux` asks of everything here.
pub fn checks() -> alloc::vec::Vec<(&'static str, bool)> {
    let mut out = elf::checks();
    out.extend(fs::checks());
    out.extend(dev::checks());
    out.extend(input::checks());
    out.extend(thread::checks());
    out.extend(unix::checks());
    out.extend(poll::checks());
    out.extend(epoll::checks());
    out.extend(signal::checks());
    out.extend(proc::checks());
    out.extend(snd::checks());
    out.extend(netctl::checks());
    out.extend(drm::checks());
    out.extend(services::checks());
    out.extend(program::checks());
    out.extend(syscall::checks());
    out.extend(load::checks());
    out.extend(session::checks());
    out.extend(i386::checks());
    out.extend(disk::checks());
    out.extend(ns::checks());
    out.extend(link::checks());
    out.extend(timer::checks());
    out.extend(pty::checks());
    out
}

/// Linux as a `job` runtime.
pub static RUNTIME: crate::job::Runtime = crate::job::Runtime {
    name: "linux",
    kill: |task| syscall::doom(task, syscall::SIGNALED | 9),
};

/// Why a new Linux session cannot start now, or `None` when it can.
///
/// Several may run at once -- each is its own pid namespace with its own
/// deadline, trace and processes (see `session`) -- up to `session::MAX`,
/// which is about the fork and thread pools they share rather than memory.
pub fn busy() -> Option<&'static str> {
    if session::live_count() >= session::MAX {
        Some("every Linux session is in use -- 'jobs' lists them, 'kill <n>' ends one")
    } else {
        None
    }
}

/// Start `path` with `args` as a background job. Answers the job number.
///
/// The bytes are read here so a wrong path is refused at the prompt; the load
/// happens on the job's own task, because a loaded `Guest` holds raw pointers
/// to its pages and is not a thing to hand between tasks.
pub fn start_job(path: &str, args: &[&str]) -> Result<u32, &'static str> {
    start_job_as(path, args, false)
}

/// A supervised service: a job whose console output goes to the service log.
pub fn start_service_job(path: &str, args: &[&str]) -> Result<u32, &'static str> {
    start_job_as(path, args, true)
}

/// `start_job`, with some variables of the environment replaced for this job
/// alone.
///
/// Applied on the job's own task, immediately before its loader reads them,
/// through the same per-task override `execve` uses. Changing the global
/// environment around a `start_job` instead was a race: the loader runs when
/// the job's task does, so the value put back afterwards was sometimes the
/// one it read -- which is how a headless LibreOffice came up on GTK anyway.
pub fn start_job_env(path: &str, args: &[&str], overrides: &[&str]) -> Result<u32, &'static str> {
    let ov: alloc::vec::Vec<alloc::string::String> = overrides.iter().map(|s| alloc::string::String::from(*s)).collect();
    start_job_inner(path, args, false, ov)
}

/// `start_job_env`, its console output kept out of the terminal the way a
/// service's is.
pub fn start_job_env_quiet(path: &str, args: &[&str], overrides: &[&str]) -> Result<u32, &'static str> {
    let ov: alloc::vec::Vec<alloc::string::String> = overrides.iter().map(|s| alloc::string::String::from(*s)).collect();
    start_job_inner(path, args, true, ov)
}

fn start_job_as(path: &str, args: &[&str], quiet: bool) -> Result<u32, &'static str> {
    start_job_inner(path, args, quiet, alloc::vec::Vec::new())
}

fn start_job_inner(
    path: &str,
    args: &[&str],
    quiet: bool,
    overrides: alloc::vec::Vec<alloc::string::String>,
) -> Result<u32, &'static str> {
    if let Some(why) = busy() {
        return Err(why);
    }
    let bytes = disk::program_bytes(path).ok_or("no such program")?;
    let mut argv: alloc::vec::Vec<alloc::string::String> = alloc::vec![alloc::string::String::from(path)];
    argv.extend(args.iter().map(|a| alloc::string::String::from(*a)));
    let title = argv.join(" ");
    // A quiet job's console output goes to the service log, and its end is
    // its caller's to report: the notice was a line per LibreOffice launch in
    // the terminal of somebody asking a spreadsheet a question.
    (if quiet { crate::job::start_quiet } else { crate::job::start })(
        &RUNTIME,
        title,
        alloc::boxed::Box::new(move || {
            let argv: alloc::vec::Vec<&str> = argv.iter().map(|s| s.as_str()).collect();
            if !overrides.is_empty() {
                let name = |e: &str| alloc::string::String::from(e.split_once('=').map_or(e, |(n, _)| n));
                let mut env = syscall::environ();
                env.retain(|e| !overrides.iter().any(|o| name(o) == name(e)));
                env.extend(overrides.iter().cloned());
                syscall::set_exec_env(env);
            }
            let mut g = match load::load(&bytes, &argv) {
                Ok(g) => g,
                Err(why) => return alloc::format!("will not run it: {}", why),
            };
            if quiet {
                session::ask_quiet();
            }
            // Safety: the same call `linux run` makes, on a task of its own
            // and with no deadline -- see `syscall::run`.
            let r = unsafe { load::run(&mut g, 0) };
            describe(r, syscall::trace().len())
        }),
    )
}

/// How a session ended, in one line.
pub fn describe(r: u64, calls: usize) -> alloc::string::String {
    if r & syscall::OVERRAN != 0 {
        alloc::format!("killed for running too long after {} syscall(s)", calls)
    } else if r & syscall::FAULTED != 0 {
        alloc::format!("killed by fault {:#04x} after {} syscall(s), machine intact", r & 0xFFFF_FFFF, calls)
    } else if r & syscall::SIGNALED != 0 {
        alloc::format!("ended by signal {} after {} syscall(s)", r & 0x7F, calls)
    } else if r & syscall::EXITED != 0 {
        alloc::format!("exited {} after {} syscall(s)", r & 0xFFFF_FFFF, calls)
    } else {
        alloc::format!("returned without exiting -- {} syscall(s)", calls)
    }
}
