//! Starting a miner from the boot volume, for a machine whose only job is that.
//!
//! A miner-only image carries no model, no store and nothing to type on: it is
//! handed to somebody who boots it and walks away. So the pool, the worker name
//! and the slice count have to arrive from somewhere, and on read-only media
//! there is exactly one place they can be -- a file put there when the image was
//! built.
//!
//! ### Read before `ExitBootServices`, like everything else that matters
//!
//! `main.rs` reads the model, the tokenizer and the root bundle before that
//! call because it is the last moment a filesystem exists. This is the same
//! constraint arriving for a different reason: an ISO is `find_esp`'s own
//! "read-only, and there is no writable ESP", so the firmware's reader is not
//! merely the easiest way in, it is the only one.
//!
//! ### Nothing the file says is executed
//!
//! `update::repairs` states the rule and it applies with more force here,
//! because this file is on media anybody can edit and it names a machine to
//! connect to. Two words are parsed into a typed field and the field is what
//! acts; a line naming something this parser does not know is dropped and
//! counted, never passed on to a shell. There is no path from this file to
//! `shell::execute` and there must not be one -- the whole difference between
//! a config and a script is that a config cannot say `rm`.
//!
//! ### The worker name is a payout address, so a default would be theft
//!
//! `pool/src/roster.rs` maps a worker name to the address that gets paid, and
//! at the account-free venues the name *is* the address. A miner that booted
//! with a built-in default would mine to whoever that default belongs to, for
//! as long as nobody noticed. So `worker` has no default: without it `plan`
//! answers `None` and the machine sits at a prompt doing nothing, which is the
//! failure that is visible rather than the one that is profitable.
use alloc::string::String;
use alloc::vec::Vec;

/// Where the file lives on the boot volume.
///
/// **Backslashes, because this path goes to the firmware.** UEFI's own file
/// protocol wants them and `uefi::read_file` passes the string through widened
/// and otherwise untouched, so a forward-slash spelling is not a different
/// separator -- it is a filename with no directories in it, which simply is not
/// there. `update::repairs` keeps two constants for exactly this reason, one
/// for the firmware and one for our own FAT writer, which splits on '/'.
///
/// Found by booting a miner ISO that came up, reached a prompt, printed nothing
/// and sat with its pool unset. There is no error path here to report: a file
/// that is absent and a file whose name cannot match are the same `None`.
pub const FILE: &str = "\\GLADOS\\MINER.TXT";

/// What the file is allowed to say. Every field is typed and none is a command.
#[derive(Debug, PartialEq)]
pub struct Plan {
    pub host: String,
    pub port: u16,
    pub user: String,
    pub pass: String,
    pub glados_proto: bool,
    pub slices: Option<u32>,
    /// Lines whose first word this parser does not know. Counted rather than
    /// ignored, because a typo in a file nobody can edit after the image is cut
    /// should be visible at boot instead of presenting as a miner that will not
    /// connect for no stated reason.
    pub unknown: usize,
}

/// Parse the file. `None` when it does not name both a pool and a worker,
/// which are the two things that have no safe default.
pub fn parse(bytes: &[u8]) -> Option<Plan> {
    let text = core::str::from_utf8(bytes).ok()?;
    let mut host = String::new();
    let mut port = 3333u16;
    let mut user = String::new();
    let mut pass = String::from("x");
    let mut glados_proto = false;
    let mut slices = None;
    let mut unknown = 0usize;

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, value) = match line.split_once(char::is_whitespace) {
            Some((k, v)) => (k, v.trim()),
            // A bare word is not a directive. Counted, so `pool` on its own --
            // which is the shape of a half-finished edit -- is reported rather
            // than read as a pool named nothing.
            None => {
                unknown += 1;
                continue;
            }
        };
        match key {
            "pool" => {
                // `stratum+tcp://` is stripped because it is what a pool's own
                // page gives you to paste, and refusing it would be refusing
                // the only form most people will ever have in front of them.
                let a = value.trim_start_matches("stratum+tcp://");
                match a.rsplit_once(':') {
                    Some((h, p)) => match p.parse::<u16>() {
                        Ok(n) => {
                            host = String::from(h);
                            port = n;
                        }
                        // A port that will not parse leaves the host unset, so
                        // the plan fails rather than silently dialling 3333 on
                        // a pool that asked for something else.
                        Err(_) => return None,
                    },
                    None => host = String::from(a),
                }
            }
            "worker" => user = String::from(value),
            "pass" => pass = String::from(value),
            "protocol" => match value {
                "glados" => glados_proto = true,
                "stratum" => glados_proto = false,
                _ => return None,
            },
            "slices" => match value.parse::<u32>() {
                Ok(n) => slices = Some(n),
                Err(_) => return None,
            },
            _ => unknown += 1,
        }
    }

    if host.is_empty() || user.is_empty() {
        return None;
    }
    Some(Plan { host, port, user, pass, glados_proto, slices, unknown })
}

/// Apply a plan and start mining. Answers a line to print.
///
/// Separate from `parse` for `update::repairs`'s reason: parsing is a pure
/// function of bytes and can be asserted without a network, a pool or a task
/// table, and everything that cannot be is here.
pub fn apply(p: &Plan) -> String {
    use alloc::format;
    {
        let mut g = super::client::CONFIG.lock_irq();
        *g = Some(super::client::Config {
            host: p.host.clone(),
            port: p.port,
            user: p.user.clone(),
            pass: p.pass.clone(),
            proto: if p.glados_proto {
                super::client::Protocol::Glados
            } else {
                super::client::Protocol::StratumV1
            },
        });
    }

    // **Standing down for the model is turned off, and it is not a micro-
    // optimisation.** `client::YIELD_TO_MODEL` defaults on because mining
    // doubles the time per token, and on an image with no checkpoint there is
    // no engine to yield to and never will be. Leaving it on costs an atomic
    // load per batch and, more to the point, leaves a switch in the machine
    // whose stated reason does not exist here.
    super::client::set_yield_to_model(false);

    let got = match p.slices {
        Some(n) => Some(super::client::set_slices(n)),
        None => None,
    };

    match super::client::start() {
        Ok(()) => {
            let mut s = format!("mining as {} at {}:{}", p.user, p.host, p.port);
            if let (Some(want), Some(n)) = (p.slices, got) {
                // Says what it got rather than what was asked for. `set_slices`
                // clamps to what the task table can actually spare, and a file
                // asking for four on a machine that can start two should say so
                // at boot rather than read as four that are not working.
                if n != want {
                    s.push_str(&format!(", {} of {} slice(s)", n, want));
                } else {
                    s.push_str(&format!(", {} slice(s)", n));
                }
            }
            if p.unknown > 0 {
                s.push_str(&format!(", {} line(s) not understood", p.unknown));
            }
            s
        }
        Err(e) => format!("miner configured but did not start: {}", e),
    }
}

/// Claims about the parser. No network, no pool, no task table.
pub fn checks() -> Vec<(bool, String)> {
    let mut out: Vec<(bool, String)> = Vec::new();
    let mut ok = |c: bool, w: &str| out.push((c, String::from(w)));

    let full = parse(b"pool p.example.com:3334\nworker 0xabc.rig1\nslices 3\n");
    ok(full.is_some(), "a file naming a pool and a worker is a plan");
    if let Some(p) = &full {
        ok(p.host == "p.example.com" && p.port == 3334, "the host and port are split on the last colon");
        ok(p.user == "0xabc.rig1", "the worker name is carried whole");
        ok(p.slices == Some(3), "and the slice count is a number");
        ok(p.pass == "x", "the password defaults to the convention every pool ignores");
        ok(!p.glados_proto, "stratum unless the file says otherwise");
    }

    ok(parse(b"worker 0xabc\n").is_none(), "a worker with no pool is not a plan");
    // The one that earns its place: mining to a default address is mining to
    // somebody else, and it is the failure that pays a stranger rather than
    // failing loudly.
    ok(parse(b"pool p.example.com:3334\n").is_none(), "and a pool with no worker is refused rather than defaulted");

    ok(parse(b"pool p.example.com\nworker w\n").map(|p| p.port) == Some(3333),
       "a pool with no port takes the stratum default");
    ok(parse(b"pool stratum+tcp://p.example.com:1\nworker w\n").map(|p| p.host)
           == Some(String::from("p.example.com")),
       "the scheme a pool's own page hands out is stripped");
    ok(parse(b"pool p:notaport\nworker w\n").is_none(), "a port that is not a number refuses the whole file");
    ok(parse(b"slices two\npool p:1\nworker w\n").is_none(), "and so does a slice count that is not one");
    ok(parse(b"protocol carrier-pigeon\npool p:1\nworker w\n").is_none(), "an unknown protocol is refused, not defaulted");
    ok(parse(b"protocol glados\npool p:1\nworker w\n").map(|p| p.glados_proto) == Some(true),
       "and a known one is taken");

    let noisy = parse(b"# a comment\n\npool p:1\nworker w\nfrobnicate 9\npool\n");
    ok(noisy.as_ref().map(|p| p.unknown) == Some(2),
       "comments and blanks are skipped, and what is left unread is counted");

    // Nothing here can reach a shell, and the claim is about the type rather
    // than about this input: `Plan` has no field a command could live in.
    ok(parse(b"pool p:1\nworker w\nrm -rf /\n").map(|p| p.unknown) == Some(1),
       "a line that looks like a command is one more thing not understood");
    ok(parse(&[0xff, 0xfe, 0x00]).is_none(), "bytes that are not text are not a plan");
    ok(parse(b"").is_none(), "and neither is an empty file");

    out
}
