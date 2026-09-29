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
//!
//! **And a typo is the same theft by a different route.** A default pays a
//! stranger; a transposed pair of characters pays nobody, and both look exactly
//! like a miner that is working. Every address form that turns up here carries a
//! checksum for precisely that reason, so `mine::addr` reads it before the first
//! share and the plan carries the verdict. Four answers rather than two, because
//! "this carries no checksum" and "this is a worker name the pool's roster maps"
//! are facts of their own and filing either under pass or fail is wrong.
use alloc::string::String;
use alloc::vec::Vec;

use super::addr;

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
    /// The WebSocket path, when the pool is reached over `wss://`.
    ///
    /// Present means TLS then a WebSocket upgrade; absent means a plain TCP
    /// connection. A Cloudflare Durable Object can only be reached the first
    /// way -- a Worker accepts no inbound TCP at all -- so this is what lets an
    /// image mine to the serverless pool with nothing beside it.
    pub ws: Option<String>,
    pub slices: Option<u32>,
    /// What could be established about where `user` pays.
    ///
    /// A field rather than a check inside `parse`, because it is a pure function
    /// of the name and so belongs in what a claim can read -- and because
    /// refusing here would throw away the reason. `apply` is what declines, where
    /// there is a line to print it on.
    pub payout: addr::Payout,
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
    let mut said_protocol = false;
    let mut ws = None;
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
                let mut a = value.trim_start_matches("stratum+tcp://");
                let mut default_port = 3333u16;
                if let Some(rest) = a.strip_prefix("wss://") {
                    // The path is everything from the first slash, and `/mine`
                    // when there is none: that is the only path the pool serves
                    // miners on, and a bare host is how people will write it.
                    let (hp, path) = match rest.find('/') {
                        Some(i) => (&rest[..i], &rest[i..]),
                        None => (rest, "/mine"),
                    };
                    ws = Some(String::from(path));
                    default_port = 443;
                    a = hp;
                }
                port = default_port;
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
                "glados" => {
                    glados_proto = true;
                    said_protocol = true;
                }
                "stratum" => {
                    glados_proto = false;
                    said_protocol = true;
                }
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
    // The serverless pool speaks the native protocol and nothing else, so a
    // `wss://` pool means it unless the file says otherwise. A miner's whole
    // configuration is then two lines: where, and whose address.
    if ws.is_some() && !said_protocol {
        glados_proto = true;
    }
    let payout = addr::judge(&user);
    Some(Plan { host, port, user, pass, glados_proto, ws, slices, payout, unknown })
}

/// Apply a plan and start mining. Answers a line to print.
///
/// Separate from `parse` for `update::repairs`'s reason: parsing is a pure
/// function of bytes and can be asserted without a network, a pool or a task
/// table, and everything that cannot be is here.
pub fn apply(p: &Plan) -> String {
    use alloc::format;

    // **Refused before anything is configured, not after.** A broken checksum
    // means the string is not an address at all, so every share found under it
    // is work given away -- and an image on read-only media will do it again
    // every boot, for as long as nobody looks. The refusal names the form and
    // the name, because the operator has to find the character that is wrong in
    // a file they can no longer edit and the next thing they do is cut another
    // image.
    if !p.payout.may_mine() {
        return format!("not mining: {} ({})", p.payout.say(), p.user);
    }
    {
        let mut g = super::client::CONFIG.lock_irq();
        *g = Some(super::client::Config {
            host: p.host.clone(),
            port: p.port,
            user: p.user.clone(),
            pass: p.pass.clone(),
            ws: p.ws.clone(),
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

    // **Take the whole screen, because the console does not have it.**
    // `desk.rs` reflows the console into the terminal *window's* grid during
    // layout, and that happens on a miner image too -- the compositor is never
    // spawned, but the one pass that ran left the console about a quarter of the
    // display wide. Drawing a dashboard into it produced a border that stopped
    // a third of the way across and nothing else visible, which reads as a
    // broken renderer and is a console doing exactly what it was told.
    //
    // `set_exclusive` is the other half: it is what makes the desktop's periodic
    // painters stand down, the same flag `port::with_screen` takes for the length
    // of a call and this holds for the life of the machine.
    if let Some(fb) = crate::gfx::primary() {
        crate::gfx::set_exclusive(true);
        crate::gfx::console::with(|c| c.reflow(0, 0, fb.width(), fb.height()));
    }

    // No task for the screen: `client::run` draws it, because this kernel has no
    // sleep and a once-a-second task can only spin. `screen::tick` is one
    // comparison on a loop that already wakes every 200 ms.
    super::screen::wipe();

    match super::client::start() {
        Ok(()) => {
            let scheme = if p.ws.is_some() { "wss://" } else { "" };
            let mut s = format!("mining as {} at {}{}:{}", p.user, scheme, p.host, p.port);
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
            // Said on the way up rather than only on a refusal: "checks out" and
            // "carries no checksum" are different assurances and the operator is
            // owed which one they have before walking away from the machine.
            s.push_str(&format!("\n  {}", p.payout.say()));
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
    ok(
        full.as_ref().map(|p| p.payout) == Some(addr::Payout::Name),
        "and a worker that is not an address shape is carried as a name",
    );
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
    let edge = parse(b"pool wss://pool.example.com/mine\nworker w\n");
    ok(
        edge.as_ref().map(|p| (p.host.as_str(), p.port, p.ws.as_deref())) == Some(("pool.example.com", 443, Some("/mine"))),
        "a wss pool is split into host, port 443 and its path",
    );
    ok(edge.map(|p| p.glados_proto) == Some(true), "and speaks the native protocol without being told");
    ok(
        parse(b"pool wss://h:8443\nworker w\n").map(|p| (p.port, p.ws)) == Some((8443, Some(String::from("/mine")))),
        "a wss pool with a port and no path takes /mine",
    );
    ok(
        parse(b"protocol stratum\npool wss://h\nworker w\n").map(|p| p.glados_proto) == Some(false),
        "a protocol line still wins over the scheme's default",
    );
    ok(parse(b"pool h:1\nworker w\n").map(|p| p.ws) == Some(None), "a plain pool is plain TCP");
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

    // **The payout verdict reaches the plan, and the broken one is the point.**
    // `addr` asserts its own arithmetic; what is asserted here is the join --
    // that a file naming a mistyped address parses into a plan that says so,
    // rather than into one that mines.
    let good = parse(b"pool p:1\nworker 0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed\n");
    ok(
        good.as_ref().map(|p| p.payout) == Some(addr::Payout::Checked(addr::Kind::Evm)),
        "an EIP-55 payout address in the file is checked and passes",
    );
    ok(good.map(|p| p.payout.may_mine()) == Some(true), "and such a plan may mine");

    let typo = parse(b"pool p:1\nworker 0x5aAeb6053F3E94C9b9A09f33669435E7Ef1Beaed\n");
    ok(
        typo.as_ref().map(|p| p.payout) == Some(addr::Payout::Broken(addr::Kind::Evm)),
        "one letter's case wrong in it is carried as broken",
    );
    ok(
        typo.as_ref().map(|p| p.payout.may_mine()) == Some(false),
        "and such a plan may not mine",
    );
    // Still a plan, deliberately: parsing succeeded and it is `apply` that
    // declines, so the reason survives to be printed instead of becoming a
    // `None` indistinguishable from an absent file.
    ok(typo.is_some(), "a broken address still parses, so the refusal can name it");

    let lower = parse(b"pool p:1\nworker 0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaed\n");
    ok(
        lower.as_ref().map(|p| p.payout) == Some(addr::Payout::Unchecked(addr::Kind::Evm)),
        "an all-lowercase address is well-formed with nothing to verify",
    );
    ok(lower.map(|p| p.payout.may_mine()) == Some(true), "and it is allowed rather than refused");

    // The suffix, through the file rather than through `judge` directly, because
    // this is the spelling every multi-rig venue's own page hands out.
    ok(
        parse(b"pool p:1\nworker 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa.gf63\n")
            .map(|p| p.payout)
            == Some(addr::Payout::Checked(addr::Kind::Base58)),
        "and an address with a rig suffix is still checked",
    );

    out.extend(addr::checks());
    out
}
