//! A command shell reading from the keyboard ring buffer.
//!
//! In TempleOS the shell and the compiler were the same thing: what you typed
//! was compiled and run. This is not that yet -- it dispatches on fixed command
//! names. M5 replaces `execute` with a tokeniser, a parser and a JIT, at which
//! point this file mostly disappears.

use crate::acpi::Acpi;
use crate::cpu::idt;
use crate::dev::{kbd, lapic};
use crate::gfx::console::{self, LTCYAN, LTGRAY, LTGREEN, LTRED, WHITE, YELLOW};
use crate::lang;
use crate::mem;
use crate::BootInfo;
use crate::{kprint, kprintln, serial_println};
use alloc::string::String;
use alloc::vec::Vec;

const PROMPT: &str = "glados> ";
/// Derived, not written out. Hardcoding this was fine until the prompt changed
/// length during the rename, at which point every cursor position in the line
/// editor was off by one.
const PROMPT_LEN: usize = PROMPT.len();
const HISTORY_MAX: usize = 64;

fn prompt() {
    console::set_color(LTGREEN);
    kprint!("\n{}", PROMPT);
    console::set_color(WHITE);
}

/// Redraw the prompt after something else has printed over it.
///
/// A background task writing to the console leaves the cursor wherever its
/// output ended, and the line the operator was typing scrolled away with it.
/// This at least restores the prompt so the shell does not look dead.
pub fn reprompt() {
    prompt();
}

/// Say why a command did nothing, when the mind holds the engine.
///
/// The refusal itself is enforced inside `with_engine`, not here. This existed
/// as a guarded match arm sitting *after* the arms it guarded, so it was
/// unreachable and protected nothing; the compiler had been reporting it as an
/// unreachable pattern the whole time.
fn note_if_mind_busy() {
    if crate::ai::engine_held_by_mind() {
        console::set_color(YELLOW);
        kprintln!("  the mind is using the model -- try again when it finishes");
        console::set_color(WHITE);
    }
}

/// Repaint the edited line and place the cursor.
///
/// Deliberately console-only rather than `kprint!`: echoing every keystroke and
/// every redraw to the serial port would bury the log in partial lines. The
/// finished line is written to serial once, when Enter is pressed.
///
/// Assumes the line fits on one row. Longer input wraps and the cursor
/// arithmetic stops being right -- a limitation, not a crash.
fn redraw(line: &str, cursor: usize, prev_len: usize) {
    console::with(|c| {
        c.set_col(PROMPT_LEN);
        c.write_bytes(line.as_bytes());
        for _ in line.len()..prev_len {
            c.put_char(b' ');
        }
        c.set_col(PROMPT_LEN + cursor);
    });
}

pub fn run(boot: &BootInfo, acpi: &Option<Acpi>) -> ! {
    console::set_color(LTCYAN);
    kprintln!("\ninteractive. type 'help', or just type code.");
    console::set_color(WHITE);

    let mut interp = lang::Interp::new();
    let mut history: Vec<String> = Vec::new();
    let mut line = String::new();
    let mut cursor = 0usize;
    // Equal to history.len() means "editing a fresh line", not browsing.
    let mut hist = 0usize;
    let mut stash = String::new();

    prompt();

    loop {
        let key = if let Some(k) = kbd::pop_any() {
            k
        } else {
            // Nothing queued: idle until the next interrupt rather than
            // spinning. The timer alone wakes us 100 times a second.
            unsafe { core::arch::asm!("hlt", options(nomem, nostack)) };
            continue;
        };

        let prev = line.len();
        match key {
            b'\n' => {
                console::with(|c| c.set_col(PROMPT_LEN + line.len()));
                kprintln!();
                // The one place the typed line reaches the serial log.
                serial_println!("{}{}", PROMPT, line);

                let trimmed = line.trim();
                if !trimmed.is_empty() && history.last().map(|h| h.as_str()) != Some(trimmed) {
                    history.push(String::from(trimmed));
                    if history.len() > HISTORY_MAX {
                        history.remove(0);
                    }
                }
                // Re-arm pacing per command, so a skip requested during one
                // command's output does not silently disable it forever.
                console::resume_pacing();
                // A pipeline runs `execute` itself, with the console captured.
                if !run_pipeline(trimmed, boot, acpi, &mut interp) {
                    execute(trimmed, boot, acpi, &mut interp);
                }
                // Between commands is the only place the namespace is
                // guaranteed to be whole, so this is where an automatic
                // snapshot is allowed to run.
                crate::sysbox::autosnap_poll();
                // Anything needing the model will have quietly done nothing if
                // the mind had it; say so once, here, rather than at each of a
                // dozen call sites that would drift out of step.
                note_if_mind_busy();

                line.clear();
                cursor = 0;
                hist = history.len();
                stash.clear();
                prompt();
            }

            8 => {
                if cursor > 0 {
                    cursor -= 1;
                    line.remove(cursor);
                    redraw(&line, cursor, prev);
                }
            }
            kbd::KEY_DELETE => {
                if cursor < line.len() {
                    line.remove(cursor);
                    redraw(&line, cursor, prev);
                }
            }

            kbd::KEY_LEFT => {
                if cursor > 0 {
                    cursor -= 1;
                    console::set_col(PROMPT_LEN + cursor);
                }
            }
            kbd::KEY_RIGHT => {
                if cursor < line.len() {
                    cursor += 1;
                    console::set_col(PROMPT_LEN + cursor);
                }
            }
            // Ctrl-A / Ctrl-E, for the same reason every other shell has them.
            kbd::KEY_HOME | 0x01 => {
                cursor = 0;
                console::set_col(PROMPT_LEN);
            }
            kbd::KEY_END | 0x05 => {
                cursor = line.len();
                console::set_col(PROMPT_LEN + cursor);
            }
            // Ctrl-U: scrap the line.
            0x15 => {
                line.clear();
                cursor = 0;
                redraw(&line, cursor, prev);
            }

            kbd::KEY_UP => {
                if hist > 0 {
                    if hist == history.len() {
                        stash = line.clone();
                    }
                    hist -= 1;
                    line = history[hist].clone();
                    cursor = line.len();
                    redraw(&line, cursor, prev);
                }
            }
            kbd::KEY_DOWN => {
                if hist < history.len() {
                    hist += 1;
                    line = if hist == history.len() {
                        stash.clone()
                    } else {
                        history[hist].clone()
                    };
                    cursor = line.len();
                    redraw(&line, cursor, prev);
                }
            }

            ch if (0x20..0x7F).contains(&ch) => {
                line.insert(cursor, ch as char);
                cursor += 1;
                redraw(&line, cursor, prev);
            }
            _ => {}
        }
    }
}

fn store_cmd(rest: &str) {
    use crate::store::{self, cas, sha256};

    match rest {
        "" | "status" => {
            console::set_color(YELLOW);
            kprintln!("[store]");
            console::set_color(WHITE);
            let sha_ok = sha256::selftest();
            console::set_color(if sha_ok { LTGREEN } else { LTRED });
            kprintln!("  sha-256 vectors: {}", if sha_ok { "pass" } else { "FAIL" });
            console::set_color(WHITE);
            if !store::mounted() {
                kprintln!("  not mounted. 'store init' to format free space.");
                return;
            }
            store::with(|s| {
                kprintln!(
                    "  region lba {}..{}  seq {}  checkpoints {}",
                    s.sb.region_start,
                    s.sb.region_start + s.sb.region_blocks,
                    s.sb.seq,
                    s.sb.checkpoints
                );
                kprintln!(
                    "  next free lba {}  ({} blocks left)",
                    s.sb.alloc_next,
                    s.free_blocks()
                );
                if s.sb.root.is_none() {
                    kprintln!("  no checkpoints yet");
                } else {
                    let h = sha256::short_hex(&s.sb.root.hash);
                    kprintln!(
                        "  root {}  at lba {} ({} B)",
                        core::str::from_utf8(&h).unwrap_or("?"),
                        s.sb.root.lba,
                        s.sb.root.len
                    );
                }
            });
        }

        "init" => match store::init() {
            Ok((start, blocks)) => {
                console::set_color(LTGREEN);
                kprintln!("  formatted lba {}..{} ({} blocks)", start, start + blocks, blocks);
                console::set_color(LTRED);
                kprintln!("  NVMe writes are now UNLOCKED for this session");
                console::set_color(WHITE);
            }
            Err(e) => {
                console::set_color(LTRED);
                match e {
                    store::InitError::NoRoom => {
                        kprintln!("  no unclaimed space on this disk -- refusing to write");
                        kprintln!("  (expected on a disk fully allocated to Windows)");
                    }
                    other => kprintln!("  init failed: {:?}", other),
                }
                console::set_color(WHITE);
            }
        },

        "unlock" => match store::unlock() {
            Ok((start, blocks)) => {
                console::set_color(LTRED);
                kprintln!("  writes UNLOCKED for lba {}..{}", start, start + blocks);
                console::set_color(WHITE);
                kprintln!("  region re-checked against the partition table first");
            }
            Err(e) => {
                console::set_color(LTRED);
                match e {
                    cas::Error::Unsafe => kprintln!("  region overlaps a partition that is not ours -- refusing"),
                    cas::Error::NotFormatted => kprintln!("  no store mounted"),
                    other => kprintln!("  {:?}", other),
                }
                console::set_color(WHITE);
            }
        },

        "test" => {
            if !store::mounted() {
                console::set_color(LTRED);
                kprintln!("  not mounted -- run 'store init' first");
                console::set_color(WHITE);
                return;
            }
            if !crate::dev::nvme::writes_unlocked() {
                console::set_color(LTRED);
                kprintln!("  writes are locked -- run 'store unlock' first");
                console::set_color(WHITE);
                return;
            }
            let ok = store::with(|s| {
                let mut entries: alloc::vec::Vec<cas::Entry> = alloc::vec::Vec::new();
                let payloads: [&[u8]; 3] = [
                    b"the first blob",
                    b"a second, longer blob that spans more of a block",
                    b"third",
                ];
                for (i, p) in payloads.iter().enumerate() {
                    match s.put(p) {
                        Ok(c) => {
                            let mut name = [0u8; cas::NAME_LEN];
                            name[0] = b'b';
                            name[1] = b'0' + i as u8;
                            entries.push(cas::Entry { name, chunk: c });
                        }
                        Err(e) => {
                            kprintln!("  put failed: {:?}", e);
                            return false;
                        }
                    }
                }
                // Read back and verify content addressing before committing.
                for (i, e) in entries.iter().enumerate() {
                    match s.get(&e.chunk) {
                        Ok(d) => {
                            if d.as_slice() != payloads[i] {
                                kprintln!("  blob {} round-trip MISMATCH", i);
                                return false;
                            }
                        }
                        Err(err) => {
                            kprintln!("  get failed: {:?}", err);
                            return false;
                        }
                    }
                }
                match s.commit(&entries) {
                    Ok(r) => {
                        let h = sha256::short_hex(&r.hash);
                        kprintln!(
                            "  committed {} entries, root {}",
                            entries.len(),
                            core::str::from_utf8(&h).unwrap_or("?")
                        );
                        true
                    }
                    Err(e) => {
                        kprintln!("  commit failed: {:?}", e);
                        false
                    }
                }
            })
            .unwrap_or(false);

            console::set_color(if ok { LTGREEN } else { LTRED });
            kprintln!("  {}", if ok { "put / get / commit verified" } else { "STORE TEST FAILED" });
            console::set_color(WHITE);
        }

        "log" => {
            if !store::mounted() {
                kprintln!("  not mounted");
                return;
            }
            store::with(|s| {
                let mut r = s.sb.root;
                let mut n = 0;
                console::set_color(YELLOW);
                kprintln!("  seq   root              entries");
                console::set_color(WHITE);
                while !r.is_none() && n < 32 {
                    match s.read_manifest(&r) {
                        Ok(m) => {
                            let h = sha256::short_hex(&r.hash);
                            kprintln!(
                                "  {:<5} {}  {}",
                                m.seq,
                                core::str::from_utf8(&h).unwrap_or("?"),
                                m.entries.len()
                            );
                            r = m.prev;
                        }
                        Err(e) => {
                            console::set_color(LTRED);
                            kprintln!("  chain broken: {:?}", e);
                            console::set_color(WHITE);
                            break;
                        }
                    }
                    n += 1;
                }
                if n == 0 {
                    kprintln!("  (no checkpoints)");
                }
            });
        }

        "rollback" => {
            if !store::mounted() {
                kprintln!("  not mounted");
                return;
            }
            store::with(|s| match s.read_manifest(&s.sb.root) {
                Ok(m) => {
                    if m.prev.is_none() {
                        kprintln!("  already at the first checkpoint");
                        return;
                    }
                    match s.rollback_to(m.prev) {
                        Ok(()) => {
                            console::set_color(LTGREEN);
                            kprintln!("  rolled back to seq {}", s.sb.seq);
                            console::set_color(WHITE);
                            kprintln!("  nothing was erased; roll forward is still possible");
                        }
                        Err(e) => kprintln!("  rollback failed: {:?}", e),
                    }
                }
                Err(e) => kprintln!("  cannot read current manifest: {:?}", e),
            });
        }

        other => {
            console::set_color(LTRED);
            kprintln!("  unknown: store {}", other);
            console::set_color(WHITE);
            kprintln!("  store [status|init|unlock|test|log|rollback]");
        }
    }
}

fn execute(line: &str, boot: &BootInfo, acpi: &Option<Acpi>, interp: &mut lang::Interp) {
    let mut parts = line.splitn(2, ' ');
    let cmd = parts.next().unwrap_or("");
    let rest = parts.next().unwrap_or("").trim();

    // sysbox first: it owns a whole vocabulary of short names, and claiming
    // them here keeps that list in one place instead of spreading twenty more
    // arms across this match.
    if crate::sysbox::dispatch(cmd, rest) {
        return;
    }

    match cmd {
        "" => {}
        "typewriter" => {
            match rest.parse::<u64>() {
                Ok(us) => {
                    console::set_pace(us);
                    if us == 0 {
                        kprintln!("  pacing off");
                    } else {
                        kprintln!("  {} us per character", us);
                    }
                }
                Err(_) => {
                    kprintln!("  {} us per character", console::pace_us());
                    kprintln!("  'typewriter <us>' to change, 'typewriter 0' for instant");
                    kprintln!("  any keypress skips pacing for the rest of a command");
                }
            }
        }
        "refresh" => console::redraw(),
        "fat" => fat_cmd(rest),
        "net" => {
            let mut it = rest.split_whitespace();
            match it.next() {
                Some("ip") => {
                    let mut c = crate::net::config();
                    if let Some(ip) = it.next().and_then(crate::net::parse_ip) {
                        c.ip = ip;
                    }
                    if let Some(gw) = it.next().and_then(crate::net::parse_ip) {
                        c.gateway = gw;
                    }
                    crate::net::set_config(c);
                    crate::net::report();
                }
                _ => crate::net::report(),
            }
        }
        "ping" => {
            let mut it = rest.split_whitespace();
            match it.next().and_then(crate::net::parse_ip) {
                Some(ip) => {
                    let n = it.next().and_then(|s| s.parse().ok()).unwrap_or(4);
                    crate::net::ping(ip, n);
                }
                None => kprintln!("  usage: ping <a.b.c.d> [count]"),
            }
        }
        "pkg" => {
            let mut it = rest.splitn(2, ' ');
            let verb = it.next().unwrap_or("");
            let arg = it.next().unwrap_or("").trim();
            match verb {
                "" | "list" => crate::pkg::list(),
                "info" => crate::pkg::info(arg),
                "remove" => crate::pkg::remove(arg),
                "add" => match crate::sysbox::read_blob(arg) {
                    Some(bytes) => {
                        crate::pkg::install(&bytes);
                    }
                    None => kprintln!("  no such file: {}", arg),
                },
                other => kprintln!("  not a pkg subcommand: {}", other),
            }
        }
        "edit" | "vi" => {
            if rest.is_empty() {
                kprintln!("  usage: edit <path>");
            } else {
                crate::edit::run(rest);
            }
        }
        "date" => match crate::dev::rtc::now() {
            Some(d) => {
                kprintln!(
                    "  {:04}-{:02}-{:02} {:02}:{:02}:{:02}",
                    d.year, d.month, d.day, d.hour, d.minute, d.second
                );
                kprintln!("  {} seconds since 1970", crate::dev::rtc::unix_seconds(&d));
            }
            None => kprintln!("  no usable RTC on this machine"),
        },
        "autosnap" => {
            let mut it = rest.split_whitespace();
            match it.next() {
                Some("on") => {
                    let secs = it.next().and_then(|s| s.parse().ok()).unwrap_or(60);
                    crate::sysbox::autosnap_configure(true, secs);
                }
                Some("off") => crate::sysbox::autosnap_configure(false, 0),
                _ => {}
            }
            crate::sysbox::autosnap_report();
        }
        "act" => {
            // Read-only unless explicitly told otherwise, and "trusted" has to
            // be typed in full. The distinction is enforced in the grammar, not
            // after the fact: with ReadOnly the mutating applets are not
            // reachable outputs at all.
            let (trust, task) = match rest.strip_prefix("trusted ") {
                Some(t) => (crate::ai::harness::Trust::Full, t.trim()),
                None => (crate::ai::harness::Trust::ReadOnly, rest),
            };
            let task = if task.is_empty() { "look at the files" } else { task };
            crate::ai::harness::report(task, trust, 1.0);
        }
        // `teach file <path>` reads "applet task" one per line, so a batch of
        // examples can be written in the editor instead of one shell line at a
        // time. Whitespace-separated rather than tab-separated: a tab is
        // awkward to type in a modal editor, and the applet name never
        // contains a space.
        "teach" if rest.starts_with("file ") => {
            let path = rest[5..].trim();
            match crate::sysbox::read_blob(path) {
                None => kprintln!("  no such file: {}", path),
                Some(bytes) => {
                    let text = core::str::from_utf8(&bytes).unwrap_or("");
                    let (mut ok, mut bad) = (0usize, 0usize);
                    for line in text.lines() {
                        let line = line.trim();
                        if line.is_empty() || line.starts_with('#') {
                            continue;
                        }
                        let mut p = line.splitn(2, char::is_whitespace);
                        match (p.next(), p.next()) {
                            (Some(applet), Some(task)) if !task.trim().is_empty() => {
                                if crate::sysbox::APPLETS.iter().any(|a| a.name == applet)
                                    && crate::ai::vocab::record(applet, task.trim())
                                {
                                    ok += 1;
                                } else {
                                    bad += 1;
                                    kprintln!("  skipped: {}", line);
                                }
                            }
                            _ => {
                                bad += 1;
                                kprintln!("  skipped: {}", line);
                            }
                        }
                    }
                    kprintln!("  {} example(s) recorded, {} skipped", ok, bad);
                    if ok > 0 {
                        kprintln!("  'fit' to rebuild the router from them");
                    }
                }
            }
        }
        "teach" => {
            let mut it = rest.splitn(2, ' ');
            match (it.next(), it.next()) {
                (Some(applet), Some(task)) if !task.trim().is_empty() => {
                    if crate::sysbox::APPLETS.iter().any(|a| a.name == applet) {
                        if crate::ai::vocab::record(applet, task.trim()) {
                            kprintln!("  recorded: {} <- \"{}\"", applet, task.trim());
                        } else {
                            kprintln!("  could not write to {}", crate::ai::vocab::CORPUS);
                        }
                    } else {
                        kprintln!("  '{}' is not an applet", applet);
                    }
                }
                _ => kprintln!("  usage: teach <applet> <task description>"),
            }
        }
        "ctx" => {
            let mut it = rest.splitn(2, ' ');
            match (it.next().unwrap_or(""), it.next().unwrap_or("").trim()) {
                ("save", name) if !name.is_empty() => match crate::ai::ctx_save(name) {
                    Some(n) => kprintln!("  saved {} B to {}/{}", n, crate::ai::CTX_DIR, name),
                    None => kprintln!("  could not save"),
                },
                ("load", name) if !name.is_empty() => match crate::ai::ctx_load(name) {
                    Some(p) => kprintln!("  restored to position {}", p),
                    None => kprintln!("  no such context, or it does not fit this model"),
                },
                _ => crate::ai::ctx_report(),
            }
        }
        "cont" => {
            let mut opts = crate::ai::GenOpts { resume: true, ..Default::default() };
            opts.steps = 60;
            crate::ai::generate(rest, &opts);
        }
        "ask" => {
            let mut opts = crate::ai::GenOpts { steps: 64, temperature: 0.3, ..Default::default() };
            let mut q = rest;
            if let Some(t) = q.strip_prefix("-n ") {
                let mut it = t.splitn(2, ' ');
                if let (Some(n), Some(tail)) = (it.next(), it.next()) {
                    opts.steps = n.parse().unwrap_or(opts.steps);
                    q = tail.trim_start();
                }
            }
            if q.is_empty() {
                kprintln!("  usage: ask [-n tokens] <question>");
            } else {
                crate::ai::chat(q, &opts);
            }
        }
        "logits" => {
            let ids: Vec<usize> = rest.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            crate::ai::logits_for(&ids);
        }
        "fit" => {
            let lambda: f32 = rest.split_whitespace().next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0);
            crate::ai::harness::fit_probe(lambda);
        }
        "route" => {
            let (trust, task) = match rest.strip_prefix("trusted ") {
                Some(t) => (crate::ai::harness::Trust::Full, t.trim()),
                None => (crate::ai::harness::Trust::ReadOnly, rest),
            };
            if task.is_empty() {
                kprintln!("  usage: route [trusted] <task>");
            } else {
                crate::ai::harness::route_report(task, trust);
            }
        }
        "window" => {
            let n: Vec<usize> = rest.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if n.len() == 2 {
                crate::ai::set_window(n[0], n[1]);
            }
            crate::ai::window_report();
        }
        "gate" => crate::ai::harness::gate_report(),
        "search" => crate::ai::harness::search_report(),
        "probe" => crate::ai::harness::probe_features(),
        "feature" => {
            use crate::ai::harness::{set_feature_mode, Feature};
            match rest {
                "hidden" => { set_feature_mode(Feature::Hidden); kprintln!("  feature = hidden state"); }
                "pooled" => { set_feature_mode(Feature::Pooled); kprintln!("  feature = pooled embedding"); }
                _ => kprintln!("  usage: feature hidden|pooled"),
            }
        }
        "zeroshot" => crate::ai::harness::zero_shot_report(if rest.is_empty() { "diff" } else { rest }),
        "train" => {
            let epochs: usize = rest.split_whitespace().next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(20);
            crate::ai::harness::train_report(epochs);
        }
        "think" => {
            if rest.is_empty() {
                kprintln!("  usage: think <prompt>   (runs in the background)");
            } else if crate::ai::think(rest) {
                kprintln!("  queued -- the shell stays yours while it runs");
            } else {
                kprintln!("  a request is already pending");
            }
        }
        "gen" => {
            // `gen -t 0 once upon a time` -- flags first, everything after is
            // the prompt verbatim, so it can contain spaces without quoting.
            let mut opts = crate::ai::GenOpts::default();
            let mut prompt = rest;
            loop {
                let mut it = prompt.splitn(3, ' ');
                let flag = it.next().unwrap_or("");
                if !matches!(flag, "-t" | "-p" | "-n") {
                    break;
                }
                let Some(value) = it.next() else { break };
                let tail = it.next().unwrap_or("");
                match flag {
                    "-t" => opts.temperature = value.parse().unwrap_or(opts.temperature),
                    "-p" => opts.topp = value.parse().unwrap_or(opts.topp),
                    _ => opts.steps = value.parse().unwrap_or(opts.steps),
                }
                prompt = tail.trim_start();
            }
            if prompt.is_empty() {
                prompt = "Once upon a time";
            }
            crate::ai::generate(prompt, &opts);
        }
        // Grouped rather than alphabetical: the list is long enough now that
        // finding a command matters more than enumerating them. Each group is
        // one subsystem, in the order you would meet them.
        "help" => {
            console::set_color(YELLOW);
            kprintln!("machine");
            console::set_color(WHITE);
            kprintln!("  mem uptime tasks cpu acpi pci video date reboot");
            kprintln!("  fault         deliberately dereference null");
            kprintln!("  clear refresh echo <text>");
            kprintln!("  typewriter    output pacing, in us per character");

            console::set_color(YELLOW);
            kprintln!("\nstorage");
            console::set_color(WHITE);
            kprintln!("  nvme disk     controller and namespace state");
            kprintln!("  store         init/unlock/test/log/rollback -- 'store' for the list");
            kprintln!("  autosnap      snapshot the namespace on every write");
            kprintln!("  fat           read a FAT16/32 volume: fat ls|cat <path>");
            kprintln!("  pkg           list/info/add/remove content-addressed packages");
            kprintln!("  edit <path>   modal editor ('vi' works too)");

            console::set_color(YELLOW);
            kprintln!("\nnetwork");
            console::set_color(WHITE);
            kprintln!("  net           link, addresses, ARP cache");
            kprintln!("  net ip <addr> [gw]    set them");
            kprintln!("  ping <addr> [count]");

            console::set_color(YELLOW);
            kprintln!("\nthe model");
            console::set_color(WHITE);
            kprintln!("  gen <prompt>  generate text     ask <prompt>  chat turn");
            kprintln!("  think <p>     run it in the background, off the shell");
            kprintln!("  act <task>    choose an applet by constrained decoding");
            kprintln!("  route <task>  choose one with the probe -- no transformer");
            kprintln!("  teach <applet> <task>   add an example ('teach file <path>' for many)");
            kprintln!("  fit [lambda]  refit the probe and the council on what it knows");
            kprintln!("  gate search   how often agreement is right; the config search");
            kprintln!("  ctx cont window logits probe feature zeroshot train");
            console::set_color(YELLOW);
            kprintln!("  'act' and 'route' are read-only unless you type 'trusted' first");
            console::set_color(WHITE);

            console::set_color(YELLOW);
            kprintln!("\nsysbox owns the namespace -- type 'sysbox' for its applets");
            kprintln!("a command may be piped into a filter: <cmd> | grep|head|tail|sort|wc");
            console::set_color(WHITE);

            console::set_color(YELLOW);
            kprintln!("\nanything else is evaluated as code -- 'words' and 'vars' to look around");
            console::set_color(WHITE);
            kprintln!("  x = 6*7               println(\"hi\", x)");
            kprintln!("  i=0 while(i<8){{ rect(i*40,300,32,32,i+1) i=i+1 }}");
            kprintln!("  hex(peek32(0xfec00000))");
        }
        "mem" => {
            let (used, total) = mem::heap::HEAP.stats();
            kprintln!("  heap  {} B used / {} B total", used, total);
            kprintln!("  free  {} KiB", (total - used) / 1024);
        }
        "uptime" => {
            let t = lapic::ticks();
            let hz = crate::TIMER_HZ as u64;
            kprintln!("  {} ticks  ({}.{:02} s at {} Hz)", t, t / hz, (t % hz) * 100 / hz, hz);
            kprintln!("  apic timer calibrated at {} Hz", lapic::timer_hz());
        }
        "acpi" => match acpi {
            Some(a) => {
                kprintln!("  revision {}  cpus {}", a.revision, a.cpus);
                kprintln!("  lapic    {:#x}  (id {})", a.lapic_addr, lapic::id());
                for i in 0..a.ioapic_count {
                    let io = a.ioapics[i];
                    kprintln!("  ioapic {} {:#x}  gsi base {}", io.id, io.addr, io.gsi_base);
                }
                for i in 0..a.override_count {
                    let o = a.overrides[i];
                    kprintln!("  irq {} -> gsi {}  flags {:#06x}", o.source, o.gsi, o.flags);
                }
            }
            None => {
                console::set_color(LTRED);
                kprintln!("  ACPI was not parsed");
                console::set_color(WHITE);
            }
        },
        "tasks" => {
            kprintln!(
                "  {} tasks, {} switches total",
                crate::task::count(),
                crate::task::total_switches()
            );
            for i in 0..crate::task::count() {
                if let Some(t) = crate::task::snapshot(i) {
                    let marker = if i == crate::task::current() { '*' } else { ' ' };
                    kprintln!(
                        "  {}{} {:<8} rsp {:#018x}  resumed {}",
                        marker,
                        i,
                        t.name,
                        t.rsp,
                        t.switches
                    );
                }
            }
            // Only advances while the clock task is actually on the CPU, so a
            // rising number here is proof of preemption rather than of a timer.
            kprintln!("  clock task iterations: {}", crate::clock_iterations());
            kprintln!(
                "  fpu state per task: {} B ({})",
                crate::task::fpu_area_bytes(),
                if crate::cpu::detected().avx_enabled { "xsave" } else { "fxsave" }
            );
            let checks = crate::ai::fpu_checks();
            let errs = crate::ai::fpu_errors();
            console::set_color(if errs == 0 { LTGREEN } else { LTRED });
            kprintln!("  ymm survived {} preemption checks, {} corrupted", checks, errs);
            console::set_color(WHITE);
        }
        "pci" => match acpi.as_ref().and_then(|a| a.mcfg) {
            Some(ecam) => {
                console::set_color(YELLOW);
                kprintln!("  bb:dd.f  vendor device  class");
                console::set_color(WHITE);
                let mut n = 0usize;
                crate::dev::pci::scan(ecam, 255, |d| {
                    n += 1;
                    let vname = crate::dev::pci::vendor_name(d.vendor);
                    kprintln!(
                        "  {:02x}:{:02x}.{}  {:04x} {:04x}  {} {}",
                        d.bus,
                        d.dev,
                        d.func,
                        d.vendor,
                        d.device,
                        crate::dev::pci::class_name(d.class, d.subclass),
                        vname
                    );
                });
                kprintln!("  {} functions found via ecam at {:#x}", n, ecam);
            }
            None => {
                console::set_color(LTRED);
                kprintln!("  no MCFG table, so no ECAM base to scan");
                console::set_color(WHITE);
            }
        },
        "nvme" => {
            let Some(ecam) = acpi.as_ref().and_then(|a| a.mcfg) else {
                console::set_color(LTRED);
                kprintln!("  no MCFG, so no ECAM base");
                console::set_color(WHITE);
                return;
            };
            if !crate::dev::nvme::present() {
                kprintln!("  probing...");
                match crate::dev::nvme::init(ecam) {
                    Ok(()) => {}
                    Err(e) => {
                        console::set_color(LTRED);
                        kprintln!("  init failed: {:?}", e);
                        console::set_color(WHITE);
                        return;
                    }
                }
            }
            crate::dev::nvme::with(|n| {
                // A nested fn, not a closure: closure lifetime elision cannot
                // tie the returned &str back to the input slice.
                fn trim(b: &[u8]) -> &str {
                    let mut end = b.len();
                    while end > 0 && (b[end - 1] == b' ' || b[end - 1] == 0) {
                        end -= 1;
                    }
                    core::str::from_utf8(&b[..end]).unwrap_or("?")
                }
                console::set_color(YELLOW);
                kprintln!("[nvme]");
                console::set_color(WHITE);
                kprintln!("  model   {}", trim(&n.model));
                kprintln!("  serial  {}", trim(&n.serial));
                kprintln!(
                    "  ns {}  {} blocks x {} B = {} MiB",
                    n.nsid,
                    n.block_count,
                    n.block_size,
                    n.capacity_bytes() / (1024 * 1024)
                );
                console::set_color(if crate::dev::nvme::writes_unlocked() { LTRED } else { LTGREEN });
                kprintln!(
                    "  writes  {}",
                    if crate::dev::nvme::writes_unlocked() { "UNLOCKED" } else { "locked (read-only)" }
                );
                console::set_color(WHITE);

                // A page-aligned DMA buffer, not a stack array. An unaligned
                // buffer can straddle a page boundary, and the tail of the
                // transfer then never arrives.
                let Some(dma) = crate::dev::nvme::alloc_dma(4096) else {
                    kprintln!("  could not allocate a DMA buffer");
                    return;
                };
                match n.read(0, 1, dma) {
                    Ok(()) => {
                        let buf = unsafe { core::slice::from_raw_parts(dma, 512) };
                        kprint!("  lba0   ");
                        for b in buf.iter().take(16) {
                            kprint!("{:02x} ", b);
                        }
                        kprintln!();
                        let sig = u16::from_le_bytes([buf[510], buf[511]]);
                        if sig == 0xAA55 {
                            console::set_color(LTGREEN);
                            kprintln!("  boot signature 0xAA55 present -- a real partition table");
                        } else {
                            kprintln!("  no 0xAA55 signature (sig {:#06x})", sig);
                        }
                        console::set_color(WHITE);
                    }
                    Err(e) => {
                        console::set_color(LTRED);
                        kprintln!("  read failed: status {:#x}", e);
                        console::set_color(WHITE);
                    }
                }
            });
        }
        "disk" => {
            if !crate::dev::nvme::present() {
                console::set_color(LTRED);
                kprintln!("  no NVMe controller initialised -- run 'nvme' first");
                console::set_color(WHITE);
                return;
            }
            match crate::store::block::scan() {
                Ok(layout) => {
                    let bs = crate::store::block::block_size() as u64;
                    let total = crate::store::block::block_count();
                    console::set_color(YELLOW);
                    kprintln!("[disk]  scheme {:?}", layout.scheme);
                    console::set_color(WHITE);
                    kprintln!("  {} blocks x {} B = {} MiB total", total, bs, total * bs / (1024 * 1024));
                    if layout.partitions.is_empty() {
                        kprintln!("  no partitions");
                    }
                    for p in &layout.partitions {
                        kprintln!(
                            "  {}: lba {:>12}..{:<12} {:>8} MiB  {}",
                            p.index,
                            p.start_lba,
                            p.end_lba(),
                            p.block_count * bs / (1024 * 1024),
                            p.kind()
                        );
                    }
                    let used = layout.highest_used_lba();
                    let free = total.saturating_sub(used);
                    console::set_color(LTCYAN);
                    kprintln!(
                        "  highest claimed lba {}, {} MiB beyond it",
                        used,
                        free * bs / (1024 * 1024)
                    );
                    console::set_color(WHITE);
                    // The whole point of this command: say plainly what would
                    // be destroyed, before anything can write.
                    console::set_color(LTRED);
                    kprintln!("  every range above is IN USE -- writing to it destroys data");
                    console::set_color(WHITE);
                }
                Err(e) => {
                    console::set_color(LTRED);
                    kprintln!("  scan failed: {:?}", e);
                    console::set_color(WHITE);
                }
            }
        }
        "store" => store_cmd(rest),
        "recovery" => {
            // Available from the shell too, but the boot-time path is the one
            // that matters -- this is a convenience, not the design.
            let _ = crate::recovery::console();
        }
        "cpu" => {
            let max_ext = crate::cpu::cpuid(0x8000_0000, 0)[0];
            if max_ext >= 0x8000_0004 {
                // The brand string is 48 bytes spread across three leaves,
                // each returning it in eax/ebx/ecx/edx little-endian.
                let mut brand = [0u8; 48];
                for (i, leaf) in [0x8000_0002u32, 0x8000_0003, 0x8000_0004].iter().enumerate() {
                    let r = crate::cpu::cpuid(*leaf, 0);
                    for (j, v) in r.iter().enumerate() {
                        brand[i * 16 + j * 4..i * 16 + j * 4 + 4]
                            .copy_from_slice(&v.to_le_bytes());
                    }
                }
                let s = core::str::from_utf8(&brand).unwrap_or("?");
                kprintln!("  {}", s.trim_end_matches('\0').trim());
            }
            let v = crate::cpu::cpuid(0, 0);
            let mut vend = [0u8; 12];
            vend[0..4].copy_from_slice(&v[1].to_le_bytes());
            vend[4..8].copy_from_slice(&v[3].to_le_bytes());
            vend[8..12].copy_from_slice(&v[2].to_le_bytes());
            kprintln!("  vendor  {}", core::str::from_utf8(&vend).unwrap_or("?"));
            let f = crate::cpu::cpuid(1, 0);
            kprintln!("  family  {:#x}  apic id {}", f[0], f[1] >> 24);
            kprintln!("  lapic   {}", lapic::id());
            let s = crate::cpu::detected();
            kprintln!(
                "  simd    sse={} sse2={} sse4.1={} avx={} avx2={} fma={} f16c={} avx512f={}",
                s.sse as u8, s.sse2 as u8, s.sse41 as u8, s.avx as u8,
                s.avx2 as u8, s.fma as u8, s.f16c as u8, s.avx512f as u8
            );
            console::set_color(if s.avx_enabled { LTGREEN } else { LTRED });
            kprintln!("  avx state enabled by the OS: {}", s.avx_enabled);
            console::set_color(WHITE);
        }
        "reboot" => {
            console::set_color(YELLOW);
            kprintln!("  resetting...");
            crate::cpu::reboot();
        }
        "tensor" => {
            let ok = crate::ai::selftest();
            if ok {
                console::set_color(LTGREEN);
                kprintln!("  all tensor selftests passed");
            } else {
                console::set_color(LTRED);
                kprintln!("  TENSOR SELFTESTS FAILED");
            }
            console::set_color(WHITE);
        }
        "bench" => crate::ai::bench(),
        "model" => crate::ai::model_demo(),
        "video" => {
            kprintln!(
                "  {}x{}  stride {}  {:?}",
                boot.fb.width(),
                boot.fb.height(),
                boot.fb.stride(),
                boot.fb.format()
            );
        }
        "echo" => kprintln!("  {}", rest),
        "clear" => console::with(|c| c.clear()),
        "fault" => {
            console::set_color(LTRED);
            kprintln!("  this will halt the machine.");
            console::set_color(LTGRAY);
            idt::trigger_page_fault();
        }
        "words" => {
            console::set_color(YELLOW);
            kprintln!("builtins");
            console::set_color(WHITE);
            let mut col = 0;
            for w in lang::eval::BUILTINS {
                kprint!("  {:<10}", w);
                col += 1;
                if col % 5 == 0 {
                    kprintln!();
                }
            }
            if col % 5 != 0 {
                kprintln!();
            }
        }
        "vars" => {
            if interp.var_count() == 0 {
                kprintln!("  none yet");
            } else {
                for (k, v) in interp.vars() {
                    kprintln!("  {} = {}", k, v.render());
                }
            }
        }
        _ => {
            // Not a command, so it is code. This is the point of the whole
            // exercise: there is no boundary between using the machine and
            // programming it.
            match lang::eval_line(interp, line) {
                Ok(lang::Value::Nil) => {}
                Ok(v) => {
                    console::set_color(LTCYAN);
                    kprintln!("  {}", v.render());
                    console::set_color(WHITE);
                }
                Err(e) => {
                    console::set_color(LTRED);
                    kprintln!("  {}", e);
                    console::set_color(WHITE);
                }
            }
        }
    }
}

// --- composition --------------------------------------------------------
//
// `cmd > /path` sends what a command printed into the namespace, and
// `cmd | filter` runs it through a text filter. Both work by capturing the
// console rather than by changing any applet, which is what makes them
// available to every command at once -- including the ones that existed long
// before this did.
//
// Filters take text and give text. They are not applets: an applet acts on the
// namespace, and these only ever look at a string that has already been
// produced.

fn strip_ansi_free(s: &str) -> alloc::string::String {
    // Console colour is set through a side channel rather than escape codes,
    // so captured text is already plain. This exists so that stays true by
    // intent rather than by accident.
    alloc::string::String::from(s)
}

fn apply_filter(text: &str, spec: &str) -> alloc::string::String {
    let mut it = spec.trim().splitn(2, ' ');
    let verb = it.next().unwrap_or("");
    let arg = it.next().unwrap_or("").trim();
    let lines: Vec<&str> = text.lines().collect();

    let mut out = alloc::string::String::new();
    match verb {
        "grep" => {
            for l in lines.iter().filter(|l| l.contains(arg)) {
                out.push_str(l);
                out.push('\n');
            }
        }
        "head" => {
            let n: usize = arg.parse().unwrap_or(10);
            for l in lines.iter().take(n) {
                out.push_str(l);
                out.push('\n');
            }
        }
        "tail" => {
            let n: usize = arg.parse().unwrap_or(10);
            let skip = lines.len().saturating_sub(n);
            for l in lines.iter().skip(skip) {
                out.push_str(l);
                out.push('\n');
            }
        }
        "sort" => {
            let mut v = lines.clone();
            v.sort_unstable();
            for l in v {
                out.push_str(l);
                out.push('\n');
            }
        }
        "count" | "wc" => {
            let words = text.split_whitespace().count();
            out = alloc::format!(
                "  {} lines, {} words, {} bytes\n",
                lines.len(),
                words,
                text.len()
            );
        }
        other => {
            out = alloc::format!("  not a filter: {}\n", other);
        }
    }
    out
}

/// Split a command line on the *last* unquoted `|` or `>`.
///
/// Last rather than first, so `grep >` reads as a filter argument rather than
/// as a redirection -- the trailing operator is the one that applies to
/// everything before it.
fn split_pipeline(line: &str) -> Option<(&str, char, &str)> {
    let mut found: Option<(usize, char)> = None;
    for (i, c) in line.char_indices() {
        if c == '|' || c == '>' {
            found = Some((i, c));
        }
    }
    let (i, c) = found?;
    Some((line[..i].trim(), c, line[i + 1..].trim()))
}

/// Run `cmd`, capturing everything it prints.
fn capture(
    cmd: &str,
    boot: &BootInfo,
    acpi: &Option<Acpi>,
    interp: &mut lang::Interp,
) -> alloc::string::String {
    console::begin_capture();
    execute(cmd, boot, acpi, interp);
    console::end_capture().unwrap_or_default()
}

/// Handle a line containing `|` or `>`. Returns false if there was neither.
pub fn run_pipeline(
    line: &str,
    boot: &BootInfo,
    acpi: &Option<Acpi>,
    interp: &mut lang::Interp,
) -> bool {
    let Some((head, op, tail)) = split_pipeline(line) else {
        return false;
    };
    if head.is_empty() || tail.is_empty() {
        return false;
    }

    let text = strip_ansi_free(&capture(head, boot, acpi, interp));

    match op {
        '|' => {
            let out = apply_filter(&text, tail);
            kprint!("{}", out);
        }
        '>' => {
            if crate::sysbox::write_text(tail, &text) {
                kprintln!("  {} bytes -> {}", text.len(), tail);
            } else {
                kprintln!("  could not write {}", tail);
            }
        }
        _ => return false,
    }
    true
}

/// Browse a real filesystem on the disk.
///
/// The firmware could read files and no longer exists; this is the kernel
/// doing it for itself. Read-only on purpose -- a tool that can inspect a
/// broken disk is useful, one that can half-write it is worse than nothing.
fn fat_cmd(rest: &str) {
    use crate::store::{block, fat};

    let mut it = rest.split_whitespace();
    let verb = it.next().unwrap_or("");
    let arg = it.next().unwrap_or("");
    let arg2 = rest.splitn(3, ' ').nth(2).unwrap_or("").trim();

    // Which partition. Named by index from `disk`, or "auto" for the first one
    // that actually parses as FAT -- which is the useful default, since the
    // partition type byte is a claim and mounting is a check.
    let layout = match block::scan() {
        Ok(l) => l,
        Err(e) => {
            kprintln!("  cannot read the partition table: {:?}", e);
            return;
        }
    };

    let pick_auto = || -> Option<(u32, fat::Volume)> {
        for p in layout.partitions.iter() {
            if let Ok(v) = fat::Volume::mount(p.start_lba) {
                return Some((p.index, v));
            }
        }
        None
    };

    match verb {
        "" | "list" if arg.is_empty() => {
            console::set_color(YELLOW);
            kprintln!("[fat]");
            console::set_color(LTGRAY);
            let mut any = false;
            for p in layout.partitions.iter() {
                if let Ok(v) = fat::Volume::mount(p.start_lba) {
                    any = true;
                    kprintln!(
                        "  partition {}  {:?}  {} clusters of {} B  ({})",
                        p.index,
                        v.kind(),
                        v.total_clusters(),
                        v.cluster_bytes(),
                        p.kind()
                    );
                }
            }
            if !any {
                kprintln!("  no FAT filesystem found on this disk");
            } else {
                kprintln!("  'fat ls <path>', 'fat cat <path>', 'fat get <path> <namespace-path>'");
            }
        }
        "ls" | "cat" | "get" => {
            let Some((idx, vol)) = pick_auto() else {
                kprintln!("  no FAT filesystem found");
                return;
            };
            let path = if verb == "get" { arg } else { rest[verb.len()..].trim() };
            match vol.find(path) {
                Err(e) => kprintln!("  {}: {:?}", path, e),
                Ok(entry) => match verb {
                    "ls" => {
                        if !entry.is_dir {
                            kprintln!("  {:>10}  {}", entry.size, entry.name);
                            return;
                        }
                        match vol.list(entry.cluster) {
                            Err(e) => kprintln!("  {:?}", e),
                            Ok(items) => {
                                kprintln!("  partition {}, {} entries", idx, items.len());
                                for e in items {
                                    if e.is_dir {
                                        console::set_color(LTCYAN);
                                        kprintln!("  {:>10}  {}/", "", e.name);
                                    } else {
                                        console::set_color(WHITE);
                                        kprintln!("  {:>10}  {}", e.size, e.name);
                                    }
                                }
                                console::set_color(LTGRAY);
                            }
                        }
                    }
                    "cat" => match vol.read_file(&entry) {
                        Err(e) => kprintln!("  {:?}", e),
                        Ok(bytes) => {
                            for line in bytes.split(|c| *c == b'\n') {
                                kprintln!(
                                    "  {}",
                                    core::str::from_utf8(line).unwrap_or("<binary>")
                                );
                            }
                        }
                    },
                    _ => match vol.read_file(&entry) {
                        Err(e) => kprintln!("  {:?}", e),
                        Ok(bytes) => {
                            let dest = if arg2.is_empty() { "/tmp/imported" } else { arg2 };
                            let n = bytes.len();
                            if crate::sysbox::write_blob(dest, bytes) {
                                kprintln!("  {} bytes -> {}", n, dest);
                            } else {
                                kprintln!("  could not write {}", dest);
                            }
                        }
                    },
                },
            }
        }
        other => kprintln!("  not a fat subcommand: {}", other),
    }
}
