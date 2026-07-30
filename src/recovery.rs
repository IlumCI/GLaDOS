//! Recovery console.
//!
//! The rule this file exists to obey: **a repair tool must not live inside the
//! thing it repairs.** Once state is persistent, "turn it off and on again"
//! stops working, because rebooting faithfully restores the corruption. So
//! this console is compiled into the boot image on the ESP, runs before any
//! persistent state is restored, and deliberately depends on as little as
//! possible -- no interpreter, no shell, no restored heap contents. Everything
//! it needs is the NVMe driver, the store, and the keyboard.
//!
//! It has its own line reader for the same reason. Reusing the shell's would
//! tie recovery to the interpreter, which is exactly the sort of dependency
//! that stops working on the day you need it.

use crate::dev::kbd;
use crate::gfx::console::{self, LTCYAN, LTGRAY, LTGREEN, LTRED, WHITE, YELLOW};
use crate::store::{self, cas, sha256};
use crate::{kprint, kprintln};
use alloc::string::String;

/// What the caller should do once the console exits.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Outcome {
    /// Continue booting and restore persistent state as normal.
    Continue,
    /// Continue booting but do not restore anything.
    ///
    /// The escape hatch for when the persistent state is itself the problem.
    SkipRestore,
}

/// Read one line, echoing to the console only.
fn read_line() -> String {
    let mut s = String::new();
    loop {
        match kbd::pop() {
            Some(b'\n') => {
                kprintln!();
                return s;
            }
            Some(8) => {
                if s.pop().is_some() {
                    console::with(|c| c.put_char(8));
                }
            }
            Some(ch) if (0x20..0x7F).contains(&ch) => {
                s.push(ch as char);
                console::with(|c| c.put_char(ch));
            }
            Some(_) => {}
            None => unsafe { core::arch::asm!("hlt", options(nomem, nostack)) },
        }
    }
}

fn hex8(h: &[u8; 32]) -> String {
    let b = sha256::short_hex(h);
    String::from(core::str::from_utf8(&b).unwrap_or("????????????????"))
}

/// Poll briefly for a key at boot.
///
/// Deliberately a fixed short window rather than a prompt that waits for
/// input: an unattended machine has to boot on its own, and a recovery console
/// that blocks the boot is its own kind of failure.
pub fn key_held(ticks: u64) -> bool {
    let start = crate::dev::lapic::ticks();
    while crate::dev::lapic::ticks() - start < ticks {
        if let Some(k) = kbd::pop() {
            // ESC or 'r'
            if k == 27 || k == b'r' || k == b'R' {
                return true;
            }
        }
        core::hint::spin_loop();
    }
    false
}

fn banner() {
    console::set_color(LTCYAN);
    kprintln!("\n=== GLaDOS recovery console ===");
    console::set_color(WHITE);
    kprintln!("running from the boot image; no persistent state has been restored");
    help();
}

fn help() {
    console::set_color(YELLOW);
    kprintln!("\ncommands");
    console::set_color(WHITE);
    kprintln!("  l          list checkpoints, newest first");
    kprintln!("  v          verify every chunk in every checkpoint");
    kprintln!("  i          store and superblock detail");
    kprintln!("  b          roll back one checkpoint");
    kprintln!("  g <seq>    roll back to a specific sequence number");
    kprintln!("  c          continue booting and restore state");
    kprintln!("  s          continue booting WITHOUT restoring state");
    kprintln!("  ?          this list");
}

fn list() {
    if !store::mounted() {
        kprintln!("  no store mounted");
        return;
    }
    store::with(|s| {
        let mut r = s.sb.root;
        if r.is_none() {
            kprintln!("  no checkpoints");
            return;
        }
        console::set_color(YELLOW);
        kprintln!("  seq    root              entries  lba");
        console::set_color(WHITE);
        let mut n = 0;
        while !r.is_none() && n < 64 {
            match s.read_manifest(&r) {
                Ok(m) => {
                    kprintln!(
                        "  {:<6} {}  {:<7}  {}",
                        m.seq,
                        hex8(&r.hash),
                        m.entries.len(),
                        r.lba
                    );
                    r = m.prev;
                }
                Err(e) => {
                    console::set_color(LTRED);
                    kprintln!("  chain breaks here: {:?}", e);
                    console::set_color(WHITE);
                    break;
                }
            }
            n += 1;
        }
    });
}

/// Walk every checkpoint and re-read every chunk, checking each against its
/// own content hash.
///
/// This is the command that answers "is anything actually damaged", which is
/// the first thing worth knowing and the hardest to guess at.
fn verify() {
    if !store::mounted() {
        kprintln!("  no store mounted");
        return;
    }
    store::with(|s| {
        let mut r = s.sb.root;
        let mut checkpoints = 0u32;
        let mut chunks = 0u32;
        let mut bad = 0u32;

        while !r.is_none() && checkpoints < 64 {
            let m = match s.read_manifest(&r) {
                Ok(m) => m,
                Err(e) => {
                    console::set_color(LTRED);
                    kprintln!("  manifest at lba {} unreadable: {:?}", r.lba, e);
                    console::set_color(WHITE);
                    bad += 1;
                    break;
                }
            };
            for e in &m.entries {
                match s.get(&e.chunk) {
                    Ok(_) => chunks += 1,
                    Err(err) => {
                        console::set_color(LTRED);
                        kprintln!(
                            "  seq {} chunk at lba {} FAILED: {:?}",
                            m.seq, e.chunk.lba, err
                        );
                        console::set_color(WHITE);
                        bad += 1;
                    }
                }
            }
            checkpoints += 1;
            r = m.prev;
        }

        console::set_color(if bad == 0 { LTGREEN } else { LTRED });
        kprintln!(
            "  {} checkpoints, {} chunks verified, {} failures",
            checkpoints, chunks, bad
        );
        console::set_color(WHITE);
        if bad > 0 {
            kprintln!("  roll back to a checkpoint older than the first failure");
        }
    });
}

fn info() {
    if !store::mounted() {
        kprintln!("  no store mounted");
        return;
    }
    store::with(|s| {
        kprintln!("  region     lba {}..{}", s.sb.region_start, s.sb.region_start + s.sb.region_blocks);
        kprintln!("  sequence   {}", s.sb.seq);
        kprintln!("  commits    {}", s.sb.checkpoints);
        kprintln!("  next free  lba {}  ({} blocks left)", s.sb.alloc_next, s.free_blocks());
        if s.sb.root.is_none() {
            kprintln!("  root       none");
        } else {
            kprintln!("  root       {} at lba {}", hex8(&s.sb.root.hash), s.sb.root.lba);
        }
    });
}

fn back() {
    store::with(|s| match s.read_manifest(&s.sb.root) {
        Ok(m) => {
            if m.prev.is_none() {
                kprintln!("  already at the oldest checkpoint");
                return;
            }
            match s.rollback_to(m.prev) {
                Ok(()) => {
                    console::set_color(LTGREEN);
                    kprintln!("  rolled back; superblock now at seq {}", s.sb.seq);
                    console::set_color(WHITE);
                    kprintln!("  nothing was erased -- the newer checkpoint is still on disk");
                }
                Err(e) => kprintln!("  failed: {:?}", e),
            }
        }
        Err(e) => kprintln!("  current manifest unreadable: {:?}", e),
    });
}

fn goto(arg: &str) {
    let Ok(want) = arg.trim().parse::<u64>() else {
        kprintln!("  usage: g <seq>");
        return;
    };
    store::with(|s| {
        let mut r = s.sb.root;
        let mut hops = 0;
        while !r.is_none() && hops < 64 {
            match s.read_manifest(&r) {
                Ok(m) => {
                    if m.seq == want {
                        match s.rollback_to(r) {
                            Ok(()) => {
                                console::set_color(LTGREEN);
                                kprintln!("  now at checkpoint seq {}", want);
                                console::set_color(WHITE);
                            }
                            Err(e) => kprintln!("  failed: {:?}", e),
                        }
                        return;
                    }
                    r = m.prev;
                }
                Err(e) => {
                    kprintln!("  chain broken before seq {}: {:?}", want, e);
                    return;
                }
            }
            hops += 1;
        }
        kprintln!("  no checkpoint with seq {} in the reachable chain", want);
    });
}

/// The console loop. Returns what boot should do next.
pub fn console() -> Outcome {
    banner();
    loop {
        console::set_color(LTCYAN);
        kprint!("\nrecovery> ");
        console::set_color(WHITE);
        let line = read_line();
        let line = line.trim();
        let (cmd, arg) = match line.split_once(' ') {
            Some((a, b)) => (a, b),
            None => (line, ""),
        };

        match cmd {
            "" => {}
            "l" | "list" => list(),
            "v" | "verify" => verify(),
            "i" | "info" => info(),
            "b" | "back" => back(),
            "g" | "goto" => goto(arg),
            "c" | "continue" => return Outcome::Continue,
            "s" | "safe" => {
                console::set_color(YELLOW);
                kprintln!("  continuing without restoring persistent state");
                console::set_color(WHITE);
                return Outcome::SkipRestore;
            }
            "?" | "help" => help(),
            other => kprintln!("  unknown: {} (try ?)", other),
        }
    }
}

/// Called during boot, after hardware is up and before state is restored.
///
/// Enters the console if the store looks damaged, or if a key is held. The
/// first condition matters more than the second: the case worth designing for
/// is the one where the machine cannot boot far enough for anyone to ask.
pub fn maybe_enter(store_damaged: bool) -> Outcome {
    if store_damaged {
        console::set_color(LTRED);
        kprintln!("\n[recovery] the checkpoint store did not verify");
        console::set_color(WHITE);
        return console();
    }

    console::set_color(LTGRAY);
    kprintln!("\n[boot] hold ESC or R for the recovery console...");
    console::set_color(WHITE);
    if key_held(crate::TIMER_HZ as u64 / 2) {
        return console();
    }
    Outcome::Continue
}
