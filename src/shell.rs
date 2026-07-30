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
        let key = if let Some(k) = kbd::pop() {
            k
        } else if let Some(b) = crate::serial::read_byte() {
            // A terminal speaks a slightly different dialect than the i8042
            // driver: Enter arrives as CR, and Backspace as DEL. Translate here
            // rather than in `read_byte`, because the keyboard's own DELETE key
            // is a different key that happens to share the 0x7F code.
            match b {
                b'\r' => b'\n',
                0x7F => 8,
                other => other,
            }
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
                execute(trimmed, boot, acpi, &mut interp);

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
        "help" => {
            console::set_color(YELLOW);
            kprintln!("commands");
            console::set_color(WHITE);
            kprintln!("  help          this list");
            kprintln!("  mem           heap usage");
            kprintln!("  uptime        ticks since boot");
            kprintln!("  acpi          parsed firmware tables");
            kprintln!("  tasks         scheduler state");
            kprintln!("  pci           enumerate PCIe devices");
            kprintln!("  cpu           processor identification");
            kprintln!("  reboot        reset the machine");
            kprintln!("  video         framebuffer geometry");
            kprintln!("  echo <text>   print text");
            kprintln!("  clear         clear the screen");
            kprintln!("  words         list language builtins");
            kprintln!("  vars          list variables you have defined");
            kprintln!("  fault         deliberately dereference null");
            kprintln!("  typewriter    output pacing, in us per character");
            kprintln!("  refresh       repaint the console");
            console::set_color(YELLOW);
            kprintln!("\nsysbox owns the namespace -- type 'sysbox' for its applets");
            console::set_color(WHITE);
            console::set_color(YELLOW);
            kprintln!("\nanything else is evaluated as code");
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
