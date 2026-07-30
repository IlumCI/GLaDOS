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
        let Some(key) = kbd::pop() else {
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

fn execute(line: &str, boot: &BootInfo, acpi: &Option<Acpi>, interp: &mut lang::Interp) {
    let mut parts = line.splitn(2, ' ');
    let cmd = parts.next().unwrap_or("");
    let rest = parts.next().unwrap_or("").trim();

    match cmd {
        "" => {}
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
