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
use crate::mem;
use crate::BootInfo;
use crate::{kprint, kprintln};
use alloc::string::String;

fn prompt() {
    console::set_color(LTGREEN);
    kprint!("\nsanctum> ");
    console::set_color(WHITE);
}

pub fn run(boot: &BootInfo, acpi: &Option<Acpi>) -> ! {
    console::set_color(LTCYAN);
    kprintln!("\ninteractive. type 'help'.");
    console::set_color(WHITE);

    let mut line = String::new();
    prompt();

    loop {
        match kbd::pop() {
            Some(b'\n') => {
                kprintln!();
                execute(line.trim(), boot, acpi);
                line.clear();
                prompt();
            }
            Some(8) => {
                // Only echo the erase if there was something to erase.
                if line.pop().is_some() {
                    kprint!("\u{8}");
                }
            }
            Some(ch) if (0x20..0x7F).contains(&ch) => {
                line.push(ch as char);
                kprint!("{}", ch as char);
            }
            Some(_) => {}
            None => {
                // Nothing queued: idle until the next interrupt rather than
                // spinning. The timer alone wakes us 100 times a second.
                unsafe { core::arch::asm!("hlt", options(nomem, nostack)) };
            }
        }
    }
}

fn execute(line: &str, boot: &BootInfo, acpi: &Option<Acpi>) {
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
            kprintln!("  video         framebuffer geometry");
            kprintln!("  echo <text>   print text");
            kprintln!("  clear         clear the screen");
            kprintln!("  fault         deliberately dereference null");
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
        }
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
        other => {
            console::set_color(LTRED);
            kprintln!("  unknown command: {}", other);
            console::set_color(WHITE);
        }
    }
}
