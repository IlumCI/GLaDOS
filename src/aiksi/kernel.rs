//! Aiksi's view of the kernel.
//!
//! GLaDOS is written in Rust and Aiksi is how anything else reaches it. That
//! makes this file the translation layer, and it has one rule:
//!
//! > **A builtin is named after the Rust path it calls, flattened.**
//! > `crate::net::tcp::connect` is `tcp_connect`. `crate::dev::rtc::now` is
//! > `rtc_now`. `crate::crypto::sha256::hash` is `sha256`.
//!
//! A rule and not a taste. The audience for this language is a 0.6B model and
//! whoever is reading the kernel source beside it, and both can apply a rule
//! they were told once to a subsystem they have never seen. A hand-picked name
//! per builtin reads better in isolation and has to be memorised one at a time,
//! which is the cost that actually matters here. Where the rule produces
//! something unbearable the rule still wins, because a single exception means
//! every name has to be checked against the list again.
//!
//! ### Why this is a separate file
//!
//! `eval.rs` owns the gate, the arity check and the table. This owns the arms.
//! The split is so that reading "what may a program do" is one screen rather
//! than a scroll through six hundred lines of subsystem plumbing -- and so that
//! adding a subsystem cannot accidentally edit the gate.
//!
//! Anything in `eval::BUILTINS` that `eval` does not handle itself arrives
//! here. A name that reaches this function without an arm is a row with no
//! implementation, which the boot selftest catches as "unknown".
//!
//! ### The shape of a kernel call
//!
//! Values in, values out, and errors are strings. Nothing here panics on bad
//! input: a program written by a model will pass a string where a number
//! belongs, and the answer to that is a message naming the builtin, not a
//! triple fault in ring 0.

use super::eval::{Interp, Value};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Dispatch a builtin that reaches a kernel subsystem.
///
/// The gate and the arity check have already run, so an arm here can trust its
/// argument count and may assume its capability was allowed. A name that
/// reaches the final arm is a table row nobody implemented.
pub fn call(it: &mut Interp, name: &str, args: &[Value]) -> Result<Value, String> {
    match name {
        // --- strings ---------------------------------------------------
        //
        // Not a kernel subsystem, and the reason they are here anyway: a
        // systems language whose only string operation is `+` cannot parse a
        // config file, split a response header or build a path. Every one of
        // these existed as a workaround written in Aiksi itself, badly, in the
        // skeletons.
        "upper" => Ok(Value::Str(text(args, 0).to_uppercase())),
        "lower" => Ok(Value::Str(text(args, 0).to_lowercase())),
        "trim" => Ok(Value::Str(text(args, 0).trim().to_string())),
        "split" => {
            let (s, sep) = (text(args, 0), text(args, 1));
            // An empty separator splits into characters, which is what every
            // caller that passes one actually wants and is otherwise an
            // infinite sequence of empty strings.
            let parts: Vec<Value> = if sep.is_empty() {
                s.chars().map(|c| Value::Str(c.to_string())).collect()
            } else {
                s.split(sep.as_str()).map(|p| Value::Str(p.to_string())).collect()
            };
            Ok(Value::List(parts))
        }
        "join" => {
            let sep = text(args, 1);
            match &args[0] {
                Value::List(v) => {
                    let parts: Vec<String> = v.iter().map(|x| x.render()).collect();
                    Ok(Value::Str(parts.join(&sep)))
                }
                other => Ok(Value::Str(other.render())),
            }
        }
        "substr" => {
            let s = text(args, 0);
            let start = int(args, 1)?.max(0) as usize;
            // Length, not end index. Both conventions are defensible and this
            // one composes with `find`, which answers a position.
            let n = int(args, 2)?.max(0) as usize;
            Ok(Value::Str(s.chars().skip(start).take(n).collect()))
        }
        "find" => {
            let (s, pat) = (text(args, 0), text(args, 1));
            // Character index, not byte offset, so it can be handed straight
            // back to `substr` and `get`. -1 for absent: a program testing the
            // result should not have to know about Nil to do it.
            Ok(Value::Int(match s.find(pat.as_str()) {
                Some(b) => s[..b].chars().count() as i64,
                None => -1,
            }))
        }
        "replace" => {
            let (s, from, to) = (text(args, 0), text(args, 1), text(args, 2));
            if from.is_empty() {
                return Ok(Value::Str(s));
            }
            Ok(Value::Str(s.replace(from.as_str(), &to)))
        }
        "starts" => Ok(Value::Int(text(args, 0).starts_with(text(args, 1).as_str()) as i64)),
        "ends" => Ok(Value::Int(text(args, 0).ends_with(text(args, 1).as_str()) as i64)),
        "contains" => match &args[0] {
            Value::List(v) => {
                let want = args[1].render();
                Ok(Value::Int(v.iter().any(|x| x.render() == want) as i64))
            }
            other => Ok(Value::Int(other.render().contains(text(args, 1).as_str()) as i64)),
        },
        "chr" => {
            let c = int(args, 0)?;
            Ok(Value::Str(
                char::from_u32(c.clamp(0, 0x10FFFF) as u32)
                    .map(|c| c.to_string())
                    .unwrap_or_default(),
            ))
        }
        "ord" => Ok(Value::Int(text(args, 0).chars().next().map(|c| c as i64).unwrap_or(-1))),
        "repeat" => {
            let s = text(args, 0);
            // Bounded, because `repeat("x", 1000000000)` in a repaint path is
            // an allocation failure in a kernel with no OOM killer and one
            // address space. The step budget does not see a single call.
            let n = int(args, 1)?.clamp(0, 65_536) as usize;
            Ok(Value::Str(s.repeat(n)))
        }
        "pad" => {
            let s = text(args, 0);
            let w = int(args, 1)?.clamp(0, 4_096) as usize;
            let have = s.chars().count();
            Ok(Value::Str(if have >= w {
                s
            } else {
                let mut out = s;
                for _ in have..w {
                    out.push(' ');
                }
                out
            }))
        }

        // --- numbers ---------------------------------------------------
        "abs" => Ok(Value::Int(int(args, 0)?.saturating_abs())),
        "min" => Ok(Value::Int(int(args, 0)?.min(int(args, 1)?))),
        "max" => Ok(Value::Int(int(args, 0)?.max(int(args, 1)?))),
        "clamp" => Ok(Value::Int(int(args, 0)?.clamp(int(args, 1)?, int(args, 2)?))),
        // Integer square root, by the same Newton iteration `gfx` uses for
        // circles. There are no floats in this language and adding them for
        // one builtin would change every arithmetic path.
        "sqrt" => {
            let n = int(args, 0)?;
            Ok(Value::Int(if n <= 0 { 0 } else { isqrt(n as u64) as i64 }))
        }
        "pow" => {
            let (b, e) = (int(args, 0)?, int(args, 1)?);
            let mut acc: i64 = 1;
            for _ in 0..e.clamp(0, 62) {
                acc = acc.saturating_mul(b);
            }
            Ok(Value::Int(acc))
        }

        // --- lists -----------------------------------------------------
        "sort" => match &args[0] {
            Value::List(v) => {
                let mut out = v.clone();
                // Numbers numerically, anything else by its rendering. A list
                // of mixed kinds sorts stably rather than refusing: a program
                // sorting rows it read from a file should not have to prove
                // they are homogeneous first.
                out.sort_by(|a, b| match (a.as_int(), b.as_int()) {
                    (Ok(x), Ok(y)) => x.cmp(&y),
                    _ => a.render().cmp(&b.render()),
                });
                Ok(Value::List(out))
            }
            _ => Err("sort wants a list".to_string()),
        },
        "reverse" => match &args[0] {
            Value::List(v) => {
                let mut out = v.clone();
                out.reverse();
                Ok(Value::List(out))
            }
            other => Ok(Value::Str(other.render().chars().rev().collect())),
        },
        "slice" => match &args[0] {
            Value::List(v) => {
                let start = int(args, 1)?.max(0) as usize;
                let n = int(args, 2)?.max(0) as usize;
                Ok(Value::List(v.iter().skip(start).take(n).cloned().collect()))
            }
            _ => Err("slice wants a list".to_string()),
        },
        "index" => match &args[0] {
            Value::List(v) => {
                let want = args[1].render();
                Ok(Value::Int(
                    v.iter().position(|x| x.render() == want).map(|i| i as i64).unwrap_or(-1),
                ))
            }
            _ => Err("index wants a list".to_string()),
        },
        "remove" => match &args[0] {
            Value::List(v) => {
                let i = int(args, 1)?;
                let mut out = v.clone();
                if i >= 0 && (i as usize) < out.len() {
                    out.remove(i as usize);
                }
                Ok(Value::List(out))
            }
            _ => Err("remove wants a list".to_string()),
        },
        "range" => {
            let (from, to) = (int(args, 0)?, int(args, 1)?);
            // Bounded for the reason `repeat` is: this is the easiest way for
            // a generated program to ask for a billion-element list.
            let n = (to - from).clamp(0, 65_536);
            Ok(Value::List((0..n).map(|i| Value::Int(from + i)).collect()))
        }

        // --- crate::time, crate::dev::rtc, crate::dev::lapic ------------
        "rtc_now" => {
            // The clock can be unreadable -- no RTC, or a read that never
            // settled -- and an empty string is the honest answer for that.
            // Inventing an epoch date would be a timestamp a program would
            // then store.
            let Some(t) = crate::dev::rtc::now() else {
                return Ok(Value::Str(String::new()));
            };
            Ok(Value::Str(format!(
                "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
                t.year, t.month, t.day, t.hour, t.minute, t.second
            )))
        }
        "rtc_unix" => Ok(Value::Int(
            crate::dev::rtc::now().map(|t| crate::dev::rtc::unix_seconds(&t) as i64).unwrap_or(-1),
        )),
        // Seconds, from the timer-interrupt count at TIMER_HZ. Not
        // `lapic::timer_hz`, which is the calibrated APIC frequency in the
        // millions and put every reading at 0s the last time the two were
        // confused.
        "uptime" => Ok(Value::Int(
            (crate::dev::lapic::ticks() / crate::TIMER_HZ as u64) as i64,
        )),
        "tsc" => Ok(Value::Int(crate::time::rdtsc() as i64)),
        "tsc_mhz" => Ok(Value::Int(crate::time::tsc_mhz() as i64)),

        // --- crate::task -----------------------------------------------
        "task_count" => Ok(Value::Int(crate::task::count() as i64)),
        "task_current" => Ok(Value::Int(crate::task::current() as i64)),
        "task_switches" => Ok(Value::Int(crate::task::total_switches() as i64)),
        "task_yield" => {
            crate::task::yield_now();
            Ok(Value::Nil)
        }

        // --- crate::mem ------------------------------------------------
        "mem_used" => Ok(Value::Int(crate::mem::heap::HEAP.stats().0 as i64)),
        "mem_total" => Ok(Value::Int(crate::mem::heap::HEAP.stats().1 as i64)),

        // --- crate::dev::pci -------------------------------------------
        //
        // The hardware inventory, as text, one device per line. A list of
        // structured values would be better and this language has no record
        // type; `split` over the lines is what a program does instead, and
        // saying so is better than pretending the shape is richer than it is.
        "pci_list" => {
            let mut out = String::new();
            // `scan` is a visitor rather than an iterator, because it walks
            // config space and there is nowhere to hold the result but the
            // callback. ECAM comes from `net`, which is the module that found
            // it; a machine with no ECAM lists nothing rather than guessing at
            // the legacy ports.
            if let Some(ecam) = crate::net::ecam() {
                crate::dev::pci::scan(ecam, 255, |d| {
                    out.push_str(&format!(
                        "{:02x}:{:02x}.{} {:04x}:{:04x} {}
",
                        d.bus,
                        d.dev,
                        d.func,
                        d.vendor,
                        d.device,
                        crate::dev::pci::class_name(d.class, d.subclass)
                    ));
                });
            }
            Ok(Value::Str(out.trim_end().to_string()))
        }

        // --- crate::net ------------------------------------------------
        "net_ready" => Ok(Value::Int(crate::net::ready() as i64)),
        "net_ifaces" => Ok(Value::List(
            crate::net::ifaces()
                .iter()
                .filter(|i| i.nic.is_some())
                .map(|i| Value::Str(i.name.to_string()))
                .collect(),
        )),
        "net_ip" => Ok(Value::Str(fmt_ip(crate::net::config().ip))),
        "net_gateway" => Ok(Value::Str(fmt_ip(crate::net::config().gateway))),
        "net_dns" => Ok(Value::Str(fmt_ip(crate::net::config().dns))),

        // --- the operator's namespace, beyond read/write ----------------
        // Hex, because that is how every other part of this system names a
        // node -- the ledgers, `app info`, the blob directory -- and a program
        // comparing a hash against one it read from a file has to see the same
        // spelling.
        "hash_of" => Ok(Value::Str(match crate::sysbox::hash_of(&text(args, 0)) {
            Some(h) => {
                let mut out = String::new();
                for b in h {
                    out.push_str(&format!("{:02x}", b));
                }
                out
            }
            None => String::new(),
        })),
        "size" => Ok(Value::Int(
            crate::sysbox::read_blob(&text(args, 0)).map(|b| b.len() as i64).unwrap_or(-1),
        )),
        "is_dir" => Ok(Value::Int(crate::sysbox::is_dir(&text(args, 0)) as i64)),
        "rm" => {
            let path = text(args, 0);
            if !it.may_write_pub(&path) {
                return Err(format!("a stored program may not remove {}", path));
            }
            Ok(Value::Int(crate::sysbox::detach(&path) as i64))
        }

        // --- encodings --------------------------------------------------
        "hexenc" => {
            let mut out = String::new();
            for b in text(args, 0).as_bytes() {
                out.push_str(&format!("{:02x}", b));
            }
            Ok(Value::Str(out))
        }
        "hexdec" => {
            let s = text(args, 0);
            let bytes: Vec<u8> = s.as_bytes().to_vec();
            let mut out = Vec::new();
            let mut i = 0;
            while i + 1 < bytes.len() {
                let hi = (bytes[i] as char).to_digit(16);
                let lo = (bytes[i + 1] as char).to_digit(16);
                match (hi, lo) {
                    (Some(h), Some(l)) => out.push((h * 16 + l) as u8),
                    _ => return Err(("hexdec: not hex".to_string())),
                }
                i += 2;
            }
            Ok(Value::Str(String::from_utf8_lossy(&out).into_owned()))
        }

        other => Err(format!("'{}' is in the table with no implementation", other)),
    }
}

fn text(args: &[Value], i: usize) -> String {
    args.get(i).map(|v| v.render()).unwrap_or_default()
}

fn int(args: &[Value], i: usize) -> Result<i64, String> {
    args.get(i).ok_or_else(|| "missing argument".to_string())?.as_int()
}

fn fmt_ip(ip: [u8; 4]) -> String {
    format!("{}.{}.{}.{}", ip[0], ip[1], ip[2], ip[3])
}

/// Integer square root by Newton's method.
fn isqrt(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}
