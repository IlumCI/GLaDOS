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
        // A `Time` record. It answered only the formatted stamp, so a program
        // wanting the hour was reduced to `substr(rtc_now(), 11, 2)` -- the
        // exact fragile re-parse a record exists to remove. The stamp survives
        // as `.text`, so nothing that displayed the time lost anything.
        "rtc_now" => {
            // The clock can be unreadable -- no RTC, or a read that never
            // settled -- and nothing is the honest answer for that. Inventing
            // an epoch date would be a timestamp a program would then store.
            let Some(d) = crate::dev::rtc::now() else {
                return Ok(Value::Nil);
            };
            rec(
                "Time",
                alloc::vec![
                    i(d.year as i64),
                    i(d.month as i64),
                    i(d.day as i64),
                    i(d.hour as i64),
                    i(d.minute as i64),
                    i(d.second as i64),
                    Value::Str(format!(
                        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
                        d.year, d.month, d.day, d.hour, d.minute, d.second
                    )),
                ],
            )
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
        // A list of `Device` records, not lines of text.
        //
        // It answered text until records existed, and every caller then wrote
        // the same fragile `split` to take the fields back apart -- which is
        // the thing a record is for. The shape is `eval::KERNEL_RECS`, so an
        // annotation checks against it.
        //
        // `scan` is a visitor rather than an iterator, because it walks config
        // space and there is nowhere to hold the result but the callback. ECAM
        // comes from `net`, which is the module that found it; a machine with
        // no ECAM lists nothing rather than guessing at the legacy ports.
        "pci_list" => {
            let mut out = Vec::new();
            if let Some(ecam) = crate::net::ecam() {
                crate::dev::pci::scan(ecam, 255, |d| {
                    out.push(Value::Rec(
                        String::from("Device"),
                        alloc::vec![
                            (String::from("bus"), Value::Int(d.bus as i64)),
                            (String::from("dev"), Value::Int(d.dev as i64)),
                            (String::from("func"), Value::Int(d.func as i64)),
                            (String::from("vendor"), Value::Int(d.vendor as i64)),
                            (String::from("device"), Value::Int(d.device as i64)),
                            (
                                String::from("class"),
                                Value::Str(
                                    crate::dev::pci::class_name(d.class, d.subclass).to_string(),
                                ),
                            ),
                        ],
                    ));
                });
            }
            Ok(Value::List(out))
        }

        // --- crate::net ------------------------------------------------
        "net_ready" => Ok(Value::Int(crate::net::ready() as i64)),
        // Was a list of names, which threw away everything else an interface
        // knows: its address, whether it is up, what it has carried. Not a
        // struct flattened into text -- a struct discarded -- which makes it
        // the clearest case in the sweep.
        "net_ifaces" => {
            let mut out = alloc::vec::Vec::new();
            for f in crate::net::ifaces().iter().filter(|f| f.nic.is_some()) {
                out.push(rec(
                    "Iface",
                    alloc::vec![
                        t(f.name),
                        Value::Str(fmt_ip(f.ip)),
                        Value::Str(fmt_ip(f.netmask)),
                        Value::Str(fmt_ip(f.gateway)),
                        Value::Str(fmt_ip(f.dns)),
                        i(f.up as i64),
                        i(f.stats.rx_packets as i64),
                        i(f.stats.rx_bytes as i64),
                        i(f.stats.tx_packets as i64),
                        i(f.stats.tx_bytes as i64),
                        i(f.stats.tx_dropped as i64),
                    ],
                )?);
            }
            Ok(Value::List(out))
        }
        "net_config" => {
            let c = crate::net::config();
            rec(
                "Config",
                alloc::vec![
                    Value::Str(fmt_ip(c.ip)),
                    Value::Str(fmt_ip(c.gateway)),
                    Value::Str(fmt_ip(c.netmask)),
                    Value::Str(fmt_ip(c.dns)),
                ],
            )
        }
        "mem_stats" => {
            let (used, total) = crate::mem::heap::HEAP.stats();
            rec("Mem", alloc::vec![i(used as i64), i(total as i64)])
        }
        "task_list" => {
            let here = crate::task::current();
            let mut out = alloc::vec::Vec::new();
            for n in 0..crate::task::count() {
                let Some(task) = crate::task::snapshot(n) else {
                    continue;
                };
                out.push(rec(
                    "Task",
                    alloc::vec![
                        i(n as i64),
                        t(task.name),
                        t(match task.state {
                            crate::task::State::Ready => "ready",
                            crate::task::State::Unused => "unused",
                        }),
                        i(task.switches as i64),
                        i((n == here) as i64),
                    ],
                )?);
            }
            Ok(Value::List(out))
        }
        "stat" => {
            let path = text(args, 0);
            let dir = crate::sysbox::is_dir(&path);
            // A miss is nothing, not a Stat full of zeroes. A record whose
            // fields all read as "absent" is indistinguishable from an empty
            // file, and a program checking existence would have to know which
            // field to trust.
            let blob = crate::sysbox::read_blob(&path);
            if !dir && blob.is_none() {
                return Ok(Value::Nil);
            }
            let hash = match crate::sysbox::hash_of(&path) {
                Some(h) => {
                    let mut out = String::new();
                    for b in h {
                        out.push_str(&format!("{:02x}", b));
                    }
                    out
                }
                None => String::new(),
            };
            rec(
                "Stat",
                alloc::vec![
                    Value::Str(
                        path.rsplit('/').next().unwrap_or(&path).to_string()
                    ),
                    Value::Str(hash),
                    i(blob.map(|b| b.len() as i64).unwrap_or(0)),
                    i(dir as i64),
                ],
            )
        }
        "net_ip" => Ok(Value::Str(fmt_ip(crate::net::config().ip))),
        "net_gateway" => Ok(Value::Str(fmt_ip(crate::net::config().gateway))),
        "net_dns" => Ok(Value::Str(fmt_ip(crate::net::config().dns))),

        // --- crate::net::dns, ::tcp, ::udp, ::tls -----------------------
        //
        // Sockets, and the thing to know before writing anything against them:
        // **there is one connection.** `tcp` holds a single TCB and `connect`
        // aborts whatever was open before it. That is not a limitation to be
        // worked around with a handle table bolted on here -- it is what the
        // transport is, and a builtin handing back a fake descriptor would be
        // lying about it. A program does one exchange at a time, in order.
        //
        // Every host argument goes through `dns::lookup`, which passes a
        // dotted address through unchanged and resolves anything else. A
        // program holding an address never pays for a query, and one holding a
        // name does not have to know which it was given.
        "dns_resolve" => Ok(Value::Str(
            crate::net::dns::lookup(&text(args, 0)).map(fmt_ip).unwrap_or_default(),
        )),
        // Success as 1/0 rather than an error, because the failures here are
        // answers: a port that refuses is the result a scanner wants, not an
        // exception it has to catch. `tcp_error` carries the distinction --
        // refused, timed out, reset -- for a program that needs it.
        "tcp_connect" => {
            let host = text(args, 0);
            let Ok(ip) = crate::net::dns::lookup(&host) else {
                return Err(alloc::format!("cannot resolve '{}'", host));
            };
            let port = int(args, 1)?.clamp(0, 65_535) as u16;
            // Bounded, because an unbounded timeout in a repaint path hangs
            // the desktop and the step budget cannot see a blocking call.
            let ms = int(args, 2)?.clamp(1, 30_000) as u64;
            match crate::net::tcp::connect(ip, port, ms) {
                Ok(()) => {
                    set_err(it, "");
                    Ok(Value::Int(1))
                }
                Err(e) => {
                    set_err(it, e.name());
                    Ok(Value::Int(0))
                }
            }
        }
        "tcp_send" => {
            let data = text(args, 0);
            let ms = int(args, 1)?.clamp(1, 30_000) as u64;
            match crate::net::tcp::send(data.as_bytes(), ms) {
                Ok(()) => Ok(Value::Int(1)),
                Err(e) => {
                    set_err(it, e.name());
                    Ok(Value::Int(0))
                }
            }
        }
        // Lossy UTF-8, because a program reading a protocol is reading text
        // and one reading bytes has `hexenc` to make them legible. Refusing a
        // response for not being valid UTF-8 would throw away the header that
        // says what encoding it is.
        "tcp_recv" => {
            let ms = int(args, 0)?.clamp(1, 30_000) as u64;
            let v = crate::net::tcp::recv(ms);
            Ok(Value::Str(String::from_utf8_lossy(&v).into_owned()))
        }
        "tcp_close" => {
            crate::net::tcp::close(2_000);
            Ok(Value::Nil)
        }
        "tcp_state" => {
            Ok(Value::Str(alloc::format!("{:?}", crate::net::tcp::state()).to_lowercase()))
        }
        "tcp_error" => Ok(Value::Str(last_err(it))),
        // State and error in one answer. Read separately they can straddle a
        // preemption and describe two different moments, which is a race a
        // program has no way to see and no way to avoid.
        "tcp_status" => {
            let st = format!("{:?}", crate::net::tcp::state()).to_lowercase();
            let why = last_err(it);
            rec("Tcp", alloc::vec![Value::Str(st), Value::Str(why)])
        }
        "http_get" => {
            let host = text(args, 0);
            let Ok(ip) = crate::net::dns::lookup(&host) else {
                return Err(alloc::format!("cannot resolve '{}'", host));
            };
            let port = int(args, 1)?.clamp(1, 65_535) as u16;
            match crate::net::tcp::http_get(ip, &host, port, &text(args, 2)) {
                Ok(v) => Ok(Value::Str(String::from_utf8_lossy(&v).into_owned())),
                Err(e) => Err(alloc::format!("http_get: {}", e.name())),
            }
        }
        // TLS here validates the chain, the transcript signature, dates and
        // name and then *reports* rather than enforcing; `tls.rs` says so in
        // its own header. So this answers the body and `https_identity`
        // answers whether it was anyone in particular. One builtin returning
        // only the body would make the unauthenticated case invisible, which
        // is the exact failure that module spends its header warning about.
        "https_get" => {
            let host = text(args, 0);
            let Ok(ip) = crate::net::dns::lookup(&host) else {
                return Err(alloc::format!("cannot resolve '{}'", host));
            };
            let port = int(args, 1)?.clamp(1, 65_535) as u16;
            match crate::net::tls::https_get(ip, &host, port, &text(args, 2)) {
                Ok((body, _, _, id, _, _)) => {
                    set_err(it, if id.ok() { "" } else { "unauthenticated" });
                    Ok(Value::Str(String::from_utf8_lossy(&body).into_owned()))
                }
                Err(e) => Err(alloc::format!("https_get: {}", e.name())),
            }
        }
        "https_identity" => Ok(Value::Int((last_err(it) != "unauthenticated") as i64)),
        "udp_send" => {
            let host = text(args, 0);
            let Ok(ip) = crate::net::dns::lookup(&host) else {
                return Err(alloc::format!("cannot resolve '{}'", host));
            };
            let dst = int(args, 1)?.clamp(0, 65_535) as u16;
            let src = int(args, 2)?.clamp(0, 65_535) as u16;
            Ok(Value::Int(
                crate::net::udp::send(ip, dst, src, text(args, 3).as_bytes()) as i64,
            ))
        }
        "ping" => {
            let host = text(args, 0);
            let Ok(ip) = crate::net::dns::lookup(&host) else {
                return Err(alloc::format!("cannot resolve '{}'", host));
            };
            crate::net::ping(ip, int(args, 1)?.clamp(1, 16) as u16);
            Ok(Value::Nil)
        }

        // --- crate::ai ---------------------------------------------------
        //
        // The model, from inside a program the model may have written. Less
        // circular than it sounds, and worth being precise about why:
        // `with_engine` refuses a second holder, so `ask` called from inside
        // an authoring run answers "another task holds it" rather than
        // decoding reentrantly. The refusal is the safety property and it is
        // the existing one, not a new check bolted on here.
        "model_ready" => Ok(Value::Int(crate::ai::engine_ready() as i64)),
        "ask" => {
            let prompt = text(args, 0);
            let steps = int(args, 1)?.clamp(1, 512) as usize;
            if !crate::ai::engine_ready() {
                return Err("ask: no model is loaded".to_string());
            }
            // `generate` writes to the console, which is how every other
            // caller consumes it. Capturing is what `applet` already does, and
            // it keeps one implementation of generation rather than a second
            // that returns a string and drifts from the first.
            let opts = crate::ai::GenOpts {
                steps,
                echo_prompt: false,
                ..Default::default()
            };
            crate::gfx::console::begin_capture();
            crate::ai::generate(&prompt, &opts);
            let out = crate::gfx::console::end_capture().unwrap_or_default();
            // `generate` signs off with a rate line for whoever is watching the
            // console. A program asked for an answer, and would otherwise have
            // to know to strip a benchmark off the end of every one.
            let body: alloc::vec::Vec<&str> =
                out.lines().filter(|l| !l.trim_start().starts_with(|c: char| c.is_ascii_digit()) || !l.contains(" tokens in ")).collect();
            Ok(Value::Str(body.join("
").trim().to_string()))
        }

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

/// Why the last socket call failed.
///
/// Kept on the interpreter rather than in a static, so two programs cannot
/// read each other's failure -- and so it is gone when the interpreter is,
/// which for an application is every repaint.
fn set_err(it: &mut Interp, why: &str) {
    it.set_note("net_error", why);
}

fn last_err(it: &Interp) -> String {
    it.note("net_error")
}

/// Build one of `eval::KERNEL_RECS` by name.
///
/// The fields are passed in declaration order and checked against the
/// declaration, so an arm that adds a field without adding it to the shape --
/// or gets the order wrong -- fails here rather than handing back a record
/// whose `.ip` is its netmask. That mistake is invisible at a glance and both
/// values are strings.
fn rec(name: &str, values: alloc::vec::Vec<Value>) -> Result<Value, String> {
    let Some((_, shape)) = super::eval::KERNEL_RECS.iter().find(|(n, _)| *n == name) else {
        return Err(format!("no kernel record '{}'", name));
    };
    if shape.len() != values.len() {
        return Err(format!(
            "{} has {} field(s), built with {}",
            name,
            shape.len(),
            values.len()
        ));
    }
    let mut out = alloc::vec::Vec::with_capacity(shape.len());
    for ((fname, ty), v) in shape.iter().zip(values.into_iter()) {
        if !v.fits(ty) {
            return Err(format!(
                "{}.{} wants {}, got {}",
                name,
                fname,
                ty.name(),
                v.type_name()
            ));
        }
        out.push((String::from(*fname), v));
    }
    Ok(Value::Rec(String::from(name), out))
}

fn i(v: i64) -> Value {
    Value::Int(v)
}

fn t(v: &str) -> Value {
    Value::Str(String::from(v))
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
