//! `/run/glados/netctl`: the network, as a ring-3 program asks for it.
//!
//! **The mechanism stays in the kernel and the policy moves out.** The
//! drivers, the 802.11 station, WPA2 and DHCP are ring 0 and stay there --
//! they are what the agent's own fetches run on. What does not need to be is
//! the *manager*: which network to join, when to rejoin, what to remember.
//! That is a program, `netd`, at ring 3, and this socket is everything it may
//! ask of the machine. A line in, lines back, the last one `ok` or `err ...`:
//!
//! ```text
//! status                 iface <name> up|down link=0|1 ip=.. gw=.. dns=..
//!                        wifi <state> ssid=<ssid|-> secure=0|1   (or: wifi none)
//! networks               net <bars> open|secured <ssid>          (last scan)
//! scan                   sweep the channels
//! join <ssid> [pass]     join; DHCP follows on its own once the link is up
//! leave
//! dhcp eth0|wlan0        ask again
//! known                  known <ssid>      (never the passphrase)
//! remember <ssid> [pass] / forget <ssid>
//! manage on|off          a ring-3 manager is deciding; the kernel's own
//!                        auto-join stands down while one is
//! ```
//!
//! Served like Skywalker: in the client's own syscalls, from `sky::server::
//! pump`, under the guest lock. Text rather than netlink because nothing at
//! ring 3 here speaks rtnetlink and a manager written for this machine
//! should be readable with `cat`.
//!
//! **Known networks are `/etc/net/known`**, `ssid<TAB>pass` per line, and
//! the kernel's own `net::autojoin` reads the same file -- so a machine with
//! no ring-3 manager running still rejoins the network it knows, and one
//! with `netd` running gets the same answer from the same record.

use super::unix::{self, Pipe, Side, Sock};
use crate::sync::Racy;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt::Write;

pub const PATH: &str = "/run/glados/netctl";
pub const KNOWN: &str = "/etc/net/known";

struct Conn {
    pipe: Rc<RefCell<Pipe>>,
    side: Side,
    inbuf: Vec<u8>,
    out: Vec<u8>,
    manages: bool,
}

static LISTENER: Racy<Option<Rc<RefCell<Sock>>>> = Racy::new(None);
static CONNS: Racy<Vec<Conn>> = Racy::new(Vec::new());

/// Whether a ring-3 manager has said it is deciding, as the session it
/// speaks from. **The session, not a flag**: a manager that is killed never
/// closes its socket in any way this side sees -- the close is only noticed
/// inside some guest's syscall, and there may be none left -- so the kernel's
/// own auto-join stood down forever behind a dead `netd`. A session that has
/// ended cannot be deciding anything.
const NOBODY: usize = usize::MAX;
static MANAGED: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(NOBODY);

pub fn managed() -> bool {
    let s = MANAGED.load(core::sync::atomic::Ordering::Acquire);
    if s == NOBODY {
        return false;
    }
    if super::session::main_task_of(s).is_none() {
        MANAGED.store(NOBODY, core::sync::atomic::Ordering::Release);
        return false;
    }
    true
}

pub fn serve() {
    let l = unsafe { LISTENER.get() };
    if l.is_some() {
        return;
    }
    let sock = Rc::new(RefCell::new(Sock::Fresh));
    if unix::bind(&sock, PATH).is_ok() && unix::listen(&sock).is_ok() {
        *l = Some(sock);
    }
}

pub fn pump() {
    let Some(listener) = unsafe { LISTENER.get() }.clone() else { return };
    let conns = unsafe { CONNS.get() };
    while let Ok(Some((pipe, side))) = unix::accept(&listener) {
        conns.push(Conn { pipe, side, inbuf: Vec::new(), out: Vec::new(), manages: false });
    }
    let mut i = 0;
    while i < conns.len() {
        let c = &mut conns[i];
        let mut buf = [0u8; 512];
        let mut gone = false;
        loop {
            match unix::read(&c.pipe, c.side, &mut buf) {
                Ok(0) => {
                    gone = true;
                    break;
                }
                Ok(n) => c.inbuf.extend_from_slice(&buf[..n]),
                Err(_) => break,
            }
        }
        while let Some(nl) = c.inbuf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = c.inbuf.drain(..=nl).collect();
            let line = String::from_utf8_lossy(&line[..line.len() - 1]).into_owned();
            let reply = answer(line.trim(), &mut c.manages);
            c.out.extend_from_slice(reply.as_bytes());
        }
        if !c.out.is_empty() {
            if let Ok(n) = unix::write(&c.pipe, c.side, &c.out) {
                c.out.drain(..n);
            }
        }
        if gone {
            if c.manages {
                MANAGED.store(NOBODY, core::sync::atomic::Ordering::Release);
            }
            unix::close(&c.pipe, c.side);
            conns.remove(i);
        } else {
            i += 1;
        }
    }
}

fn ip(a: [u8; 4]) -> String {
    alloc::format!("{}.{}.{}.{}", a[0], a[1], a[2], a[3])
}

/// The known networks, in the order they were remembered.
pub fn known() -> Vec<(String, String)> {
    let Some(b) = crate::sysbox::read_blob(KNOWN) else { return Vec::new() };
    String::from_utf8_lossy(&b)
        .lines()
        .filter_map(|l| {
            let mut p = l.splitn(2, '\t');
            let s = p.next()?.trim();
            (!s.is_empty()).then(|| (String::from(s), String::from(p.next().unwrap_or(""))))
        })
        .collect()
}

fn store_known(k: &[(String, String)]) -> bool {
    let mut t = String::new();
    for (s, p) in k {
        let _ = writeln!(t, "{}\t{}", s, p);
    }
    crate::sysbox::write_blob(KNOWN, t.into_bytes())
}

/// Remember a network, replacing what was known of it.
pub fn remember(ssid: &str, pass: &str) -> bool {
    if ssid.is_empty() || ssid.contains('\t') || ssid.contains('\n') || pass.contains('\n') {
        return false;
    }
    let mut k = known();
    k.retain(|(s, _)| s != ssid);
    k.push((String::from(ssid), String::from(pass)));
    store_known(&k)
}

pub fn forget(ssid: &str) -> bool {
    let mut k = known();
    let n = k.len();
    k.retain(|(s, _)| s != ssid);
    k.len() != n && store_known(&k)
}

/// One request, one reply. Pure in the sense that matters for a test: the
/// whole of what a client sees comes back as the string.
pub fn answer(line: &str, manages: &mut bool) -> String {
    let mut out = String::new();
    let mut words = line.splitn(3, ' ');
    let verb = words.next().unwrap_or("");
    let a = words.next().unwrap_or("").trim();
    let b = words.next().unwrap_or("").trim();
    let ok = match verb {
        "status" => {
            for n in [crate::net::ETH0, crate::net::WLAN0] {
                let i = &mut crate::net::ifaces()[n];
                if i.nic.is_none() {
                    continue;
                }
                let link = i.usable() as u8;
                let _ = writeln!(
                    out,
                    "iface {} {} link={} ip={} gw={} dns={}",
                    i.name,
                    if i.up { "up" } else { "down" },
                    link,
                    ip(i.ip),
                    ip(i.gateway),
                    ip(i.dns)
                );
            }
            let _claim = crate::net::claim_wifi();
            match crate::net::wlan() {
                None => out.push_str("wifi none\n"),
                Some(w) => {
                    let (state, secure) = w.status();
                    let ssid = w.ssid().unwrap_or_else(|| String::from("-"));
                    let _ = writeln!(out, "wifi {} ssid={} secure={}", state, ssid, secure as u8);
                }
            }
            Ok(())
        }
        "networks" => {
            let _claim = crate::net::claim_wifi();
            match crate::net::wlan() {
                None => Err("no wireless driver"),
                Some(w) => {
                    for n in w.networks() {
                        let _ = writeln!(
                            out,
                            "net {} {} {}",
                            crate::net::wifi::bars(n.rssi),
                            if n.secured { "secured" } else { "open" },
                            n.ssid
                        );
                    }
                    Ok(())
                }
            }
        }
        "scan" => {
            let _claim = crate::net::claim_wifi();
            match crate::net::wlan() {
                None => Err("no wireless driver"),
                Some(w) => {
                    w.join("", "", crate::net::now_ms());
                    Ok(())
                }
            }
        }
        "join" if !a.is_empty() => {
            let _claim = crate::net::claim_wifi();
            match crate::net::wlan() {
                None => Err("no wireless driver"),
                Some(w) => {
                    // A passphrase left off means the known one, if there is.
                    let pass = if b.is_empty() {
                        known().into_iter().find(|(s, _)| s == a).map(|(_, p)| p).unwrap_or_default()
                    } else {
                        String::from(b)
                    };
                    w.join(a, &pass, crate::net::now_ms());
                    Ok(())
                }
            }
        }
        "leave" => {
            let _claim = crate::net::claim_wifi();
            match crate::net::wlan() {
                None => Err("no wireless driver"),
                Some(w) => {
                    w.leave_net();
                    Ok(())
                }
            }
        }
        "dhcp" => {
            let n = match a {
                "eth0" => Some(crate::net::ETH0),
                "wlan0" => Some(crate::net::WLAN0),
                _ => None,
            };
            match n {
                Some(n) if crate::net::ifaces()[n].nic.is_some() => {
                    // The shell's idle loop runs it: DHCP blocks for seconds and
                    // a guest's syscall must not.
                    crate::net::ifaces()[n].dhcp_wanted = true;
                    Ok(())
                }
                _ => Err("no such interface"),
            }
        }
        "known" => {
            for (s, _) in known() {
                let _ = writeln!(out, "known {}", s);
            }
            Ok(())
        }
        "remember" if !a.is_empty() => {
            if remember(a, b) { Ok(()) } else { Err("could not write /etc/net/known") }
        }
        "forget" if !a.is_empty() => {
            if forget(a) { Ok(()) } else { Err("not known") }
        }
        "manage" => {
            let on = a != "off";
            *manages = on;
            let who = if on { super::session::current().unwrap_or(NOBODY) } else { NOBODY };
            MANAGED.store(who, core::sync::atomic::Ordering::Release);
            Ok(())
        }
        "" => Ok(()),
        _ => Err("unknown request; try status, networks, scan, join, leave, dhcp, known, remember, forget, manage"),
    };
    match ok {
        Ok(()) => out.push_str("ok\n"),
        Err(why) => {
            let _ = writeln!(out, "err {}", why);
        }
    }
    out
}

pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();
    let mut m = false;
    out.push(("an unknown request is refused by name", answer("frobnicate", &mut m).starts_with("err unknown request")));
    out.push(("an empty line is answered, not dropped", answer("", &mut m) == "ok\n"));
    let s = answer("status", &mut m);
    out.push(("status ends ok and always says what wifi is", s.ends_with("ok\n") && s.contains("wifi ")));
    out.push(("an ssid with a tab cannot be remembered", !remember("a\tb", "x")));
    out
}
