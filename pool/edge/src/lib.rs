//! The pool's rules, as exports a Durable Object can call.
//!
//! **This file decides nothing, and that is the whole design.** Everything a
//! miner is held to lives in `session::Conn` and `pool::Pool`, compiled here from
//! the same source the TCP daemon runs. JavaScript owns what JavaScript must --
//! the WebSocket, storage, timers -- and hands every line to `edge_line`; what
//! comes back is exactly what `server.rs` would have written to a socket.
//!
//! ### The boundary
//!
//! Strings cross it as bytes in this module's memory. JavaScript asks
//! `edge_alloc` for room, copies a line in, calls the function, and frees the
//! room. Results are left in one output buffer that `edge_out_ptr` points at and
//! the call's return value measures, so no pointer is ever handed back that the
//! caller would have to remember to free.
//!
//! A session's answer is a sequence of records, `[kind][u32 length][bytes]`, with
//! kind `S` for a message to the miner, `L` for a log line and `C` for "hang up".
//! Tagged rather than delimited, because a message may contain any byte a JSON
//! string can, and a delimiter is one more thing to escape and get wrong.

use std::collections::BTreeMap;
use std::sync::Mutex;

use glados_pool::pool::Pool;
use glados_pool::session::{Conn, Out};
use glados_pool::upstream::{UpOut, Upstream};

struct State {
    pool: Mutex<Pool>,
    conns: BTreeMap<u32, Conn>,
    next: u32,
    /// One per coin that has an upstream, in slot order. JavaScript owns the
    /// sockets and addresses these by index.
    ups: Vec<Upstream>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
static OUT: Mutex<Vec<u8>> = Mutex::new(Vec::new());

fn set_out(bytes: &[u8]) -> u32 {
    let mut o = OUT.lock().unwrap();
    o.clear();
    o.extend_from_slice(bytes);
    o.len() as u32
}

fn text(ptr: *const u8, len: u32) -> String {
    if ptr.is_null() || len == 0 {
        return String::new();
    }
    let b = unsafe { core::slice::from_raw_parts(ptr, len as usize) };
    String::from_utf8_lossy(b).into_owned()
}

fn encode(out: &Out) -> Vec<u8> {
    let mut v = Vec::new();
    let mut push = |kind: u8, bytes: &[u8]| {
        v.push(kind);
        v.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        v.extend_from_slice(bytes);
    };
    for m in &out.send {
        push(b'S', m.as_bytes());
    }
    for l in &out.log {
        push(b'L', l.as_bytes());
    }
    if out.close {
        push(b'C', &[]);
    }
    v
}

#[no_mangle]
pub extern "C" fn edge_alloc(len: u32) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len as usize);
    let p = v.as_mut_ptr();
    core::mem::forget(v);
    p
}

#[no_mangle]
pub extern "C" fn edge_free(ptr: *mut u8, len: u32) {
    if !ptr.is_null() {
        unsafe { drop(Vec::from_raw_parts(ptr, 0, len as usize)) };
    }
}

#[no_mangle]
pub extern "C" fn edge_out_ptr() -> *const u8 {
    OUT.lock().unwrap().as_ptr()
}

/// Build the pool from coin specs, one per line.
///
/// Answers the number of slots, or zero with the reason left in the output
/// buffer -- a coin that will not parse is an error the operator should read,
/// not a pool that starts with one coin fewer than it was told.
#[no_mangle]
pub extern "C" fn edge_init(ptr: *const u8, len: u32, share_secs: u32, window: f64) -> u32 {
    let specs = text(ptr, len);
    let mut coins = Vec::new();
    for line in specs.lines().map(str::trim).filter(|l| !l.is_empty()) {
        match glados_pool::spec::parse_coin(line) {
            Ok(c) => coins.push(c),
            Err(e) => {
                set_out(format!("{line}: {e}").as_bytes());
                return 0;
            }
        }
    }
    if coins.is_empty() {
        set_out(b"no coins");
        return 0;
    }
    glados_pool::vardiff::set_target_secs(share_secs as u64);
    let mut pool = Pool::new(coins);
    if window > 0.0 {
        pool.set_window(window as u64);
    }
    let slots = pool.slots();
    let ups = (0..slots as usize).filter_map(|i| Upstream::for_slot(&pool, i)).collect();
    *STATE.lock().unwrap() = Some(State { pool: Mutex::new(pool), conns: BTreeMap::new(), next: 1, ups });
    slots
}

/// A connection arrived. Answers its id, or zero before `edge_init`.
#[no_mangle]
pub extern "C" fn edge_open(ptr: *const u8, len: u32) -> u32 {
    let peer = text(ptr, len);
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return 0 };
    let id = s.next;
    s.next = s.next.wrapping_add(1).max(1);
    s.conns.insert(id, Conn::new(&peer));
    id
}

/// One line from a miner. Answers the length of the records left in the buffer.
#[no_mangle]
pub extern "C" fn edge_line(conn: u32, ptr: *const u8, len: u32) -> u32 {
    let line = text(ptr, len);
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return 0 };
    let Some(c) = s.conns.get_mut(&conn) else { return 0 };
    let out = c.on_line(&s.pool, &line);
    set_out(&encode(&out))
}

/// A job period passed with nothing from this miner.
#[no_mangle]
pub extern "C" fn edge_idle(conn: u32) -> u32 {
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return 0 };
    let Some(c) = s.conns.get_mut(&conn) else { return 0 };
    let out = c.on_idle(&s.pool);
    set_out(&encode(&out))
}

#[no_mangle]
pub extern "C" fn edge_close(conn: u32) {
    if let Some(s) = STATE.lock().unwrap().as_mut() {
        s.conns.remove(&conn);
    }
}

/// The share log as canonical JSON, the same document `--ledger` writes.
#[no_mangle]
pub extern "C" fn edge_ledger(epoch: f64, generated_at: f64) -> u32 {
    let g = STATE.lock().unwrap();
    let Some(s) = g.as_ref() else { return 0 };
    let doc = s.pool.lock().unwrap().ledger_json(epoch as u64, generated_at as u64);
    set_out(doc.as_bytes())
}

/// Restore a ledger written earlier. Answers records loaded, or -1.
#[no_mangle]
pub extern "C" fn edge_load_ledger(ptr: *const u8, len: u32) -> i32 {
    let doc = text(ptr, len);
    let g = STATE.lock().unwrap();
    let Some(s) = g.as_ref() else { return -1 };
    let r = s.pool.lock().unwrap().load_ledger(&doc);
    match r {
        Ok(n) => n as i32,
        Err(e) => {
            drop(g);
            set_out(e.as_bytes());
            -1
        }
    }
}

// --- Upstream -----------------------------------------------------------------
//
// The same `Upstream` the native daemon drives from a thread, driven here from
// a Durable Object's outbound `connect()`. Records as for a session, plus `W`
// for "work was installed, hand miners fresh jobs", and `C` carrying the reason
// the connection is finished.

fn encode_up(out: &UpOut) -> Vec<u8> {
    let mut v = Vec::new();
    let mut push = |kind: u8, bytes: &[u8]| {
        v.push(kind);
        v.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        v.extend_from_slice(bytes);
    };
    for m in &out.send {
        push(b'S', m.as_bytes());
    }
    for l in &out.log {
        push(b'L', l.as_bytes());
    }
    if out.work {
        push(b'W', &[]);
    }
    if let Some(why) = &out.close {
        push(b'C', why.as_bytes());
    }
    v
}

fn with_up(i: u32, f: impl FnOnce(&mut Upstream, &Mutex<Pool>) -> UpOut) -> u32 {
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return 0 };
    let State { pool, ups, .. } = s;
    let Some(u) = ups.get_mut(i as usize) else { return 0 };
    let out = f(u, pool);
    set_out(&encode_up(&out))
}

/// How many upstreams there are.
#[no_mangle]
pub extern "C" fn edge_up_count() -> u32 {
    STATE.lock().unwrap().as_ref().map(|s| s.ups.len() as u32).unwrap_or(0)
}

/// `host\tport` of upstream `i`, left in the output buffer.
#[no_mangle]
pub extern "C" fn edge_up_where(i: u32) -> u32 {
    let g = STATE.lock().unwrap();
    let Some(u) = g.as_ref().and_then(|s| s.ups.get(i as usize)) else { return 0 };
    let text = format!("{}\t{}", u.host, u.port);
    drop(g);
    set_out(text.as_bytes())
}

#[no_mangle]
pub extern "C" fn edge_up_open(i: u32) -> u32 {
    with_up(i, |u, _| u.open())
}

#[no_mangle]
pub extern "C" fn edge_up_bytes(i: u32, ptr: *const u8, len: u32) -> u32 {
    let bytes = if ptr.is_null() || len == 0 {
        Vec::new()
    } else {
        unsafe { core::slice::from_raw_parts(ptr, len as usize) }.to_vec()
    };
    with_up(i, |u, p| u.on_bytes(p, &bytes))
}

#[no_mangle]
pub extern "C" fn edge_up_tick(i: u32) -> u32 {
    with_up(i, |u, p| u.on_tick(p))
}

/// Fresh jobs for one connection, after upstream installed new work.
#[no_mangle]
pub extern "C" fn edge_work(conn: u32) -> u32 {
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return 0 };
    let Some(c) = s.conns.get_mut(&conn) else { return 0 };
    let out = c.on_work(&s.pool);
    set_out(&encode(&out))
}
