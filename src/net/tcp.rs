//! TCP: one connection at a time, actively opened.
//!
//! Enough to fetch something over HTTP, which is the point -- a byte stream is
//! the thing every useful protocol is built on, and until one exists the
//! network can only answer questions about itself.
//!
//! ### What is here
//!
//! The RFC 793 state machine for an active open, RFC 6298 retransmission with
//! Jacobson/Karels round-trip estimation and Karn's algorithm, sequence
//! arithmetic that survives wraparound, an MSS option, and a receive buffer
//! that advertises a real window.
//!
//! ### What is deliberately not here, and what it costs
//!
//!   * **No reassembly queue.** An out-of-order segment is dropped and
//!     acknowledged with the sequence we still want, which is a duplicate ACK
//!     and makes the peer resend. Correct, and slower than it needs to be
//!     exactly when the network is already losing packets.
//!   * **No congestion control.** There is a fixed four-segment cap on flight
//!     size standing in for it. That is not TCP-friendly and would be rude at
//!     scale; it is defensible while every send is a short request.
//!   * **No Nagle and no delayed ACK.** Both are latency-for-efficiency
//!     trades. Sending immediately is easier to reason about, and this stack
//!     has no bulk traffic to be inefficient with.
//!   * **One connection.** A second `connect` replaces the first. The control
//!     block is a single static, matching the one-entry ARP cache next door.
//!   * **No interrupt-driven receive.** The card is polled: during a blocking
//!     call by `wait_until`, and otherwise by `service` from the shell's idle
//!     loop. So the stack advances at best 100 times a second while the
//!     machine sits at a prompt, and not at all while a long command runs.
//!   * **Shortened TIME_WAIT.** 2 seconds rather than 2*MSL. The hazard 2*MSL
//!     guards against is an old duplicate landing on a new incarnation of the
//!     same four-tuple; the local port is drawn fresh from the TSC on every
//!     connect, so a new incarnation almost never reuses the tuple.
//!
//! ### The part that is easy to get wrong
//!
//! Sequence numbers are u32 and they wrap. Comparing them with `<` is correct
//! for about 49 days of a busy connection and then silently is not. Every
//! comparison here goes through `seq_lt` and friends, which subtract and
//! interpret the result as a signed 32-bit number -- the RFC 793 rule, and the
//! reason a connection that wraps mid-transfer does not stall.

use super::{checksum, config, send_ipv4, Ipv4, PROTO_TCP};
use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED, YELLOW};
use crate::kprintln;
use crate::sync::Racy;
use alloc::vec::Vec;

const FIN: u8 = 0x01;
const SYN: u8 = 0x02;
const RST: u8 = 0x04;
const PSH: u8 = 0x08;
const ACK: u8 = 0x10;

/// 1460 = 1500 byte Ethernet MTU - 20 IP - 20 TCP. The card's receive buffers
/// are 2048 bytes, so a full-size segment fits with room to spare.
const MSS: usize = 1460;

/// What we advertise. Large enough that a peer sending a small HTTP response
/// never stalls waiting for a window update.
const RCV_CAPACITY: usize = 32768;

/// Standing in for congestion control. See the module note.
const MAX_FLIGHT_SEGMENTS: usize = 4;

/// RFC 6298 puts the floor at 1 second. Kept, even though round trips here are
/// measured in microseconds: the floor exists to avoid spurious retransmits
/// when the RTT estimate is young, not to match the network.
const RTO_MIN_TICKS: u64 = crate::TIMER_HZ as u64;
const RTO_MAX_TICKS: u64 = 60 * crate::TIMER_HZ as u64;
const TIME_WAIT_TICKS: u64 = 2 * crate::TIMER_HZ as u64;

/// After this many retransmissions of the same segment, give up and reset.
const MAX_RETRIES: u32 = 8;

/// Bound on the inbox. TCP already has a story for a dropped segment, so
/// dropping under pressure is safe in a way that growing without limit is not.
const MAX_INBOX: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum State {
    Closed,
    SynSent,
    Established,
    FinWait1,
    FinWait2,
    Closing,
    TimeWait,
    CloseWait,
    LastAck,
}

impl State {
    pub fn name(self) -> &'static str {
        match self {
            State::Closed => "CLOSED",
            State::SynSent => "SYN_SENT",
            State::Established => "ESTABLISHED",
            State::FinWait1 => "FIN_WAIT_1",
            State::FinWait2 => "FIN_WAIT_2",
            State::Closing => "CLOSING",
            State::TimeWait => "TIME_WAIT",
            State::CloseWait => "CLOSE_WAIT",
            State::LastAck => "LAST_ACK",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Error {
    NoNic,
    Timeout,
    Refused,
    Reset,
    NotConnected,
}

impl Error {
    pub fn name(self) -> &'static str {
        match self {
            Error::NoNic => "no NIC",
            Error::Timeout => "timed out",
            Error::Refused => "connection refused",
            Error::Reset => "connection reset",
            Error::NotConnected => "not connected",
        }
    }
}

// --- sequence arithmetic -------------------------------------------------
//
// RFC 793's rule: a is before b when their difference, read as a signed 32-bit
// number, is negative. This is what makes wraparound a non-event.

fn seq_lt(a: u32, b: u32) -> bool {
    (a.wrapping_sub(b) as i32) < 0
}
fn seq_le(a: u32, b: u32) -> bool {
    (a.wrapping_sub(b) as i32) <= 0
}
fn seq_gt(a: u32, b: u32) -> bool {
    (a.wrapping_sub(b) as i32) > 0
}
fn seq_ge(a: u32, b: u32) -> bool {
    (a.wrapping_sub(b) as i32) >= 0
}

struct Tcb {
    state: State,
    remote: Ipv4,
    remote_port: u16,
    local_port: u16,

    /// Oldest unacknowledged sequence number.
    snd_una: u32,
    /// Next sequence number to send.
    snd_nxt: u32,
    /// The peer's advertised receive window.
    snd_wnd: u16,
    /// Initial send sequence, kept to recognise the SYN-ACK.
    iss: u32,

    /// Next sequence number expected from the peer.
    rcv_nxt: u32,

    /// Unacknowledged outbound bytes, beginning at `snd_una`. A byte leaves
    /// here only when the peer acknowledges it, because until then it may need
    /// to be sent again.
    send_buf: Vec<u8>,
    /// Delivered inbound bytes, waiting to be read.
    recv_buf: Vec<u8>,

    /// Set once close() has been asked for; the FIN goes out after the last
    /// byte of `send_buf` has been sent.
    closing: bool,
    fin_sent: bool,
    /// Sequence number the FIN occupied. A FIN consumes one number without
    /// carrying a byte, which is the source of most off-by-one bugs here.
    fin_seq: u32,
    peer_fin: bool,

    /// Retransmission, in scheduler ticks. 0 means disarmed.
    retx_deadline: u64,
    rto: u64,
    retries: u32,

    /// Jacobson/Karels, in microseconds.
    srtt: u64,
    rttvar: u64,
    /// Karn's algorithm: a retransmitted segment is not a valid RTT sample,
    /// because there is no way to tell which transmission the ACK answers.
    timing: bool,
    timed_seq: u32,
    timed_at: u64,

    /// Set when the peer resets us, so a blocking call can report why.
    reset: bool,

    deadline_wait: u64,
}

static TCB: Racy<Option<Tcb>> = Racy::new(None);
static INBOX: Racy<Vec<(Ipv4, Vec<u8>)>> = Racy::new(Vec::new());

fn ticks() -> u64 {
    crate::dev::lapic::ticks()
}

fn now_us() -> u64 {
    let mhz = crate::time::tsc_mhz();
    if mhz == 0 {
        // No calibrated TSC: fall back to tick granularity so the estimator
        // still gets a monotonic number rather than a constant.
        return ticks() * (1_000_000 / crate::TIMER_HZ as u64);
    }
    crate::time::rdtsc() / mhz
}

/// Both the initial sequence number and the ephemeral port come from here.
///
/// RFC 793 wants an ISS that advances with time so that segments from a
/// previous incarnation of a connection cannot be mistaken for current ones.
/// The TSC is exactly such a clock.
fn entropy() -> u32 {
    crate::time::rdtsc() as u32
}

// --- segment construction ------------------------------------------------

fn tcp_checksum(src: Ipv4, dst: Ipv4, segment: &[u8]) -> u16 {
    // The pseudo-header is not transmitted; it exists so the checksum covers
    // the addresses, which catches a segment delivered to the wrong host.
    let mut buf = Vec::with_capacity(12 + segment.len());
    buf.extend_from_slice(&src);
    buf.extend_from_slice(&dst);
    buf.push(0);
    buf.push(PROTO_TCP);
    buf.extend_from_slice(&(segment.len() as u16).to_be_bytes());
    buf.extend_from_slice(segment);
    checksum(&buf)
}

impl Tcb {
    fn window(&self) -> u16 {
        RCV_CAPACITY.saturating_sub(self.recv_buf.len()).min(u16::MAX as usize) as u16
    }

    /// Build one segment. `with_mss` adds the option, which is only legal on a
    /// segment carrying SYN.
    fn segment(&self, flags: u8, seq: u32, payload: &[u8], with_mss: bool) -> Vec<u8> {
        let header_words = if with_mss { 6 } else { 5 };
        let mut s = Vec::with_capacity(header_words * 4 + payload.len());
        s.extend_from_slice(&self.local_port.to_be_bytes());
        s.extend_from_slice(&self.remote_port.to_be_bytes());
        s.extend_from_slice(&seq.to_be_bytes());
        let ack_num = if flags & ACK != 0 { self.rcv_nxt } else { 0 };
        s.extend_from_slice(&ack_num.to_be_bytes());
        s.push((header_words as u8) << 4);
        s.push(flags);
        s.extend_from_slice(&self.window().to_be_bytes());
        s.extend_from_slice(&[0, 0]); // checksum, filled below
        s.extend_from_slice(&[0, 0]); // urgent pointer
        if with_mss {
            s.push(2); // kind: maximum segment size
            s.push(4); // length
            s.extend_from_slice(&(MSS as u16).to_be_bytes());
        }
        s.extend_from_slice(payload);

        let c = tcp_checksum(config().ip, self.remote, &s);
        s[16..18].copy_from_slice(&c.to_be_bytes());
        s
    }

    fn in_flight(&self) -> u32 {
        self.snd_nxt.wrapping_sub(self.snd_una)
    }

    fn arm_retx(&mut self) {
        self.retx_deadline = ticks() + self.rto;
    }

    fn disarm_retx(&mut self) {
        self.retx_deadline = 0;
        self.retries = 0;
    }

    /// Fold a fresh round-trip sample into the estimator (RFC 6298 §2).
    fn observe_rtt(&mut self, sample_us: u64) {
        if self.srtt == 0 {
            self.srtt = sample_us;
            self.rttvar = sample_us / 2;
        } else {
            let delta = self.srtt.abs_diff(sample_us);
            self.rttvar = (3 * self.rttvar + delta) / 4;
            self.srtt = (7 * self.srtt + sample_us) / 8;
        }
        // RTO = SRTT + 4*RTTVAR, converted to ticks and clamped. The clock
        // granularity is one tick, so anything smaller rounds up to one.
        let us = self.srtt + 4 * self.rttvar;
        let per_tick = 1_000_000 / crate::TIMER_HZ as u64;
        self.rto = ((us + per_tick - 1) / per_tick).clamp(RTO_MIN_TICKS, RTO_MAX_TICKS);
    }
}

/// Everything a state transition wants to put on the wire, collected so it can
/// be sent *after* the borrow of the control block ends. See the module note
/// in `net` on re-entrancy.
type Outbox = Vec<Vec<u8>>;

fn flush(remote: Ipv4, out: Outbox) {
    for seg in out {
        send_ipv4(remote, PROTO_TCP, &seg);
    }
}

fn with_tcb<R>(f: impl FnOnce(&mut Tcb) -> R) -> Option<R> {
    unsafe { TCB.get().as_mut().map(f) }
}

// --- inbound -------------------------------------------------------------

/// Queue a segment. Called from `net::poll`; does no work beyond validation.
pub fn deliver(src: Ipv4, segment: &[u8]) {
    if segment.len() < 20 {
        return;
    }
    // Verify the checksum here, once, so the state machine never has to
    // consider whether what it is reading is real.
    if tcp_checksum(src, config().ip, segment) != 0 {
        return;
    }
    let inbox = unsafe { &mut *INBOX.get() };
    if inbox.len() < MAX_INBOX {
        inbox.push((src, segment.to_vec()));
    }
}

/// Give the stack a slice of an otherwise idle moment.
///
/// The shell calls this each time it finds no keystroke waiting, which is the
/// only reason a connection makes progress between commands. Without it a
/// peer's FIN sits unread in the receive ring: the connection stays
/// ESTABLISHED long after the other end has finished, the peer retransmits its
/// FIN into silence, and `report` prints a state that stopped being true
/// seconds ago. That is exactly what happened before this existed.
///
/// It is bounded rather than draining the ring, so a burst of traffic cannot
/// stall the prompt. It is also what makes the machine answer a ping while
/// sitting idle, which it previously only did while running `ping` itself.
pub fn service() {
    if !super::ready() {
        return;
    }
    for _ in 0..8 {
        if matches!(super::poll(), super::Event::None) {
            break;
        }
    }
    pump();
}

/// Drain the inbox and run the timers. Every blocking operation calls this in
/// its wait loop, and `service` calls it when the shell is idle.
pub fn pump() {
    // Take the queue before processing so that a `send_ipv4` triggered from
    // inside a transition -- which may poll, which may enqueue -- is writing
    // to an empty inbox rather than the one being iterated.
    let batch = core::mem::take(unsafe { &mut *INBOX.get() });
    for (src, seg) in batch {
        let (remote, out) = match with_tcb(|t| (t.remote, on_segment(t, src, &seg))) {
            Some(v) => v,
            None => {
                // Nothing is listening. Tell the peer rather than making it
                // wait for a timeout, but never answer a reset with a reset.
                reject(src, &seg);
                continue;
            }
        };
        flush(remote, out);
    }

    let (remote, out) = match with_tcb(|t| (t.remote, on_tick(t))) {
        Some(v) => v,
        None => return,
    };
    flush(remote, out);

    // A finished connection is dropped here rather than inside the borrow.
    let done = with_tcb(|t| t.state == State::Closed).unwrap_or(false);
    if done {
        let keep = with_tcb(|t| core::mem::take(&mut t.recv_buf)).unwrap_or_default();
        let last = with_tcb(|t| (t.state, t.reset));
        if let Some((_, reset)) = last {
            unsafe { *TCB.get() = None };
            LAST_RESET.set(reset);
            LAST_DATA.set(keep);
        }
    }
}

/// Carried across the drop of a control block so `recv` and the shell can
/// still report what arrived and why it ended.
struct Cell<T>(Racy<Option<T>>);
impl<T> Cell<T> {
    const fn new() -> Self {
        Cell(Racy::new(None))
    }
    fn set(&self, v: T) {
        unsafe { *self.0.get() = Some(v) };
    }
    fn take(&self) -> Option<T> {
        unsafe { self.0.get().take() }
    }
}
static LAST_DATA: Cell<Vec<u8>> = Cell::new();
static LAST_RESET: Cell<bool> = Cell::new();

/// Answer a segment that matches no connection.
fn reject(src: Ipv4, seg: &[u8]) {
    let flags = seg[13];
    if flags & RST != 0 {
        return;
    }
    let their_seq = u32::from_be_bytes([seg[4], seg[5], seg[6], seg[7]]);
    let their_ack = u32::from_be_bytes([seg[8], seg[9], seg[10], seg[11]]);
    let data_off = ((seg[12] >> 4) as usize) * 4;
    let payload_len = seg.len().saturating_sub(data_off);
    let syn_fin = ((flags & SYN != 0) as u32) + ((flags & FIN != 0) as u32);

    let mut r = Vec::with_capacity(20);
    r.extend_from_slice(&seg[2..4]); // their destination port is ours
    r.extend_from_slice(&seg[0..2]);
    if flags & ACK != 0 {
        // Their ACK tells us which sequence number they expect from us.
        r.extend_from_slice(&their_ack.to_be_bytes());
        r.extend_from_slice(&0u32.to_be_bytes());
        r.push(5 << 4);
        r.push(RST);
    } else {
        r.extend_from_slice(&0u32.to_be_bytes());
        let ack = their_seq
            .wrapping_add(payload_len as u32)
            .wrapping_add(syn_fin);
        r.extend_from_slice(&ack.to_be_bytes());
        r.push(5 << 4);
        r.push(RST | ACK);
    }
    r.extend_from_slice(&0u16.to_be_bytes()); // window
    r.extend_from_slice(&[0, 0]); // checksum
    r.extend_from_slice(&[0, 0]); // urgent
    let c = tcp_checksum(config().ip, src, &r);
    r[16..18].copy_from_slice(&c.to_be_bytes());
    send_ipv4(src, PROTO_TCP, &r);
}

fn on_segment(t: &mut Tcb, src: Ipv4, seg: &[u8]) -> Outbox {
    let mut out = Outbox::new();

    let src_port = u16::from_be_bytes([seg[0], seg[1]]);
    let dst_port = u16::from_be_bytes([seg[2], seg[3]]);
    if src != t.remote || src_port != t.remote_port || dst_port != t.local_port {
        return out;
    }

    let seg_seq = u32::from_be_bytes([seg[4], seg[5], seg[6], seg[7]]);
    let seg_ack = u32::from_be_bytes([seg[8], seg[9], seg[10], seg[11]]);
    let data_off = ((seg[12] >> 4) as usize) * 4;
    let flags = seg[13];
    let wnd = u16::from_be_bytes([seg[14], seg[15]]);
    if data_off < 20 || data_off > seg.len() {
        return out;
    }
    let payload = &seg[data_off..];

    if flags & RST != 0 {
        t.state = State::Closed;
        t.reset = true;
        return out;
    }

    if t.state == State::SynSent {
        // The only segment that opens the connection is a SYN-ACK
        // acknowledging exactly our ISS+1.
        if flags & SYN != 0 && flags & ACK != 0 {
            if seg_ack != t.iss.wrapping_add(1) {
                // Acknowledges something we never sent.
                t.state = State::Closed;
                t.reset = true;
                return out;
            }
            t.rcv_nxt = seg_seq.wrapping_add(1);
            t.snd_una = seg_ack;
            t.snd_wnd = wnd;
            t.state = State::Established;
            t.disarm_retx();
            if t.timing && seq_ge(seg_ack, t.timed_seq) {
                let s = now_us().saturating_sub(t.timed_at);
                t.observe_rtt(s);
                t.timing = false;
            }
            out.push(t.segment(ACK, t.snd_nxt, &[], false));
            queue_pending(t, &mut out);
        }
        return out;
    }

    // --- acknowledgement ---
    if flags & ACK != 0 && seq_gt(seg_ack, t.snd_una) && seq_le(seg_ack, t.snd_nxt) {
        let acked = seg_ack.wrapping_sub(t.snd_una) as usize;
        // A FIN consumes a sequence number but occupies no byte of send_buf,
        // so the drain is capped by what is actually buffered.
        let drain = acked.min(t.send_buf.len());
        t.send_buf.drain(..drain);
        t.snd_una = seg_ack;

        if t.timing && seq_ge(seg_ack, t.timed_seq) {
            let s = now_us().saturating_sub(t.timed_at);
            t.observe_rtt(s);
            t.timing = false;
        }
        t.retries = 0;
        if t.in_flight() == 0 {
            t.disarm_retx();
        } else {
            t.arm_retx();
        }
    }
    t.snd_wnd = wnd;

    // --- inbound data ---
    let mut need_ack = false;
    if !payload.is_empty() {
        if seg_seq == t.rcv_nxt {
            let room = RCV_CAPACITY.saturating_sub(t.recv_buf.len());
            let take = payload.len().min(room);
            t.recv_buf.extend_from_slice(&payload[..take]);
            t.rcv_nxt = t.rcv_nxt.wrapping_add(take as u32);
        }
        // Either way the peer gets an ACK. In order it is an acknowledgement;
        // out of order it is a duplicate ACK naming what we still want, which
        // is how the peer learns to resend without waiting for its timer.
        need_ack = true;
    }

    // A FIN is only consumed if everything before it has been. That test is
    // what stops an out-of-order FIN from closing the connection early.
    if flags & FIN != 0 && seg_seq.wrapping_add(payload.len() as u32) == t.rcv_nxt {
        t.rcv_nxt = t.rcv_nxt.wrapping_add(1);
        t.peer_fin = true;
        need_ack = true;
    }

    // --- state transitions ---
    let fin_acked = t.fin_sent && seq_gt(t.snd_una, t.fin_seq);
    match t.state {
        State::Established => {
            if t.peer_fin {
                t.state = State::CloseWait;
            }
        }
        State::FinWait1 => {
            if fin_acked && t.peer_fin {
                t.state = State::TimeWait;
                t.deadline_wait = ticks() + TIME_WAIT_TICKS;
            } else if fin_acked {
                t.state = State::FinWait2;
            } else if t.peer_fin {
                t.state = State::Closing;
            }
        }
        State::FinWait2 => {
            if t.peer_fin {
                t.state = State::TimeWait;
                t.deadline_wait = ticks() + TIME_WAIT_TICKS;
            }
        }
        State::Closing => {
            if fin_acked {
                t.state = State::TimeWait;
                t.deadline_wait = ticks() + TIME_WAIT_TICKS;
            }
        }
        State::LastAck => {
            if fin_acked {
                t.state = State::Closed;
            }
        }
        _ => {}
    }

    if t.state != State::Closed {
        queue_pending(t, &mut out);
        if need_ack && out.is_empty() {
            out.push(t.segment(ACK, t.snd_nxt, &[], false));
        }
    }
    out
}

/// Send whatever the window allows, then the FIN if one is due.
fn queue_pending(t: &mut Tcb, out: &mut Outbox) {
    if t.state == State::SynSent || t.state == State::Closed {
        return;
    }

    let mut sent_any = false;
    loop {
        let unsent_off = t.snd_nxt.wrapping_sub(t.snd_una) as usize;
        if unsent_off >= t.send_buf.len() {
            break;
        }
        let flight = t.in_flight() as usize;
        if flight >= MAX_FLIGHT_SEGMENTS * MSS {
            break;
        }
        let window = t.snd_wnd as usize;
        if flight >= window {
            break;
        }
        let allowed = (window - flight)
            .min(MAX_FLIGHT_SEGMENTS * MSS - flight)
            .min(MSS);
        let avail = t.send_buf.len() - unsent_off;
        let n = allowed.min(avail);
        if n == 0 {
            break;
        }
        let chunk = t.send_buf[unsent_off..unsent_off + n].to_vec();
        let seq = t.snd_nxt;
        out.push(t.segment(ACK | PSH, seq, &chunk, false));
        t.snd_nxt = t.snd_nxt.wrapping_add(n as u32);
        if !t.timing {
            t.timing = true;
            t.timed_seq = seq.wrapping_add(n as u32);
            t.timed_at = now_us();
        }
        sent_any = true;
    }

    // The FIN goes out only once every buffered byte has been transmitted.
    if t.closing && !t.fin_sent {
        let all_sent = t.snd_nxt.wrapping_sub(t.snd_una) as usize >= t.send_buf.len();
        if all_sent {
            t.fin_seq = t.snd_nxt;
            out.push(t.segment(ACK | FIN, t.snd_nxt, &[], false));
            t.snd_nxt = t.snd_nxt.wrapping_add(1);
            t.fin_sent = true;
            sent_any = true;
        }
    }

    if sent_any && t.retx_deadline == 0 {
        t.arm_retx();
    }
}

fn on_tick(t: &mut Tcb) -> Outbox {
    let mut out = Outbox::new();
    let now = ticks();

    if t.state == State::TimeWait && now >= t.deadline_wait {
        t.state = State::Closed;
        return out;
    }

    if t.retx_deadline == 0 || now < t.retx_deadline {
        return out;
    }

    if t.retries >= MAX_RETRIES {
        t.state = State::Closed;
        t.reset = true;
        return out;
    }
    t.retries += 1;
    // Exponential backoff, and Karn: no RTT sample may be taken from a
    // retransmission, because the ACK cannot be attributed to a transmission.
    t.rto = (t.rto * 2).min(RTO_MAX_TICKS);
    t.timing = false;

    match t.state {
        State::SynSent => {
            out.push(t.segment(SYN, t.iss, &[], true));
        }
        _ => {
            // Go back to the oldest unacknowledged byte and resend from there.
            let n = t.send_buf.len().min(MSS);
            if n > 0 {
                let chunk = t.send_buf[..n].to_vec();
                out.push(t.segment(ACK | PSH, t.snd_una, &chunk, false));
            } else if t.fin_sent && seq_le(t.snd_una, t.fin_seq) {
                out.push(t.segment(ACK | FIN, t.fin_seq, &[], false));
            } else {
                out.push(t.segment(ACK, t.snd_nxt, &[], false));
            }
        }
    }
    t.arm_retx();
    out
}

// --- the blocking API ----------------------------------------------------

/// Run the stack for up to `ms`, stopping as soon as `done` is satisfied.
///
/// Idles on `hlt` rather than calling `task::yield_now`: there is nothing to
/// do until a packet or a timer tick arrives, and `hlt` waits for exactly that
/// without burning the CPU. The preemptive timer still switches tasks, so the
/// shell and the model are not starved.
///
/// This loop is also what exposed the scheduler bug documented on
/// `task::yield_now` -- yielding a hundred times a second turned a race that
/// had survived unnoticed into a hang on every run.
fn wait_until(ms: u64, mut done: impl FnMut() -> bool) -> bool {
    let deadline = ticks() + (ms * crate::TIMER_HZ as u64) / 1000 + 1;
    loop {
        for _ in 0..16 {
            super::poll();
        }
        pump();
        if done() {
            return true;
        }
        if ticks() >= deadline {
            return false;
        }
        unsafe { core::arch::asm!("hlt", options(nomem, nostack)) };
    }
}

pub fn state() -> State {
    with_tcb(|t| t.state).unwrap_or(State::Closed)
}

/// Open a connection, replacing any existing one.
pub fn connect(dst: Ipv4, port: u16, timeout_ms: u64) -> Result<(), Error> {
    if !super::ready() {
        return Err(Error::NoNic);
    }
    abort();
    LAST_DATA.take();
    LAST_RESET.take();

    let iss = entropy();
    // Ephemeral range. Drawn fresh each time so a new connection almost never
    // reuses a four-tuple a previous one has just finished with.
    let local_port = 49152 + (entropy() % 16384) as u16;

    let tcb = Tcb {
        state: State::SynSent,
        remote: dst,
        remote_port: port,
        local_port,
        snd_una: iss,
        snd_nxt: iss.wrapping_add(1),
        snd_wnd: MSS as u16,
        iss,
        rcv_nxt: 0,
        send_buf: Vec::new(),
        recv_buf: Vec::new(),
        closing: false,
        fin_sent: false,
        fin_seq: 0,
        peer_fin: false,
        retx_deadline: 0,
        rto: RTO_MIN_TICKS,
        retries: 0,
        srtt: 0,
        rttvar: 0,
        timing: true,
        timed_seq: iss.wrapping_add(1),
        timed_at: now_us(),
        reset: false,
        deadline_wait: 0,
    };
    unsafe { *TCB.get() = Some(tcb) };

    let (remote, syn) = with_tcb(|t| {
        t.arm_retx();
        (t.remote, t.segment(SYN, t.iss, &[], true))
    })
    .ok_or(Error::NotConnected)?;
    send_ipv4(remote, PROTO_TCP, &syn);

    let ok = wait_until(timeout_ms, || {
        !matches!(state(), State::SynSent)
    });

    match state() {
        State::Established => Ok(()),
        _ if !ok => {
            abort();
            Err(Error::Timeout)
        }
        // A RST in response to a SYN is a refusal, which is worth
        // distinguishing from silence: one means nothing is listening on that
        // port, the other means nothing is there at all.
        _ => {
            let refused = LAST_RESET.take().unwrap_or(false);
            abort();
            Err(if refused { Error::Refused } else { Error::Reset })
        }
    }
}

/// Queue bytes and push them out, waiting for the window if it is closed.
pub fn send(data: &[u8], timeout_ms: u64) -> Result<(), Error> {
    if !matches!(state(), State::Established | State::CloseWait) {
        return Err(Error::NotConnected);
    }
    let (remote, out) = with_tcb(|t| {
        t.send_buf.extend_from_slice(data);
        let mut out = Outbox::new();
        queue_pending(t, &mut out);
        (t.remote, out)
    })
    .ok_or(Error::NotConnected)?;
    flush(remote, out);

    // Wait for it to be acknowledged, so that a caller which sends then closes
    // does not close over data the peer never confirmed.
    let done = wait_until(timeout_ms, || {
        with_tcb(|t| t.send_buf.is_empty()).unwrap_or(true)
    });
    match state() {
        State::Closed => Err(if LAST_RESET.take().unwrap_or(false) {
            Error::Reset
        } else {
            Error::NotConnected
        }),
        _ if !done => Err(Error::Timeout),
        _ => Ok(()),
    }
}

/// Read whatever has arrived, waiting up to `timeout_ms` for the first byte.
///
/// Returns as soon as data is available rather than trying to fill anything --
/// a stream has no record boundaries to wait for, and the caller knows what it
/// is expecting better than this does.
pub fn recv(timeout_ms: u64) -> Vec<u8> {
    wait_until(timeout_ms, || {
        with_tcb(|t| !t.recv_buf.is_empty() || t.peer_fin).unwrap_or(true)
    });
    if let Some(v) = with_tcb(|t| core::mem::take(&mut t.recv_buf)) {
        if !v.is_empty() {
            return v;
        }
    }
    LAST_DATA.take().unwrap_or_default()
}

/// Read until the peer closes or `timeout_ms` elapses.
pub fn recv_to_end(timeout_ms: u64) -> Vec<u8> {
    let mut all = Vec::new();
    let deadline = ticks() + (timeout_ms * crate::TIMER_HZ as u64) / 1000 + 1;
    loop {
        let chunk = recv(200);
        all.extend_from_slice(&chunk);
        let finished = with_tcb(|t| t.peer_fin && t.recv_buf.is_empty()).unwrap_or(true);
        if finished {
            // The connection may have been dropped by pump() with data still
            // in the carry-over cell.
            if let Some(rest) = LAST_DATA.take() {
                all.extend_from_slice(&rest);
            }
            break;
        }
        if ticks() >= deadline {
            break;
        }
    }
    all
}

/// Close politely: FIN, and wait for the exchange to finish.
pub fn close(timeout_ms: u64) {
    let Some((remote, out)) = with_tcb(|t| {
        t.closing = true;
        if t.state == State::Established {
            t.state = State::FinWait1;
        } else if t.state == State::CloseWait {
            t.state = State::LastAck;
        }
        let mut out = Outbox::new();
        queue_pending(t, &mut out);
        (t.remote, out)
    }) else {
        return;
    };
    flush(remote, out);
    wait_until(timeout_ms, || matches!(state(), State::Closed));
    abort();
}

/// Drop the connection without ceremony.
fn abort() {
    unsafe { *TCB.get() = None };
    unsafe { (*INBOX.get()).clear() };
}

pub fn report() {
    console::set_color(YELLOW);
    kprintln!("[tcp]");
    console::set_color(LTGRAY);
    match with_tcb(|t| {
        (
            t.state,
            t.remote,
            t.remote_port,
            t.local_port,
            t.send_buf.len(),
            t.recv_buf.len(),
            t.srtt,
            t.rto,
            t.retries,
            t.snd_wnd,
        )
    }) {
        None => kprintln!("  no connection"),
        Some((st, r, rp, lp, sb, rb, srtt, rto, retries, wnd)) => {
            kprintln!("  {}", st.name());
            kprintln!("  {}.{}.{}.{}:{} from :{}", r[0], r[1], r[2], r[3], rp, lp);
            kprintln!("  unacked {} B   readable {} B   peer window {} B", sb, rb, wnd);
            if srtt > 0 {
                kprintln!("  srtt {} us   rto {} ms   retransmits {}",
                    srtt, rto * (1000 / crate::TIMER_HZ as u64), retries);
            } else {
                kprintln!("  no round trip measured yet");
            }
        }
    }
}

/// Fetch one resource over HTTP/1.0 and return the whole response.
///
/// HTTP/1.0 with `Connection: close` on purpose: the end of the body is the
/// end of the stream, so this needs no chunked-transfer decoder and no
/// Content-Length parser to know when it is done.
pub fn http_get(dst: Ipv4, port: u16, path: &str) -> Result<Vec<u8>, Error> {
    connect(dst, port, 5000)?;

    let mut req = alloc::string::String::new();
    req.push_str("GET ");
    req.push_str(if path.is_empty() { "/" } else { path });
    req.push_str(" HTTP/1.0\r\nHost: ");
    let c = dst;
    req.push_str(&alloc::format!("{}.{}.{}.{}", c[0], c[1], c[2], c[3]));
    req.push_str("\r\nUser-Agent: glados/0.1\r\nConnection: close\r\n\r\n");

    send(req.as_bytes(), 5000)?;
    let body = recv_to_end(10000);
    close(2000);
    Ok(body)
}

/// Print a summary of a fetch rather than dumping the whole body.
pub fn http_report(dst: Ipv4, port: u16, path: &str) {
    console::set_color(YELLOW);
    kprintln!("[http] {}.{}.{}.{}:{}{}", dst[0], dst[1], dst[2], dst[3], port,
        if path.is_empty() { "/" } else { path });
    console::set_color(LTGRAY);

    let t0 = crate::time::rdtsc();
    match http_get(dst, port, path) {
        Err(e) => {
            console::set_color(LTRED);
            kprintln!("  {}", e.name());
            console::set_color(LTGRAY);
        }
        Ok(body) => {
            let mhz = crate::time::tsc_mhz().max(1);
            let ms = (crate::time::rdtsc() - t0) / mhz / 1000;
            let text = alloc::string::String::from_utf8_lossy(&body);
            let mut lines = text.lines();
            if let Some(status) = lines.next() {
                console::set_color(LTGREEN);
                kprintln!("  {}", status.trim());
                console::set_color(LTGRAY);
            }
            // Headers end at the first blank line; show the body after it.
            let split = text.find("\r\n\r\n").map(|i| i + 4)
                .or_else(|| text.find("\n\n").map(|i| i + 2));
            match split {
                Some(i) => {
                    let b = &text[i..];
                    kprintln!("  {} B in {} ms, {} B of body", body.len(), ms, b.len());
                    for line in b.lines().take(12) {
                        kprintln!("  | {}", line);
                    }
                    if b.lines().count() > 12 {
                        kprintln!("  | ... {} more lines", b.lines().count() - 12);
                    }
                }
                None => kprintln!("  {} B in {} ms, no header break found", body.len(), ms),
            }
        }
    }
}
