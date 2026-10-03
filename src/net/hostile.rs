//! What a radio actually hears, which is not what a fixture sends.
//!
//! **Every wireless claim in this tree so far has been fed well-formed
//! frames.** `Loopback` delivers exactly what the fake access point built,
//! instantly, in order, never corrupted and never from anybody else. A real
//! radio on a real channel hears the opposite: every network in the building,
//! frames addressed to other stations, frames cut short by interference, and
//! -- on an unencrypted link, which is where every association starts --
//! whatever anyone in the room chooses to send.
//!
//! That matters more here than in most kernels. There is no unwinder, one
//! address space and no process isolation, so an index out of bounds in a
//! frame parser is not an error, it is the machine stopping. A wireless
//! parser is the largest piece of attacker-controlled input this system
//! accepts, and it accepts some of it **before there is any key at all**,
//! because EAPOL has to cross an unencrypted link in order to encrypt it.
//!
//! ### Mutation from valid seeds, not random bytes
//!
//! Random bytes almost never get past a type check, so a random fuzzer reports
//! a large number that means nothing. These start from real frames -- a
//! beacon, an authentication, an association response, a data frame, an
//! EAPOL-Key -- and truncate, flip, splice and extend them, which is what
//! keeps the input plausible enough to reach the code past the first `if`.
//!
//! The count of cases *accepted* by some parser is therefore reported and
//! claimed, not just the count run. A fuzzer whose every case is refused at
//! the first check has exercised the first check, and `smp.rs` already
//! records what a suite that cannot fail looks like from the outside.
//!
//! ### Deterministic, because a fuzzer that cannot be re-run is an anecdote
//!
//! One xorshift seeded by a constant. The case that fails is the same case on
//! every machine and every boot, so the index printed beside a failure is
//! enough to find it again -- and a fix can be shown to fix *that* case rather
//! than to have moved the search somewhere else.

use alloc::vec::Vec;

use crate::cpu::recover::{self, Caught};
use crate::dev::radio::Radio;
use crate::net::iface::Nic;
use crate::net::ieee80211 as dot11;
use crate::net::softmac::{Link, Loopback, ETHERTYPE_EAPOL};
use crate::net::{ccmp, mlme, wpa2, Mac};

use core::sync::atomic::{AtomicU32, Ordering};

/// Accounting, in statics because a case runs inside `FnOnce` and the guard
/// hands nothing back but a verdict.
static CASES: AtomicU32 = AtomicU32::new(0);
static ACCEPTED: AtomicU32 = AtomicU32::new(0);
static PANICS: AtomicU32 = AtomicU32::new(0);
static FIRST_BAD: AtomicU32 = AtomicU32::new(u32::MAX);

const SEED: u64 = 0x5EED_1EE8_0211_C0DE;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64. Not for anything that needs to be unpredictable -- the
        // point here is the opposite, that it is entirely predictable.
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next() % n as u64) as usize
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 24) as u8
    }
}

/// Well-formed frames of every kind this stack reads.
fn seeds(me: Mac, ap: Mac, tk: &[u8; 16]) -> Vec<Vec<u8>> {
    let mut v = Vec::new();
    v.push(dot11::beacon(&ap, 1, "glados", 6, true));
    v.push(dot11::probe_response(&me, &ap, 2, "glados", 6, true));
    v.push(dot11::auth_response(&me, &ap, 3, 0));
    v.push(dot11::assoc_response(&me, &ap, 4, 0, 7));
    v.push(dot11::deauth(&me, &ap, 5, 3));
    v.push(dot11::disassoc(&me, &ap, 6, 8));
    v.push(dot11::probe_request(me, "glados", dot11::BASIC_RATES));

    let ip = dot11::snap_wrap(0x0800, b"an ordinary packet");
    v.push(dot11::data_from_ds(&me, &ap, &ap, 7, &ip));
    v.push(dot11::data_to_ds(&ap, &me, &ap, 8, &ip));

    // An EAPOL-Key frame, which is the one thing a station reads from a
    // stranger before any key exists.
    let mut auth = wpa2::Authenticator::new("correct horse", b"glados", ap, me);
    let m1 = auth.message1();
    v.push(dot11::data_from_ds(&me, &ap, &ap, 9, &dot11::snap_wrap(ETHERTYPE_EAPOL, &m1)));
    // And the bare EAPOL payload, so the key parser is reached without having
    // to survive the 802.11 layer first.
    v.push(m1);

    // A protected frame, so the cipher path is fuzzed and not only the
    // plaintext one.
    if let Some(p) = ccmp::protect(tk, &dot11::data_to_ds(&ap, &me, &ap, 10, &ip), 9, 0) {
        v.push(p);
    }
    v
}

/// Bend one frame out of shape, in a way that keeps it plausible.
fn mutate(src: &[u8], other: &[u8], r: &mut Rng) -> Vec<u8> {
    let mut f = src.to_vec();
    match r.below(10) {
        // Cut it short. The commonest thing a real radio delivers: a frame
        // that hit interference partway through.
        0 => {
            let n = r.below(f.len() + 1);
            f.truncate(n);
        }
        // Flip a bit. Reaches every field eventually, including the ones that
        // are lengths.
        1 => {
            if !f.is_empty() {
                let i = r.below(f.len());
                f[i] ^= 1 << r.below(8);
            }
        }
        // A length field claiming far more than is there, which is the
        // classic 802.11 element-walker bug and the reason `elements` stops
        // at the first bad length instead of failing the frame.
        2 => {
            if !f.is_empty() {
                let i = r.below(f.len());
                f[i] = 0xFF;
            }
        }
        3 => {
            if !f.is_empty() {
                let i = r.below(f.len());
                f[i] = 0;
            }
        }
        // Longer than it should be. A parser trusting a declared length over
        // the buffer reads somebody else's bytes; one trusting the buffer
        // over the length reads padding as content.
        4 => {
            let n = r.below(64);
            for _ in 0..n {
                f.push(r.byte());
            }
        }
        // Two frames spliced, so the header says one thing and the body is
        // another kind entirely.
        5 => {
            let at = r.below(f.len() + 1);
            f.truncate(at);
            let take = r.below(other.len() + 1);
            f.extend_from_slice(&other[..take]);
        }
        // Pure noise at a plausible length, which is what a frame from a
        // network using a rate this radio cannot demodulate looks like.
        6 => {
            f.clear();
            let n = r.below(300);
            for _ in 0..n {
                f.push(r.byte());
            }
        }
        // The boundaries, named rather than waited for: a random walk reaches
        // "exactly the header length" about once in three hundred.
        7 => {
            let n = match r.below(8) {
                0 => 0,
                1 => 1,
                2 => 23,
                3 => 24,
                4 => dot11::MGMT_HDR + 5,
                5 => dot11::MGMT_HDR + 6,
                6 => 98,
                _ => 99,
            };
            f.truncate(n.min(f.len()));
            while f.len() < n {
                f.push(r.byte());
            }
        }
        // A declared length set *smaller* than the fields behind it.
        //
        // **Lengths get their own mutation because uniform random search is
        // bad at them.** A frame has two or three length bytes in a hundred,
        // and the dangerous values are the small ones -- so a random position
        // crossed with a random value reaches "a length claiming less than is
        // read" about once in ten thousand cases. That is exactly how this
        // fuzzer missed the EAPOL bug the named case below found on its first
        // try. Positions are biased to the front, where headers are, and
        // values to the low end, where the danger is.
        9 => {
            if !f.is_empty() {
                let i = r.below(f.len().min(64));
                f[i] = r.below(96) as u8;
            }
        }
        // Every byte the same, which finds a parser that happens to work
        // because real frames have structure.
        _ => {
            let b = if r.below(2) == 0 { 0x00 } else { 0xFF };
            let n = r.below(200);
            f.clear();
            for _ in 0..n {
                f.push(b);
            }
        }
    }
    f
}

/// Put one frame through every parser that could ever see it.
///
/// Returns nothing, because it runs inside a landing pad; what it learned goes
/// into `ACCEPTED`. "Accepted" means some parser answered `Some` -- the thing
/// that has to happen often enough for the run to mean anything.
fn hammer(f: &[u8], tk: &[u8; 16], me: &Mac) {
    let mut got = false;
    got |= dot11::mgmt_subtype(f).is_some();
    got |= dot11::mgmt_addrs(f).is_some();
    got |= dot11::parse_auth(f).is_some();
    got |= dot11::parse_assoc_resp(f).is_some();
    got |= dot11::parse_reason(f).is_some();
    got |= dot11::parse_beacon(f).is_some();
    got |= dot11::data_addrs(f).is_some();
    got |= dot11::snap_unwrap(f).is_some();
    got |= !dot11::elements(f).is_empty();
    got |= dot11::addressed_to(f, me);
    got |= dot11::is_beacon_like(f);

    got |= ccmp::parse(f).is_some();
    got |= ccmp::unprotect(tk, f).is_some();
    if let Some(p) = ccmp::parse(f) {
        // The two slices `softmac` takes right after this, which is where a
        // header length larger than the frame would land.
        got |= p.hdr_len <= f.len();
    }

    // The EAPOL body, reached both ways a real one is: bare, and inside a
    // SNAP-wrapped data frame.
    got |= wpa2::parse(f).is_some();
    if let Some((_, body)) = dot11::snap_unwrap(f) {
        got |= wpa2::parse(body).is_some();
    }
    if got {
        ACCEPTED.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn selftest() -> bool {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED, YELLOW};
    let mut ok = true;
    let mut check = |what: &str, pass: bool| {
        console::set_color(if pass { LTGREEN } else { LTRED });
        crate::kprintln!("  {}  {}", if pass { "ok  " } else { "FAIL" }, what);
        console::set_color(LTGRAY);
        ok &= pass;
    };

    let me: Mac = [0x02, 0, 0, 0, 0, 0x11];
    let ap: Mac = [0x02, 0, 0, 0, 0, 0xAA];
    let other: Mac = [0x02, 0, 0, 0, 0, 0x33];
    let tk = [0x5Au8; 16];

    // A fuzzer with no landing pad is a fuzzer that kills the machine the
    // first time it works. Checked rather than assumed, because `slot()`
    // answers `None` before per-core storage is armed and `guarded` then runs
    // the closure plainly -- which is honest and is not something to fuzz in.
    let armed = matches!(recover::guarded(|| {}), Caught::Ran);
    check("there is a landing pad, so a panic here is catchable", armed);
    if !armed {
        console::set_color(YELLOW);
        crate::kprintln!("  skipped: without a pad the first real find is a dead machine");
        console::set_color(LTGRAY);
        return false;
    }

    CASES.store(0, Ordering::Relaxed);
    ACCEPTED.store(0, Ordering::Relaxed);
    PANICS.store(0, Ordering::Relaxed);
    FIRST_BAD.store(u32::MAX, Ordering::Relaxed);

    let pool = seeds(me, ap, &tk);
    let mut r = Rng(SEED);
    let was = recover::in_selftest();
    recover::selftest_window(true);

    const ROUNDS: u32 = 4000;
    for i in 0..ROUNDS {
        let a = r.below(pool.len());
        let b = r.below(pool.len());
        let f = mutate(&pool[a], &pool[b], &mut r);
        CASES.fetch_add(1, Ordering::Relaxed);
        match recover::guarded(|| hammer(&f, &tk, &me)) {
            Caught::Ran => {}
            _ => {
                PANICS.fetch_add(1, Ordering::Relaxed);
                let _ = FIRST_BAD.compare_exchange(
                    u32::MAX,
                    i,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
            }
        }
    }

    // The same frames through the whole receive path, which is where a length
    // computed by one layer indexes a buffer held by another.
    let mut link = Link::new(Loopback::new(me));
    let _ = link.radio_mut().start();
    link.join(ap);
    let _ = link.keyed(&tk, 0);
    for i in 0..ROUNDS {
        let a = r.below(pool.len());
        let b = r.below(pool.len());
        let f = mutate(&pool[a], &pool[b], &mut r);
        link.radio_mut().inbox.push(f);
        CASES.fetch_add(1, Ordering::Relaxed);
        match recover::guarded(|| {
            let _ = link.receive();
        }) {
            Caught::Ran => {}
            _ => {
                PANICS.fetch_add(1, Ordering::Relaxed);
                let _ = FIRST_BAD.compare_exchange(
                    u32::MAX,
                    ROUNDS + i,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
            }
        }
    }

    // And into the state machine, in every state it has -- the point being
    // that a frame arriving in the wrong state is the ordinary case on a busy
    // channel, not an edge one.
    let mut sta = mlme::Station::new(Loopback::new(me));
    sta.start("glados", "correct horse", 0);
    let mut t = 0u64;
    const STA_ROUNDS: u32 = 1200;
    for i in 0..STA_ROUNDS {
        let a = r.below(pool.len());
        let b = r.below(pool.len());
        let f = mutate(&pool[a], &pool[b], &mut r);
        sta.link_mut().radio_mut().inbox.push(f);
        t += 40;
        CASES.fetch_add(1, Ordering::Relaxed);
        match recover::guarded(|| {
            sta.poll(t);
            let _ = sta.receive();
        }) {
            Caught::Ran => {}
            _ => {
                PANICS.fetch_add(1, Ordering::Relaxed);
                let _ = FIRST_BAD.compare_exchange(
                    u32::MAX,
                    2 * ROUNDS + i,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
            }
        }
        if matches!(sta.state(), mlme::State::Failed(_)) {
            sta.start("glados", "correct horse", t);
        }
    }

    recover::selftest_window(was);

    let cases = CASES.load(Ordering::Relaxed);
    let accepted = ACCEPTED.load(Ordering::Relaxed);
    let panics = PANICS.load(Ordering::Relaxed);
    let bad = FIRST_BAD.load(Ordering::Relaxed);

    crate::kprintln!(
        "  {} hostile frame(s), {} of them got past a type check",
        cases,
        accepted
    );
    if panics > 0 {
        console::set_color(LTRED);
        crate::kprintln!(
            "  {} case(s) stopped the machine; the first was case {}",
            panics,
            bad
        );
        console::set_color(LTGRAY);
    }
    check("not one of them stopped the machine", panics == 0);
    // Without this the claim above is satisfied by a parser that refuses
    // everything, which is the failure `smp.rs` records about a one-shot check
    // that had never once reported a difference.
    check(
        "and enough got through for that to be a claim about the parsers",
        accepted >= 200,
    );

    // --- the cases worth naming, rather than waiting for -----------------
    recover::selftest_window(true);

    // An EAPOL frame long enough to pass the length check whose *declared*
    // body is shorter than the fields the parser then reads. This is the
    // shape that matters most in the whole file: EAPOL crosses an unencrypted
    // link by design, so this frame is accepted from anybody in the room
    // before there is any key to refuse them with.
    let mut runt = alloc::vec![0u8; 99];
    runt[0] = 1;
    runt[1] = 3; // EAPOL-Key
    runt[2] = 0;
    runt[3] = 0; // and a declared body of nothing at all
    let short_body = matches!(recover::guarded(|| { let _ = wpa2::parse(&runt); }), Caught::Ran);
    check(
        "an EAPOL frame declaring a body shorter than its own fields is refused",
        short_body && wpa2::parse(&runt).is_none(),
    );

    // Every declared body length from nothing up to a whole one.
    let mut every = true;
    for n in 0..=200usize {
        let mut p = alloc::vec![0u8; 4 + 200];
        p[0] = 1;
        p[1] = 3;
        p[2] = (n >> 8) as u8;
        p[3] = n as u8;
        p[4] = 2; // KEY_TYPE_RSN, so the first check does not refuse it
        if !matches!(recover::guarded(|| { let _ = wpa2::parse(&p); }), Caught::Ran) {
            every = false;
        }
    }
    check("and so is every other declared length, from zero upwards", every);

    // A CCMP frame whose header length exceeds what is there. `softmac` takes
    // two slices at `hdr_len` immediately after parsing, so a parser that
    // answered a length without checking it hands them an index past the end.
    let mut qos = dot11::data_to_ds(&ap, &me, &ap, 1, b"x");
    qos[0] |= 0x80; // QoS subtype, so the header is two bytes longer
    let mut trimmed = true;
    for n in 0..=40usize {
        let f: Vec<u8> = qos.iter().copied().take(n).collect();
        if !matches!(
            recover::guarded(|| {
                if let Some(p) = ccmp::parse(&f) {
                    let _ = &f[..p.hdr_len];
                    let _ = &f[p.hdr_len..];
                }
            }),
            Caught::Ran
        ) {
            trimmed = false;
        }
    }
    check(
        "a header length is never larger than the frame it was read from",
        trimmed,
    );

    // An information element claiming 255 bytes inside a four-byte body.
    let mut lying = dot11::beacon(&ap, 1, "glados", 6, true);
    let at = dot11::MGMT_HDR + 12;
    if lying.len() > at + 1 {
        lying[at + 1] = 0xFF;
        lying.truncate(at + 4);
    }
    check(
        "an element claiming more than the frame holds truncates the walk",
        matches!(recover::guarded(|| { let _ = dot11::parse_beacon(&lying); }), Caught::Ran)
            && dot11::elements(&lying[dot11::MGMT_HDR + 12..]).is_empty(),
    );

    recover::selftest_window(was);

    // --- and the frames that are well-formed and not ours ----------------
    //
    // These do not crash anything; they are wrong answers rather than dead
    // machines, and a real channel is full of them.
    let mut sta = mlme::Station::new(Loopback::new(me));
    let mut fake = mlme::Ap::new(ap, me, "glados", "correct horse", 6);
    sta.start("glados", "correct horse", 0);
    let mut t = 0u64;
    for _ in 0..200 {
        fake.serve(sta.link_mut().radio_mut());
        t += mlme::DWELL_MS;
        if matches!(sta.poll(t), mlme::State::Running) {
            break;
        }
    }
    check("a station on a busy channel still gets on the network", {
        sta.state() == mlme::State::Running
    });

    // A deauthentication with the right destination and the wrong source. On a
    // real channel this is what an attacker sends first, and it costs nothing
    // to send -- 802.11w is what would authenticate it and is not here, so the
    // only defence is that the source has to be the access point we are
    // talking to.
    let forged = dot11::deauth(&me, &other, 0, 7);
    sta.link_mut().radio_mut().inbox.push(forged);
    sta.poll(t);
    check(
        "a deauthentication from anyone but our own access point is ignored",
        sta.state() == mlme::State::Running && sta.reason != 7,
    );

    // A beacon for a network we did not ask for, mid-association.
    let elsewhere = dot11::beacon(&other, 0, "somebody else", 6, true);
    sta.link_mut().radio_mut().inbox.push(elsewhere);
    sta.poll(t);
    check(
        "a beacon from another network does not disturb an association",
        sta.state() == mlme::State::Running && sta.secured(),
    );

    // A data frame addressed to a different station, which every station on a
    // channel receives and every station but one must discard.
    let theirs = dot11::data_from_ds(&other, &ap, &ap, 0, &dot11::snap_wrap(0x0800, b"not yours"));
    sta.link_mut().radio_mut().inbox.push(theirs);
    sta.poll(t);
    check(
        "an unprotected frame for another station is dropped, not delivered",
        sta.receive().is_none(),
    );

    // The management queue is bounded, so a flood cannot take the heap. An
    // unattended machine beside a busy access point hears beacons all night.
    let mut flood = Link::new(Loopback::new(me));
    let _ = flood.radio_mut().start();
    for i in 0..500u16 {
        flood.radio_mut().inbox.push(dot11::beacon(&ap, i, "glados", 6, true));
    }
    let _ = flood.receive();
    check(
        "a flood of beacons is bounded rather than growing without limit",
        flood.take_mgmt().len() <= crate::net::softmac::MGMT_QUEUE,
    );

    ok
}
