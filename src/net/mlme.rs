//! Getting onto a network: scan, authenticate, associate, handshake, keys.
//!
//! This is the MLME -- the station state machine 802.11 calls a management
//! entity -- and like everything else in `softmac` it is written once for every
//! chip rather than once per driver. A FullMAC part's firmware does all of it
//! internally and a SoftMAC part does none of it, which is the entire practical
//! difference between the two and the reason `Caps::softmac` exists.
//!
//! ### The clock is an argument, and that is the whole testability argument
//!
//! `poll(now_ms)` takes the time rather than reading it. Every state here is a
//! deadline and a retry count, so a machine that read its own clock could only
//! be tested by waiting -- seconds per claim, at boot, on every machine
//! forever. Handed the time, the whole association path runs in a loop with no
//! delay at all and a timeout is asserted by naming a number.
//!
//! It is the same reason `update::decide`, `repair::rank` and `code::locate`
//! are pure functions over their inputs: the states worth checking are the ones
//! that are expensive or impossible to arrange for real.
//!
//! ### What it will not do, said here rather than found later
//!
//! No 802.11w protected management frames, so a deauthentication is
//! unauthenticated and anybody in earshot can send one -- which is true of most
//! deployed networks and is the attack every "wifi jammer" performs. No power
//! save, no roaming between access points of one network, no 802.1X/EAP (this
//! is WPA2-Personal, a passphrase and a PSK), and no group rekeying on a timer:
//! the GTK arrives once in message 3 and a network that rotates it will
//! eventually stop delivering broadcast traffic here.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::dev::radio::{self, JoinTarget, Radio, Rx};
use crate::net::iface::{Kind, Nic, Wlan};
use crate::net::ieee80211 as dot11;
use crate::net::softmac::{Link, ETHERTYPE_EAPOL};
use crate::net::wpa2;
use crate::net::Mac;

/// How long to sit on one channel before moving to the next.
///
/// A beacon interval is typically 100 ms, so anything under that can miss a
/// network entirely on a passive channel -- which is the failure that reads as
/// "it does not see my access point" and is really a scan that did not wait.
pub const DWELL_MS: u64 = 120;
pub const AUTH_MS: u64 = 500;
pub const ASSOC_MS: u64 = 500;
/// Longer than the other two on purpose: the four-way handshake is four frames
/// and two of them are the access point's, and an access point that is busy
/// takes its time about message 3.
pub const HANDSHAKE_MS: u64 = 4000;
pub const TRIES: u8 = 3;
/// How long an offloaded scan may run beyond what the host-driven one would
/// have taken. A firmware scan that never reports completion must still end,
/// and it must end as a failure the operator can read rather than as a scan
/// that is quietly still going.
pub const OFFLOAD_SLACK_MS: u64 = 2000;

/// Received frames held for the layer above while the MLME is the one draining
/// the radio. Bounded for the reason the management queue is.
const RX_QUEUE: usize = 32;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum State {
    Idle,
    Scanning,
    Authenticating,
    Associating,
    /// Associated, unencrypted, running the four-way handshake. **This is the
    /// dangerous state**, and it is named rather than folded into the one
    /// before it: the link carries frames in the clear here, which is correct
    /// and is also exactly where an attacker wants it kept.
    Handshaking,
    Running,
    Failed(&'static str),
}

impl State {
    pub fn name(&self) -> &'static str {
        match self {
            State::Idle => "idle",
            State::Scanning => "scanning",
            State::Authenticating => "authenticating",
            State::Associating => "associating",
            State::Handshaking => "handshaking",
            State::Running => "running",
            State::Failed(why) => why,
        }
    }
}

/// One network heard during a scan.
#[derive(Clone)]
pub struct Bss {
    pub ssid: String,
    pub bssid: Mac,
    pub channel: u8,
    pub secured: bool,
    pub rsn: bool,
    pub rssi: i8,
    pub beacon_int: u16,
    pub dtim: Option<u8>,
}

impl Bss {
    fn join_target(&self) -> JoinTarget {
        JoinTarget { bssid: self.bssid, channel: self.channel, beacon_int: self.beacon_int, dtim: self.dtim }
    }
}

pub struct Station<R: Radio> {
    link: Link<R>,
    ssid: String,
    pass: String,
    state: State,
    /// When the current state was entered, in the caller's milliseconds.
    since: u64,
    tries: u8,
    plan: Vec<u8>,
    at: usize,
    sup: Option<wpa2::Supplicant>,
    target: Option<Bss>,
    rx_q: Vec<Vec<u8>>,
    eapol_q: Vec<Vec<u8>>,
    /// Everything heard in the last scan, strongest signal first.
    pub seen: Vec<Bss>,
    pub aid: u16,
    /// The reason code from the last deauthentication or disassociation.
    pub reason: u16,
    pub mgmt_tx: u32,
    pub mgmt_rx: u32,
    pub eapol_rx: u32,
    /// The radio is scanning on its own, and the plan is only a deadline.
    offloaded: bool,
    /// `prepare_join` succeeded and `left` has not been called since, so the
    /// radio holds state about an access point that has to be taken down.
    prepared: bool,
}

impl<R: Radio> Station<R> {
    pub fn new(radio: R) -> Station<R> {
        Station {
            link: Link::new(radio),
            ssid: String::new(),
            pass: String::new(),
            state: State::Idle,
            since: 0,
            tries: 0,
            plan: Vec::new(),
            at: 0,
            sup: None,
            target: None,
            rx_q: Vec::new(),
            eapol_q: Vec::new(),
            seen: Vec::new(),
            aid: 0,
            reason: 0,
            mgmt_tx: 0,
            mgmt_rx: 0,
            eapol_rx: 0,
            offloaded: false,
            prepared: false,
        }
    }

    pub fn state(&self) -> State {
        self.state
    }

    pub fn link_mut(&mut self) -> &mut Link<R> {
        &mut self.link
    }

    pub fn target(&self) -> Option<&Bss> {
        self.target.as_ref()
    }

    pub fn secured(&self) -> bool {
        self.link.secured()
    }

    /// Begin. An empty passphrase means an open network, which is a decision
    /// the caller makes and not one this infers from the beacon: a network that
    /// advertises no encryption and is joined by a station that expected some
    /// is a station about to send its traffic in the clear.
    pub fn start(&mut self, ssid: &str, pass: &str, now: u64) {
        self.abort_scan();
        self.ssid = ssid.to_string();
        self.pass = pass.to_string();
        self.seen.clear();
        self.rx_q.clear();
        self.eapol_q.clear();
        self.target = None;
        self.sup = None;
        self.aid = 0;
        self.reason = 0;
        self.link.leave();
        self.unprepare();

        if self.link.radio_mut().start().is_err() {
            self.state = State::Failed("the radio would not start");
            return;
        }

        let band5 = self.link.caps().band5;
        self.plan = radio::channels(radio::Band::G24);
        if band5 {
            self.plan.extend(radio::channels(radio::Band::G5));
        }
        self.at = 0;
        self.state = State::Scanning;
        self.tries = 0;
        self.since = now;
        // A wildcard here too, for the reason `tune_and_probe` gives.
        let plan = self.plan.clone();
        match self.link.radio_mut().scan_offload("", &plan) {
            None => {
                self.offloaded = false;
                self.tune_and_probe(now);
            }
            Some(Ok(())) => self.offloaded = true,
            Some(Err(why)) => {
                self.offloaded = false;
                self.state = State::Failed(why);
            }
        }
    }

    /// Tell the radio it no longer has an access point, if it was told it had
    /// one. Idempotent, because three different paths end an association and
    /// making each of them remember whether another already did is how one of
    /// them forgets.
    fn unprepare(&mut self) {
        if self.prepared {
            self.prepared = false;
            self.link.radio_mut().left();
        }
    }

    /// Say goodbye properly and stand down.
    ///
    /// The deauthentication is sent rather than the link simply being
    /// forgotten, because an access point holds an association -- and its
    /// buffered frames, and one of its finite association identifiers -- until
    /// it is told or times out. Leaving silently is how a network fills up with
    /// stations that are not there.
    /// Call off a firmware scan still running. Idempotent.
    fn abort_scan(&mut self) {
        if self.offloaded && self.state == State::Scanning {
            self.link.radio_mut().scan_abort();
        }
        self.offloaded = false;
    }

    pub fn stop(&mut self) {
        self.abort_scan();
        self.goodbye();
        self.sup = None;
        self.state = State::Idle;
    }

    /// The deauthentication, if the access point holds state about us, then
    /// the link and the radio let go -- **in that order**, because a part told
    /// it has no access point refuses to send to one, and the goodbye went out
    /// after that told it nothing.
    ///
    /// The trigger is whether the *access point* has state about us, not
    /// whether we succeeded: a handshake that timed out leaves an association
    /// held at the other end exactly as a working one does, and that is the
    /// case where going quiet is worst.
    fn goodbye(&mut self) {
        if self.link.bssid() != [0u8; 6] {
            if let Some(b) = self.target.clone() {
                let me = self.link.mac();
                let seq = self.link.next_seq();
                let f = dot11::deauth(&b.bssid, &me, seq, dot11::REASON_LEAVING);
                if self.link.tx_raw(&f) {
                    self.mgmt_tx += 1;
                }
            }
        }
        self.link.leave();
        self.unprepare();
    }

    // --- the loop ---------------------------------------------------------

    /// One turn of the state machine. Cheap, and safe to call as often as the
    /// caller likes: everything it does is gated on a deadline or on a frame
    /// having arrived.
    pub fn poll(&mut self, now: u64) -> State {
        // **This order is the protocol's and not a preference.** An access
        // point sends the association response and the handshake's message 1
        // back to back, so both are taken off the radio in one drain -- and
        // the supplicant that must read message 1 does not exist until the
        // association response has been read. Feeding data as it came off the
        // radio drops the first message of every handshake, which presents as
        // a network that associates and then times out, and is the single
        // worst place for an ordering bug because everything visible works.
        self.drain();
        self.handle_mgmt(now);
        self.feed_eapol();

        match self.state {
            State::Scanning if self.offloaded => {
                if self.link.radio_mut().scan_done() {
                    self.offloaded = false;
                    self.choose(now);
                } else if now.saturating_sub(self.since) >= self.plan.len() as u64 * DWELL_MS + OFFLOAD_SLACK_MS {
                    self.abort_scan();
                    self.state = State::Failed("the radio's own scan never reported finishing");
                }
            }
            State::Scanning => {
                if now.saturating_sub(self.since) >= DWELL_MS {
                    self.at += 1;
                    if self.at >= self.plan.len() {
                        self.choose(now);
                    } else {
                        self.tune_and_probe(now);
                    }
                }
            }
            State::Authenticating => {
                if now.saturating_sub(self.since) >= AUTH_MS {
                    self.retry(now, "no answer to the authentication", |s, n| s.send_auth(n));
                }
            }
            State::Associating => {
                if now.saturating_sub(self.since) >= ASSOC_MS {
                    self.retry(now, "no answer to the association", |s, n| s.send_assoc(n));
                }
            }
            State::Handshaking => {
                if now.saturating_sub(self.since) >= HANDSHAKE_MS {
                    // Not retried. A handshake that does not complete is
                    // nearly always a wrong passphrase, and the access point
                    // will not say so -- message 3 simply never arrives, or
                    // arrives with a MIC that does not verify. Retrying hides
                    // the one diagnosis worth printing.
                    self.state = State::Failed("the handshake did not finish");
                }
            }
            _ => {}
        }
        // However it failed, a radio still holding an access point is told it
        // does not: a firmware time event left standing pins the part to one
        // channel until something else happens to clear it.
        if matches!(self.state, State::Failed(_)) {
            self.goodbye();
        }
        self.state
    }

    fn retry(&mut self, now: u64, why: &'static str, again: fn(&mut Self, u64)) {
        self.tries += 1;
        if self.tries >= TRIES {
            self.state = State::Failed(why);
            return;
        }
        again(self, now);
    }

    fn tune_and_probe(&mut self, now: u64) {
        self.since = now;
        let ch = self.plan[self.at];
        if self.link.radio_mut().set_channel(ch).is_err() {
            return;
        }
        // Passive on the radar channels: nothing here can listen for radar and
        // vacate, so a probe request there is a transmission that must not
        // happen. Beacons are still heard, which is what "passive scan" means.
        if radio::needs_dfs(ch) {
            return;
        }
        let me = self.link.mac();
        // A wildcard probe, so every access point in earshot answers and the
        // scan finds networks it was not told to look for. A named probe is
        // the only way to find a hidden network and is also how a station
        // announces what it is looking for to everybody in the room.
        let f = dot11::probe_request(me, "", dot11::BASIC_RATES);
        if self.link.tx_raw(&f) {
            self.mgmt_tx += 1;
        }
    }

    fn choose(&mut self, now: u64) {
        // Strongest signal wins among access points carrying the network asked
        // for. Several is the ordinary case rather than a corner one -- that is
        // what a mesh or an office is -- and picking the first heard means
        // picking the one whose beacon happened to land first.
        let mut best: Option<Bss> = None;
        for b in self.seen.iter() {
            if b.ssid != self.ssid {
                continue;
            }
            if best.as_ref().map(|c| b.rssi > c.rssi).unwrap_or(true) {
                best = Some(b.clone());
            }
        }
        // A scan with no network named is a scan, and it is over. Reporting
        // "no access point is carrying that network" would be true of the
        // empty string and useless to read: the operator asked what is in the
        // air, got an answer, and is not looking for anything yet.
        if self.ssid.is_empty() {
            self.state = State::Idle;
            return;
        }
        let b = match best {
            Some(b) => b,
            None => {
                self.state = State::Failed("no access point is carrying that network");
                return;
            }
        };
        // A secured network with no passphrase, or a passphrase for an open
        // one, is a mismatch the operator should hear about rather than a
        // connection that half works.
        if b.secured && self.pass.is_empty() {
            self.state = State::Failed("that network is encrypted and no passphrase was given");
            return;
        }
        if b.secured && !b.rsn {
            self.state = State::Failed("that network is WEP, which is not security");
            return;
        }
        // The radio's turn: a plain part tunes, a firmware-assisted one sets up
        // everything it must know before a frame can reach this access point.
        // A refusal ends here, because an authentication sent without it is a
        // frame that never leaves the part and times out three times over.
        if let Err(why) = self.link.radio_mut().prepare_join(&b.join_target()) {
            self.state = State::Failed(why);
            return;
        }
        self.prepared = true;
        self.target = Some(b);
        self.tries = 0;
        self.state = State::Authenticating;
        self.send_auth(now);
    }

    fn send_auth(&mut self, now: u64) {
        self.since = now;
        let bssid = match &self.target {
            Some(b) => b.bssid,
            None => return,
        };
        let me = self.link.mac();
        let seq = self.link.next_seq();
        let f = dot11::auth_open(&bssid, &me, seq, 1);
        if self.link.tx_raw(&f) {
            self.mgmt_tx += 1;
        }
    }

    fn send_assoc(&mut self, now: u64) {
        self.since = now;
        let (bssid, rsn) = match &self.target {
            Some(b) => (b.bssid, b.rsn),
            None => return,
        };
        let me = self.link.mac();
        let seq = self.link.next_seq();
        let ssid = self.ssid.clone();
        let f = dot11::assoc_request(&bssid, &me, seq, &ssid, dot11::BASIC_RATES, rsn);
        if self.link.tx_raw(&f) {
            self.mgmt_tx += 1;
        }
    }

    // --- what arrives -----------------------------------------------------

    /// Drain the radio once, sorting what comes off it into three piles.
    ///
    /// **One drain, in one place.** `Link::receive` is what reads the radio and
    /// it sets management frames aside as it goes, so the MLME and the IP stack
    /// cannot consume each other's frames whichever of them is called first.
    /// Nothing is *acted* on here -- see `poll` for why the order matters.
    fn drain(&mut self) {
        loop {
            let eth = match self.link.receive() {
                Some(e) => e,
                None => break,
            };
            if eth.len() >= 14 && u16::from_be_bytes([eth[12], eth[13]]) == ETHERTYPE_EAPOL {
                self.eapol_rx += 1;
                if self.eapol_q.len() >= RX_QUEUE {
                    self.eapol_q.remove(0);
                }
                self.eapol_q.push(eth);
                continue;
            }
            if self.rx_q.len() >= RX_QUEUE {
                self.rx_q.remove(0);
            }
            self.rx_q.push(eth);
        }
    }

    fn feed_eapol(&mut self) {
        let frames = core::mem::take(&mut self.eapol_q);
        for eth in frames {
            self.on_eapol(&eth[14..]);
        }
    }

    fn on_eapol(&mut self, payload: &[u8]) {
        let (reply, done) = match self.sup.as_mut() {
            Some(s) => {
                let r = s.on_frame(payload);
                (r, s.done)
            }
            None => return,
        };
        if let Some(r) = reply {
            let bssid = self.link.bssid();
            let me = self.link.mac();
            let mut eth = Vec::with_capacity(14 + r.len());
            eth.extend_from_slice(&bssid);
            eth.extend_from_slice(&me);
            eth.extend_from_slice(&ETHERTYPE_EAPOL.to_be_bytes());
            eth.extend_from_slice(&r);
            let _ = self.link.transmit(&eth);
        }
        if done && self.state == State::Handshaking {
            self.install();
        }
    }

    /// Message 4 has gone out; install the temporal key.
    ///
    /// **The order is the point.** The key goes in only after the supplicant
    /// has finished, because installing it earlier would stop the link
    /// accepting the plaintext EAPOL frames the handshake itself is made of --
    /// a station that encrypted halfway through the exchange that establishes
    /// encryption, which is an easy mistake to make and presents as a handshake
    /// that always stalls at message 3.
    fn install(&mut self) {
        let tk = match self.sup.as_ref().and_then(|s| s.ptk.as_ref()) {
            Some(p) => wpa2::Ptk(p).tk().to_vec(),
            None => {
                self.state = State::Failed("the handshake finished with no key");
                return;
            }
        };
        if self.link.keyed(&tk, 0) {
            self.state = State::Running;
        } else {
            self.state = State::Failed("the key was refused");
        }
    }

    fn handle_mgmt(&mut self, now: u64) {
        let frames = self.link.take_mgmt();
        for got in frames {
            self.mgmt_rx += 1;
            self.on_mgmt(&got, now);
        }
    }

    fn on_mgmt(&mut self, got: &Rx, now: u64) {
        let frame = &got.frame[..];
        let me = self.link.mac();

        if dot11::is_beacon_like(frame) {
            if self.state == State::Scanning {
                self.note(frame, got.channel, got.rssi);
            }
            return;
        }
        // Everything below is a frame in a conversation we are having, so it
        // has to be addressed to us and come from the access point we are
        // talking to. Without the second check anybody in the room can end an
        // association by sending a deauthentication with a forged source, and
        // with it they merely have to forge the right one -- which is what
        // 802.11w exists for and this does not have.
        if !dot11::addressed_to(frame, &me) {
            return;
        }
        let from = match dot11::mgmt_addrs(frame) {
            Some((_, sa, _)) => sa,
            None => return,
        };
        let bssid = match &self.target {
            Some(b) => b.bssid,
            None => return,
        };
        if from != bssid {
            return;
        }

        if let Some((sub, reason)) = dot11::parse_reason(frame) {
            // Only about a conversation still going. After `leave` or a failure
            // the target is remembered for the status line, and a stale
            // disassociation from it must not start associating again behind
            // the operator's back -- on a radio that has already let go.
            if !matches!(
                self.state,
                State::Authenticating | State::Associating | State::Handshaking | State::Running
            ) {
                return;
            }
            self.reason = reason;
            self.link.leave();
            self.sup = None;
            // Disassociation keeps the radio's state, since it re-associates
            // with the same access point; deauthentication ends it.
            if sub == 12 {
                self.unprepare();
            }
            // Deauthentication undoes everything; disassociation leaves the
            // station authenticated, so the right answer to it is to associate
            // again rather than to start over. Two frames exist because the
            // two mean different things, and treating them alike throws away
            // the whole authentication for a state that did not need it.
            if sub == 12 {
                self.state = State::Failed("the access point sent a deauthentication");
            } else {
                self.tries = 0;
                self.state = State::Associating;
                self.send_assoc(now);
            }
            return;
        }

        if self.state == State::Authenticating {
            if let Some(a) = dot11::parse_auth(frame) {
                if a.seq_no != 2 {
                    return;
                }
                if a.status != dot11::STATUS_SUCCESS {
                    self.state = State::Failed("the access point refused the authentication");
                    return;
                }
                self.tries = 0;
                self.state = State::Associating;
                self.send_assoc(now);
                return;
            }
        }

        if self.state == State::Associating {
            if let Some(r) = dot11::parse_assoc_resp(frame) {
                if r.status != dot11::STATUS_SUCCESS {
                    self.state = State::Failed("the access point refused the association");
                    return;
                }
                self.aid = r.aid;
                self.link.join(bssid);
                if let Some(t) = self.target.as_ref().map(|b| b.join_target()) {
                    self.link.radio_mut().associated(r.aid, &t);
                }
                let rsn = self.target.as_ref().map(|b| b.rsn).unwrap_or(false);
                if !rsn {
                    // An open network is usable the moment it is associated,
                    // and `secured()` says so afterwards rather than this
                    // pretending otherwise.
                    self.state = State::Running;
                    return;
                }
                let ssid = self.ssid.clone();
                let pass = self.pass.clone();
                self.sup = Some(wpa2::Supplicant::new(&pass, ssid.as_bytes(), bssid, me));
                self.since = now;
                self.state = State::Handshaking;
            }
        }
    }

    fn note(&mut self, frame: &[u8], heard_on: u8, rssi: i8) {
        let b = match dot11::parse_beacon(frame) {
            Some(b) => b,
            None => return,
        };
        // The DS Parameter element is the authority. The channel the radio says
        // it heard the frame on is only right while frames are taken off the
        // part promptly, and a scan moves on; it is the fallback because 5 GHz
        // beacons carry the number in a different element this does not read.
        let channel = b.channel.unwrap_or(heard_on);
        let bss = Bss {
            ssid: b.ssid,
            bssid: b.bssid,
            channel,
            secured: b.secured,
            rsn: b.rsn,
            rssi,
            beacon_int: b.beacon_int,
            dtim: b.dtim,
        };
        // One entry per BSSID: an access point beacons ten times a second and
        // a scan that recorded each one would report a list of duplicates and
        // pick between them by nothing.
        if let Some(old) = self.seen.iter_mut().find(|o| o.bssid == bss.bssid) {
            // A probe response carries no TIM, so it must not erase the DTIM
            // period a beacon from the same access point already gave.
            let dtim = bss.dtim.or(old.dtim);
            *old = bss;
            old.dtim = dtim;
            return;
        }
        self.seen.push(bss);
    }
}

impl<R: Radio> Wlan for Station<R> {
    fn join(&mut self, ssid: &str, pass: &str, now_ms: u64) {
        self.start(ssid, pass, now_ms);
    }

    fn leave_net(&mut self) {
        self.stop();
    }

    fn poll_mlme(&mut self, now_ms: u64) {
        self.poll(now_ms);
    }

    fn status(&self) -> (&'static str, bool) {
        (self.state.name(), self.secured())
    }

    fn networks(&self) -> Vec<crate::net::wifi::Network> {
        let mut v: Vec<crate::net::wifi::Network> =
            self.seen.iter().map(crate::net::wifi::Network::from_bss).collect();
        // Strongest first, which is the order a person wants and also the
        // order `choose` picks in -- a list sorted differently from the
        // decision would have the operator reading one thing and the machine
        // doing another.
        v.sort_by(|a, b| b.rssi.cmp(&a.rssi));
        v
    }

    fn ssid(&self) -> Option<String> {
        if self.state != State::Running {
            return None;
        }
        self.target.as_ref().map(|b| b.ssid.clone())
    }

    fn joined_ap(&self) -> Option<Mac> {
        if self.state != State::Running {
            return None;
        }
        self.target.as_ref().map(|b| b.bssid)
    }

    fn radio_name(&self) -> &'static str {
        self.link.name()
    }
}

impl<R: Radio> Nic for Station<R> {
    fn mac(&self) -> Mac {
        self.link.mac()
    }

    fn link_up(&mut self) -> bool {
        self.state == State::Running
    }

    fn transmit(&mut self, eth: &[u8]) -> bool {
        self.link.transmit(eth)
    }

    fn receive(&mut self) -> Option<Vec<u8>> {
        // Drained and not acted on: EAPOL waits for `poll`, which handles the
        // management frames first. Feeding it here dropped message 1 of every
        // live handshake -- see the claim about the IP stack in `selftest`.
        self.drain();
        if self.rx_q.is_empty() {
            return None;
        }
        Some(self.rx_q.remove(0))
    }

    fn kind(&self) -> Kind {
        Kind::Wireless
    }

    fn wireless(&mut self) -> Option<&mut dyn Wlan> {
        Some(self)
    }
}

// ---------------------------------------------------------------------------
// An access point that exists so the station can be driven
// ---------------------------------------------------------------------------

/// The other end, in as much detail as the station's path requires.
///
/// **Not an access point**, and the difference is worth being plain about: it
/// beacons only when probed, has one station, holds no queues, never rekeys and
/// does nothing on a timer. What it is is the smallest thing that makes every
/// transition in `Station` reachable at boot with no hardware -- the same
/// bargain `mkelf.py` and `mkwad.py` make, and the same reason `Loopback`
/// exists one layer down.
///
/// The parts that matter are the ones it does *not* fake. The authentication
/// and association frames are built by `ieee80211`'s own builders and read by
/// its own parsers; the handshake is `wpa2::Authenticator`, which derives the
/// PTK with the same function the supplicant does; and the data frames after it
/// are opened with `ccmp::Keys`, so a key the station installed wrongly fails
/// here rather than being compared against a copy of itself.
pub struct Ap {
    pub bssid: Mac,
    pub ssid: &'static str,
    pub pass: &'static str,
    pub channel: u8,
    pub rsn: bool,
    sta: Mac,
    seq: u16,
    auth: Option<wpa2::Authenticator>,
    keys: Option<crate::net::ccmp::Keys>,
    /// Ethernet payloads received from the station after the handshake.
    pub got: Vec<Vec<u8>>,
    pub refuse_auth: bool,
    pub refuse_assoc: bool,
    pub authed: bool,
    pub assoced: bool,
}

impl Ap {
    pub fn new(bssid: Mac, sta: Mac, ssid: &'static str, pass: &'static str, channel: u8) -> Ap {
        Ap {
            bssid,
            ssid,
            pass,
            channel,
            rsn: !pass.is_empty(),
            sta,
            seq: 0,
            auth: None,
            keys: None,
            got: Vec::new(),
            refuse_auth: false,
            refuse_assoc: false,
            authed: false,
            assoced: false,
        }
    }

    fn next_seq(&mut self) -> u16 {
        let s = self.seq;
        self.seq = (self.seq + 1) & 0x0FFF;
        s
    }

    /// Take everything the radio transmitted and queue what this would answer.
    ///
    /// Frames sent while the station is tuned elsewhere are **discarded rather
    /// than answered**, which is what makes a scan mean anything: a fake that
    /// replied on every channel would let a station find a network without ever
    /// tuning to it, and the channel plan would be decoration.
    pub fn serve(&mut self, radio: &mut crate::net::softmac::Loopback) {
        let on = radio.channel();
        let sent = core::mem::take(&mut radio.sent);
        let mut out: Vec<Vec<u8>> = Vec::new();
        // A firmware scan probes one channel per turn on its own. The probe is
        // the radio's and not the host's, so it is answered only when the
        // firmware is on this access point's channel -- the same rule a host
        // probe is held to below, which is what keeps the offloaded scan a
        // scan rather than a list handed over.
        if let Some((plan, at)) = radio.fw_scan.as_mut() {
            if let Some(&ch) = plan.get(*at) {
                *at += 1;
                let probe = dot11::probe_request(radio.mac(), "", dot11::BASIC_RATES);
                self.handle(&probe, ch, &mut out);
            }
        }
        for f in sent {
            self.handle(&f, on, &mut out);
        }
        for f in out {
            radio.inbox.push(f);
        }
    }

    /// One frame heard, and whatever this would say back.
    ///
    /// Split out of `serve` so something that is not a `Loopback` can drive an
    /// access point -- `net::rehearsal` puts several of these behind one radio
    /// to make a room. The channel is an argument because that is the whole of
    /// whether this access point can hear the frame at all.
    pub fn handle(&mut self, f: &[u8], on: u8, out: &mut Vec<Vec<u8>>) {
        if on != self.channel {
            return;
        }
        match dot11::mgmt_subtype(f) {
            Some(s) => self.on_mgmt(s, f, out),
            None => self.on_data(f, out),
        }
    }

    /// Let the station address this access point at all.
    ///
    /// A fake built for one station knows who it is talking to at
    /// construction. A room does not: every access point in it has to answer
    /// whichever station probes, so the address is set when one does.
    pub fn talking_to(&mut self, sta: Mac) {
        if self.sta != sta {
            self.sta = sta;
            self.authed = false;
            self.assoced = false;
            self.auth = None;
            self.keys = None;
        }
    }

    fn on_mgmt(&mut self, sub: u8, f: &[u8], out: &mut Vec<Vec<u8>>) {
        match sub {
            // Probe request. A wildcard one is answered; a named one only if
            // the name is ours.
            4 => {
                let asked = dot11::elements(&f[dot11::MGMT_HDR..])
                    .into_iter()
                    .find(|ie| ie.id == 0)
                    .map(|ie| ie.data.to_vec())
                    .unwrap_or_default();
                if !asked.is_empty() && asked != self.ssid.as_bytes() {
                    return;
                }
                let seq = self.next_seq();
                out.push(dot11::probe_response(
                    &self.sta,
                    &self.bssid,
                    seq,
                    self.ssid,
                    self.channel,
                    self.rsn,
                ));
            }
            // Authentication.
            11 => {
                let a = match dot11::parse_auth(f) {
                    Some(a) => a,
                    None => return,
                };
                if a.seq_no != 1 || a.alg != dot11::AUTH_OPEN {
                    return;
                }
                let status = if self.refuse_auth { 1 } else { dot11::STATUS_SUCCESS };
                self.authed = status == dot11::STATUS_SUCCESS;
                let seq = self.next_seq();
                out.push(dot11::auth_response(&self.sta, &self.bssid, seq, status));
            }
            // Association request.
            0 => {
                if !self.authed {
                    return;
                }
                let status = if self.refuse_assoc { 30 } else { dot11::STATUS_SUCCESS };
                let seq = self.next_seq();
                out.push(dot11::assoc_response(&self.sta, &self.bssid, seq, status, 7));
                if status != dot11::STATUS_SUCCESS {
                    return;
                }
                self.assoced = true;
                if !self.rsn {
                    return;
                }
                let mut a =
                    wpa2::Authenticator::new(self.pass, self.ssid.as_bytes(), self.bssid, self.sta);
                let m1 = a.message1();
                self.auth = Some(a);
                let down = self.eapol_down(&m1);
                out.push(down);
            }
            // Deauthentication or disassociation from the station.
            12 | 10 => {
                self.authed = false;
                self.assoced = false;
                self.keys = None;
                self.auth = None;
            }
            _ => {}
        }
    }

    fn on_data(&mut self, f: &[u8], out: &mut Vec<Vec<u8>>) {
        use crate::net::ccmp;
        let protected = f.len() >= 2 && u16::from_le_bytes([f[0], f[1]]) & 0x4000 != 0;
        let body = if protected {
            match self.keys.as_mut().and_then(|k| k.unprotect(f)) {
                Some((_, b)) => b,
                None => return,
            }
        } else {
            match ccmp::parse(f) {
                Some(p) => f[p.hdr_len..].to_vec(),
                None => return,
            }
        };
        let (et, payload) = match dot11::snap_unwrap(&body) {
            Some(v) => v,
            None => return,
        };
        if et != ETHERTYPE_EAPOL {
            self.got.push(payload.to_vec());
            return;
        }
        let reply = match self.auth.as_mut() {
            Some(a) => a.on_frame(payload),
            None => return,
        };
        if let Some(r) = reply {
            let down = self.eapol_down(&r);
            out.push(down);
        }
        // Message 4 finishes it, and the key goes in here for the same reason
        // it goes in there: not before the exchange that establishes it is over.
        let done = self.auth.as_ref().map(|a| a.done).unwrap_or(false);
        if done && self.keys.is_none() {
            if let Some(tk) = self.auth.as_ref().and_then(|a| a.tk()) {
                self.keys = ccmp::Keys::new(&tk, 0);
            }
        }
    }

    fn eapol_down(&mut self, payload: &[u8]) -> Vec<u8> {
        let seq = self.next_seq();
        let body = dot11::snap_wrap(ETHERTYPE_EAPOL, payload);
        dot11::data_from_ds(&self.sta, &self.bssid, &self.bssid, seq, &body)
    }

    /// Send an ordinary frame down to the station, protected once there is a
    /// key. The station refuses an unprotected one at that point, which is the
    /// property being checked.
    pub fn send_down(&mut self, ethertype: u16, payload: &[u8], radio: &mut crate::net::softmac::Loopback) {
        let seq = self.next_seq();
        let body = dot11::snap_wrap(ethertype, payload);
        let f = dot11::data_from_ds(&self.sta, &self.bssid, &self.bssid, seq, &body);
        let f = match self.keys.as_mut() {
            Some(k) => match k.protect(&f) {
                Some(p) => p,
                None => return,
            },
            None => f,
        };
        radio.inbox.push(f);
    }

    pub fn deauth(&mut self, reason: u16, radio: &mut crate::net::softmac::Loopback) {
        let seq = self.next_seq();
        let f = dot11::deauth(&self.sta, &self.bssid, seq, reason);
        // Addressed to the station, sent by the access point: the reverse of
        // the direction `dot11::deauth` is usually called in, and the argument
        // order is what says so.
        radio.inbox.push(f);
        self.authed = false;
        self.assoced = false;
    }
}

pub fn selftest() -> bool {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED};
    use crate::net::softmac::Loopback;

    let mut ok = true;
    let mut check = |what: &str, pass: bool| {
        console::set_color(if pass { LTGREEN } else { LTRED });
        crate::kprintln!("  {}  {}", if pass { "ok  " } else { "FAIL" }, what);
        console::set_color(LTGRAY);
        ok &= pass;
    };

    let me: Mac = [0x02, 0, 0, 0, 0, 0x11];
    let ap_mac: Mac = [0x02, 0, 0, 0, 0, 0xAA];

    // --- the frames, built and read back ------------------------------
    let f = dot11::auth_open(&ap_mac, &me, 5, 1);
    let a = dot11::parse_auth(&f);
    check(
        "an authentication frame says Open System, transaction one, no error",
        a.as_ref()
            .map(|a| a.alg == dot11::AUTH_OPEN && a.seq_no == 1 && a.status == 0)
            .unwrap_or(false),
    );
    check(
        "and it is addressed to the access point, from us, on its BSSID",
        dot11::mgmt_addrs(&f) == Some((ap_mac, me, ap_mac)),
    );
    // The transaction sequence number and the frame's sequence control are two
    // different counters with nearly the same name, sitting nine bytes apart.
    check(
        "the frame's own sequence control is not the transaction number",
        u16::from_le_bytes([f[22], f[23]]) >> 4 == 5 && dot11::parse_auth(&f).unwrap().seq_no == 1,
    );

    let r = dot11::auth_response(&me, &ap_mac, 1, 0);
    check(
        "the answer comes back the other way round",
        dot11::mgmt_addrs(&r) == Some((me, ap_mac, ap_mac))
            && dot11::parse_auth(&r).map(|a| a.seq_no) == Some(2),
    );

    let req = dot11::assoc_request(&ap_mac, &me, 6, "glados", dot11::BASIC_RATES, true);
    let ies = dot11::elements(&req[dot11::MGMT_HDR + 4..]);
    check(
        "an association request carries the SSID, the rates and the RSN choice",
        ies.iter().any(|i| i.id == 0 && i.data == b"glados")
            && ies.iter().any(|i| i.id == 1)
            && ies.iter().any(|i| i.id == 48),
    );
    // The choice is CCMP and only CCMP: an access point picks from what the
    // station offers, so offering TKIP is volunteering for the downgrade.
    check(
        "and the only pairwise cipher it offers is CCMP",
        dot11::RSN_CCMP_PSK[10..14] == [0x00, 0x0F, 0xAC, 0x04]
            && dot11::RSN_CCMP_PSK[8..10] == [0x01, 0x00],
    );
    check(
        "an open network's request sets no privacy bit and no RSN element",
        {
            let o = dot11::assoc_request(&ap_mac, &me, 6, "glados", dot11::BASIC_RATES, false);
            u16::from_le_bytes([o[dot11::MGMT_HDR], o[dot11::MGMT_HDR + 1]]) & (1 << 4) == 0
                && !dot11::elements(&o[dot11::MGMT_HDR + 4..]).iter().any(|i| i.id == 48)
        },
    );

    let resp = dot11::assoc_response(&me, &ap_mac, 2, 0, 7);
    check(
        "an association identifier is fourteen bits, not sixteen",
        dot11::parse_assoc_resp(&resp).map(|r| (r.status, r.aid)) == Some((0, 7))
            && resp[dot11::MGMT_HDR + 5] & 0xC0 == 0xC0,
    );

    let d = dot11::deauth(&ap_mac, &me, 3, dot11::REASON_LEAVING);
    check(
        "a deauthentication and a disassociation are told apart, with reasons",
        dot11::parse_reason(&d) == Some((12, dot11::REASON_LEAVING))
            && dot11::parse_reason(&dot11::disassoc(&ap_mac, &me, 3, 8)) == Some((10, 8))
            && dot11::parse_reason(&req).is_none(),
    );
    check(
        "a beacon goes to everybody and a probe response to whoever asked",
        {
            let b = dot11::beacon(&ap_mac, 0, "glados", 6, true);
            let pr = dot11::probe_response(&me, &ap_mac, 0, "glados", 6, true);
            // One parser reads both, which is what lets a scan use whichever
            // arrives first -- so what has to differ is the addressing, and
            // what must not is anything the parser reads.
            let (bd, _, _) = dot11::mgmt_addrs(&b).unwrap();
            let (pd, _, _) = dot11::mgmt_addrs(&pr).unwrap();
            let pb = dot11::parse_beacon(&b).unwrap();
            let pp = dot11::parse_beacon(&pr).unwrap();
            bd == dot11::BROADCAST
                && pd == me
                && pb.ssid == pp.ssid
                && pb.channel == Some(6)
                && pp.channel == Some(6)
                && pb.rsn
                && pp.rsn
        },
    );
    check(
        "a deauthentication to everybody is addressed to us as well",
        dot11::addressed_to(&dot11::deauth(&dot11::BROADCAST, &ap_mac, 0, 1), &me)
            && !dot11::addressed_to(&dot11::deauth(&ap_mac, &ap_mac, 0, 1), &me),
    );
    check(
        "the two data directions put the same three addresses in different orders",
        {
            let up = dot11::data_to_ds(&ap_mac, &me, &ap_mac, 0, b"x");
            let down = dot11::data_from_ds(&me, &ap_mac, &ap_mac, 0, b"x");
            dot11::data_addrs(&up) == Some((ap_mac, me))
                && dot11::data_addrs(&down) == Some((me, ap_mac))
                && up[4..10] != down[4..10]
        },
    );

    // --- the key wrap the handshake needs -----------------------------
    check(
        "RFC 3394 wrap and unwrap are inverses, which the group key rests on",
        {
            let kek = [0x0Au8; 16];
            let gtk = [0x47u8; 16];
            crate::crypto::aes::key_wrap(&kek, &gtk)
                .and_then(|w| crate::crypto::aes::key_unwrap(&kek, &w))
                .as_deref()
                == Some(&gtk[..])
        },
    );

    // --- the whole path, with no hardware -----------------------------
    let mut sta = Station::new(Loopback::new(me));
    let mut ap = Ap::new(ap_mac, me, "glados", "correct horse", 6);
    let mut now = 0u64;
    let mut steps = 0;
    sta.start("glados", "correct horse", now);
    while steps < 200 && !matches!(sta.state(), State::Running | State::Failed(_)) {
        ap.serve(sta.link_mut().radio_mut());
        now += DWELL_MS;
        sta.poll(now);
        steps += 1;
    }

    check("the scan found the network", sta.seen.iter().any(|b| b.ssid == "glados"));
    check(
        "and it tuned to the channel the beacon named",
        sta.target().map(|b| b.channel) == Some(6)
            && sta.link_mut().radio_mut().channel() == 6,
    );
    check("it authenticated and associated", ap.authed && ap.assoced && sta.aid == 7);
    check(
        "the four-way handshake finished and the link is encrypted",
        sta.state() == State::Running && sta.secured(),
    );
    check(
        "and both ends derived the same temporal key",
        {
            let mut eth: Vec<u8> = Vec::new();
            eth.extend_from_slice(&ap_mac);
            eth.extend_from_slice(&me);
            eth.extend_from_slice(&0x0800u16.to_be_bytes());
            eth.extend_from_slice(b"through the whole path");
            let sent = sta.transmit(&eth);
            ap.serve(sta.link_mut().radio_mut());
            sent && ap.got.last().map(|p| &p[..]) == Some(&b"through the whole path"[..])
        },
    );
    check(
        "the group key came out of message three, wrapped under the KEK",
        sta.sup.as_ref().and_then(|s| s.gtk.as_ref()).map(|g| g.len()) == Some(16),
    );
    check(
        "a frame from the access point comes back up as Ethernet",
        {
            ap.send_down(0x0800, b"and back down", sta.link_mut().radio_mut());
            sta.receive().map(|e| e[14..].to_vec()) == Some(b"and back down".to_vec())
        },
    );

    // Leaving says so. An access point holds an association until it is told,
    // so a station that simply stops transmitting leaves a stale one behind
    // with an association identifier still spent on it.
    check(
        "stopping says goodbye rather than going quiet, and lands back at idle",
        {
            sta.stop();
            ap.serve(sta.link_mut().radio_mut());
            sta.state() == State::Idle
                && sta.state().name() == "idle"
                && !ap.assoced
                && !sta.secured()
        },
    );

    // Re-associating afterwards has to work, or `stop` is a one-way door and
    // a network dropped at three in the morning is a machine off the network
    // until somebody reboots it.
    sta.start("glados", "correct horse", now);
    let mut steps = 0;
    while steps < 200 && !matches!(sta.state(), State::Running | State::Failed(_)) {
        ap.serve(sta.link_mut().radio_mut());
        now += DWELL_MS;
        sta.poll(now);
        steps += 1;
    }
    check(
        "and it can join again afterwards, with a key of its own",
        sta.state() == State::Running && sta.secured(),
    );

    // --- with the IP stack reading the interface in between ----------
    //
    // On a live machine `receive` is called by the IP stack on its own
    // schedule, not only `poll`, and it drains the radio too. The association
    // response and message 1 arrive together, so a `receive` that fed EAPOL
    // as it drained handed message 1 to a supplicant that did not exist yet --
    // the ordering `poll` documents, broken from the other entry point. Every
    // live join of a WPA2 network failed with "the handshake did not finish"
    // while this suite passed, because nothing here called `receive` mid-join.
    let mut sta = Station::new(Loopback::new(me));
    let mut ap = Ap::new(ap_mac, me, "glados", "correct horse", 6);
    let mut now = 0u64;
    sta.start("glados", "correct horse", now);
    for _ in 0..200 {
        if matches!(sta.state(), State::Running | State::Failed(_)) {
            break;
        }
        ap.serve(sta.link_mut().radio_mut());
        let _ = Nic::receive(&mut sta);
        now += DWELL_MS;
        sta.poll(now);
    }
    check(
        "the handshake survives the IP stack draining the interface between polls",
        sta.state() == State::Running && sta.secured(),
    );

    // --- the same path through a firmware-assisted part ---------------
    //
    // A radio that scans for itself and refuses every frame until it has been
    // told which access point it is for. The station has to reach Running
    // through the hooks alone; a station that ignored them sends its probes and
    // its authentication into a part that drops them, and `refused` counts each.
    // Scoped, so the stations below are not these: a claim further down
    // reused `sta` and tested a failed one for a long time without saying so.
    {
        let mut lb = Loopback::new(me);
        lb.offload = true;
        let mut sta = Station::new(lb);
        let mut ap = Ap::new(ap_mac, me, "glados", "correct horse", 6);
        let mut now = 0u64;
        sta.start("glados", "correct horse", now);
        for _ in 0..200 {
            if matches!(sta.state(), State::Running | State::Failed(_)) {
                break;
            }
            ap.serve(sta.link_mut().radio_mut());
            now += DWELL_MS;
            sta.poll(now);
        }
        check(
            "a part that scans in firmware is asked to, and gets on without a frame refused",
            sta.state() == State::Running
                && sta.secured()
                && sta.seen.iter().any(|b| b.ssid == "glados")
                && sta.link_mut().radio_mut().refused == 0,
        );
        check(
            "it was told the access point, its channel and its beacon interval before anything was sent",
            sta.link_mut().radio_mut().prepared
                == Some(JoinTarget { bssid: ap_mac, channel: 6, beacon_int: 100, dtim: None }),
        );
        check(
            "and the association identifier once there was one",
            sta.link_mut().radio_mut().assoc_aid == Some(7),
        );
        sta.stop();
        ap.serve(sta.link_mut().radio_mut());
        check(
            "stopping takes the radio's state down once, after the goodbye went out",
            sta.link_mut().radio_mut().left_count == 1
                && sta.link_mut().radio_mut().prepared.is_none()
                && !ap.assoced,
        );

        // A firmware scan that never reports finishing has to end anyway, and end
        // as a failure that says whose scan it was.
        let mut lb = Loopback::new(me);
        lb.offload = true;
        let mut sta = Station::new(lb);
        sta.start("glados", "", 0);
        let mut now = 0u64;
        for _ in 0..400 {
            now += DWELL_MS;
            if matches!(sta.poll(now), State::Failed(_)) {
                break;
            }
        }
        check(
            "a firmware scan that never finishes fails, naming the radio's scan, and is called off",
            sta.state() == State::Failed("the radio's own scan never reported finishing")
            && sta.link_mut().radio_mut().aborts == 1
            && sta.link_mut().radio_mut().fw_scan.is_none(),
        );

        // Leaving while the firmware scans calls the scan off: a part left hopping
    // channels refuses the next scan and drifts under whatever comes after.
    {
        let mut lb = Loopback::new(me);
        lb.offload = true;
        let mut sta = Station::new(lb);
        sta.start("glados", "", 0);
        sta.poll(DWELL_MS);
        sta.stop();
        sta.stop();
        check(
            "stopping during a firmware scan calls it off, once",
            sta.link_mut().radio_mut().aborts == 1 && sta.link_mut().radio_mut().fw_scan.is_none(),
        );
    }

    // A join that fails after the radio was prepared must not leave it prepared:
        // that is a firmware time event pinning the part to one channel.
        let mut lb = Loopback::new(me);
        lb.offload = true;
        let mut sta = Station::new(lb);
        let mut ap = Ap::new(ap_mac, me, "glados", "correct horse", 6);
        ap.refuse_assoc = true;
        sta.start("glados", "correct horse", 0);
        let mut now = 0u64;
        for _ in 0..200 {
            ap.serve(sta.link_mut().radio_mut());
            now += DWELL_MS;
            if matches!(sta.poll(now), State::Failed(_)) {
                break;
            }
        }
        check(
            "a refused association takes the radio's state down with it",
            sta.state() == State::Failed("the access point refused the association")
                && sta.link_mut().radio_mut().prepared.is_none()
                && sta.link_mut().radio_mut().left_count == 1,
        );
    }

    // A wrong passphrase on a part that keeps station state: the handshake times
    // out with the access point still holding an association, and the goodbye
    // has to leave *before* the radio is told it has no access point -- a part
    // told that refuses to send to one, and the deauthentication was sent after.
    {
        let mut lb = Loopback::new(me);
        lb.offload = true;
        let mut sta = Station::new(lb);
        let mut ap = Ap::new(ap_mac, me, "glados", "correct horse", 6);
        sta.start("glados", "wrong horse", 0);
        let mut now = 0u64;
        let mut held = false;
        for _ in 0..400 {
            ap.serve(sta.link_mut().radio_mut());
            held |= ap.assoced;
            now += DWELL_MS;
            if matches!(sta.poll(now), State::Failed(_)) {
                break;
            }
        }
        ap.serve(sta.link_mut().radio_mut());
        check(
            "a handshake that fails on such a part still says goodbye, then lets the radio go",
            held
                && sta.state() == State::Failed("the handshake did not finish")
                && !ap.assoced
                && sta.link_mut().radio_mut().prepared.is_none()
                && sta.link_mut().radio_mut().left_count == 1,
        );
    }

    // The DTIM period comes from the TIM element, which only a beacon carries,
    // and a probe response heard afterwards must not erase it.
    {
        let mut b = dot11::beacon(&ap_mac, 0, "glados", 6, true);
        b.extend_from_slice(&[5, 4, 0, 3, 0, 0]);
        let pr = dot11::probe_response(&me, &ap_mac, 0, "glados", 6, true);
        let pb = dot11::parse_beacon(&b);
        let pp = dot11::parse_beacon(&pr);
        let mut sta = Station::new(Loopback::new(me));
        sta.note(&b, 6, -40);
        sta.note(&pr, 6, -41);
        check(
            "a beacon's DTIM period is read, and a later probe response does not erase it",
            pb.as_ref().map(|x| (x.beacon_int, x.dtim)) == Some((100, Some(3)))
                && pp.as_ref().map(|x| x.dtim) == Some(None)
                && sta.seen.len() == 1
                && sta.seen[0].dtim == Some(3)
                && sta.seen[0].rssi == -41,
        );
    }

    // --- and through the trait objects the interface layer holds -------
    //
    // `wlan0` holds a driver behind `dyn Nic` and asks it for `wireless()`, so
    // a `Station` that works as a concrete type and is unreachable through the
    // vtable would be a stack nothing can drive. Driven here through
    // `&mut dyn Nic`, which carries the same vtable a `Box` does and differs
    // only in who owns the value -- and the fixture has to keep owning it,
    // because the access point reaches the radio underneath.
    let mut sta2 = Station::new(Loopback::new(me));
    let mut ap2 = Ap::new(ap_mac, me, "glados", "correct horse", 6);
    let mut t = 0u64;
    {
        let n: &mut dyn Nic = &mut sta2;
        if let Some(w) = n.wireless() {
            w.join("glados", "correct horse", t);
        }
    }
    let mut running = false;
    for _ in 0..200 {
        ap2.serve(sta2.link_mut().radio_mut());
        t += DWELL_MS;
        let n: &mut dyn Nic = &mut sta2;
        match n.wireless() {
            Some(w) => {
                w.poll_mlme(t);
                if w.status().0 == "running" {
                    running = true;
                }
            }
            None => break,
        }
        if running {
            break;
        }
    }
    {
        let n: &mut dyn Nic = &mut sta2;
        let kind = n.kind();
        let (state, secure) = n.wireless().map(|w| w.status()).unwrap_or(("none", false));
        check(
            "a driver behind the vtable answers the wireless half and gets on",
            running && state == "running" && secure && kind == Kind::Wireless,
        );
    }
    {
        let n: &mut dyn Nic = &mut sta2;
        let list = n.wireless().map(|w| w.networks()).unwrap_or_default();
        check(
            "and it reports what the scan heard, as a name and a signal",
            list.iter().any(|x| x.ssid == "glados" && x.secured && x.rssi < 0),
        );
    }
    // A wired card answers None to the same question, which is the whole
    // reason the method has a default rather than the interface layer keeping
    // a second handle and a flag saying which one is real.
    let mut wired = crate::net::iface::Loopback::new();
    {
        let n: &mut dyn Nic = &mut wired;
        let kind = n.kind();
        check(
            "an interface that is not wireless says so rather than pretending",
            n.wireless().is_none() && kind != Kind::Wireless,
        );
    }

    // --- and installed as wlan0, which is all a driver has to do -------
    //
    // `attach_radio` is the whole interface between a wireless driver and this
    // stack, so a claim that it works is a claim that writing `impl Radio` is
    // sufficient. Run only on a machine whose `wlan0` is empty, which is every
    // machine so far: clobbering a real driver to test the thing that installs
    // drivers would be the suite breaking what it checks.
    if crate::net::ifaces()[crate::net::WLAN0].nic.is_none() {
        let mut fullmac = Loopback::new(me);
        fullmac.softmac = false;
        check(
            "a FullMAC part is refused here, because its firmware ran the MLME",
            !crate::net::attach_radio(fullmac)
                && crate::net::ifaces()[crate::net::WLAN0].nic.is_none(),
        );

        let attached = crate::net::attach_radio(Loopback::new(me));
        check(
            "and a SoftMAC part becomes wlan0, reachable as wireless",
            attached
                && crate::net::wlan().is_some()
                && crate::net::ifaces()[crate::net::WLAN0]
                    .nic
                    .as_ref()
                    .map(|n| n.kind() == Kind::Wireless)
                    .unwrap_or(false),
        );
        // The clock the real path uses, rather than the fixture's counter.
        //
        // **The units are the claim, not that it moves.** Every deadline here
        // is a number of milliseconds, and `now_ms` is a division -- so a
        // clock answering microseconds makes `DWELL_MS` a hundred and twenty
        // *microseconds*, the scan blows through thirty-eight channels before
        // one beacon interval has elapsed, and every network in the building
        // is missed. On hardware that reads as a radio that hears nothing,
        // which is the most expensive symptom to debug and the cheapest thing
        // to check: sleep a known time and see what the clock says.
        let t0 = crate::net::now_ms();
        crate::time::delay_us(50_000);
        let moved = crate::net::now_ms().saturating_sub(t0);
        check(
            "a 50 ms wait reads as about 50 ms, so the deadlines are in the unit they say",
            (40..=70).contains(&moved),
        );
        if !(40..=70).contains(&moved) {
            crate::kprintln!("        50 ms of delay_us read as {} ms", moved);
        }
        crate::net::wifi_service();
        check(
            "and a poll through the interface leaves the station where it was",
            crate::net::wlan().map(|w| w.status().0) == Some("idle"),
        );

        // Put it back. `wlan0` is empty on a machine with no wireless driver
        // and that is exactly what it was a moment ago, so this restores the
        // interface table rather than approximating it.
        let w = &mut crate::net::ifaces()[crate::net::WLAN0];
        w.nic = None;
        w.up = false;
    }

    // --- and every way it goes wrong ----------------------------------
    // A deauthentication ends it. Unauthenticated, because 802.11w is not here,
    // which is the honest state of nearly every deployed network.
    ap.deauth(7, sta.link_mut().radio_mut());
    sta.poll(now);
    check(
        "a deauthentication ends the association and carries its reason",
        matches!(sta.state(), State::Failed(_)) && sta.reason == 7 && !sta.secured(),
    );

    let mut sta = Station::new(Loopback::new(me));
    let mut ap = Ap::new(ap_mac, me, "glados", "correct horse", 6);
    ap.refuse_auth = true;
    sta.start("glados", "correct horse", 0);
    let mut now = 0u64;
    for _ in 0..200 {
        ap.serve(sta.link_mut().radio_mut());
        now += DWELL_MS;
        if matches!(sta.poll(now), State::Failed(_)) {
            break;
        }
    }
    check(
        "an access point that refuses the authentication is not retried at",
        sta.state() == State::Failed("the access point refused the authentication"),
    );

    let mut sta = Station::new(Loopback::new(me));
    let mut ap = Ap::new(ap_mac, me, "glados", "correct horse", 6);
    ap.refuse_assoc = true;
    sta.start("glados", "correct horse", 0);
    let mut now = 0u64;
    for _ in 0..200 {
        ap.serve(sta.link_mut().radio_mut());
        now += DWELL_MS;
        if matches!(sta.poll(now), State::Failed(_)) {
            break;
        }
    }
    check(
        "and neither is one that refuses the association",
        sta.state() == State::Failed("the access point refused the association"),
    );

    // A wrong passphrase is the interesting failure, because the access point
    // never says so: message 3's MIC simply does not verify, the supplicant
    // answers nothing, and what the station sees is silence.
    let mut sta = Station::new(Loopback::new(me));
    let mut ap = Ap::new(ap_mac, me, "glados", "correct horse", 6);
    sta.start("glados", "wrong horse", 0);
    let mut now = 0u64;
    for _ in 0..400 {
        ap.serve(sta.link_mut().radio_mut());
        now += DWELL_MS;
        if matches!(sta.poll(now), State::Failed(_)) {
            break;
        }
    }
    check(
        "a wrong passphrase fails in the handshake and never installs a key",
        sta.state() == State::Failed("the handshake did not finish") && !sta.secured(),
    );

    // Nothing answers at all: the scan finds nothing and says which half failed.
    let mut sta = Station::new(Loopback::new(me));
    sta.start("glados", "", 0);
    let mut now = 0u64;
    for _ in 0..200 {
        now += DWELL_MS;
        if matches!(sta.poll(now), State::Failed(_)) {
            break;
        }
    }
    check(
        "an empty room fails in the scan, naming the scan",
        sta.state() == State::Failed("no access point is carrying that network"),
    );

    // An encrypted network and no passphrase is refused before anything is
    // sent, rather than associating and then finding out.
    let mut sta = Station::new(Loopback::new(me));
    let mut ap = Ap::new(ap_mac, me, "glados", "correct horse", 6);
    sta.start("glados", "", 0);
    let mut now = 0u64;
    for _ in 0..200 {
        ap.serve(sta.link_mut().radio_mut());
        now += DWELL_MS;
        if matches!(sta.poll(now), State::Failed(_)) {
            break;
        }
    }
    check(
        "an encrypted network with no passphrase is refused before authenticating",
        sta.state() == State::Failed("that network is encrypted and no passphrase was given")
            && !ap.authed,
    );

    // An open network needs no handshake and is running the moment it is
    // associated -- and says plainly that it is not secured.
    let mut sta = Station::new(Loopback::new(me));
    let mut ap = Ap::new(ap_mac, me, "cafe", "", 11);
    sta.start("cafe", "", 0);
    let mut now = 0u64;
    for _ in 0..200 {
        ap.serve(sta.link_mut().radio_mut());
        now += DWELL_MS;
        if matches!(sta.poll(now), State::Running | State::Failed(_)) {
            break;
        }
    }
    check(
        "an open network associates with no handshake and reports itself unsecured",
        sta.state() == State::Running && !sta.secured() && ap.assoced,
    );

    ok
}
