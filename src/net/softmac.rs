//! Ethernet on one side, 802.11 on the other, for every part that needs it.
//!
//! A `Radio` moves 802.11 frames. The rest of the kernel speaks Ethernet. This
//! is the layer between them, and it is shared: encapsulation, sequence
//! numbers and CCMP are the same on every SoftMAC chip in the world, so they
//! are written once here rather than once per driver.
//!
//! What a driver is left with is a register file and an endpoint.
//!
//! ### The three things it does, and why each is here rather than there
//!
//! **Encapsulation.** An Ethernet frame is destination, source, type, payload.
//! An 802.11 data frame puts the addresses in its own header -- in an order
//! that depends on which way the frame is going -- and carries the type in an
//! LLC/SNAP prefix. So a frame changes shape in both directions and the
//! addresses have to be put back where Ethernet expects them.
//!
//! **Sequence numbers.** Twelve bits, incremented per frame, wrapping. They are
//! not covered by CCMP's authentication (`ccmp::aad` masks them, and says why),
//! so they are the transmitter's own bookkeeping and belong to whoever builds
//! the frame -- which is here.
//!
//! **Encryption, when the chip does not.** `Caps::hw_ccmp` decides, and a part
//! that answers true is trusted with it. Everything else goes through
//! `ccmp::Keys`, which brings the replay counter with it.
//!
//! ### Before the handshake there is no key, and that is a state
//!
//! A link with no keys sends in the clear. That is correct -- it is how the
//! authentication and association frames get out, and how EAPOL reaches the
//! access point before there is anything to encrypt with -- and it is also
//! exactly the state an attacker would like the link left in. So `secured`
//! reports it, `transmit` refuses a data frame while unkeyed unless it is
//! EAPOL, and a received data frame that is *not* protected is dropped once
//! keys exist. A network that encrypts in one direction only is a network that
//! does not encrypt.

use alloc::vec::Vec;

use crate::dev::radio::{Caps, Radio, Rx};
use crate::net::ccmp;
use crate::net::iface::{Kind, Nic};
use crate::net::ieee80211 as dot11;
use crate::net::Mac;

/// EAPOL, which must cross an unencrypted link because it is what makes the
/// link encrypted.
pub const ETHERTYPE_EAPOL: u16 = 0x888E;

/// How many management frames may pile up before the oldest is dropped.
///
/// Bounded because a machine parked next to a busy access point hears beacons
/// forever, and an unattended one would grow this queue until the heap gave
/// out. Small because the MLME drains it every poll and anything older than a
/// poll is stale by definition.
// As many as one `receive` can take, so a burst of beacons -- which is what a
// firmware scan delivers -- cannot evict its own head before the MLME reads it.
pub const MGMT_QUEUE: usize = 64;

pub struct Link<R: Radio> {
    radio: R,
    bssid: Mac,
    keys: Option<ccmp::Keys>,
    /// The group key, for frames addressed to more than one station. Its own
    /// replay counter: the access point numbers group frames separately.
    group: Option<ccmp::Keys>,
    seq: u16,
    joined: bool,
    /// Management frames seen while draining the radio for data, kept whole
    /// with what the radio reported about them -- the signal strength is how a
    /// scan picks between two access points carrying the same network, and it
    /// exists nowhere but here.
    mgmt: Vec<Rx>,
    /// Counted rather than logged: a frame dropped for one of these reasons is
    /// ordinary in ones and a fault in thousands, and a line each would bury
    /// the shell under a busy access point.
    pub dropped_unprotected: u32,
    pub dropped_replay: u32,
    pub dropped_malformed: u32,
    /// Our own frames, reflected back by the access point.
    pub dropped_own: u32,
}

impl<R: Radio> Link<R> {
    pub fn new(radio: R) -> Link<R> {
        Link {
            radio,
            bssid: [0; 6],
            keys: None,
            group: None,
            seq: 0,
            joined: false,
            mgmt: Vec::new(),
            dropped_unprotected: 0,
            dropped_replay: 0,
            dropped_malformed: 0,
            dropped_own: 0,
        }
    }

    pub fn caps(&self) -> Caps {
        self.radio.caps()
    }

    pub fn radio_mut(&mut self) -> &mut R {
        &mut self.radio
    }

    /// Associated with this BSS, but not yet keyed.
    pub fn join(&mut self, bssid: Mac) {
        self.bssid = bssid;
        self.joined = true;
        self.keys = None;
        self.group = None;
    }

    /// The group key from the handshake or a group rekey, under its key id.
    /// Offered to the radio first, as the pairwise key is.
    pub fn keyed_group(&mut self, gtk: &[u8], key_id: u8) -> bool {
        if gtk.len() != 16 {
            return false;
        }
        if self.radio.caps().hw_ccmp {
            let mut k = crate::dev::radio::Key { idx: key_id & 3, tk: [0; 16], pairwise: false, peer: [0xff; 6] };
            k.tk.copy_from_slice(gtk);
            if self.radio.set_key(&k) {
                self.group = None;
                return true;
            }
        }
        self.group = ccmp::Keys::new(gtk, key_id);
        self.group.is_some()
    }

    /// The four-way handshake finished and produced a temporal key.
    ///
    /// Offered to the radio first. A part that takes it encrypts in hardware
    /// and this keeps no software key at all -- which also means no software
    /// replay counter, so a part that claims the key is claiming that too.
    pub fn keyed(&mut self, tk: &[u8], key_id: u8) -> bool {
        if tk.len() != 16 {
            return false;
        }
        if self.radio.caps().hw_ccmp {
            let mut k = crate::dev::radio::Key {
                idx: key_id,
                tk: [0; 16],
                pairwise: true,
                peer: self.bssid,
            };
            k.tk.copy_from_slice(tk);
            if self.radio.set_key(&k) {
                self.keys = None;
                return true;
            }
            // Said it could and then would not. Fall through to software
            // rather than leaving the link in the clear believing otherwise.
        }
        self.keys = ccmp::Keys::new(tk, key_id);
        self.keys.is_some()
    }

    pub fn secured(&self) -> bool {
        self.keys.is_some() || (self.joined && self.radio.caps().hw_ccmp)
    }

    pub fn leave(&mut self) {
        self.joined = false;
        self.keys = None;
        self.group = None;
        self.bssid = [0; 6];
    }

    /// The next sequence number, for whoever is building the frame.
    ///
    /// Public because management frames need one too and the MLME builds
    /// those. **One counter for both**, which is what the standard says and
    /// also the only arrangement that cannot produce two frames a moment apart
    /// carrying the same number -- a duplicate as far as the receiver is
    /// concerned, and silently discarded by it.
    pub fn next_seq(&mut self) -> u16 {
        let s = self.seq;
        self.seq = (self.seq + 1) & 0x0FFF;
        s
    }

    pub fn bssid(&self) -> Mac {
        self.bssid
    }

    /// What the part underneath calls itself.
    pub fn name(&self) -> &'static str {
        self.radio.name()
    }

    /// Take the management frames seen since the last call.
    ///
    /// **There is one drain on a radio**, and whichever of the two consumers
    /// calls first must not consume what the other needs. So `receive` sets
    /// these aside as it goes rather than the MLME reading the radio itself,
    /// and an ordinary data path that never asks simply lets them expire.
    pub fn take_mgmt(&mut self) -> Vec<Rx> {
        core::mem::take(&mut self.mgmt)
    }

    /// Send one raw 802.11 frame, unencrypted. For management and EAPOL.
    pub fn tx_raw(&mut self, frame: &[u8]) -> bool {
        self.radio.tx(frame).is_ok()
    }

    // There is deliberately no `rx_raw`. A second way to take a frame off the
    // radio is a second consumer racing the first for the same queue, and
    // whichever called would silently eat what the other needed -- which is
    // the whole reason `receive` sets management frames aside instead.
}

impl<R: Radio> Nic for Link<R> {
    fn mac(&self) -> Mac {
        self.radio.mac()
    }

    fn link_up(&mut self) -> bool {
        self.joined
    }

    fn transmit(&mut self, eth: &[u8]) -> bool {
        if eth.len() < 14 || !self.joined {
            return false;
        }
        let mut da = [0u8; 6];
        let mut sa = [0u8; 6];
        da.copy_from_slice(&eth[0..6]);
        sa.copy_from_slice(&eth[6..12]);
        let ethertype = u16::from_be_bytes([eth[12], eth[13]]);

        // Unkeyed, only EAPOL goes out. Everything else waits for the
        // handshake rather than leaving in the clear, which is the failure
        // nobody would see from this end.
        if self.keys.is_none() && !self.radio.caps().hw_ccmp && ethertype != ETHERTYPE_EAPOL {
            return false;
        }

        let body = dot11::snap_wrap(ethertype, &eth[14..]);
        let seq = self.next_seq();
        let frame = dot11::data_to_ds(&self.bssid, &sa, &da, seq, &body);

        match &mut self.keys {
            Some(k) => match k.protect(&frame) {
                Some(p) => self.radio.tx(&p).is_ok(),
                // The packet number is spent. Refusing is the only safe answer
                // -- see `ccmp::Keys::next_pn`.
                None => false,
            },
            None => self.radio.tx(&frame).is_ok(),
        }
    }

    fn receive(&mut self) -> Option<Vec<u8>> {
        // **Bounded, because the loop's exit condition belongs to the driver.**
        // `rx` returning `None` is what ends this, and a part that always has
        // one more frame -- a ring the driver never advances, a fake that
        // manufactures beacons on demand -- turns this into a kernel that
        // stops answering with no fault and no message. That is not
        // hypothetical: the rehearsal radio did exactly it, and what it looked
        // like was a shell that never gave the prompt back.
        //
        // Sixty-four is well past what a real part delivers between polls and
        // far short of a hang; what is left in the queue is taken on the next
        // call, which is the next trip round the idle loop.
        const PER_CALL: usize = 64;
        for _ in 0..PER_CALL {
            let got = self.radio.rx()?;
            if got.frame.len() < 24 {
                self.dropped_malformed += 1;
                continue;
            }
            // A management frame is not data and is not malformed either:
            // authentication, association and both kinds of goodbye all arrive
            // here and belong to the MLME above. Set aside rather than dropped.
            if dot11::mgmt_subtype(&got.frame).is_some() {
                if self.mgmt.len() >= MGMT_QUEUE {
                    self.mgmt.remove(0);
                }
                self.mgmt.push(got);
                continue;
            }
            let frame = got.frame;
            let protected = u16::from_le_bytes([frame[0], frame[1]]) & 0x4000 != 0;

            // Group-addressed frames are under the group key, named by the id
            // in their CCMP header; everything else under the pairwise key.
            let group_addressed = frame.len() >= 10 && frame[4] & 1 != 0;
            let named_id = ccmp::parse(&frame)
                .and_then(|f| frame.get(f.hdr_len + 3))
                .map(|b| b >> 6);
            let (hdr, body) = if protected {
                let key = if group_addressed {
                    self.group.as_mut().filter(|g| Some(g.key_id) == named_id)
                } else {
                    self.keys.as_mut()
                };
                match key {
                    Some(k) => match k.unprotect(&frame) {
                        Some(v) => v,
                        None => {
                            // Either a forgery or a replay, and the counter
                            // knows which. Both are dropped; only the tally
                            // tells them apart afterwards.
                            self.dropped_replay += 1;
                            continue;
                        }
                    },
                    // Protected, and no key to open it with. Not malformed --
                    // a group-addressed frame under a key we were never given
                    // looks exactly like this.
                    None => {
                        self.dropped_unprotected += 1;
                        continue;
                    }
                }
            } else {
                // In the clear. Allowed only while there is no key, and only
                // for EAPOL once there is -- a network that encrypts one way
                // does not encrypt.
                let f = match ccmp::parse(&frame) {
                    Some(f) => f,
                    None => {
                        self.dropped_malformed += 1;
                        continue;
                    }
                };
                if self.keys.is_some() {
                    let is_eapol = dot11::snap_unwrap(&frame[f.hdr_len..])
                        .map(|(t, _)| t == ETHERTYPE_EAPOL)
                        .unwrap_or(false);
                    if !is_eapol {
                        self.dropped_unprotected += 1;
                        continue;
                    }
                }
                (frame[..f.hdr_len].to_vec(), frame[f.hdr_len..].to_vec())
            };

            let (da, sa) = match dot11::data_addrs(&hdr) {
                Some(v) => v,
                None => {
                    self.dropped_malformed += 1;
                    continue;
                }
            };
            // Our own broadcast, relayed back by the access point to everybody
            // including us. Taken up, our own ARP request teaches the stack
            // that our address is somebody else's.
            if sa == self.radio.mac() {
                self.dropped_own += 1;
                continue;
            }
            let (ethertype, payload) = match dot11::snap_unwrap(&body) {
                Some(v) => v,
                None => {
                    self.dropped_malformed += 1;
                    continue;
                }
            };

            let mut eth = Vec::with_capacity(14 + payload.len());
            eth.extend_from_slice(&da);
            eth.extend_from_slice(&sa);
            eth.extend_from_slice(&ethertype.to_be_bytes());
            eth.extend_from_slice(payload);
            return Some(eth);
        }
        None
    }

    fn kind(&self) -> Kind {
        Kind::Wireless
    }
}

/// A radio that transmits into a queue and receives from one.
///
/// **The whole point is that the association path can be driven with no
/// hardware in the machine.** Every real part needs a dongle on the GF63 and
/// QEMU models none of them, so without this the shared layer could be written
/// and never once executed -- which is the state `rtl8188eu`'s descriptors were
/// in for as long as they existed.
///
/// It is the same bargain `mkelf.py` and `mkwad.py` make: build the fixture so
/// the negatives are reachable.
pub struct Loopback {
    mac: Mac,
    ch: u8,
    started: bool,
    /// Frames this radio was asked to send, oldest first.
    pub sent: Vec<Vec<u8>>,
    /// Frames queued for it to receive.
    pub inbox: Vec<Vec<u8>>,
    pub hw_ccmp: bool,
    /// False makes it a FullMAC part: firmware runs the MLME, so such a part
    /// implements `Nic` directly and `softmac` is not in its path at all.
    pub softmac: bool,
    /// A part that claims hardware crypto and then refuses the key, which is
    /// the one lie the interface cannot catch and the fallback it forces.
    pub refuse_key: bool,
    pub keys_taken: u32,
    /// Behave as a firmware-assisted part: scan in "firmware", and refuse to
    /// send any frame until `prepare_join` has named an access point. The
    /// refusal is what makes the fixture a test of the hooks rather than a
    /// second copy of the host-driven path -- an MLME that skipped them sends
    /// its authentication into a part that drops it.
    pub offload: bool,
    /// The firmware scan in progress: the channel plan and how far it has got.
    /// Advanced by whoever plays the air (`mlme::Ap::serve`), one channel per
    /// turn, which is the same pacing the host-driven scan gets.
    pub fw_scan: Option<(Vec<u8>, usize)>,
    /// What `prepare_join` was told, what `associated` was told, and how often
    /// `left` was called. Recorded so the suite can ask.
    pub prepared: Option<crate::dev::radio::JoinTarget>,
    pub assoc_aid: Option<u16>,
    pub left_count: u32,
    /// Frames refused because nothing had been prepared. Zero is the claim.
    pub refused: u32,
    /// Firmware scans called off.
    pub aborts: u32,
}

impl Loopback {
    pub fn new(mac: Mac) -> Loopback {
        Loopback {
            mac,
            ch: 1,
            started: false,
            sent: Vec::new(),
            inbox: Vec::new(),
            hw_ccmp: false,
            softmac: true,
            refuse_key: false,
            keys_taken: 0,
            offload: false,
            fw_scan: None,
            prepared: None,
            assoc_aid: None,
            left_count: 0,
            refused: 0,
            aborts: 0,
        }
    }

    /// Hand back what was last transmitted, as though it had been received.
    pub fn echo(&mut self) {
        if let Some(f) = self.sent.last() {
            self.inbox.push(f.clone());
        }
    }
}

impl Radio for Loopback {
    fn name(&self) -> &'static str {
        "loopback"
    }

    fn caps(&self) -> Caps {
        Caps { softmac: self.softmac, hw_ccmp: self.hw_ccmp, band5: true, max_frame: 2304 }
    }

    fn mac(&self) -> Mac {
        self.mac
    }

    fn start(&mut self) -> Result<(), &'static str> {
        self.started = true;
        Ok(())
    }

    fn stop(&mut self) {
        self.started = false;
    }

    fn set_channel(&mut self, ch: u8) -> Result<(), &'static str> {
        // The plan is the plan, even for a radio that tunes nothing.
        if crate::dev::radio::channel_mhz(ch).is_none() {
            return Err("no such channel");
        }
        self.ch = ch;
        Ok(())
    }

    fn channel(&self) -> u8 {
        self.ch
    }

    fn tx(&mut self, frame: &[u8]) -> Result<(), &'static str> {
        if !self.started {
            return Err("radio is not started");
        }
        if self.offload && self.prepared.is_none() {
            self.refused += 1;
            return Err("the firmware has not been told about an access point");
        }
        self.sent.push(frame.to_vec());
        Ok(())
    }

    fn rx(&mut self) -> Option<Rx> {
        if self.inbox.is_empty() {
            return None;
        }
        let frame = self.inbox.remove(0);
        Some(Rx { frame, rssi: -42, channel: self.ch })
    }

    fn set_key(&mut self, _key: &crate::dev::radio::Key) -> bool {
        if self.refuse_key {
            return false;
        }
        self.keys_taken += 1;
        self.hw_ccmp
    }

    fn scan_offload(&mut self, _ssid: &str, chans: &[u8]) -> Option<Result<(), &'static str>> {
        if !self.offload {
            return None;
        }
        self.fw_scan = Some((chans.to_vec(), 0));
        Some(Ok(()))
    }

    fn scan_done(&mut self) -> bool {
        match &self.fw_scan {
            Some((plan, at)) if *at >= plan.len() => {
                self.fw_scan = None;
                true
            }
            Some(_) => false,
            None => true,
        }
    }

    fn scan_abort(&mut self) {
        if self.fw_scan.take().is_some() {
            self.aborts += 1;
        }
    }

    fn prepare_join(&mut self, t: &crate::dev::radio::JoinTarget) -> Result<(), &'static str> {
        self.set_channel(t.channel)?;
        if self.offload {
            self.prepared = Some(*t);
        }
        Ok(())
    }

    fn associated(&mut self, aid: u16, _t: &crate::dev::radio::JoinTarget) {
        self.assoc_aid = Some(aid);
    }

    fn left(&mut self) {
        self.prepared = None;
        self.assoc_aid = None;
        self.left_count += 1;
    }
}

pub fn selftest() -> bool {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED};
    let mut ok = true;
    let mut check = |what: &str, pass: bool| {
        console::set_color(if pass { LTGREEN } else { LTRED });
        crate::kprintln!("  {}  {}", if pass { "ok  " } else { "FAIL" }, what);
        console::set_color(LTGRAY);
        ok &= pass;
    };

    let me: Mac = [0x02, 0, 0, 0, 0, 0x11];
    let peer: Mac = [0x02, 0, 0, 0, 0, 0x22];
    let bssid: Mac = [0x02, 0, 0, 0, 0, 0xAA];
    let tk = [0x5Au8; 16];

    // An ordinary Ethernet frame: destination, source, IPv4, payload.
    let mut eth: Vec<u8> = Vec::new();
    eth.extend_from_slice(&peer);
    eth.extend_from_slice(&me);
    eth.extend_from_slice(&0x0800u16.to_be_bytes());
    eth.extend_from_slice(b"a packet");

    // --- SNAP and addressing, which are pure --------------------------
    let w = dot11::snap_wrap(0x0800, b"a packet");
    check(
        "SNAP adds eight bytes and gives them back",
        w.len() == eth.len() - 6 && dot11::snap_unwrap(&w) == Some((0x0800, &b"a packet"[..])),
    );
    check(
        "a body that is not SNAP is refused rather than read from byte nine",
        dot11::snap_unwrap(b"not snap at all").is_none() && dot11::snap_unwrap(&w[..4]).is_none(),
    );
    // The half-right bug: A1 is the destination on a frame from the access
    // point and the BSSID on a frame to it, so reading A1 works on everything
    // sent and fails on everything received.
    let to_ap = dot11::data_to_ds(&bssid, &me, &peer, 7, &w);
    check(
        "a frame to the access point puts the BSSID first and the destination third",
        to_ap[4..10] == bssid && to_ap[10..16] == me && to_ap[16..22] == peer,
    );
    check(
        "and its real source and destination come back out",
        dot11::data_addrs(&to_ap) == Some((peer, me)),
    );
    let mut from_ap = to_ap.clone();
    from_ap[1] = 0x02; // FromDS instead of ToDS
    check(
        "a frame from the access point is read with the other layout",
        dot11::data_addrs(&from_ap) == Some((bssid, peer)),
    );
    check(
        "the sequence number lands in the top twelve bits",
        u16::from_le_bytes([to_ap[22], to_ap[23]]) == 7 << 4,
    );

    // --- unkeyed: EAPOL only ------------------------------------------
    let mut link = Link::new(Loopback::new(me));
    let _ = link.radio_mut().start();
    check("a link that has not joined sends nothing", !link.transmit(&eth));
    link.join(bssid);
    check(
        "joined but unkeyed, an ordinary frame is refused rather than sent in the clear",
        !link.transmit(&eth) && link.radio_mut().sent.is_empty(),
    );
    let mut eapol = eth.clone();
    eapol[12..14].copy_from_slice(&ETHERTYPE_EAPOL.to_be_bytes());
    check(
        "but EAPOL goes out, because it is what makes the link encrypted",
        link.transmit(&eapol) && link.radio_mut().sent.len() == 1,
    );
    check("and it is not secured yet", !link.secured());

    // --- keyed: the round trip ----------------------------------------
    check("the handshake's key is accepted", link.keyed(&tk, 0) && link.secured());
    link.radio_mut().sent.clear();
    check("a data frame now goes out", link.transmit(&eth));
    let wire = link.radio_mut().sent[0].clone();
    check(
        "it is protected, and the payload is not on the air",
        wire[1] & 0x40 != 0 && !wire.windows(8).any(|x| x == b"a packet"),
    );
    check(
        "and it is longer than the Ethernet frame by the header, SNAP and MIC",
        wire.len() == 24 + ccmp::HDR_LEN + 8 + (eth.len() - 14) + ccmp::MIC_LEN,
    );

    // Received by a second link with the same key, which is what an access
    // point is from this side of the conversation.
    let mut other = Link::new(Loopback::new(peer));
    let _ = other.radio_mut().start();
    other.join(bssid);
    let _ = other.keyed(&tk, 0);
    other.radio_mut().inbox.push(wire.clone());
    check(
        "the other end gets back exactly the Ethernet frame that went in",
        other.receive().as_deref() == Some(&eth[..]),
    );
    // Genuine, correctly signed, and refused -- the replay counter is carried
    // through the layer rather than left behind in `ccmp`.
    other.radio_mut().inbox.push(wire.clone());
    check(
        "and replaying it is refused, with the drop counted",
        other.receive().is_none() && other.dropped_replay == 1,
    );

    // --- a protected link must not accept plaintext -------------------
    let clear = dot11::data_to_ds(&bssid, &me, &peer, 9, &w);
    other.radio_mut().inbox.push(clear.clone());
    check(
        "an unprotected data frame is dropped once keys exist",
        other.receive().is_none() && other.dropped_unprotected == 1,
    );
    // Except EAPOL, which is how rekeying reaches a link that is already up.
    let ew = dot11::snap_wrap(ETHERTYPE_EAPOL, b"key msg");
    other.radio_mut().inbox.push(dot11::data_to_ds(&bssid, &me, &peer, 10, &ew));
    check(
        "but EAPOL in the clear is still delivered",
        other.receive().map(|e| e[12..14].to_vec()) == Some(ETHERTYPE_EAPOL.to_be_bytes().to_vec()),
    );

    // --- malformed, and the counters that tell the reasons apart ------
    //
    // **Three reasons, and the order they are checked in matters.** A frame in
    // the clear on a keyed link is dropped as *unprotected* before anything
    // looks at its body -- whether the body would have parsed is not the
    // interesting fact about it. So a body that is not SNAP only reaches the
    // malformed count when it arrives properly protected, which is the case
    // this builds on purpose. The first version of this claim expected the
    // plaintext one to count as malformed and it does not, correctly.
    other.radio_mut().inbox.push(alloc::vec![0u8; 10]);
    check(
        "a runt is malformed",
        other.receive().is_none() && other.dropped_malformed == 1,
    );
    let before = other.dropped_unprotected;
    other.radio_mut().inbox.push(dot11::data_to_ds(&bssid, &me, &peer, 11, b"not snap"));
    check(
        "a plaintext frame is unprotected first and never examined further",
        other.receive().is_none()
            && other.dropped_unprotected == before + 1
            && other.dropped_malformed == 1,
    );
    let odd = ccmp::protect(&tk, &dot11::data_to_ds(&bssid, &me, &peer, 12, b"not snap here"), 500, 0);
    other.radio_mut().inbox.push(odd.unwrap_or_default());
    check(
        "and a properly protected body that is not SNAP is malformed",
        other.receive().is_none() && other.dropped_malformed == 2,
    );

    // --- sequence numbers ---------------------------------------------
    link.radio_mut().sent.clear();
    let _ = link.transmit(&eth);
    let _ = link.transmit(&eth);
    let seq_of = |f: &[u8]| u16::from_le_bytes([f[22], f[23]]) >> 4;
    check(
        "each frame takes the next sequence number",
        seq_of(&link.radio_mut().sent[0]) + 1 == seq_of(&link.radio_mut().sent[1]),
    );
    // Twelve bits, so it wraps after four thousand and ninety-six frames --
    // which on a link carrying anything at all is minutes, not a corner case.
    // A counter that ran on into the fragment number would put a 1 in the low
    // four bits and every receiver would read the frame as fragment one of
    // something it never saw the start of.
    {
        let mut w = Link::new(Loopback::new(me));
        let _ = w.radio_mut().start();
        for _ in 0..4095 {
            let _ = w.next_seq();
        }
        let last = w.next_seq();
        let wrapped = w.next_seq();
        check(
            "and after four thousand and ninety-six it wraps to zero, not to 4096",
            last == 4095 && wrapped == 0,
        );
        w.join(bssid);
        let _ = w.keyed(&tk, 0);
        let _ = w.transmit(&eth);
        check(
            "a frame built on the wrapped counter still has a clear fragment number",
            w.radio_mut()
                .sent
                .last()
                .map(|f| u16::from_le_bytes([f[22], f[23]]) & 0x000F == 0)
                .unwrap_or(false),
        );
    }

    // --- hardware crypto, and a part that claims it and then will not --
    let mut hw = Loopback::new(me);
    hw.hw_ccmp = true;
    let mut hwlink = Link::new(hw);
    let _ = hwlink.radio_mut().start();
    hwlink.join(bssid);
    check(
        "a part that encrypts in hardware takes the key and keeps none in software",
        hwlink.keyed(&tk, 0) && hwlink.secured() && hwlink.radio_mut().keys_taken == 1,
    );
    hwlink.radio_mut().sent.clear();
    check(
        "and its frames go out unwrapped, for the chip to protect",
        hwlink.transmit(&eth) && hwlink.radio_mut().sent[0][1] & 0x40 == 0,
    );
    // The one lie the interface cannot catch: claiming hardware crypto and
    // then refusing the key. Falling back is the only answer that does not
    // leave the link in the clear believing otherwise.
    let mut liar = Loopback::new(me);
    liar.hw_ccmp = true;
    liar.refuse_key = true;
    let mut liarlink = Link::new(liar);
    let _ = liarlink.radio_mut().start();
    liarlink.join(bssid);
    check(
        "a part that claims hardware crypto and refuses the key falls back to software",
        liarlink.keyed(&tk, 0) && liarlink.secured() && {
            liarlink.radio_mut().sent.clear();
            liarlink.transmit(&eth) && liarlink.radio_mut().sent[0][1] & 0x40 != 0
        },
    );

    // --- the radio's own refusals --------------------------------------
    let mut r = Loopback::new(me);
    check(
        "a radio that has not been started refuses to transmit",
        r.tx(b"anything").is_err() && r.rx().is_none(),
    );
    let _ = r.start();
    check(
        "and it tunes only to channels that exist",
        r.set_channel(6).is_ok() && r.set_channel(36).is_ok() && r.set_channel(37).is_err()
            && r.channel() == 36,
    );

    ok
}
