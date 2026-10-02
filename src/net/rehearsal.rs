//! A radio with no chip behind it, and a room for it to hear.
//!
//! **This is not a driver and it does not pretend to be one.** `Radio::name`
//! answers "rehearsal", every interface that shows an adapter shows that word,
//! and the network manager says so across the top of its window. It exists for
//! one reason: everything above `dev::radio::Radio` is finished and there is no
//! part in this machine to attach it to, so without this the whole wireless
//! stack, the interface layer above it and the window above that could be
//! written and never once looked at.
//!
//! It is the same argument `Loopback` makes one layer down and `mkwad.py`
//! makes for DOOM: build the fixture so the thing can be exercised, and be
//! loud about which is which.
//!
//! ### It is also the worked example a driver author needs
//!
//! Nine methods and one of them optional. Start the part, say what it can do,
//! tune it, move frames. Everything a real chip adds -- registers, an endpoint,
//! firmware -- sits under exactly these, and nothing above this file changes
//! when a real one arrives. `net::attach_radio` takes this the same way it will
//! take an `ath9k`.
//!
//! ### What it models, and what it does not
//!
//! Several access points on several channels, each with its own BSSID, name,
//! security and signal strength, and **a station only hears the ones on the
//! channel it is tuned to** -- which is what makes a scan mean something here
//! rather than being a list handed over for free. Two of them carry the same
//! network at different strengths, because that is the case `mlme::choose`
//! exists for and the case a list sorted by name would get wrong.
//!
//! It does not model: interference, retransmission, hidden nodes, rate
//! selection, or an access point that stops answering. Those are the things a
//! real room does that this cannot, and they are why `diag hostile` exists
//! beside it.

use alloc::string::String;
use alloc::vec::Vec;

use crate::dev::radio::{Caps, JoinTarget, Key, Radio, Rx};
use crate::net::mlme::Ap;
use crate::net::Mac;

/// One access point in the rehearsal room.
struct Spot {
    ap: Ap,
    rssi: i8,
    /// Beacons go out on their own, as a real one does ten times a second --
    /// so a scan that never probes still finds the network, which is what a
    /// passive scan on a radar channel depends on.
    beacon_seq: u16,
}

/// How often each access point beacons. A real one is near this, and the
/// number matters: `mlme::DWELL_MS` is 120, so a scan has to sit on a channel
/// long enough to hear one, and a room that beaconed more slowly than the scan
/// dwells would be a room a correct scan misses.
const BEACON_MS: u64 = 100;

pub struct Rehearsal {
    mac: Mac,
    ch: u8,
    started: bool,
    spots: Vec<Spot>,
    /// Frames waiting for the station, with the signal each arrived at.
    inbox: Vec<(Vec<u8>, i8)>,
    /// When this room last beaconed.
    ///
    /// **Frames come from time passing, never from being asked for one.** The
    /// first version beaconed whenever `rx` found the queue empty, which meant
    /// `rx` never answered `None` -- and `Link::receive` drains until it does.
    /// The machine reached a prompt, took one command, and stopped; no fault,
    /// no message, the clock task still printing. A radio that manufactures a
    /// frame on demand is a radio that hangs whatever polls it.
    beaconed: u64,
    /// Play a firmware-assisted part: scan by itself and refuse every frame
    /// until it has been told which access point it is for. The shape an Intel
    /// part has, rehearsed before one can be driven.
    offload: bool,
    /// The scan in progress: the plan, how far it has got, and when it last
    /// moved.
    scan: Option<(Vec<u8>, usize, u64)>,
    prepared: bool,
}

/// How long a firmware scan sits on each channel. Shorter than the host's
/// dwell, because a part that scans for itself is not waiting on a host to be
/// scheduled -- and long enough here that a beacon is heard on every one.
const FW_DWELL_MS: u64 = BEACON_MS + 10;

/// The room, as it is built. Two access points carry `glados` so the strongest
/// has to be chosen rather than the first heard.
const ROOM: &[(&str, &str, u8, i8, [u8; 6])] = &[
    ("glados", "correct horse", 6, -42, [0x02, 0x47, 0x4C, 0x41, 0x44, 0x01]),
    ("glados", "correct horse", 11, -71, [0x02, 0x47, 0x4C, 0x41, 0x44, 0x02]),
    ("aperture-guest", "", 1, -58, [0x02, 0x41, 0x50, 0x45, 0x52, 0x01]),
    ("black-mesa", "unforeseen", 36, -66, [0x02, 0x42, 0x4D, 0x52, 0x46, 0x01]),
    ("Enrichment Center", "cake is a lie", 44, -79, [0x02, 0x45, 0x4E, 0x52, 0x49, 0x01]),
];

impl Rehearsal {
    pub fn new(mac: Mac) -> Rehearsal {
        let mut spots = Vec::new();
        for (ssid, pass, ch, rssi, bssid) in ROOM {
            spots.push(Spot {
                // The station address is unknown until one probes; `talking_to`
                // sets it when one does.
                ap: Ap::new(*bssid, [0; 6], ssid, pass, *ch),
                rssi: *rssi,
                beacon_seq: 0,
            });
        }
        Rehearsal {
            mac,
            ch: 1,
            started: false,
            spots,
            inbox: Vec::new(),
            beaconed: 0,
            offload: false,
            scan: None,
            prepared: false,
        }
    }

    /// The same room, heard through a part that scans in firmware.
    pub fn offloading(mac: Mac) -> Rehearsal {
        let mut r = Rehearsal::new(mac);
        r.offload = true;
        r
    }

    /// What is in the room, for anything that wants to say so plainly.
    pub fn room() -> Vec<String> {
        ROOM.iter()
            .map(|(ssid, pass, ch, rssi, _)| {
                alloc::format!(
                    "{} ch {} {} dBm {}",
                    ssid,
                    ch,
                    rssi,
                    if pass.is_empty() { "open" } else { "WPA2" }
                )
            })
            .collect()
    }

    /// Beacons from every access point on this channel.
    ///
    /// A real one transmits these whether or not anybody asks, which is the
    /// whole of a passive scan -- and the only way a station finds a network on
    /// a radar channel, where it must not probe.
    fn beacon(&mut self) {
        let ch = self.ch;
        self.beaconed = crate::net::now_ms();
        for s in self.spots.iter_mut() {
            if s.ap.channel != ch {
                continue;
            }
            s.beacon_seq = (s.beacon_seq + 1) & 0x0FFF;
            let f = crate::net::ieee80211::beacon(
                &s.ap.bssid,
                s.beacon_seq,
                s.ap.ssid,
                s.ap.channel,
                s.ap.rsn,
            );
            self.inbox.push((f, s.rssi));
        }
    }
}

impl Radio for Rehearsal {
    fn name(&self) -> &'static str {
        if self.offload { "rehearsal (offload)" } else { "rehearsal" }
    }

    fn caps(&self) -> Caps {
        // SoftMAC, because the point is to exercise the host's own MLME. A
        // part claiming otherwise would route around everything this is for.
        Caps { softmac: true, hw_ccmp: false, band5: true, max_frame: 2304 }
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
        self.inbox.clear();
    }

    fn set_channel(&mut self, ch: u8) -> Result<(), &'static str> {
        if crate::dev::radio::channel_mhz(ch).is_none() {
            return Err("no such channel");
        }
        // Tuning away drops whatever had not been taken off the part, which is
        // what a real radio does and is why a scan has to dwell.
        self.inbox.clear();
        self.ch = ch;
        // One round immediately, so a scan hears the channel it just tuned to
        // without waiting a whole beacon interval for it.
        self.beacon();
        Ok(())
    }

    fn channel(&self) -> u8 {
        self.ch
    }

    fn tx(&mut self, frame: &[u8]) -> Result<(), &'static str> {
        if !self.started {
            return Err("radio is not started");
        }
        if self.offload && !self.prepared {
            return Err("the firmware has not been told about an access point");
        }
        // Whoever is transmitting is the station these access points are
        // talking to. Read off the frame rather than configured, because that
        // is what an access point in a real room has to do.
        if let Some((_, sa, _)) = crate::net::ieee80211::mgmt_addrs(frame) {
            for s in self.spots.iter_mut() {
                s.ap.talking_to(sa);
            }
        }
        let ch = self.ch;
        let mut out: Vec<Vec<u8>> = Vec::new();
        let mut rssi: Vec<i8> = Vec::new();
        for s in self.spots.iter_mut() {
            let before = out.len();
            s.ap.handle(frame, ch, &mut out);
            for _ in before..out.len() {
                rssi.push(s.rssi);
            }
        }
        for (f, r) in out.into_iter().zip(rssi) {
            self.inbox.push((f, r));
        }
        Ok(())
    }

    fn rx(&mut self) -> Option<Rx> {
        // A beacon interval has passed, so every access point on this channel
        // sends one -- which is the whole of a passive scan and the only way a
        // radar channel is ever heard. Gated on the clock and not on the queue
        // being empty: see `beaconed`.
        let now = crate::net::now_ms();
        let mut stepped = false;
        if let Some((plan, at, last)) = self.scan.as_mut() {
            if *at < plan.len() && now.saturating_sub(*last) >= FW_DWELL_MS {
                self.ch = plan[*at];
                *at += 1;
                *last = now;
                stepped = true;
            }
        }
        if stepped || (self.started && now.saturating_sub(self.beaconed) >= BEACON_MS) {
            self.beacon();
        }
        if self.inbox.is_empty() {
            return None;
        }
        let (frame, rssi) = self.inbox.remove(0);
        Some(Rx { frame, rssi, channel: self.ch })
    }

    fn set_key(&mut self, _key: &Key) -> bool {
        // No hardware to put it in, so the honest answer, and `softmac` then
        // does CCMP itself -- which is the path worth rehearsing anyway.
        false
    }

    fn scan_offload(&mut self, _ssid: &str, chans: &[u8]) -> Option<Result<(), &'static str>> {
        if !self.offload {
            return None;
        }
        self.inbox.clear();
        // Backdated a dwell, so the first channel is heard on the first poll.
        let now = crate::net::now_ms();
        self.scan = Some((chans.to_vec(), 0, now.saturating_sub(FW_DWELL_MS)));
        Some(Ok(()))
    }

    fn scan_done(&mut self) -> bool {
        match &self.scan {
            // Done once the last channel has had its dwell, not the moment it
            // was tuned: otherwise the last channel's beacons are never heard.
            Some((plan, at, last)) if *at >= plan.len() => {
                if crate::net::now_ms().saturating_sub(*last) >= FW_DWELL_MS {
                    self.scan = None;
                    return true;
                }
                false
            }
            Some(_) => false,
            None => true,
        }
    }

    fn prepare_join(&mut self, t: &JoinTarget) -> Result<(), &'static str> {
        self.scan = None;
        self.set_channel(t.channel)?;
        self.prepared = true;
        Ok(())
    }

    fn left(&mut self) {
        self.prepared = false;
    }
}
