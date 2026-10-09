//! The held part as a `dev::radio::Radio`, which is what makes it `wlan0`.
//!
//! **A handle, not an owner.** The part lives in `iwx::HELD` behind its lock,
//! because two tasks service it and `iwx down` must be able to stop it whatever
//! holds this. So every method here takes the lock for the length of one call,
//! and a part that has been stopped underneath answers as a radio that is not
//! started rather than as one that hangs.
//!
//! ### What it does, and what it refuses by name
//!
//! It scans, through `scan_offload`: the firmware builds the probes, visits the
//! channels regulatory allows, and every beacon it hears comes back through
//! `rx`. That is enough for `wifi scan` and for the network manager to list what
//! is in the air.
//!
//! It joins, through `join.rs`: `prepare_join` puts up the PHY context, the
//! MAC context, the binding, the station and two queues, `tx` sends the
//! authentication and association `mlme` builds through the management queue,
//! `associated` tells the firmware the id, and `left` takes it all down.
//! **Untested on the part until the next trip**: every layout is asserted
//! against the headers and every step is journalled (`iwx journal` prints
//! it), which is what a trip with no serial line can bring home.

use crate::dev::radio::{Caps, Radio, Rx};

pub struct Air;

impl Radio for Air {
    fn name(&self) -> &'static str {
        "iwx"
    }

    fn caps(&self) -> Caps {
        let band5 = super::with_held(|h| h.facts.band_5).unwrap_or(false);
        // SoftMAC: the host runs the MLME and the handshake. No hardware CCMP
        // until keys can be installed in the part, which needs a station.
        Caps { softmac: true, hw_ccmp: false, band5, max_frame: 2304 }
    }

    fn mac(&self) -> [u8; 6] {
        super::with_held(|h| h.facts.mac).unwrap_or([0; 6])
    }

    fn start(&mut self) -> Result<(), &'static str> {
        match super::with_held(|h| h.stopped.is_none()) {
            Some(true) => Ok(()),
            Some(false) => Err("the part was stopped; `iwx boot` brings it up again"),
            None => Err("no part is held; `iwx boot` brings one up"),
        }
    }

    /// Nothing: stopping the station is not stopping the part. `iwx down` is.
    fn stop(&mut self) {}

    /// The part tunes through a PHY context, which `prepare_join` adds; a bare
    /// channel change has nothing to attach to and is refused by name.
    fn set_channel(&mut self, _ch: u8) -> Result<(), &'static str> {
        Err("this part tunes only through a scan or a PHY context; join through `prepare_join`")
    }

    fn channel(&self) -> u8 {
        super::with_held(|h| h.link.as_ref().map(|l| l.target.channel).unwrap_or(0)).unwrap_or(0)
    }

    fn tx(&mut self, frame: &[u8]) -> Result<(), &'static str> {
        super::with_held(|h| h.tx(frame)).unwrap_or(Err("no part is held"))
    }

    fn rx(&mut self) -> Option<Rx> {
        // The ring is drained into the inbox once the inbox is empty, not once
        // per frame: a burst of sixty-four was sixty-four ring reads.
        if super::with_held(|h| h.inbox.frames.is_empty()).unwrap_or(true) {
            super::service();
        }
        super::with_held(|h| h.inbox.frames.pop_front())
            .flatten()
            .map(|f| Rx { frame: f.frame, rssi: f.rssi, channel: f.channel })
    }

    fn scan_offload(&mut self, ssid: &str, chans: &[u8]) -> Option<Result<(), &'static str>> {
        Some(
            super::with_held(|h| h.scan(ssid.as_bytes(), chans))
            .unwrap_or(Err("no part is held")),
        )
    }

    /// Done once the part says so **and** every frame it delivered is read.
    /// The beacons from a scan's last channel arrive in the same ring batch as
    /// its end, and answering on the end alone had the station choose before
    /// reading them -- the only access point on the last channel, never seen.
    fn scan_done(&mut self) -> bool {
        super::service();
        super::with_held(|h| !h.scanning && h.inbox.frames.is_empty()).unwrap_or(true)
    }

    fn scan_abort(&mut self) {
        let _ = super::with_held(|h| h.scan_abort());
    }

    fn prepare_join(&mut self, t: &crate::dev::radio::JoinTarget) -> Result<(), &'static str> {
        super::with_held(|h| h.prepare_join(t)).unwrap_or(Err("no part is held"))
    }

    fn associated(&mut self, aid: u16, t: &crate::dev::radio::JoinTarget) {
        let _ = super::with_held(|h| h.associated(aid, t));
    }

    fn left(&mut self) {
        let _ = super::with_held(|h| h.leave());
    }
}
