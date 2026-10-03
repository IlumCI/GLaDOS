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
//! **It does not join yet.** `prepare_join` is where the PHY context, the MAC
//! context, the binding, the station and the time event belong, and none is
//! written -- so it refuses, by name, and `mlme` turns that into a failure the
//! operator can read rather than an authentication sent into a part that drops
//! it. `tx` refuses for the same reason: without a station there is no queue a
//! frame can go out on.

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

    fn set_channel(&mut self, _ch: u8) -> Result<(), &'static str> {
        Err("this part tunes only through a scan or a PHY context, and joining is not written")
    }

    fn channel(&self) -> u8 {
        0
    }

    fn tx(&mut self, _frame: &[u8]) -> Result<(), &'static str> {
        Err("this part sends frames only for a station it has been told about, and joining is not written")
    }

    fn rx(&mut self) -> Option<Rx> {
        super::service();
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

    fn prepare_join(&mut self, _t: &crate::dev::radio::JoinTarget) -> Result<(), &'static str> {
        Err("this driver scans and does not join yet: the station contexts are not written")
    }
}
