//! The seam a wireless chip plugs into: 802.11 frames in and out.
//!
//! **Not Ethernet, and that is the whole point.** `net::iface::Nic` takes an
//! Ethernet frame, which is right for an Ethernet card and right for a FullMAC
//! wireless part whose firmware hides 802.11 behind one. It is wrong for
//! everything else, and everything else is most of `dev::registry`'s wireless
//! rows: on a SoftMAC part the radio moves 802.11 frames and the *host* does
//! association, sequencing and encryption.
//!
//! So the seam is here, one layer down, and `net::softmac` turns a `Radio` into
//! a `Nic` for the parts that need it. A FullMAC driver skips that and
//! implements `Nic` directly -- which is why `Caps::softmac` exists rather than
//! two traits: the difference is a property of the chip, reported by the chip.
//!
//! ### What a driver has to supply, and what it does not
//!
//! Six methods. Start the part, say what it is, tune it, and move frames. That
//! is deliberately almost nothing: the MLME, CCMP, sequence numbers,
//! fragmentation and the supplicant are all above this and shared, so adding a
//! chip is a register file and a bulk endpoint rather than a wireless stack.
//!
//! `set_key` is the one optional method. A part that encrypts in hardware takes
//! the key and says so; everything else leaves the default, `false`, and
//! `softmac` does CCMP itself. Answering `true` without doing it is the one
//! lie this interface cannot catch, which is why the default is the honest one.
//!
//! ### The channel plan lives here, not in a driver
//!
//! It is the same plan for every part in the world. It was in `rtl8188eu`,
//! which is exactly the kind of chip-independent knowledge that ends up
//! duplicated once there are two drivers.

use alloc::vec::Vec;

/// Which band a channel is in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Band {
    G24,
    G5,
}

/// The centre frequency of a channel, in MHz, or nothing if it is not one.
///
/// **The numbering is dense and the plan is not**, so this is a list rather
/// than arithmetic with a range check: channel 37 is a number, not a channel,
/// and computing 5185 for it would produce a scan that tunes somewhere nobody
/// transmits and reports it as empty.
///
/// Channel 14 is absent. It is 2.4 GHz, 802.11b only, and legal in one country;
/// there is no regulatory database here to know which, and a scan that probes
/// it where it is forbidden is a transmission that should not happen. That
/// reasoning came from `rtl8188eu` and is the same for every part.
pub fn channel_mhz(ch: u8) -> Option<u16> {
    match ch {
        1..=13 => Some(2407 + 5 * ch as u16),
        36 | 40 | 44 | 48 => Some(5000 + 5 * ch as u16),
        52 | 56 | 60 | 64 => Some(5000 + 5 * ch as u16),
        100 | 104 | 108 | 112 | 116 | 120 | 124 | 128 | 132 | 136 | 140 | 144 => {
            Some(5000 + 5 * ch as u16)
        }
        149 | 153 | 157 | 161 | 165 => Some(5000 + 5 * ch as u16),
        _ => None,
    }
}

pub fn band_of(ch: u8) -> Option<Band> {
    match ch {
        1..=13 => Some(Band::G24),
        _ => channel_mhz(ch).map(|_| Band::G5),
    }
}

/// Does this channel share spectrum with radar?
///
/// **A scan must not transmit on one of these.** Dynamic frequency selection
/// means listening for radar and vacating, and nothing here can do that -- so
/// the honest behaviour is to receive on them and never send a probe request,
/// which is what "passive scan" means. Named rather than assumed, because a
/// driver that probes actively on every channel it can tune is breaking a rule
/// that has nothing to do with whether it works.
pub fn needs_dfs(ch: u8) -> bool {
    matches!(ch, 52..=144)
}

/// Every channel a band offers, in order.
pub fn channels(band: Band) -> Vec<u8> {
    (1u8..=165).filter(|c| band_of(*c) == Some(band)).collect()
}

/// What a part can do for itself.
#[derive(Clone, Copy)]
pub struct Caps {
    /// The host runs the MLME and the crypto. False means firmware does, and
    /// such a part implements `Nic` directly instead of going through
    /// `softmac`.
    pub softmac: bool,
    /// The part encrypts CCMP in hardware once it has been given a key.
    pub hw_ccmp: bool,
    /// Whether it can tune 5 GHz at all.
    pub band5: bool,
    /// The largest frame it will carry, including the 802.11 header.
    pub max_frame: usize,
}

/// A key handed to a part that does its own crypto.
pub struct Key {
    pub idx: u8,
    pub tk: [u8; 16],
    /// Pairwise keys are per peer; a group key is for the whole BSS.
    pub pairwise: bool,
    pub peer: [u8; 6],
}

/// One frame off the air, with what the radio knows about how it arrived.
pub struct Rx {
    pub frame: Vec<u8>,
    /// Received signal strength in dBm. Negative, and closer to zero is better.
    pub rssi: i8,
    pub channel: u8,
}

/// The network a station is about to join, as a part that keeps its own state
/// about one needs it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct JoinTarget {
    pub bssid: [u8; 6],
    pub channel: u8,
    /// Time units of 1024 us; zero when the beacon did not say.
    pub beacon_int: u16,
    /// `None` when only a probe response was heard, which carries no TIM.
    pub dtim: Option<u8>,
}

/// A wireless part, at the 802.11 frame.
///
/// ### Two kinds of part behind one seam
///
/// A plain SoftMAC part does what the required methods say and nothing else:
/// tune, send, receive. **A firmware-assisted part does not let the host do
/// that.** Intel's firmware will not transmit a probe request the host built,
/// and will not let a frame reach an access point until it has been told about
/// that access point -- a PHY context on its channel, a MAC context, a binding
/// between them, a station entry, and a time event holding the radio there.
/// The host still builds the authentication and association frames and still
/// runs the handshake, which is why such a part is SoftMAC in `Caps` and goes
/// through `mlme` rather than implementing `Nic` itself.
///
/// The hooks below are where the two kinds differ, and **every one has a
/// default that is exactly what the host-driven path already did**, so a part
/// that overrides none of them behaves as every part did before they existed.
/// That is asserted rather than intended: `diag mlme` runs the whole path twice,
/// once through the defaults and once through a radio that overrides them.
pub trait Radio {
    fn name(&self) -> &'static str;
    fn caps(&self) -> Caps;
    fn mac(&self) -> [u8; 6];

    fn start(&mut self) -> Result<(), &'static str>;
    fn stop(&mut self);

    fn set_channel(&mut self, ch: u8) -> Result<(), &'static str>;
    fn channel(&self) -> u8;

    /// Send one 802.11 frame, header and all.
    fn tx(&mut self, frame: &[u8]) -> Result<(), &'static str>;
    /// Take one received frame, if there is one. Never blocks.
    fn rx(&mut self) -> Option<Rx>;

    /// Install a key in hardware. `false` -- the default -- means the part does
    /// not do this and `softmac` will encrypt in software.
    fn set_key(&mut self, _key: &Key) -> bool {
        false
    }

    /// Scan in firmware. `None`, the default, means this part cannot and the
    /// host tunes channel by channel and sends its own probes. `Some(Ok)` means
    /// the scan has started and what it hears arrives through `rx` like any
    /// other frame, until `scan_done`. `Some(Err)` is a part that can scan and
    /// would not, which is a failure and not a reason to fall back: a
    /// host-driven scan on such a part transmits nothing.
    fn scan_offload(&mut self, _ssid: &str, _chans: &[u8]) -> Option<Result<(), &'static str>> {
        None
    }

    /// Whether an offloaded scan has finished. Never asked otherwise.
    fn scan_done(&mut self) -> bool {
        true
    }

    /// Stop an offloaded scan the station has given up on, or is leaving. A
    /// part left scanning refuses the next scan and goes on hopping channels
    /// under whatever the station does next. Never asked of a host-driven scan.
    fn scan_abort(&mut self) {}

    /// Get ready to talk to one access point. The default tunes to its channel,
    /// which is all a plain part needs.
    ///
    /// **An `Err` must leave nothing behind.** `left` is called only after an
    /// `Ok`, so a part that sets up several contexts and fails at the third
    /// undoes the first two itself before answering.
    fn prepare_join(&mut self, t: &JoinTarget) -> Result<(), &'static str> {
        self.set_channel(t.channel)
    }

    /// The association succeeded.
    fn associated(&mut self, _aid: u16, _t: &JoinTarget) {}

    /// The station has stopped talking to the access point it joined, for any
    /// reason. A part that set up state in `prepare_join` takes it down here.
    fn left(&mut self) {}
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

    check(
        "the 2.4 GHz plan is 2407 plus five per channel",
        channel_mhz(1) == Some(2412) && channel_mhz(6) == Some(2437)
            && channel_mhz(13) == Some(2472),
    );
    check(
        "and the 5 GHz plan is 5000 plus five",
        channel_mhz(36) == Some(5180) && channel_mhz(165) == Some(5825),
    );
    // The plan is a list because the numbering is dense and the plan is not.
    // Arithmetic with a range check would answer 5185 for channel 37 and tune
    // a radio somewhere nobody transmits.
    check(
        "a number between two channels is not a channel",
        channel_mhz(37).is_none() && channel_mhz(150).is_none() && channel_mhz(99).is_none(),
    );
    check(
        "channel 14 is absent, for a regulatory reason and not a technical one",
        channel_mhz(14).is_none() && channel_mhz(0).is_none(),
    );
    check(
        "bands are told apart by the channel, not by a flag somebody sets",
        band_of(1) == Some(Band::G24)
            && band_of(36) == Some(Band::G5)
            && band_of(37).is_none(),
    );
    // Radar channels must be received on and never transmitted on, because
    // nothing here can detect radar and vacate.
    check(
        "the radar channels are named, and the ones either side are not",
        needs_dfs(52) && needs_dfs(144) && !needs_dfs(48) && !needs_dfs(149),
    );
    check(
        "every listed channel has a frequency and every frequency a channel",
        channels(Band::G24).len() == 13
            && channels(Band::G5).len() == 25
            && channels(Band::G5).iter().all(|c| channel_mhz(*c).is_some()),
    );

    ok
}
