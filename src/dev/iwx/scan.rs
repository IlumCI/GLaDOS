//! Asking the part to scan, in the shape its firmware declares.
//!
//! **The firmware scans; the host only asks.** Intel parts will not transmit a
//! probe request the host built and handed over as a frame -- the scan engine
//! builds them out of a template in the request, tunes, dwells and moves on by
//! itself, hands every beacon and probe response it hears up the receive ring as
//! an `RX_MPDU`, and says when it is done. So this module is one command and one
//! notification, and `dev::radio::Radio::scan_offload` is the seam it plugs into.
//!
//! ### Version 17, because that is what the image says
//!
//! `SCAN_REQ_UMAC` has had more versions than any other command in the driver
//! and upstream carries two of them. The GF63's image declares **17**, read out
//! of its command-version table in the kernel, so 17 is the one built here; the
//! 22000 family's images declare 14, which differs in the general parameters
//! (v10 against v11) and the channel entry (v4 against v5), and is not written.
//! A firmware declaring anything else is refused by name rather than sent a
//! layout it does not speak.
//!
//! 1,940 bytes: an 8-byte header, general parameters 36, channel parameters 540
//! (67 entries of eight), periodic 12, probe parameters 1,344 (a 512-byte probe
//! template, twenty direct-SSID slots, short SSIDs and BSSIDs). Too large for a
//! command entry, which is why the queue grew a large-command path first.
//!
//! **Every offset is walked as well as written**, as `gen3` does for its
//! structures: the builder writes at named offsets and the claims recompute each
//! from the C declaration's field widths, so a field mistyped by a byte
//! disagrees with the walk rather than shifting everything after it into
//! somebody else's field with no error anywhere.

use alloc::vec::Vec;

pub const SCAN_REQ_UMAC: u8 = 0x0d;
pub const SCAN_COMPLETE_UMAC: u8 = 0x0f;
pub const SCAN_ITERATION_COMPLETE_UMAC: u8 = 0xb5;

/// The only layout written.
pub const VERSION: u8 = 17;
pub const LEN: usize = 1940;

/// Upstream's numbers for a foreground scan.
const DWELL_ACTIVE: u8 = 10;
const DWELL_PASSIVE: u8 = 110;
const ADWELL_DEFAULT_LB_N_APS: u8 = 2;
const ADWELL_DEFAULT_HB_N_APS: u8 = 8;
const ADWELL_DEFAULT_N_APS_SOCIAL: u8 = 10;
const ADWELL_MAX_BUDGET_FULL_SCAN: u16 = 300;
const ADWELL_N_APS_GO_FRIENDLY: u8 = 10;
const ADWELL_N_APS_SOCIAL_CHS: u8 = 2;
const PRIORITY_EXT_6: u32 = 6;

const GEN_PASS_ALL: u16 = 1 << 1;
const GEN_NTFY_ITER_COMPLETE: u16 = 1 << 2;
const GEN_ADAPTIVE_DWELL: u16 = 1 << 7;
const GEN_FORCE_PASSIVE: u16 = 1 << 11;

const CHANNEL_FLAG_ENABLE_CHAN_ORDER: u8 = 1 << 5;
const CHAN_CFG_FLAGS_BAND_POS: u32 = 30;
const PHY_BAND_5: u32 = 0;
const PHY_BAND_24: u32 = 1;

/// The firmware capability that asks for a DS parameter element in the probe
/// template, with its channel left for the firmware to fill.
pub const CAPA_DS_PARAM_SET_IE: usize = 9;

const MAX_CHANNELS: usize = 67;
const PROBE_BUF: usize = 512;
const SSID_SLOTS: usize = 20;
const SSID_LEN: usize = 32;

/// Where each part of the request begins. Recomputed in the claims from the
/// declaration's widths, which is the point of naming them.
pub mod at {
    pub const UID: usize = 0;
    pub const OOC_PRIORITY: usize = 4;
    pub const GENERAL: usize = 8;
    pub const CHANNELS: usize = GENERAL + 36;
    pub const PERIODIC: usize = CHANNELS + 4 + 67 * 8;
    pub const PROBE: usize = PERIODIC + 2 * 4 + 4;
    /// The probe template's buffer, after five four-byte segment descriptors.
    pub const PROBE_BUF: usize = PROBE + 5 * 4;
    pub const DIRECT_SCAN: usize = PROBE_BUF + 512 + 4;
    pub const SHORT_SSID: usize = DIRECT_SCAN + 20 * 34;
    pub const BSSIDS: usize = SHORT_SSID + 8 * 4;
    pub const END: usize = BSSIDS + 16 * 6;
}

/// What a scan is asked to do.
pub struct Request<'a> {
    /// The station's own address, which goes in the probe template.
    pub mac: [u8; 6],
    /// Channels to visit, by number. 2.4 GHz are 1 to 14; everything above is
    /// 5 GHz, which is all `dev::radio` numbers.
    pub channels: &'a [u8],
    /// A network to probe for by name. Empty is a passive scan, upstream's rule:
    /// with nothing to ask for, the part listens rather than announcing a
    /// wildcard probe on every channel.
    pub ssid: &'a [u8],
    /// Whether the part supports 5 GHz, from the NVM.
    pub band_5: bool,
    /// Whether the firmware wants a DS parameter element in the template.
    pub ds_param: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Refused {
    /// The image declares a scan layout this driver does not write.
    Version(Option<u8>),
    /// No channel the part may use was asked for.
    NoChannels,
    /// An SSID longer than 802.11 allows.
    LongSsid,
}

impl Refused {
    pub fn why(&self) -> &'static str {
        match self {
            Refused::Version(_) => "the firmware declares a scan layout other than version 17, which is the only one written",
            Refused::NoChannels => "no channel the part may use was asked for",
            Refused::LongSsid => "an SSID is at most thirty-two bytes",
        }
    }
}

fn put16(v: &mut [u8], at: usize, x: u16) {
    v[at..at + 2].copy_from_slice(&x.to_le_bytes());
}
fn put32(v: &mut [u8], at: usize, x: u32) {
    v[at..at + 4].copy_from_slice(&x.to_le_bytes());
}

/// 802.11g's rates in 500 kb/s units, the four DSSS ones basic. Eight go in the
/// Supported Rates element and the rest in Extended, which is the 802.11 limit
/// and the split net80211 makes.
const RATES_24: [u8; 12] = [0x82, 0x84, 0x8b, 0x96, 12, 18, 24, 36, 48, 72, 96, 108];
/// 802.11a's, with 6, 12 and 24 basic.
const RATES_5: [u8; 8] = [0x8c, 18, 0x98, 36, 0xb0, 72, 96, 108];

/// The probe template and where its parts are, as `(mac header, 2.4, 5, 6,
/// common)` segments of `(offset, length)` -- five, because the structure is
/// `mac_header, band_data[3], common_data`. Four put "common" in the 6 GHz band's
/// slot, harmless only while common is empty.
fn template(r: &Request) -> (Vec<u8>, [(u16, u16); 5]) {
    let mut f = Vec::with_capacity(64);
    f.extend_from_slice(&[0x40, 0x00, 0, 0]); // probe request, no DS bits, duration
    f.extend_from_slice(&[0xff; 6]);
    f.extend_from_slice(&r.mac);
    f.extend_from_slice(&[0xff; 6]);
    f.extend_from_slice(&[0, 0]); // sequence, filled by the part
    f.extend_from_slice(&[0, 0]); // an empty SSID element: the part inserts the name
    let hdr = (0u16, f.len() as u16);
    let b24 = f.len();
    f.push(1);
    f.push(8);
    f.extend_from_slice(&RATES_24[..8]);
    f.push(50);
    f.push(4);
    f.extend_from_slice(&RATES_24[8..]);
    if r.ds_param {
        f.extend_from_slice(&[3, 1, 0]);
    }
    let seg24 = (b24 as u16, (f.len() - b24) as u16);
    let seg5 = if r.band_5 {
        let b5 = f.len();
        f.push(1);
        f.push(8);
        f.extend_from_slice(&RATES_5);
        (b5 as u16, (f.len() - b5) as u16)
    } else {
        (0, 0)
    };
    // No HT or VHT capabilities: legacy rates only, which is what this station
    // can then receive. The common segment is empty and points at the end.
    let common = (f.len() as u16, 0u16);
    (f, [hdr, seg24, seg5, (0, 0), common])
}

/// Build the request.
pub fn request(declared: Option<u8>, r: &Request) -> Result<Vec<u8>, Refused> {
    if declared != Some(VERSION) {
        return Err(Refused::Version(declared));
    }
    if r.ssid.len() > SSID_LEN {
        return Err(Refused::LongSsid);
    }
    let chans: Vec<u8> = r
        .channels
        .iter()
        .copied()
        .filter(|&c| c != 0 && (c <= 14 || r.band_5))
        .take(MAX_CHANNELS)
        .collect();
    if chans.is_empty() {
        return Err(Refused::NoChannels);
    }
    let mut v = alloc::vec![0u8; LEN];
    put32(&mut v, at::UID, 0);
    put32(&mut v, at::OOC_PRIORITY, PRIORITY_EXT_6);

    // General parameters, v11.
    let g = at::GENERAL;
    let mut flags = GEN_PASS_ALL | GEN_NTFY_ITER_COMPLETE | GEN_ADAPTIVE_DWELL;
    if r.ssid.is_empty() {
        flags |= GEN_FORCE_PASSIVE;
    }
    put16(&mut v, g, flags);
    v[g + 3] = 0; // scan relative to MAC 0
    v[g + 4] = DWELL_ACTIVE;
    v[g + 5] = DWELL_ACTIVE;
    v[g + 6] = ADWELL_DEFAULT_LB_N_APS;
    v[g + 7] = ADWELL_DEFAULT_HB_N_APS;
    v[g + 8] = ADWELL_DEFAULT_N_APS_SOCIAL;
    put16(&mut v, g + 10, ADWELL_MAX_BUDGET_FULL_SCAN);
    // max_out_of_time and suspend_time stay zero: a foreground scan.
    put32(&mut v, g + 28, PRIORITY_EXT_6);
    v[g + 32] = DWELL_PASSIVE;
    v[g + 33] = DWELL_PASSIVE;

    // Channel parameters, v7, with v5 entries.
    let c = at::CHANNELS;
    v[c] = CHANNEL_FLAG_ENABLE_CHAN_ORDER;
    v[c + 1] = chans.len() as u8;
    v[c + 2] = ADWELL_N_APS_GO_FRIENDLY;
    v[c + 3] = ADWELL_N_APS_SOCIAL_CHS;
    // The low bits of an entry's flags say which direct-SSID slots to probe
    // for on it: slot zero when a name was given, none for a passive scan.
    let ssid_bits: u32 = if r.ssid.is_empty() { 0 } else { 1 };
    for (i, &ch) in chans.iter().enumerate() {
        let e = c + 4 + i * 8;
        let band = if ch <= 14 { PHY_BAND_24 } else { PHY_BAND_5 };
        put32(&mut v, e, ssid_bits | (band << CHAN_CFG_FLAGS_BAND_POS));
        v[e + 4] = ch;
        v[e + 5] = 0x80; // psd_20 = -128: no power-spectral-density limit given
        v[e + 6] = 1; // one iteration
        v[e + 7] = 0;
    }

    // Periodic: one iteration, no interval -- a single scan.
    put16(&mut v, at::PERIODIC, 0);
    v[at::PERIODIC + 2] = 1;

    // Probe parameters, v4.
    let (t, segs) = template(r);
    for (i, (off, len)) in segs.iter().enumerate() {
        put16(&mut v, at::PROBE + i * 4, *off);
        put16(&mut v, at::PROBE + i * 4 + 2, *len);
    }
    v[at::PROBE_BUF..at::PROBE_BUF + t.len()].copy_from_slice(&t);
    if !r.ssid.is_empty() {
        v[at::DIRECT_SCAN] = 0; // the SSID element id
        v[at::DIRECT_SCAN + 1] = r.ssid.len() as u8;
        v[at::DIRECT_SCAN + 2..at::DIRECT_SCAN + 2 + r.ssid.len()].copy_from_slice(r.ssid);
    }
    Ok(v)
}

/// What the part said when the scan ended.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Done {
    /// One for completed, two for aborted; anything else is reported raw.
    pub status: u8,
    pub iteration: bool,
}

/// Recognise either notification that ends a scan, upstream's rule: both end it.
/// Group zero or the long group, as every narrow notification may arrive.
pub fn done(group: u8, code: u8, payload: &[u8]) -> Option<Done> {
    if group > 1 {
        return None;
    }
    match code {
        SCAN_COMPLETE_UMAC if payload.len() >= 16 => Some(Done { status: payload[6], iteration: false }),
        SCAN_ITERATION_COMPLETE_UMAC if payload.len() >= 16 => Some(Done { status: payload[5], iteration: true }),
        _ => None,
    }
}

pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();
    // The walk, from the C declarations' field widths.
    let general = 2 + 1 + 1 + 2 + 1 + 1 + 1 + 1 + 2 + 2 * 4 + 2 * 4 + 4 + 2 + 2;
    let channels = 1 + 1 + 2 + 67 * (4 + 1 + 1 + 1 + 1);
    let periodic = 2 * (2 + 1 + 1) + 2 + 2;
    let preq = 5 * (2 + 2) + 512;
    let probe = preq + 1 + 1 + 2 + 20 * (1 + 1 + 32) + 8 * 4 + 16 * 6;
    out.push((
        "the general parameters are thirty-six bytes and the channel block 540",
        general == 36 && channels == 540 && at::PERIODIC - at::CHANNELS == channels,
    ));
    out.push((
        "and every section begins where the one before it ends",
        at::GENERAL == 8
            && at::CHANNELS == at::GENERAL + general
            && at::PROBE == at::PERIODIC + periodic
            && at::END == at::PROBE + probe,
    ));
    out.push(("so the request is 1,940 bytes", at::END == LEN && LEN == 1940));
    out.push((
        "which is too large for a command entry and inside what a command may carry",
        LEN + super::cmd::HDR_WIDE > super::cmd::ENTRY && LEN <= super::cmd::MAX_PAYLOAD,
    ));

    let me = [2, 0, 0, 0, 0, 0x11];
    let passive = Request { mac: me, channels: &[1, 6, 11, 36, 52], ssid: b"", band_5: true, ds_param: true };
    let v = request(Some(17), &passive);
    out.push(("a version-17 image gets a request", v.as_ref().map(|v| v.len()) == Ok(LEN)));
    out.push((
        "any other version is refused rather than sent a layout it does not speak",
        request(Some(14), &passive) == Err(Refused::Version(Some(14))) && request(None, &passive).is_err(),
    ));
    if let Ok(v) = v {
        let flags = u16::from_le_bytes([v[at::GENERAL], v[at::GENERAL + 1]]);
        out.push((
            "with nothing to probe for, the scan is passive and every frame is passed up",
            flags & GEN_FORCE_PASSIVE != 0 && flags & GEN_PASS_ALL != 0 && flags & GEN_NTFY_ITER_COMPLETE != 0,
        ));
        out.push((
            "the dwell times are upstream's: ten active, a hundred and ten passive",
            v[at::GENERAL + 4] == 10 && v[at::GENERAL + 32] == 110 && v[at::GENERAL + 33] == 110,
        ));
        let entry = |i: usize| {
            let e = at::CHANNELS + 4 + i * 8;
            (u32::from_le_bytes(v[e..e + 4].try_into().unwrap()), v[e + 4], v[e + 6])
        };
        out.push((
            "five channels, each with its band in bit 30 -- 2.4 GHz is one, 5 GHz zero",
            v[at::CHANNELS + 1] == 5
                && entry(0) == (1 << 30, 1, 1)
                && entry(2) == (1 << 30, 11, 1)
                && entry(3) == (0, 36, 1)
                && entry(5) == (0, 0, 0),
        ));
        let t = &v[at::PROBE_BUF..];
        out.push((
            "the probe template is a probe request from this station to everybody",
            t[0] == 0x40 && t[4..10] == [0xff; 6] && t[10..16] == me && t[16..22] == [0xff; 6],
        ));
        let seg = |i: usize| {
            let o = at::PROBE + i * 4;
            (u16::from_le_bytes([v[o], v[o + 1]]), u16::from_le_bytes([v[o + 2], v[o + 3]]))
        };
        out.push((
            "its header segment ends at an empty SSID element the part fills in",
            seg(0) == (0, 26) && t[24] == 0 && t[25] == 0,
        ));
        out.push((
            "the 2.4 GHz segment is eight rates, four extended, and the DS element asked for",
            seg(1) == (26, 2 + 8 + 2 + 4 + 3) && t[26] == 1 && t[27] == 8 && t[36] == 50 && t[42] == 3,
        ));
        out.push(("and the 5 GHz one follows it", seg(2) == (26 + 19, 10) && t[45] == 1));
        out.push((
            "the 6 GHz slot is empty and common is the fifth, pointing at the template's end",
            seg(3) == (0, 0) && seg(4) == (26 + 19 + 10, 0),
        ));
    }
    let named = Request { mac: me, channels: &[6], ssid: b"glados", band_5: false, ds_param: false };
    if let Ok(v) = request(Some(17), &named) {
        let flags = u16::from_le_bytes([v[at::GENERAL], v[at::GENERAL + 1]]);
        let e = at::CHANNELS + 4;
        out.push((
            "a named network is probed for actively, from direct-SSID slot zero on every channel",
            flags & GEN_FORCE_PASSIVE == 0
                && v[at::DIRECT_SCAN + 1] == 6
                && &v[at::DIRECT_SCAN + 2..at::DIRECT_SCAN + 8] == b"glados"
                && v[e] & 1 == 1,
        ));
    }
    out.push((
        "a part without 5 GHz is not asked to visit it, and an empty plan is refused",
        request(Some(17), &Request { mac: me, channels: &[36, 40], ssid: b"", band_5: false, ds_param: false })
            == Err(Refused::NoChannels),
    ));
    out.push((
        "an SSID past thirty-two bytes is refused",
        request(Some(17), &Request { mac: me, channels: &[1], ssid: &[b'x'; 33], band_5: false, ds_param: false })
            == Err(Refused::LongSsid),
    ));
    let mut complete = [0u8; 16];
    complete[6] = 1;
    let mut iter = [0u8; 16];
    iter[5] = 2;
    out.push((
        "either notification ends a scan, read at its own status offset, in either group",
        done(0, SCAN_COMPLETE_UMAC, &complete) == Some(Done { status: 1, iteration: false })
            && done(1, SCAN_ITERATION_COMPLETE_UMAC, &iter) == Some(Done { status: 2, iteration: true })
            && done(5, SCAN_COMPLETE_UMAC, &complete).is_none()
            && done(0, SCAN_COMPLETE_UMAC, &complete[..8]).is_none(),
    ));
    out
}
