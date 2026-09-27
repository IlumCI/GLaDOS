//! How much the radio may sleep, which is two commands and only one of them yet.
//!
//! **`iwx_set_pslevel` sends two commands and returns after the first at
//! initialisation time**, which is the finding that shaped this file. It sends
//! `POWER_TABLE_CMD` -- four bytes, the *device's* policy -- and then:
//!
//! ```text
//!     if ((sc->sc_flags & IWX_FLAG_MAC_ACTIVE) == 0)
//!             return 0;
//! ```
//!
//! At `iwx_init_hw` no MAC context has been added, so that flag is clear and the
//! function returns there. The forty-byte `MAC_PM_POWER_TABLE` -- the one a reader
//! would call "the power table" -- is never sent at initialisation at all. It
//! needs a station id, a colour, and a DTIM period out of a beacon, none of which
//! exists before association.
//!
//! So what this file ships is the device command, wired into the configuration
//! sequence, and the MAC command's **layout, builder and arithmetic**, asserted and
//! deliberately not sent. That split is the honest one: the layout is an ABI and
//! can be checked now, where sending it would mean inventing a station.
//!
//! ### The default is not to sleep, and that is a decision
//!
//! `init_hw` asks for level 3 when the stack has power management on and level 0
//! otherwise. Nothing here turns it on, so the payload is four zero bytes -- and
//! the zeroes are the message rather than an unfilled buffer. On a machine whose
//! job is mining or serving, a radio that sleeps between beacons trades latency for
//! power nobody asked to save. `Facts::level` is how somebody changes that.
//!
//! ### The table is in units of 1024
//!
//! `iwx_pmgt` holds timeouts as small integers and the wire wants
//! `pmgt->rxtimeout * 1024`. A driver writing 200 where firmware expects 204,800
//! has asked for a fifth of a millisecond instead of a fifth of a second, which is
//! a radio that never sleeps while reporting that it does.
//!
//! ### Provenance
//!
//! As the rest of `iwx`: numbers from OpenBSD's `iwx(4)`, Intel's dual BSD/GPLv2
//! headers underneath, BSD arm, `NOTICE.md`.

use alloc::string::String;
use alloc::vec::Vec;

/// The device's own policy. Four bytes, and what `init_hw` sends.
pub const POWER_TABLE_CMD: u8 = 0x77;
/// A station's policy. Forty bytes, and unreachable until one exists.
pub const MAC_PM_POWER_TABLE: u8 = 0xa9;

/// The only bit the device command has.
pub const DEVICE_POWER_SAVE_ENA: u16 = 1 << 0;

/// The MAC command's flags.
pub const POWER_SAVE_ENA: u16 = 1 << 0;
pub const POWER_MANAGEMENT_ENA: u16 = 1 << 1;
pub const SKIP_OVER_DTIM: u16 = 1 << 2;

/// How long firmware should keep a station alive with no traffic, in seconds, as a
/// floor under whatever the beacon interval implies.
pub const KEEP_ALIVE_PERIOD_SEC: u32 = 25;

/// One row of the power-management ladder.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Pmgt {
    /// In units of 1024 on the wire. Held as upstream holds it, so the
    /// multiplication happens in one place and is visible there.
    pub rx_timeout: u32,
    pub tx_timeout: u32,
    /// How many DTIM periods may be slept through.
    pub skip_dtim: u8,
}

pub const NDTIMRANGES: usize = 3;
pub const NPOWERLEVELS: usize = 6;

/// The ladder, three DTIM ranges by six levels.
///
/// **Two of the three rows are identical**, which is upstream's table and not a
/// transcription slip: the short and medium DTIM ranges have the same six levels,
/// and only the long range differs -- there the two deepest levels stop skipping
/// DTIM periods, because a beacon interval already long enough to put the range at
/// eleven leaves nothing worth skipping. Kept as three rows rather than collapsed
/// to two, so the indexing matches the source it came from.
pub const PMGT: [[Pmgt; NPOWERLEVELS]; NDTIMRANGES] = [
    // DTIM <= 2
    [
        Pmgt { rx_timeout: 0, tx_timeout: 0, skip_dtim: 0 }, // continuously aware
        Pmgt { rx_timeout: 200, tx_timeout: 500, skip_dtim: 0 },
        Pmgt { rx_timeout: 200, tx_timeout: 300, skip_dtim: 0 },
        Pmgt { rx_timeout: 50, tx_timeout: 100, skip_dtim: 0 },
        Pmgt { rx_timeout: 50, tx_timeout: 25, skip_dtim: 1 },
        Pmgt { rx_timeout: 25, tx_timeout: 25, skip_dtim: 2 },
    ],
    // 3 <= DTIM <= 10
    [
        Pmgt { rx_timeout: 0, tx_timeout: 0, skip_dtim: 0 },
        Pmgt { rx_timeout: 200, tx_timeout: 500, skip_dtim: 0 },
        Pmgt { rx_timeout: 200, tx_timeout: 300, skip_dtim: 0 },
        Pmgt { rx_timeout: 50, tx_timeout: 100, skip_dtim: 0 },
        Pmgt { rx_timeout: 50, tx_timeout: 25, skip_dtim: 1 },
        Pmgt { rx_timeout: 25, tx_timeout: 25, skip_dtim: 2 },
    ],
    // DTIM >= 11
    [
        Pmgt { rx_timeout: 0, tx_timeout: 0, skip_dtim: 0 },
        Pmgt { rx_timeout: 200, tx_timeout: 500, skip_dtim: 0 },
        Pmgt { rx_timeout: 200, tx_timeout: 300, skip_dtim: 0 },
        Pmgt { rx_timeout: 50, tx_timeout: 100, skip_dtim: 0 },
        Pmgt { rx_timeout: 50, tx_timeout: 25, skip_dtim: 0 },
        Pmgt { rx_timeout: 25, tx_timeout: 25, skip_dtim: 0 },
    ],
];

/// Which row of the ladder a DTIM period falls in.
///
/// The boundaries are upstream's: two and ten. Written as a function rather than
/// inline because getting `<=` wrong at either end silently picks a neighbouring
/// row, and on the short and medium rows those are identical -- so the mistake is
/// invisible for every DTIM under eleven and changes the two deepest levels above
/// it.
pub fn range_of(dtim: u32) -> usize {
    if dtim <= 2 {
        0
    } else if dtim <= 10 {
        1
    } else {
        2
    }
}

/// The wire value of a timeout out of the ladder.
///
/// The one place the 1024 lives. A driver writing the table's own number asks for a
/// fifth of a millisecond where firmware was told a fifth of a second.
pub fn timeout_on_wire(t: u32) -> u32 {
    t * 1024
}

// --- the device command ------------------------------------------------------

/// Build the four-byte device policy.
///
/// Level zero means continuously aware and the payload is four zeroes. That is the
/// default here and the reason is in the module header: a machine that mines or
/// serves should not have its radio asleep between beacons.
pub fn device_body(level: u8) -> Vec<u8> {
    let flags: u16 = if level != 0 { DEVICE_POWER_SAVE_ENA } else { 0 };
    let mut v = alloc::vec![0u8; 4];
    v[0..2].copy_from_slice(&flags.to_le_bytes());
    v
}

// --- the MAC command, laid out and not sent ----------------------------------

/// Offsets in the forty-byte station policy.
pub mod at {
    pub const ID_AND_COLOR: usize = 0;
    pub const FLAGS: usize = 4;
    pub const KEEP_ALIVE_SECONDS: usize = 6;
    pub const RX_DATA_TIMEOUT: usize = 8;
    pub const TX_DATA_TIMEOUT: usize = 12;
    pub const RX_DATA_TIMEOUT_UAPSD: usize = 16;
    pub const TX_DATA_TIMEOUT_UAPSD: usize = 20;
    pub const LPRX_RSSI_THRESHOLD: usize = 24;
    pub const SKIP_DTIM_PERIODS: usize = 25;
    pub const SNOOZE_INTERVAL: usize = 26;
    pub const SNOOZE_WINDOW: usize = 28;
    pub const SNOOZE_STEP: usize = 30;
    pub const QNDP_TID: usize = 31;
    pub const UAPSD_AC_FLAGS: usize = 32;
    pub const UAPSD_MAX_SP: usize = 33;
    pub const HEAVY_TX_THLD_PACKETS: usize = 34;
    pub const HEAVY_RX_THLD_PACKETS: usize = 35;
    pub const HEAVY_TX_THLD_PERCENTAGE: usize = 36;
    pub const HEAVY_RX_THLD_PERCENTAGE: usize = 37;
    pub const LIMITED_PS_THRESHOLD: usize = 38;
    pub const RESERVED: usize = 39;
}

/// Its length.
pub const MAC_POWER_LEN: usize = 40;

/// A firmware context's identity: an eight-bit id and an eight-bit colour in one
/// word.
///
/// The colour is a generation counter, so firmware can tell a context that was
/// removed and re-added from the one it answered about a moment ago. Packing it
/// into the same word is why both are bytes and why a driver that used the id alone
/// would address the wrong generation after a reconfiguration.
pub fn id_and_color(id: u8, color: u8) -> u32 {
    (id as u32) | ((color as u32) << 8)
}

/// How long firmware should keep a sleeping station alive.
///
/// Three beacon intervals, floored at twenty-five seconds, rounded **up** to whole
/// seconds. Rounding down would ask for less time than three beacons take, so a
/// station on a slow network would be dropped by the firmware that was told to keep
/// it.
pub fn keep_alive_seconds(dtim_period: u32, beacon_interval_ms: u32) -> u16 {
    let dtim_msec = dtim_period.max(1).saturating_mul(beacon_interval_ms);
    let ms = (3u32.saturating_mul(dtim_msec)).max(1000 * KEEP_ALIVE_PERIOD_SEC);
    // Round up, then to seconds.
    (ms.saturating_add(999) / 1000) as u16
}

/// What a station's policy needs that is not a constant.
#[derive(Clone, Copy)]
pub struct Station {
    pub id: u8,
    pub color: u8,
    pub level: u8,
    /// Out of the beacon. There is no beacon yet, which is why nothing calls this.
    pub dtim_period: u32,
    pub beacon_interval_ms: u32,
}

/// Build the station policy.
///
/// **Nothing calls this yet, on purpose.** It needs a station id and a DTIM period
/// out of a beacon; both arrive with association. Written now because the layout is
/// an ABI and a forty-byte structure with one field at the wrong offset is not a
/// smaller command, it is a different one -- so the offsets are worth asserting
/// before anything depends on them.
///
/// uAPSD is deliberately absent. It needs the station to have advertised it and the
/// firmware to have declared support, and a driver filling those fields without
/// both would be describing a negotiation that never happened.
pub fn mac_body(s: &Station) -> Vec<u8> {
    let mut v = alloc::vec![0u8; MAC_POWER_LEN];
    let put32 = |v: &mut Vec<u8>, at: usize, x: u32| v[at..at + 4].copy_from_slice(&x.to_le_bytes());
    let put16 = |v: &mut Vec<u8>, at: usize, x: u16| v[at..at + 2].copy_from_slice(&x.to_le_bytes());

    put32(&mut v, at::ID_AND_COLOR, id_and_color(s.id, s.color));
    put16(
        &mut v,
        at::KEEP_ALIVE_SECONDS,
        keep_alive_seconds(s.dtim_period, s.beacon_interval_ms),
    );
    // Level zero leaves every other field zero, which is what continuously aware
    // is: no flags, no timeouts, nothing skipped.
    if s.level != 0 {
        let row = PMGT[range_of(s.dtim_period)][(s.level as usize).min(NPOWERLEVELS - 1)];
        let mut flags = POWER_SAVE_ENA | POWER_MANAGEMENT_ENA;
        put32(&mut v, at::RX_DATA_TIMEOUT, timeout_on_wire(row.rx_timeout));
        put32(&mut v, at::TX_DATA_TIMEOUT, timeout_on_wire(row.tx_timeout));
        if row.skip_dtim != 0 {
            flags |= SKIP_OVER_DTIM;
            // **One more than the table says.** The field counts periods to skip
            // *including* the one being slept through, so writing the table's own
            // number asks for one fewer than the level means.
            v[at::SKIP_DTIM_PERIODS] = row.skip_dtim + 1;
        }
        put16(&mut v, at::FLAGS, flags);
    }
    v
}

/// Claims. No radio, and no register touched.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();
    let mut ok = |c: bool, w: &'static str| out.push((w, c));

    // --- the two commands are not the same size, or the same command ---------

    ok(POWER_TABLE_CMD == 0x77 && MAC_PM_POWER_TABLE == 0xa9, "the device and station policies are two opcodes");
    ok(device_body(0).len() == 4, "the device policy is four bytes");
    ok(MAC_POWER_LEN == 40, "and the station policy forty");
    // The finding this file opens with: only the small one is sent at
    // initialisation, because the other needs a station.
    ok(
        device_body(0).len() < MAC_POWER_LEN,
        "so the one sent at initialisation is the small one, the other needing a station",
    );

    // --- the device policy --------------------------------------------------

    ok(device_body(0).iter().all(|&b| b == 0), "level zero is four zeroes, which is continuously aware");
    ok(
        u16::from_le_bytes([device_body(3)[0], device_body(3)[1]]) == DEVICE_POWER_SAVE_ENA,
        "and any level above it sets the one bit the command has",
    );
    // The default, and the reason. A machine that mines should not sleep its radio.
    ok(device_body(0) != device_body(1), "so the default differs from the shallowest saving level");

    // --- the ladder ---------------------------------------------------------

    ok(PMGT.len() == NDTIMRANGES && PMGT[0].len() == NPOWERLEVELS, "the ladder is three ranges by six levels");
    ok(PMGT[0][0] == Pmgt { rx_timeout: 0, tx_timeout: 0, skip_dtim: 0 }, "level zero is all zeroes in every range");
    // Two of three rows identical is upstream's table, not a slip -- asserted so
    // that stays deliberate and a future edit to one row is noticed.
    ok(PMGT[0] == PMGT[1], "the short and medium DTIM ranges are the same six levels");
    ok(PMGT[0] != PMGT[2], "and the long range differs");
    ok(
        PMGT[2][4].skip_dtim == 0 && PMGT[0][4].skip_dtim == 1,
        "where it differs is that the deep levels stop skipping DTIM periods",
    );
    ok(
        PMGT[2][4].rx_timeout == PMGT[0][4].rx_timeout,
        "and only in that -- the timeouts are the same",
    );

    // The boundaries. Getting either wrong picks a neighbouring row, and for
    // everything under eleven the neighbour is identical -- so the mistake is
    // invisible until a slow network shows up.
    ok(range_of(1) == 0 && range_of(2) == 0, "two is in the short range");
    ok(range_of(3) == 1 && range_of(10) == 1, "three to ten is the medium one");
    ok(range_of(11) == 2, "and eleven is the long one");
    ok(range_of(0) == 0, "a DTIM of zero falls in the short range rather than panicking");

    // The unit. 200 on the wire is a fifth of a millisecond; 204,800 is a fifth of
    // a second, and the difference is a radio that never sleeps while saying it
    // does.
    ok(timeout_on_wire(200) == 204_800, "a timeout is multiplied by 1024 on the wire");
    ok(timeout_on_wire(0) == 0, "and zero stays zero");

    // --- the station policy's layout ----------------------------------------

    // Walked as well as written down, as `gen3` does: two statements from one C
    // declaration, so a mistyped constant disagrees with the walk.
    let mut w = 0usize;
    ok(w == at::ID_AND_COLOR, "the context identity is at 0");
    w += 4;
    ok(w == at::FLAGS, "the flags at 4");
    w += 2;
    ok(w == at::KEEP_ALIVE_SECONDS, "keep-alive at 6, sharing the word with them");
    w += 2;
    ok(w == at::RX_DATA_TIMEOUT, "the receive timeout at 8");
    w += 4;
    ok(w == at::TX_DATA_TIMEOUT, "the transmit one at 12");
    w += 4 + 4 + 4; // tx, then the two uAPSD timeouts
    ok(w == at::LPRX_RSSI_THRESHOLD, "the low-power receive threshold at 24");
    w += 1;
    ok(w == at::SKIP_DTIM_PERIODS, "and the DTIM skip count beside it at 25");
    w += 1;
    ok(w == at::SNOOZE_INTERVAL, "the snooze interval at 26");
    w += 2 + 2 + 1 + 1 + 1 + 1; // interval, window, step, tid, ac flags, max sp
    ok(w == at::HEAVY_TX_THLD_PACKETS, "the four heavy-traffic thresholds at 34");
    w += 4;
    ok(w == at::LIMITED_PS_THRESHOLD, "the limited-saving threshold at 38");
    w += 1;
    ok(w == at::RESERVED, "one reserved byte at 39");
    w += 1;
    ok(w == MAC_POWER_LEN, "and the whole command is forty bytes");

    // --- the identity word --------------------------------------------------

    ok(id_and_color(0, 0) == 0, "a zero context is a zero word");
    ok(id_and_color(3, 0) == 3, "the id is the low byte");
    ok(id_and_color(0, 7) == 0x0700, "and the colour the next one up");
    // The colour is a generation counter, so a driver using the id alone addresses
    // the wrong generation after a context is removed and re-added.
    ok(id_and_color(3, 7) != id_and_color(3, 8), "so two generations of one id are different words");

    // --- the keep-alive arithmetic ------------------------------------------

    // The floor, for a fast network where three beacons are nothing.
    ok(keep_alive_seconds(1, 100) == KEEP_ALIVE_PERIOD_SEC as u16, "a fast network takes the twenty-five second floor");
    // And above it: three DTIM periods of 10 x 1000 ms is 30 s.
    ok(keep_alive_seconds(10, 1000) == 30, "a slow one takes three DTIM periods instead");
    // Rounded up, not down. Down would ask for less time than three beacons take,
    // so a station on a slow network is dropped by the firmware told to keep it.
    ok(keep_alive_seconds(1, 8501) == 26, "and the rounding is up, since down would ask for too little");
    ok(keep_alive_seconds(0, 100) == KEEP_ALIVE_PERIOD_SEC as u16, "a DTIM of zero is treated as one rather than as no time");

    // --- the station policy's contents --------------------------------------

    let s = Station { id: 1, color: 2, level: 0, dtim_period: 1, beacon_interval_ms: 100 };
    let v = mac_body(&s);
    ok(v.len() == MAC_POWER_LEN, "a station policy is forty bytes");
    ok(
        u32::from_le_bytes([v[0], v[1], v[2], v[3]]) == id_and_color(1, 2),
        "carrying the context identity",
    );
    ok(
        u16::from_le_bytes([v[at::KEEP_ALIVE_SECONDS], v[at::KEEP_ALIVE_SECONDS + 1]]) == 25,
        "and the keep-alive even at level zero, which is not conditional",
    );
    ok(
        u16::from_le_bytes([v[at::FLAGS], v[at::FLAGS + 1]]) == 0,
        "level zero sets no flags",
    );
    ok(
        v[at::RX_DATA_TIMEOUT..at::RX_DATA_TIMEOUT + 8].iter().all(|&b| b == 0),
        "and no timeouts, which is what continuously aware means",
    );

    // A deep level, where the skip count and the +1 matter.
    let deep = Station { level: 5, ..s };
    let v = mac_body(&deep);
    ok(
        u16::from_le_bytes([v[at::FLAGS], v[at::FLAGS + 1]])
            == POWER_SAVE_ENA | POWER_MANAGEMENT_ENA | SKIP_OVER_DTIM,
        "a deep level sets saving, management and the DTIM skip",
    );
    ok(
        u32::from_le_bytes([v[8], v[9], v[10], v[11]]) == timeout_on_wire(25),
        "with the ladder's timeout multiplied by 1024",
    );
    // **One more than the table.** The field counts the period slept through, so
    // writing the table's own number asks for one fewer than the level means.
    ok(
        v[at::SKIP_DTIM_PERIODS] == PMGT[0][5].skip_dtim + 1,
        "and a skip count one greater than the ladder's, which counts the period slept through",
    );
    // A level whose row says not to skip leaves the bit clear, rather than setting
    // it with a count of one.
    let mid = Station { level: 3, ..s };
    let v = mac_body(&mid);
    ok(
        u16::from_le_bytes([v[at::FLAGS], v[at::FLAGS + 1]]) & SKIP_OVER_DTIM == 0
            && v[at::SKIP_DTIM_PERIODS] == 0,
        "a level that does not skip leaves the bit clear and the count zero",
    );
    // uAPSD is absent, which is a decision: filling it needs a negotiation that has
    // not happened.
    ok(
        v[at::RX_DATA_TIMEOUT_UAPSD..at::TX_DATA_TIMEOUT_UAPSD + 4].iter().all(|&b| b == 0)
            && v[at::UAPSD_AC_FLAGS] == 0 && v[at::UAPSD_MAX_SP] == 0,
        "and no uAPSD, which would describe a negotiation that never happened",
    );
    // A level past the ladder is clamped rather than indexing out of range.
    ok(mac_body(&Station { level: 99, ..s }).len() == MAC_POWER_LEN, "a level past the ladder is clamped, not a panic");

    out
}
