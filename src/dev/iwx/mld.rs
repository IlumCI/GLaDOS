//! Joining an access point, through the firmware's MLD API.
//!
//! `scan.rs` is where this driver stopped, and the reason it stopped there is
//! that joining is five contexts and a queue where scanning is one command. The
//! firmware this part runs (`so-a0-hr-b0-89`) declares `MLD_API_SUPPORT`, and a
//! firmware that does is driven through `MAC_CONF_GROUP`'s link-aware commands
//! -- `MAC_CONFIG_CMD`, `LINK_CONFIG_CMD`, `STA_CONFIG_CMD` -- rather than the
//! older `MAC_CONTEXT_CMD` / `BINDING_CONTEXT_CMD` / `ADD_STA` trio OpenBSD's
//! `iwx(4)` sends. Both families exist in the image; the newer one is what
//! Linux sends it, and sending the older to a part whose driver has moved on
//! is a path nobody exercises any more.
//!
//! ### The sequence
//!
//! ```text
//!   PHY_CONTEXT_CMD   add      the channel
//!   MAC_CONFIG_CMD    add      who we are, not associated, hear beacons
//!   LINK_CONFIG_CMD   add      a link on that MAC, no PHY yet
//!   LINK_CONFIG_CMD   modify   active, on that PHY, rates and timing
//!   STA_CONFIG_CMD             the access point, as a peer station
//!   SCD_QUEUE_CONFIG  add x2   a management queue and a data queue
//!   SESSION_PROTECTION add     stay on channel for the handshake
//! ```
//!
//! Then `mlme` authenticates and associates through `tx`, and `associated`
//! tells the firmware the association id. `left` takes it all down in reverse.
//!
//! ### Layouts are facts about the firmware, and are asserted
//!
//! Every structure here is a packed little-endian layout read from Intel's
//! headers (dual BSD/GPLv2, the BSD arm, see `NOTICE.md`), and every builder
//! asserts the size it produces against the number computed from those headers
//! -- a structure one field short is not a smaller command, it is a different
//! command the firmware parses to the end of. Versions are checked against the
//! image's command-version table before anything is sent, and a version this
//! module does not know is refused by name rather than sent in the hope that
//! the layout did not move.
//!
//! ### What is deliberately not done
//!
//! **Rates are fixed at the lowest legacy rate**, 1 Mb/s on 2.4 GHz and 6 Mb/s
//! on 5 GHz, with `IWL_TX_FLAGS_CMD_RATE` set on every frame. Letting the
//! firmware choose needs `TLC_MNG_CONFIG_CMD` and the station's whole rate
//! table, and a link that works slowly is the rung before a link that works
//! fast. **Keys stay in software**: `wlan.rs` keeps `hw_ccmp` false, so
//! `softmac` encrypts and the frame goes out with `IWL_TX_FLAGS_ENCRYPT_DIS`.
//! Both are stated here so the next trip knows what it is measuring.

use alloc::vec::Vec;

use super::cmd;
use crate::dev::dma::Dma;

// --- groups and commands -----------------------------------------------------

pub const MAC_CONF_GROUP: u8 = 0x3;
pub const DATA_PATH_GROUP: u8 = 0x5;

/// In `LONG_GROUP`.
pub const PHY_CONTEXT_CMD: u8 = 0x08;
pub const TX_CMD: u8 = 0x1c;

/// In `MAC_CONF_GROUP`.
pub const SESSION_PROTECTION_CMD: u8 = 0x05;
pub const MAC_CONFIG_CMD: u8 = 0x08;
pub const LINK_CONFIG_CMD: u8 = 0x09;
pub const STA_CONFIG_CMD: u8 = 0x0a;
pub const STA_REMOVE_CMD: u8 = 0x0c;
pub const SESSION_PROTECTION_NOTIF: u8 = 0xfb;

/// In `DATA_PATH_GROUP`.
pub const SCD_QUEUE_CONFIG_CMD: u8 = 0x17;

/// Capability bits the join path rests on.
pub const CAPA_ULTRA_HB_CHANNELS: usize = 48;
pub const CAPA_MLD_API: usize = 121;

pub const ACTION_ADD: u32 = 1;
pub const ACTION_MODIFY: u32 = 2;
pub const ACTION_REMOVE: u32 = 3;
pub const CTXT_INVALID: u32 = 0xffff_ffff;

/// The one id this driver uses for everything: one PHY, one MAC, one link, one
/// station. Colour zero. `id | colour << 8`.
pub const ID: u32 = 0;
pub const STA_ID: u32 = 0;

/// Sizes, computed from the headers. A builder that produces another number has
/// a field wrong.
pub const PHY_LEN: usize = 32;
pub const MAC_LEN: usize = 64;
pub const LINK_LEN: usize = 208;
pub const STA_V1_LEN: usize = 96;
pub const STA_V2_LEN: usize = 104;
pub const SPROT_LEN: usize = 24;
pub const SCD_LEN: usize = 36;
pub const SCD_RSP_LEN: usize = 8;
/// The TX command proper, between the wide header and the 802.11 header.
pub const TX_CMD_LEN: usize = 28;

// --- what the image has to say -----------------------------------------------

/// The command versions the join path sends by, read off the image once.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Versions {
    pub phy: Option<u8>,
    pub mac: Option<u8>,
    pub link: Option<u8>,
    pub sta: Option<u8>,
    pub tx: Option<u8>,
    pub tx_notif: Option<u8>,
    pub scd: Option<u8>,
    pub sprot: Option<u8>,
    pub ultra_hb: bool,
    pub mld: bool,
}

impl Versions {
    pub fn of(image: &super::fw::Image) -> Versions {
        let l = cmd::LONG_GROUP;
        Versions {
            phy: image.cmd_ver(l, PHY_CONTEXT_CMD),
            mac: image.cmd_ver(MAC_CONF_GROUP, MAC_CONFIG_CMD),
            link: image.cmd_ver(MAC_CONF_GROUP, LINK_CONFIG_CMD),
            sta: image.cmd_ver(MAC_CONF_GROUP, STA_CONFIG_CMD),
            tx: image.cmd_ver(l, TX_CMD),
            tx_notif: image.notif_ver(l, TX_CMD),
            scd: image.cmd_ver(DATA_PATH_GROUP, SCD_QUEUE_CONFIG_CMD),
            sprot: image.cmd_ver(MAC_CONF_GROUP, SESSION_PROTECTION_CMD),
            ultra_hb: image.has_capa(CAPA_ULTRA_HB_CHANNELS),
            mld: image.has_capa(CAPA_MLD_API),
        }
    }

    /// Whether this module knows how to talk to this image, and if not, which
    /// command it would get wrong. A version absent from the table is the
    /// firmware's default, which for these commands is the first layout.
    pub fn check(&self) -> Result<(), &'static str> {
        if !self.mld {
            return Err("the image does not declare the MLD API, and the legacy MAC/binding path is not written");
        }
        if !self.ultra_hb {
            return Err("the image lacks ULTRA_HB_CHANNELS, so its channel info is the four-byte v1 this module does not build");
        }
        match self.phy.unwrap_or(1) {
            3 | 4 => {}
            _ => return Err("PHY_CONTEXT_CMD is a version other than 3 or 4"),
        }
        match self.mac.unwrap_or(1) {
            2 | 3 => {}
            _ => return Err("MAC_CONFIG_CMD is a version other than 2 or 3"),
        }
        match self.link.unwrap_or(1) {
            1..=6 => {}
            _ => return Err("LINK_CONFIG_CMD is past version 6, where the layout grew"),
        }
        match self.sta.unwrap_or(0) {
            0..=2 => {}
            _ => return Err("STA_CONFIG_CMD is version 3, whose link mask this module does not build"),
        }
        if self.tx.unwrap_or(0) != 10 {
            return Err("TX_CMD is not version 10, the only layout written here");
        }
        if self.tx_notif.unwrap_or(0) < 7 {
            return Err("TX_CMD's notification is below 7, so rates are the old format");
        }
        if self.scd.unwrap_or(0) != 3 {
            return Err("SCD_QUEUE_CONFIG_CMD is not version 3; the queue-add shape differs");
        }
        match self.sprot.unwrap_or(1) {
            1 | 2 => {}
            _ => return Err("SESSION_PROTECTION_CMD is past version 2"),
        }
        Ok(())
    }

    /// The `STA_CONFIG_CMD` length this image wants.
    pub fn sta_len(&self) -> usize {
        if self.sta.unwrap_or(0) >= 2 { STA_V2_LEN } else { STA_V1_LEN }
    }
}

// --- small writers -----------------------------------------------------------

fn w32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}
fn w16(b: &mut [u8], at: usize, v: u16) {
    b[at..at + 2].copy_from_slice(&v.to_le_bytes());
}
fn w64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

// --- PHY_CONTEXT_CMD, version 4 ----------------------------------------------

pub const PHY_BAND_5: u8 = 0;
pub const PHY_BAND_24: u8 = 1;
/// `IWL_PHY_CHANNEL_MODE20`.
pub const CHANNEL_MODE20: u8 = 0;
/// `IWL_PHY_CTRL_POS_ABOVE`: a secondary channel location. On a 20 MHz
/// channel there is none, and upstream still writes "the other side" into the
/// NPCA field "to help firmware".
pub const CTRL_POS_ABOVE: u8 = 0x4;

/// The PHY context: one channel, 20 MHz wide, on LMAC 0.
///
/// `rxchain_info` is reserved in v4 and left zero; v3 read it and firmware at
/// v3 is not what this part carries.
pub fn phy_context(action: u32, channel: u8, band24: bool) -> [u8; PHY_LEN] {
    let mut b = [0u8; PHY_LEN];
    w32(&mut b, 0, ID);
    w32(&mut b, 4, action);
    // iwl_fw_channel_info, v2: channel u32, band u8, width u8, ctrl_pos u8.
    w32(&mut b, 8, channel as u32);
    b[12] = if band24 { PHY_BAND_24 } else { PHY_BAND_5 };
    b[13] = CHANNEL_MODE20;
    b[14] = 0;
    w32(&mut b, 16, 0); // lmac_id
    w32(&mut b, 20, 0); // rxchain_info, reserved in v4
    w32(&mut b, 24, 0); // dsp_cfg_flags
    b[28] = CTRL_POS_ABOVE; // secondary_ctrl_chnl_loc = ctrl_pos ^ ABOVE
    b
}

// --- MAC_CONFIG_CMD, version 2 -----------------------------------------------

pub const MAC_TYPE_BSS_STA: u32 = 5;
pub const FILTER_PROMISC: u32 = 1 << 0;
pub const FILTER_CONTROL_AND_MGMT: u32 = 1 << 1;
pub const FILTER_ACCEPT_GRP: u32 = 1 << 2;
pub const FILTER_ACCEPT_BEACON: u32 = 1 << 6;
pub const FILTER_ACCEPT_PROBE_REQ: u32 = 1 << 12;

/// The station MAC. `assoc` carries the association id once there is one;
/// before that the filter admits beacons, which is how the host sees the
/// network it is joining, and after it the firmware tracks the beacon itself.
///
/// **Adding with `is_assoc` set is refused by firmware**, upstream's comment,
/// so an association id is only ever carried on a modify.
pub fn mac_config(action: u32, addr: [u8; 6], assoc: Option<u16>) -> [u8; MAC_LEN] {
    let mut b = [0u8; MAC_LEN];
    w32(&mut b, 0, ID);
    w32(&mut b, 4, action);
    w32(&mut b, 8, MAC_TYPE_BSS_STA);
    b[12..18].copy_from_slice(&addr);
    let mut filter = FILTER_ACCEPT_GRP;
    let assoc = if action == ACTION_ADD { None } else { assoc };
    if assoc.is_none() {
        filter |= FILTER_ACCEPT_BEACON;
    }
    w32(&mut b, 20, filter);
    // wifi_gen_v2: he_support u16, he_ap_support u16, eht_support u32 -- none.
    // nic_not_ack_enabled: set, because NIC ack is an HE feature.
    w32(&mut b, 40, 1);
    // client data at 44: is_assoc u8, esr_transition_timeout u8,
    // medium_sync_delay u16, assoc_id u16, reserved u16, data_policy u16,
    // reserved u16, ctwin u32.
    b[44] = assoc.is_some() as u8;
    w16(&mut b, 48, assoc.unwrap_or(0));
    b
}

// --- LINK_CONFIG_CMD, version 2 ----------------------------------------------

pub const MODIFY_ACTIVE: u32 = 1 << 0;
pub const MODIFY_RATES_INFO: u32 = 1 << 1;
pub const MODIFY_PROTECT_FLAGS: u32 = 1 << 2;
pub const MODIFY_QOS_PARAMS: u32 = 1 << 3;
pub const MODIFY_BEACON_TIMING: u32 = 1 << 4;
pub const MODIFY_HE_PARAMS: u32 = 1 << 5;
pub const MODIFY_ALL: u32 = 0xff;
/// `MAC_QOS_FLG_UPDATE_EDCA`.
pub const QOS_UPDATE_EDCA: u32 = 1 << 0;

/// Basic rates as the firmware's bitmaps: CCK bit `i` is 1, 2, 5.5, 11 Mb/s;
/// OFDM bit `i` is 6, 9, 12, 18, 24, 36, 48, 54. The mandatory sets, which is
/// what upstream fills in when the beacon's basic rates are not consulted.
pub const CCK_MANDATORY: u32 = 0b1111;
pub const OFDM_MANDATORY: u32 = 0b0001_0101;

/// What a link modify carries about the network.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinkParams {
    pub band24: bool,
    /// Beacon interval, TU.
    pub bi: u16,
    /// DTIM period, beacons. Zero when unknown; one is assumed then.
    pub dtim: u8,
}

/// Firmware AC order is BK, BE, VI, VO; the transmit FIFO for each on this
/// generation is 1, 2, 3, 4. The EDCA parameters are 802.11's defaults for a
/// non-WMM station, which is what `softmac` sends as.
const EDCA: [(u16, u16, u8, u8, u16); 4] = [
    (15, 1023, 7, 1, 0),  // BK
    (15, 1023, 3, 2, 0),  // BE
    (7, 15, 2, 3, 94 * 32), // VI
    (3, 7, 2, 4, 47 * 32),  // VO
];

fn link_common(action: u32, addr: [u8; 6]) -> [u8; LINK_LEN] {
    let mut b = [0u8; LINK_LEN];
    w32(&mut b, 0, action);
    w32(&mut b, 4, ID); // link_id
    w32(&mut b, 8, ID); // mac_id
    w32(&mut b, 12, CTXT_INVALID); // phy_id, until the link is activated
    b[16..22].copy_from_slice(&addr);
    b[166] = 0; // spec_link_id
    b
}

/// A link on the MAC, with no PHY yet: the shape upstream adds with.
pub fn link_add(addr: [u8; 6]) -> [u8; LINK_LEN] {
    link_common(ACTION_ADD, addr)
}

pub fn link_remove(addr: [u8; 6]) -> [u8; LINK_LEN] {
    link_common(ACTION_REMOVE, addr)
}

/// Activate or deactivate the link on PHY context `ID`, carrying rates, slot
/// time, EDCA and beacon timing. The PHY id is honoured only while the link is
/// inactive, upstream's note, which is why activation is the one modify that
/// names it.
pub fn link_modify(addr: [u8; 6], active: bool, p: &LinkParams) -> [u8; LINK_LEN] {
    let mut b = link_common(ACTION_MODIFY, addr);
    w32(&mut b, 12, if active { ID } else { CTXT_INVALID });
    w32(
        &mut b,
        24,
        MODIFY_ACTIVE | MODIFY_RATES_INFO | MODIFY_PROTECT_FLAGS | MODIFY_QOS_PARAMS | MODIFY_BEACON_TIMING,
    );
    w32(&mut b, 28, active as u32);
    if !active {
        b[32] = 1; // block_tx
    }
    w32(&mut b, 36, if p.band24 { CCK_MANDATORY } else { 0 });
    w32(&mut b, 40, OFDM_MANDATORY);
    w32(&mut b, 44, 0); // cck_short_preamble
    w32(&mut b, 48, (!p.band24) as u32); // short_slot: always on 5 GHz
    w32(&mut b, 52, 0); // protection_flags
    w32(&mut b, 56, QOS_UPDATE_EDCA);
    for (i, &(cw_min, cw_max, aifsn, fifo, txop)) in EDCA.iter().enumerate() {
        let at = 60 + i * 8;
        w16(&mut b, at, cw_min);
        w16(&mut b, at + 2, cw_max);
        b[at + 4] = aifsn;
        b[at + 5] = 1 << fifo;
        w16(&mut b, at + 6, txop);
    }
    let bi = if p.bi == 0 { 100 } else { p.bi } as u32;
    let dtim = if p.dtim == 0 { 1 } else { p.dtim } as u32;
    w32(&mut b, 136, bi);
    w32(&mut b, 140, bi * dtim);
    b
}

// --- STA_CONFIG_CMD, versions 1 and 2 ----------------------------------------

pub const STATION_TYPE_PEER: u32 = 0;

/// The access point as a peer station on link `ID`. There is no action: the
/// same command adds and modifies, and `assoc` is what changes between them.
pub fn sta_config(len: usize, peer: [u8; 6], assoc: Option<u16>) -> Vec<u8> {
    let mut b = alloc::vec![0u8; len];
    w32(&mut b, 0, STA_ID);
    w32(&mut b, 4, ID); // link_id
    b[8..14].copy_from_slice(&peer); // peer_mld_address
    b[16..22].copy_from_slice(&peer); // peer_link_address
    w32(&mut b, 24, STATION_TYPE_PEER);
    w32(&mut b, 28, assoc.unwrap_or(0) as u32);
    // beamform_flags 32, mfp 36, mimo 40, mimo_protection 44, ack_enabled 48,
    // trig_rnd_alloc 52, tx_ampdu_spacing 56, tx_ampdu_max_size 60,
    // sp_length 64, uapsd_acs 68, pkt_ext 72..92, htc_flags 92: all zero.
    b
}

pub fn sta_remove() -> [u8; 4] {
    STA_ID.to_le_bytes()
}

// --- SESSION_PROTECTION_CMD, version 2 ---------------------------------------

pub const SESSION_PROTECT_CONF_ASSOC: u32 = 0;

/// Milliseconds to time units, upstream's `MSEC_TO_TU`.
pub const fn msec_to_tu(ms: u32) -> u32 {
    ms * 1000 / 1024
}

/// Hold the channel for `duration_tu`, so a scan or a sleep does not take the
/// radio away between the authentication and the association response.
pub fn session_protection(action: u32, duration_tu: u32) -> [u8; SPROT_LEN] {
    let mut b = [0u8; SPROT_LEN];
    w32(&mut b, 0, ID); // the link's id and colour
    w32(&mut b, 4, action);
    w32(&mut b, 8, SESSION_PROTECT_CONF_ASSOC);
    w32(&mut b, 12, duration_tu);
    // repetition_count 16, interval 20: one-shot.
    b
}

// --- SCD_QUEUE_CONFIG_CMD, version 3 -----------------------------------------

pub const SCD_ADD: u32 = 0;
pub const SCD_REMOVE: u32 = 1;

/// `TFD_QUEUE_CB_SIZE(n)`: log2 of the entry count, less three.
pub const fn cb_size(entries: usize) -> u32 {
    (entries.trailing_zeros()) - 3
}

pub fn scd_queue_add(sta_mask: u32, tid: u8, entries: usize, bc_pa: u64, tfd_pa: u64) -> [u8; SCD_LEN] {
    let mut b = [0u8; SCD_LEN];
    w32(&mut b, 0, SCD_ADD);
    w32(&mut b, 4, sta_mask);
    b[8] = tid;
    w32(&mut b, 12, 0); // flags
    w32(&mut b, 16, cb_size(entries));
    w64(&mut b, 20, bc_pa);
    w64(&mut b, 28, tfd_pa);
    b
}

pub fn scd_queue_remove(sta_mask: u32, tid: u8) -> [u8; SCD_LEN] {
    let mut b = [0u8; SCD_LEN];
    w32(&mut b, 0, SCD_REMOVE);
    w32(&mut b, 4, sta_mask);
    b[8] = tid;
    b
}

/// The reply: which queue the firmware gave and where its write pointer starts.
pub fn scd_queue_rsp(payload: &[u8]) -> Option<(u16, u16)> {
    if payload.len() < SCD_RSP_LEN {
        return None;
    }
    let q = u16::from_le_bytes([payload[0], payload[1]]);
    let wr = u16::from_le_bytes([payload[4], payload[5]]);
    Some((q, wr))
}

// --- TX_CMD, version 10 ------------------------------------------------------

pub const TX_FLAGS_CMD_RATE: u16 = 1 << 0;
pub const TX_FLAGS_ENCRYPT_DIS: u16 = 1 << 1;
pub const TX_FLAGS_HIGH_PRI: u16 = 1 << 2;
const OFFLD_MH_SIZE: u32 = 8;
const OFFLD_PAD: u32 = 13;

/// `rate_n_flags`, in the format firmware past TX notification 6 reads: the
/// modulation type at bit 8 (CCK 0, legacy OFDM 1), the rate index in the low
/// three bits, and the antenna at bit 14. The lowest legacy rate of the band,
/// on one antenna.
pub fn legacy_rate(band24: bool, ant: u8) -> u32 {
    let ant = if ant == 0 { 1 } else { ant & 0x3 };
    let modulation = if band24 { 0 } else { 1 << 8 };
    modulation | ((ant as u32) << 14)
}

/// How long an 802.11 header is, from its frame control.
pub fn hdr_len(frame: &[u8]) -> usize {
    if frame.len() < 2 {
        return frame.len();
    }
    let fc = u16::from_le_bytes([frame[0], frame[1]]);
    let ftype = (fc >> 2) & 0x3;
    let mut n = 24;
    if ftype == 2 {
        // Data: a fourth address when both DS bits are set, a QoS control word
        // when the subtype's QoS bit is set.
        if fc & 0x0300 == 0x0300 {
            n += 6;
        }
        if fc & 0x0080 != 0 {
            n += 2;
        }
    } else if ftype == 1 {
        // Control frames are shorter and never go out through here.
        n = 16;
    }
    n.min(frame.len())
}

pub fn is_mgmt(frame: &[u8]) -> bool {
    frame.len() >= 2 && (frame[0] >> 2) & 0x3 == 0
}

/// Lay one frame out as the firmware fetches it, into a slot the caller owns:
///
/// ```text
///   [0..8)    wide command header   TX_CMD, LONG_GROUP, slot index, queue id
///   [8..36)   the TX command        len, flags, offload, dram_info, rate
///   [36..)    the 802.11 header     copied, padded to four
///   body      the rest of the frame
/// ```
///
/// Answers `(header bytes, body offset, body length)`. The first transmit
/// buffer is the first twenty bytes, the second is the rest of the command and
/// the header, the third is the body -- upstream's three, because the firmware
/// reads the command and the header out of the first two and DMAs the body.
pub fn tx_slot(slot: &mut [u8], idx: u8, queue: u8, frame: &[u8], rate: u32, flags: u16) -> Option<(usize, usize, usize)> {
    let h = hdr_len(frame);
    let padded = (h + 3) & !3;
    let body = frame.len() - h;
    let need = cmd::HDR_WIDE + TX_CMD_LEN + padded + body;
    if need > slot.len() || frame.len() > 0xffff {
        return None;
    }
    for x in slot[..need].iter_mut() {
        *x = 0;
    }
    slot[0] = TX_CMD;
    slot[1] = cmd::LONG_GROUP;
    slot[2] = idx;
    slot[3] = queue;
    // Length and version bytes are the queue's business, not the firmware's,
    // for a data frame; zero is what upstream leaves.
    let c = cmd::HDR_WIDE;
    w16(slot, c, frame.len() as u16);
    w16(slot, c + 2, flags);
    let mut offload = ((h / 2) as u32) << OFFLD_MH_SIZE;
    if h % 4 != 0 {
        offload |= 1 << OFFLD_PAD;
    }
    w32(slot, c + 4, offload);
    // dram_info 12..20: zero. rate at 20.
    w32(slot, c + 20, rate);
    let hdr_at = c + TX_CMD_LEN;
    slot[hdr_at..hdr_at + h].copy_from_slice(&frame[..h]);
    let body_at = hdr_at + padded;
    slot[body_at..body_at + body].copy_from_slice(&frame[h..]);
    Some((h, body_at, body))
}

// --- a transmit queue --------------------------------------------------------

/// Entries per queue. A power of two, as the write pointer arithmetic needs,
/// and small: the station sends a handful of frames at a time and the
/// byte-count table has room for 1024.
pub const Q_ENTRIES: usize = 64;
/// Bytes per slot: header, command, and a frame up to the radio's `max_frame`,
/// rounded to the 64 the first buffer wants aligned.
pub const SLOT: usize = 2560;
/// The byte-count table on this family: 1024 sixteen-bit entries.
pub const BC_ENTRIES: usize = 1024;
/// The management TID, which is how a frame that is not data is queued.
pub const MGMT_TID: u8 = 15;
pub const DATA_TID: u8 = 0;

/// One hardware transmit queue: its descriptors, byte-count table and slots,
/// and the firmware's number for it once the add has been answered.
pub struct TxQueue {
    pub tid: u8,
    /// What the firmware called it. `None` until `SCD_QUEUE_CONFIG_CMD` answers.
    pub id: Option<u8>,
    tfds: Dma,
    bc: Dma,
    slots: Dma,
    cur: usize,
    cur_hw: u32,
    pub sent: u32,
}

impl TxQueue {
    pub fn new(tid: u8) -> Option<TxQueue> {
        Some(TxQueue {
            tid,
            id: None,
            tfds: Dma::new(Q_ENTRIES * super::ctxt::TFD_SIZE, 4096)?,
            bc: Dma::new(BC_ENTRIES * 2, 4096)?,
            slots: Dma::new(Q_ENTRIES * SLOT, 4096)?,
            cur: 0,
            cur_hw: 0,
            sent: 0,
        })
    }

    /// The command that asks the firmware for this queue.
    pub fn add_body(&self) -> [u8; SCD_LEN] {
        scd_queue_add(1 << STA_ID, self.tid, Q_ENTRIES, self.bc.pa(), self.tfds.pa())
    }

    pub fn remove_body(&self) -> [u8; SCD_LEN] {
        scd_queue_remove(1 << STA_ID, self.tid)
    }

    /// Take the firmware's answer: its queue number and the write pointer the
    /// ring starts at, which is the sequence number's slot and not zero.
    pub fn activated(&mut self, rsp: &[u8]) -> Result<(), &'static str> {
        let (q, wr) = scd_queue_rsp(rsp).ok_or("the queue reply is shorter than eight bytes")?;
        if q > 31 {
            return Err("the firmware named a queue past 31");
        }
        self.id = Some(q as u8);
        self.cur_hw = wr as u32 % cmd::HW_WRAP;
        self.cur = wr as usize % Q_ENTRIES;
        Ok(())
    }

    /// Write one frame into the ring and ring for it.
    ///
    /// # Safety
    /// `bar0` must be a mapped aperture for a part whose firmware is alive and
    /// which has been told about this queue.
    pub unsafe fn send(&mut self, bar0: u64, frame: &[u8], rate: u32, flags: u16) -> Result<(), &'static str> {
        let id = self.id.ok_or("the queue has not been activated")?;
        let idx = self.cur;
        let slot_pa = self.slots.pa() + (idx * SLOT) as u64;
        let (h, body_at, body) = {
            let s = &mut self.slots.as_mut_slice()[idx * SLOT..(idx + 1) * SLOT];
            tx_slot(s, idx as u8, id, frame, rate, flags).ok_or("the frame does not fit a slot")?
        };
        let padded = (h + 3) & !3;
        let tb1 = cmd::HDR_WIDE + TX_CMD_LEN + padded - cmd::FIRST_TB;
        let n: u16 = if body > 0 { 3 } else { 2 };
        {
            let at = idx * super::ctxt::TFD_SIZE;
            let d = &mut self.tfds.as_mut_slice()[at..at + super::ctxt::TFD_SIZE];
            for x in d.iter_mut() {
                *x = 0;
            }
            d[0..2].copy_from_slice(&n.to_le_bytes());
            d[2..4].copy_from_slice(&(cmd::FIRST_TB as u16).to_le_bytes());
            d[4..12].copy_from_slice(&slot_pa.to_le_bytes());
            d[12..14].copy_from_slice(&(tb1 as u16).to_le_bytes());
            d[14..22].copy_from_slice(&(slot_pa + cmd::FIRST_TB as u64).to_le_bytes());
            if n == 3 {
                d[22..24].copy_from_slice(&(body as u16).to_le_bytes());
                d[24..32].copy_from_slice(&(slot_pa + body_at as u64).to_le_bytes());
            }
        }
        // The byte-count entry: the frame's length, with the number of 64-byte
        // chunks of descriptor to fetch (less one) in the top two bits. Three
        // buffers is thirty-two bytes of descriptor, one chunk, zero.
        {
            let e = &mut self.bc.as_mut_slice()[idx * 2..idx * 2 + 2];
            e.copy_from_slice(&(frame.len() as u16 & 0x3fff).to_le_bytes());
        }
        self.cur = (self.cur + 1) % Q_ENTRIES;
        self.cur_hw = (self.cur_hw + 1) % cmd::HW_WRAP;
        self.sent += 1;
        core::ptr::write_volatile((bar0 + HBUS_TARG_WRPTR) as *mut u32, ((id as u32) << 16) | self.cur_hw);
        Ok(())
    }
}

const HBUS_TARG_WRPTR: u64 = 0x400 + 0x060;

/// Where a transmit response says how it went: `iwl_tx_resp.status[0].status`,
/// at byte 36. The low byte is the status; 1 is success, 2 is direct done.
pub fn tx_status(payload: &[u8]) -> Option<u8> {
    if payload.len() < 38 {
        return None;
    }
    Some(payload[36])
}

// --- claims ------------------------------------------------------------------

pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();
    let addr = [0x28, 0xc5, 0xd2, 0x06, 0x00, 0x72];
    let peer = [0x00, 0x11, 0x22, 0x33, 0x44, 0x55];

    let p = phy_context(ACTION_ADD, 36, false);
    out.push(("mld: a PHY context is 32 bytes with the channel at 8 and the band at 12", p.len() == PHY_LEN && p[8] == 36 && p[12] == PHY_BAND_5));
    out.push(("mld: 2.4 GHz is band 1 in a PHY context, 5 GHz band 0", phy_context(ACTION_ADD, 6, true)[12] == PHY_BAND_24));

    let m = mac_config(ACTION_ADD, addr, Some(7));
    out.push(("mld: a MAC config is 64 bytes and an add never claims association", m.len() == MAC_LEN && m[44] == 0 && m[48] == 0));
    let m = mac_config(ACTION_MODIFY, addr, Some(7));
    out.push(("mld: a MAC modify carries is_assoc and the association id at 44 and 48", m[44] == 1 && m[48] == 7 && m[49] == 0));
    out.push(("mld: an unassociated MAC admits beacons and an associated one does not",
        mac_config(ACTION_ADD, addr, None)[20] & (FILTER_ACCEPT_BEACON as u8) != 0 && m[20] & (FILTER_ACCEPT_BEACON as u8) == 0));

    let l = link_add(addr);
    out.push(("mld: a link add is 208 bytes with no PHY", l.len() == LINK_LEN && l[12..16] == [0xff; 4] && l[16..22] == addr));
    let lp = LinkParams { band24: true, bi: 100, dtim: 2 };
    let l = link_modify(addr, true, &lp);
    out.push(("mld: an activating modify names PHY 0, active 1, and the DTIM interval at 140", l[12..16] == [0; 4] && l[28] == 1 && u32::from_le_bytes([l[136], l[137], l[138], l[139]]) == 100 && u32::from_le_bytes([l[140], l[141], l[142], l[143]]) == 200));
    out.push(("mld: the four ACs sit at 60 in BK, BE, VI, VO order with fifos 1..4", l[65] == 1 << 1 && l[73] == 1 << 2 && l[81] == 1 << 3 && l[89] == 1 << 4));
    out.push(("mld: 2.4 GHz carries the CCK mandatory set and long slot; 5 GHz neither",
        l[36] == CCK_MANDATORY as u8 && l[48] == 0 && {
            let l5 = link_modify(addr, true, &LinkParams { band24: false, bi: 100, dtim: 1 });
            l5[36] == 0 && l5[48] == 1
        }));
    let l = link_modify(addr, false, &lp);
    out.push(("mld: a deactivating modify drops the PHY and blocks transmit", l[12..16] == [0xff; 4] && l[28] == 0 && l[32] == 1));

    let s = sta_config(STA_V1_LEN, peer, None);
    out.push(("mld: a v1 station is 96 bytes, peer at 8 and 16, type peer, link 0", s.len() == 96 && s[8..14] == peer && s[16..22] == peer && s[24] == 0 && s[4] == 0));
    out.push(("mld: a v2 station is 104 bytes", sta_config(STA_V2_LEN, peer, Some(3)).len() == 104));
    out.push(("mld: a station modify carries the association id at 28", sta_config(STA_V1_LEN, peer, Some(3))[28] == 3));

    let sp = session_protection(ACTION_ADD, msec_to_tu(900));
    out.push(("mld: a session protection is 24 bytes, conf 0, with 900 ms as 878 TU", sp.len() == SPROT_LEN && sp[8] == 0 && u32::from_le_bytes([sp[12], sp[13], sp[14], sp[15]]) == 878));

    let q = scd_queue_add(1, MGMT_TID, 64, 0x1000, 0x2000);
    out.push(("mld: a queue add is 36 bytes, cb_size 3 for 64 entries, tables at 20 and 28",
        q.len() == SCD_LEN && q[8] == 15 && q[16] == 3 && q[21] == 0x10 && q[20] == 0 && q[29] == 0x20 && q[28] == 0));
    out.push(("mld: cb_size is log2 less three", cb_size(256) == 5 && cb_size(16) == 1));
    out.push(("mld: a queue reply yields the queue number and write pointer", scd_queue_rsp(&[5, 0, 0, 0, 0x34, 0x12, 0, 0]) == Some((5, 0x1234)) && scd_queue_rsp(&[1; 7]).is_none()));

    out.push(("mld: the lowest legacy rate is CCK on 2.4 GHz and OFDM on 5, antenna A", legacy_rate(true, 1) == 0x4000 && legacy_rate(false, 1) == 0x4100 && legacy_rate(false, 0) == 0x4100));

    // A data frame: fc 0x0008 (data, to DS clear), 24-byte header, 10-byte body.
    let mut f = alloc::vec![0u8; 34];
    f[0] = 0x08;
    f[1] = 0x01;
    for (i, x) in f[24..].iter_mut().enumerate() {
        *x = 0xa0 + i as u8;
    }
    out.push(("mld: a plain data header is 24 bytes; QoS is 26; management is 24", hdr_len(&f) == 24 && hdr_len(&[0x88, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) == 26 && hdr_len(&[0xb0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) == 24));
    out.push(("mld: authentication is management and data is not", is_mgmt(&[0xb0, 0]) && !is_mgmt(&f)));
    let mut slot = [0u8; SLOT];
    let r = tx_slot(&mut slot, 9, 3, &f, 0x4100, TX_FLAGS_CMD_RATE | TX_FLAGS_ENCRYPT_DIS);
    out.push(("mld: a slot opens with TX_CMD, LONG_GROUP, its index and queue", slot[0] == TX_CMD && slot[1] == cmd::LONG_GROUP && slot[2] == 9 && slot[3] == 3));
    out.push(("mld: the command carries the frame length, the flags and the rate at 8, 10 and 28",
        slot[8] == 34 && slot[10] == 3 && u32::from_le_bytes([slot[28], slot[29], slot[30], slot[31]]) == 0x4100));
    out.push(("mld: offload assist says twelve header words and no pad for a 24-byte header",
        u32::from_le_bytes([slot[12], slot[13], slot[14], slot[15]]) == 12 << 8));
    out.push(("mld: the header is copied to 36 and the body follows it", r == Some((24, 60, 10)) && slot[36] == 0x08 && slot[60] == 0xa0 && slot[69] == 0xa9));
    let mut qf = alloc::vec![0u8; 30];
    qf[0] = 0x88;
    let r = tx_slot(&mut slot, 0, 0, &qf, 0, 0);
    out.push(("mld: a QoS header is padded to 28 and the pad bit is set", r == Some((26, 64, 4)) && slot[13] & (1 << (OFFLD_PAD - 8)) != 0));
    out.push(("mld: a frame larger than a slot is refused", tx_slot(&mut slot, 0, 0, &alloc::vec![0u8; SLOT], 0, 0).is_none()));
    out.push(("mld: a transmit status is read at 36", tx_status(&[0u8; 38]).is_some() && tx_status(&[0u8; 36]).is_none()));

    // The versions this part reported on the laptop pass; each refusal fires.
    let real = Versions { phy: Some(4), mac: Some(2), link: Some(2), sta: None, tx: Some(10), tx_notif: Some(7), scd: Some(3), sprot: Some(2), ultra_hb: true, mld: true };
    out.push(("mld: the GF63's firmware versions are accepted and want a v1 station", real.check().is_ok() && real.sta_len() == STA_V1_LEN));
    out.push(("mld: a non-MLD image is refused by name", Versions { mld: false, ..real }.check().is_err()));
    out.push(("mld: a TX_CMD other than version 10 is refused", Versions { tx: Some(9), ..real }.check().is_err()));
    out.push(("mld: a queue-config other than version 3 is refused", Versions { scd: Some(0), ..real }.check().is_err()));
    out.push(("mld: a version 2 station wants 104 bytes", Versions { sta: Some(2), ..real }.sta_len() == STA_V2_LEN));
    out
}
