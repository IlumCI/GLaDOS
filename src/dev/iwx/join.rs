//! Joining an access point, through the firmware's legacy contexts.
//!
//! `scan.rs` is where this driver stopped, and the reason it stopped there is
//! that joining is four contexts and a queue where scanning is one command.
//! **This image does not declare `MLD_API_SUPPORT`** -- capability 121 is
//! absent from `so-a0-hr-b0-89`, read off the file on the host -- so the
//! contexts are the ones Linux's mvm and OpenBSD's `iwx(4)` send such a
//! firmware: `MAC_CONTEXT_CMD`, `BINDING_CONTEXT_CMD` and `ADD_STA`, with
//! `PHY_CONTEXT_CMD` in front of them. The first version of this file was
//! written against the MLD-era `MAC_CONFIG`/`LINK_CONFIG`/`STA_CONFIG` trio on
//! the strength of a wrong note, and `Versions::check` refused it on the part
//! before a byte was sent -- which is what the check is for.
//!
//! ### The sequence, `iwx_auth`'s
//!
//! ```text
//!   PHY_CONTEXT_CMD      add       the channel
//!   RLC_CONFIG_CMD                 which receive chains it listens on
//!   MAC_CONTEXT_CMD      add       who we are, not associated, hear beacons
//!   BINDING_CONTEXT_CMD  add       tie the MAC to the PHY
//!   ADD_STA                        the access point, as a link station
//!   SCD_QUEUE_CONFIG     add x2    a management queue and a data queue
//!   SESSION_PROTECTION   add       stay on channel for the handshake
//! ```
//!
//! Then `mlme` authenticates and associates through `tx`; `associated` is
//! `iwx_run`'s half -- `ADD_STA` as an update, `MAC_CONTEXT_CMD` modified
//! with the association id -- and `left` takes it all down in reverse.
//!
//! ### Layouts are facts about the firmware, and are asserted
//!
//! Every structure here is a packed little-endian layout read from Intel's
//! headers (dual BSD/GPLv2, the BSD arm, see `NOTICE.md`) and cross-checked
//! against OpenBSD's `if_iwxreg.h`, and every builder asserts the size it
//! produces -- a structure one field short is not a smaller command, it is a
//! different command the firmware parses to the end of. Versions are checked
//! against the image's command-version table before anything is sent, and a
//! version this module does not know is refused by name rather than sent in
//! the hope that the layout did not move.
//!
//! ### What is deliberately not done
//!
//! **Rates are fixed at the lowest legacy rate**, 1 Mb/s on 2.4 GHz and 6 Mb/s
//! on 5 GHz, with `IWL_TX_FLAGS_CMD_RATE` set on every frame. Letting the
//! firmware choose needs `TLC_MNG_CONFIG_CMD` and the station's whole rate
//! table, and a link that works slowly is the rung before a link that works
//! fast. **Keys stay in software**: `wlan.rs` keeps `hw_ccmp` false, so
//! `softmac` encrypts and the frame goes out with `IWL_TX_FLAGS_ENCRYPT_DIS`.
//! And `iwx_run`'s `SF_CFG_CMD`, `MCAST_FILTER_CMD` and power command are not
//! sent: each tunes a link that already works, and none is needed for one to.

use alloc::vec::Vec;

use super::cmd;
use crate::dev::dma::Dma;

// --- groups and commands -----------------------------------------------------

pub const MAC_CONF_GROUP: u8 = 0x3;
pub const DATA_PATH_GROUP: u8 = 0x5;

/// In `LONG_GROUP`.
pub const PHY_CONTEXT_CMD: u8 = 0x08;
pub const ADD_STA: u8 = 0x18;
pub const REMOVE_STA: u8 = 0x19;
pub const TX_CMD: u8 = 0x1c;
pub const MAC_CONTEXT_CMD: u8 = 0x28;
pub const BINDING_CONTEXT_CMD: u8 = 0x2b;

/// In `MAC_CONF_GROUP`.
pub const SESSION_PROTECTION_CMD: u8 = 0x05;
pub const SESSION_PROTECTION_NOTIF: u8 = 0xfb;

/// In `DATA_PATH_GROUP`.
pub const RLC_CONFIG_CMD: u8 = 0x08;
pub const SCD_QUEUE_CONFIG_CMD: u8 = 0x17;

/// Capability bits the join path rests on.
pub const CAPA_ULTRA_HB_CHANNELS: usize = 48;
pub const CAPA_MLD_API: usize = 121;
/// Two LMACs, one per band: 5 GHz contexts live on LMAC 1 on such a part.
pub const CAPA_CDB_SUPPORT: usize = 40;

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
pub const RLC_LEN: usize = 32;
pub const MAC_LEN: usize = 148;
pub const BINDING_LEN: usize = 28;
pub const STA_LEN: usize = 48;
pub const SPROT_LEN: usize = 24;
pub const SCD_LEN: usize = 36;
pub const SCD_RSP_LEN: usize = 8;
/// The TX command proper, between the command header and the 802.11 header.
pub const TX_CMD_LEN: usize = 28;
/// A transmit frame's command header: `iwl_cmd_header`, cmd, group, sequence.
pub const TX_HDR: usize = 4;

// --- what the image has to say -----------------------------------------------

/// The command versions the join path sends by, read off the image once.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Versions {
    pub phy: Option<u8>,
    pub mac_ctx: Option<u8>,
    pub binding: Option<u8>,
    pub add_sta: Option<u8>,
    pub rlc: Option<u8>,
    pub tx: Option<u8>,
    pub tx_notif: Option<u8>,
    pub scd: Option<u8>,
    pub sprot: Option<u8>,
    pub ultra_hb: bool,
    pub mld: bool,
    pub cdb: bool,
}

impl Versions {
    pub fn of(image: &super::fw::Image) -> Versions {
        let l = cmd::LONG_GROUP;
        Versions {
            phy: image.cmd_ver(l, PHY_CONTEXT_CMD),
            mac_ctx: image.cmd_ver(l, MAC_CONTEXT_CMD),
            binding: image.cmd_ver(l, BINDING_CONTEXT_CMD),
            add_sta: image.cmd_ver(l, ADD_STA),
            rlc: image.cmd_ver(DATA_PATH_GROUP, RLC_CONFIG_CMD),
            tx: image.cmd_ver(l, TX_CMD),
            tx_notif: image.notif_ver(l, TX_CMD),
            scd: image.cmd_ver(DATA_PATH_GROUP, SCD_QUEUE_CONFIG_CMD),
            sprot: image.cmd_ver(MAC_CONF_GROUP, SESSION_PROTECTION_CMD),
            ultra_hb: image.has_capa(CAPA_ULTRA_HB_CHANNELS),
            mld: image.has_capa(CAPA_MLD_API),
            cdb: image.has_capa(CAPA_CDB_SUPPORT),
        }
    }

    /// Whether this module knows how to talk to this image, and if not, which
    /// command it would get wrong. A version absent from the table is the
    /// firmware's default, which for these commands is the first layout.
    pub fn check(&self) -> Result<(), &'static str> {
        if self.mld {
            return Err("the image declares the MLD API, and this module speaks the legacy contexts its predecessor did not");
        }
        if !self.ultra_hb {
            return Err("the image lacks ULTRA_HB_CHANNELS, so its channel info is the four-byte v1 this module does not build");
        }
        match self.phy.unwrap_or(1) {
            3 | 4 => {}
            _ => return Err("PHY_CONTEXT_CMD is a version other than 3 or 4"),
        }
        // The three legacy contexts have no entry in this image's table, which
        // is their first layout; a later one would be a different structure.
        if self.mac_ctx.unwrap_or(1) != 1 {
            return Err("MAC_CONTEXT_CMD is past version 1");
        }
        if self.binding.unwrap_or(1) > 2 {
            return Err("BINDING_CONTEXT_CMD is past version 2");
        }
        if self.add_sta.unwrap_or(10) > 12 {
            return Err("ADD_STA is past version 12");
        }
        match self.rlc.unwrap_or(1) {
            1 | 2 => {}
            _ => return Err("RLC_CONFIG_CMD is past version 2"),
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

    /// Whether the receive chains travel in `RLC_CONFIG_CMD` (version 2) or in
    /// the PHY context's own `rxchain_info` (anything older).
    pub fn rlc_separate(&self) -> bool {
        self.rlc == Some(2)
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
/// `IWX_PHY_VHT_CTRL_POS_1_BELOW`, which is what a 20 MHz context carries.
pub const CTRL_POS_1_BELOW: u8 = 0;

/// The PHY context: one channel, 20 MHz wide.
///
/// `rxchain` is the receive-chain word, carried here only when the image's
/// `RLC_CONFIG_CMD` is older than version 2 -- at 2 the firmware ignores this
/// field and `rlc_config` carries it instead, which is upstream's rule.
/// `cdb` is the dual-LMAC capability: on such a part the 5 GHz band is LMAC 1,
/// and a PHY context on the wrong LMAC is accepted and never hears anything.
pub fn phy_context(action: u32, channel: u8, band24: bool, cdb: bool, rxchain: Option<u32>) -> [u8; PHY_LEN] {
    let mut b = [0u8; PHY_LEN];
    w32(&mut b, 0, ID);
    w32(&mut b, 4, action);
    // iwl_fw_channel_info, v2: channel u32, band u8, width u8, ctrl_pos u8.
    w32(&mut b, 8, channel as u32);
    b[12] = if band24 { PHY_BAND_24 } else { PHY_BAND_5 };
    b[13] = CHANNEL_MODE20;
    b[14] = CTRL_POS_1_BELOW;
    w32(&mut b, 16, if cdb && !band24 { 1 } else { 0 }); // lmac_id
    w32(&mut b, 20, rxchain.unwrap_or(0)); // rxchain_info
    w32(&mut b, 24, 0); // dsp_cfg_flags
    // 28..32 reserved. (Linux's header calls it secondary_ctrl_chnl_loc for
    // later versions; OpenBSD leaves it zero on this one, and so does this.)
    b
}

// --- MAC_CONTEXT_CMD, the legacy layout -------------------------------------

pub const MAC_TYPE_BSS_STA: u32 = 5;
pub const TSF_ID_A: u32 = 0;
/// `MAC_FILTER_*` of the legacy command: beacons at bit 6, probe requests at
/// bit 12. (The MLD-era `MAC_CONFIG_CMD` packs them at 3 and 5; it is not what
/// this image speaks, and the two numberings were confused here once.)
pub const FILTER_ACCEPT_GRP: u32 = 1 << 2;
pub const FILTER_IN_BEACON: u32 = 1 << 6;
pub const FILTER_IN_PROBE_REQUEST: u32 = 1 << 12;
/// `IWX_MAC_FLG_SHORT_SLOT` is bit 4 of `short_slot`, not a boolean.
pub const MAC_FLG_SHORT_SLOT: u32 = 1 << 4;
/// `MAC_QOS_FLG_UPDATE_EDCA`.
pub const QOS_UPDATE_EDCA: u32 = 1 << 0;

/// Basic rates as the firmware's bitmaps: CCK bit `i` is 1, 2, 5.5, 11 Mb/s;
/// OFDM bit `i` is 6, 9, 12, 18, 24, 36, 48, 54. `iwx_ack_rates` with no
/// beacon rate set consulted ends at exactly these: every CCK rate, and OFDM
/// 6, 12 and 24.
pub const CCK_MANDATORY: u32 = 0b1111;
pub const OFDM_MANDATORY: u32 = 0b0001_0101;

/// What the MAC context carries about the network.
#[derive(Clone, Copy, Debug, Default)]
pub struct MacParams {
    pub band24: bool,
    /// Beacon interval, TU. Zero when unknown; 100 is assumed then.
    pub bi: u16,
    /// DTIM period, beacons. Zero when unknown; one is assumed then.
    pub dtim: u8,
    /// The association id, once there is one. `None` before association,
    /// and that is also what keeps beacons flowing to the host.
    pub assoc: Option<u16>,
}

/// The EDCA table, indexed by the **transmit FIFO** the firmware wants it
/// under -- `iwx_mac_ctxt_cmd_common` writes `cmd->ac[txf]`, and on this
/// generation the FIFOs are BK 1, BE 2, VI 3, VO 4 with 0 the command FIFO.
/// 802.11's defaults for a non-WMM station, which is what `softmac` sends as.
const EDCA: [(usize, u16, u16, u8, u16); 4] = [
    (1, 15, 1023, 7, 0),      // BK
    (2, 15, 1023, 3, 0),      // BE
    (3, 7, 15, 2, 94 * 32),   // VI
    (4, 3, 7, 2, 47 * 32),    // VO
];

/// `iwl_mac_ctx_cmd`: the common header, then `iwl_mac_data_sta` in the union,
/// whose size is the largest member's (`p2p_sta`, 48). 148 bytes in all.
pub fn mac_context(action: u32, addr: [u8; 6], bssid: [u8; 6], p: &MacParams) -> [u8; MAC_LEN] {
    let mut b = [0u8; MAC_LEN];
    w32(&mut b, 0, ID);
    w32(&mut b, 4, action);
    if action == ACTION_REMOVE {
        return b;
    }
    w32(&mut b, 8, MAC_TYPE_BSS_STA);
    w32(&mut b, 12, TSF_ID_A);
    b[16..22].copy_from_slice(&addr);
    b[24..30].copy_from_slice(&bssid);
    w32(&mut b, 32, if p.band24 { CCK_MANDATORY } else { 0 });
    w32(&mut b, 36, OFDM_MANDATORY);
    w32(&mut b, 40, 0); // protection_flags
    w32(&mut b, 44, 0); // cck_short_preamble
    w32(&mut b, 48, if p.band24 { 0 } else { MAC_FLG_SHORT_SLOT });
    // Beacons reach the host until the firmware has an association and a
    // DTIM to track them by; upstream's condition exactly.
    let mut filter = FILTER_ACCEPT_GRP;
    if p.assoc.is_none() || p.dtim == 0 {
        filter |= FILTER_IN_BEACON;
    }
    w32(&mut b, 52, filter);
    w32(&mut b, 56, 0); // qos_flags: not a QoS station
    for &(txf, cw_min, cw_max, aifsn, txop) in EDCA.iter() {
        let at = 60 + txf * 8;
        w16(&mut b, at, cw_min);
        w16(&mut b, at + 2, cw_max);
        b[at + 4] = aifsn;
        b[at + 5] = 1 << txf;
        w16(&mut b, at + 6, txop);
    }
    // iwl_mac_data_sta at 100: is_assoc, dtim_time, dtim_tsf u64, bi,
    // reserved, dtim_interval, data_policy, listen_interval, assoc_id,
    // assoc_beacon_arrive_time. The DTIM time and TSF are left zero: they are
    // the beacon's own timestamps, which `mlme` does not carry up, and
    // firmware tracks the beacon itself once associated.
    let bi = if p.bi == 0 { 100 } else { p.bi } as u32;
    let dtim = if p.dtim == 0 { 1 } else { p.dtim } as u32;
    w32(&mut b, 100, p.assoc.is_some() as u32);
    w32(&mut b, 116, bi);
    w32(&mut b, 124, bi * dtim);
    w32(&mut b, 128, 0); // data_policy
    w32(&mut b, 132, 10); // listen_interval, upstream's constant
    w32(&mut b, 136, p.assoc.unwrap_or(0) as u32);
    b
}

// --- BINDING_CONTEXT_CMD, version 2 ------------------------------------------

/// Tie MAC `ID` to PHY `ID`. Version 2 because `BINDING_CDB_SUPPORT` (capa
/// 39) is declared; `lmac_id` is 0 on a part with one LMAC, and this part has
/// one (`CDB_SUPPORT`, capa 40, is not declared).
pub fn binding(action: u32, lmac: u32) -> [u8; BINDING_LEN] {
    let mut b = [0u8; BINDING_LEN];
    w32(&mut b, 0, ID); // the PHY's id and colour
    w32(&mut b, 4, action);
    w32(&mut b, 8, ID); // macs[0]
    w32(&mut b, 12, CTXT_INVALID);
    w32(&mut b, 16, CTXT_INVALID);
    w32(&mut b, 20, ID); // phy
    w32(&mut b, 24, lmac);
    b
}

// --- ADD_STA, version 10, and REMOVE_STA -------------------------------------

/// `IWX_STA_LINK`: an ordinary peer.
pub const STA_LINK: u8 = 0;
pub const STA_FLG_FAT_EN_MSK: u32 = 3 << 26;
pub const STA_FLG_MIMO_EN_MSK: u32 = 3 << 28;
pub const ADD_STA_SUCCESS: u8 = 1;

/// The access point as station `STA_ID`. `update` is the modify after
/// association; the address goes only on the add, as upstream does. 20 MHz,
/// one spatial stream, no aggregation: the masks say those fields are being
/// set and the zeros say to what.
pub fn add_sta(update: bool, bssid: [u8; 6]) -> [u8; STA_LEN] {
    let mut b = [0u8; STA_LEN];
    b[0] = update as u8; // add_modify
    w32(&mut b, 4, ID); // mac_id_n_color
    if !update {
        b[8..14].copy_from_slice(&bssid);
    }
    b[16] = STA_ID as u8;
    w32(&mut b, 24, 0); // station_flags
    w32(&mut b, 28, STA_FLG_FAT_EN_MSK | STA_FLG_MIMO_EN_MSK);
    b[37] = STA_LINK; // station_type
    // assoc_id 38, beamform_flags 40, tfd_queue_msk 42, rx_ba_window 46,
    // sp_length 47, uapsd_acs 48: zero. (Offsets per the header: 0 add_modify,
    // 1 awake_acs, 2 tid_disable_tx, 4 mac_id_n_color, 8 addr, 14 res,
    // 16 sta_id, 17 modify_mask, 18 res, 20 station_flags, 24 flags_msk.)
    b
}

/// The reply's status, low byte of its first word.
pub fn add_sta_ok(reply: &[u8]) -> bool {
    reply.len() >= 4 && reply[0] & 0xff == ADD_STA_SUCCESS
}

pub fn sta_remove() -> [u8; 4] {
    [STA_ID as u8, 0, 0, 0]
}

// --- RLC_CONFIG_CMD, version 2 ------------------------------------------------

/// Which receive chains a PHY context listens on. With `RLC_CONFIG_CMD` at
/// version 2 the PHY context's own `rxchain_info` is ignored and this carries
/// it instead -- one valid-antenna mask, one idle chain, one active chain, the
/// values `iwx_auth` sends. 32 bytes: phy_id, rlc {rx_chain_info, reserved},
/// sad {4 words}, flags u8, reserved[3].
pub fn rlc_config(rx_ant: u8) -> [u8; RLC_LEN] {
    let mut b = [0u8; RLC_LEN];
    w32(&mut b, 0, ID);
    w32(&mut b, 4, rx_chain_info(rx_ant));
    b
}

/// `valid << 1 | idle << 10 | active << 12`, with one chain each way.
pub fn rx_chain_info(rx_ant: u8) -> u32 {
    let valid = if rx_ant == 0 { 1 } else { rx_ant & 0x3 } as u32;
    (valid << 1) | (1 << 10) | (1 << 12)
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
///   [0..4)    command header        TX_CMD, group 0, slot index, queue id
///   [4..32)   the TX command        len, flags, offload, dram_info, rate
///   [32..)    the 802.11 header     copied, padded to four
///   body      the rest of the frame
/// ```
///
/// **The header is the four-byte `iwl_cmd_header`, not the eight-byte wide one
/// the command queue frames with.** Upstream's `iwl_device_tx_cmd` carries the
/// short header, with the slot index in the low byte of its sequence word and
/// the queue in the high, and `tx-gen2.c` sizes its second buffer from that.
/// Written with the wide header first, from memory, and caught by reading.
///
/// Answers `(header bytes, body offset, body length)`. The first transmit
/// buffer is the first twenty bytes, the second is the rest of the command and
/// the header, the third is the body -- upstream's three, because the firmware
/// reads the command and the header out of the first two and DMAs the body.
pub fn tx_slot(slot: &mut [u8], idx: u8, queue: u8, frame: &[u8], rate: u32, flags: u16) -> Option<(usize, usize, usize)> {
    let h = hdr_len(frame);
    let padded = (h + 3) & !3;
    let body = frame.len() - h;
    let need = TX_HDR + TX_CMD_LEN + padded + body;
    if need > slot.len() || frame.len() > 0xffff {
        return None;
    }
    for x in slot[..need].iter_mut() {
        *x = 0;
    }
    slot[0] = TX_CMD;
    slot[1] = 0; // group: upstream leaves it, and the data path does not read it
    slot[2] = idx;
    slot[3] = queue;
    let c = TX_HDR;
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
        let tb1 = TX_HDR + TX_CMD_LEN + padded - cmd::FIRST_TB;
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
    let r32 = |b: &[u8], at: usize| u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);

    let p = phy_context(ACTION_ADD, 36, false, false, None);
    out.push(("join: a PHY context is 32 bytes with the channel at 8, the band at 12 and ctrl_pos below", p.len() == PHY_LEN && p[8] == 36 && p[12] == PHY_BAND_5 && p[14] == 0 && p[28] == 0));
    out.push(("join: 2.4 GHz is band 1 in a PHY context, 5 GHz band 0", phy_context(ACTION_ADD, 6, true, false, None)[12] == PHY_BAND_24));
    out.push(("join: on a dual-LMAC part 5 GHz is LMAC 1 and 2.4 GHz LMAC 0; single-LMAC is always 0", phy_context(ACTION_ADD, 36, false, true, None)[16] == 1 && phy_context(ACTION_ADD, 6, true, true, None)[16] == 0 && p[16] == 0));
    out.push(("join: the receive chains land at 20 only when handed in", r32(&phy_context(ACTION_ADD, 36, false, false, Some(0x1406)), 20) == 0x1406 && r32(&p, 20) == 0));
    out.push(("join: rx_chain_info is valid<<1 | one idle at 10 | one active at 12", rx_chain_info(0b11) == (3 << 1) | (1 << 10) | (1 << 12) && rx_chain_info(0) == (1 << 1) | (1 << 10) | (1 << 12)));
    let rl = rlc_config(0b11);
    out.push(("join: an RLC config is 32 bytes, PHY 0, chains at 4", rl.len() == RLC_LEN && r32(&rl, 0) == 0 && r32(&rl, 4) == rx_chain_info(0b11)));

    let mp = MacParams { band24: true, bi: 100, dtim: 2, assoc: None };
    let m = mac_context(ACTION_ADD, addr, peer, &mp);
    out.push(("join: a MAC context is 148 bytes: type 5, tsf A, our address at 16, the BSSID at 24", m.len() == MAC_LEN && r32(&m, 8) == MAC_TYPE_BSS_STA && r32(&m, 12) == 0 && m[16..22] == addr && m[24..30] == peer));
    out.push(("join: 2.4 GHz carries the CCK mandatory set and long slot; 5 GHz neither",
        r32(&m, 32) == CCK_MANDATORY && r32(&m, 36) == OFDM_MANDATORY && r32(&m, 48) == 0 && {
            let m5 = mac_context(ACTION_ADD, addr, peer, &MacParams { band24: false, ..mp });
            r32(&m5, 32) == 0 && r32(&m5, 48) == MAC_FLG_SHORT_SLOT
        }));
    out.push(("join: the legacy filter is group plus beacons at bit 6 before association", r32(&m, 52) == FILTER_ACCEPT_GRP | FILTER_IN_BEACON && FILTER_IN_BEACON == 64 && FILTER_IN_PROBE_REQUEST == 4096));
    out.push(("join: the EDCA rows sit at 60 + 8*fifo, BK in FIFO 1 through VO in FIFO 4, each naming its FIFO", m[73] == 1 << 1 && m[81] == 1 << 2 && m[89] == 1 << 3 && m[97] == 1 << 4 && m[60..68] == [0; 8] && m[68..70] == [15, 0] && m[72] == 7));
    out.push(("join: the station data at 100 says not associated, bi 100, DTIM interval 200, listen 10", r32(&m, 100) == 0 && r32(&m, 116) == 100 && r32(&m, 124) == 200 && r32(&m, 132) == 10 && r32(&m, 136) == 0));
    let ma = mac_context(ACTION_MODIFY, addr, peer, &MacParams { assoc: Some(7), ..mp });
    out.push(("join: a modify after association says so, carries the id, and stops asking for beacons", r32(&ma, 4) == ACTION_MODIFY && r32(&ma, 100) == 1 && r32(&ma, 136) == 7 && r32(&ma, 52) == FILTER_ACCEPT_GRP));
    out.push(("join: an association with no DTIM known keeps beacons flowing", r32(&mac_context(ACTION_MODIFY, addr, peer, &MacParams { assoc: Some(7), dtim: 0, ..mp }), 52) & FILTER_IN_BEACON != 0));
    out.push(("join: a MAC remove is the header alone", { let r = mac_context(ACTION_REMOVE, addr, peer, &mp); r32(&r, 4) == ACTION_REMOVE && r[8..].iter().all(|&x| x == 0) }));

    let bd = binding(ACTION_ADD, 0);
    out.push(("join: a binding is 28 bytes: PHY 0, MAC 0 then two invalid, PHY again, LMAC 0", bd.len() == BINDING_LEN && r32(&bd, 8) == 0 && r32(&bd, 12) == CTXT_INVALID && r32(&bd, 16) == CTXT_INVALID && r32(&bd, 20) == 0 && r32(&bd, 24) == 0));

    let st = add_sta(false, peer);
    out.push(("join: an ADD_STA is 48 bytes: add, MAC 0, the BSSID at 8, station 0, type link", st.len() == STA_LEN && st[0] == 0 && st[8..14] == peer && st[16] == 0 && st[37] == STA_LINK));
    out.push(("join: it sets the width and MIMO masks and zero flags, which is 20 MHz single-stream", r32(&st, 28) == (3 << 26) | (3 << 28) && r32(&st, 24) == 0));
    out.push(("join: the update after association is add_modify 1 with no address", { let u = add_sta(true, peer); u[0] == 1 && u[8..14] == [0; 6] }));
    out.push(("join: a reply is success only when its low status byte is 1", add_sta_ok(&[1, 0, 0, 0]) && !add_sta_ok(&[0, 0, 0, 0]) && !add_sta_ok(&[1, 0]) && add_sta_ok(&[0x01, 0x80, 0, 0])));
    out.push(("join: a station remove names station 0 in four bytes", sta_remove() == [0, 0, 0, 0]));

    let sp = session_protection(ACTION_ADD, msec_to_tu(900));
    out.push(("join: a session protection is 24 bytes, conf 0, with 900 ms as 878 TU", sp.len() == SPROT_LEN && sp[8] == 0 && r32(&sp, 12) == 878));

    let q = scd_queue_add(1, MGMT_TID, 64, 0x1000, 0x2000);
    out.push(("join: a queue add is 36 bytes, cb_size 3 for 64 entries, tables at 20 and 28",
        q.len() == SCD_LEN && q[8] == 15 && q[16] == 3 && q[21] == 0x10 && q[20] == 0 && q[29] == 0x20 && q[28] == 0));
    out.push(("join: cb_size is log2 less three", cb_size(256) == 5 && cb_size(16) == 1));
    out.push(("join: a queue reply yields the queue number and write pointer", scd_queue_rsp(&[5, 0, 0, 0, 0x34, 0x12, 0, 0]) == Some((5, 0x1234)) && scd_queue_rsp(&[1; 7]).is_none()));

    out.push(("join: the lowest legacy rate is CCK on 2.4 GHz and OFDM on 5, antenna A", legacy_rate(true, 1) == 0x4000 && legacy_rate(false, 1) == 0x4100 && legacy_rate(false, 0) == 0x4100));

    // A data frame: fc 0x0008 (data, to DS clear), 24-byte header, 10-byte body.
    let mut f = alloc::vec![0u8; 34];
    f[0] = 0x08;
    f[1] = 0x01;
    for (i, x) in f[24..].iter_mut().enumerate() {
        *x = 0xa0 + i as u8;
    }
    out.push(("join: a plain data header is 24 bytes; QoS is 26; management is 24", hdr_len(&f) == 24 && hdr_len(&[0x88, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) == 26 && hdr_len(&[0xb0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) == 24));
    out.push(("join: authentication is management and data is not", is_mgmt(&[0xb0, 0]) && !is_mgmt(&f)));
    let mut slot = [0u8; SLOT];
    let r = tx_slot(&mut slot, 9, 3, &f, 0x4100, TX_FLAGS_CMD_RATE | TX_FLAGS_ENCRYPT_DIS);
    out.push(("join: a slot opens with the four-byte header: TX_CMD, group 0, its index and queue", slot[0] == TX_CMD && slot[1] == 0 && slot[2] == 9 && slot[3] == 3));
    out.push(("join: the command carries the frame length, the flags and the rate at 4, 6 and 24",
        slot[4] == 34 && slot[6] == 3 && r32(&slot, 24) == 0x4100));
    out.push(("join: offload assist says twelve header words and no pad for a 24-byte header", r32(&slot, 8) == 12 << 8));
    out.push(("join: the header is copied to 32 and the body follows it", r == Some((24, 56, 10)) && slot[32] == 0x08 && slot[56] == 0xa0 && slot[65] == 0xa9));
    out.push(("join: the second transmit buffer is twelve bytes plus the padded header", TX_HDR + TX_CMD_LEN + 24 - cmd::FIRST_TB == 36));
    let mut qf = alloc::vec![0u8; 30];
    qf[0] = 0x88;
    let r = tx_slot(&mut slot, 0, 0, &qf, 0, 0);
    out.push(("join: a QoS header is padded to 28 and the pad bit is set", r == Some((26, 60, 4)) && slot[9] & (1 << (OFFLD_PAD - 8)) != 0));
    out.push(("join: a frame larger than a slot is refused", tx_slot(&mut slot, 0, 0, &alloc::vec![0u8; SLOT], 0, 0).is_none()));
    out.push(("join: a transmit status is read at 36", tx_status(&[0u8; 38]).is_some() && tx_status(&[0u8; 36]).is_none()));

    // The versions this part reported on the laptop pass; each refusal fires.
    let real = Versions { phy: Some(4), mac_ctx: None, binding: None, add_sta: None, rlc: Some(2), tx: Some(10), tx_notif: Some(7), scd: Some(3), sprot: Some(2), ultra_hb: true, mld: false, cdb: false };
    out.push(("join: the GF63's firmware versions are accepted, and want the chains in RLC_CONFIG", real.check().is_ok() && real.rlc_separate()));
    out.push(("join: an MLD image is refused by name, since these are the legacy contexts", Versions { mld: true, ..real }.check().is_err()));
    out.push(("join: a TX_CMD other than version 10 is refused", Versions { tx: Some(9), ..real }.check().is_err()));
    out.push(("join: a queue-config other than version 3 is refused", Versions { scd: Some(0), ..real }.check().is_err()));
    out.push(("join: a MAC_CONTEXT_CMD past version 1 is refused", Versions { mac_ctx: Some(2), ..real }.check().is_err()));
    out.push(("join: an older RLC puts the chains back in the PHY context", !Versions { rlc: None, ..real }.rlc_separate()));
    out
}
