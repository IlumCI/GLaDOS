//! Telling a configured part how to behave, which is where `iwx_init_hw` goes.
//!
//! After the handshake in `init` the firmware is alive and knows nothing about
//! how this host wants it to run. `iwx_init_hw` is a dozen commands under the MAC
//! access lock, each one small, most of them unconditional, and several gated on
//! what the firmware file said it could do.
//!
//! ### This sequence is incomplete, and says so rather than pretending
//!
//! Seven of the twelve are here, in upstream's relative order, and they are the
//! seven whose payload is small and exactly known. **Five are absent** and
//! naming them is the point, because a table that silently skipped them would read
//! as a part that had been fully configured:
//!
//! - `PHY_CONFIGURATION_CMD`, which upstream sends only when the part wants
//!   single-stream diversity. This one is genuinely not owed: `8086:51f0` sets
//!   `tx_with_siso_diversity` to zero, so the command is skipped on the real part
//!   and its absence here is upstream's behaviour rather than a gap.
//! - `TEMP_REPORTING_THRESHOLDS_CMD`, which carries a threshold table.
//! - `MAC_PM_POWER_TABLE`, which needs a station. Upstream does not send it at
//!   initialisation either: `iwx_set_pslevel` returns after the *device* policy
//!   when no MAC context is active, which it is not here -- so the command in this
//!   table is the four-byte one and `power.rs` says why.
//! - `MCC_UPDATE_CMD`, which has a response to parse.
//! - the scan configuration, which is the largest command in the driver.
//!
//! Two of the five are not owed at initialisation, so **three** are the gap:
//! temperature thresholds, `MCC_UPDATE` and the scan configuration. So
//! `configure` leaves the part able to be told things and not yet able to scan,
//! and the shell says that in those words.
//!
//! ### What the capability bitmap actually said, measured
//!
//! Read off `iwlwifi-so-a0-hr-b0-89.ucode`, which is what the GF63 loads:
//!
//!     capabilities  9def037e ff7f7aef db91eedb 193fdfbc 00000000
//!     api           fd9bfffb fff7ffff 0000001f 00000000
//!     LAR true   DQA false   CT-kill true   MLD true
//!
//! Two of those are worth knowing before writing any condition.
//!
//! **`DQA_SUPPORT` is not declared**, so `DQA_ENABLE_CMD` is skipped on the real
//! part. A driver sending it unconditionally would be sending a command this
//! firmware never advertised, and the gate is what stops that -- which is the whole
//! reason the bitmaps were parsed before this table was written.
//!
//! **`MLD_API_SUPPORT` *is* declared and must not be believed.** Upstream enables
//! that path only on MAC type MA or family Bz, with the comment that "the API 77
//! firmware on some of the older firmware devices also claims support, but doesn't
//! actually work" -- and this part is Snow Owl. So the bit is set, the feature does
//! not work, and a driver reading the bit alone enables a path Intel's own driver
//! declines. `mld_usable` is that refusal, and it is a claim rather than a comment.
//!
//! ### Provenance
//!
//! As the rest of `iwx`: numbers from OpenBSD's `iwx(4)`, Intel's dual BSD/GPLv2
//! headers underneath, BSD arm, `NOTICE.md`.

use alloc::string::String;
use alloc::vec::Vec;

use super::alive::{Buffers, Rx};
use super::cmd::{self, Queue};
use super::ctxt::Rings;
use super::fw::{capa, Image};
use super::init::SYSTEM_GROUP;
use super::power;
use super::{Family, Mac};

/// The data-path group, which the queue-enable command lives in.
pub const DATA_PATH_GROUP: u8 = 0x5;

pub const TX_ANT_CONFIGURATION_CMD: u8 = 0x98;
pub const BT_CONFIG: u8 = 0x9b;
pub const SOC_CONFIGURATION_CMD: u8 = 0x01;
pub const DQA_ENABLE_CMD: u8 = 0x00;
pub const LTR_CONFIG: u8 = 0xee;
pub const REPLY_BEACON_FILTERING_CMD: u8 = 0xd2;

/// Bluetooth coexistence: let Wi-Fi win. The only mode this driver ever asks for,
/// because arbitration needs a Bluetooth stack to arbitrate with and there is none.
pub const BT_COEX_WIFI: u32 = 0x3;

/// The part is a discrete card rather than one in the chipset.
///
/// **Set for `8086:51f0`, which reads oddly and is upstream's own answer.** That
/// id is a PCH function, so "discrete" is the last word anybody would choose for
/// it -- and `iwx`'s product table sets `sc_integrated = 0` for it, which takes the
/// version-1 branch of this command and sends exactly this flag. Followed rather
/// than corrected: the alternative is guessing that a field named `integrated`
/// means what the English word means, against a table written by people with the
/// part in front of them.
pub const SOC_CONFIG_DISCRETE: u32 = 1 << 0;

/// Turn the latency-tolerance feature on. The rest of that command's thirty-two
/// bytes stay zero.
pub const LTR_CFG_FLAG_FEATURE_ENABLE: u32 = 0x1;

/// Is the multi-link API usable, whatever the bitmap says?
///
/// **The bit is not the answer, and this is the one place in the driver where that
/// is true.** Upstream reads the capability *and* requires MAC type MA or family
/// Bz, because firmware on older parts claims the API and does not implement it.
/// Snow Owl declares it and is neither, so a driver trusting the bitmap turns on a
/// path that does not work and has no error to report.
pub fn mld_usable(mac: Mac, image: &Image) -> bool {
    if !image.has_capa(capa::MLD_API_SUPPORT) {
        return false;
    }
    matches!(mac, Mac::Ma) || mac.family() == Some(Family::Bz)
}

/// Why a step might not be sent.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Gate {
    /// Always sent.
    Always,
    /// Only when the firmware declares a capability.
    Capa(usize),
    /// Only when the PCIe device says latency tolerance is enabled. Not a firmware
    /// property at all, which is why it is its own gate rather than a capability
    /// bit -- the answer is in config space.
    LtrEnabled,
}

/// What a step's payload is.
///
/// Named rather than carried as bytes, because one of them is a value out of the
/// NVM and a table of literals could not hold it. The same arrangement `Kick`
/// makes for its addresses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Body {
    /// One word: which transmit chains the part was fused with. From the NVM, so
    /// this command cannot be sent before it has been read.
    TxAnt,
    /// Two words: the coexistence mode and an empty module mask.
    BtCoex,
    /// Two words: the platform flags and the crystal latency.
    Soc,
    /// One word: which queue is the command queue.
    Dqa,
    /// Thirty-two bytes, of which one word is set.
    Ltr,
    /// Sixty zero bytes, which is what disabling beacon filtering is.
    BeaconFilterOff,
    /// Two words: whether the device may sleep, and a reserved half.
    DevicePower,
}

impl Body {
    pub fn len(self) -> usize {
        match self {
            Body::TxAnt | Body::Dqa | Body::DevicePower => 4,
            Body::BtCoex | Body::Soc => 8,
            Body::Ltr => 32,
            Body::BeaconFilterOff => 60,
        }
    }
}

/// One command in the sequence.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Step {
    pub group: u8,
    pub code: u8,
    pub body: Body,
    pub gate: Gate,
}

/// The sequence, in upstream's relative order.
pub const CONFIG: &[Step] = &[
    // The antennas first, because everything about rates and scanning is
    // expressed in terms of chains the part does not have if this is wrong.
    Step { group: 0, code: TX_ANT_CONFIGURATION_CMD, body: Body::TxAnt, gate: Gate::Always },
    Step { group: 0, code: BT_CONFIG, body: Body::BtCoex, gate: Gate::Always },
    Step { group: SYSTEM_GROUP, code: SOC_CONFIGURATION_CMD, body: Body::Soc, gate: Gate::Always },
    // Skipped on the real part, which does not declare the capability -- measured,
    // not assumed.
    Step { group: DATA_PATH_GROUP, code: DQA_ENABLE_CMD, body: Body::Dqa, gate: Gate::Capa(capa::DQA_SUPPORT) },
    Step { group: 0, code: LTR_CONFIG, body: Body::Ltr, gate: Gate::LtrEnabled },
    // The device's sleep policy, after the latency configuration and before the
    // beacon filter, which is where `init_hw` puts it.
    Step { group: 0, code: power::POWER_TABLE_CMD, body: Body::DevicePower, gate: Gate::Always },
    // Last, as upstream has it.
    Step { group: 0, code: REPLY_BEACON_FILTERING_CMD, body: Body::BeaconFilterOff, gate: Gate::Always },
];

/// Everything the payloads need that is not a constant.
#[derive(Clone, Copy)]
pub struct Facts {
    /// Which transmit chains, out of the NVM.
    pub tx_ant: u8,
    /// Whether this part is described as discrete. Upstream's flag for `8086:51f0`.
    pub discrete: bool,
    /// The crystal latency, which is zero on this part.
    pub xtal_latency: u32,
    /// Whether the PCIe function says latency tolerance is on.
    pub ltr_enabled: bool,
    /// How much the radio may sleep. Zero, and `power.rs` argues for zero: a
    /// machine that mines or serves should not have its radio asleep between
    /// beacons.
    pub power_level: u8,
}

/// Build one step's payload.
pub fn body(b: Body, f: &Facts) -> Vec<u8> {
    let mut v = alloc::vec![0u8; b.len()];
    match b {
        Body::TxAnt => v[0..4].copy_from_slice(&(f.tx_ant as u32).to_le_bytes()),
        Body::BtCoex => {
            v[0..4].copy_from_slice(&BT_COEX_WIFI.to_le_bytes());
            // The module mask stays zero: enabling a coexistence module means
            // having something to coexist with, and there is no Bluetooth stack.
        }
        Body::Soc => {
            let flags = if f.discrete { SOC_CONFIG_DISCRETE } else { 0 };
            v[0..4].copy_from_slice(&flags.to_le_bytes());
            v[4..8].copy_from_slice(&f.xtal_latency.to_le_bytes());
        }
        Body::Dqa => v[0..4].copy_from_slice(&(cmd::CMD_QUEUE).to_le_bytes()),
        Body::Ltr => v[0..4].copy_from_slice(&LTR_CFG_FLAG_FEATURE_ENABLE.to_le_bytes()),
        // Sixty zeroes, and the zeroes are the message: every threshold at zero
        // with the enable word clear is what "do not filter beacons" is spelt as.
        Body::BeaconFilterOff => {}
        Body::DevicePower => v.copy_from_slice(&power::device_body(f.power_level)),
    }
    v
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Fault {
    /// A step was refused. Carries the index, so the transcript names which.
    At(usize, cmd::CmdError),
}

impl Fault {
    pub fn why(&self) -> String {
        match self {
            Fault::At(i, e) => match CONFIG.get(*i) {
                Some(s) => alloc::format!(
                    "configuration step {} (group {:#04x} code {:#04x}) was refused: {}",
                    i, s.group, s.code, e.why()
                ),
                None => alloc::format!("configuration step {}: {}", i, e.why()),
            },
        }
    }
}

/// What a run of the sequence did.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Done {
    pub sent: usize,
    pub skipped: usize,
}

impl Done {
    pub fn say(&self) -> String {
        alloc::format!(
            "{} configuration command(s) sent, {} skipped; three more are owed and not written, so it cannot scan yet",
            self.sent, self.skipped
        )
    }
}

/// Send the sequence.
///
/// **No replies are waited for.** Every one of these is a command firmware
/// acknowledges by acting rather than by answering, and upstream sends them with
/// no response buffer. A driver waiting for one would hang on the first.
///
/// # Safety
/// `bar0` must be a mapped aperture for a part whose firmware is alive and which
/// has been through `init::handshake`.
pub unsafe fn configure(
    bar0: u64,
    rings: &mut Rings,
    q: &mut Queue,
    image: &Image,
    f: &Facts,
) -> Result<Done, Fault> {
    let mut d = Done { sent: 0, skipped: 0 };
    for (i, s) in CONFIG.iter().enumerate() {
        let allowed = match s.gate {
            Gate::Always => true,
            Gate::Capa(n) => image.has_capa(n),
            Gate::LtrEnabled => f.ltr_enabled,
        };
        if !allowed {
            d.skipped += 1;
            continue;
        }
        q.send(bar0, rings, s.group, s.code, 0, &body(s.body, f)).map_err(|e| Fault::At(i, e))?;
        d.sent += 1;
    }
    Ok(d)
}

/// Read whether the PCIe function has latency tolerance enabled.
///
/// **Not a firmware capability**, which is why it is not in the bitmap: the answer
/// is bit 10 of the device control register 2, inside the PCI Express capability,
/// and reaching it means walking the capability list. A part with it off and a
/// driver that sent the command anyway would be configuring a feature the link
/// does not have.
pub fn ltr_enabled(ecam: u64, d: &crate::dev::pci::Device) -> bool {
    use crate::dev::pci;
    const DCSR2: u64 = 0x28;
    const LTREN: u32 = 1 << 10;
    pci::find_cap(ecam, d, pci::CAP_PCIE).is_some_and(|at| pci::cfg_read32(ecam, d, at + DCSR2) & LTREN != 0)
}

/// Claims. No radio, and no register touched.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();
    let mut ok = |c: bool, w: &'static str| out.push((w, c));

    // --- the order ----------------------------------------------------------

    let idx = |c: u8| CONFIG.iter().position(|s| s.code == c);
    ok(CONFIG.len() == 7, "seven of the twelve configuration commands are written");
    ok(
        idx(TX_ANT_CONFIGURATION_CMD) == Some(0),
        "the antenna configuration goes first, since rates are expressed in chains",
    );
    ok(idx(BT_CONFIG) < idx(SOC_CONFIGURATION_CMD), "coexistence before the platform configuration");
    ok(idx(SOC_CONFIGURATION_CMD) < idx(DQA_ENABLE_CMD), "which comes before the queue enable");
    ok(idx(DQA_ENABLE_CMD) < idx(LTR_CONFIG), "and that before the latency configuration");
    ok(
        idx(LTR_CONFIG) < idx(power::POWER_TABLE_CMD),
        "then the sleep policy, which is where init_hw puts it",
    );
    ok(
        idx(REPLY_BEACON_FILTERING_CMD) == Some(CONFIG.len() - 1),
        "and beacon filtering is last, as upstream has it",
    );

    // --- the payload lengths ------------------------------------------------

    ok(
        Body::TxAnt.len() == 4 && Body::Dqa.len() == 4 && Body::DevicePower.len() == 4,
        "three commands are one word",
    );
    ok(Body::BtCoex.len() == 8 && Body::Soc.len() == 8, "two are two words");
    ok(Body::Ltr.len() == 32, "the latency command is thirty-two bytes");
    // Sixty and not sixty-four: eleven words, then two pairs. A structure padded
    // to a round number would be a different structure with the enable word in the
    // wrong place.
    ok(Body::BeaconFilterOff.len() == 60, "and the beacon filter sixty, which is not a round number");
    // Two of them are longer than the first transmit buffer, so this sequence is
    // the first thing in the driver that needs the two-descriptor path for real.
    ok(
        CONFIG.iter().any(|s| super::cmd::HDR_WIDE + s.body.len() > super::cmd::FIRST_TB),
        "and two of them need the second transmit buffer, which nothing before this did",
    );

    // --- the payloads -------------------------------------------------------

    let f = Facts { tx_ant: 0b11, discrete: true, xtal_latency: 0, ltr_enabled: true, power_level: 0 };
    let v = body(Body::TxAnt, &f);
    ok(u32::from_le_bytes([v[0], v[1], v[2], v[3]]) == 3, "the antenna mask is the NVM's, widened to a word");
    let v = body(Body::BtCoex, &f);
    ok(u32::from_le_bytes([v[0], v[1], v[2], v[3]]) == BT_COEX_WIFI, "coexistence asks for Wi-Fi to win");
    ok(
        u32::from_le_bytes([v[4], v[5], v[6], v[7]]) == 0,
        "with no modules enabled, there being no Bluetooth stack to coexist with",
    );
    let v = body(Body::Soc, &f);
    ok(
        u32::from_le_bytes([v[0], v[1], v[2], v[3]]) == SOC_CONFIG_DISCRETE,
        "a part upstream calls discrete sends that flag",
    );
    let v = body(Body::Soc, &Facts { discrete: false, ..f });
    ok(u32::from_le_bytes([v[0], v[1], v[2], v[3]]) == 0, "and one it does not sends none");
    let v = body(Body::Ltr, &f);
    ok(
        u32::from_le_bytes([v[0], v[1], v[2], v[3]]) == 1 && v[4..].iter().all(|&b| b == 0),
        "the latency command sets one word and leaves thirty-two bytes' worth of zeroes",
    );
    ok(
        body(Body::BeaconFilterOff, &f).iter().all(|&b| b == 0),
        "and disabling beacon filtering is sixty zeroes, which is what that means",
    );

    // --- the gates ----------------------------------------------------------

    // A firmware declaring nothing gets the unconditional commands and no others.
    let bare = super::fw::Image {
        human: String::new(), ver: 0, build: 0, sections: Vec::new(), cpus: None,
        capa: [0; super::fw::CAPA_WORDS], api: [0; super::fw::API_WORDS],
        iml: None, records: Vec::new(),
    };
    let unconditional = CONFIG.iter().filter(|s| s.gate == Gate::Always).count();
    ok(unconditional == 5, "five of the seven are unconditional");
    ok(
        CONFIG.iter().filter(|s| matches!(s.gate, Gate::Capa(_))).count() == 1,
        "one is gated on a firmware capability",
    );
    // The sleep policy is four zero bytes by default, and the zeroes are the
    // decision rather than an unfilled buffer.
    ok(
        body(Body::DevicePower, &f).iter().all(|&b| b == 0),
        "the sleep policy defaults to not sleeping, which is what a mining machine wants",
    );
    ok(
        body(Body::DevicePower, &Facts { power_level: 3, ..f })[0] == 1,
        "and asking for a level sets the bit",
    );
    ok(
        CONFIG.iter().filter(|s| s.gate == Gate::LtrEnabled).count() == 1,
        "and one on something in config space rather than in the firmware",
    );
    ok(
        !bare.has_capa(capa::DQA_SUPPORT),
        "a firmware declaring nothing does not get the queue-enable command",
    );

    // **The real firmware does not declare DQA**, which is measured rather than
    // assumed: `iwlwifi-so-a0-hr-b0-89.ucode`'s first capability word is
    // 0x9def037e and bit 12 of it is clear. So this command is skipped on the part
    // this driver is for, and a driver sending it unconditionally would be sending
    // one that firmware never advertised.
    ok((0x9def_037eu32 >> 12) & 1 == 0, "and the real firmware's word confirms it does not");
    // The bits that are set in the same word, so the claim above is about a clear
    // bit in a live bitmap rather than about an empty one.
    ok((0x9def_037eu32 >> 1) & 1 == 1, "where the regulatory capability beside it is set");

    // --- the multi-link trap ------------------------------------------------

    // The one place a capability bit is not the answer. Snow Owl declares it and
    // upstream refuses it, so a driver reading the bitmap alone enables a path
    // Intel's own driver declines -- and there is no error to report when it does
    // not work.
    let mut mld = bare.clone();
    mld.capa[capa::MLD_API_SUPPORT / 32] = 1 << (capa::MLD_API_SUPPORT % 32);
    ok(mld.has_capa(capa::MLD_API_SUPPORT), "a firmware can declare the multi-link API");
    ok(!mld_usable(Mac::So, &mld), "and Snow Owl declaring it is still refused");
    ok(!mld_usable(Mac::Sof, &mld), "as is its Alder Lake spin");
    ok(mld_usable(Mac::Ma, &mld), "where Ma declaring it is taken");
    ok(mld_usable(Mac::Bz, &mld), "and so is Bz");
    ok(!mld_usable(Mac::Ma, &bare), "but a part that does not declare it is refused whatever it is");

    // --- what is missing, asserted as missing -------------------------------

    // The sequence is seven of twelve and the count is the claim: a table that grew
    // silently to twelve would pass every ordering claim above and still be a
    // different sequence, so the shortfall is named where somebody will see it.
    ok(
        CONFIG.len() < 12,
        "the sequence is deliberately short of upstream's twelve, and the report says so",
    );
    ok(
        Done { sent: 4, skipped: 2 }.say().contains("cannot scan yet"),
        "and a successful run says the part still cannot scan",
    );

    out
}
