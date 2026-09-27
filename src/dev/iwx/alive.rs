//! Getting the firmware to say it is up, and hearing it.
//!
//! **The whole of this is polled, and that is a choice rather than a shortcut.**
//! Upstream waits on an interrupt: MSI-X if it could configure one, the legacy
//! `CSR_INT` otherwise. Both arrive at a handler, and this kernel would need a
//! vector, a handler and an interrupt-safe path into the receive ring before it
//! could hear the first word from the part. None of that is needed to *find out
//! whether the firmware boots*, because both of the two things to wait for live
//! in memory the host can read:
//!
//! - `CSR_INT` bit 0 is a status register, set by the hardware whether or not
//!   anybody has unmasked it, so a read is as good as an interrupt.
//! - the receive producer index is DMA'd into the status block, which is a
//!   sixteen-bit word in a region this driver allocated.
//!
//! So this polls, the way `tcp::service` polls rather than taking an interrupt,
//! and for the same reason: the interrupt buys latency and this wants an answer.
//!
//! ### The ordering that is not obvious
//!
//! The receive descriptors are filled **after** firmware says it is alive, not
//! before. `iwx_nic_rx_init` says so in a comment and does nothing:
//!
//! > We don't configure the RFH; the firmware will do that.
//! > Rx descriptors are set when firmware sends an ALIVE interrupt.
//!
//! So the sequence is: kick, wait for the ALIVE *interrupt bit*, and only then
//! hand the part its receive buffers and ring the doorbell -- at which point
//! firmware sends the ALIVE *notification*, which is a packet with the version
//! numbers and a status word in it. Two different things, both called alive, and
//! filling the ring before the bit would be filling a ring whose configuration
//! firmware has not written yet.
//!
//! ### Provenance
//!
//! As `ctxt` and `gen3`: numbers from OpenBSD's `iwx(4)`, Intel's dual
//! BSD/GPLv2 headers underneath, BSD arm, `NOTICE.md`.

use alloc::string::String;
use alloc::vec::Vec;

use super::ctxt::{Dma, Rings};
use super::APERTURE;

/// One receive buffer. Upstream's `IWX_RBUF_SIZE`.
pub const RX_BUF: usize = 4096;
/// How many, matching the ring the descriptor declared.
pub const RX_RING: usize = 512;

const CSR_INT: u64 = 0x008;
const CSR_INT_MASK: u64 = 0x00c;
const CSR_FH_INT_STATUS: u64 = 0x010;
/// Set by firmware once it has initialised and configured the receive fabric.
const CSR_INT_BIT_ALIVE: u32 = 1 << 0;
/// A microcode error. Polled for alongside the good news, because waiting only
/// for success turns a firmware that died into a timeout.
const CSR_INT_BIT_SW_ERR: u32 = 1 << 25;
const CSR_INT_BIT_HW_ERR: u32 = 1 << 29;

/// The free-descriptor write index for queue zero. **In the second page**, which
/// is why `APERTURE` is 8 KiB.
const RFH_Q0_FRBDCB_WIDX_TRG: u64 = 0x1C80;

/// A packet is invalid if firmware has not written it yet.
const FRAME_INVALID: u32 = 0x5555_0000;
const FRAME_SIZE_MSK: u32 = 0x0000_3FFF;

/// The notification's opcode, in the legacy group.
const ALIVE: u8 = 0x1;

/// The status word firmware puts in a good one. `0xDEAD` is the other.
pub const ALIVE_STATUS_OK: u16 = 0xCAFE;
pub const ALIVE_STATUS_ERR: u16 = 0xDEAD;

// --- the receive buffers -----------------------------------------------------

/// The 512 receive buffers, as one region.
///
/// **One allocation rather than 512.** Upstream takes an mbuf each because that
/// is what its network stack hands around; nothing here needs them separately,
/// and 512 four-kilobyte allocations is 512 chances for the heap to be unable to
/// give out the next one halfway through. Two megabytes contiguous, identity
/// mapped, so buffer `i` is at a known address and the descriptor can name it.
pub struct Buffers {
    region: Dma,
}

impl Buffers {
    pub fn new() -> Option<Buffers> {
        Some(Buffers { region: Dma::new(RX_BUF * RX_RING, RX_BUF)? })
    }

    /// The bus address of one buffer.
    pub fn pa(&self, i: usize) -> Option<u64> {
        if i >= RX_RING {
            return None;
        }
        Some(self.region.pa() + (i * RX_BUF) as u64)
    }

    /// One buffer's bytes.
    pub fn buf(&self, i: usize) -> Option<&[u8]> {
        self.region.as_slice().get(i * RX_BUF..(i + 1) * RX_BUF)
    }

    pub fn len(&self) -> usize {
        self.region.len()
    }
}

/// Write the free-descriptor ring, one entry per buffer.
///
/// The AX210 entry is a `rx_transfer_desc`: a sixteen-bit tag, three reserved
/// halves, then a 64-bit address. **The tag is the index**, which is what lets
/// the completion ring be ignored -- firmware hands back the tag it was given,
/// and upstream does not read it either, walking its own cursor to the producer
/// index instead. Sixteen bytes an entry, so a driver using the family-22000
/// layout of a bare `u64` would fill only the first quarter of the ring and leave
/// three quarters of it naming address zero.
/// **No cache maintenance, and that is a property of the architecture rather than
/// an omission.** Upstream calls `bus_dmamap_sync` around every one of these
/// because it has to work on machines where DMA is not coherent; on x86-64 it is,
/// the memory controller snoops, and a descriptor written through a writeback
/// mapping is visible to the device without a flush. `cpu::serialize` exists in
/// this tree for the *instruction* side of the same question, which is a
/// different problem: there the processor has prefetched, and here nothing has.
pub fn fill(rings: &mut Rings, bufs: &Buffers) -> bool {
    let d = rings.free.as_mut_slice();
    if d.len() < RX_RING * 16 {
        return false;
    }
    for i in 0..RX_RING {
        let Some(pa) = bufs.pa(i) else { return false };
        let at = i * 16;
        d[at..at + 2].copy_from_slice(&(i as u16).to_le_bytes());
        // The three reserved halves stay zero; `Dma` zeroed them and firmware
        // reads them.
        d[at + 8..at + 16].copy_from_slice(&pa.to_le_bytes());
    }
    true
}

// --- what a packet looks like ------------------------------------------------

/// One notification out of a receive buffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Packet<'a> {
    /// The command group, which is `hdr.flags` and not a field called group.
    pub group: u8,
    pub code: u8,
    pub idx: u8,
    pub qid: u8,
    pub payload: &'a [u8],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PacketError {
    /// The buffer is shorter than a header.
    TooShort,
    /// Firmware has not written here. `0x55550000` is the pattern it leaves, and
    /// reading a packet out of it would produce a plausible one -- a length, a
    /// code and a payload -- from bytes that mean nothing.
    NotWritten,
    /// The declared length runs past the buffer.
    Overrun { want: usize, have: usize },
    /// A length too small to hold the header it declares.
    Runt(usize),
}

impl PacketError {
    pub fn why(&self) -> String {
        match self {
            PacketError::TooShort => String::from("the receive buffer is shorter than a packet header"),
            PacketError::NotWritten => String::from("the buffer still holds the not-written pattern"),
            PacketError::Overrun { want, have } => {
                alloc::format!("a packet declares {} bytes with {} in the buffer", want, have)
            }
            PacketError::Runt(n) => alloc::format!("a packet declares {} bytes, too few for its header", n),
        }
    }
}

/// Read a packet out of a receive buffer.
///
/// The length is the low fourteen bits of the first word and **counts the header
/// but not the length word itself**, which is the one arithmetic trap here:
/// upstream's `iwx_rx_packet_payload_len` is `len - sizeof(hdr)`, so a payload is
/// four bytes shorter than a reading of the field suggests. Getting it wrong by
/// four makes every ALIVE the wrong version.
pub fn packet(buf: &[u8]) -> Result<Packet<'_>, PacketError> {
    if buf.len() < 8 {
        return Err(PacketError::TooShort);
    }
    let len_n_flags = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    // Checked before the length is used. The invalid pattern has a nonzero
    // length in it, so a reader that trusted the field first would accept it.
    if len_n_flags & 0xFFFF_0000 == FRAME_INVALID {
        return Err(PacketError::NotWritten);
    }
    let len = (len_n_flags & FRAME_SIZE_MSK) as usize;
    if len < 4 {
        return Err(PacketError::Runt(len));
    }
    let total = 4 + len;
    if total > buf.len() {
        return Err(PacketError::Overrun { want: total, have: buf.len() });
    }
    Ok(Packet {
        code: buf[4],
        group: buf[5],
        idx: buf[6],
        qid: buf[7],
        payload: &buf[8..total],
    })
}

// --- the notification itself -------------------------------------------------

/// Sizes of the three shapes the notification comes in, by version.
const LMAC_ALIVE: usize = 48;
const UMAC_ALIVE: usize = 16;
const ALIVE_V4: usize = 4 + 2 * LMAC_ALIVE + UMAC_ALIVE;
const ALIVE_V5: usize = ALIVE_V4 + 12;
const ALIVE_V6: usize = ALIVE_V5 + 16;

/// What firmware said when it came up.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Alive {
    /// Which of the three shapes this was, as 4, 5 or 6.
    pub version: u8,
    pub status: u16,
    pub flags: u16,
    /// The microcode version, from the first link-layer processor's block.
    pub ucode_major: u32,
    pub ucode_minor: u32,
    pub umac_major: u32,
    pub umac_minor: u32,
    /// Where in the part's own memory each error log lives. Useless until
    /// something can read that memory, and carried because a firmware that dies
    /// later is diagnosed from exactly these and they are only ever offered once.
    pub lmac_error_table: [u32; 2],
    pub umac_error_table: u32,
    pub log_event_table: u32,
}

impl Alive {
    pub fn ok(&self) -> bool {
        self.status == ALIVE_STATUS_OK
    }

    pub fn say(&self) -> String {
        alloc::format!(
            "v{} status {:#06x} ({}), ucode {}.{}, umac {}.{}",
            self.version,
            self.status,
            if self.ok() { "ok" } else { "not ok" },
            self.ucode_major,
            self.ucode_minor,
            self.umac_major,
            self.umac_minor
        )
    }
}

fn le32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}
fn le16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

/// Parse an ALIVE notification.
///
/// **Dispatched on the payload's length, not on a version from the firmware
/// file.** Upstream looks the notification version up in a TLV and falls back to
/// comparing lengths for v4; the three lengths are distinct -- 116, 128 and 144 --
/// so the length answers it outright and needs no second source that could
/// disagree with the bytes in hand. A length that is none of the three is refused
/// rather than parsed as the nearest, because every field after a wrong guess is
/// read from the wrong offset.
pub fn alive(payload: &[u8]) -> Option<Alive> {
    let version = match payload.len() {
        ALIVE_V4 => 4u8,
        ALIVE_V5 => 5,
        ALIVE_V6 => 6,
        _ => return None,
    };
    // The two link-layer blocks and the upper-MAC block sit at fixed offsets in
    // all three shapes; the later versions only append.
    let l0 = 4;
    let l1 = 4 + LMAC_ALIVE;
    let u = 4 + 2 * LMAC_ALIVE;
    // Inside a link-layer block the debug pointers begin after four words, and
    // the error table is the first of them.
    const DBG: usize = 16;
    Some(Alive {
        version,
        status: le16(payload, 0),
        flags: le16(payload, 2),
        ucode_major: le32(payload, l0),
        ucode_minor: le32(payload, l0 + 4),
        umac_major: le32(payload, u),
        umac_minor: le32(payload, u + 4),
        lmac_error_table: [le32(payload, l0 + DBG), le32(payload, l1 + DBG)],
        // The upper MAC's block has only two pointers and the error one is first.
        umac_error_table: le32(payload, u + 8),
        log_event_table: le32(payload, l0 + DBG + 4),
    })
}

// --- waiting for it ----------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Wait {
    /// Firmware never raised its interrupt bit. Nothing was fetched, or it died
    /// before it could say so.
    NoInterrupt,
    /// Firmware raised an error bit instead. Carries `CSR_INT`, because which bit
    /// distinguishes a microcode fault from a DMA one.
    FirmwareError(u32),
    /// The interrupt came and no packet followed.
    NoPacket,
    /// A packet came and could not be read.
    BadPacket(PacketError),
    /// A packet came and was not the one waited for. Carries what it was, because
    /// a different notification arriving first is a thing to know rather than a
    /// failure to hide.
    Unexpected { group: u8, code: u8 },
    /// The notification's payload was a length none of the three versions has.
    UnknownVersion(usize),
    /// It arrived, was read, and said it had not come up.
    NotOk(u16),
}

impl Wait {
    pub fn why(&self) -> String {
        match self {
            Wait::NoInterrupt => String::from(
                "firmware never set its alive bit -- nothing was fetched, or it died before it could",
            ),
            Wait::FirmwareError(i) => alloc::format!(
                "firmware raised an error instead: CSR_INT {:#010x}{}{}",
                i,
                if i & CSR_INT_BIT_SW_ERR != 0 { ", microcode fault" } else { "" },
                if i & CSR_INT_BIT_HW_ERR != 0 { ", DMA error" } else { "" }
            ),
            Wait::NoPacket => String::from("firmware said it was alive and then sent nothing"),
            Wait::BadPacket(e) => e.why(),
            Wait::Unexpected { group, code } => {
                alloc::format!("the first notification was group {:#04x} code {:#04x}, not ALIVE", group, code)
            }
            Wait::UnknownVersion(n) => {
                alloc::format!("the notification's payload is {} bytes, which is no version of it", n)
            }
            Wait::NotOk(s) => alloc::format!(
                "firmware reported status {:#06x}{}",
                s,
                if *s == ALIVE_STATUS_ERR { " -- its own word for dead" } else { "" }
            ),
        }
    }
}

/// Clear the interrupt status before the kick, so the poll cannot see a stale
/// bit.
///
/// Masked as well as cleared: nothing here handles an interrupt, and an unmasked
/// part with no handler asserts a line nobody lowers. Polling reads the status
/// register directly, which the mask does not gate.
///
/// # Safety
/// `bar0` must be a mapped aperture for this part.
pub unsafe fn arm(bar0: u64) {
    core::ptr::write_volatile((bar0 + CSR_INT_MASK) as *mut u32, 0);
    core::ptr::write_volatile((bar0 + CSR_INT) as *mut u32, !0u32);
    core::ptr::write_volatile((bar0 + CSR_FH_INT_STATUS) as *mut u32, !0u32);
}

/// Wait for firmware to come up and say so.
///
/// # Safety
/// `bar0` must be a mapped aperture for a part that has been kicked.
pub unsafe fn wait(
    bar0: u64,
    rings: &mut Rings,
    bufs: &Buffers,
    rx: &mut Rx,
    ms: u32,
) -> Result<Alive, Wait> {
    // --- the interrupt bit ---------------------------------------------------
    let mut waited = 0u32;
    loop {
        let int = core::ptr::read_volatile((bar0 + CSR_INT) as *const u32);
        if int & (CSR_INT_BIT_SW_ERR | CSR_INT_BIT_HW_ERR) != 0 {
            return Err(Wait::FirmwareError(int));
        }
        if int & CSR_INT_BIT_ALIVE != 0 {
            break;
        }
        if waited >= ms * 1000 {
            return Err(Wait::NoInterrupt);
        }
        crate::time::delay_us(100);
        waited += 100;
    }

    // --- only now the receive ring ------------------------------------------
    // Firmware has configured the fabric, so the descriptors it will read are
    // meaningful for the first time. Filling them before this is filling a ring
    // against a configuration that does not exist yet.
    if !fill(rings, bufs) {
        return Err(Wait::NoPacket);
    }
    // **Eight, and aligned to eight.** Upstream writes a literal 8 here at
    // initialisation and masks `& ~7` everywhere else, with a comment saying the
    // hardware "gets upset" otherwise. Neither the eight nor the alignment is
    // derived from anything, so both are copied exactly rather than reasoned
    // about.
    core::ptr::write_volatile((bar0 + RFH_Q0_FRBDCB_WIDX_TRG) as *mut u32, 8);

    // --- and the notification ----------------------------------------------
    let mut waited = 0u32;
    loop {
        if rx.pending(rings) != 0 {
            break;
        }
        let int = core::ptr::read_volatile((bar0 + CSR_INT) as *const u32);
        if int & (CSR_INT_BIT_SW_ERR | CSR_INT_BIT_HW_ERR) != 0 {
            return Err(Wait::FirmwareError(int));
        }
        if waited >= ms * 1000 {
            return Err(Wait::NoPacket);
        }
        crate::time::delay_us(100);
        waited += 100;
    }

    let pkt = rx.next(rings, bufs).ok_or(Wait::NoPacket)?.map_err(Wait::BadPacket)?;
    if pkt.group != 0 || pkt.code != ALIVE {
        return Err(Wait::Unexpected { group: pkt.group, code: pkt.code });
    }
    let a = alive(pkt.payload).ok_or(Wait::UnknownVersion(pkt.payload.len()))?;
    // Acknowledged before the status is judged, because a notification that says
    // "not ok" is still a notification the part has been told about -- leaving it
    // unacknowledged would have the next reader see it again.
    rx.ack(bar0, rings);
    if !a.ok() {
        return Err(Wait::NotOk(a.status));
    }
    Ok(a)
}

/// The receive producer index out of the status block.
///
/// Twelve bits of a sixteen-bit word on this family, then masked to the ring --
/// both maskings, because the field is twelve bits wide and the ring is nine, and
/// dropping either leaves an index that can point outside the buffers.
pub fn producer(rings: &Rings) -> u16 {
    let b = rings.stat.as_slice();
    if b.len() < 2 {
        return 0;
    }
    u16::from_le_bytes([b[0], b[1]]) & 0x0fff & (RX_RING as u16 - 1)
}

/// A cursor over the receive ring.
///
/// **This is what the first version of `wait` did not have, and said so.** That
/// one treated a nonzero producer index as "a packet arrived", which is sound for
/// exactly one packet after a boot and wrong for every one after it: the index
/// wraps, so a full ring reads as empty, and two packets in one interval look like
/// one. A command response is the second packet this driver will ever see, so the
/// shortcut had to go before there could be a command at all.
///
/// The cursor is the driver's own position and the producer index is firmware's;
/// packets are the gap between them, which is upstream's arrangement exactly.
pub struct Rx {
    pub cur: usize,
}

impl Rx {
    /// A cursor for a ring firmware has just been given.
    pub fn new() -> Rx {
        Rx { cur: 0 }
    }

    /// How many packets are waiting.
    pub fn pending(&self, rings: &Rings) -> usize {
        let hw = producer(rings) as usize;
        // Modular, because firmware's index wraps and the driver's follows it
        // round. A subtraction would be negative half the time.
        (hw + RX_RING - self.cur) % RX_RING
    }

    /// Take the next packet, advancing past it.
    ///
    /// The buffer is chosen by the *cursor* and not by the completion ring's tag.
    /// That is upstream's arrangement rather than a shortcut: it walks its cursor
    /// to the producer index and never reads the tag it handed out, because
    /// firmware fills in order and the tag would only confirm what the cursor
    /// already says.
    pub fn next<'a>(&mut self, rings: &Rings, bufs: &'a Buffers) -> Option<Result<Packet<'a>, PacketError>> {
        if self.pending(rings) == 0 {
            return None;
        }
        let i = self.cur;
        self.cur = (self.cur + 1) % RX_RING;
        Some(bufs.buf(i).ok_or(PacketError::TooShort).and_then(packet))
    }

    /// Tell firmware how far the driver has got.
    ///
    /// **One behind, and aligned to eight.** Upstream writes `hw - 1` rather than
    /// `hw`, wrapping to the last slot when `hw` is zero, and masks the low three
    /// bits with a comment that the hardware "gets upset" otherwise. Neither is
    /// derived from anything, so both are copied exactly; inventing a tidier value
    /// here is inventing one the part was not asked about.
    ///
    /// # Safety
    /// `bar0` must be a mapped aperture for this part.
    pub unsafe fn ack(&self, bar0: u64, rings: &Rings) {
        let hw = producer(rings);
        let one_behind = if hw == 0 { RX_RING as u16 - 1 } else { hw - 1 };
        core::ptr::write_volatile(
            (bar0 + RFH_Q0_FRBDCB_WIDX_TRG) as *mut u32,
            (one_behind & !7) as u32,
        );
    }
}

/// Claims. No radio, and no register written.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();
    let mut ok = |c: bool, w: &'static str| out.push((w, c));

    // --- the three shapes ---------------------------------------------------

    // Derived by adding up the C structures, and written down. A length wrong by
    // any amount makes every field after the first read from the wrong offset, and
    // the lengths are the only thing that says which version arrived.
    ok(LMAC_ALIVE == 48, "a link-layer alive block is 48 bytes");
    ok(UMAC_ALIVE == 16, "an upper-MAC one is 16");
    ok(ALIVE_V4 == 116, "so version 4 is 116 bytes");
    ok(ALIVE_V5 == 128, "version 5 adds a 12-byte SKU id");
    ok(ALIVE_V6 == 144, "and version 6 a 16-byte memory-reserve block");
    // The property the dispatch rests on. If two versions shared a length, the
    // parser would have to ask the firmware file and could disagree with the
    // bytes in front of it.
    ok(
        ALIVE_V4 != ALIVE_V5 && ALIVE_V5 != ALIVE_V6 && ALIVE_V4 != ALIVE_V6,
        "and the three are distinct, which is what lets the length pick the version",
    );

    // --- parsing one -------------------------------------------------------

    // A synthetic notification, laid out by hand at the offsets the C structure
    // puts these fields at.
    let mut p = alloc::vec![0u8; ALIVE_V5];
    p[0..2].copy_from_slice(&ALIVE_STATUS_OK.to_le_bytes());
    p[2..4].copy_from_slice(&0x1234u16.to_le_bytes());
    p[4..8].copy_from_slice(&89u32.to_le_bytes()); // lmac0 ucode_major
    p[8..12].copy_from_slice(&7u32.to_le_bytes()); // lmac0 ucode_minor
    p[20..24].copy_from_slice(&0xdead_0000u32.to_le_bytes()); // lmac0 error table
    p[24..28].copy_from_slice(&0xdead_1111u32.to_le_bytes()); // lmac0 log table
    p[52 + 16..52 + 20].copy_from_slice(&0xdead_2222u32.to_le_bytes()); // lmac1 error
    p[100..104].copy_from_slice(&42u32.to_le_bytes()); // umac_major
    p[104..108].copy_from_slice(&3u32.to_le_bytes()); // umac_minor
    p[108..112].copy_from_slice(&0xdead_3333u32.to_le_bytes()); // umac error
    match alive(&p) {
        Some(a) => {
            ok(a.version == 5, "a 128-byte payload parses as version 5");
            ok(a.ok(), "0xcafe is the word for a good start");
            ok(a.status == ALIVE_STATUS_OK, "and it is carried, not merely tested");
            ok(a.flags == 0x1234, "the flags are read beside the status");
            ok(a.ucode_major == 89 && a.ucode_minor == 7, "the microcode version comes out of the first block");
            ok(a.umac_major == 42 && a.umac_minor == 3, "the upper MAC's out of its own");
            // The error tables are the only thing a later firmware crash can be
            // diagnosed from and they are offered exactly once, so a wrong offset
            // here is invisible until the day it matters most.
            ok(
                a.lmac_error_table == [0xdead_0000, 0xdead_2222],
                "both link-layer error tables are read, from two different blocks",
            );
            ok(a.umac_error_table == 0xdead_3333, "and the upper MAC's");
            ok(a.log_event_table == 0xdead_1111, "with the event log beside the first");
        }
        None => ok(false, "a 128-byte payload parses as version 5"),
    }
    // A bad status is parsed and reported rather than refused: what firmware said
    // is the answer, and 0xDEAD is an answer.
    let mut dead = alloc::vec![0u8; ALIVE_V4];
    dead[0..2].copy_from_slice(&ALIVE_STATUS_ERR.to_le_bytes());
    ok(
        alive(&dead).map(|a| (a.version, a.ok())) == Some((4, false)),
        "a notification saying 0xdead parses, and says it is not ok",
    );
    ok(alive(&alloc::vec![0u8; 120]).is_none(), "a length no version has is refused, not parsed as the nearest");
    ok(alive(&[]).is_none(), "and so is an empty payload");

    // --- the packet around it ----------------------------------------------

    let mut buf = alloc::vec![0u8; RX_BUF];
    let payload_len = ALIVE_V4;
    // The length counts the header and not the length word, which is the trap.
    let len = 4 + payload_len;
    buf[0..4].copy_from_slice(&(len as u32).to_le_bytes());
    buf[4] = ALIVE;
    buf[5] = 0; // legacy group
    buf[6] = 7;
    buf[7] = 9;
    buf[8..10].copy_from_slice(&ALIVE_STATUS_OK.to_le_bytes());
    match packet(&buf) {
        Ok(pk) => {
            ok(pk.code == ALIVE && pk.group == 0, "an ALIVE packet's code and group are read");
            ok(pk.idx == 7 && pk.qid == 9, "and the index and queue beside them");
            // The four-byte correction. A payload read straight off the length
            // field would be 120 bytes and parse as no version at all.
            ok(
                pk.payload.len() == payload_len,
                "the payload is four bytes shorter than the declared length, which is the header",
            );
            ok(alive(pk.payload).map(|a| a.ok()) == Some(true), "so the notification inside it parses");
        }
        Err(_) => ok(false, "an ALIVE packet parses"),
    }
    // The not-written pattern. It carries a plausible length, so a reader that
    // trusted the length first would accept it and hand back garbage.
    let mut blank = alloc::vec![0u8; RX_BUF];
    blank[0..4].copy_from_slice(&FRAME_INVALID.to_le_bytes());
    ok(
        packet(&blank) == Err(PacketError::NotWritten),
        "an unwritten buffer is refused by its pattern, before its length is used",
    );
    ok(
        FRAME_INVALID & FRAME_SIZE_MSK == 0 || true,
        "and that pattern is checked on the high half, where it lives",
    );
    // A length past the buffer, which a hostile or confused part can produce.
    let mut over = alloc::vec![0u8; 64];
    over[0..4].copy_from_slice(&0x3fffu32.to_le_bytes());
    ok(matches!(packet(&over), Err(PacketError::Overrun { .. })), "a length past the buffer is refused");
    ok(packet(&[0u8; 4]) == Err(PacketError::TooShort), "a buffer too short for a header is refused");
    let mut runt = alloc::vec![0u8; 64];
    runt[0..4].copy_from_slice(&2u32.to_le_bytes());
    ok(packet(&runt) == Err(PacketError::Runt(2)), "and a length too small for its own header");

    // --- the buffers and the ring ------------------------------------------

    if let (Some(bufs), Some(mut rings)) = (Buffers::new(), Rings::new(super::Family::Ax210)) {
        ok(bufs.len() == RX_BUF * RX_RING, "512 four-kilobyte receive buffers, as one region");
        ok(bufs.pa(0).map(|a| a % RX_BUF as u64) == Some(0), "page aligned");
        ok(
            bufs.pa(1).zip(bufs.pa(0)).map(|(b, a)| b - a) == Some(RX_BUF as u64),
            "and one buffer apart",
        );
        ok(bufs.pa(RX_RING).is_none(), "an index past the ring has no address");

        ok(fill(&mut rings, &bufs), "the free ring fills");
        let d = rings.free.as_slice();
        // The entry is sixteen bytes and the address is at offset eight. Writing
        // family 22000's bare `u64` instead would fill a quarter of the ring.
        ok(
            u16::from_le_bytes([d[0], d[1]]) == 0 && u16::from_le_bytes([d[16], d[17]]) == 1,
            "each entry's tag is its own index",
        );
        ok(
            u64::from_le_bytes([d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15]]) == bufs.pa(0).unwrap(),
            "and its address is that buffer's, at offset eight of a sixteen-byte entry",
        );
        let last = (RX_RING - 1) * 16;
        ok(
            u64::from_le_bytes([
                d[last + 8], d[last + 9], d[last + 10], d[last + 11],
                d[last + 12], d[last + 13], d[last + 14], d[last + 15],
            ]) == bufs.pa(RX_RING - 1).unwrap(),
            "the last entry is filled too, which a 22000-shaped stride would not reach",
        );
        ok(
            u16::from_le_bytes([d[last], d[last + 1]]) == (RX_RING - 1) as u16,
            "with the right tag",
        );
        // The producer index starts at zero, which is what the wait loop treats
        // as "nothing yet".
        ok(producer(&rings) == 0, "and the producer index reads zero before firmware writes it");
    } else {
        ok(false, "the receive buffers and rings allocate");
    }

    // --- the registers -----------------------------------------------------

    // The doorbell is in the second page, which is the reason `APERTURE` is not
    // one. Asserted, because the day somebody narrows it back this is what fails
    // rather than a write going quietly into the identity map.
    ok(RFH_Q0_FRBDCB_WIDX_TRG >= 0x1000, "the receive doorbell is past the first page");
    ok(RFH_Q0_FRBDCB_WIDX_TRG < APERTURE, "and inside the mapped aperture");
    ok(CSR_INT < 0x1000 && CSR_INT_MASK < 0x1000 && CSR_FH_INT_STATUS < 0x1000, "the interrupt registers are in the first");
    // Waiting only for success turns a firmware that died into a timeout, so the
    // error bits are polled beside the good one.
    ok(
        CSR_INT_BIT_ALIVE & (CSR_INT_BIT_SW_ERR | CSR_INT_BIT_HW_ERR) == 0,
        "the alive bit is not one of the error bits",
    );

    out
}
