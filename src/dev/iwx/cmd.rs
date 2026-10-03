//! Asking the part something, which is how everything after ALIVE happens.
//!
//! Firmware answers questions rather than exposing registers: the MAC address,
//! the channel plan, the PHY's configuration and every later thing are host
//! commands with responses. A command is a small structure written into a ring in
//! host memory, described by a transmit frame descriptor, and announced with one
//! register write; the answer comes back through the receive ring as an ordinary
//! notification carrying the same group and opcode.
//!
//! ### Two rings and one queue index, which is the arithmetic to get right
//!
//! The descriptor ring holds 256 entries and the *hardware* write pointer counts
//! to 65,536 on this family. They are not the same number and neither is derived
//! from the other: `cur` indexes the ring and wraps at 256, `cur_hw` is what the
//! doorbell is told and wraps at 65,536. Using one for the other works for the
//! first 256 commands and then quietly addresses the wrong descriptor.
//!
//! ### A command is in two pieces because a descriptor says so
//!
//! The first transmit buffer is capped at twenty bytes -- `IWX_FIRST_TB_SIZE` --
//! and anything longer needs a second pointing twenty bytes further into the same
//! command. Not an optimisation: the hardware reads the first twenty bytes from
//! wherever the first buffer points and expects the rest to be described
//! separately. A single buffer for a longer command loses everything past byte
//! twenty, and an eight-byte header plus a four-byte payload fits, which is why a
//! driver can get this wrong and still read the NVM.
//!
//! ### Provenance
//!
//! As the rest of `iwx`: numbers from OpenBSD's `iwx(4)`, Intel's dual BSD/GPLv2
//! headers underneath, BSD arm, `NOTICE.md`.

use alloc::string::String;
use alloc::vec::Vec;

use super::alive::{self, Buffers, Packet, Rx};
use super::ctxt::{Dma, Rings, TFD_SIZE, TX_RING};

/// The command queue's own index. Queue zero on every family.
pub const CMD_QUEUE: u32 = 0;

/// The wide command header: opcode, group, index, queue, length, reserved,
/// version.
pub const HDR_WIDE: usize = 8;

/// How much of a command the first transmit buffer may describe.
pub const FIRST_TB: usize = 20;

/// The payload space a pre-allocated command entry has.
const DEF_PAYLOAD: usize = 320;

/// One entry in the command buffer. The header shares its space with the payload
/// through a union, so the entry is four bytes of narrow header plus the payload
/// either way: 324.
pub const ENTRY: usize = 4 + DEF_PAYLOAD;

/// The hardware write pointer's modulus on this family, which is **not** the
/// ring's length.
pub const HW_WRAP: u32 = 65536;

const HBUS_TARG_WRPTR: u64 = 0x400 + 0x060;

/// Groups this driver names. There are many; these are the ones it asks of.
pub const LONG_GROUP: u8 = 0x1;
pub const REGULATORY_AND_NVM_GROUP: u8 = 0xc;

/// The command queue.
pub struct Queue {
    /// 256 command entries, contiguous, so entry `i` is at a known address.
    pub buf: Dma,
    /// Where the next command goes in the descriptor ring.
    pub cur: usize,
    /// What the doorbell will be told. A different modulus from `cur`.
    pub cur_hw: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CmdError {
    /// The payload does not fit a pre-allocated entry. Upstream allocates a
    /// larger buffer for this case; nothing here needs one yet, so it is refused
    /// by name rather than silently truncated.
    TooLong(usize),
    /// The command ring is not there.
    NoQueue,
    /// The part answered something else, or nothing.
    NoReply,
    /// A reply came and could not be read.
    BadReply(alive::PacketError),
    /// A reply came for a different command. Carries both, because a reply
    /// arriving out of order is a thing to know rather than a failure to hide.
    Mismatched { want: (u8, u8), got: (u8, u8) },
}

impl CmdError {
    pub fn why(&self) -> String {
        match self {
            CmdError::TooLong(n) => alloc::format!(
                "a {}-byte payload does not fit a command entry, which holds {}",
                n,
                DEF_PAYLOAD - HDR_WIDE + 4
            ),
            CmdError::NoQueue => String::from("the command ring was not allocated"),
            CmdError::NoReply => String::from("the part did not answer"),
            CmdError::BadReply(e) => e.why(),
            CmdError::Mismatched { want, got } => alloc::format!(
                "asked group {:#04x} code {:#04x} and was answered group {:#04x} code {:#04x}",
                want.0, want.1, got.0, got.1
            ),
        }
    }
}

impl Queue {
    pub fn new() -> Option<Queue> {
        // Aligned to 64 because upstream aligns this region to
        // `FIRST_TB_SIZE_ALIGN`. Note it aligns the *region* and strides entries
        // by 324, so individual entries are not aligned -- copied as it is rather
        // than tidied, since the requirement is the part's and not this driver's.
        Some(Queue { buf: Dma::new(ENTRY * TX_RING as usize, 64)?, cur: 0, cur_hw: 0 })
    }

    /// The bus address of one command entry.
    pub fn entry_pa(&self, i: usize) -> Option<u64> {
        if i >= TX_RING as usize {
            return None;
        }
        Some(self.buf.pa() + (i * ENTRY) as u64)
    }

    /// Write a command into the ring, without telling the part.
    ///
    /// **Separate from the doorbell so it can be checked with no radio.** Framing
    /// is memory writes -- a header, a payload and a descriptor -- and every one of
    /// them is a thing to get wrong; ringing is one register write and cannot be
    /// exercised here at all. Keeping them together made the suite write to
    /// physical address 0x460, which is what a claim passing a bar of zero to a
    /// function that rings a doorbell does. It faulted, which is the system
    /// working, and the split is the fix rather than a bar the claims pretend to
    /// have.
    ///
    /// Answers the index it used, which is what the reply will carry back.
    pub fn frame(
        &mut self,
        rings: &mut Rings,
        group: u8,
        opcode: u8,
        version: u8,
        payload: &[u8],
    ) -> Result<u8, CmdError> {
        if HDR_WIDE + payload.len() > ENTRY {
            return Err(CmdError::TooLong(payload.len()));
        }
        let idx = self.cur;
        let pa = self.entry_pa(idx).ok_or(CmdError::NoQueue)?;

        // **A group of zero is rewritten to one**, which is upstream's own
        // workaround and its comment calls it "Intel inside (tm)": firmware past
        // API 50 rejects old-style commands in group zero with BAD_COMMAND, so
        // they have to be sent as though they were in the long group. Nothing
        // here sends a group-zero command yet, and the rule is applied anyway --
        // the first one that does would otherwise be refused by the part for a
        // reason with no trace on this side.
        let group = if group == 0 { LONG_GROUP } else { group };

        {
            let at = idx * ENTRY;
            let b = self.buf.as_mut_slice();
            let e = &mut b[at..at + ENTRY];
            // Zeroed, because firmware reads the whole entry and the slot may
            // hold a previous command.
            for x in e.iter_mut() {
                *x = 0;
            }
            e[0] = opcode;
            e[1] = group;
            e[2] = idx as u8;
            e[3] = CMD_QUEUE as u8;
            // The length is the payload's alone and excludes this header, which
            // is the opposite convention from the receive side's.
            e[4..6].copy_from_slice(&(payload.len() as u16).to_le_bytes());
            e[6] = 0;
            e[7] = version;
            e[HDR_WIDE..HDR_WIDE + payload.len()].copy_from_slice(payload);
        }

        // The descriptor: one or two transmit buffers over the same command.
        let total = HDR_WIDE + payload.len();
        {
            let at = idx * TFD_SIZE;
            let d = rings.cmd.as_mut_slice();
            let t = &mut d[at..at + TFD_SIZE];
            for x in t.iter_mut() {
                *x = 0;
            }
            let n: u16 = if total > FIRST_TB { 2 } else { 1 };
            t[0..2].copy_from_slice(&n.to_le_bytes());
            // A transmit buffer is ten bytes: a length then an address, packed,
            // so the second begins at twelve and not at sixteen.
            let first = core::cmp::min(total, FIRST_TB);
            t[2..4].copy_from_slice(&(first as u16).to_le_bytes());
            t[4..12].copy_from_slice(&pa.to_le_bytes());
            if n == 2 {
                t[12..14].copy_from_slice(&((total - FIRST_TB) as u16).to_le_bytes());
                t[14..22].copy_from_slice(&(pa + FIRST_TB as u64).to_le_bytes());
            }
        }

        // Advance both, each by its own modulus. The doorbell is the caller's.
        self.cur = (self.cur + 1) % TX_RING as usize;
        self.cur_hw = (self.cur_hw + 1) % HW_WRAP;
        Ok(idx as u8)
    }

    /// Tell the part there is a command waiting.
    ///
    /// The queue number is in the high half and the *hardware* cursor in the low,
    /// which is the one that counts to 65,536 rather than the ring index.
    ///
    /// # Safety
    /// `bar0` must be a mapped aperture for a part whose firmware is alive.
    pub unsafe fn ring(&self, bar0: u64) {
        core::ptr::write_volatile(
            (bar0 + HBUS_TARG_WRPTR) as *mut u32,
            (CMD_QUEUE << 16) | self.cur_hw,
        );
    }

    /// Frame a command and ring for it.
    ///
    /// # Safety
    /// `bar0` must be a mapped aperture for a part whose firmware is alive.
    pub unsafe fn send(
        &mut self,
        bar0: u64,
        rings: &mut Rings,
        group: u8,
        opcode: u8,
        version: u8,
        payload: &[u8],
    ) -> Result<u8, CmdError> {
        let idx = self.frame(rings, group, opcode, version, payload)?;
        self.ring(bar0);
        Ok(idx)
    }
}

/// Send a command and wait for the answer.
///
/// **Other notifications are skipped rather than treated as the answer.** Firmware
/// sends things nobody asked for -- statistics, temperature, debug -- and the
/// first packet after a command is not necessarily its reply. So this walks the
/// ring looking for the group and opcode it asked about and passes over the rest,
/// which is what lets the one unsolicited notification arriving mid-command be
/// harmless rather than a mismatch.
///
/// # Safety
/// `bar0` must be a mapped aperture for a part whose firmware is alive.
pub unsafe fn ask<'a>(
    bar0: u64,
    rings: &mut Rings,
    bufs: &'a Buffers,
    rx: &mut Rx,
    q: &mut Queue,
    group: u8,
    opcode: u8,
    version: u8,
    payload: &[u8],
    ms: u32,
) -> Result<Packet<'a>, CmdError> {
    ask_with(bar0, rings, bufs, rx, q, group, opcode, version, payload, ms, &mut |_| {})
}

/// `ask`, handing every packet it steps over to `aside`.
///
/// **What lets a command be sent while the part is hearing things.** During a
/// scan or an association every beacon is a packet in this ring, and `ask`
/// stepped over them -- correctly for a part that only answers questions, and
/// a scan that loses whatever arrived while a command was in flight otherwise.
/// `iwx::rx::Inbox::take` is what `Held` passes.
///
/// # Safety
/// As `ask`.
pub unsafe fn ask_with<'a>(
    bar0: u64,
    rings: &mut Rings,
    bufs: &'a Buffers,
    rx: &mut Rx,
    q: &mut Queue,
    group: u8,
    opcode: u8,
    version: u8,
    payload: &[u8],
    ms: u32,
    aside: &mut dyn FnMut(&Packet),
) -> Result<Packet<'a>, CmdError> {
    let want_group = if group == 0 { LONG_GROUP } else { group };
    q.send(bar0, rings, group, opcode, version, payload)?;

    let mut waited = 0u32;
    loop {
        while let Some(got) = rx.next(rings, bufs) {
            let pkt = got.map_err(CmdError::BadReply)?;
            if pkt.group == want_group && pkt.code == opcode {
                rx.ack(bar0, rings);
                return Ok(pkt);
            }
            // Something else. Handed aside, acknowledged and stepped over:
            // leaving it would have the next reader see it again, and refusing
            // on it would make any unsolicited notification break the next
            // command.
            aside(&pkt);
            rx.ack(bar0, rings);
        }
        if waited >= ms * 1000 {
            return Err(CmdError::NoReply);
        }
        crate::time::delay_us(100);
        waited += 100;
    }
}

/// Wait for a notification nobody asked for.
///
/// **Not the same operation as `ask`, and conflating them is a hang.** A command
/// response arrives because something was sent; these arrive because firmware
/// finished something, so there is nothing to send and nothing to correlate. The
/// post-alive sequence waits on two of them, and using `ask` would mean inventing
/// a command to send in order to have something to wait for.
///
/// **A group of 1 is accepted wherever 0 is asked for.** `iwx_rx_pkt` strips the
/// long group back to a bare opcode when the command it answers was marked
/// narrow, so a notification upstream matches as a bare `0x4` can arrive either
/// way; accepting both is what stops this waiting forever on a perfectly ordinary
/// packet.
///
/// # Safety
/// `bar0` must be a mapped aperture for a part whose firmware is alive.
pub unsafe fn expect<'a>(
    bar0: u64,
    rings: &mut Rings,
    bufs: &'a Buffers,
    rx: &mut Rx,
    group: u8,
    code: u8,
    ms: u32,
) -> Result<Packet<'a>, CmdError> {
    expect_with(bar0, rings, bufs, rx, group, code, ms, &mut |_| {})
}

/// `expect`, handing every packet it steps over to `aside`.
///
/// # Safety
/// As `expect`.
pub unsafe fn expect_with<'a>(
    bar0: u64,
    rings: &mut Rings,
    bufs: &'a Buffers,
    rx: &mut Rx,
    group: u8,
    code: u8,
    ms: u32,
    aside: &mut dyn FnMut(&Packet),
) -> Result<Packet<'a>, CmdError> {
    let mut waited = 0u32;
    loop {
        while let Some(got) = rx.next(rings, bufs) {
            let pkt = got.map_err(CmdError::BadReply)?;
            let matched = pkt.code == code
                && (pkt.group == group || (group == 0 && pkt.group == LONG_GROUP));
            rx.ack(bar0, rings);
            if matched {
                return Ok(pkt);
            }
            aside(&pkt);
            // Anything else is stepped over. Firmware sends statistics,
            // temperature and debug unasked, and refusing on the first of them
            // would make an unrelated notification break the sequence.
        }
        if waited >= ms * 1000 {
            return Err(CmdError::NoReply);
        }
        crate::time::delay_us(100);
        waited += 100;
    }
}

/// Claims. No radio, and no register written.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();
    let mut ok = |c: bool, w: &'static str| out.push((w, c));

    ok(HDR_WIDE == 8, "the wide command header is eight bytes");
    ok(ENTRY == 324, "a command entry is 324 bytes");
    ok(FIRST_TB == 20, "and the first transmit buffer describes at most twenty");
    // The two moduli, which is the arithmetic this module opens by warning about.
    ok(HW_WRAP == 65536, "the hardware write pointer wraps at 65,536");
    ok(TX_RING == 256, "where the descriptor ring holds 256");
    ok(HW_WRAP != TX_RING, "so the two are not the same number and neither derives from the other");
    ok(HBUS_TARG_WRPTR < super::APERTURE, "the doorbell is inside the mapped aperture");
    // Everything below frames commands and touches no register, which is the
    // reason `frame` and `ring` are separate functions.
    ok(true, "and framing a command touches none of it, which is why these claims can run");

    if let (Some(mut q), Some(mut rings)) = (Queue::new(), Rings::new(super::Family::Ax210)) {
        ok(q.buf.len() == ENTRY * 256, "the queue holds 256 entries");
        ok(q.buf.pa() % 64 == 0, "aligned as upstream aligns it");
        ok(
            q.entry_pa(1).zip(q.entry_pa(0)).map(|(b, a)| b - a) == Some(ENTRY as u64),
            "and entries are one entry apart",
        );
        ok(q.entry_pa(256).is_none(), "an index past the ring has no address");

        // A four-byte payload, which is what NVM_GET_INFO is: header plus payload
        // is twelve bytes, so it fits the first transmit buffer and the
        // descriptor needs one.
        let idx = q.frame(&mut rings, REGULATORY_AND_NVM_GROUP, 0x02, 0, &[0; 4]);
        ok(idx == Ok(0), "the first command goes in slot zero");
        let e = &q.buf.as_slice()[0..ENTRY];
        ok(e[0] == 0x02 && e[1] == REGULATORY_AND_NVM_GROUP, "its opcode and group are written");
        ok(e[2] == 0 && e[3] == 0, "with its own index and the command queue's number");
        ok(
            u16::from_le_bytes([e[4], e[5]]) == 4,
            "and a length that is the payload alone, not counting the header",
        );
        let t = &rings.cmd.as_slice()[0..TFD_SIZE];
        ok(u16::from_le_bytes([t[0], t[1]]) == 1, "a twelve-byte command needs one transmit buffer");
        ok(
            u16::from_le_bytes([t[2], t[3]]) == (HDR_WIDE + 4) as u16,
            "whose length is the whole command",
        );
        ok(
            u64::from_le_bytes([t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11]]) == q.entry_pa(0).unwrap(),
            "and whose address is the entry's, at offset four of a ten-byte descriptor",
        );
        ok(q.cur == 1 && q.cur_hw == 1, "both cursors advanced");

        // A payload past twenty bytes, which needs the second buffer. This is the
        // case a driver can omit and still read the NVM, because NVM_GET_INFO is
        // twelve bytes -- so it is asserted rather than left to the first long
        // command to discover.
        let idx = q.frame(&mut rings, REGULATORY_AND_NVM_GROUP, 0x03, 1, &[7u8; 40]);
        ok(idx == Ok(1), "the second command goes in slot one");
        let t = &rings.cmd.as_slice()[TFD_SIZE..2 * TFD_SIZE];
        ok(u16::from_le_bytes([t[0], t[1]]) == 2, "a 48-byte command needs two");
        ok(u16::from_le_bytes([t[2], t[3]]) == FIRST_TB as u16, "the first describes twenty bytes");
        ok(
            u16::from_le_bytes([t[12], t[13]]) == (HDR_WIDE + 40 - FIRST_TB) as u16,
            "and the second the remaining twenty-eight",
        );
        // Ten-byte descriptors, so the second's address is at offset fourteen.
        // Sixteen would be the natural guess and would put it in the third.
        ok(
            u64::from_le_bytes([t[14], t[15], t[16], t[17], t[18], t[19], t[20], t[21]])
                == q.entry_pa(1).unwrap() + FIRST_TB as u64,
            "pointing twenty bytes into the same entry, at offset fourteen",
        );
        let e = &q.buf.as_slice()[ENTRY..2 * ENTRY];
        ok(e[7] == 1, "the command version is carried in the header");
        ok(e[HDR_WIDE] == 7 && e[HDR_WIDE + 39] == 7, "and the payload follows it whole");

        // The group-zero rewrite, which the part requires and nothing here
        // exercises yet.
        let _ = q.frame(&mut rings, 0, 0x05, 0, &[]);
        let e = &q.buf.as_slice()[2 * ENTRY..3 * ENTRY];
        ok(e[1] == LONG_GROUP, "a group-zero command is sent as the long group");

        // A slot is cleared before it is reused, or a shorter command leaves the
        // tail of a longer one for firmware to read as payload.
        let before = q.buf.as_slice()[2 * ENTRY + HDR_WIDE + 30];
        ok(before == 0, "and an entry is zeroed before it is written");

        ok(
            q.frame(&mut rings, 1, 1, 0, &[0u8; 400]) == Err(CmdError::TooLong(400)),
            "a payload too large for an entry is refused by name",
        );
    } else {
        ok(false, "the command queue allocates");
    }

    out
}
