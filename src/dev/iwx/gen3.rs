//! What an AX210-family part boots out of, which is what is in the GF63.
//!
//! Family 22000 hands the device one descriptor and a pointer to it. AX210 hands
//! it *three* objects and a fourth to bootstrap with:
//!
//! - a **peripheral scratch** area, which is where the firmware image addresses
//!   and the receive ring now live -- everything `ctxt::ContextInfo` carried;
//! - a **peripheral info** page, which firmware writes its own boot progress and
//!   queue indices into, so it is the only region here the device *writes*;
//! - a small **context info** header, 104 bytes, which is little more than
//!   pointers to those two plus the message rings;
//! - an **image loader**, a separate blob out of the firmware file, which the
//!   part runs in order to fetch everything else.
//!
//! And the entry point changes with it: `CSR_CTXT_INFO_ADDR` rather than
//! `CSR_CTXT_INFO_BA`, plus the loader's address and length, plus a boot-enable
//! bit. Five register pairs where family 22000 had one.
//!
//! ### These are byte layouts and not `repr(C)` structs, and that is forced
//!
//! `ctxt::ContextInfo` could be `repr(C)` because every field in it happens to
//! be naturally aligned. Neither structure here is: `prph_scratch_rbd_cfg` is a
//! `u64` and a `u32`, twelve bytes, so everything after it sits four bytes out of
//! phase and the DRAM array of `u64` begins at offset 124. `context_info_gen3`
//! is worse -- `mtr_base_addr` is a `u64` at offset 52. Under `__packed` that is
//! exactly what the hardware reads; under `repr(C)` the compiler would insert
//! four bytes and every address after it would be read from the wrong place.
//!
//! `repr(C, packed)` would express it and then forbid taking a reference to any
//! field, so every write would need `write_unaligned` anyway. Writing the bytes
//! at named offsets is the same operation with the offsets visible, which is what
//! the claims need to address -- and it is the shape `linux::signal` uses for
//! `rt_sigframe` for the same reason.
//!
//! ### Provenance
//!
//! As `ctxt`: read from OpenBSD's `iwx(4)`, which took the numbers from Intel's
//! dual BSD/GPLv2 headers, and what is used is the BSD arm -- reproduced in full,
//! with its conditions and disclaimer, in `NOTICE.md` at the repository root. The
//! offsets are the silicon's; the names and the prose are this tree's.

use alloc::string::String;
use alloc::vec::Vec;

use super::ctxt::{self, Dma, Placed, Rings};
use super::fw;

// --- the peripheral scratch --------------------------------------------------

/// Offsets inside the peripheral scratch area.
///
/// Derived by walking the packed C structure field by field, and every one is
/// asserted against the walk as well as written down -- two statements, so a
/// slip in either shows as a disagreement.
pub mod scratch_at {
    // ctrl_cfg.version
    pub const MAC_ID: usize = 0;
    pub const VERSION: usize = 2;
    pub const SIZE_DW: usize = 4;
    // ctrl_cfg.control
    pub const CONTROL_FLAGS: usize = 8;
    // ctrl_cfg.pnvm_cfg
    pub const PNVM_BASE: usize = 16;
    pub const PNVM_SIZE: usize = 24;
    // ctrl_cfg.hwm_cfg
    pub const HWM_BASE: usize = 32;
    pub const HWM_SIZE: usize = 40;
    pub const DEBUG_TOKEN: usize = 44;
    // ctrl_cfg.rbd_cfg -- **twelve bytes, not sixteen**, which is what throws
    // everything below it four bytes out of phase.
    pub const FREE_RBD_ADDR: usize = 48;
    pub const RBD_RESERVED: usize = 56;
    // ctrl_cfg.reduce_power_cfg
    pub const REDUCE_POWER_BASE: usize = 60;
    pub const REDUCE_POWER_SIZE: usize = 68;
    /// End of `ctrl_cfg`, and it is not a multiple of eight.
    pub const CTRL_CFG_END: usize = 76;
    pub const RESERVED: usize = 76;
    /// The three firmware image arrays, the same `context_info_dram` family
    /// 22000 carries -- at an offset that is 4-aligned and not 8.
    pub const UMAC_IMG: usize = 124;
    pub const LMAC_IMG: usize = 124 + 512;
    pub const VIRTUAL_IMG: usize = 124 + 1024;
}

/// The whole scratch area: 76 of control, 48 reserved, 1536 of image addresses.
pub const SCRATCH_SIZE: usize = 1660;

// --- the context info header -------------------------------------------------

/// Offsets inside the 104-byte context-info header.
pub mod at {
    pub const VERSION: usize = 0;
    pub const SIZE: usize = 2;
    pub const CONFIG: usize = 4;
    pub const PRPH_INFO_BASE: usize = 8;
    /// Where firmware reports the receive producer index.
    pub const CR_HEAD_IDX_ARR: usize = 16;
    pub const TR_TAIL_IDX_ARR: usize = 24;
    pub const CR_TAIL_IDX_ARR: usize = 32;
    pub const TR_HEAD_IDX_ARR: usize = 40;
    pub const CR_IDX_ARR_SIZE: usize = 48;
    pub const TR_IDX_ARR_SIZE: usize = 50;
    /// The transmit message ring: the command queue. **A `u64` at 52**, which is
    /// the field that makes `repr(C)` unusable for this structure.
    pub const MTR_BASE: usize = 52;
    pub const MCR_BASE: usize = 60;
    pub const MTR_SIZE: usize = 68;
    pub const MCR_SIZE: usize = 70;
    pub const MTR_DOORBELL_VEC: usize = 72;
    pub const MCR_DOORBELL_VEC: usize = 74;
    pub const MTR_MSI_VEC: usize = 76;
    pub const MCR_MSI_VEC: usize = 78;
    pub const MTR_OPT_HEADER_SIZE: usize = 80;
    pub const MTR_OPT_FOOTER_SIZE: usize = 81;
    pub const MCR_OPT_HEADER_SIZE: usize = 82;
    pub const MCR_OPT_FOOTER_SIZE: usize = 83;
    pub const MSG_RINGS_CTRL_FLAGS: usize = 84;
    pub const PRPH_INFO_MSI_VEC: usize = 86;
    pub const PRPH_SCRATCH_BASE: usize = 88;
    pub const PRPH_SCRATCH_SIZE: usize = 96;
    pub const RESERVED: usize = 100;
}

/// The header's length.
pub const GEN3_SIZE: usize = 104;

/// The peripheral info page.
///
/// **A whole page for a sixteen-byte structure, and the page is the point.** The
/// structure itself is four words of boot progress at offset zero, but the
/// context info also points firmware at two index arrays *inside the same
/// region*, at half and three quarters of a page. Allocating only the structure
/// would have the device write those indices past the end of it.
pub const PRPH_INFO_SIZE: usize = 4096;
/// Where the transmit tail indices go: half a page in.
pub const TR_TAIL_OFFSET: usize = PRPH_INFO_SIZE / 2;
/// And the completion tail indices: three quarters.
pub const CR_TAIL_OFFSET: usize = PRPH_INFO_SIZE * 3 / 4;

// --- the control word --------------------------------------------------------

const PRPH_SCRATCH_RB_SIZE_4K: u32 = 1 << 16;
const PRPH_SCRATCH_MTR_MODE: u32 = 1 << 17;
const PRPH_SCRATCH_MTR_FORMAT: u32 = (1 << 18) | (1 << 19);
const PRPH_MTR_FORMAT_256B: u32 = 0xC_0000;

/// The scratch control word.
///
/// **The receive-buffer size moved from a shifted code to a single bit.** Family
/// 22000 encodes it as a four-bit ladder at position 9; here 4K is one bit at 16
/// and there is no ladder, so carrying the gen1 encoding across would set bits
/// 9..13 and leave the size unspecified.
///
/// `MTR_FORMAT_256B` is masked before it is or-ed, which looks redundant and is
/// not: the constant is `0xC0000`, which is bits 18 and 19 *already in place*,
/// and the mask is the assertion that it does not reach outside the field. Kept
/// as upstream writes it, because a constant that is silently wider than its
/// field is exactly what a mask catches.
pub fn control_flags() -> u32 {
    PRPH_SCRATCH_RB_SIZE_4K | PRPH_SCRATCH_MTR_MODE | (PRPH_MTR_FORMAT_256B & PRPH_SCRATCH_MTR_FORMAT)
}

// --- writing bytes at offsets ------------------------------------------------

fn put16(b: &mut [u8], at: usize, v: u16) {
    b[at..at + 2].copy_from_slice(&v.to_le_bytes());
}
fn put32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}
fn put64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

/// Read a `u64` back out, for the suite and for `iwx ctxt`.
pub fn get64(d: &Dma, at: usize) -> Option<u64> {
    let b = d.as_slice().get(at..at + 8)?;
    Some(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
}
pub fn get32(d: &Dma, at: usize) -> Option<u32> {
    let b = d.as_slice().get(at..at + 4)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}
pub fn get16(d: &Dma, at: usize) -> Option<u16> {
    let b = d.as_slice().get(at..at + 2)?;
    Some(u16::from_le_bytes([b[0], b[1]]))
}

// --- what a boot needs held alive -------------------------------------------

/// Every region the part will read or write, kept together.
///
/// **Held as one object because the lifetimes are one fact.** The device fetches
/// from all of these after the kick and there is no completion to wait on until
/// firmware reports alive, so dropping any of them early hands the allocator
/// memory a DMA is still walking. One owner, one drop, and the ordering problem
/// disappears rather than being documented.
pub struct Boot {
    pub info: Dma,
    pub scratch: Dma,
    pub prph_info: Dma,
    pub iml: Dma,
    pub rings: Rings,
    /// One region per loadable firmware section.
    pub sections: Vec<Dma>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Error {
    /// The firmware file carries no image loader. On this family that is fatal
    /// and it is the *file* that is wrong, not the part: the loader is what
    /// fetches everything else, so there is nothing to fall back to.
    NoImageLoader,
    /// A section's bytes were not where the parse said they were.
    BadSection,
    /// Out of memory for one of the regions.
    NoMemory,
    Group(ctxt::GroupError),
}

impl Error {
    pub fn why(&self) -> String {
        match self {
            Error::NoImageLoader => String::from(
                "the firmware file carries no image loader (TLV 52), which this family boots through",
            ),
            Error::BadSection => String::from("a firmware section's bytes are outside the file"),
            Error::NoMemory => String::from("not enough contiguous memory for the boot regions"),
            Error::Group(g) => g.why(),
        }
    }
}

/// Build everything an AX210 part needs to boot.
///
/// `hw_rev` is the raw `CSR_HW_REV`; its low sixteen bits are echoed back in the
/// scratch area, as on family 22000 -- but into the *scratch* and not into the
/// context info, which carries no revision at all.
pub fn build(hw_rev: u32, image: &fw::Image, file: &[u8]) -> Result<Boot, Error> {
    let iml_range = image.iml.ok_or(Error::NoImageLoader)?;
    let placed = ctxt::group(&image.sections).map_err(Error::Group)?;
    let rings = Rings::new(super::Family::Ax210).ok_or(Error::NoMemory)?;

    // --- the scratch ---------------------------------------------------------
    let mut scratch = Dma::new(SCRATCH_SIZE, 8).ok_or(Error::NoMemory)?;
    let mut sections: Vec<Dma> = Vec::new();
    {
        let b = scratch.as_mut_slice();
        put16(b, scratch_at::MAC_ID, hw_rev as u16);
        put16(b, scratch_at::VERSION, 0);
        put16(b, scratch_at::SIZE_DW, (SCRATCH_SIZE / 4) as u16);
        put32(b, scratch_at::CONTROL_FLAGS, control_flags());
        // **Only the free ring goes here.** The used ring and the status block
        // move into the context info on this family, which is the transposition
        // most likely to be got wrong by carrying gen1's grouping across.
        put64(b, scratch_at::FREE_RBD_ADDR, rings.free.pa());
        // Platform NVM is left zero, and that is not the same as owing nothing.
        // A nonzero address here would have the part fetch a platform NVM at boot
        // from memory nothing has written; what firmware is owed instead is the
        // *handshake* in `init`, which rings a doorbell saying to proceed without
        // one. This tree recorded "no PNVM is needed for this part" for a while,
        // which was right about the file and wrong about the handshake -- and the
        // wrong half would have hung the sequence a step later.
    }
    for p in &placed {
        let end = p.sect.at.checked_add(p.sect.len).ok_or(Error::BadSection)?;
        let src = file.get(p.sect.at..end).ok_or(Error::BadSection)?;
        let mut d = Dma::new(p.sect.len.max(1), 8).ok_or(Error::NoMemory)?;
        d.as_mut_slice()[..src.len()].copy_from_slice(src);
        let pa = d.pa();
        let slot = p.slot * 8;
        let base = match p.dest {
            ctxt::Dest::Umac => scratch_at::UMAC_IMG,
            ctxt::Dest::Lmac => scratch_at::LMAC_IMG,
            ctxt::Dest::Paging => scratch_at::VIRTUAL_IMG,
        };
        put64(scratch.as_mut_slice(), base + slot, pa);
        sections.push(d);
    }

    // --- the page firmware writes into ---------------------------------------
    let prph_info = Dma::new(PRPH_INFO_SIZE, 8).ok_or(Error::NoMemory)?;

    // --- the image loader ----------------------------------------------------
    let (iml_at, iml_len) = iml_range;
    let iml_src = file.get(iml_at..iml_at + iml_len).ok_or(Error::BadSection)?;
    let mut iml = Dma::new(iml_len.max(1), 8).ok_or(Error::NoMemory)?;
    iml.as_mut_slice()[..iml_src.len()].copy_from_slice(iml_src);

    // --- the header ----------------------------------------------------------
    let mut info = Dma::new(GEN3_SIZE, 8).ok_or(Error::NoMemory)?;
    {
        let b = info.as_mut_slice();
        // `version` and `size` are left zero, which is upstream's behaviour and
        // worth stating because gen1 sets its length field and a reader moving
        // between the two would fill this one in. The header's length is implied
        // by the family here.
        put64(b, at::PRPH_INFO_BASE, prph_info.pa());
        put64(b, at::PRPH_SCRATCH_BASE, scratch.pa());
        put32(b, at::PRPH_SCRATCH_SIZE, SCRATCH_SIZE as u32);
        // The status block, which on this family is two bytes rather than
        // sixteen and holds only the completion head.
        put64(b, at::CR_HEAD_IDX_ARR, rings.stat.pa());
        // And the two tail arrays, which live *inside the peripheral info page*
        // at fixed fractions of it rather than in regions of their own.
        put64(b, at::TR_TAIL_IDX_ARR, prph_info.pa() + TR_TAIL_OFFSET as u64);
        put64(b, at::CR_TAIL_IDX_ARR, prph_info.pa() + CR_TAIL_OFFSET as u64);
        // The message rings: transmit is the command queue, completion is the
        // used receive ring. Both were in the scratch's business on gen1.
        put64(b, at::MTR_BASE, rings.cmd.pa());
        put64(b, at::MCR_BASE, rings.used.pa());
        // Sizes as exponents, from the same two encodings family 22000 uses --
        // which is why they stayed in `ctxt` rather than being copied here.
        put16(b, at::MTR_SIZE, ctxt::tfd_cb_size(ctxt::TX_RING).unwrap_or(0) as u16);
        put16(b, at::MCR_SIZE, ctxt::rb_cb_size(ctxt::RX_RING).unwrap_or(0) as u16);
    }

    Ok(Boot { info, scratch, prph_info, iml, rings, sections })
}

/// Claims. No radio, no firmware file, no register touched.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();
    let mut ok = |c: bool, w: &'static str| out.push((w, c));

    // --- the two layouts, walked as well as written down ---------------------

    // **Every offset is re-derived by adding up field widths**, and compared
    // against the table. Both statements come from the same C declaration and
    // neither is the other's source, so a mistyped constant disagrees with the
    // walk and a miscounted field width disagrees with the constant. Under
    // `repr(C)` the compiler was the second opinion; these are byte layouts, so
    // the arithmetic has to be.
    let mut w = 0usize;
    ok(w == scratch_at::MAC_ID, "scratch: mac_id at 0");
    w += 2;
    ok(w == scratch_at::VERSION, "version follows it");
    w += 2;
    ok(w == scratch_at::SIZE_DW, "then the length in double words");
    w += 2 + 2; // size, reserved
    ok(w == scratch_at::CONTROL_FLAGS, "the control word at 8");
    w += 4 + 4; // control_flags, reserved
    ok(w == scratch_at::PNVM_BASE, "platform NVM at 16");
    w += 8;
    ok(w == scratch_at::PNVM_SIZE, "its size at 24");
    w += 4 + 4;
    ok(w == scratch_at::HWM_BASE, "the hardware monitor at 32");
    w += 8;
    ok(w == scratch_at::HWM_SIZE, "its size at 40");
    w += 4;
    ok(w == scratch_at::DEBUG_TOKEN, "the debug token at 44");
    w += 4;
    ok(w == scratch_at::FREE_RBD_ADDR, "the free receive ring at 48");
    w += 8;
    ok(w == scratch_at::RBD_RESERVED, "and one reserved word after it, not two");
    w += 4;
    // The whole reason this file is byte layouts. Twelve bytes for the ring
    // configuration puts everything below it four out of phase.
    ok(w == scratch_at::REDUCE_POWER_BASE, "so reduce-power lands at 60, not 64");
    w += 8;
    ok(w == scratch_at::REDUCE_POWER_SIZE, "its size at 68");
    w += 4 + 4;
    ok(w == scratch_at::CTRL_CFG_END, "and the control block ends at 76");
    ok(scratch_at::CTRL_CFG_END % 8 != 0, "which is not a multiple of eight");
    w += 48; // reserved[12]
    ok(w == scratch_at::UMAC_IMG, "the UMAC image array at 124");
    ok(scratch_at::UMAC_IMG % 8 != 0, "at an offset no repr(C) would put a u64 at");
    ok(
        scratch_at::LMAC_IMG - scratch_at::UMAC_IMG == 512,
        "each image array is 64 addresses, as on family 22000",
    );
    ok(
        scratch_at::VIRTUAL_IMG + 512 == SCRATCH_SIZE,
        "and the paged array is the last thing in the scratch",
    );
    ok(SCRATCH_SIZE == 1660, "the scratch is 1660 bytes");
    ok(SCRATCH_SIZE / 4 == 415, "which is 415 double words, the figure firmware is told");

    // The header, walked the same way.
    let mut h = 0usize;
    ok(h == at::VERSION, "header: version at 0");
    h += 2;
    ok(h == at::SIZE, "size at 2");
    h += 2;
    ok(h == at::CONFIG, "config at 4");
    h += 4;
    ok(h == at::PRPH_INFO_BASE, "the peripheral info page at 8");
    h += 8;
    ok(h == at::CR_HEAD_IDX_ARR, "the completion head array at 16");
    h += 8 * 4; // cr_head, tr_tail, cr_tail, tr_head
    ok(h == at::CR_IDX_ARR_SIZE, "the array sizes at 48");
    h += 2 + 2;
    // The field that forces the whole file's shape.
    ok(h == at::MTR_BASE, "and the transmit ring at 52, a u64 on a four-byte boundary");
    ok(at::MTR_BASE % 8 != 0, "which is what makes repr(C) unusable here");
    h += 8;
    ok(h == at::MCR_BASE, "the completion ring at 60");
    h += 8;
    ok(h == at::MTR_SIZE, "their sizes at 68 and 70");
    h += 2 + 2 + 2 + 2 + 2 + 2; // mtr/mcr size, doorbell, msi
    ok(h == at::MTR_OPT_HEADER_SIZE, "the four optional-header bytes at 80");
    h += 4;
    ok(h == at::MSG_RINGS_CTRL_FLAGS, "the ring control flags at 84");
    h += 2 + 2;
    ok(h == at::PRPH_SCRATCH_BASE, "the scratch pointer at 88");
    h += 8;
    ok(h == at::PRPH_SCRATCH_SIZE, "its size at 96");
    h += 4 + 4;
    ok(h == GEN3_SIZE, "and the header is 104 bytes");

    // --- the control word ---------------------------------------------------

    // 4K as one bit at 16, where family 22000 spells it as a four-bit code at
    // position 9. Carrying gen1's encoding across would set bits 9..13 and leave
    // the size unspecified, which is the mistake this claim exists to catch.
    ok(control_flags() & (1 << 16) != 0, "4K buffers are one bit here, not a code");
    ok(
        control_flags() != ctxt::control_flags(ctxt::RX_RING, ctxt::RB_SIZE_4K).unwrap_or(0),
        "so the two families' control words are not the same value",
    );
    ok(control_flags() & (1 << 17) != 0, "the transmit ring is in message mode");
    ok(control_flags() == 0x3_0000 | 0xC_0000, "and the whole word is 0xf0000");
    // The mask on the format constant, which looks redundant: the constant is
    // already positioned, so the mask asserts it does not reach past its field.
    ok(
        PRPH_MTR_FORMAT_256B & !PRPH_SCRATCH_MTR_FORMAT == 0,
        "the 256-byte format code fits inside its own two bits",
    );

    // --- the peripheral info page -------------------------------------------

    // A page for sixteen bytes, because two index arrays live inside it at fixed
    // fractions. Allocating the structure alone would have the device write past
    // its end.
    ok(PRPH_INFO_SIZE == 4096, "the peripheral info region is a whole page");
    ok(TR_TAIL_OFFSET == 2048 && CR_TAIL_OFFSET == 3072, "with the tail arrays half and three quarters in");
    ok(
        16 < TR_TAIL_OFFSET,
        "and the structure itself fits below the first of them",
    );

    // --- the whole build ----------------------------------------------------

    // An image with an image loader, which this family cannot boot without.
    let img = fw::build(
        "so-a0-hr-b0",
        1,
        7,
        &[
            (19, fw::section_body(0x0080_0000, &[1, 2, 3, 4])),
            (19, fw::section_body(fw::CPU1_CPU2_SEPARATOR, &[])),
            (19, fw::section_body(0x00a0_0000, &[5, 6])),
            (52, alloc::vec![0xaa, 0xbb, 0xcc, 0xdd]),
        ],
    );
    let parsed = fw::parse(&img);
    ok(parsed.is_ok(), "an image carrying a loader parses");
    if let Ok(image) = &parsed {
        ok(image.iml == Some((image.iml.unwrap().0, 4)), "and the loader's four bytes are located");
        match build(0x0000_0433, image, &img) {
            Ok(b) => {
                ok(b.info.len() == GEN3_SIZE, "the header builds at 104 bytes");
                ok(b.scratch.len() == SCRATCH_SIZE, "and the scratch at 1660");
                ok(b.iml.len() == 4, "the loader is copied into its own region");
                ok(
                    b.iml.as_slice() == [0xaa, 0xbb, 0xcc, 0xdd],
                    "byte for byte, since the part executes it",
                );
                // The header is pointers, and every one of them must be the
                // region it names. A transposition here is a part that fetches
                // its firmware addresses out of the page it was meant to write
                // its boot progress into.
                ok(
                    get64(&b.info, at::PRPH_SCRATCH_BASE) == Some(b.scratch.pa()),
                    "the header points at the scratch",
                );
                ok(
                    get32(&b.info, at::PRPH_SCRATCH_SIZE) == Some(SCRATCH_SIZE as u32),
                    "and says how long it is",
                );
                ok(
                    get64(&b.info, at::PRPH_INFO_BASE) == Some(b.prph_info.pa()),
                    "and at the page firmware writes into",
                );
                ok(
                    get64(&b.info, at::TR_TAIL_IDX_ARR) == Some(b.prph_info.pa() + 2048),
                    "the transmit tail array is inside that page, not beside it",
                );
                ok(
                    get64(&b.info, at::CR_TAIL_IDX_ARR) == Some(b.prph_info.pa() + 3072),
                    "and the completion tail array three quarters in",
                );
                ok(
                    get64(&b.info, at::MTR_BASE) == Some(b.rings.cmd.pa())
                        && get64(&b.info, at::MCR_BASE) == Some(b.rings.used.pa()),
                    "the message rings are the command queue and the used receive ring",
                );
                ok(
                    get64(&b.info, at::CR_HEAD_IDX_ARR) == Some(b.rings.stat.pa()),
                    "and the completion head array is the status block",
                );
                ok(get16(&b.info, at::MTR_SIZE) == Some(5), "the transmit ring's exponent is 5");
                ok(get16(&b.info, at::MCR_SIZE) == Some(9), "and the completion ring's is 9");
                // Upstream leaves both of these zero and gen1 does not, so a
                // reader moving between the two families would fill them in.
                ok(
                    get16(&b.info, at::VERSION) == Some(0) && get16(&b.info, at::SIZE) == Some(0),
                    "the header's own version and size stay zero, unlike gen1's",
                );
                // **Only the free ring is in the scratch.** The used ring and the
                // status block moved into the header on this family, which is the
                // transposition most likely to survive being carried across from
                // gen1 unnoticed.
                ok(
                    get64(&b.scratch, scratch_at::FREE_RBD_ADDR) == Some(b.rings.free.pa()),
                    "the scratch carries the free receive ring",
                );
                ok(
                    get16(&b.scratch, scratch_at::SIZE_DW) == Some(415),
                    "and its own length in double words",
                );
                ok(
                    get16(&b.scratch, scratch_at::MAC_ID) == Some(0x0433),
                    "the revision is echoed into the scratch, not the header",
                );
                ok(
                    get32(&b.scratch, scratch_at::CONTROL_FLAGS) == Some(control_flags()),
                    "and so is the control word",
                );
                ok(
                    get64(&b.scratch, scratch_at::PNVM_BASE) == Some(0),
                    "platform NVM is left zero, being loaded by a later command",
                );
                // The firmware sections, in the scratch's arrays rather than the
                // header's -- there are none in the header.
                ok(
                    get64(&b.scratch, scratch_at::LMAC_IMG) != Some(0)
                        && get64(&b.scratch, scratch_at::UMAC_IMG) != Some(0),
                    "both images' addresses are in the scratch",
                );
                ok(
                    get64(&b.scratch, scratch_at::LMAC_IMG) != get64(&b.scratch, scratch_at::UMAC_IMG),
                    "and they are different regions",
                );
                ok(b.sections.len() == 2, "one region per loadable section");
                ok(
                    b.sections.first().map(|d| d.as_slice().first().copied()) == Some(Some(1)),
                    "and the bytes really got there",
                );
                // The rings are AX210's shapes, which is what makes the family
                // parameter load-bearing rather than decorative.
                ok(
                    b.rings.free.len() == 16 * ctxt::RX_RING as usize,
                    "the free ring uses 16-byte transfer descriptors",
                );
                ok(
                    b.rings.used.len() == 32 * ctxt::RX_RING as usize,
                    "and the used ring 32-byte completion descriptors",
                );
                ok(b.rings.stat.len() == 2, "with a two-byte status block");
            }
            Err(_) => ok(false, "the boot regions build"),
        }
    }

    // **An image with no loader is refused**, and this is the one refusal that
    // matters on this family: the loader is what fetches everything else, so
    // there is no degraded mode to fall back to. Family 22000 needs no loader at
    // all, which is why an image can be perfectly valid and still unbootable
    // here.
    let no_iml = fw::build("no loader", 1, 1, &[(19, fw::section_body(0x1000, &[1]))]);
    if let Ok(image) = fw::parse(&no_iml) {
        ok(image.iml.is_none(), "an image with no loader record says so");
        ok(
            build(0, &image, &no_iml).map(|_| ()) == Err(Error::NoImageLoader),
            "and building an AX210 boot from it is refused by name",
        );
    }

    out
}

// --- starting it -------------------------------------------------------------

/// Registers the kick touches. All inside the first page, which is all that is
/// mapped.
const CSR_CTXT_INFO_ADDR: u64 = 0x118;
const CSR_IML_DATA_ADDR: u64 = 0x120;
const CSR_IML_SIZE_ADDR: u64 = 0x128;
/// **The same register as `CSR_HW_IF_CONFIG_REG`.** Intel's header spells the
/// boot-control bit at offset zero, which is where the interface configuration
/// lives, so the enable is a bit in a register the power-up sequence has already
/// been writing. That is why it is set rather than written: writing it whole
/// would clear `NIC_READY` and the prepare bit along with it.
const CSR_CTXT_INFO_BOOT_CTRL: u64 = 0x0;
const CSR_AUTO_FUNC_BOOT_ENA: u32 = 1 << 1;
const CSR_LTR_LONG_VAL_AD: u64 = 0x0d4;
const CSR_GP_CNTRL: u64 = 0x024;
const HBUS_TARG_PRPH_WADDR: u64 = 0x400 + 0x044;
const HBUS_TARG_PRPH_WDAT: u64 = 0x400 + 0x04c;

const GP_CNTRL_MAC_ACCESS_REQ: u32 = 0x0000_0008;
const GP_CNTRL_MAC_ACCESS_EN: u32 = 0x0000_0001;
const GP_CNTRL_GOING_TO_SLEEP: u32 = 0x0000_0010;
const GP_CNTRL_MAC_CLOCK_READY: u32 = 0x0000_0001;

/// Where the "run the boot code" doorbell is, in peripheral address space.
const UREG_CPU_INIT_RUN: u32 = 0xa0_5c44;
/// AX210 puts the upper MAC's peripheral registers at an offset.
const UMAC_PRPH_OFFSET: u32 = 0x30_0000;
/// The peripheral address mask on this family. Family 22000 uses 0x000fffff, so
/// carrying that across would truncate every upper-MAC address.
const PRPH_ADDR_MASK: u32 = 0x00ff_ffff;

/// Which address a step writes, named so the sequence can be a table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum What {
    ContextInfo,
    ImageLoader,
    ImageLoaderLen,
}

/// One step of the kick.
///
/// A table for `POWER_UP`'s reason: no emulator models this part, so the ordering
/// is the thing that cannot otherwise be checked anywhere. The values are runtime
/// addresses, so what the table carries is *which* address goes where -- enough
/// for a claim to say the loader is given to the part before the boot bit is set,
/// which is the ordering that matters.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kick {
    /// A 64-bit register, **written as two 32-bit halves, low first.**
    ///
    /// Not an optimisation and not a portability hedge. Upstream's own comment
    /// says it: the device expects a 64-bit address but a single 64-bit write
    /// "won't work on some devices, such as the AX201" -- which is exactly the
    /// part this is for. A `write_volatile` of a `u64` would compile to one
    /// instruction and the part would take half an address.
    Addr64(u64, What),
    Word(u64, What),
    SetBit(u64, u32),
    /// Take the MAC access lock, which peripheral writes require.
    Lock,
    /// The latency tolerance workaround, before firmware sets its own.
    Ltr,
    /// A write into upper-MAC peripheral space.
    Prph(u32, u32),
    Unlock,
}

/// The sequence, in order.
pub const KICK: &[Kick] = &[
    // The descriptor first: the part must know where to read from before it is
    // told to read.
    Kick::Addr64(CSR_CTXT_INFO_ADDR, What::ContextInfo),
    // Then the loader and its length. The length is a plain 32-bit word, which is
    // why it is a separate variant rather than a short `Addr64`.
    Kick::Addr64(CSR_IML_DATA_ADDR, What::ImageLoader),
    Kick::Word(CSR_IML_SIZE_ADDR, What::ImageLoaderLen),
    // Only now enable automatic boot. Setting this before the loader's address
    // was written would start a part whose loader pointer is whatever the
    // register held.
    Kick::SetBit(CSR_CTXT_INFO_BOOT_CTRL, CSR_AUTO_FUNC_BOOT_ENA),
    // Peripheral space needs the lock, and the latency workaround has to be in
    // before firmware starts touching the link.
    Kick::Lock,
    Kick::Ltr,
    // And the doorbell, which is the instruction that actually starts it.
    Kick::Prph(UREG_CPU_INIT_RUN, 1),
    Kick::Unlock,
];

/// The latency-tolerance word, derived from each field's own mask.
///
/// **The reference implementation gets this wrong and the claims below catch
/// it.** OpenBSD's `iwx_set_ltr` places each field with an explicit shift
/// constant beside a mask, and for the two scale fields the pair disagrees:
/// `NO_SNOOP_SCALE_MASK` is `0x1c000000`, which is bits 26 to 28, while
/// `NO_SNOOP_SCALE_SHIFT` is 24. So `(2 << 24) & 0x1c000000` is **zero** -- the
/// scale is masked away, both fields come out empty, and the register is written
/// as `0x80fa80fa` where the function's own comment says it is setting "the LTR
/// to ~250 usec". A scale of zero is nanoseconds, so it asks for 250 ns: a
/// thousandfold tighter than intended, on the workaround that exists to stop the
/// link misbehaving during boot.
///
/// Linux places the same fields with `u32_encode_bits`, which shifts by the
/// mask's own lowest set bit and therefore cannot disagree with it, and produces
/// `0x88fa88fa`. That is what this computes.
///
/// So the shift is *derived* from the mask here rather than carried beside it.
/// Two constants that must agree and are written down twice are two constants
/// that will not -- which is the argument this tree makes about `knob.py`
/// checking its table against the source, arriving on a pair of #defines.
pub fn ltr_value() -> u32 {
    const NO_SNOOP_REQ: u32 = 0x8000_0000;
    const NO_SNOOP_SCALE: u32 = 0x1c00_0000;
    const NO_SNOOP_VAL: u32 = 0x03ff_0000;
    const SNOOP_REQ: u32 = 0x0000_8000;
    const SNOOP_SCALE: u32 = 0x0000_1c00;
    const SNOOP_VAL: u32 = 0x0000_03ff;
    /// The PCIe latency-tolerance scale code for microseconds.
    const SCALE_USEC: u32 = 2;
    const USEC: u32 = 250;
    NO_SNOOP_REQ
        | place(SCALE_USEC, NO_SNOOP_SCALE)
        | place(USEC, NO_SNOOP_VAL)
        | SNOOP_REQ
        | place(SCALE_USEC, SNOOP_SCALE)
        | place(USEC, SNOOP_VAL)
}

/// Put a value in a field, positioned by the field's own mask.
///
/// The value is masked *after* shifting, so one too large for its field is
/// truncated rather than spilling into the field above it. That is the safer
/// direction here: a truncated latency figure is a wrong number and a spilled one
/// is a wrong number in somebody else's field.
const fn place(v: u32, mask: u32) -> u32 {
    (v << mask.trailing_zeros()) & mask
}

/// The word written to the peripheral address register.
///
/// The `3 << 24` is a byte-enable field: all four bytes. Without it the write
/// lands with no bytes enabled, which is a write that does nothing and reports
/// nothing.
pub fn prph_waddr(addr: u32) -> u32 {
    (addr & PRPH_ADDR_MASK) | (3 << 24)
}

/// An upper-MAC peripheral address.
pub fn umac_prph(addr: u32) -> u32 {
    addr + UMAC_PRPH_OFFSET
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KickFault {
    /// The MAC access lock was never granted, so a peripheral write would have
    /// gone nowhere. Refused rather than attempted, because a peripheral write
    /// without the lock is silently dropped.
    NoLock,
}

impl KickFault {
    pub fn why(&self) -> &'static str {
        "the MAC access lock was not granted, so the boot doorbell would have been dropped"
    }
}

/// Run the kick. **This starts the firmware.**
///
/// Takes the built regions rather than addresses, so a caller cannot pass the
/// scratch where the context info belongs -- and holds them by reference for the
/// length of the call, which is the beginning of the lifetime `Boot` owns and not
/// the end of it: the part goes on fetching after this returns, until it reports
/// alive.
///
/// # Safety
///
/// `bar0` must be a mapped register aperture for an AX210-family part whose
/// power-up sequence has completed. Nothing here can check that, which is why it
/// is `unsafe` and why `Radio::boot` is the way in.
pub unsafe fn kick(bar0: u64, b: &Boot) -> Result<(), KickFault> {
    for step in KICK {
        match *step {
            Kick::Addr64(reg, what) => {
                let pa = match what {
                    What::ContextInfo => b.info.pa(),
                    What::ImageLoader => b.iml.pa(),
                    What::ImageLoaderLen => 0,
                };
                // Low half first. The order is not arbitrary: the part latches on
                // the high half, so writing high-then-low would latch a
                // half-formed address.
                core::ptr::write_volatile((bar0 + reg) as *mut u32, pa as u32);
                core::ptr::write_volatile((bar0 + reg + 4) as *mut u32, (pa >> 32) as u32);
            }
            Kick::Word(reg, what) => {
                let v = match what {
                    What::ImageLoaderLen => b.iml.len() as u32,
                    _ => 0,
                };
                core::ptr::write_volatile((bar0 + reg) as *mut u32, v);
            }
            Kick::SetBit(reg, bit) => {
                let p = (bar0 + reg) as *mut u32;
                core::ptr::write_volatile(p, core::ptr::read_volatile(p) | bit);
            }
            Kick::Lock => {
                if !lock(bar0) {
                    return Err(KickFault::NoLock);
                }
            }
            Kick::Ltr => {
                core::ptr::write_volatile((bar0 + CSR_LTR_LONG_VAL_AD) as *mut u32, ltr_value());
            }
            Kick::Prph(addr, val) => prph_write(bar0, umac_prph(addr), val),
            Kick::Unlock => {
                let p = (bar0 + CSR_GP_CNTRL) as *mut u32;
                core::ptr::write_volatile(p, core::ptr::read_volatile(p) & !GP_CNTRL_MAC_ACCESS_REQ);
            }
        }
    }
    Ok(())
}

/// Write one word into peripheral space.
///
/// The address must be latched before the data, and nothing in the type system
/// orders two volatile writes to different addresses -- volatile does, which is
/// why both are volatile rather than only the second.
///
/// Takes an address already in upper-MAC space, so a caller that forgot
/// `umac_prph` writes somewhere real rather than being corrected here. That is the
/// honest split: this function cannot tell which space an address belongs to.
///
/// # Safety
/// `bar0` must be a mapped aperture for a part holding the MAC access lock.
pub unsafe fn prph_write(bar0: u64, addr: u32, val: u32) {
    core::ptr::write_volatile((bar0 + HBUS_TARG_PRPH_WADDR) as *mut u32, prph_waddr(addr));
    core::ptr::write_volatile((bar0 + HBUS_TARG_PRPH_WDAT) as *mut u32, val);
}

/// Release the MAC access lock.
///
/// Separate from `kick`'s own release because `nvm` takes the lock for a pair of
/// register reads and has nothing else to do under it -- and a lock taken and not
/// released leaves the part unable to sleep.
///
/// # Safety
/// `bar0` must be a mapped aperture for this part.
pub unsafe fn unlock(bar0: u64) {
    let p = (bar0 + CSR_GP_CNTRL) as *mut u32;
    core::ptr::write_volatile(p, core::ptr::read_volatile(p) & !GP_CNTRL_MAC_ACCESS_REQ);
}

/// Ask for access to the MAC's own registers.
///
/// The wait is for `MAC_ACCESS_EN` **with `GOING_TO_SLEEP` clear**, and both
/// halves matter: a part on its way into a low-power state can report access
/// granted and then take it away, so the mask covers the sleep bit and the wanted
/// value does not include it.
pub fn lock(bar0: u64) -> bool {
    // Safety: the caller's aperture, and the offset is inside the first page.
    unsafe {
        let p = (bar0 + CSR_GP_CNTRL) as *mut u32;
        core::ptr::write_volatile(p, core::ptr::read_volatile(p) | GP_CNTRL_MAC_ACCESS_REQ);
    }
    crate::time::delay_us(2);
    let mut waited = 0u32;
    loop {
        // Safety: as above.
        let v = unsafe { core::ptr::read_volatile((bar0 + CSR_GP_CNTRL) as *const u32) };
        if v & (GP_CNTRL_MAC_CLOCK_READY | GP_CNTRL_GOING_TO_SLEEP) == GP_CNTRL_MAC_ACCESS_EN {
            return true;
        }
        if waited >= 150_000 {
            return false;
        }
        crate::time::delay_us(10);
        waited += 10;
    }
}

/// Claims about the kick. Nothing is written to any register.
pub fn kick_checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();
    let mut ok = |c: bool, w: &'static str| out.push((w, c));

    let idx = |pred: fn(&Kick) -> bool| KICK.iter().position(pred);
    let ctx = idx(|k| matches!(k, Kick::Addr64(_, What::ContextInfo)));
    let iml = idx(|k| matches!(k, Kick::Addr64(_, What::ImageLoader)));
    let len = idx(|k| matches!(k, Kick::Word(_, What::ImageLoaderLen)));
    let ena = idx(|k| matches!(k, Kick::SetBit(_, b) if *b == CSR_AUTO_FUNC_BOOT_ENA));
    let lock = idx(|k| matches!(k, Kick::Lock));
    let ltr = idx(|k| matches!(k, Kick::Ltr));
    let bell = idx(|k| matches!(k, Kick::Prph(a, _) if *a == UREG_CPU_INIT_RUN));
    let unlock = idx(|k| matches!(k, Kick::Unlock));

    // The ordering that matters: everything the part will read must be in a
    // register before the bit that tells it to read.
    ok(ctx < ena, "the descriptor's address is written before boot is enabled");
    ok(iml < ena, "and so is the loader's");
    ok(len < ena, "and its length");
    ok(iml < len || len < iml, "the loader and its length are both written");
    ok(ena < bell, "boot is enabled before the doorbell is rung");
    // The lock is what makes a peripheral write land at all.
    ok(lock < bell, "the MAC lock is taken before the doorbell, which is a peripheral write");
    ok(bell < unlock, "and released after it");
    ok(lock < ltr && ltr < bell, "the latency workaround goes in under the lock, before the doorbell");
    ok(unlock == Some(KICK.len() - 1), "the lock is released last, on the way out");

    // Every register inside the one mapped page.
    ok(
        KICK.iter().all(|k| match k {
            Kick::Addr64(r, _) => *r + 4 < 0x1000,
            Kick::Word(r, _) | Kick::SetBit(r, _) => *r < 0x1000,
            Kick::Lock | Kick::Unlock => CSR_GP_CNTRL < 0x1000,
            Kick::Ltr => CSR_LTR_LONG_VAL_AD < 0x1000,
            Kick::Prph(_, _) => HBUS_TARG_PRPH_WADDR < 0x1000 && HBUS_TARG_PRPH_WDAT < 0x1000,
        }),
        "every register the kick touches is inside the mapped aperture",
    );
    // **A 64-bit register is written as two halves**, which is the AX201's own
    // documented quirk, so a variant that wrote eight bytes at once would be the
    // bug. Asserted as the absence of any single wide write.
    ok(
        KICK.iter().any(|k| matches!(k, Kick::Addr64(_, _))),
        "addresses go through the two-halves variant",
    );
    // The boot-enable bit shares a register with the interface configuration, so
    // it must be set and never written.
    ok(
        CSR_CTXT_INFO_BOOT_CTRL == 0x0,
        "the boot control bit lives at offset zero, with the interface configuration",
    );
    ok(
        KICK.iter().all(|k| !matches!(k, Kick::Word(0x0, _))),
        "so nothing writes that register whole",
    );

    // The derived constants.
    ok(ltr_value() == 0x88fa_88fa, "the latency word derives to 0x88fa88fa");
    ok(
        ltr_value() & 0xffff == (ltr_value() >> 16) & 0xffff,
        "and its two halves are identical, which is what the two requests mean",
    );
    // **The reference's own value, asserted as the wrong one.** OpenBSD places
    // the scale with an explicit shift of 24 against a mask covering bits 26 to
    // 28, so the field is masked away and the register is written as 0x80fa80fa
    // -- a scale of zero, which is nanoseconds. Kept as a claim rather than a
    // comment because it is the one number here that looks plausible while being
    // a thousandfold wrong, and because re-introducing the explicit shift is the
    // obvious thing for the next reader of that header to do.
    ok(
        (2u32 << 24) & 0x1c00_0000 == 0,
        "placing the scale by an explicit shift of 24 would zero it, as upstream does",
    );
    ok(
        ltr_value() != 0x80fa_80fa,
        "so this is not the value the reference implementation writes",
    );
    ok(
        place(2, 0x1c00_0000) == 0x0800_0000,
        "the scale goes at bit 26, which is where its mask begins",
    );
    // Derived positions cannot drift from their masks, which is the whole reason
    // this is `place` and not a shift constant per field.
    ok(place(1, 0x0000_03ff) == 1, "a field at bit zero is not shifted");
    ok(place(0xffff, 0x0000_03ff) == 0x3ff, "and a value too large is truncated, not spilled");
    ok(prph_waddr(0) == 3 << 24, "a peripheral write enables all four bytes");
    ok(
        prph_waddr(0xffff_ffff) == 0x03ff_ffff,
        "and the address is masked to this family's 24 bits, not 22000's 20",
    );
    ok(umac_prph(UREG_CPU_INIT_RUN) == 0xd0_5c44, "the doorbell is at 0xd05c44 in upper-MAC space");
    ok(
        prph_waddr(umac_prph(UREG_CPU_INIT_RUN)) & 0x00ff_ffff == 0xd0_5c44,
        "which survives the mask, where 22000's would have truncated it",
    );
    // The sleep bit in the lock's mask. Without it a part on its way down reports
    // access granted and then removes it.
    ok(
        GP_CNTRL_GOING_TO_SLEEP != 0 && GP_CNTRL_MAC_ACCESS_EN & GP_CNTRL_GOING_TO_SLEEP == 0,
        "the lock waits for access granted with the sleep bit clear",
    );

    out
}
