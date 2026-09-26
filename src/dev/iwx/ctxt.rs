//! The structure a family-22000 radio boots out of, and the memory it reads it
//! from.
//!
//! **Family 22000, which the GF63's part is not.** That was found after this was
//! written and it is the more useful half of the file: `8086:51f0` decodes to
//! Snow Owl, so it is AX210 family and takes a *different* descriptor -- a
//! `context_info_gen3` plus a peripheral scratch area plus an image loader,
//! through `CSR_CTXT_INFO_ADDR` rather than `CSR_CTXT_INFO_BA`. The part is sold
//! as an AX201 because the marketing name follows the radio module, which is why
//! the wrong family looked right for as long as it did.
//!
//! So what is here is correct, asserted, and for the AX200 and the AX201 as
//! fitted to earlier platforms. It is kept rather than deleted because the pieces
//! underneath it are the same on both -- `Dma`, the section grouping, the two
//! size encodings -- and because the gen3 descriptor is the next increment rather
//! than a rewrite of this one. `iwx::Family` is what chooses between them and it
//! refuses rather than guessing.
//!
//! Family 22000 does not take its firmware a register at a time. The driver builds one 1,792-byte descriptor in host memory, fills
//! it with the physical addresses of everything the part will need, writes that
//! descriptor's own address to `CSR_CTXT_INFO_BA`, and tells the part to go. The
//! firmware then fetches its own image by DMA. So nearly all of the bring-up is
//! *laying out memory correctly*, and only the last two writes touch a register.
//!
//! **Which is why this is the half that can be built without the radio.** No
//! emulator models an Intel wireless part, so every register write here is
//! unverifiable until the laptop is in front of somebody. The descriptor is not:
//! it is bytes at offsets, the grouping of firmware sections is a pure function
//! of a parsed image, and the two size encodings are arithmetic. All of that is
//! asserted at boot against a synthetic image, and what is left for the GF63 is
//! whether the part answers.
//!
//! ### It is an ABI, so a field one short is a different structure
//!
//! The same argument `linux::signal` makes about `rt_sigframe`. Firmware reads
//! `dram.lmac_img` at offset 192 + 512 whatever this kernel believes, so a
//! missing reserved field does not make a smaller descriptor -- it makes one
//! where every address after the gap is read as something else, and the failure
//! is a part that fetches its microcode from whatever the heap held. Every offset
//! is therefore a claim, and so is the total.
//!
//! Written flat rather than as nine nested structures. The C header nests them
//! and the nesting carries no information the offsets do not: flat is what the
//! claims can address, and a comment names which sub-structure each group was.
//!
//! ### Provenance, and the licence
//!
//! The offsets, the flag positions and the two `*_CB_SIZE` encodings were read
//! from OpenBSD's `iwx(4)` (`if_iwxreg.h`, `if_iwxvar.h` and `if_iwx.c` at
//! rev 1.75/1.50), which took them in turn from Intel's own headers -- those are
//! **dual BSD/GPLv2**, Copyright(c) 2017 Intel Deutschland GmbH and
//! Copyright(c) 2018-2019 Intel Corporation, and what is used here is the BSD
//! arm. The bytes below are a hardware interface rather than anybody's
//! expression: no code was copied, the names and every comment are this tree's,
//! and what was taken is the set of numbers the silicon dictates. Recorded the
//! way `dev/registry` records a support level, because a number with no source
//! is a number the next reader cannot check.
//!
//! ### DMA, in a kernel with one address space and no IOMMU
//!
//! Every address in this structure is one the device will read or write with no
//! translation and nothing to stop it. There is no protection to add -- an IOMMU
//! is not set up here and this kernel is identity mapped, so the only safety
//! property available is that the device is never told about memory that is not
//! ours. That is why `Dma` owns its allocation and hands out a physical address
//! only for memory it allocated itself, and why `Drop` is the hazard: freeing a
//! region the part is still fetching from hands the allocator memory the next
//! DMA will overwrite. The fifth instance in this tree of "put it back before
//! giving it away", and the first where the other party is not the processor.

use alloc::string::String;
use alloc::vec::Vec;

use super::fw;

// --- DMA memory --------------------------------------------------------------

/// A region the radio may read or write, owned by us.
///
/// **Physical equals virtual, and that is load-bearing rather than convenient.**
/// `build_identity_map` maps everything at its own address, so a heap pointer
/// *is* a bus address and there is no translation to get wrong -- the same
/// property `cpu::code` leans on to execute from an allocation and `mem::space`
/// checks before writing CR3. It is asserted rather than assumed, because the day
/// it stops being true this module silently hands the device the wrong addresses
/// and the symptom is a part that fetches microcode from somebody else's memory.
pub struct Dma {
    ptr: *mut u8,
    layout: core::alloc::Layout,
}

impl Dma {
    /// Allocate `len` bytes aligned to `align`, zeroed.
    ///
    /// Zeroed because the descriptor has reserved fields and unused DRAM slots,
    /// and firmware reads all of them. A slot holding heap debris is an address
    /// the part will try to fetch from.
    pub fn new(len: usize, align: usize) -> Option<Dma> {
        // A zero-length region has no address worth handing over, and `Layout`
        // refuses a non-power-of-two alignment -- both are programming errors
        // here rather than conditions, so they answer `None` rather than panic.
        if len == 0 || !align.is_power_of_two() {
            return None;
        }
        let layout = core::alloc::Layout::from_size_align(len, align).ok()?;
        // Safety: the layout is non-zero-sized and validated above.
        let ptr = unsafe { alloc::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return None;
        }
        Some(Dma { ptr, layout })
    }

    /// The address to give the device. Identity mapped, so this is the virtual
    /// address unchanged.
    pub fn pa(&self) -> u64 {
        self.ptr as u64
    }

    pub fn len(&self) -> usize {
        self.layout.size()
    }

    pub fn as_slice(&self) -> &[u8] {
        // Safety: our own allocation, of exactly this length.
        unsafe { core::slice::from_raw_parts(self.ptr, self.layout.size()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // Safety: as above, and `&mut self` is the only writer.
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.layout.size()) }
    }
}

impl Drop for Dma {
    fn drop(&mut self) {
        // **Nothing here can stop the device first**, and that is stated rather
        // than solved: a `Dma` must outlive every fetch the part will make from
        // it, which is an ordering its owner is responsible for. The context
        // info may be released once firmware reports alive; the paging regions
        // may not be released until the device is down. Upstream keeps those in
        // two different lifetimes for exactly this reason.
        //
        // Safety: allocated by us with this layout and not freed twice.
        unsafe { alloc::alloc::dealloc(self.ptr, self.layout) }
    }
}

// --- the two size encodings --------------------------------------------------

/// The cyclic-buffer size exponent: the field holds the log, not the count.
///
/// `IWX_RX_QUEUE_CB_SIZE` is `fls(x) - 1`, which for a power of two is its
/// base-two logarithm. Written as the logarithm rather than transcribing the
/// find-last-set spelling, because `fls` on a non-power-of-two is a different
/// function from `ilog2` and every ring size here is a power of two -- so the two
/// agree on every input this can be given, and the logarithm is the one whose
/// name says what the field means.
///
/// Four bits wide, so a ring of 32,768 or more would silently write into the
/// field above it. `entries` is refused past that rather than masked.
pub fn rb_cb_size(entries: u32) -> Option<u32> {
    if !entries.is_power_of_two() {
        return None;
    }
    let e = entries.trailing_zeros();
    // Upstream asserts `< 0xF`. The field is four bits, so 15 would be the
    // largest representable value and the assertion excludes it; kept exactly,
    // because a bound copied loosely is a bound that stops holding.
    if e >= 0xF {
        return None;
    }
    Some(e)
}

/// The transmit queue's size encoding, which is the same logarithm less three.
///
/// `IWX_TFD_QUEUE_CB_SIZE(x)` is `IWX_RX_QUEUE_CB_SIZE(x) - 3`. The three is not
/// a fudge: the field counts in units of eight descriptors, so the smallest
/// queue it can describe is eight and a ring of four would encode as a negative
/// number. Refused rather than wrapped, which as a `u8` on the wire would read
/// as an enormous queue.
pub fn tfd_cb_size(entries: u32) -> Option<u8> {
    let e = rb_cb_size(entries)?;
    if e < 3 {
        return None;
    }
    Some((e - 3) as u8)
}

/// Receive-buffer size codes. Only the one this part uses is named; the rest of
/// the ladder is in the header and none of it is reachable from here.
pub const RB_SIZE_4K: u32 = 0x4;

const CTXT_INFO_TFD_FORMAT_LONG: u32 = 1 << 8;
const CTXT_INFO_RB_CB_SIZE_POS: u32 = 4;
const CTXT_INFO_RB_SIZE_POS: u32 = 9;

/// The control word.
///
/// **The long TFD format is not optional.** Upstream's own comment says the
/// short format is not supported by the driver, and the descriptor layout this
/// module allocates is the long one -- 256 bytes with twenty-five buffer
/// descriptors. Setting the sizes without this bit would have the part read
/// those rings in a format nothing here writes.
///
/// Two shifts sitting five bits apart with a flag between them, which is exactly
/// the arithmetic that is wrong silently: `RB_SIZE_POS` is **9** and not the 12
/// it would be if the two four-bit fields were adjacent. Derived here and
/// asserted against the number for this part's own configuration, so a wrong
/// position shows as a disagreement rather than as a radio that does not answer.
pub fn control_flags(rx_entries: u32, rb_size: u32) -> Option<u32> {
    let cb = rb_cb_size(rx_entries)?;
    Some(CTXT_INFO_TFD_FORMAT_LONG | (cb << CTXT_INFO_RB_CB_SIZE_POS) | (rb_size << CTXT_INFO_RB_SIZE_POS))
}

// --- the descriptor ----------------------------------------------------------

/// How many DRAM addresses each of the three images may carry.
pub const MAX_DRAM_ENTRY: usize = 64;

/// The whole descriptor's length, which the part is also told in double words.
pub const SIZE: usize = 1792;

/// The device INIT configuration, flat.
///
/// Field names follow the C header's, flattened with the sub-structure's name as
/// a prefix where it disambiguates. Every reserved field is present and named,
/// because a reserved field left out is not a gap -- it is every later offset
/// moved.
#[repr(C)]
pub struct ContextInfo {
    // version
    pub mac_id: u16,
    pub version: u16,
    /// The descriptor's own length in double words. Firmware reads this to know
    /// how much to fetch, so a wrong value here mis-reads everything at once.
    pub size_dw: u16,
    pub version_reserved: u16,
    // control
    pub control_flags: u32,
    pub control_reserved: u32,
    pub reserved0: u64,
    // rbd_cfg
    pub free_rbd_addr: u64,
    pub used_rbd_addr: u64,
    pub status_wr_ptr: u64,
    // hcmd_cfg
    pub cmd_queue_addr: u64,
    pub cmd_queue_size: u8,
    pub hcmd_reserved: [u8; 7],
    pub reserved1: [u32; 4],
    // dump_cfg
    pub core_dump_addr: u64,
    pub core_dump_size: u32,
    pub dump_reserved: u32,
    // edbg_cfg
    pub early_debug_addr: u64,
    pub early_debug_size: u32,
    pub edbg_reserved: u32,
    // pnvm_cfg -- left zero on this family. Platform NVM is an AX210 mechanism
    // and a nonzero address here would have a 22000 part fetch one anyway.
    pub platform_nvm_addr: u64,
    pub platform_nvm_size: u32,
    pub pnvm_reserved: u32,
    pub reserved2: [u32; 16],
    // dram: note umac comes *first* in the structure and second in the section
    // list, which is the transposition this module exists to get right.
    pub umac_img: [u64; MAX_DRAM_ENTRY],
    pub lmac_img: [u64; MAX_DRAM_ENTRY],
    pub virtual_img: [u64; MAX_DRAM_ENTRY],
    pub reserved3: [u32; 16],
}

/// The offsets, as the C header lays them out. Named so a claim reads as a
/// comparison against the header rather than against arithmetic this file did.
pub mod at {
    pub const MAC_ID: usize = 0;
    pub const VERSION: usize = 2;
    pub const SIZE_DW: usize = 4;
    pub const CONTROL_FLAGS: usize = 8;
    pub const RESERVED0: usize = 16;
    pub const FREE_RBD_ADDR: usize = 24;
    pub const USED_RBD_ADDR: usize = 32;
    pub const STATUS_WR_PTR: usize = 40;
    pub const CMD_QUEUE_ADDR: usize = 48;
    pub const CMD_QUEUE_SIZE: usize = 56;
    pub const RESERVED1: usize = 64;
    pub const CORE_DUMP_ADDR: usize = 80;
    pub const EARLY_DEBUG_ADDR: usize = 96;
    pub const PLATFORM_NVM_ADDR: usize = 112;
    pub const RESERVED2: usize = 128;
    pub const UMAC_IMG: usize = 192;
    pub const LMAC_IMG: usize = 704;
    pub const VIRTUAL_IMG: usize = 1216;
    pub const RESERVED3: usize = 1728;
}

// --- splitting the section list ----------------------------------------------

/// Where a firmware section is meant to end up.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dest {
    /// The link-layer processor's image, which comes first in the file.
    Lmac,
    /// The upper MAC's image, after the first separator.
    Umac,
    /// Paged code, after the second. Released only when the device goes down,
    /// where the other two may be released as soon as firmware reports alive.
    Paging,
}

/// A section with the image it belongs to and its slot in that image's array.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Placed {
    pub dest: Dest,
    pub slot: usize,
    pub sect: fw::Section,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum GroupError {
    /// More sections for one image than the descriptor has slots. Carries which
    /// image and how many, because writing slot 64 would land in the next array
    /// and the symptom is the wrong processor's code.
    TooMany(Dest, usize),
    /// A separator appeared twice. Two `CPU1_CPU2` markers means the list cannot
    /// be split, and guessing which one is real is guessing where the UMAC image
    /// starts.
    RepeatedSeparator,
    /// Nothing to load at all.
    Empty,
}

impl GroupError {
    pub fn why(&self) -> String {
        match self {
            GroupError::TooMany(d, n) => alloc::format!(
                "{} sections for the {:?} image, and the descriptor holds {}",
                n, d, MAX_DRAM_ENTRY
            ),
            GroupError::RepeatedSeparator => String::from("a separator appeared twice, so the list cannot be split"),
            GroupError::Empty => String::from("no loadable sections"),
        }
    }
}

/// Split a parsed image's sections into the three arrays.
///
/// **Assigned by which separators have been passed, not by counting.** Upstream
/// counts to the first separator, then starts again one past it, then two past
/// the pair -- correct on a well-formed image and quietly wrong on one where the
/// paging separator arrives without the CPU separator, which files paged code as
/// the upper MAC's. A walk carrying its own state gives the same answer on every
/// image with both markers in order and a defined answer on the rest, and has no
/// `+1`/`+2` to get wrong.
///
/// The separators themselves are dropped rather than placed: `fw::Section`
/// already knows one when it sees it, and their `offset` is a marker where every
/// other section's is a destination.
pub fn group(sections: &[fw::Section]) -> Result<Vec<Placed>, GroupError> {
    let mut dest = Dest::Lmac;
    let mut seen_cpu = false;
    let mut seen_paging = false;
    let mut n = [0usize; 3];
    let mut out: Vec<Placed> = Vec::new();

    for s in sections {
        if s.offset == fw::CPU1_CPU2_SEPARATOR {
            if seen_cpu {
                return Err(GroupError::RepeatedSeparator);
            }
            seen_cpu = true;
            dest = Dest::Umac;
            continue;
        }
        if s.offset == fw::PAGING_SEPARATOR {
            if seen_paging {
                return Err(GroupError::RepeatedSeparator);
            }
            seen_paging = true;
            dest = Dest::Paging;
            continue;
        }
        let i = match dest {
            Dest::Lmac => 0,
            Dest::Umac => 1,
            Dest::Paging => 2,
        };
        if n[i] >= MAX_DRAM_ENTRY {
            return Err(GroupError::TooMany(dest, n[i] + 1));
        }
        out.push(Placed { dest, slot: n[i], sect: *s });
        n[i] += 1;
    }

    if out.is_empty() {
        return Err(GroupError::Empty);
    }
    Ok(out)
}

// --- the rings ---------------------------------------------------------------

/// Receive ring depth, and the transmit ring's.
///
/// Both are the family's own figures rather than choices: the receive count is
/// what the control word's exponent is computed from and the transmit count is
/// what the command queue's encoding is, so a different number here is a
/// different word written to the part.
pub const RX_RING: u32 = 512;
pub const TX_RING: u32 = 256;

/// One long-format transmit frame descriptor: two bytes of count, twenty-five
/// ten-byte buffer descriptors, four of padding.
pub const TFD_SIZE: usize = 256;

/// The four regions the descriptor points at.
///
/// **The entry sizes are the family's, and this is where it is easiest to reach
/// for the wrong ones.** An AX210 free-descriptor entry is a sixteen-byte
/// `rx_transfer_desc` and its used entry a thirty-two-byte completion
/// descriptor; on 22000 they are a bare `u64` address and a bare `u32` tag. The
/// structures exist in the same header and are the natural thing to use, and
/// using them here would have the part stride the ring at twice and eight times
/// the pitch.
pub struct Rings {
    /// Addresses of free receive buffers, 256-byte aligned.
    pub free: Dma,
    /// Tags of used ones, 256-byte aligned.
    pub used: Dma,
    /// The status block firmware writes its producer indices into.
    pub stat: Dma,
    /// The command queue's descriptors, 256-byte aligned.
    pub cmd: Dma,
}

/// The three per-family entry sizes, which are the trap this type exists around.
///
/// **All three differ between the families and none of them is the obvious
/// choice.** On family 22000 a free-descriptor entry is a bare `u64` address, a
/// used entry a bare `u32` tag, and the status block the sixteen-byte
/// `rb_status`. On AX210 the first two become real descriptor structures --
/// sixteen and thirty-two bytes -- and the status block shrinks to a single
/// `u16`. So reaching for `rx_transfer_desc` on 22000 strides the ring at twice
/// the pitch, and reaching for `u64` on AX210 strides it at half, and both are
/// silent: the part walks a ring of the right length at the wrong step.
pub struct RingSizes {
    pub free_entry: usize,
    pub used_entry: usize,
    pub stat: usize,
}

pub fn ring_sizes(family: super::Family) -> RingSizes {
    match family {
        super::Family::F22000 => RingSizes { free_entry: 8, used_entry: 4, stat: 16 },
        // `rx_transfer_desc` is 16 -- a tag, three reserved halves and a 64-bit
        // address -- and `rx_completion_desc` is 32. The status block is two
        // bytes, because on this family firmware keeps its indices in the
        // peripheral info page instead.
        super::Family::Ax210 => RingSizes { free_entry: 16, used_entry: 32, stat: 2 },
        // Bz shrinks the completion descriptor again, to four. Named so the table
        // is complete; nothing here drives that family.
        super::Family::Bz => RingSizes { free_entry: 16, used_entry: 4, stat: 2 },
    }
}

impl Rings {
    /// Allocate the four regions for a family.
    ///
    /// Takes the family rather than defaulting to one, because a default here is
    /// the mistake `ring_sizes` documents and it cannot be caught by anything
    /// downstream -- every address in the descriptor would be correct.
    pub fn new(family: super::Family) -> Option<Rings> {
        let z = ring_sizes(family);
        Some(Rings {
            free: Dma::new(z.free_entry * RX_RING as usize, 256)?,
            used: Dma::new(z.used_entry * RX_RING as usize, 256)?,
            // Sixteen-byte aligned whatever its length, which is upstream's
            // figure and not derived from the size -- a two-byte block aligned to
            // two would be legal arithmetic and the wrong alignment.
            stat: Dma::new(z.stat, 16)?,
            cmd: Dma::new(TFD_SIZE * TX_RING as usize, 256)?,
        })
    }
}

/// Build the descriptor into a DMA region, given the firmware's placed sections.
///
/// `hw_rev` is the raw `CSR_HW_REV` word; its low sixteen bits are the SKU and
/// revision the part expects to see echoed back. Passed in rather than read
/// here, because this function touches no register and is asserted with none.
///
/// Answers the region, and the DMA regions holding the firmware sections, which
/// the caller must keep alive. They are returned rather than stored for the
/// reason `Drop` gives: their lifetime is the caller's problem and hiding them in
/// a static would make it nobody's.
pub fn build(
    hw_rev: u32,
    placed: &[Placed],
    file: &[u8],
    rings: &Rings,
) -> Option<(Dma, Vec<Dma>)> {
    let mut sections: Vec<Dma> = Vec::new();
    let mut ci = ContextInfo {
        mac_id: hw_rev as u16,
        version: 0,
        size_dw: (SIZE / 4) as u16,
        version_reserved: 0,
        control_flags: control_flags(RX_RING, RB_SIZE_4K)?,
        control_reserved: 0,
        reserved0: 0,
        free_rbd_addr: rings.free.pa(),
        used_rbd_addr: rings.used.pa(),
        status_wr_ptr: rings.stat.pa(),
        cmd_queue_addr: rings.cmd.pa(),
        cmd_queue_size: tfd_cb_size(TX_RING)?,
        hcmd_reserved: [0; 7],
        reserved1: [0; 4],
        core_dump_addr: 0,
        core_dump_size: 0,
        dump_reserved: 0,
        early_debug_addr: 0,
        early_debug_size: 0,
        edbg_reserved: 0,
        platform_nvm_addr: 0,
        platform_nvm_size: 0,
        pnvm_reserved: 0,
        reserved2: [0; 16],
        umac_img: [0; MAX_DRAM_ENTRY],
        lmac_img: [0; MAX_DRAM_ENTRY],
        virtual_img: [0; MAX_DRAM_ENTRY],
        reserved3: [0; 16],
    };

    for p in placed {
        // The section's bytes are still in the file buffer; `fw::parse`
        // deliberately described them by range rather than copying, so this is
        // the first and only copy and it goes straight to the region the device
        // will read.
        let end = p.sect.at.checked_add(p.sect.len)?;
        let src = file.get(p.sect.at..end)?;
        let mut d = Dma::new(p.sect.len.max(1), 8)?;
        d.as_mut_slice()[..src.len()].copy_from_slice(src);
        let pa = d.pa();
        match p.dest {
            Dest::Lmac => ci.lmac_img[p.slot] = pa,
            Dest::Umac => ci.umac_img[p.slot] = pa,
            Dest::Paging => ci.virtual_img[p.slot] = pa,
        }
        sections.push(d);
    }

    let mut region = Dma::new(SIZE, 8)?;
    // Safety: `ContextInfo` is `repr(C)` and exactly `SIZE` bytes, which the
    // suite asserts, and the region is that long. Little-endian by the
    // architecture rather than by conversion, which is why there is no swap
    // here and a claim says the first word reads back as written.
    unsafe {
        core::ptr::copy_nonoverlapping(
            &ci as *const ContextInfo as *const u8,
            region.as_mut_slice().as_mut_ptr(),
            SIZE,
        );
    }
    Some((region, sections))
}

/// Read a `u64` out of a built descriptor, for the suite and for `iwx ctxt`.
pub fn field64(region: &Dma, off: usize) -> Option<u64> {
    let b = region.as_slice().get(off..off + 8)?;
    Some(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
}

/// Read a `u16`. The version block is three of them and the length is one, so
/// reading it as half of a `u32` is one shift away from answering about the
/// neighbouring field -- which is what the suite caught on the first run.
pub fn field16(region: &Dma, off: usize) -> Option<u16> {
    let b = region.as_slice().get(off..off + 2)?;
    Some(u16::from_le_bytes([b[0], b[1]]))
}

/// Read a `u32`.
pub fn field32(region: &Dma, off: usize) -> Option<u32> {
    let b = region.as_slice().get(off..off + 4)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Claims. No radio, no firmware file, and no register touched.
///
/// The offsets are checked against `at::`, which is the C header's own numbers,
/// *and* against the compiler's idea of where each field landed. Two independent
/// statements about one layout: a transcription error in the table shows as a
/// disagreement with `repr(C)`, and a `repr(C)` that inserted padding shows as a
/// disagreement with the table. Either alone would pass a wrong layout.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();

    // --- the layout ---------------------------------------------------------

    out.push(("the descriptor is 1792 bytes", core::mem::size_of::<ContextInfo>() == SIZE));
    out.push(("which is 448 double words, the figure firmware is told", SIZE / 4 == 448));
    // A `repr(C)` whose alignment forced trailing padding would still be 1792
    // only by luck, so the alignment is asserted too: eight, from the u64s.
    out.push(("and it is eight-byte aligned, so no tail padding was added", core::mem::align_of::<ContextInfo>() == 8));

    // Every offset the firmware reads. Computed from a real instance rather than
    // from `offset_of!`, which this toolchain may not have for nested paths --
    // a pointer difference against the base is the same statement.
    let z = ContextInfo {
        mac_id: 0, version: 0, size_dw: 0, version_reserved: 0,
        control_flags: 0, control_reserved: 0, reserved0: 0,
        free_rbd_addr: 0, used_rbd_addr: 0, status_wr_ptr: 0,
        cmd_queue_addr: 0, cmd_queue_size: 0, hcmd_reserved: [0; 7],
        reserved1: [0; 4], core_dump_addr: 0, core_dump_size: 0, dump_reserved: 0,
        early_debug_addr: 0, early_debug_size: 0, edbg_reserved: 0,
        platform_nvm_addr: 0, platform_nvm_size: 0, pnvm_reserved: 0,
        reserved2: [0; 16],
        umac_img: [0; MAX_DRAM_ENTRY], lmac_img: [0; MAX_DRAM_ENTRY],
        virtual_img: [0; MAX_DRAM_ENTRY], reserved3: [0; 16],
    };
    let base = &z as *const ContextInfo as usize;
    let off = |p: usize| p - base;

    out.push(("mac_id is at 0", off(&z.mac_id as *const _ as usize) == at::MAC_ID));
    out.push(("the control word is at 8", off(&z.control_flags as *const _ as usize) == at::CONTROL_FLAGS));
    out.push(("the free receive ring is at 24", off(&z.free_rbd_addr as *const _ as usize) == at::FREE_RBD_ADDR));
    out.push(("the used receive ring is at 32", off(&z.used_rbd_addr as *const _ as usize) == at::USED_RBD_ADDR));
    out.push(("the status write pointer is at 40", off(&z.status_wr_ptr as *const _ as usize) == at::STATUS_WR_PTR));
    out.push(("the command queue is at 48", off(&z.cmd_queue_addr as *const _ as usize) == at::CMD_QUEUE_ADDR));
    out.push(("its size byte is at 56", off(&z.cmd_queue_size as *const _ as usize) == at::CMD_QUEUE_SIZE));
    out.push(("the core dump block is at 80", off(&z.core_dump_addr as *const _ as usize) == at::CORE_DUMP_ADDR));
    out.push(("early debug is at 96", off(&z.early_debug_addr as *const _ as usize) == at::EARLY_DEBUG_ADDR));
    out.push(("platform NVM is at 112", off(&z.platform_nvm_addr as *const _ as usize) == at::PLATFORM_NVM_ADDR));
    // The three DRAM arrays, which are the whole point of the structure and the
    // ones where being wrong loads one processor's code into another.
    out.push(("the UMAC image array is at 192", off(&z.umac_img as *const _ as usize) == at::UMAC_IMG));
    out.push(("the LMAC image array is at 704", off(&z.lmac_img as *const _ as usize) == at::LMAC_IMG));
    out.push(("the paged image array is at 1216", off(&z.virtual_img as *const _ as usize) == at::VIRTUAL_IMG));
    out.push(("and the tail reserve is at 1728", off(&z.reserved3 as *const _ as usize) == at::RESERVED3));
    // **UMAC comes first in the structure and second in the file.** The one
    // transposition that produces a part which fetches a complete, valid image
    // into the wrong processor, so it is asserted as an inequality rather than
    // left to the offsets above.
    out.push((
        "the UMAC array precedes the LMAC one, which the section order does not",
        at::UMAC_IMG < at::LMAC_IMG,
    ));
    out.push(("each image array holds 64 addresses", at::LMAC_IMG - at::UMAC_IMG == MAX_DRAM_ENTRY * 8));

    // --- the encodings ------------------------------------------------------

    out.push(("a 512-entry receive ring encodes as the exponent 9", rb_cb_size(512) == Some(9)));
    out.push(("a 256-entry transmit ring encodes as 5, three less", tfd_cb_size(256) == Some(5)));
    // The bound upstream asserts. 32,768 would encode as 15, which the four-bit
    // field cannot hold without touching the flag above it.
    out.push(("a ring needing exponent 15 is refused, not masked", rb_cb_size(32768).is_none()));
    out.push(("and one below eight descriptors is refused by the transmit encoding", tfd_cb_size(4).is_none()));
    out.push(("a count that is not a power of two is refused", rb_cb_size(500).is_none()));

    // The control word for this part's configuration, derived and written down.
    // 0x990 is long TFD (bit 8), exponent 9 at position 4, and the 4K code at
    // position 9 -- and it is the claim that catches `RB_SIZE_POS` being the 12
    // it would be if the two fields were adjacent, which they are not.
    out.push((
        "the control word for a 512-entry 4K ring is 0x990",
        control_flags(RX_RING, RB_SIZE_4K) == Some(0x990),
    ));
    out.push((
        "the long TFD format bit is set, since the short one is not implemented",
        control_flags(RX_RING, RB_SIZE_4K).map(|f| f & (1 << 8) != 0) == Some(true),
    ));
    out.push((
        "and the two size fields do not overlap",
        (9u32 << CTXT_INFO_RB_CB_SIZE_POS) & (RB_SIZE_4K << CTXT_INFO_RB_SIZE_POS) == 0,
    ));

    // --- splitting the section list -----------------------------------------

    // A synthetic image shaped like a real one: two LMAC sections, a separator,
    // one UMAC, a separator, two paged. Built through `fw::build` and read back
    // through `fw::parse`, so what is grouped is what the parser really produces
    // rather than a hand-made list the parser would never emit.
    let img = fw::build(
        "synthetic",
        1,
        2,
        &[
            (19, fw::section_body(0x0080_0000, &[1, 2, 3, 4])),
            (19, fw::section_body(0x0090_0000, &[5, 6, 7, 8])),
            (19, fw::section_body(fw::CPU1_CPU2_SEPARATOR, &[])),
            (19, fw::section_body(0x00a0_0000, &[9, 10])),
            (19, fw::section_body(fw::PAGING_SEPARATOR, &[])),
            (19, fw::section_body(0x00b0_0000, &[11])),
            (19, fw::section_body(0x00c0_0000, &[12])),
        ],
    );
    let parsed = fw::parse(&img);
    out.push(("the synthetic image parses", parsed.is_ok()));
    if let Ok(image) = &parsed {
        let g = group(&image.sections);
        out.push(("and its sections group", g.is_ok()));
        if let Ok(p) = &g {
            let n = |d: Dest| p.iter().filter(|x| x.dest == d).count();
            out.push(("two sections before the first separator are the LMAC image", n(Dest::Lmac) == 2));
            out.push(("one between the separators is the UMAC image", n(Dest::Umac) == 1));
            out.push(("two after the second are paged", n(Dest::Paging) == 2));
            out.push(("the separators themselves are not placed", p.len() == 5));
            // Slots restart per image. Sharing one counter would put the UMAC
            // image at index 2 of its own array and leave 0 and 1 as addresses
            // firmware would fetch from.
            out.push((
                "each image's slots start at zero",
                p.iter().filter(|x| x.dest == Dest::Umac).all(|x| x.slot == 0)
                    && p.iter().filter(|x| x.dest == Dest::Paging).map(|x| x.slot).min() == Some(0),
            ));
            out.push((
                "and they are consecutive within an image",
                p.iter().filter(|x| x.dest == Dest::Lmac).map(|x| x.slot).collect::<Vec<_>>() == alloc::vec![0, 1],
            ));
        }
    }

    // No separator at all: everything is the first processor's, which is what
    // upstream's counting also answers.
    let flat = fw::build("flat", 1, 1, &[(19, fw::section_body(0x1000, &[1]))]);
    if let Ok(image) = fw::parse(&flat) {
        let g = group(&image.sections);
        out.push((
            "an image with no separator is all LMAC",
            g.map(|p| p.len() == 1 && p[0].dest == Dest::Lmac) == Ok(true),
        ));
    }

    // Sixty-five sections for one image. The slot past the last would be the
    // first address of the next array, so this must refuse rather than write it.
    let mut many: Vec<(u32, Vec<u8>)> = Vec::new();
    for i in 0..(MAX_DRAM_ENTRY + 1) {
        many.push((19, fw::section_body(0x1000 + i as u32, &[0])));
    }
    let big = fw::build("too many", 1, 1, &many);
    if let Ok(image) = fw::parse(&big) {
        out.push((
            "sixty-five sections for one image are refused, not written past the array",
            matches!(group(&image.sections), Err(GroupError::TooMany(Dest::Lmac, 65))),
        ));
    }

    // A repeated separator cannot be split, and guessing is guessing where the
    // upper MAC's image starts.
    let dup = fw::build(
        "two separators",
        1,
        1,
        &[
            (19, fw::section_body(0x1000, &[1])),
            (19, fw::section_body(fw::CPU1_CPU2_SEPARATOR, &[])),
            (19, fw::section_body(0x2000, &[2])),
            (19, fw::section_body(fw::CPU1_CPU2_SEPARATOR, &[])),
        ],
    );
    if let Ok(image) = fw::parse(&dup) {
        out.push((
            "a repeated separator is refused rather than guessed at",
            group(&image.sections) == Err(GroupError::RepeatedSeparator),
        ));
    }

    // --- DMA memory ---------------------------------------------------------

    if let Some(d) = Dma::new(4096, 256) {
        out.push(("a DMA region honours its alignment", d.pa() % 256 == 0));
        out.push(("and comes back zeroed, since firmware reads every reserved byte", d.as_slice().iter().all(|&b| b == 0)));
        // The property the whole module rests on. `paging::query` reads the real
        // tables rather than a shadow, so this is the hardware's answer and not
        // a belief about the map.
        out.push((
            "and is identity mapped, so its address is the one to give the device",
            crate::mem::paging::query(d.pa()).is_some(),
        ));
    } else {
        out.push(("a DMA region could be allocated", false));
    }
    out.push(("a zero-length region is refused", Dma::new(0, 8).is_none()));
    out.push(("and so is an alignment that is not a power of two", Dma::new(64, 3).is_none()));

    // --- the whole thing ----------------------------------------------------

    if let (Ok(image), Some(rings)) = (fw::parse(&img), Rings::new(super::Family::F22000)) {
        // The ring regions, whose alignments the part strides by.
        out.push(("the free receive ring is 256-byte aligned", rings.free.pa() % 256 == 0));
        out.push(("the used one too", rings.used.pa() % 256 == 0));
        out.push(("the status block is 16-byte aligned", rings.stat.pa() % 16 == 0));
        out.push(("the command queue is 256-byte aligned", rings.cmd.pa() % 256 == 0));
        // Entry sizes are the family's, not AX210's. A `u64` per free entry and a
        // `u32` per used one; the sixteen- and thirty-two-byte descriptor structs
        // in the same header belong to the later part.
        out.push(("on family 22000 the free ring is one bare address per entry", rings.free.len() == 8 * RX_RING as usize));
        out.push(("and the used ring one bare tag", rings.used.len() == 4 * RX_RING as usize));
        // The pair that says the family really is consulted. Same ring depth,
        // different pitch, and nothing downstream could tell.
        out.push((
            "AX210 uses real descriptors instead, so its rings are wider",
            ring_sizes(super::Family::Ax210).free_entry == 16
                && ring_sizes(super::Family::Ax210).used_entry == 32,
        ));
        out.push((
            "and its status block is two bytes where 22000's is sixteen",
            ring_sizes(super::Family::Ax210).stat == 2 && ring_sizes(super::Family::F22000).stat == 16,
        ));
        out.push(("and the command queue is 256 long-format descriptors", rings.cmd.len() == TFD_SIZE * TX_RING as usize));

        if let Ok(placed) = group(&image.sections) {
            match build(0x0000_0351, &placed, &img, &rings) {
                Some((region, sections)) => {
                    out.push(("the descriptor builds", region.len() == SIZE));
                    // Read back through the bytes rather than through the struct,
                    // because what firmware sees is the bytes.
                    out.push((
                        "its length field reads back as 448 double words",
                        field16(&region, at::SIZE_DW) == Some(448),
                    ));
                    // And the version is zero, which is the field the first
                    // version of the claim above was accidentally reading -- kept
                    // as a claim of its own so the two can never be confused
                    // again by a shift.
                    out.push((
                        "and the version field beside it is zero",
                        field16(&region, at::VERSION) == Some(0),
                    ));
                    out.push((
                        "the low half of CSR_HW_REV is echoed into mac_id",
                        field16(&region, at::MAC_ID) == Some(0x0351),
                    ));
                    out.push((
                        "the control word is in the descriptor, not only in the function",
                        field32(&region, at::CONTROL_FLAGS) == Some(0x990),
                    ));
                    out.push((
                        "the ring addresses are the rings'",
                        field64(&region, at::FREE_RBD_ADDR) == Some(rings.free.pa())
                            && field64(&region, at::USED_RBD_ADDR) == Some(rings.used.pa())
                            && field64(&region, at::STATUS_WR_PTR) == Some(rings.stat.pa())
                            && field64(&region, at::CMD_QUEUE_ADDR) == Some(rings.cmd.pa()),
                    ));
                    // Three sections' addresses in three different arrays, which
                    // is the join this module exists for: two LMAC, one UMAC, two
                    // paged, and none of them in each other's slots.
                    out.push((
                        "two LMAC addresses landed in the LMAC array",
                        field64(&region, at::LMAC_IMG) != Some(0)
                            && field64(&region, at::LMAC_IMG + 8) != Some(0)
                            && field64(&region, at::LMAC_IMG + 16) == Some(0),
                    ));
                    out.push((
                        "one UMAC address landed in the UMAC array",
                        field64(&region, at::UMAC_IMG) != Some(0)
                            && field64(&region, at::UMAC_IMG + 8) == Some(0),
                    ));
                    out.push((
                        "two paged addresses landed in the paged array",
                        field64(&region, at::VIRTUAL_IMG) != Some(0)
                            && field64(&region, at::VIRTUAL_IMG + 8) != Some(0)
                            && field64(&region, at::VIRTUAL_IMG + 16) == Some(0),
                    ));
                    // No address appears in two arrays, which a shared slot
                    // counter or a transposed `match` arm would produce.
                    out.push((
                        "and no address appears in two arrays",
                        field64(&region, at::LMAC_IMG) != field64(&region, at::UMAC_IMG)
                            && field64(&region, at::UMAC_IMG) != field64(&region, at::VIRTUAL_IMG),
                    ));
                    out.push(("one region per loadable section is returned", sections.len() == 5));
                    // The bytes really got there. A descriptor full of correct
                    // addresses pointing at empty memory is the failure that
                    // looks identical from every field above.
                    out.push((
                        "and the section's bytes were copied into the region the address names",
                        sections.first().map(|d| d.as_slice().first().copied()) == Some(Some(1)),
                    ));
                    out.push((
                        "every reserved field is left zero",
                        field64(&region, at::RESERVED0) == Some(0)
                            && field64(&region, at::RESERVED2) == Some(0)
                            && field64(&region, at::RESERVED3) == Some(0)
                            && field64(&region, at::PLATFORM_NVM_ADDR) == Some(0),
                    ));
                    out.push((
                        "the command queue size byte is 5, in the descriptor",
                        region.as_slice().get(at::CMD_QUEUE_SIZE).copied() == Some(5),
                    ));
                }
                None => out.push(("the descriptor builds", false)),
            }
        }
    }

    out
}
