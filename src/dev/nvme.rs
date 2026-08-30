//! NVMe block device driver.
//!
//! Chosen over USB Mass Storage because it is roughly a third of the work: an
//! NVMe controller is a handful of MMIO registers plus a pair of ring buffers
//! in ordinary memory, with no host controller stack, no enumeration, no
//! endpoints and no transfer-descriptor lists underneath it.
//!
//! Three things make this driver possible at all here, all of them consequences
//! of decisions made earlier:
//!
//! * Identity mapping means virtual == physical, so a heap pointer *is* a DMA
//!   address. No IOMMU setup, no translation, no bounce buffers.
//! * The ECAM enumeration from `dev::pci` finds the controller and its BAR.
//! * Non-RAM pages are mapped uncacheable, so doorbell writes actually reach
//!   the device instead of sitting in a write-back cache line.
//!
//! SAFETY POSTURE: writes are refused unless explicitly unlocked. The only
//! NVMe device in this laptop is the internal Kingston, which holds Windows,
//! and a stray LBA would be unrecoverable. Reads are harmless and prove
//! everything except the write path; the write path is exercised against a
//! throwaway QEMU image instead.

#![allow(dead_code)]

use super::pci;
use crate::sync::Racy;
use alloc::alloc::{alloc_zeroed, Layout};
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{AtomicBool, Ordering};

// --- controller registers, offsets from BAR0 ---
const REG_CAP: u64 = 0x00; // 64-bit capabilities
const REG_VS: u64 = 0x08;
const REG_INTMS: u64 = 0x0C;
const REG_INTMC: u64 = 0x10;
const REG_CC: u64 = 0x14; // controller configuration
const REG_CSTS: u64 = 0x1C; // controller status
const REG_AQA: u64 = 0x24; // admin queue attributes
const REG_ASQ: u64 = 0x28; // admin submission queue base
const REG_ACQ: u64 = 0x30; // admin completion queue base

const CC_EN: u32 = 1 << 0;
const CSTS_RDY: u32 = 1 << 0;
const CSTS_CFS: u32 = 1 << 1; // controller fatal status

const ADMIN_QUEUE_LEN: usize = 32;
const IO_QUEUE_LEN: usize = 64;

const OPC_ADMIN_CREATE_SQ: u8 = 0x01;
const OPC_ADMIN_CREATE_CQ: u8 = 0x05;
const OPC_ADMIN_IDENTIFY: u8 = 0x06;
const OPC_IO_WRITE: u8 = 0x01;
const OPC_IO_READ: u8 = 0x02;

/// A submission queue entry. Exactly 64 bytes; the layout is the wire format.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Command {
    opc: u8,
    flags: u8,
    cid: u16,
    nsid: u32,
    _rsvd2: u64,
    mptr: u64,
    prp1: u64,
    prp2: u64,
    cdw10: u32,
    cdw11: u32,
    cdw12: u32,
    cdw13: u32,
    cdw14: u32,
    cdw15: u32,
}

/// A completion queue entry. Exactly 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Completion {
    result: u32,
    _rsvd: u32,
    sq_head: u16,
    sq_id: u16,
    cid: u16,
    status: u16,
}

/// Second PRP entry for a two-page transfer, or 0 if one page suffices.
///
/// Superseded by `Nvme::prp_for`, which handles any size. Kept because it is
/// the readable statement of the rule the list obeys, and because a caller
/// that knows it is under two pages need not touch the shared list page.
#[allow(dead_code)]
///
/// PRP1 may point anywhere within a page, but the controller will only fill to
/// the end of that page; anything beyond needs PRP2. The size of the transfer
/// is therefore not the whole question -- the buffer's *offset within its page*
/// matters just as much. A 512-byte read from an unaligned stack buffer can
/// straddle a boundary, and the symptom is subtle: the leading bytes arrive
/// correctly and the tail is silently left untouched, which reads as corrupt
/// data rather than as a failed transfer.
fn prp2_for(buf: u64, bytes: usize) -> u64 {
    let page_off = buf & 0xFFF;
    let in_first_page = 4096 - page_off;
    if bytes as u64 > in_first_page {
        (buf - page_off) + 4096
    } else {
        0
    }
}

impl Nvme {
    /// PRP2 for a transfer of any size.
    ///
    /// The NVMe rule: PRP1 is the buffer, which may start mid-page. If what is
    /// left runs into a second page, PRP2 is that page. If it runs past two,
    /// PRP2 becomes the address of a *list* of page addresses.
    ///
    /// Building that list is the part that is awkward on a normal OS -- the
    /// pages must be pinned and each virtual address translated to a physical
    /// one. Here the address space is identity-mapped, so a page's address is
    /// already the address the controller wants, and the list is a loop.
    fn prp_for(&mut self, buf: u64, bytes: usize) -> u64 {
        let page_off = buf & 0xFFF;
        let in_first = 4096 - page_off;
        if bytes as u64 <= in_first {
            return 0;
        }
        let rest = bytes as u64 - in_first;
        let second = (buf - page_off) + 4096;
        if rest <= 4096 {
            return second;
        }
        let pages = rest.div_ceil(4096) as usize;
        if self.prp_list.is_null() || pages > 4096 / 8 {
            // No list to build, or more than one page can describe. The caller
            // clamps to `max_transfer_blocks` so this should not be reachable;
            // answering the two-page form keeps it a short read rather than a
            // command that reads into memory nobody owns.
            return second;
        }
        for i in 0..pages {
            unsafe { self.prp_list.add(i).write_volatile(second + (i as u64) * 4096) };
        }
        self.prp_list as u64
    }
}


/// Allocate page-aligned, zeroed DMA memory.
///
/// Page alignment is required for queues and for PRP entries. Because the
/// address space is identity-mapped, the pointer returned is also the physical
/// address the controller will use.
/// Page-aligned buffer for callers doing I/O. Using this avoids the
/// straddling-PRP case entirely rather than relying on it being handled.
pub fn alloc_dma(bytes: usize) -> Option<*mut u8> {
    dma_alloc(bytes)
}

fn dma_alloc(bytes: usize) -> Option<*mut u8> {
    let layout = Layout::from_size_align(bytes, 4096).ok()?;
    let p = unsafe { alloc_zeroed(layout) };
    if p.is_null() {
        None
    } else {
        Some(p)
    }
}

struct Queue {
    sq: *mut Command,
    cq: *mut Completion,
    len: usize,
    sq_tail: usize,
    cq_head: usize,
    /// Flipped each time the completion queue wraps. The controller toggles
    /// the phase bit in each entry it writes, so "is this entry new?" is
    /// answered by comparing against the expected phase -- there is no other
    /// valid flag to test.
    phase: bool,
    id: u16,
}

pub struct Nvme {
    bar: u64,
    doorbell_stride: u64,
    admin: Queue,
    io: Option<Queue>,
    pub max_transfer_blocks: u32,
    /// One page of PRP entries, allocated once and reused by every command.
    ///
    /// Not per-call: `dma_alloc` is `alloc_zeroed` and nothing here frees, so
    /// a list allocated per read would leak a page per read. A single command
    /// is in flight at a time, so one page is all that is ever needed.
    prp_list: *mut u64,
    pub nsid: u32,
    pub block_size: u32,
    pub block_count: u64,
    pub model: [u8; 40],
    pub serial: [u8; 20],
}

impl Nvme {
    #[inline]
    unsafe fn r32(&self, off: u64) -> u32 {
        unsafe { read_volatile((self.bar + off) as *const u32) }
    }
    #[inline]
    unsafe fn w32(&self, off: u64, v: u32) {
        unsafe { write_volatile((self.bar + off) as *mut u32, v) }
    }
    #[inline]
    unsafe fn r64(&self, off: u64) -> u64 {
        unsafe { read_volatile((self.bar + off) as *const u64) }
    }
    #[inline]
    unsafe fn w64(&self, off: u64, v: u64) {
        unsafe { write_volatile((self.bar + off) as *mut u64, v) }
    }

    /// Doorbell register for a queue.
    ///
    /// Layout is `0x1000 + (2*qid + is_cq) * (4 << CAP.DSTRD)`. The stride is
    /// not always 4: CAP.DSTRD exists so controllers can space doorbells onto
    /// separate cache lines, and hardcoding 4 works right up until it silently
    /// does not.
    #[inline]
    fn doorbell(&self, qid: u16, is_cq: bool) -> u64 {
        let idx = 2 * qid as u64 + is_cq as u64;
        0x1000 + idx * (4u64 << self.doorbell_stride)
    }

    /// Submit one command and spin until its completion arrives.
    ///
    /// Polled rather than interrupt-driven. That is deliberate for now: it
    /// keeps the driver usable from any context, including the recovery path,
    /// where the interrupt controller may not be configured.
    fn submit(&mut self, admin: bool, mut cmd: Command) -> Result<u32, u16> {
        let q = if admin {
            &mut self.admin
        } else {
            self.io.as_mut().ok_or(0xFFFFu16)?
        };

        let cid = q.sq_tail as u16;
        cmd.cid = cid;
        unsafe { write_volatile(q.sq.add(q.sq_tail), cmd) };

        q.sq_tail = (q.sq_tail + 1) % q.len;
        let (qid, tail) = (q.id, q.sq_tail as u32);
        let db = self.doorbell(qid, false);
        // Stamped across the doorbell so the delta below is the controller's
        // own latency and not this function's bookkeeping.
        let issued = crate::time::rdtsc();
        unsafe { self.w32(db, tail) };

        // Poll the completion queue for an entry whose phase bit differs from
        // the last pass.
        let q = if admin {
            &mut self.admin
        } else {
            self.io.as_mut().unwrap()
        };
        let mut spins: u64 = 0;
        loop {
            let c = unsafe { read_volatile(q.cq.add(q.cq_head)) };
            let p = (c.status & 1) != 0;
            if p == q.phase {
                let status = c.status >> 1;
                q.cq_head += 1;
                if q.cq_head == q.len {
                    q.cq_head = 0;
                    q.phase = !q.phase;
                }
                let (qid, head) = (q.id, q.cq_head as u32);
                let db = self.doorbell(qid, true);
                unsafe { self.w32(db, head) };

                // How long the controller took, into the entropy pool.
                //
                // NAND timing, internal scheduling and wear levelling all
                // land in this number, and none of them is predictable from
                // outside the machine. It matters because the other source is
                // keyboard and mouse: a machine left running overnight sees
                // none of those, and this is the one thing that still arrives
                // while nobody is present.
                //
                // Only on a real completion. The timeout path below is two
                // hundred million spins and says more about this loop than
                // about the device.
                crate::rng::add_device_entropy(
                    crate::time::rdtsc().wrapping_sub(issued),
                );
                return if status == 0 { Ok(c.result) } else { Err(status) };
            }

            spins += 1;
            if spins > 200_000_000 {
                return Err(0xFFFE); // timeout
            }
            core::hint::spin_loop();
        }
    }

    fn identify(&mut self, cns: u32, nsid: u32, buf: *mut u8) -> Result<(), u16> {
        let cmd = Command {
            opc: OPC_ADMIN_IDENTIFY,
            nsid,
            prp1: buf as u64,
            cdw10: cns,
            ..Default::default()
        };
        self.submit(true, cmd).map(|_| ())
    }

    /// Read `count` blocks starting at `lba` into `buf`.
    ///
    /// Uses PRP1 plus, when the transfer crosses a page, PRP2 as a second
    /// page pointer. That caps a single command at two pages here; larger
    /// transfers would need a PRP list, which is why callers chunk.
    pub fn read(&mut self, lba: u64, count: u16, buf: *mut u8) -> Result<(), u16> {
        let bytes = count as usize * self.block_size as usize;
        if count as u32 > self.max_transfer_blocks {
            return Err(0xFFFD);
        }
        let prp2 = self.prp_for(buf as u64, bytes);
        let cmd = Command {
            opc: OPC_IO_READ,
            nsid: self.nsid,
            prp1: buf as u64,
            prp2,
            cdw10: lba as u32,
            cdw11: (lba >> 32) as u32,
            cdw12: (count as u32).saturating_sub(1), // zero-based count
            ..Default::default()
        };
        self.submit(false, cmd).map(|_| ())
    }

    /// Write `count` blocks. Refused unless writes have been unlocked.
    ///
    /// The gate is not ceremony. The only NVMe device in this machine holds
    /// Windows, and there is no undo for a misplaced LBA.
    pub fn write(&mut self, lba: u64, count: u16, buf: *const u8) -> Result<(), u16> {
        if !writes_unlocked() {
            return Err(0xFFFC);
        }
        // Outside the claimed region is refused, and this is the check the
        // gate was missing. `store` unlocks for its own region; without this
        // the unlock was a licence to write the partition table.
        if !may_write(lba, count as u32) {
            return Err(0xFFFB);
        }
        let bytes = count as usize * self.block_size as usize;
        if count as u32 > self.max_transfer_blocks {
            return Err(0xFFFD);
        }
        let prp2 = self.prp_for(buf as u64, bytes);
        let cmd = Command {
            opc: OPC_IO_WRITE,
            nsid: self.nsid,
            prp1: buf as u64,
            prp2,
            cdw10: lba as u32,
            cdw11: (lba >> 32) as u32,
            cdw12: (count as u32).saturating_sub(1),
            ..Default::default()
        };
        self.submit(false, cmd).map(|_| ())
    }

    pub fn capacity_bytes(&self) -> u64 {
        self.block_count * self.block_size as u64
    }
}

static WRITES: AtomicBool = AtomicBool::new(false);

/// The only blocks a write may touch, as `[start, end)`.
///
/// **The lock used to be a bit, and a bit is the wrong shape.** Unlocking said
/// "writes are allowed" and nothing said *where*, so from the moment
/// `store::init` succeeded, every LBA on the device was writable -- the
/// partition table at zero, the EFI system partition, and the Windows volume
/// that is still the only other thing on this disk. The gate's own doc comment
/// says there is no undo for a misplaced LBA, and then the gate did not check
/// the LBA.
///
/// It is a window now, and every caller that unlocks has to name one. That is
/// not extra ceremony: both callers already had the region in hand from
/// `find_store_region`, and a caller that cannot say which blocks it means is
/// exactly the caller that should not be writing.
static WIN_START: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static WIN_END: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

pub fn writes_unlocked() -> bool {
    WRITES.load(Ordering::Relaxed)
}

/// The window writes are confined to, or `None` when locked.
pub fn write_window() -> Option<(u64, u64)> {
    if !writes_unlocked() {
        return None;
    }
    Some((WIN_START.load(Ordering::Relaxed), WIN_END.load(Ordering::Relaxed)))
}

/// Would this write be allowed?
///
/// Separated from `write` so the gate can be checked without a device and
/// without writing anything -- the property is arithmetic, and a safety gate
/// nobody can test is a safety gate nobody has tested.
pub fn may_write(lba: u64, count: u32) -> bool {
    let Some((start, end)) = write_window() else { return false };
    let Some(last) = lba.checked_add(count as u64) else { return false };
    // An empty window admits nothing, which is the right answer for a caller
    // that unlocked with a zero-length region.
    start < end && lba >= start && last <= end
}

/// Deliberately explicit, and deliberately not wired to a shell command that
/// could be typed by accident.
///
/// `start` and `blocks` are the region the caller is claiming. Everything
/// outside it stays refused whether or not this succeeds.
pub fn unlock_writes(confirm: u64, start: u64, blocks: u64) -> bool {
    if confirm != 0xD15EA5E {
        return false;
    }
    let Some(end) = start.checked_add(blocks) else { return false };
    WIN_START.store(start, Ordering::Relaxed);
    WIN_END.store(end, Ordering::Relaxed);
    // The window before the bit: a reader that sees writes unlocked must never
    // see the previous caller's window still standing.
    WRITES.store(true, Ordering::Relaxed);
    true
}

/// Put the lock back.
///
/// Needed because unlocking was one-way. `store::init` unlocked writes and then
/// formatted, so a region that failed its safety check returned an error while
/// leaving the disk writable -- the failure of a guard made the system less
/// safe than not having tried. Anything that unlocks speculatively must be able
/// to undo it.
pub fn lock_writes() {
    WRITES.store(false, Ordering::Relaxed);
    // The window goes too. A stale one left behind would be a window somebody
    // could re-open into by unlocking with a bad confirmation and getting the
    // early return -- which does not set it, and so would inherit this.
    WIN_START.store(0, Ordering::Relaxed);
    WIN_END.store(0, Ordering::Relaxed);
}

/// The gate, checked without a device and without writing anything.
///
/// Every claim here is about `may_write`, which is the whole of the decision.
/// It restores whatever window was in force, because this runs at boot on a
/// machine that may already have mounted a store.
pub fn gate_selftest() -> bool {
    use crate::kprintln;
    let saved = write_window();
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };

    lock_writes();
    claim("locked, nothing may be written", !may_write(0, 1) && !may_write(1000, 1));

    unlock_writes(0xD15EA5E, 2048, 512);
    claim("unlocked, the claimed region may be written", may_write(2048, 512));
    claim("and the first block outside it may not", !may_write(2560, 1));
    claim("nor the one before it", !may_write(2047, 1));
    // The case that matters most on this machine: LBA 0 is the partition
    // table, and before the window existed an unlocked store made it writable.
    claim("nor the partition table", !may_write(0, 1));
    // A write that starts inside and runs out the far end is the shape that
    // gets a bounds check wrong.
    claim("nor one that starts inside and overruns", !may_write(2560 - 4, 8));
    // Length is attacker-controlled in the sense that matters: a caller with
    // an arithmetic bug should not wrap into a pass.
    claim("nor one whose length overflows", !may_write(u64::MAX - 1, 8));

    unlock_writes(0xD15EA5E, 2048, 0);
    claim("an empty claim admits nothing", !may_write(2048, 1));

    claim("a wrong confirmation does not unlock", !unlock_writes(1, 0, u64::MAX));

    lock_writes();
    if let Some((start, end)) = saved {
        unlock_writes(0xD15EA5E, start, end.saturating_sub(start));
    }
    ok
}

static CONTROLLER: Racy<Option<Nvme>> = Racy::new(None);

pub fn with<R>(f: impl FnOnce(&mut Nvme) -> R) -> Option<R> {
    unsafe { CONTROLLER.get().as_mut().map(f) }
}

pub fn present() -> bool {
    unsafe { CONTROLLER.get().is_some() }
}

#[derive(Debug)]
pub enum InitError {
    NoEcam,
    NoController,
    NoBar,
    NotReady,
    Alloc,
    Admin(u16),
    NoNamespace,
}

/// Find and bring up the first NVMe controller.
pub fn init(ecam: u64) -> Result<(), InitError> {
    // Class 01h subclass 08h: NVM Express.
    let mut found: Option<pci::Device> = None;
    pci::scan(ecam, 255, |d| {
        if d.class == 0x01 && d.subclass == 0x08 && found.is_none() {
            found = Some(d);
        }
    });
    let dev = found.ok_or(InitError::NoController)?;

    let bar = pci::bar(ecam, &dev, 0).ok_or(InitError::NoBar)?;
    if bar == 0 {
        return Err(InitError::NoBar);
    }
    // The BAR is very likely outside the boot-time identity map: 64-bit BARs
    // land wherever the firmware's PCI window is, which on QEMU is 768 GiB and
    // on this laptop is well above RAM. Map it uncacheable before the first
    // register read, or that read faults.
    if !crate::mem::paging::map_range(bar, 0x2000, true) {
        return Err(InitError::NoBar);
    }
    // Without bus-master the controller can accept commands and never DMA a
    // completion back, which is indistinguishable from a hang.
    pci::enable_bus_master(ecam, &dev);

    let mut n = Nvme {
        bar,
        doorbell_stride: 0,
        admin: Queue {
            sq: core::ptr::null_mut(),
            cq: core::ptr::null_mut(),
            len: ADMIN_QUEUE_LEN,
            sq_tail: 0,
            cq_head: 0,
            phase: true,
            id: 0,
        },
        io: None,
        max_transfer_blocks: 16,
        prp_list: core::ptr::null_mut(),
        nsid: 1,
        block_size: 512,
        block_count: 0,
        model: [0; 40],
        serial: [0; 20],
    };

    let cap = unsafe { n.r64(REG_CAP) };
    n.doorbell_stride = (cap >> 32) & 0xF;

    // Reset: a controller left enabled by firmware must be disabled and
    // observed to go not-ready before its admin queues can be repointed.
    unsafe {
        let cc = n.r32(REG_CC);
        n.w32(REG_CC, cc & !CC_EN);
        let mut spins = 0u64;
        while n.r32(REG_CSTS) & CSTS_RDY != 0 {
            spins += 1;
            if spins > 50_000_000 {
                return Err(InitError::NotReady);
            }
            core::hint::spin_loop();
        }
    }

    let asq = dma_alloc(ADMIN_QUEUE_LEN * 64).ok_or(InitError::Alloc)?;
    let acq = dma_alloc(ADMIN_QUEUE_LEN * 16).ok_or(InitError::Alloc)?;
    n.admin.sq = asq as *mut Command;
    n.admin.cq = acq as *mut Completion;

    unsafe {
        // AQA holds zero-based sizes in two 12-bit fields.
        let aqa = ((ADMIN_QUEUE_LEN as u32 - 1) << 16) | (ADMIN_QUEUE_LEN as u32 - 1);
        n.w32(REG_AQA, aqa);
        n.w64(REG_ASQ, asq as u64);
        n.w64(REG_ACQ, acq as u64);

        // Enable with 4 KiB pages, NVM command set, 64-byte SQ / 16-byte CQ
        // entries encoded as their base-2 logarithms.
        let cc: u32 = CC_EN | (0 << 4) | (0 << 7) | (6 << 16) | (4 << 20);
        n.w32(REG_CC, cc);

        let mut spins = 0u64;
        loop {
            let s = n.r32(REG_CSTS);
            if s & CSTS_CFS != 0 {
                return Err(InitError::NotReady);
            }
            if s & CSTS_RDY != 0 {
                break;
            }
            spins += 1;
            if spins > 50_000_000 {
                return Err(InitError::NotReady);
            }
            core::hint::spin_loop();
        }
    }

    // Identify controller: model and serial live at fixed offsets.
    let idbuf = dma_alloc(4096).ok_or(InitError::Alloc)?;
    let cap_for_mps = unsafe { n.r64(REG_CAP) };
    n.identify(1, 0, idbuf).map_err(InitError::Admin)?;
    unsafe {
        core::ptr::copy_nonoverlapping(idbuf.add(4), n.serial.as_mut_ptr(), 20);
        core::ptr::copy_nonoverlapping(idbuf.add(24), n.model.as_mut_ptr(), 40);

        // MDTS: the largest transfer the controller will accept, as a power of
        // two multiple of the minimum page size. It was never read, and the
        // field that holds it was never used -- every transfer went through a
        // hardcoded two-page cap instead, which is 16 blocks. Streaming a
        // model's worth of weights at 8 KiB a command is tens of thousands of
        // round trips for work a handful could do.
        //
        // Zero means "no limit", which is not an invitation: the PRP list
        // below is one page, so the ceiling is what it can describe.
        let mdts = read_volatile(idbuf.add(77));
        let mps_min = 12 + ((cap_for_mps >> 48) & 0xF) as u32; // CAP.MPSMIN
        let by_mdts = if mdts == 0 || mdts > 20 {
            u32::MAX
        } else {
            (1u32 << mdts) << mps_min.saturating_sub(9)
        };
        // One page of 8-byte entries, each describing a 4 KiB page.
        let by_list = (4096 / 8) * (4096 / 512);
        n.max_transfer_blocks = by_mdts.min(by_list).max(16);
    }
    n.prp_list = dma_alloc(4096).ok_or(InitError::Alloc)? as *mut u64;

    // Create the I/O completion queue before the submission queue: the SQ
    // creation names the CQ it reports into, so the CQ has to exist first.
    let iocq = dma_alloc(IO_QUEUE_LEN * 16).ok_or(InitError::Alloc)?;
    let iosq = dma_alloc(IO_QUEUE_LEN * 64).ok_or(InitError::Alloc)?;

    let create_cq = Command {
        opc: OPC_ADMIN_CREATE_CQ,
        prp1: iocq as u64,
        cdw10: ((IO_QUEUE_LEN as u32 - 1) << 16) | 1, // size-1, qid 1
        cdw11: 1,                                     // physically contiguous
        ..Default::default()
    };
    n.submit(true, create_cq).map_err(InitError::Admin)?;

    let create_sq = Command {
        opc: OPC_ADMIN_CREATE_SQ,
        prp1: iosq as u64,
        cdw10: ((IO_QUEUE_LEN as u32 - 1) << 16) | 1,
        cdw11: (1 << 16) | 1, // reports into CQ 1, physically contiguous
        ..Default::default()
    };
    n.submit(true, create_sq).map_err(InitError::Admin)?;

    n.io = Some(Queue {
        sq: iosq as *mut Command,
        cq: iocq as *mut Completion,
        len: IO_QUEUE_LEN,
        sq_tail: 0,
        cq_head: 0,
        phase: true,
        id: 1,
    });

    // Identify namespace 1 for its size and block format.
    n.identify(0, 1, idbuf).map_err(InitError::Admin)?;
    unsafe {
        n.block_count = read_volatile(idbuf as *const u64); // NSZE
        let flbas = read_volatile(idbuf.add(26)) & 0x0F;
        // LBA format table starts at byte 128, 4 bytes per entry; byte 2 of
        // each entry is the block size as a power of two.
        let lbaf = idbuf.add(128 + flbas as usize * 4);
        let lbads = read_volatile(lbaf.add(2));
        n.block_size = if lbads >= 9 && lbads <= 16 {
            1u32 << lbads
        } else {
            512
        };
    }
    if n.block_count == 0 {
        return Err(InitError::NoNamespace);
    }

    unsafe { *CONTROLLER.get() = Some(n) };
    Ok(())
}
