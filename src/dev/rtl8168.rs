//! Realtek RTL8168/8111 gigabit Ethernet.
//!
//! The wired NIC in the GF63, confirmed as `10ec:8168` at PCI 04:00.0. This is
//! the tractable half of "networking on real hardware": descriptor rings much
//! like the e1000 next door, no firmware blob, and a register set that has
//! been stable across a decade of parts.
//!
//! It sits behind `net::iface::Nic`, so nothing above Ethernet knows or cares
//! which card is underneath.
//!
//! ### Where this differs from the e1000, and why it matters
//!
//! **Descriptors are owned by a bit, not by a tail pointer.** The e1000 has
//! head and tail registers and the driver hands descriptors over by moving the
//! tail. Realtek instead puts an OWN bit in each descriptor: set means the
//! card owns it, clear means we do. There is no tail to advance, and the ring
//! wraps because the *last* descriptor carries an EOR (end of ring) bit rather
//! than because a register says how long the ring is. Forgetting EOR gives a
//! card that walks off the end of the ring into whatever follows it.
//!
//! **Transmit needs a poke.** Filling a descriptor is not enough; the card
//! only looks when `TPPOLL` is written. The e1000 notices a tail write on its
//! own.
//!
//! **The config registers are locked.** `CFG9346` has to be set to 0xC0 before
//! several registers accept writes, and returned to 0x00 afterwards. Writes
//! made while locked are silently dropped, which presents as a card that
//! initialises cleanly and does nothing.
//!
//! **Rings must be 256-byte aligned.** The low bits of the descriptor base
//! registers are not implemented, so a misaligned ring is silently rounded
//! down and the card reads descriptors that were never written.

use super::pci;
use crate::net::iface::{Kind, Nic};
use alloc::vec::Vec;

const VENDOR_REALTEK: u16 = 0x10EC;
/// The RTL8168/8111 family. 0x8161 and 0x8136 (RTL810x Fast Ethernet) use the
/// same register layout, so they are accepted too -- a machine that has one
/// instead is better served by a driver that tries than by one that refuses.
const DEVICES: [u16; 4] = [0x8168, 0x8161, 0x8167, 0x8136];

// --- registers, offsets from the MMIO base -------------------------------

const REG_IDR0: u32 = 0x00; // MAC address, 6 bytes
const REG_TNPDS: u32 = 0x20; // transmit descriptor ring, 64-bit
const REG_CR: u32 = 0x37; // command
const REG_TPPOLL: u32 = 0x38; // transmit poll
const REG_IMR: u32 = 0x3C; // interrupt mask
const REG_ISR: u32 = 0x3E; // interrupt status
const REG_TCR: u32 = 0x40; // transmit configuration
const REG_RCR: u32 = 0x44; // receive configuration
const REG_CFG9346: u32 = 0x50; // register write lock
const REG_PHYSTATUS: u32 = 0x6C;
const REG_RDSAR: u32 = 0xE4; // receive descriptor ring, 64-bit
const REG_CPCR: u32 = 0xE0; // C+ command
const REG_MTPS: u32 = 0xEC; // max transmit packet size
const REG_RMS: u32 = 0xDA; // receive packet max size

const CR_RST: u8 = 0x10;
const CR_RE: u8 = 0x08;
const CR_TE: u8 = 0x04;

const CFG9346_UNLOCK: u8 = 0xC0;
const CFG9346_LOCK: u8 = 0x00;

const TPPOLL_NPQ: u8 = 0x40; // normal priority queue has work

/// Accept broadcast, multicast, and frames addressed to us. Deliberately not
/// promiscuous: everything above expects to be handed its own traffic, and
/// accepting the rest would only cost time in `poll`.
const RCR_ACCEPT: u32 = 0x0000_000E;
/// Unlimited receive DMA burst, and no rx FIFO threshold.
const RCR_DMA: u32 = 0x0000_7700;

const TCR_DMA_UNLIMITED: u32 = 0x0700_0000;
/// Standard 96-bit inter-frame gap, which is what every switch expects.
const TCR_IFG: u32 = 0x0300_0000;

const CPCR_RX_CHKSUM: u16 = 1 << 5;
const CPCR_PCI_MULRW: u16 = 1 << 3;

const PHYSTATUS_LINKOK: u8 = 1 << 1;

const OWN: u32 = 0x8000_0000;
const EOR: u32 = 0x4000_0000;
const FS: u32 = 0x2000_0000;
const LS: u32 = 0x1000_0000;

const N_RX: usize = 32;
const N_TX: usize = 32;
/// Room for a full frame plus the 4-byte CRC the card appends on receive.
const BUF_SIZE: usize = 2048;

/// 16 bytes, and the layout is fixed by the hardware.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct Desc {
    /// OWN, EOR, FS, LS in the high bits; frame length in the low 14.
    flags: u32,
    vlan: u32,
    addr_lo: u32,
    addr_hi: u32,
}

pub struct Rtl8168 {
    mmio: u64,
    mac: [u8; 6],
    rx: *mut Desc,
    tx: *mut Desc,
    rx_bufs: Vec<*mut u8>,
    tx_bufs: Vec<*mut u8>,
    /// Next descriptor we expect the card to have filled. Realtek has no head
    /// register to read, so the driver tracks its own position and trusts the
    /// OWN bit to say whether it has caught up.
    rx_cur: usize,
    tx_cur: usize,
}

#[derive(Debug)]
pub enum InitError {
    NotFound,
    NoBar,
    NoMemory,
    NoMac,
    ResetTimeout,
}

impl Rtl8168 {
    #[inline]
    fn r8(&self, off: u32) -> u8 {
        unsafe { core::ptr::read_volatile((self.mmio + off as u64) as *const u8) }
    }
    #[inline]
    fn w8(&self, off: u32, v: u8) {
        unsafe { core::ptr::write_volatile((self.mmio + off as u64) as *mut u8, v) }
    }
    #[inline]
    fn w16(&self, off: u32, v: u16) {
        unsafe { core::ptr::write_volatile((self.mmio + off as u64) as *mut u16, v) }
    }
    #[inline]
    fn r32(&self, off: u32) -> u32 {
        unsafe { core::ptr::read_volatile((self.mmio + off as u64) as *const u32) }
    }
    #[inline]
    fn w32(&self, off: u32, v: u32) {
        unsafe { core::ptr::write_volatile((self.mmio + off as u64) as *mut u32, v) }
    }
    fn w64(&self, off: u32, v: u64) {
        // Two 32-bit halves, low first. The card latches on the high write, so
        // the order is not optional.
        self.w32(off, v as u32);
        self.w32(off + 4, (v >> 32) as u32);
    }

    pub fn mac(&self) -> [u8; 6] {
        self.mac
    }

    pub fn link_up(&self) -> bool {
        self.r8(REG_PHYSTATUS) & PHYSTATUS_LINKOK != 0
    }

    pub fn transmit(&mut self, frame: &[u8]) -> bool {
        if frame.len() > BUF_SIZE {
            return false;
        }
        let i = self.tx_cur;
        let d = unsafe { &mut *self.tx.add(i) };
        // Still owned by the card: the ring is full and the oldest frame has
        // not gone out yet.
        if unsafe { core::ptr::read_volatile(&d.flags) } & OWN != 0 {
            return false;
        }

        unsafe {
            core::ptr::copy_nonoverlapping(
                frame.as_ptr(),
                self.tx_bufs[i],
                frame.len(),
            );
        }

        // The card pads to the 60-byte minimum itself, so a short frame needs
        // no help here.
        let mut flags = OWN | FS | LS | (frame.len() as u32 & 0x3FFF);
        if i == N_TX - 1 {
            flags |= EOR;
        }
        unsafe { core::ptr::write_volatile(&mut d.flags, flags) };

        // Descriptor first, then the poke. Reversed, the card can look before
        // the descriptor is visible and find a stale OWN bit.
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
        self.w8(REG_TPPOLL, TPPOLL_NPQ);

        self.tx_cur = (i + 1) % N_TX;
        true
    }

    pub fn receive(&mut self) -> Option<Vec<u8>> {
        let i = self.rx_cur;
        let d = unsafe { &mut *self.rx.add(i) };
        let flags = unsafe { core::ptr::read_volatile(&d.flags) };
        if flags & OWN != 0 {
            return None; // still the card's
        }

        // The low 14 bits are the length *including* the 4-byte CRC the card
        // leaves on the end. Handing that upward would make every IPv4 packet
        // four bytes too long, which the length check in `net::poll` would
        // then reject as malformed.
        let len = (flags & 0x3FFF) as usize;
        let out = if len > 4 {
            let n = len - 4;
            let mut v = alloc::vec![0u8; n];
            unsafe {
                core::ptr::copy_nonoverlapping(self.rx_bufs[i] as *const u8, v.as_mut_ptr(), n)
            };
            Some(v)
        } else {
            None
        };

        // Hand the descriptor back. EOR has to be rewritten with it -- the
        // whole flags word was overwritten by the card.
        let mut give = OWN | (BUF_SIZE as u32 & 0x3FFF);
        if i == N_RX - 1 {
            give |= EOR;
        }
        unsafe { core::ptr::write_volatile(&mut d.flags, give) };

        self.rx_cur = (i + 1) % N_RX;
        out
    }
}

// Single core, one owner, and the pointers address identity-mapped DMA memory.
unsafe impl Send for Rtl8168 {}
unsafe impl Sync for Rtl8168 {}

impl Nic for Rtl8168 {
    fn mac(&self) -> [u8; 6] {
        Rtl8168::mac(self)
    }
    fn link_up(&mut self) -> bool {
        Rtl8168::link_up(self)
    }
    fn transmit(&mut self, frame: &[u8]) -> bool {
        Rtl8168::transmit(self, frame)
    }
    fn receive(&mut self) -> Option<Vec<u8>> {
        Rtl8168::receive(self)
    }
    fn kind(&self) -> Kind {
        Kind::Ethernet
    }
}

pub fn probe(ecam: u64) -> Result<Rtl8168, InitError> {
    let mut found = None;
    pci::scan(ecam, 255, |d| {
        if d.vendor == VENDOR_REALTEK && DEVICES.contains(&d.device) && found.is_none() {
            found = Some(d);
        }
    });
    let dev = found.ok_or(InitError::NotFound)?;

    // BAR2 is the memory window on every part in this family. BAR0 is a
    // legacy I/O port range and BAR1 is its upper half; using either would
    // mean port I/O for every register access, which the descriptor rings
    // cannot work through anyway.
    let bar = pci::bar(ecam, &dev, 2).ok_or(InitError::NoBar)?;
    if bar == 0 {
        return Err(InitError::NoBar);
    }
    pci::enable_bus_master(ecam, &dev);

    let rx = super::nvme::alloc_dma(N_RX * core::mem::size_of::<Desc>())
        .ok_or(InitError::NoMemory)? as *mut Desc;
    let tx = super::nvme::alloc_dma(N_TX * core::mem::size_of::<Desc>())
        .ok_or(InitError::NoMemory)? as *mut Desc;
    // The base registers ignore their low 8 bits, so a ring that is not
    // 256-byte aligned is silently rounded down and the card reads
    // descriptors nobody wrote.
    if (rx as u64) & 0xFF != 0 || (tx as u64) & 0xFF != 0 {
        return Err(InitError::NoMemory);
    }

    let mut rx_bufs = Vec::with_capacity(N_RX);
    let mut tx_bufs = Vec::with_capacity(N_TX);
    for _ in 0..N_RX {
        rx_bufs.push(super::nvme::alloc_dma(BUF_SIZE).ok_or(InitError::NoMemory)?);
    }
    for _ in 0..N_TX {
        tx_bufs.push(super::nvme::alloc_dma(BUF_SIZE).ok_or(InitError::NoMemory)?);
    }

    let mut nic = Rtl8168 {
        mmio: bar,
        mac: [0; 6],
        rx,
        tx,
        rx_bufs,
        tx_bufs,
        rx_cur: 0,
        tx_cur: 0,
    };

    // --- reset ---
    nic.w8(REG_CR, CR_RST);
    let deadline = crate::dev::lapic::ticks() + crate::TIMER_HZ as u64;
    while nic.r8(REG_CR) & CR_RST != 0 {
        if crate::dev::lapic::ticks() > deadline {
            return Err(InitError::ResetTimeout);
        }
        core::hint::spin_loop();
    }

    // --- MAC address, straight out of IDR0 ---
    for i in 0..6 {
        nic.mac[i] = nic.r8(REG_IDR0 + i as u32);
    }
    if nic.mac.iter().all(|b| *b == 0) || nic.mac.iter().all(|b| *b == 0xFF) {
        return Err(InitError::NoMac);
    }

    // --- rings ---
    for i in 0..N_RX {
        let d = unsafe { &mut *rx.add(i) };
        d.vlan = 0;
        let a = nic.rx_bufs[i] as u64;
        d.addr_lo = a as u32;
        d.addr_hi = (a >> 32) as u32;
        d.flags = OWN | (BUF_SIZE as u32 & 0x3FFF) | if i == N_RX - 1 { EOR } else { 0 };
    }
    for i in 0..N_TX {
        let d = unsafe { &mut *tx.add(i) };
        d.vlan = 0;
        let a = nic.tx_bufs[i] as u64;
        d.addr_lo = a as u32;
        d.addr_hi = (a >> 32) as u32;
        d.flags = if i == N_TX - 1 { EOR } else { 0 };
    }

    // --- configure, with the lock open ---
    nic.w8(REG_CFG9346, CFG9346_UNLOCK);

    // C+ mode has to be set before the rings are handed over.
    nic.w16(REG_CPCR, CPCR_RX_CHKSUM | CPCR_PCI_MULRW);
    nic.w64(REG_RDSAR, rx as u64);
    nic.w64(REG_TNPDS, tx as u64);

    nic.w32(REG_TCR, TCR_DMA_UNLIMITED | TCR_IFG);
    nic.w32(REG_RCR, RCR_ACCEPT | RCR_DMA);
    nic.w16(REG_RMS, BUF_SIZE as u16);
    nic.w8(REG_MTPS, 0x3B); // 0x3B * 128 bytes, comfortably over one frame

    // Interrupts stay masked: everything above polls, exactly as with the
    // e1000, and a NIC that raises an interrupt nobody handles is worse than
    // one that stays quiet.
    nic.w16(REG_IMR, 0);
    nic.w16(REG_ISR, 0xFFFF); // clear anything latched during reset

    nic.w8(REG_CR, CR_RE | CR_TE);
    nic.w8(REG_CFG9346, CFG9346_LOCK);

    Ok(nic)
}
