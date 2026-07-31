//! Intel 8254x gigabit Ethernet.
//!
//! QEMU's default `-device e1000` is an 82540EM, which is the best documented
//! NIC there is -- Intel's software developer manual describes every register
//! used here -- and the same driver covers a good number of real Intel parts.
//!
//! Polled rather than interrupt-driven, to begin with. A ring the card writes
//! into and we read from needs no interrupt to be correct, only to be
//! efficient, and adding an IRQ before the descriptor handling is known-good
//! means debugging two things at once.
//!
//! # Descriptor rings
//!
//! Both directions use a circular array of 16-byte descriptors, each pointing
//! at a buffer, with the card and the driver chasing each other around it. The
//! card owns everything between HEAD and TAIL; we own the rest. For receive we
//! hand it buffers and advance TAIL behind it; for transmit we fill a
//! descriptor and advance TAIL to say so.
//!
//! Two details are easy to get wrong and silent when wrong. The ring's base
//! address must be *physical*, which here is also its virtual address because
//! the map is the identity -- but only because of that. And the ring must be
//! 16-byte aligned with a length that is a multiple of 128 bytes, or the card
//! quietly does nothing.

use super::pci;
use crate::mem::paging;
use alloc::vec;
use alloc::vec::Vec;
use core::ptr::{read_volatile, write_volatile};

// Registers, byte offsets into the MMIO BAR.
const REG_CTRL: u32 = 0x0000;
const REG_STATUS: u32 = 0x0008;
const REG_ICR: u32 = 0x00C0;
const REG_IMC: u32 = 0x00D8;
const REG_RCTL: u32 = 0x0100;
const REG_TCTL: u32 = 0x0400;
const REG_RDBAL: u32 = 0x2800;
const REG_RDBAH: u32 = 0x2804;
const REG_RDLEN: u32 = 0x2808;
const REG_RDH: u32 = 0x2810;
const REG_RDT: u32 = 0x2818;
const REG_TDBAL: u32 = 0x3800;
const REG_TDBAH: u32 = 0x3804;
const REG_TDLEN: u32 = 0x3808;
const REG_TDH: u32 = 0x3810;
const REG_TDT: u32 = 0x3818;
const REG_RAL: u32 = 0x5400;
const REG_RAH: u32 = 0x5404;
const REG_MTA: u32 = 0x5200;

const CTRL_RST: u32 = 1 << 26;
const CTRL_ASDE: u32 = 1 << 5;
const CTRL_SLU: u32 = 1 << 6;

const RCTL_EN: u32 = 1 << 1;
const RCTL_BAM: u32 = 1 << 15; // accept broadcast
const RCTL_SECRC: u32 = 1 << 26; // strip the Ethernet CRC
const RCTL_BSIZE_2048: u32 = 0; // with BSEX clear

const TCTL_EN: u32 = 1 << 1;
const TCTL_PSP: u32 = 1 << 3; // pad short packets

const TXD_CMD_EOP: u8 = 1 << 0;
const TXD_CMD_IFCS: u8 = 1 << 1;
const TXD_CMD_RS: u8 = 1 << 3;
const TXD_STAT_DD: u8 = 1 << 0;
const RXD_STAT_DD: u8 = 1 << 0;
const RXD_STAT_EOP: u8 = 1 << 1;

/// Ring lengths. Must keep the byte size a multiple of 128; 32 descriptors of
/// 16 bytes is 512, which satisfies that with room to spare.
const N_RX: usize = 32;
const N_TX: usize = 32;
const BUF_SIZE: usize = 2048;

#[repr(C)]
#[derive(Clone, Copy)]
struct RxDesc {
    addr: u64,
    length: u16,
    checksum: u16,
    status: u8,
    errors: u8,
    special: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TxDesc {
    addr: u64,
    length: u16,
    cso: u8,
    cmd: u8,
    status: u8,
    css: u8,
    special: u16,
}

pub struct E1000 {
    bar: u64,
    mac: [u8; 6],
    rx: *mut RxDesc,
    tx: *mut TxDesc,
    rx_bufs: Vec<*mut u8>,
    tx_bufs: Vec<*mut u8>,
    rx_cur: usize,
    tx_cur: usize,
}

// One core, and the NIC is only touched from the shell task for now.
unsafe impl Send for E1000 {}
unsafe impl Sync for E1000 {}

impl E1000 {
    #[inline]
    fn write(&self, reg: u32, v: u32) {
        unsafe { write_volatile((self.bar + reg as u64) as *mut u32, v) }
    }

    #[inline]
    fn read(&self, reg: u32) -> u32 {
        unsafe { read_volatile((self.bar + reg as u64) as *const u32) }
    }

    pub fn mac(&self) -> [u8; 6] {
        self.mac
    }

    pub fn link_up(&self) -> bool {
        self.read(REG_STATUS) & (1 << 1) != 0
    }

    /// Send one frame. Returns false if the ring is full.
    pub fn transmit(&mut self, frame: &[u8]) -> bool {
        if frame.len() > BUF_SIZE {
            return false;
        }
        let i = self.tx_cur;
        let d = unsafe { &mut *self.tx.add(i) };

        // The card sets DD when it is done with a descriptor. A descriptor we
        // previously used whose DD is still clear is one the card has not
        // finished, so reusing it would overwrite a frame in flight.
        if d.cmd != 0 && d.status & TXD_STAT_DD == 0 {
            return false;
        }

        unsafe {
            core::ptr::copy_nonoverlapping(frame.as_ptr(), self.tx_bufs[i], frame.len());
        }
        d.addr = self.tx_bufs[i] as u64;
        d.length = frame.len() as u16;
        d.cso = 0;
        // EOP: this descriptor ends the packet. IFCS: let the card append the
        // Ethernet CRC. RS: report status, which is what sets DD.
        d.cmd = TXD_CMD_EOP | TXD_CMD_IFCS | TXD_CMD_RS;
        d.status = 0;

        self.tx_cur = (i + 1) % N_TX;
        self.write(REG_TDT, self.tx_cur as u32);
        true
    }

    /// Take one received frame, if the card has left one.
    pub fn receive(&mut self) -> Option<Vec<u8>> {
        let i = self.rx_cur;
        let d = unsafe { &mut *self.rx.add(i) };
        if d.status & RXD_STAT_DD == 0 {
            return None;
        }
        // A frame split across descriptors would need reassembly. With 2 KiB
        // buffers and a 1500-byte MTU it cannot happen, so this reports rather
        // than silently truncating.
        if d.status & RXD_STAT_EOP == 0 {
            d.status = 0;
            self.advance_rx();
            return None;
        }

        let len = d.length as usize;
        let mut out = vec![0u8; len];
        unsafe {
            core::ptr::copy_nonoverlapping(self.rx_bufs[i], out.as_mut_ptr(), len);
        }

        // Hand the descriptor back before advancing the tail, or the card can
        // fill a buffer we have not finished copying out of.
        d.status = 0;
        self.advance_rx();
        Some(out)
    }

    fn advance_rx(&mut self) {
        // TAIL points at the last descriptor the card may use, so it trails the
        // one we are about to read by one.
        let prev = self.rx_cur;
        self.rx_cur = (prev + 1) % N_RX;
        self.write(REG_RDT, prev as u32);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InitError {
    NotFound,
    NoBar,
    NoMemory,
    NoMac,
}

/// Intel parts this driver is known to match. QEMU's default is 100E.
const SUPPORTED: [u16; 4] = [0x100E, 0x1533, 0x10D3, 0x153A];

pub fn probe(ecam: u64) -> Result<E1000, InitError> {
    let mut found: Option<pci::Device> = None;
    pci::scan(ecam, 255, |d| {
        if d.vendor == 0x8086 && SUPPORTED.contains(&d.device) && found.is_none() {
            found = Some(d);
        }
    });
    let dev = found.ok_or(InitError::NotFound)?;

    let bar = pci::bar(ecam, &dev, 0).ok_or(InitError::NoBar)?;
    if bar == 0 {
        return Err(InitError::NoBar);
    }
    // Same reasoning as NVMe: the BAR is generally outside the boot-time
    // identity map, and it is device memory, so it must be mapped uncacheable
    // before the first register read.
    if !paging::map_range(bar, 0x20000, true) {
        return Err(InitError::NoBar);
    }
    // Without bus-master the card can be configured perfectly and will never
    // DMA anything, which looks exactly like a dead link.
    pci::enable_bus_master(ecam, &dev);

    let rx = super::nvme::alloc_dma(N_RX * core::mem::size_of::<RxDesc>())
        .ok_or(InitError::NoMemory)? as *mut RxDesc;
    let tx = super::nvme::alloc_dma(N_TX * core::mem::size_of::<TxDesc>())
        .ok_or(InitError::NoMemory)? as *mut TxDesc;

    let mut rx_bufs = Vec::new();
    let mut tx_bufs = Vec::new();
    for _ in 0..N_RX {
        rx_bufs.push(super::nvme::alloc_dma(BUF_SIZE).ok_or(InitError::NoMemory)?);
    }
    for _ in 0..N_TX {
        tx_bufs.push(super::nvme::alloc_dma(BUF_SIZE).ok_or(InitError::NoMemory)?);
    }

    let mut nic = E1000 {
        bar,
        mac: [0; 6],
        rx,
        tx,
        rx_bufs,
        tx_bufs,
        rx_cur: 0,
        tx_cur: 0,
    };

    // Reset, then wait for the bit to clear itself.
    nic.write(REG_CTRL, nic.read(REG_CTRL) | CTRL_RST);
    for _ in 0..1000 {
        if nic.read(REG_CTRL) & CTRL_RST == 0 {
            break;
        }
        crate::time::delay_us(100);
    }
    // Mask every interrupt: this driver polls, and an unhandled IRQ line from a
    // card nobody is listening to would fire continuously.
    nic.write(REG_IMC, 0xFFFF_FFFF);
    let _ = nic.read(REG_ICR);

    // Link up, and let the card negotiate speed rather than forcing it.
    nic.write(REG_CTRL, nic.read(REG_CTRL) | CTRL_SLU | CTRL_ASDE);

    // The MAC. Firmware has usually already loaded it into RAL/RAH from the
    // EEPROM, which avoids implementing the EEPROM read protocol for the
    // common case.
    let low = nic.read(REG_RAL);
    let high = nic.read(REG_RAH);
    if low == 0 && high & 0xFFFF == 0 {
        return Err(InitError::NoMac);
    }
    nic.mac = [
        low as u8,
        (low >> 8) as u8,
        (low >> 16) as u8,
        (low >> 24) as u8,
        high as u8,
        (high >> 8) as u8,
    ];

    // Clear the multicast table; leftover entries would accept traffic meant
    // for somebody else.
    for i in 0..128 {
        nic.write(REG_MTA + i * 4, 0);
    }

    // Receive ring.
    for i in 0..N_RX {
        unsafe {
            (*nic.rx.add(i)).addr = nic.rx_bufs[i] as u64;
            (*nic.rx.add(i)).status = 0;
        }
    }
    nic.write(REG_RDBAL, (rx as u64) as u32);
    nic.write(REG_RDBAH, ((rx as u64) >> 32) as u32);
    nic.write(REG_RDLEN, (N_RX * 16) as u32);
    nic.write(REG_RDH, 0);
    // TAIL is the last descriptor the card may write, so it sits one behind
    // HEAD around the ring -- setting it to zero here would tell the card the
    // ring is empty.
    nic.write(REG_RDT, (N_RX - 1) as u32);
    nic.write(REG_RCTL, RCTL_EN | RCTL_BAM | RCTL_SECRC | RCTL_BSIZE_2048);

    // Transmit ring.
    for i in 0..N_TX {
        unsafe {
            (*nic.tx.add(i)).addr = nic.tx_bufs[i] as u64;
            (*nic.tx.add(i)).cmd = 0;
            (*nic.tx.add(i)).status = TXD_STAT_DD;
        }
    }
    nic.write(REG_TDBAL, (tx as u64) as u32);
    nic.write(REG_TDBAH, ((tx as u64) >> 32) as u32);
    nic.write(REG_TDLEN, (N_TX * 16) as u32);
    nic.write(REG_TDH, 0);
    nic.write(REG_TDT, 0);
    nic.write(REG_TCTL, TCTL_EN | TCTL_PSP);

    Ok(nic)
}
