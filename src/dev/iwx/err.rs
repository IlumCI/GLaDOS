//! What a dead firmware left behind: its error tables.
//!
//! A command the part does not answer is one of two things, and the journal
//! could not tell them apart: a firmware that is slow, and a firmware that has
//! **asserted** and will never answer anything again. On the laptop the second
//! read as `MAC_CONTEXT_CMD add failed: the part did not answer` followed by a
//! `PHY_CONTEXT_CMD remove` that did not answer either -- two silences in a
//! row from a queue that had just answered twice -- and nothing said which
//! field the firmware had objected to.
//!
//! The firmware says, in two places. `CSR_INT` bit 25 is raised when the
//! microcode faults, and the ALIVE notification handed over the addresses of
//! three error tables in the part's own SRAM, "carried because a firmware that
//! dies later is diagnosed from exactly these and they are only ever offered
//! once". This module reads them, through the same indirect window
//! `iwx_read_mem` uses: latch an address at `HBUS_TARG_MEM_RADDR`, read words
//! out of `HBUS_TARG_MEM_RDAT`, which auto-increments. Under the MAC access
//! lock, since the SRAM is behind it.
//!
//! The layouts are `iwl_error_event_table` (LMAC, `LOG_ERROR_TABLE_API_S_VER_3`)
//! and `iwl_umac_error_event_table`, from Intel's headers under the BSD arm
//! (`NOTICE.md`). Only the head of each is read: `valid`, `error_id`, the
//! branch and interrupt links, the three data words, and -- the field that
//! answers the question -- the **last host command header the firmware
//! handled**, which is the command it was parsing when it died.

use alloc::string::String;
use alloc::vec::Vec;

use super::gen3;

const HBUS_TARG_MEM_RADDR: u64 = 0x400 + 0x00c;
const HBUS_TARG_MEM_RDAT: u64 = 0x400 + 0x01c;

/// `CSR_INT` and the two bits that mean the firmware is gone.
const CSR_INT: u64 = 0x008;
pub const CSR_INT_BIT_SW_ERR: u32 = 1 << 25;
pub const CSR_INT_BIT_HW_ERR: u32 = 1 << 29;

/// Words read from the LMAC table. The header has 38; `hcmd` is word 23 and
/// `last_cmd_id` word 29, so 30 covers everything printed.
pub const LMAC_WORDS: usize = 30;
/// The whole UMAC table is 15 words.
pub const UMAC_WORDS: usize = 15;

/// Error ids, `iwx_desc_lookup`'s table. Only the ones a host command can
/// provoke are named; anything else prints as its number.
pub fn error_name(id: u32) -> &'static str {
    match id {
        0x0000 => "no error",
        0x0066 => "NMI_INTERRUPT_HOST",
        0x1000 => "SYSASSERT",
        0x2000 => "ADVANCED_SYSASSERT",
        0x3000 => "NMI_INTERRUPT_UNKNOWN",
        0x3453 => "LOG_FLOW_FAIL",
        0x5000 => "MAC_ASSERT",
        0x6000 => "UNKNOWN_TRM",
        0x7000 => "NMI_INTERRUPT_HOST_SIMPLE",
        0x8000 => "NMI_INTERRUPT_ACTION_PT",
        0x9000 => "DBG_ASSERT",
        0xA000 => "NMI_INTERRUPT_DATA_ACTION_PT",
        0xC000 => "NMI_INTERRUPT_WDG",
        0xE000 => "NMI_INTERRUPT_BREAK_POINT",
        0xF000 => "NMI_TRM_HW_ERR",
        0x2e4d => "PCI_FATAL",
        _ => "?",
    }
}

/// One table, decoded. Both layouts open `valid, error_id`, then links and
/// data words at positions that differ, so each is parsed by its own function
/// and the printed shape is one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Entry {
    pub valid: u32,
    pub error_id: u32,
    pub ilink1: u32,
    pub ilink2: u32,
    pub data1: u32,
    pub data2: u32,
    pub data3: u32,
    /// The last host command header the firmware handled, low byte the
    /// opcode, next the group, then the sequence.
    pub cmd_header: u32,
    /// Where the firmware was executing when it logged this: the LMAC's
    /// `log_pc`, the UMAC's `frame_pointer`. Not decodable without Intel's
    /// symbols, but it is the field that pins an assert to one place, so a
    /// repeat with a byte-identical command is told apart from a new cause.
    pub pc: u32,
}

impl Entry {
    pub fn render(&self, which: &str) -> String {
        let (op, grp) = ((self.cmd_header & 0xff) as u8, ((self.cmd_header >> 8) & 0xff) as u8);
        alloc::format!(
            "{} error table: valid {:#x} id {:#06x} {} ilink {:#x}/{:#x} data {:#x} {:#x} {:#x}; last command group {:#04x} code {:#04x}",
            which, self.valid, self.error_id, error_name(self.error_id), self.ilink1, self.ilink2,
            self.data1, self.data2, self.data3, grp, op
        ) + &alloc::format!(" pc {:#x}", self.pc)
    }
}

/// `iwl_error_event_table`: valid 0, error_id 1, trm_hw_status 2-3, blink2 4,
/// ilink1 5, ilink2 6, data1-3 7-9, ..., hcmd 23, ..., last_cmd_id 29.
pub fn lmac(w: &[u32]) -> Option<Entry> {
    if w.len() < LMAC_WORDS {
        return None;
    }
    Some(Entry { valid: w[0], error_id: w[1], ilink1: w[5], ilink2: w[6], data1: w[7], data2: w[8], data3: w[9], cmd_header: w[23], pc: w[20] })
}

/// `iwl_umac_error_event_table`: valid 0, error_id 1, blink1 2, blink2 3,
/// ilink1 4, ilink2 5, data1-3 6-8, umac major/minor 9-10, frame/stack
/// pointer 11-12, cmd_header 13, nic_isr_pref 14.
pub fn umac(w: &[u32]) -> Option<Entry> {
    if w.len() < UMAC_WORDS {
        return None;
    }
    Some(Entry { valid: w[0], error_id: w[1], ilink1: w[4], ilink2: w[5], data1: w[6], data2: w[7], data3: w[8], cmd_header: w[13], pc: w[11] })
}

/// The interrupt status, and whether either error bit is in it.
///
/// # Safety
/// `bar0` must be a mapped aperture for this part.
pub unsafe fn int_status(bar0: u64) -> u32 {
    core::ptr::read_volatile((bar0 + CSR_INT) as *const u32)
}

pub fn is_error(int: u32) -> bool {
    int & (CSR_INT_BIT_SW_ERR | CSR_INT_BIT_HW_ERR) != 0
}

/// Read `n` words of the part's memory at `addr`, or nothing when the MAC
/// access lock cannot be had -- a part asleep, which a dead one may be.
///
/// # Safety
/// `bar0` must be a mapped aperture for this part, and `addr` a place in its
/// SRAM; `iwx_nic_error` refuses a table address under `0x400000` as
/// "invalid error log pointer", so does this.
pub unsafe fn read_mem(bar0: u64, addr: u32, n: usize) -> Option<Vec<u32>> {
    if addr < 0x40_0000 || n == 0 {
        return None;
    }
    if !gen3::lock(bar0) {
        return None;
    }
    core::ptr::write_volatile((bar0 + HBUS_TARG_MEM_RADDR) as *mut u32, addr);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(core::ptr::read_volatile((bar0 + HBUS_TARG_MEM_RDAT) as *const u32));
    }
    gen3::unlock(bar0);
    Some(out)
}

/// Every table ALIVE named, read and rendered: one line per table, plus the
/// interrupt status. A table that reads as `valid 0` is one the firmware
/// never wrote, which is itself a fact -- a firmware that is merely slow has
/// written none of them.
///
/// # Safety
/// As `read_mem`.
pub unsafe fn report(bar0: u64, a: &super::alive::Alive) -> Vec<String> {
    let mut out = Vec::new();
    let int = int_status(bar0);
    out.push(alloc::format!(
        "CSR_INT {:#010x}{}{}",
        int,
        if int & CSR_INT_BIT_SW_ERR != 0 { " -- microcode fault" } else { "" },
        if int & CSR_INT_BIT_HW_ERR != 0 { " -- DMA error" } else { "" }
    ));
    for (i, &t) in a.lmac_error_table.iter().enumerate() {
        if t == 0 {
            continue;
        }
        match read_mem(bar0, t, LMAC_WORDS).and_then(|w| lmac(&w)) {
            Some(e) => out.push(e.render(if i == 0 { "lmac0" } else { "lmac1" })),
            None => out.push(alloc::format!("lmac{} error table at {:#x} could not be read", i, t)),
        }
    }
    if a.umac_error_table != 0 {
        match read_mem(bar0, a.umac_error_table, UMAC_WORDS).and_then(|w| umac(&w)) {
            Some(e) => out.push(e.render("umac")),
            None => out.push(alloc::format!("umac error table at {:#x} could not be read", a.umac_error_table)),
        }
    }
    out
}

pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();
    let mut w = [0u32; LMAC_WORDS];
    w[0] = 1;
    w[1] = 0x2000;
    w[5] = 0x1111;
    w[6] = 0x2222;
    w[7] = 7;
    w[8] = 8;
    w[9] = 9;
    w[23] = 0x0000_0128; // group 1, MAC_CONTEXT_CMD
    let l = lmac(&w);
    out.push(("err: the LMAC table's links, data and last command are read from where VER_3 puts them", l == Some(Entry { valid: 1, error_id: 0x2000, ilink1: 0x1111, ilink2: 0x2222, data1: 7, data2: 8, data3: 9, cmd_header: 0x128, pc: 0 })));
    out.push(("err: a short LMAC table is refused rather than read past", lmac(&w[..LMAC_WORDS - 1]).is_none()));
    let mut u = [0u32; UMAC_WORDS];
    u[0] = 1;
    u[1] = 0x1000;
    u[4] = 0xa;
    u[5] = 0xb;
    u[6] = 1;
    u[7] = 2;
    u[8] = 3;
    u[13] = 0x0000_052b;
    out.push(("err: the UMAC table's links sit two words earlier and its command header at 13", umac(&u) == Some(Entry { valid: 1, error_id: 0x1000, ilink1: 0xa, ilink2: 0xb, data1: 1, data2: 2, data3: 3, cmd_header: 0x52b, pc: 0 })));
    let r = l.unwrap().render("lmac0");
    out.push(("err: the rendering names the assertion and splits the last command into group and code", r.contains("ADVANCED_SYSASSERT") && r.contains("group 0x01 code 0x28")));
    out.push(("err: the two error bits are the ones alive.rs polls for, and neither is the alive bit", is_error(CSR_INT_BIT_SW_ERR) && is_error(CSR_INT_BIT_HW_ERR) && !is_error(1 << 0) && CSR_INT_BIT_SW_ERR == 0x0200_0000));
    out.push(("err: the memory window is inside the first page of the aperture", HBUS_TARG_MEM_RADDR < 0x1000 && HBUS_TARG_MEM_RDAT < 0x1000 && HBUS_TARG_MEM_RADDR == 0x40c && HBUS_TARG_MEM_RDAT == 0x41c));
    out.push(("err: an unnamed error id prints as a number rather than as something plausible", error_name(0x1234) == "?" && error_name(0x2000) == "ADVANCED_SYSASSERT"));
    out
}
