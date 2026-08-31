//! ACPI table discovery: RSDP -> XSDT -> MADT / FADT / HPET / MCFG.
//!
//! Everything here reads by explicit byte offset with `read_unaligned`, never
//! by declaring a `#[repr(C)]` struct and dereferencing it. That is deliberate.
//! ACPI tables are byte-packed and several fields sit at offsets Rust would
//! never choose: the HPET base address is at offset 44, which is 4-aligned but
//! not 8-aligned, so a `repr(C)` struct would quietly place that `u64` at 48
//! and read four bytes of the wrong field. `repr(C, packed)` avoids the padding
//! but then taking a reference to a field is itself unsound. Offsets it is.
//!
//! Reference: ACPI Specification 6.5, sections 5.2.5 (RSDP), 5.2.8 (XSDT),
//! 5.2.12 (MADT).

#![allow(dead_code)]

pub mod aml;
pub mod eval;

use core::ffi::c_void;
use core::ptr::read_unaligned;

#[inline]
unsafe fn rd_u8(p: *const u8, off: usize) -> u8 {
    unsafe { read_unaligned(p.add(off)) }
}
#[inline]
unsafe fn rd_u16(p: *const u8, off: usize) -> u16 {
    unsafe { read_unaligned(p.add(off) as *const u16) }
}
#[inline]
unsafe fn rd_u32(p: *const u8, off: usize) -> u32 {
    unsafe { read_unaligned(p.add(off) as *const u32) }
}
#[inline]
unsafe fn rd_u64(p: *const u8, off: usize) -> u64 {
    unsafe { read_unaligned(p.add(off) as *const u64) }
}

/// All ACPI tables checksum to zero over their whole declared length.
unsafe fn checksum_ok(p: *const u8, len: usize) -> bool {
    let mut sum: u8 = 0;
    for i in 0..len {
        sum = sum.wrapping_add(unsafe { rd_u8(p, i) });
    }
    sum == 0
}

const SDT_HEADER_LEN: usize = 36;

#[derive(Clone, Copy, Default)]
pub struct IoApicInfo {
    pub id: u8,
    pub addr: u64,
    pub gsi_base: u32,
}

/// An MADT interrupt source override: "ISA IRQ `source` is really GSI `gsi`".
///
/// These exist because the legacy PIC IRQ numbering and the IOAPIC's global
/// system interrupt numbering are not the same map. On most machines the
/// timer's IRQ 0 is remapped; keyboard IRQ 1 usually is not, but assuming that
/// is how you write a keyboard driver that works in QEMU and not on hardware.
#[derive(Clone, Copy, Default)]
pub struct OverrideInfo {
    pub source: u8,
    pub gsi: u32,
    pub flags: u16,
}

/// One table as it was found: where it is, how long it says it is, and
/// whether its bytes add up.
///
/// Kept, where before every pointer was read once and dropped. Five integers
/// were all anything wanted while ACPI meant the MADT and a timer port; the
/// moment the DSDT is going to be *executed* the questions become which tables
/// exist, whether each one is intact, and where the bytecode starts.
#[derive(Clone, Copy, Default)]
pub struct Table {
    pub sig: [u8; 4],
    pub addr: u64,
    pub len: u32,
    /// Whether the whole declared length sums to zero.
    pub sound: bool,
}

impl Table {
    pub fn name(&self) -> &str {
        core::str::from_utf8(&self.sig).unwrap_or("????")
    }

    /// The bytes after the 36-byte header, which for a DSDT or SSDT is AML.
    ///
    /// # Safety
    /// Only meaningful for a table this module found and checksummed. ACPI
    /// memory is identity-mapped write-back and is never handed to the frame
    /// allocator or the heap, so the pointer stays valid for the life of the
    /// machine.
    pub unsafe fn body(&self) -> &'static [u8] {
        let len = (self.len as usize).saturating_sub(SDT_HEADER_LEN);
        unsafe { core::slice::from_raw_parts((self.addr as *const u8).add(SDT_HEADER_LEN), len) }
    }
}

/// How many tables are recorded. A laptop has a dozen; the SSDT count is what
/// varies and eight of those is already unusual.
pub const MAX_TABLES: usize = 32;

/// Enough for any laptop, and a bound rather than an allocation: the MADT is
/// parsed before the heap exists.
pub const MAX_CPUS: usize = 64;

pub const MAX_IOAPICS: usize = 4;
pub const MAX_OVERRIDES: usize = 24;

#[derive(Clone, Copy)]
pub struct Acpi {
    pub revision: u8,
    pub lapic_addr: u64,
    pub cpus: usize,
    /// The local APIC id of each enabled processor.
    ///
    /// The count alone was enough while there was one core; starting the other
    /// seven needs to address them, and an APIC id is not its index -- on a
    /// hybrid part the P-core threads and the E-cores are not numbered
    /// contiguously, so guessing `0..cpus` sends INIT to ids that answer to
    /// nobody.
    pub apic_ids: [u32; MAX_CPUS],
    pub ioapics: [IoApicInfo; MAX_IOAPICS],
    pub ioapic_count: usize,
    pub overrides: [OverrideInfo; MAX_OVERRIDES],
    pub override_count: usize,
    pub hpet: Option<u64>,
    pub mcfg: Option<u64>,
    /// ACPI PM timer I/O port. A fixed 3.579545 MHz clock, and a useful
    /// independent reference if PIT calibration ever looks wrong.
    pub pm_timer: Option<u32>,
    /// Every table found, including the ones nothing parses. Recorded so a
    /// missing one is a visible absence rather than a silent `_ => {}`.
    pub tables: [Table; MAX_TABLES],
    pub table_count: usize,
    /// The DSDT, reached through the FADT rather than through the XSDT, since
    /// it is the one table the root list does not point at.
    pub dsdt: Option<Table>,
    /// How many tables failed their checksum. Loud, because a table that does
    /// not add up is one this kernel is about to read as bytecode.
    pub unsound: usize,
}

impl Acpi {
    const fn new() -> Self {
        Self {
            revision: 0,
            lapic_addr: 0xFEE0_0000,
            cpus: 0,
            apic_ids: [0; MAX_CPUS],
            ioapics: [IoApicInfo { id: 0, addr: 0, gsi_base: 0 }; MAX_IOAPICS],
            ioapic_count: 0,
            overrides: [OverrideInfo { source: 0, gsi: 0, flags: 0 }; MAX_OVERRIDES],
            override_count: 0,
            hpet: None,
            mcfg: None,
            pm_timer: None,
            tables: [Table { sig: [0; 4], addr: 0, len: 0, sound: false }; MAX_TABLES],
            table_count: 0,
            dsdt: None,
            unsound: 0,
        }
    }

    /// Record one table. Silently drops past `MAX_TABLES`, and the count says
    /// so by reaching the cap.
    fn note(&mut self, sig: [u8; 4], addr: u64, len: u32, sound: bool) {
        if !sound {
            self.unsound += 1;
        }
        if self.table_count < MAX_TABLES {
            self.tables[self.table_count] = Table { sig, addr, len, sound };
            self.table_count += 1;
        }
    }

    /// Every table whose signature matches, in the order found.
    ///
    /// A plural because SSDTs are a plural: firmware routinely splits the
    /// namespace across several, and taking only the first is how half a
    /// machine's devices go missing.
    pub fn tables_named(&self, sig: &[u8; 4]) -> impl Iterator<Item = Table> + '_ {
        let sig = *sig;
        self.tables[..self.table_count].iter().copied().filter(move |t| t.sig == sig)
    }

    /// Every table carrying AML: the DSDT, then each SSDT in order.
    ///
    /// Order matters and is not cosmetic. An SSDT may extend a scope the DSDT
    /// defined, so loading them out of order leaves names unresolvable.
    pub fn aml_tables(&self) -> impl Iterator<Item = Table> + '_ {
        self.dsdt.into_iter().chain(self.tables_named(b"SSDT"))
    }

    /// Resolve a legacy ISA IRQ to its global system interrupt, honouring any
    /// override. Returns `(gsi, flags)`; identity-mapped when no override says
    /// otherwise.
    pub fn gsi_for_irq(&self, irq: u8) -> (u32, u16) {
        for i in 0..self.override_count {
            let o = self.overrides[i];
            if o.source == irq {
                return (o.gsi, o.flags);
            }
        }
        (irq as u32, 0)
    }

    pub fn primary_ioapic(&self) -> Option<IoApicInfo> {
        if self.ioapic_count == 0 {
            None
        } else {
            Some(self.ioapics[0])
        }
    }
}

/// Walk the ACPI tables starting from the RSDP the firmware handed us.
///
/// # Safety
/// `rsdp` must be the pointer taken from the UEFI configuration table, and the
/// tables must be identity-mapped.
pub unsafe fn parse(rsdp: *const c_void) -> Option<Acpi> {
    if rsdp.is_null() {
        return None;
    }
    let p = rsdp as *const u8;

    unsafe {
        // "RSD PTR "
        let mut sig = [0u8; 8];
        for (i, b) in sig.iter_mut().enumerate() {
            *b = rd_u8(p, i);
        }
        if &sig != b"RSD PTR " {
            return None;
        }
        // The v1 checksum covers only the first 20 bytes, even on a v2 RSDP.
        if !checksum_ok(p, 20) {
            return None;
        }

        let mut acpi = Acpi::new();
        acpi.revision = rd_u8(p, 15);

        // Revision >= 2 means an XSDT with 64-bit pointers. Below that we only
        // have the 32-bit RSDT.
        let (sdt, entry_size) = if acpi.revision >= 2 {
            if !checksum_ok(p, rd_u32(p, 20) as usize) {
                return None;
            }
            (rd_u64(p, 24), 8usize)
        } else {
            (rd_u32(p, 16) as u64, 4usize)
        };

        if sdt == 0 {
            return None;
        }
        let sdt = sdt as *const u8;
        let sdt_len = rd_u32(sdt, 4) as usize;
        if sdt_len < SDT_HEADER_LEN {
            return None;
        }

        let count = (sdt_len - SDT_HEADER_LEN) / entry_size;
        for i in 0..count {
            let off = SDT_HEADER_LEN + i * entry_size;
            let table = if entry_size == 8 {
                rd_u64(sdt, off)
            } else {
                rd_u32(sdt, off) as u64
            };
            if table == 0 {
                continue;
            }
            let t = table as *const u8;
            let mut tsig = [0u8; 4];
            for (j, b) in tsig.iter_mut().enumerate() {
                *b = rd_u8(t, j);
            }
            let len = rd_u32(t, 4) as usize;
            // Checked here, once, for every table. Before this only the RSDP
            // was validated, which was defensible while the tables supplied
            // five integers and stops being defensible now that one of them is
            // about to be run as bytecode.
            let sound = len >= SDT_HEADER_LEN && checksum_ok(t, len);
            acpi.note(tsig, table, len as u32, sound);

            match &tsig {
                b"APIC" => parse_madt(t, len, &mut acpi),
                b"HPET" => acpi.hpet = Some(rd_u64(t, 44)),
                b"MCFG" => {
                    // First allocation entry starts at 44 (36 header + 8 reserved).
                    if len >= 44 + 16 {
                        acpi.mcfg = Some(rd_u64(t, 44));
                    }
                }
                b"FACP" => {
                    // FADT: PM_TMR_BLK at offset 76, PM_TMR_LEN at 91.
                    if len >= 92 && rd_u8(t, 91) == 4 {
                        let port = rd_u32(t, 76);
                        if port != 0 {
                            acpi.pm_timer = Some(port);
                        }
                    }
                    // The DSDT is the one table the root list does not point
                    // at: it hangs off the FADT instead. X_DSDT at 140 is the
                    // 64-bit field and wins when it is present, because on a
                    // machine with tables above 4 GiB the 32-bit field at 40
                    // is zero rather than wrong.
                    let mut d = 0u64;
                    if len >= 148 {
                        d = rd_u64(t, 140);
                    }
                    if d == 0 && len >= 44 {
                        d = rd_u32(t, 40) as u64;
                    }
                    if d != 0 {
                        let p = d as *const u8;
                        let dlen = rd_u32(p, 4) as usize;
                        if dlen >= SDT_HEADER_LEN {
                            let ok = checksum_ok(p, dlen);
                            acpi.note(*b"DSDT", d, dlen as u32, ok);
                            acpi.dsdt =
                                Some(Table { sig: *b"DSDT", addr: d, len: dlen as u32, sound: ok });
                        }
                    }
                }
                _ => {}
            }
        }

        unsafe { *PARSED.get() = Some(acpi) };
        Some(acpi)
    }
}

unsafe fn parse_madt(t: *const u8, len: usize, acpi: &mut Acpi) {
    unsafe {
        acpi.lapic_addr = rd_u32(t, 36) as u64;

        let mut off = 44;
        while off + 2 <= len {
            let kind = rd_u8(t, off);
            let elen = rd_u8(t, off + 1) as usize;
            // A zero-length entry would spin here forever.
            if elen < 2 || off + elen > len {
                break;
            }

            match kind {
                // Processor Local APIC. Bit 0 of flags = enabled.
                0 => {
                    if rd_u32(t, off + 4) & 1 != 0 {
                        if acpi.cpus < MAX_CPUS {
                            acpi.apic_ids[acpi.cpus] = rd_u8(t, off + 3) as u32;
                        }
                        acpi.cpus += 1;
                    }
                }
                // I/O APIC.
                1 => {
                    if acpi.ioapic_count < MAX_IOAPICS {
                        acpi.ioapics[acpi.ioapic_count] = IoApicInfo {
                            id: rd_u8(t, off + 2),
                            addr: rd_u32(t, off + 4) as u64,
                            gsi_base: rd_u32(t, off + 8),
                        };
                        acpi.ioapic_count += 1;
                    }
                }
                // Interrupt Source Override.
                2 => {
                    if acpi.override_count < MAX_OVERRIDES {
                        acpi.overrides[acpi.override_count] = OverrideInfo {
                            source: rd_u8(t, off + 3),
                            gsi: rd_u32(t, off + 4),
                            flags: rd_u16(t, off + 8),
                        };
                        acpi.override_count += 1;
                    }
                }
                // Local APIC Address Override. A 64-bit address at offset 4,
                // which is exactly the misalignment that motivates reading by
                // offset rather than by struct.
                5 => acpi.lapic_addr = rd_u64(t, off + 4),
                // Processor Local x2APIC.
                9 => {
                    if rd_u32(t, off + 8) & 1 != 0 {
                        if acpi.cpus < MAX_CPUS {
                            acpi.apic_ids[acpi.cpus] = rd_u32(t, off + 4);
                        }
                        acpi.cpus += 1;
                    }
                }
                _ => {}
            }

            off += elen;
        }
    }
}

/// Every table found, with its verdict.
pub fn report(a: &Acpi) {
    crate::kprintln!("  {} table(s) from the root list", a.table_count);
    for i in 0..a.table_count {
        let t = a.tables[i];
        crate::kprintln!(
            "  {}  {:#012x}  {:>7} bytes  {}",
            t.name(),
            t.addr,
            t.len,
            if t.sound { "ok" } else { "CHECKSUM FAILED" }
        );
    }
    match a.dsdt {
        Some(d) => {
            let aml: usize = a.aml_tables().map(|t| t.len as usize).sum::<usize>()
                - (a.aml_tables().count() * SDT_HEADER_LEN);
            crate::kprintln!(
                "  {} byte(s) of AML across {} table(s); the DSDT is {} of it",
                aml,
                a.aml_tables().count(),
                d.len as usize - SDT_HEADER_LEN
            );
        }
        None => crate::kprintln!("  no DSDT: the FADT named none, so there is no namespace"),
    }
    if a.unsound > 0 {
        crate::kprintln!("  {} table(s) did not checksum -- do not trust what they say", a.unsound);
    }
}

/// What can be checked about ACPI without a machine to check against.
pub fn selftest(a: &Option<Acpi>) -> bool {
    let mut ok = true;
    fn claim(ok: &mut bool, good: bool, what: &str) {
        crate::kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
        *ok &= good;
    }

    let Some(a) = a else {
        claim(&mut ok, false, "ACPI was parsed at all");
        return false;
    };

    claim(&mut ok, a.table_count > 0, "the root list named at least one table");
    claim(&mut ok, a.unsound == 0, "and every table checksums");

    // The FADT is what points at the DSDT, so a machine with one and no DSDT
    // has a pointer this code failed to follow rather than firmware with no
    // namespace.
    let fadt = a.tables_named(b"FACP").next();
    claim(&mut ok, fadt.is_some(), "there is a FADT");
    if fadt.is_some() {
        claim(&mut ok, a.dsdt.is_some(), "and the DSDT it names was reached");
    }

    if let Some(d) = a.dsdt {
        claim(&mut ok, d.len as usize > SDT_HEADER_LEN, "the DSDT has a body");
        // Read the first and last byte. If the table were not mapped this
        // would fault rather than answer, which is the claim: ACPI memory
        // survives ExitBootServices and is never handed to the allocator.
        let body = unsafe { d.body() };
        let touched = !body.is_empty() && (body[0] as usize + body[body.len() - 1] as usize) < 512;
        claim(&mut ok, touched, "and its first and last byte are readable");
    }

    claim(
        &mut ok,
        a.aml_tables().count() >= a.dsdt.iter().count(),
        "the AML list starts with the DSDT and adds every SSDT",
    );
    ok
}

// --- the namespace, built once ------------------------------------------

/// The namespace, built on first use and kept.
///
/// Behind a lock rather than `Racy` because the battery reading refreshes from
/// a timer while the shell can ask for the same tree, and building it twice
/// would be two trees that disagree about which one a node index refers to.
static NAMESPACE: crate::sync::Spin<Option<aml::Namespace>> = crate::sync::Spin::new(None);

/// What building it established, kept so a report does not have to rebuild.
static SUMMARY: crate::sync::Racy<Option<NsSummary>> = crate::sync::Racy::new(None);

#[derive(Clone, Copy)]
pub struct NsSummary {
    pub tables: usize,
    pub nodes: usize,
    pub skipped: usize,
    pub redefinitions: usize,
    /// Where a walk stopped short, if one did. The whole point of the walk is
    /// that this is `None`.
    pub stop: Option<aml::Stop>,
    /// Bytes offered to the walk, and bytes it consumed.
    pub offered: usize,
}

/// Build the namespace from every AML table, in order.
///
/// Order is not cosmetic: an SSDT may extend a scope the DSDT opened, so
/// loading them out of order leaves names unresolvable.
pub fn load_namespace(a: &Acpi) -> NsSummary {
    let mut ns = aml::Namespace::new();
    let mut sum =
        NsSummary { tables: 0, nodes: 0, skipped: 0, redefinitions: 0, stop: None, offered: 0 };
    for t in a.aml_tables() {
        if !t.sound {
            // A table that does not add up is not walked. Executing bytes that
            // failed their own checksum is the one thing worth refusing here.
            continue;
        }
        let body = unsafe { t.body() };
        sum.offered += body.len();
        let r = ns.load(body);
        sum.tables += 1;
        sum.nodes += r.nodes;
        sum.skipped += r.skipped_conditionals;
        if sum.stop.is_none() {
            sum.stop = r.stop;
        }
    }
    sum.redefinitions = ns.redefinitions;
    *NAMESPACE.lock() = Some(ns);
    unsafe { *SUMMARY.get() = Some(sum) };
    sum
}

/// Run `f` against the namespace, building it first if nobody has.
pub fn with_namespace<R>(a: &Acpi, f: impl FnOnce(&aml::Namespace) -> R) -> Option<R> {
    if NAMESPACE.lock().is_none() {
        load_namespace(a);
    }
    NAMESPACE.lock().as_ref().map(f)
}

/// The parsed tables, kept so anything can reach them without being handed a
/// reference through every call.
///
/// `parse` returns by value and `main` passes that value down, which was
/// fine while ACPI meant five integers read once at boot. A battery reading
/// refreshes on a timer and a diag suite takes no arguments, so both need to
/// find it themselves.
static PARSED: crate::sync::Racy<Option<Acpi>> = crate::sync::Racy::new(None);

pub fn parsed() -> Option<Acpi> {
    unsafe { *PARSED.get() }
}

/// The AML suite, for `diag`, which hands its suites no arguments.
pub fn diag_acpi() -> bool {
    aml_selftest(&parsed())
}

pub fn ns_summary() -> Option<NsSummary> {
    unsafe { *SUMMARY.get() }
}

/// Print the namespace tree from `root`, or a summary when asked for nothing.
pub fn ns_report(a: &Acpi, filter: &str) {
    let sum = match ns_summary() {
        Some(s) => s,
        None => load_namespace(a),
    };
    crate::kprintln!(
        "  {} node(s) from {} table(s), {} byte(s) of AML",
        sum.nodes,
        sum.tables,
        sum.offered
    );
    match sum.stop {
        None => crate::kprintln!("  every table walked to its last byte"),
        Some(s) => {
            crate::console::set_color(crate::gfx::console::LTRED);
            crate::kprintln!(
                "  table {} stopped at byte {}: {:?}",
                s.table,
                s.at,
                s.why
            );
            crate::kprintln!("  the namespace past that point is not real");
            crate::console::set_color(crate::gfx::console::LTGRAY);
            // The bytes it choked on. An offset alone says where and not what,
            // and the what is the whole fix.
            if let Some(t) = a.aml_tables().nth(s.table) {
                let body = unsafe { t.body() };
                let from = s.at.saturating_sub(8);
                let to = (s.at + 24).min(body.len());
                let mut line = alloc::string::String::new();
                for (k, b) in body[from..to].iter().enumerate() {
                    let mark = if from + k == s.at { '>' } else { ' ' };
                    line.push(mark);
                    let hi = b >> 4;
                    let lo = b & 0xF;
                    line.push(char::from_digit(hi as u32, 16).unwrap_or('?'));
                    line.push(char::from_digit(lo as u32, 16).unwrap_or('?'));
                }
                crate::kprintln!("  bytes {}..{}:{}", from, to, line);
            }
        }
    }
    if sum.skipped > 0 {
        crate::kprintln!(
            "  {} conditional block(s) skipped whole; names inside are not defined",
            sum.skipped
        );
    }
    if sum.redefinitions > 0 {
        crate::kprintln!("  {} name(s) defined more than once", sum.redefinitions);
    }

    with_namespace(a, |ns| {
        let mut shown = 0usize;
        for i in 0..ns.len() {
            let n = ns.node(i);
            if matches!(n.kind, aml::Kind::Field { .. }) && filter.is_empty() {
                // A machine has hundreds of these and they are only meaningful
                // beside their region. Listed when asked for by name.
                continue;
            }
            let p = ns.path(i);
            if !filter.is_empty() && !p.contains(filter) {
                continue;
            }
            if shown >= 200 {
                crate::kprintln!("  ... and more; give a path to narrow it");
                break;
            }
            shown += 1;
            match n.kind {
                aml::Kind::Method { args, serialized } => crate::kprintln!(
                    "  {:<34} method, {} arg(s){}",
                    p,
                    args,
                    if serialized { ", serialized" } else { "" }
                ),
                aml::Kind::OpRegion { space } => {
                    crate::kprintln!("  {:<34} region in {}", p, space_name(space))
                }
                k => crate::kprintln!("  {:<34} {:?}", p, k),
            }
        }
        if shown == 0 {
            crate::kprintln!("  nothing matches '{}'", filter);
        }
    });
}

/// The address spaces a region can live in. Named because the number alone
/// says nothing about whether this kernel can reach it.
pub fn space_name(space: u8) -> &'static str {
    match space {
        0 => "system memory",
        1 => "system I/O",
        2 => "PCI config",
        3 => "embedded controller",
        4 => "SMBus",
        5 => "CMOS",
        6 => "PCI bar target",
        7 => "IPMI",
        8 => "general purpose I/O",
        9 => "generic serial bus",
        10 => "platform communications",
        _ => "an unnamed space",
    }
}

// --- evaluating, from the shell -----------------------------------------

/// Turn a written path into one the namespace can resolve.
///
/// Accepts what a person types: `\_SB.PCI0._PRT`, `_SB.PCI0`, or a bare
/// `_PRT` to be searched for. Segments are padded to four characters with
/// underscores, which is what ACPI stores.
pub fn parse_path(text: &str) -> aml::Path {
    let mut p = aml::Path::default();
    let mut rest = text.trim();
    if let Some(r) = rest.strip_prefix('\\') {
        p.rooted = true;
        rest = r;
    }
    while let Some(r) = rest.strip_prefix('^') {
        p.parents += 1;
        rest = r;
    }
    for part in rest.split('.').filter(|s| !s.is_empty()) {
        let mut seg = [b'_'; 4];
        for (i, c) in part.bytes().take(4).enumerate() {
            seg[i] = c.to_ascii_uppercase();
        }
        p.segs.push(seg);
    }
    p
}

/// Render a value the way a person reads one.
///
/// Packages are printed one element per line and indented, because a `_BST` is
/// a four-element package and its meaning is entirely positional: printing it
/// on one line makes the reader count commas to find the one they want.
fn show(ns: &aml::Namespace, v: &eval::Value, indent: usize, out: &mut alloc::string::String) {
    let pad = "                ";
    let lead = &pad[..indent.min(pad.len())];
    match v {
        eval::Value::Int(i) => {
            out.push_str(&alloc::format!("{}{} ({:#x})\n", lead, i, i));
        }
        eval::Value::Str(s) => out.push_str(&alloc::format!("{}\"{}\"\n", lead, s)),
        eval::Value::Buf(b) => {
            out.push_str(&alloc::format!("{}buffer of {} byte(s):", lead, b.len()));
            for byte in b.iter().take(16) {
                out.push_str(&alloc::format!(" {:02x}", byte));
            }
            if b.len() > 16 {
                out.push_str(" ...");
            }
            out.push('\n');
        }
        eval::Value::Pkg(items) => {
            out.push_str(&alloc::format!("{}package of {}:\n", lead, items.len()));
            for (i, item) in items.iter().take(24).enumerate() {
                out.push_str(&alloc::format!("{}  [{}]\n", lead, i));
                show(ns, item, indent + 4, out);
            }
            if items.len() > 24 {
                out.push_str(&alloc::format!("{}  ... {} more\n", lead, items.len() - 24));
            }
        }
        eval::Value::Node(n) => {
            out.push_str(&alloc::format!("{}-> {}\n", lead, ns.path(*n)));
        }
        eval::Value::Uninit => out.push_str(&alloc::format!("{}nothing\n", lead)),
    }
}

/// Evaluate one name and print what it answered.
pub fn eval_report(a: &Acpi, text: &str) {
    let mut words = text.split_whitespace();
    let Some(target) = words.next() else {
        crate::kprintln!("  acpi eval <path> [args...]   run a method, or read a name");
        return;
    };
    let mut args: alloc::vec::Vec<eval::Value> = alloc::vec::Vec::new();
    for w in words {
        let v = if let Some(hex) = w.strip_prefix("0x") {
            u64::from_str_radix(hex, 16).unwrap_or(0)
        } else {
            w.parse::<u64>().unwrap_or(0)
        };
        args.push(eval::Value::Int(v));
    }

    let p = parse_path(target);
    let done = with_namespace(a, |ns| {
        let Some(node) = ns.resolve(0, &p) else {
            crate::kprintln!("  no such name: {}", target);
            return;
        };
        let mut it = eval::Interp::new(ns);
        let r = it.eval_node(node, &args);
        match r {
            Ok(v) => {
                let mut s = alloc::string::String::new();
                show(ns, &v, 2, &mut s);
                crate::kprintln!("  {} in {} step(s):", ns.path(node), it.steps());
                for line in s.lines() {
                    crate::kprintln!("{}", line);
                }
            }
            Err(e) => {
                crate::console::set_color(crate::gfx::console::LTRED);
                crate::kprintln!("  {} failed after {} step(s): {}", ns.path(node), it.steps(), fault_text(&e));
                crate::console::set_color(crate::gfx::console::LTGRAY);
            }
        }
        if let Some(d) = it.debug {
            crate::kprintln!("  the firmware wrote to the debug object: {}", d);
        }
    });
    if done.is_none() {
        crate::kprintln!("  no namespace");
    }
}

/// A fault as a sentence. The opcode and its offset are in there because the
/// fix for an unimplemented one is a match arm, and the arm needs the number.
pub fn fault_text(f: &eval::Fault) -> alloc::string::String {
    match f {
        eval::Fault::Budget => alloc::format!(
            "it ran past {} steps, which is a loop waiting on something that never answered",
            eval::BUDGET
        ),
        eval::Fault::Depth => alloc::string::String::from("it nested too deep"),
        eval::Fault::Opcode(b, at) => {
            alloc::format!("opcode {:#04x} at byte {} is not implemented", b, at)
        }
        eval::Fault::ExtOpcode(b, at) => {
            alloc::format!("extended opcode {:#04x} at byte {} is not implemented", b, at)
        }
        eval::Fault::NotFound(n) => alloc::format!("it referred to '{}', which does not exist", n),
        eval::Fault::Type(want) => alloc::format!("something was not {}", want),
        eval::Fault::Region(s) => {
            alloc::format!("it reads a region in {}, which has no handler yet", space_name(*s))
        }
        eval::Fault::Truncated => alloc::string::String::from("the bytecode ended mid-term"),
        eval::Fault::DivideByZero => alloc::string::String::from("it divided by zero"),
        eval::Fault::Args => alloc::string::String::from("it wanted more arguments than it got"),
    }
}

/// Four declarations, hand-assembled, covering what the claims below need.
///
/// Written by hand rather than by a compiler because there is no `iasl` on the
/// machine that builds this, and that turns out to be an advantage: a fixture
/// assembled here is shaped to hit the paths this implementation has, which is
/// the same argument `tools/hybtest.py` makes for building a model to exercise
/// every layer kind rather than taking whatever a real checkpoint happens to
/// contain.
///
///     08 'TST1' 0A 2A                     Name (TST1, 42)
///     14 0B 'TST2' 01  A4 72 68 01 00     Method (TST2, 1) { Return (Arg0 + 1) }
///     14 09 'TST3' 00  A2 02 01           Method (TST3, 0) { While (One) {} }
///     14 07 'TST4' 00  6F                 Method (TST4, 0) { <no such opcode> }
///
/// The package lengths count themselves, which is the rule most easily got
/// wrong: 0x0B is one length byte plus four name bytes plus one flags byte
/// plus five of body.
#[rustfmt::skip]
const FIXTURE: &[u8] = &[
    0x08, 0x54, 0x53, 0x54, 0x31, 0x0A, 0x2A,
    0x14, 0x0B, 0x54, 0x53, 0x54, 0x32, 0x01, 0xA4, 0x72, 0x68, 0x01, 0x00,
    0x14, 0x09, 0x54, 0x53, 0x54, 0x33, 0x00, 0xA2, 0x02, 0x01,
    0x14, 0x07, 0x54, 0x53, 0x54, 0x34, 0x00, 0x6F,
];

/// What can be checked about AML on any machine, plus what this one says.
pub fn aml_selftest(a: &Option<Acpi>) -> bool {
    let mut ok = true;
    fn claim(ok: &mut bool, good: bool, what: &str) {
        crate::kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
        *ok &= good;
    }

    // --- the fixture, which is the same on every machine -----------------
    let mut ns = aml::Namespace::new();
    let r = ns.load(FIXTURE);
    claim(&mut ok, r.stop.is_none(), "a hand-assembled table walks to its last byte");
    claim(&mut ok, r.nodes == 4, "and declares exactly the four names in it");

    let find = |ns: &aml::Namespace, n: &str| ns.resolve(0, &parse_path(n));

    // A data name reads back as what it was given. The first thing that would
    // break if package lengths or name lengths were off by one.
    match find(&ns, "TST1") {
        Some(node) => {
            let mut it = eval::Interp::new(&ns);
            let got = it.eval_node(node, &[]);
            claim(&mut ok, got == Ok(eval::Value::Int(42)), "a name reads back as its value");
        }
        None => claim(&mut ok, false, "TST1 is in the namespace"),
    }

    // A method takes an argument, does arithmetic, and returns.
    match find(&ns, "TST2") {
        Some(node) => {
            let mut it = eval::Interp::new(&ns);
            let got = it.eval_node(node, &[eval::Value::Int(41)]);
            claim(&mut ok, got == Ok(eval::Value::Int(42)), "a method adds one to its argument");
            let mut it = eval::Interp::new(&ns);
            claim(
                &mut ok,
                it.eval_node(node, &[]) == Err(eval::Fault::Args),
                "and refuses to run with fewer arguments than it declared",
            );
        }
        None => claim(&mut ok, false, "TST2 is in the namespace"),
    }

    // The claim this whole evaluator is bounded for. `While (One) {}` is legal
    // AML and a machine that ran it would simply stop.
    match find(&ns, "TST3") {
        Some(node) => {
            let mut it = eval::Interp::new(&ns);
            let got = it.eval_node(node, &[]);
            claim(&mut ok, got == Err(eval::Fault::Budget), "a loop that never ends is stopped");
            claim(
                &mut ok,
                it.steps() > eval::BUDGET,
                "and it is the budget that stopped it, not an accident",
            );
        }
        None => claim(&mut ok, false, "TST3 is in the namespace"),
    }

    // An opcode with no arm is named rather than guessed at. This is the
    // property that makes a partial evaluator honest: the next machine that
    // needs something adds one match arm, and the number is in the message.
    match find(&ns, "TST4") {
        Some(node) => {
            let mut it = eval::Interp::new(&ns);
            let got = it.eval_node(node, &[]);
            claim(
                &mut ok,
                matches!(got, Err(eval::Fault::Opcode(0x6F, _))),
                "an opcode with no arm is refused, and says which opcode",
            );
        }
        None => claim(&mut ok, false, "TST4 is in the namespace"),
    }

    // A name that is not there is an error rather than a zero, because a
    // caller testing `_STA` against zero would read an absent device as one
    // that is present and switched off.
    claim(&mut ok, find(&ns, "NOPE").is_none(), "a name that was never declared is not found");

    // --- and what this particular firmware says --------------------------
    let Some(a) = a else {
        claim(&mut ok, false, "ACPI was parsed");
        return false;
    };
    let sum = ns_summary().unwrap_or_else(|| load_namespace(a));
    claim(&mut ok, sum.nodes > 0, "this machine's own tables declare a namespace");
    // The one that cannot be faked by a fixture: real firmware, walked whole.
    claim(&mut ok, sum.stop.is_none(), "and every one of them walked to its last byte");
    crate::kprintln!(
        "  {} node(s) from {} byte(s) of this machine's AML",
        sum.nodes,
        sum.offered
    );
    ok
}
