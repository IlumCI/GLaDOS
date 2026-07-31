//! glados -- a from-scratch ring-0 operating system for the MSI MS-16R8.
//!
//! There is no bootloader stage. UEFI has already put us in long mode, at
//! CPL 0, with an identity map, so this UEFI application simply *is* the
//! kernel: take what we need from the firmware, leave boot services, install
//! our own descriptor tables and page tables in place, and keep running. That
//! deletes ELF loading, relocation, and an entire handoff ABI -- the most
//! TempleOS-shaped option the hardware allows.

#![no_std]
#![no_main]
// Exception handlers need the compiler to emit an iretq-shaped prologue and
// epilogue, which no stable ABI provides.
#![feature(abi_x86_interrupt)]

extern crate alloc;

mod acpi;
mod ai;
mod cpu;
mod crypto;
mod dev;
mod edit;
mod gfx;
mod lang;
mod mem;
mod net;
mod pkg;
mod recovery;
mod serial;
mod shell;
mod store;
mod sync;
mod sysbox;
mod task;
mod time;
mod uefi;

use core::sync::atomic::{AtomicU64, Ordering};

use core::ffi::c_void;
use core::panic::PanicInfo;
use core::ptr;

use gfx::console::{self, LTCYAN, LTGREEN, LTRED, WHITE, YELLOW};
use gfx::{palette, Format, Framebuffer};
use uefi::*;

/// Everything we must extract from the firmware before it goes away.
///
/// After `ExitBootServices` there is no way back: no protocol lookups, no
/// config table, no console. Anything not captured here is gone for good.
pub struct BootInfo {
    pub fb: Framebuffer,
    /// ACPI RSDP. The root of MADT/FADT/MCFG/HPET discovery in M4.
    pub rsdp: *const c_void,
    pub mmap: *mut u8,
    pub mmap_size: usize,
    /// Firmware-reported stride. **Never** use `size_of::<MemoryDescriptor>()`.
    pub desc_size: usize,
    /// The framebuffer aperture. On this laptop it is a 64-bit BAR at
    /// 0x40_0000_0000 -- 256 GiB, far above the 18 GiB of RAM -- so the map
    /// limit has to be widened to reach it, and it must not be treated as
    /// device memory for caching purposes.
    pub fb_start: u64,
    /// One past the last byte of the framebuffer aperture. Our page tables have
    /// to cover this or the first write after `activate` faults.
    pub fb_end: u64,
    /// A llama2.c checkpoint, read off the boot volume before the firmware's
    /// filesystem went away. `None` is normal -- the system boots without one.
    pub model: Option<Blob>,
    /// The matching tokenizer.
    pub tokenizer: Option<Blob>,
    /// A DER bundle of root certificates. `None` means TLS can encrypt and
    /// cannot authenticate, which is reported rather than assumed.
    pub roots: Option<Blob>,
}

/// Where the weights live on the boot volume. Backslashes: this is a UEFI path
/// on the ESP, not a namespace path.
pub const MODEL_PATH: &str = "\\GLADOS\\model.bin";
pub const TOKENIZER_PATH: &str = "\\GLADOS\\tokenizer.bin";

#[no_mangle]
pub extern "efiapi" fn efi_main(image: Handle, st: *mut SystemTable) -> Status {
    serial::init();
    serial_println!("\n\nglados: entered efi_main");

    let st = unsafe { &mut *st };
    let bs = unsafe { &mut *st.boot_services };

    // UEFI arms a 5-minute watchdog for boot applications. We are never going
    // to return, so if we leave it armed the firmware resets the machine
    // mid-boot and it looks like a kernel hang.
    (bs.set_watchdog_timer)(0, 0, 0, ptr::null_mut());
    serial_println!("glados: watchdog disarmed");

    // --- Graphics Output Protocol ---
    let mut gop_ptr: *mut c_void = ptr::null_mut();
    let s = (bs.locate_protocol)(
        &GRAPHICS_OUTPUT_PROTOCOL_GUID,
        ptr::null_mut(),
        &mut gop_ptr,
    );
    if is_error(s) || gop_ptr.is_null() {
        con_out(st, "glados: no Graphics Output Protocol\r\n");
        serial_println!("glados: locate_protocol(GOP) failed: {:#x}", s);
        halt();
    }

    let gop = unsafe { &mut *(gop_ptr as *mut GraphicsOutputProtocol) };
    let mode = unsafe { &*gop.mode };
    let info = unsafe { &*mode.info };

    let format = match info.pixel_format {
        0 => Format::Rgbx,
        1 => Format::Bgrx,
        // BitMask would need us to derive shifts from the channel masks;
        // BltOnly means there is no linear framebuffer at all and the only
        // draw call lives in boot services, which we are about to leave.
        other => {
            con_out(st, "glados: unsupported GOP pixel format\r\n");
            serial_println!("glados: pixel_format {} unsupported", other);
            halt();
        }
    };

    let fb = unsafe {
        Framebuffer::new(
            mode.frame_buffer_base,
            info.horizontal_resolution,
            info.vertical_resolution,
            info.pixels_per_scan_line,
            format,
        )
    };

    let fb_start = mode.frame_buffer_base;
    let fb_end = mode.frame_buffer_base + mode.frame_buffer_size as u64;

    serial_println!(
        "glados: fb base={:#x} {}x{} stride={} format={:?}",
        mode.frame_buffer_base,
        info.horizontal_resolution,
        info.vertical_resolution,
        info.pixels_per_scan_line,
        format
    );

    // --- ACPI RSDP, while the configuration table still exists ---
    let rsdp = find_rsdp(st);
    serial_println!("glados: rsdp={:?}", rsdp);

    // --- Anything that needs a filesystem, while there still is one ---
    //
    // This has to happen before the memory map is sized: allocate_pool for a
    // 1 MiB model perturbs the map, and ExitBootServices rejects a stale key.
    // Loading afterwards would mean re-reading the map, which is exactly the
    // retry loop below and not worth entangling.
    let model = uefi::read_file(bs, image, MODEL_PATH);
    let tokenizer = uefi::read_file(bs, image, TOKENIZER_PATH);
    // The root bundle comes off the same volume for the same reason: this is
    // the only moment there is a filesystem to read it from.
    let roots = uefi::read_file(bs, image, net::trust::ROOTS_PATH);
    match &roots {
        Some(b) => serial_println!("glados: roots {} bytes from {}", b.len, net::trust::ROOTS_PATH),
        None => serial_println!("glados: no roots at {}", net::trust::ROOTS_PATH),
    }
    match &model {
        Some(b) => serial_println!("glados: model {} bytes from {}", b.len, MODEL_PATH),
        None => serial_println!("glados: no model at {}", MODEL_PATH),
    }
    match &tokenizer {
        Some(b) => serial_println!("glados: tokenizer {} bytes from {}", b.len, TOKENIZER_PATH),
        None => serial_println!("glados: no tokenizer at {}", TOKENIZER_PATH),
    }

    // --- Memory map, then leave the firmware behind ---
    let mut map_size: usize = 0;
    let mut map_key: usize = 0;
    let mut desc_size: usize = 0;
    let mut desc_ver: u32 = 0;

    // First call fails with BUFFER_TOO_SMALL and fills in the required size.
    (bs.get_memory_map)(
        &mut map_size,
        ptr::null_mut(),
        &mut map_key,
        &mut desc_size,
        &mut desc_ver,
    );

    // Slack, because allocate_pool below perturbs the very map we just sized.
    map_size += desc_size * 16;

    let mut buf: *mut u8 = ptr::null_mut();
    if is_error((bs.allocate_pool)(MemoryType::LoaderData, map_size, &mut buf)) {
        con_out(st, "glados: allocate_pool for memory map failed\r\n");
        halt();
    }

    // ExitBootServices rejects a stale map key, and re-reading the map can
    // itself change it. Retry, without allocating in between.
    let mut attempts = 0;
    let final_size = loop {
        let mut sz = map_size;
        let s = (bs.get_memory_map)(
            &mut sz,
            buf,
            &mut map_key,
            &mut desc_size,
            &mut desc_ver,
        );
        if is_error(s) {
            con_out(st, "glados: get_memory_map failed\r\n");
            halt();
        }

        if (bs.exit_boot_services)(image, map_key) == SUCCESS {
            break sz;
        }

        attempts += 1;
        if attempts > 8 {
            con_out(st, "glados: exit_boot_services kept failing\r\n");
            halt();
        }
    };

    // ---------------------------------------------------------------
    // Past this line the firmware is gone. No boot services, no con_out,
    // no protocols. Serial and the framebuffer are all we have.
    // ---------------------------------------------------------------
    serial_println!("glados: exited boot services after {} retries", attempts);

    let boot = BootInfo {
        fb,
        rsdp,
        mmap: buf,
        mmap_size: final_size,
        desc_size,
        fb_start,
        fb_end,
        model,
        tokenizer,
        roots,
    };

    console::init(boot.fb, 2, palette::BLACK);
    gfx::set_primary(boot.fb);

    // Replace the firmware's descriptor tables with ours. Until the IDT is in,
    // any fault is a triple fault: an instant reboot with no diagnostic.
    cpu::gdt::init();
    cpu::idt::init();
    kprintln!("[boot] gdt + idt installed");

    // Before any floating point runs. Detection alone is not enough: AVX
    // instructions fault with #UD until the OS sets CR4.OSXSAVE and declares
    // the wider register state in XCR0.
    let simd = cpu::enable_simd();
    kprintln!(
        "[boot] simd  sse2={} sse4.1={} avx={} avx2={} fma={}  avx enabled={}",
        simd.sse2 as u8,
        simd.sse41 as u8,
        simd.avx as u8,
        simd.avx2 as u8,
        simd.fma as u8,
        simd.avx_enabled as u8
    );

    // One allocator for the whole early bring-up: page tables first, then the
    // heap. Sharing it means the heap can never be handed frames that paging
    // already took.
    let mut frames = unsafe {
        mem::frame::EarlyFrames::new(boot.mmap, boot.mmap_size, boot.desc_size)
    };

    install_paging(&boot, &mut frames);
    init_heap(&mut frames);

    let acpi = unsafe { acpi::parse(boot.rsdp) };

    banner(&boot, &acpi);
    init_interrupts(&acpi);
    init_keyboard(&acpi);
    selftest();

    // Adopt the current thread of execution as task 0, then give it company.
    task::init("shell");
    console::set_color(YELLOW);
    kprintln!("\n[tasks]");
    console::set_color(LTGRAY_IDX);
    match task::spawn("clock", clock_task) {
        Some(i) => kprintln!("  spawned '{}' as task {}", "clock", i),
        None => kprintln!("  could not spawn the clock task"),
    }
    task::enable();
    kprintln!("  preemption enabled at {} Hz", TIMER_HZ);

    // Storage comes up before anything is restored, and the recovery console
    // gets its chance before that too. Ordering is the whole point: a repair
    // tool that only runs after a successful restore is useless on the day the
    // restore is what is broken.
    // Networking needs the same ECAM window storage does, and nothing later
    // depends on it -- so a machine with no supported NIC just reports that
    // and carries on.
    if let Some(ecam) = acpi.as_ref().and_then(|a| a.mcfg) {
        net::init(ecam, boot.roots.as_ref().map(|b| b.as_slice()));
    }

    let damaged = init_storage(&acpi);
    let restore = match recovery::maybe_enter(damaged) {
        recovery::Outcome::Continue => true,
        recovery::Outcome::SkipRestore => {
            console::set_color(YELLOW);
            kprintln!("[boot] persistent state will not be restored this boot");
            console::set_color(LTGRAY_IDX);
            false
        }
    };

    // The namespace exists whether or not there is a disk; a store only lets it
    // outlive a reboot. Restoring is skipped when the recovery console said so,
    // because "the last snapshot is what broke it" has to be a recoverable
    // situation.
    sysbox::init();
    if restore {
        sysbox::restore_latest();
    }

    // After storage, so a future version can pull weights out of the store
    // rather than off the ESP.
    ai::init(boot.model, boot.tokenizer);

    // The model becomes a resident task rather than a blocking command. This
    // has to come after ai::init: the task starts running as soon as it is
    // spawned, and it expects an engine to exist.
    if ai::engine_ready() && ai::spawn_mind() {
        kprintln!("  mind spawned -- 'think <prompt>' runs in the background");
    }

    shell::run(&boot, &acpi);
}

/// Bring up NVMe and attach to an existing checkpoint store, if there is one.
///
/// Returns true if the store exists but does not verify, which is the
/// condition that forces the recovery console open without being asked.
fn init_storage(acpi: &Option<acpi::Acpi>) -> bool {
    console::set_color(YELLOW);
    kprintln!("\n[storage]");
    console::set_color(LTGRAY_IDX);

    let Some(ecam) = acpi.as_ref().and_then(|a| a.mcfg) else {
        kprintln!("  no ECAM, so no PCIe enumeration");
        return false;
    };

    match dev::nvme::init(ecam) {
        Ok(()) => {
            dev::nvme::with(|n| {
                kprintln!(
                    "  nvme {} blocks x {} B = {} MiB",
                    n.block_count,
                    n.block_size,
                    n.capacity_bytes() / (1024 * 1024)
                );
            });
        }
        Err(e) => {
            kprintln!("  no NVMe controller ({:?})", e);
            return false;
        }
    }

    // The store's location is derived, not remembered: a partition tagged with
    // the GLaDOS type GUID if one exists, otherwise unclaimed space. So
    // mounting needs nothing recorded anywhere else. Read-only -- mounting
    // never writes.
    match store::cas::find_store_region(store::MIN_REGION_BLOCKS) {
        Some((start, _)) => match store::mount(start) {
            Ok(()) => {
                let mut bad = false;
                store::with(|s| {
                    kprintln!("  store at lba {}, seq {}, {} commits", start, s.sb.seq, s.sb.checkpoints);
                    // Cheap integrity probe: the root manifest must be
                    // readable and must match its own hash.
                    if !s.sb.root.is_none() && s.read_manifest(&s.sb.root).is_err() {
                        bad = true;
                    }
                });
                bad
            }
            Err(_) => {
                kprintln!("  no store here yet ('store init' to create one)");
                false
            }
        },
        None => {
            kprintln!("  no unclaimed space for a store on this disk");
            false
        }
    }
}

static CLOCK_ITERS: AtomicU64 = AtomicU64::new(0);

pub fn clock_iterations() -> u64 {
    CLOCK_ITERS.load(Ordering::Relaxed)
}

/// A second task, deliberately CPU-bound.
///
/// It never yields, never sleeps and never blocks -- so if the shell stays
/// responsive while this runs, that is preemption doing it and nothing else.
/// The iteration counter is the headless proof: it can only advance while this
/// task holds the CPU.
fn clock_task() {
    let mut last = u64::MAX;
    loop {
        CLOCK_ITERS.fetch_add(1, Ordering::Relaxed);

        // Continuously verify that this task's AVX registers survive being
        // preempted while the shell runs entirely different floating point
        // work. Counts failures rather than reporting them, so `tasks` shows
        // an ongoing verdict instead of a one-off test result.
        ai::fpu_guard(1000.0);

        // Only raises a flag; the shell does the writing. See
        // `sysbox::autosnap_poll` for why it cannot happen here.
        sysbox::autosnap_tick();

        let tenths = dev::lapic::ticks() * 10 / TIMER_HZ as u64;
        if tenths != last {
            last = tenths;
            if let Some(fb) = gfx::primary() {
                let text = alloc::format!(
                    " up {}.{}s  switches {} ",
                    tenths / 10,
                    tenths % 10,
                    task::total_switches()
                );
                let width = text.len() as u32 * gfx::font::GLYPH_W * 2;
                let x = fb.width().saturating_sub(width + 8);
                fb.draw_text(x, 4, &text, palette::YELLOW, palette::BLUE, 2);
            }
        }
        core::hint::spin_loop();
    }
}

/// Silence the PIC, bring up the local APIC, and start the periodic timer.
fn init_interrupts(acpi: &Option<acpi::Acpi>) {
    console::set_color(YELLOW);
    kprintln!("\n[apic]");
    console::set_color(LTGRAY_IDX);

    let Some(a) = acpi else {
        console::set_color(LTRED);
        kprintln!("  no ACPI -- cannot locate the APIC, staying on polling only");
        console::set_color(LTGRAY_IDX);
        return;
    };

    // Order matters: silence the PIC before enabling anything that could
    // deliver, or a stray legacy IRQ arrives on an exception vector.
    dev::pic::disable();
    kprintln!("  8259 remapped to 0x30 and fully masked");

    dev::lapic::init(a.lapic_addr);
    kprintln!("  lapic enabled, id {}", dev::lapic::id());

    if let Some(io) = a.primary_ioapic() {
        dev::ioapic::mask_all(&io);
        kprintln!(
            "  ioapic {} masked, {} redirection entries",
            io.id,
            dev::ioapic::max_redirection_entries(&io)
        );
    }

    // The 8254 PIT is the traditional reference but is not guaranteed present
    // on modern chipsets. The ACPI PM timer is: fixed at 3.579545 MHz, and its
    // port comes straight out of the FADT.
    let mut hz = dev::lapic::calibrate();
    let mut source = "PIT";
    if hz == 0 {
        console::set_color(YELLOW);
        kprintln!("  PIT did not answer");
        console::set_color(LTGRAY_IDX);
        if let Some(port) = a.pm_timer {
            hz = dev::lapic::calibrate_pm(port as u16);
            source = "ACPI PM timer";
        }
    }
    if hz == 0 {
        console::set_color(LTRED);
        kprintln!("  timer calibration FAILED -- neither the PIT nor the PM timer responded");
        console::set_color(LTGRAY_IDX);
        return;
    }
    kprintln!("  apic timer {} Hz measured against the {}", hz, source);

    if dev::lapic::start_timer(TIMER_HZ) {
        cpu::enable_interrupts();
        kprintln!("  timer running at {} Hz, interrupts enabled", TIMER_HZ);

        // Needs the timer already ticking and interrupts on, so it cannot move
        // any earlier than this.
        time::calibrate();
        if time::is_calibrated() {
            kprintln!("  tsc {} MHz", time::tsc_mhz());
        } else {
            kprintln!("  tsc not calibrated -- console pacing disabled");
        }
    } else {
        console::set_color(LTRED);
        kprintln!("  could not program the timer");
        console::set_color(LTGRAY_IDX);
    }
}

/// Scheduler tick rate. 100 Hz is a 10 ms quantum -- responsive enough for a
/// shell without spending the machine's time in the timer handler.
pub const TIMER_HZ: u32 = 100;

/// Bring up the i8042 and route its IRQ.
fn init_keyboard(acpi: &Option<acpi::Acpi>) {
    console::set_color(YELLOW);
    kprintln!("\n[i8042]");
    console::set_color(LTGRAY_IDX);

    let Some(a) = acpi else {
        console::set_color(LTRED);
        kprintln!("  no ACPI -- cannot route IRQ 1");
        console::set_color(LTGRAY_IDX);
        return;
    };

    let report = dev::kbd::init(a, dev::lapic::id());

    match report.self_test {
        // 0x55 is the controller's pass code.
        Some(0x55) => kprintln!("  controller self-test passed (0x55)"),
        Some(other) => {
            console::set_color(LTRED);
            kprintln!("  controller self-test returned {:#04x}, expected 0x55", other);
            console::set_color(LTGRAY_IDX);
        }
        None => {
            console::set_color(LTRED);
            kprintln!("  controller did not answer -- no i8042 present?");
            console::set_color(LTGRAY_IDX);
        }
    }

    match report.config {
        Some(c) => kprintln!(
            "  config {:#04x}  irq1={}  translate={}",
            c,
            c & 1,
            (c >> 6) & 1
        ),
        None => kprintln!("  config unreadable"),
    }

    match report.routed_gsi {
        Some(gsi) => kprintln!("  irq1 routed via gsi {} to vector {:#04x}", gsi, dev::VECTOR_KEYBOARD),
        None => {
            console::set_color(LTRED);
            kprintln!("  FAILED to route irq1 through the ioapic");
            console::set_color(LTGRAY_IDX);
        }
    }
}

/// 64 MiB of kernel heap.
///
/// Grown twice, each time by a model. 4 MiB was ample until weights had to fit
/// at all; 16 MiB was enough until a 30-layer one arrived. SmolLM2's KV cache
/// alone is 23.8 MiB at seq 512, and its 49152-entry vocabulary costs a few
/// more in tokenizer tables -- about 29 MiB resident once the corpus and the
/// namespace are in, both of which live entirely in RAM.
///
/// The weights themselves are *not* here -- they stay in the LoaderData pool
/// the firmware filled, referenced in place.
const HEAP_PAGES: usize = 16384;

fn init_heap(frames: &mut mem::frame::EarlyFrames) {
    match frames.alloc_contiguous(HEAP_PAGES) {
        Some(base) => {
            let size = HEAP_PAGES * mem::PAGE_SIZE as usize;
            unsafe { mem::heap::HEAP.add_region(base as usize, size) };
            kprintln!(
                "[boot] heap {} KiB at {:#x}",
                size / 1024,
                base
            );
        }
        None => {
            console::set_color(LTRED);
            kprintln!("[boot] heap allocation FAILED -- no contiguous region");
            console::set_color(LTGRAY_IDX);
        }
    }
}

/// Build and install our own identity map, replacing the firmware's.
///
/// Failure here is survivable: UEFI's page tables are still loaded and still
/// correct, so we report and carry on rather than halting. Everything through
/// M4 works fine on the firmware's map -- we just do not own it.
fn install_paging(boot: &BootInfo, frames: &mut mem::frame::EarlyFrames) {
    let top = mem::frame::max_ram_address(boot.mmap, boot.mmap_size, boot.desc_size);

    // Belt and braces. The allowlist in max_ram_address should already keep
    // this sane, but a firmware map we have not seen must not be able to turn
    // into a multi-terabyte map build. 64 GiB is comfortably above this
    // board's 64 GiB maximum populated DRAM.
    const MAP_CEILING: u64 = 64 * mem::GIB;

    // Always cover the low 4 GiB: the legacy MMIO hole lives there, and so does
    // the framebuffer aperture on both QEMU and the Intel iGPU. Fold fb_end in
    // last so the aperture is covered even if it somehow sits above the clamp.
    let limit = top.max(4 * mem::GIB).min(MAP_CEILING).max(boot.fb_end);

    kprintln!(
        "[boot] mapping to {:#x} (ram top {:#x}, fb end {:#x})",
        limit,
        top,
        boot.fb_end
    );

    match mem::paging::build_identity_map(
        frames,
        limit,
        boot.mmap,
        boot.mmap_size,
        boot.desc_size,
        boot.fb_start,
        boot.fb_end,
    ) {
        Some(pml4) => {
            unsafe { mem::paging::activate(pml4) };
            // Reaching this line means the map covered our code, our stack and
            // the framebuffer -- if it had not, we would already be gone.
            kprintln!(
                "[boot] paging active  cr3={:#x}  mapped {} MiB  ({} frames)",
                cpu::read_cr3(),
                limit / (1024 * 1024),
                frames.allocated_frames()
            );
        }
        None => {
            console::set_color(LTRED);
            kprintln!("[boot] page table build FAILED, staying on firmware map");
            console::set_color(LTGRAY_IDX);
        }
    }
}

/// Prove the exception path works while we are still expecting it to.
fn selftest() {
    console::set_color(LTGREEN);
    kprintln!("\n[selftest] heap:");
    console::set_color(LTGRAY_IDX);
    {
        use alloc::format;
        use alloc::vec::Vec;

        // Pushing past capacity repeatedly forces grow-realloc-free cycles,
        // which is what actually exercises split and coalesce.
        let mut v: Vec<u64> = Vec::new();
        for i in 0..256u64 {
            v.push(i * i);
        }
        let s = format!("  vec len {}  v[255]={}  v[16]={}", v.len(), v[255], v[16]);
        kprintln!("{}", s);
        let (used, total) = mem::heap::HEAP.stats();
        kprintln!("  in use {} B of {} B", used, total);
    }
    // Everything above is dropped. If alloc and dealloc round sizes the same
    // way, this is exactly zero; any other number is a per-allocation leak.
    let (used, _) = mem::heap::HEAP.stats();
    if used == 0 {
        console::set_color(LTGREEN);
        kprintln!("  after drop: 0 B in use -- alloc/dealloc are exact inverses");
    } else {
        console::set_color(LTRED);
        kprintln!("  after drop: {} B LEAKED", used);
    }

    // The timer is the first thing in this kernel that runs without being
    // called. If ticks advance, the LAPIC, the IDT vector, the EOI path and
    // the calibration are all correct at once.
    console::set_color(LTGREEN);
    kprintln!("\n[selftest] timer:");
    console::set_color(LTGRAY_IDX);
    let start = dev::lapic::ticks();
    let want = start + TIMER_HZ as u64 / 2; // half a second
    let mut spins: u64 = 0;
    while dev::lapic::ticks() < want {
        spins += 1;
        if spins > 200_000_000 {
            break; // timer is dead; do not hang the boot waiting for it
        }
        core::hint::spin_loop();
    }
    let elapsed = dev::lapic::ticks() - start;
    if elapsed >= TIMER_HZ as u64 / 2 {
        console::set_color(LTGREEN);
        kprintln!("  {} ticks in ~0.5 s -- interrupts are firing", elapsed);
    } else {
        console::set_color(LTRED);
        kprintln!("  only {} ticks -- timer is not delivering", elapsed);
    }

    console::set_color(LTGREEN);
    kprintln!("\n[selftest] clock:");
    console::set_color(LTGRAY_IDX);
    if dev::rtc::selftest() {
        console::set_color(LTGREEN);
        kprintln!("  ok   calendar round-trips, including leap years and 2000");
    } else {
        console::set_color(LTRED);
        kprintln!("  FAIL calendar arithmetic is wrong");
    }
    console::set_color(LTGRAY_IDX);
    match dev::rtc::now() {
        Some(d) => kprintln!(
            "  now  {:04}-{:02}-{:02} {:02}:{:02}:{:02}",
            d.year, d.month, d.day, d.hour, d.minute, d.second
        ),
        None => kprintln!("  no usable RTC -- snapshots will record no time"),
    }

    console::set_color(LTGREEN);
    kprintln!("\n[selftest] sysbox namespace:");
    console::set_color(LTGRAY_IDX);
    sysbox::selftest();

    // The RFC vectors, at every boot. 25 ms, and it is the only thing standing
    // between a broken field arithmetic and a TLS handshake that fails with
    // nothing to point at -- crypto is the one place where wrong code still
    // produces perfectly plausible output.
    crypto::selftest();

    console::set_color(LTGREEN);
    kprintln!("\n[selftest] int3 should report and resume:");
    console::set_color(LTGRAY_IDX);
    unsafe {
        core::arch::asm!("int3", options(nomem, nostack));
    }

    console::set_color(LTGREEN);
    kprintln!("[selftest] survived int3 -- idt is live.");
    console::set_color(LTGRAY_IDX);

    // The deliberate null dereference now lives behind the shell's `fault`
    // command. It is fatal by design, so running it during boot would mean the
    // shell never starts.
}

fn banner(boot: &BootInfo, acpi: &Option<acpi::Acpi>) {
    console::set_color(LTCYAN);
    kprintln!("glados");
    console::set_color(WHITE);
    kprintln!("a ring-0 kernel for MSI MS-16R8\n");

    console::set_color(YELLOW);
    kprintln!("[boot]");
    console::set_color(WHITE);
    kprintln!(
        "  framebuffer {}x{}  stride {}  {:?}",
        boot.fb.width(),
        boot.fb.height(),
        boot.fb.stride(),
        boot.fb.format()
    );
    kprintln!("  acpi rsdp   {:?}", boot.rsdp);

    let (usable, regions) = survey_memory(boot);
    kprintln!(
        "  usable ram  {} MiB across {} regions",
        usable / (1024 * 1024),
        regions
    );

    let (used, total) = mem::heap::HEAP.stats();
    kprintln!("  heap        {} KiB free of {} KiB", (total - used) / 1024, total / 1024);

    console::set_color(YELLOW);
    kprintln!("\n[acpi]");
    console::set_color(WHITE);
    match acpi {
        Some(a) => {
            kprintln!("  revision    {}   cpus {}", a.revision, a.cpus);
            kprintln!("  lapic       {:#x}", a.lapic_addr);
            for i in 0..a.ioapic_count {
                let io = a.ioapics[i];
                kprintln!(
                    "  ioapic {}    {:#x}  gsi base {}",
                    io.id,
                    io.addr,
                    io.gsi_base
                );
            }
            kprintln!("  overrides   {}", a.override_count);
            let (kbd_gsi, _) = a.gsi_for_irq(1);
            kprintln!("  irq1 -> gsi {}   <-- keyboard", kbd_gsi);
            match a.hpet {
                Some(h) => kprintln!("  hpet        {:#x}", h),
                None => kprintln!("  hpet        absent"),
            }
            match a.mcfg {
                Some(m) => kprintln!("  pcie ecam   {:#x}", m),
                None => kprintln!("  pcie ecam   absent"),
            }
            match a.pm_timer {
                Some(t) => kprintln!("  pm timer    port {:#x}", t),
                None => kprintln!("  pm timer    absent"),
            }
        }
        None => {
            console::set_color(LTRED);
            kprintln!("  ACPI PARSE FAILED");
            console::set_color(WHITE);
        }
    }

    kprintln!("\n  boot services released, running on our own.");
    kprintln!(
        "  serial in   {}",
        if serial::is_present() { "COM1 answers -- shell reads it" } else { "no UART" }
    );

    // Pixel-format check. If Rgbx/Bgrx were misdetected, red and blue swap and
    // the bars below read blue-green-red instead. Faster to see than to reason
    // about.
    console::set_color(LTGREEN);
    kprintln!("\n[selftest] bars should read RED GREEN BLUE, left to right:");
    let y = boot.fb.height().saturating_sub(80);
    let w = boot.fb.width() / 6;
    boot.fb.rect(w, y, w, 40, palette::LTRED);
    boot.fb.rect(w * 2, y, w, 40, palette::LTGREEN);
    boot.fb.rect(w * 3, y, w, 40, palette::LTBLUE);

    console::set_color(LTGRAY_IDX);
}

const LTGRAY_IDX: u8 = 7;

/// Total bytes usable after boot services are gone, and the region count.
fn survey_memory(boot: &BootInfo) -> (u64, usize) {
    let mut total = 0u64;
    let mut count = 0usize;
    let n = boot.mmap_size / boot.desc_size;
    for i in 0..n {
        // Stride by the firmware's descriptor_size, not by size_of.
        let d = unsafe { &*(boot.mmap.add(i * boot.desc_size) as *const MemoryDescriptor) };
        if d.is_usable_after_exit() {
            total += d.num_pages * 4096;
            count += 1;
        }
    }
    (total, count)
}

fn find_rsdp(st: &SystemTable) -> *const c_void {
    let mut acpi1: *const c_void = ptr::null();
    for i in 0..st.number_of_table_entries {
        let e = unsafe { &*st.configuration_table.add(i) };
        if e.vendor_guid == ACPI_20_TABLE_GUID {
            return e.vendor_table; // ACPI 2.0+ preferred: it has the XSDT.
        }
        if e.vendor_guid == ACPI_10_TABLE_GUID {
            acpi1 = e.vendor_table;
        }
    }
    acpi1
}

/// Firmware text console. Only valid *before* `ExitBootServices`.
fn con_out(st: &mut SystemTable, s: &str) {
    let mut buf = [0u16; 160];
    let mut i = 0;
    for ch in s.chars() {
        if i >= buf.len() - 1 {
            break;
        }
        buf[i] = ch as u16;
        i += 1;
    }
    buf[i] = 0;
    unsafe {
        ((*st.con_out).output_string)(st.con_out, buf.as_ptr());
    }
}

fn halt() -> ! {
    cpu::halt()
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    serial_println!("\n*** PANIC *** {}", info);
    if console::is_ready() {
        console::set_color(LTRED);
        console::_print(format_args!("\n*** KERNEL PANIC ***\n{}\n", info));
    }
    halt()
}
