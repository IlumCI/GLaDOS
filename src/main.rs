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
mod bench;
mod boot_report;
mod repair;
mod ai;
mod app;
mod cpu;
mod crypto;
mod diag;
/// DOOM, ported. Not our code -- see the module doc.
mod doom;
mod dev;
mod edit;
mod fmt;
mod gfx;
mod gpu;
mod json;
mod aiksi;
mod linux;
mod log;
mod mem;
mod mine;
mod net;
mod pkg;
/// What a program written somewhere else may ask of this machine.
mod port;
mod recovery;
mod rng;
mod serial;
mod shell;
mod sky;
mod smp;
mod store;
mod sync;
mod sysbox;
mod task;
mod time;
mod uefi;
mod update;

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
    /// A DOOM WAD, if one was on the boot volume. `None` is the ordinary
    /// case: this is not a machine that needs one to run.
    pub wad: Option<Blob>,
    /// A DER bundle of root certificates. `None` means TLS can encrypt and
    /// cannot authenticate, which is reported rather than assumed.
    pub roots: Option<Blob>,
}

/// Where the weights live on the boot volume. Backslashes: this is a UEFI path
/// on the ESP, not a namespace path.
pub const MODEL_PATH: &str = "\\GLADOS\\model.bin";
pub const TOKENIZER_PATH: &str = "\\GLADOS\\tokenizer.bin";
/// A WAD, if the operator put one there. Absent is the ordinary case.
///
/// Read at boot for the reason the model is: this is the only moment there is
/// a filesystem, and `read_file` leaves it in a firmware pool that nothing
/// frees, so a four-megabyte IWAD costs the heap nothing and can be sliced
/// rather than copied for the life of the machine.
pub const WAD_PATH: &str = "\\GLADOS\\DOOM.WAD";

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

    // Where the firmware put us. Every fault RIP is meaningless without this:
    // RIP minus the base is an offset into the binary sitting in target/, and
    // a disassembly of that names the faulting function.
    //
    // The Loaded Image protocol is asked first, but it cannot be trusted on
    // every path: one firmware reported revision fine and image_base as zero,
    // so the answer is confirmed against the image itself -- walking page-
    // aligned addresses down from our own entry point until one carries an
    // MZ header whose PE signature lands where e_lfanew says. Identity
    // mapping makes the read legal; 64 MiB of scan bounds a corrupt map.
    let mut li_ptr: *mut c_void = ptr::null_mut();
    let mut base = 0u64;
    if !is_error((bs.handle_protocol)(image, &LOADED_IMAGE_PROTOCOL_GUID, &mut li_ptr))
        && !li_ptr.is_null()
    {
        let li = unsafe { &*(li_ptr as *mut LoadedImageProtocol) };
        serial_println!(
            "glados: loaded_image reports base {:#x} size {:#x} rev {:#x}",
            li.image_base,
            li.image_size,
            li.revision
        );
        base = li.image_base;
        cpu::idt::IMAGE_SIZE.store(li.image_size, Ordering::Relaxed);
    }
    if base == 0 {
        let mut probe = efi_main as usize & !0xFFF;
        for _ in 0..(64 * 1024 * 1024 / 0x1000) {
            unsafe {
                if *(probe as *const u16) == 0x5A4D {
                    let lfanew = *((probe + 0x3C) as *const u32) as usize;
                    if lfanew < 0x400 && *((probe + lfanew) as *const u32) == 0x0000_4550 {
                        base = probe as u64;
                        break;
                    }
                }
            }
            probe -= 0x1000;
        }
    }
    cpu::idt::IMAGE_BASE.store(base, Ordering::Relaxed);
    serial_println!("glados: image base {:#x}", base);

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
    // Keep the runtime table before the boot-services era closes. It is what
    // `reboot` and `shutdown` call, and after ExitBootServices there is no
    // other way to reach the firmware.
    cpu::set_runtime(unsafe { (*st).runtime_services });

    // A heap for the update hook, which needs one before there is one.
    //
    // `init_heap` runs *after* `ExitBootServices`, and the hook has to run
    // before it -- the ESP is only writable while the firmware's FAT driver
    // exists. Everything else the hook does is stack work, but verifying a
    // P-256 signature is big-integer arithmetic and allocates, so the first
    // real end-to-end test of an update died at
    // `memory allocation of 32 bytes failed` before it reached the swap.
    //
    // A static arena rather than `allocate_pool`: `EarlyFrames` builds the
    // real heap from Conventional memory only, and this lives in the loaded
    // image, so the two cannot hand out the same page. Pool memory would be
    // `LoaderData`, and whether that is safe depends on a memory-type filter
    // several files away agreeing with an assumption made here.
    //
    // It is not wasted afterwards. `add_region` is additive, so this stays in
    // the free list and `init_heap` grows the heap around it.
    const EARLY_ARENA_LEN: usize = 1 << 20;
    static mut EARLY_ARENA: [u8; EARLY_ARENA_LEN] = [0; EARLY_ARENA_LEN];
    unsafe {
        let base = core::ptr::addr_of_mut!(EARLY_ARENA) as usize;
        mem::heap::HEAP.add_region(base, EARLY_ARENA_LEN);
    }

    // --- a staged update, while the ESP is still writable ---
    //
    // Before the model, because applying one ends in a reboot and there is no
    // sense reading 570 MB of weights first. After `set_runtime`, because that
    // reboot goes through the runtime table.
    //
    // Inert unless somebody has staged an update *and* provisioned a signing
    // key: with `UPDATE_KEY` zeroed, `verify` answers `NoKey` and the decision
    // refuses. The mechanism ships built, tested and unable to fire.
    let staged = update::hook(bs, image);
    if let update::Outcome::Said(line) | update::Outcome::Armed(line) = staged {
        serial_println!("glados: {}", line);
        con_out(st, "glados: ");
        con_out(st, line);
        con_out(st, "
");
    }

    // --- repairs this machine decided for itself on an earlier boot ---
    //
    // Here because it is the earliest point there is: the ESP is readable, and
    // every subsystem a repair could be protecting initialises later. A repair
    // adopted after `power` has already faulted is a repair that arrives one
    // boot late, which is exactly what persisting it is for.
    //
    // Nothing the file says is executed. Two words are resolved against
    // `repair::ACTIONS`, the row is what runs, and a line naming an action that
    // does not exist -- or aiming a narrow one at a subsystem it was never
    // offered for -- gets nothing.
    let (persisted, repair_note) = update::repairs::at_boot(bs, image);
    if let Some(line) = &repair_note {
        serial_println!("glados: {}", line);
        con_out(st, "glados: ");
        con_out(st, line);
        con_out(st, "
");
    }
    for e in &persisted {
        match repair::apply_named(&e.subsystem, &e.action) {
            Some((sub, act)) => {
                repair::note_from_disk(sub, act);
                serial_println!("glados: repair '{}' for {}", act, sub);
            }
            None => serial_println!(
                "glados: the boot volume asks for '{}' for {}, which is not a repair this kernel has",
                e.action,
                e.subsystem
            ),
        }
    }

    // Beside the model, and for the same reason. `None` here is not a
    // failure: most machines have no WAD and boot exactly as before.
    let wad = uefi::read_file(bs, image, WAD_PATH);
    match &wad {
        Some(b) => serial_println!("glados: wad {} bytes from {}", b.len, WAD_PATH),
        None => serial_println!("glados: no wad at {}", WAD_PATH),
    }

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

    // The image got here, which is the whole of what this hook can watch: it
    // ran, it drew, and it read every file it needs off the ESP. Past the
    // memory map there is no filesystem to record anything in, so this is the
    // last moment a trial can be resolved.
    // Not when a trial was armed a moment ago: that trial belongs to the image
    // that boots next, and clearing it here would pass a verdict on a run that
    // has not happened -- an update accepted by the image it replaces.
    if !matches!(staged, update::Outcome::Armed(_)) {
        update::mark_healthy(bs, image);
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
        wad,
        roots,
    };

    console::init(boot.fb, 2, palette::BLACK);
    gfx::set_primary(boot.fb);

    // From here until `finish`, the console writes to its shadow grid without
    // painting. Nothing is lost -- the whole log is repainted at the end -- and
    // the slow parts of boot get a progress bar instead of a blank panel.
    gfx::splash::begin();

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

    gfx::splash::stage("memory map and page tables");
    install_paging(&boot, &mut frames);
    init_heap(&mut frames);

    // Record what is left, while the firmware's map is still readable and
    // while the one allocator that took anything from it is still in scope.
    // After this the map is never consulted again, and `mem::fixed` is the
    // only thing that can say whether a fixed-address image is placeable.
    //
    // Refused rather than approximated when the allocator lost a handout: a
    // free set that is missing a taken range would place a guest on top of the
    // page tables, and the symptom would be the machine rewriting its own
    // translations while a program runs.
    match frames.handouts() {
        Some(taken) => unsafe {
            mem::fixed::snapshot(boot.mmap, boot.mmap_size, boot.desc_size, taken)
        },
        None => kprintln!("[boot] placement table skipped: the early allocator lost a handout"),
    }
    let (free, run) = mem::fixed::totals();
    kprintln!(
        "[boot] placeable  {} MiB free below the heap and above it, largest run {} MiB",
        free / 1024 / 1024,
        run / 1024 / 1024,
    );

    let acpi = unsafe { acpi::parse(boot.rsdp) };

    banner(&boot, &acpi);
    gfx::splash::stage("interrupts and keyboard");
    init_interrupts(&acpi);
    init_smp(&acpi);
    init_keyboard(&acpi);
    gfx::splash::stage("self-test");
    // **The window in which a panicking selftest is survivable**, opened here
    // and shut on the next line. A check that asserts its way out has said its
    // subsystem is broken, which is information; a panic anywhere else in this
    // kernel still halts, which is why the window is two lines wide and not a
    // policy.
    cpu::recover::selftest_window(true);
    selftest(&acpi);
    cpu::recover::selftest_window(false);

    // Adopt the current thread of execution as task 0, then give it company.
    gfx::splash::stage("scheduler");
    task::init("shell");
    console::set_color(YELLOW);
    kprintln!("\n[tasks]");
    console::set_color(LTGRAY_IDX);
    match task::spawn("clock", clock_task) {
        Some(i) => kprintln!("  spawned '{}' as task {}", "clock", i),
        None => kprintln!("  could not spawn the clock task"),
    }
    // The thing that owns the frame.
    //
    // Painting used to be push-model with no owner: sixteen scattered
    // `desk::draw()` calls, plus the shell's idle loop, so a task inside a long
    // command painted nothing and the screen stopped. Measured, `diag all`:
    // one frame in seventy-three seconds and the screen still for fifty-nine
    // of them, while the clock task painted a hundred and eighty-seven times
    // beside it. That contrast is the whole diagnosis -- the machine was
    // running, and nobody was responsible for the picture.
    match task::spawn("comp", comp_task) {
        Some(i) => {
            // Told, not guessed. The watchdog reports what the scheduler makes
            // of this task when it stops beating, and `task::current()` cannot
            // answer who it is: 0 means both task 0 and an idle core.
            gfx::render::watching(i);
            kprintln!("  spawned '{}' as task {}", "comp", i);
        }
        None => kprintln!("  could not spawn the compositor task"),
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
    gfx::splash::stage("network");
    if let Some(ecam) = acpi.as_ref().and_then(|a| a.mcfg) {
        net::init(ecam, boot.roots.as_ref().map(|b| b.as_slice()));
    }

    // After the network, deliberately: bringing the controller up resets the
    // bus, and if a USB Ethernet adapter is going to claim a device it should
    // do so before anything else walks past it. Ports it took are skipped
    // here rather than enumerated a second time.
    if let Some(ecam) = acpi.as_ref().and_then(|a| a.mcfg) {
        match dev::usbhid::probe(ecam) {
            Ok(0) => {}
            Ok(n) => kprintln!("[boot] usb    {} input device(s) on the boot protocol", n),
            Err(e) => kprintln!("[boot] usb    no input: {}", e),
        }
    }

    gfx::splash::stage("storage");
    let damaged = init_storage(&acpi);
    // The recovery prompt is a question, and it is being asked of a console
    // that is not currently on screen.
    gfx::splash::note("hold ESC or R for the recovery console");
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
    gfx::splash::stage("namespace");
    sysbox::init();
    if restore {
        sysbox::restore_latest();
    }

    // A ported program's data, filed where `port::files` can find it. After
    // the heap, because the registry is a `Vec`; before the shell, because a
    // command asking for it must not race the registration.
    //
    // The bytes are not copied. `Blob::as_slice` hands out the firmware pool
    // allocation itself, which nothing frees, so a four-megabyte IWAD is
    // borrowed for the life of the machine and costs the heap one pointer.
    if let Some(b) = boot.wad {
        port::files::provide(doom::WAD_FILE, b.as_slice());
    }

    // After storage, so a future version can pull weights out of the store
    // rather than off the ESP.
    gfx::splash::stage("loading the model");
    ai::init(boot.model, boot.tokenizer);

    // The model becomes a resident task rather than a blocking command. This
    // has to come after ai::init: the task starts running as soon as it is
    // spawned, and it expects an engine to exist.
    if ai::engine_ready() && ai::spawn_mind() {
        kprintln!("  mind spawned -- 'think <prompt>' runs in the background");
    }
    // The agent task is resident for the same reason; it does nothing until
    // an episode is queued, and shares the mind's engine-ownership rule.
    if ai::engine_ready() && ai::spawn_agent() {
        kprintln!("  agent ready -- 'agent <goal>' queues an episode");
    }
    // The initiative loop is the resident mind: it perceives, journals, and
    // occasionally gives itself a small read-only goal. It stands down while
    // the operator is present; 'initiative off' quiets it entirely.
    if ai::engine_ready() && ai::initiative::spawn() {
        kprintln!("  initiative resident -- the machine thinks between your commands");
    }

    // Try to fix what broke, before saying what is missing -- so the summary
    // reports the state the machine is actually in rather than the one it was
    // in a moment ago.
    repair::attempt_all();

    // And the other direction: a repair still applied to a subsystem that has
    // started passing without it is a workaround that outlived its bug.
    repair::recheck_persisted();

    // Said here rather than only where it happened: by now the fault itself
    // has scrolled past a hundred ok lines, and the line that matters is
    // "this machine is running without X".
    boot_report::report();

    // The repairs applied at the hook did not stop this boot, which is the
    // whole of what the trial flag asks. Deliberately here rather than beside
    // `update::mark_healthy`: that one is bounded by `ExitBootServices` because
    // it guards a boot image, and this one is not, so it covers the selftests,
    // storage, the model and the desktop instead of none of them.
    update::repairs::survived();

    gfx::splash::stage("ready");
    gfx::splash::finish();

    // Everything up to here was the machine reporting on itself and belongs
    // on the executive console. From here the operator is driving, and what
    // they type and what it answers belongs on theirs.
    gfx::console::set_default_channel(gfx::console::USER);
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
                // What one command can move. It was two pages, hardcoded,
                // because MDTS was never read and a PRP list was never built.
                kprintln!(
                    "  max transfer {} KiB per command",
                    n.max_transfer_blocks * n.block_size / 1024
                );
            });
        }
        Err(e) => {
            kprintln!("  no NVMe controller ({:?})", e);
            return false;
        }
    }

    // Late on purpose: `repair::attempt_all` decided this before the
    // controller existed, because the boot summary has to describe the machine
    // as it now is. This is the first moment there is anywhere to write.
    repair::persist_adopted();

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
/// Compose a frame when one is owed, and never mind who asked.
///
/// The loop is deliberately dull. It does not know what changed, only that
/// something did, and `desk::draw` repaints everything anyway -- total repaint
/// is what makes the window manager obviously correct, and `compose::present`
/// is what makes it cheap, writing only the rows that actually differ.
///
/// **The flag is taken before the frame, not after.** A change arriving while
/// a frame is being composed has to survive that frame: taking it afterwards
/// would clear a request that came in halfway through and leave the screen one
/// update behind, which is the shape of bug that is invisible until somebody
/// moves a window during a long paint.
/// How often a frame may be composed, at most.
///
/// Sixty, because a pointer drag and a window move are what a person actually
/// watches and thirty reads as stepping on those. The frame is 2,143 us
/// measured (`video bench`, one core), so this is about 13% of a core while
/// something is continuously changing and **nothing at all when it is not** --
/// a clean frame composes no pixels, so the cost is the deadline check.
const FRAME_HZ: u64 = 60;

fn comp_task() {
    // `rdtsc` and not `lapic::ticks()`. That counter is incremented by the
    // bootstrap processor's timer and is right for wall-clock-ish elapsed
    // time, but this tree has already paid once for deriving a *duration* from
    // it -- every network timeout was short by the core count and the check
    // that should have caught it divided the error straight back out.
    // Anything measuring an interval uses the TSC.
    let mut next = time::rdtsc();
    loop {
        // **The heartbeat, at the top and unconditionally.**
        //
        // Every path below this line either yields or composes, so a beat here
        // means one full turn of the loop happened whatever branch it took --
        // including `render off` and a full-screen program owning the screen,
        // neither of which is a fault and neither of which should raise an
        // alarm. The clock task reads it; see `render::watch`.
        gfx::render::beat(gfx::render::Phase::Turn);

        // Standing it down is what the machine did before this task existed,
        // which is how the before-number is taken without a second build.
        if !gfx::render::enabled() {
            task::yield_now();
            continue;
        }

        // Ops first, and before the deadline check.
        //
        // These come from tasks that do not own the desktop, and applying them
        // here is the whole point: this is the one moment nothing else is
        // looking at the window list. Draining before composing means a window
        // asked for during the last frame appears in this one rather than the
        // next.
        if !gfx::exclusive() {
            // The phase before each call that leaves this file, so a stall
            // names the thing it is stuck inside rather than the loop.
            gfx::render::beat(gfx::render::Phase::Ops);
            gfx::desk::drain_ops();
            // **The pointer, every loop rather than every frame.**
            //
            // This used to run from the shell's idle loop, so it stopped the
            // moment the shell entered a command -- which is why `pump_cursor`
            // existed at all: a motion-only stand-in bolted onto the clock task
            // so the arrow would keep moving while the real handler was
            // unreachable. The compositor runs whatever the shell is doing, so
            // the stand-in is not needed and is gone.
            //
            // Before the deadline check on purpose. Composing is rate-limited
            // because a frame is expensive; reading the mouse is not, and
            // capping it at 60 Hz would add up to sixteen milliseconds of lag
            // to every click for no saving.
            //
            // Outside the frame's borrow, also on purpose: a press can open a
            // window, start an app, or run an Aiksi program under DRAW_BUDGET.
            // That work is allowed to overrun and make the next frame late; it
            // is not allowed to happen underneath one.
            gfx::render::beat(gfx::render::Phase::Pointer);
            gfx::desk::poll_mouse();
        }

        // **Back to `Turn` before the wait, or the resting state lies.**
        //
        // Most turns of this loop end at the deadline check below, and with no
        // beat here the last phase announced was whatever ran before it -- so
        // a perfectly healthy idle compositor reported "reading the pointer",
        // and so did one genuinely stuck inside `poll_mouse`. Those are a
        // non-event and a hung input path, and the whole value of recording a
        // phase is telling them apart. Measured: a healthy loop reported
        // `Pointer` 13 ms ago, and a 43-second stall reported `Pointer` too.
        gfx::render::beat(gfx::render::Phase::Turn);

        let now = time::rdtsc();
        if now < next {
            // **`yield_now` and deliberately not `hlt`.**
            //
            // `hlt` is the obvious way to wait for a deadline and it is wrong
            // here, because this scheduler has no blocked state: `State` is
            // `Unused`, `Ready` and `Running`, and a task cannot stop being
            // runnable. So `hlt` halts *the core*, not this task -- and the
            // next thing to run does not run until an interrupt arrives, which
            // at 100 Hz is up to ten milliseconds. That is ten milliseconds
            // taken from the shell, once per round trip, during exactly the
            // long foreground commands this whole change exists to fix.
            //
            // Yielding costs a `rdtsc` and a compare per visit and hands the
            // core to whoever is actually ready. Real sleeping needs a blocked
            // state and a wake from `render::invalidate`, which is surgery on
            // the most delicate loop in the tree and buys nothing while there
            // is always a shell wanting the core.
            task::yield_now();
            continue;
        }
        let mhz = time::tsc_mhz().max(1) as u64;
        next = now + (mhz * 1_000_000) / FRAME_HZ;

        // `render stall` lands here, and nowhere in a shipped path.
        gfx::render::stall_hook();

        // A full-screen program owns the screen outright -- DOOM, the editor,
        // a guest holding /dev/fb0 -- so the desktop stands down rather than
        // contending. The repaint is owed for when it gives the screen back,
        // so the flag goes back too.
        if gfx::exclusive() {
            if gfx::render::take_dirty() {
                gfx::render::restore_dirty();
            }
            task::yield_now();
            continue;
        }
        let mut composed = false;
        if gfx::render::take_dirty() {
            // `draw` takes the painter's claim itself and refuses if another
            // task holds it, which is the right answer: that task is painting
            // the same desktop. What must not happen is losing the request, so
            // it goes back if the frame did not land.
            let before = gfx::render::stats().draws;
            gfx::render::beat(gfx::render::Phase::Frame);
            gfx::desk::draw();
            if gfx::render::stats().draws == before {
                gfx::render::restore_dirty();
            } else {
                composed = true;
            }
        }

        // The taskbar's two readouts, after the frame rather than before it.
        //
        // `draw` paints the *well* they sit in and not the text inside it, so
        // a composed frame erases them and they have to go back on top. Doing
        // it in the other order leaves the tray blank until the next tenth of
        // a second ticks, which is the flicker that had no name while the
        // clock task owned this and could only repaint on its own schedule.
        gfx::render::beat(gfx::render::Phase::Tray);
        gfx::desk::paint_tray(composed);
    }
}

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

        // The wireless state machine, so a scan keeps moving while the shell
        // is inside a long command. `wifi_poll` and not `wifi_service`: the
        // second one draws, and the compositor's back buffer belongs to the
        // shell's task. Claimed against the idle loop, which calls it too.
        net::wifi_poll();

        let tenths = dev::lapic::ticks() * 10 / TIMER_HZ as u64;
        if tenths != last {
            let crossed_second = tenths / 10 != last / 10;
            last = tenths;
            // Once a second, record the machine's state for the Oracle to fit
            // its self-prediction from. Here rather than on demand because a
            // future is only projectable from a history, and the history has
            // to have been accruing before anyone asks.
            if crossed_second {
                // **Somebody has to notice the screen's owner is gone, and it
                // cannot be the owner.**
                //
                // Every painter but this one now goes through the compositor,
                // so if that task stops the machine keeps running and shows
                // nothing -- with no line in the log, because the thing that
                // would have written one is what stopped. This task is the
                // right watcher: it is independent of the desktop, it never
                // blocks, and it is already awake once a second.
                //
                // `kprintln!` and not `serial_println!`, because it reaches
                // all three sinks. The console paints through
                // `compose::flush_rect` and takes no desktop claim, so it is
                // the one painter that still works while the compositor holds
                // the claim and is stuck inside a frame.
                match gfx::render::watch() {
                    Some(gfx::render::Verdict::Stalled(h)) => {
                        console::on_channel(console::EXEC, || {
                            kprintln!(
                                "[gfx] the compositor has been quiet for {} ms while {} -- the screen is stopped",
                                h.quiet_ms,
                                h.phase.name()
                            );
                            // The scheduler's own word for it, because "not
                            // turning" has three causes it tells apart and
                            // nothing else does: starved, claimed by a core
                            // that is not running it, or stranded mid-switch.
                            if let Some(st) = gfx::render::comp_state() {
                                kprintln!("       the scheduler has its task as '{}'", st);
                            }
                        });
                    }
                    Some(gfx::render::Verdict::Recovered(h)) => {
                        console::on_channel(console::EXEC, || {
                            kprintln!(
                                "[gfx] the compositor is painting again after {} ms",
                                h.worst_ms
                            );
                        });
                    }
                    None => {}
                }

                ai::futures::sample();
                // The thermal policy and the power-source policy, both of
                // which were written and neither of which had a caller. A
                // rule nothing runs is a rule that does not exist, and both
                // say so out loud when they act, because a machine that
                // quietly changes its own clock is one nobody can explain.
                for what in [dev::power::tick(), dev::battery::policy_tick()] {
                    if let Some(msg) = what {
                        console::on_channel(console::EXEC, || kprintln!("[power] {}", msg));
                    }
                }
            }
            // **The clock task no longer touches the framebuffer at all.**
            //
            // It painted the uptime and the charge straight into the taskbar
            // from here, on its own quantum, while `draw` ran on another task
            // -- two writers on one back buffer, which is exactly what the
            // paint claim existed to referee. The compositor owns both
            // readouts now (`desk::paint_tray`), so there is nobody to
            // referee.
            //
            // That also ends the symptom this whole rearrangement started
            // from: a desktop frozen solid with the uptime still ticking in
            // the corner of it. The readouts stop when the compositor stops,
            // so a stopped screen looks stopped, and what reports liveness
            // instead is the watchdog above -- in words, on a channel the
            // screen cannot take away.
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
            // Which clock, not just the number. A TSC calibrated from the tick
            // counter is not a second opinion about the tick counter, and the
            // figure alone cannot say which kind it is.
            kprintln!(
                "  tsc {} MHz, measured against {}",
                time::tsc_mhz(),
                if time::reference_is_independent() {
                    source
                } else {
                    "the tick counter, so the two cannot be compared"
                }
            );
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
/// Start the other cores.
///
/// After `init_interrupts`, which is not incidental: this needs the heap for
/// AP stacks, the LAPIC to send INIT, and a calibrated TSC to time the pause
/// between INIT and SIPI. All three land in or before that call.
fn init_smp(acpi: &Option<acpi::Acpi>) {
    console::set_color(LTGREEN);
    kprintln!("\n[smp]");
    console::set_color(LTGRAY_IDX);

    // Core 0 gets its block on every path, including the two that start no
    // other core.
    //
    // Per-core storage is not only about other cores, and treating it that way
    // was a real bug rather than an inelegance. `recover::slot` reads it to
    // find where a guarded fault should land and `mem::census` reads it to bill
    // an allocation, so both were dead on a single-core machine and on one with
    // no ACPI tables: the returns below came before `percpu::arm`, `armed()`
    // stayed false forever, `billed()` answered `None`, and **every fault
    // inside a guard was fatal**. On a machine whose stated reason for having
    // guards is that it runs programs it wrote itself.
    //
    // Measured before the fix, `-smp 1`: `diag recover` halted the machine at
    // the third claim, twice out of twice, and `diag all` never reached the
    // sixteen suites after it. At four cores the same suite passed. Nothing in
    // that difference was about parallelism.
    //
    // Only the percpu half is hoisted. `gdt::prepare` builds a TSS and two
    // 16 KiB interrupt stacks per core for application processors to load, and
    // core 0 is already running on the boot tables, so calling it here would
    // allocate for a core that will never adopt them.
    let one_core = |why: core::fmt::Arguments| {
        kprintln!("{}", why);
        cpu::percpu::prepare(1);
        cpu::percpu::adopt(0);
        cpu::percpu::arm();
    };
    let Some(a) = acpi else {
        one_core(format_args!("  no acpi tables -- staying on one core"));
        return;
    };
    if a.cpus <= 1 {
        one_core(format_args!(
            "  firmware declares {} cpu -- nothing to start",
            a.cpus
        ));
        return;
    }

    // Core 0's own block first. Arming comes after every core has one, so
    // nothing reads through a GS base that is still zero.
    // Tables and per-core blocks for every core the firmware declares, built
    // here where a fault is reportable, before any core is started.
    cpu::gdt::prepare(a.cpus.min(task::MAX_CPUS));
    cpu::percpu::prepare(a.cpus.min(task::MAX_CPUS));
    cpu::percpu::adopt(0);
    let started = smp::init(a);
    // Every core has its block now, so per-core storage may be read, and only
    // then are the cores let go.
    cpu::percpu::arm();
    smp::release();
    let answered = started + 1;
    if answered == a.cpus {
        console::set_color(LTGREEN);
    } else {
        console::set_color(YELLOW);
    }
    kprintln!("  {} of {} cores answered", answered, a.cpus);
    console::set_color(LTGRAY_IDX);
    if answered < a.cpus {
        kprintln!("  the rest stay parked; work still runs, just narrower");
    }
    if started > 0 {
        let ok = smp::selftest();
        if !ok {
            console::set_color(LTRED);
        }
        kprintln!(
            "  {}  a split matvec and its adjoint equal whole ones, bit for bit",
            if ok { "ok " } else { "FAIL" }
        );
        console::set_color(LTGRAY_IDX);
    }
}

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

    match serial::attach_irq(a, dev::lapic::id()) {
        Some(gsi) => kprintln!("  serial   interrupt on gsi {}", gsi),
        None => kprintln!("  serial   polled -- no ioapic route"),
    }

    let report = dev::kbd::init(a, dev::lapic::id());
    let mouse = dev::mouse::init(a, dev::lapic::id());
    if let Some(fb) = gfx::primary() {
        dev::mouse::set_bounds(fb.width() as i32, fb.height() as i32);
    }

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

    if mouse.present {
        kprintln!(
            "  mouse   id {:?}{}, irq12 via gsi {:?}",
            mouse.id,
            if mouse.wheel { " with wheel" } else { "" },
            mouse.routed_gsi
        );
    } else {
        kprintln!("  mouse   no answer on port 2");
    }
}

/// Kernel heap sizes to try, largest first.
///
/// Grown three times, each time by a model. 4 MiB was ample until weights had
/// to fit at all; 16 MiB was enough until a 30-layer one arrived; 64 MiB held
/// SmolLM2's 23.8 MiB KV cache with room. Qwen3-0.6B needs 112 MiB of KV cache
/// alone at seq 512 -- 28 layers of 1024-wide keys and values -- and snapshotting
/// it into the store transiently wants that much again.
///
/// A ladder rather than a constant because this is one allocation of *physically
/// contiguous* frames, and the only machine that matters cannot be tested from
/// here. A fixed 320 MiB that the GF63's memory map cannot satisfy is an
/// unbootable system; falling back to 64 MiB is a system that boots and says so.
/// The sizes it lands on are all ones that have run.
const HEAP_LADDER: [usize; 5] = [81920, 65536, 32768, 16384, 4096];

fn init_heap(frames: &mut mem::frame::EarlyFrames) {
    // Measure, then ask once. The obvious shape -- try each rung until one
    // succeeds -- does not work with this allocator and silently did not:
    // `alloc_contiguous` advances its region index on every rejection and never
    // rewinds, so a failed first rung leaves it at the end of the map and every
    // smaller rung fails immediately. The ladder degraded to all-or-nothing
    // while its comment claimed otherwise.
    let span = frames.largest_span();
    let free = frames.total_free();

    // The largest single region is what bounds an allocation; the total is not.
    // Printed because the KV cache is the largest thing this system allocates
    // and this is the number that decides how much context fits -- and on the
    // one machine that matters it has never been measured.
    kprintln!(
        "[boot] phys  {} MiB free, largest contiguous region {} MiB",
        free * mem::PAGE_SIZE as usize / 1024 / 1024,
        span * mem::PAGE_SIZE as usize / 1024 / 1024,
    );

    let Some((i, pages)) = HEAP_LADDER
        .iter()
        .copied()
        .enumerate()
        .find(|&(_, pages)| pages <= span)
    else {
        console::set_color(LTRED);
        kprintln!(
            "[boot] heap allocation FAILED -- largest region is {} MiB, smallest rung is {} MiB",
            span * mem::PAGE_SIZE as usize / 1024 / 1024,
            HEAP_LADDER[HEAP_LADDER.len() - 1] * mem::PAGE_SIZE as usize / 1024 / 1024,
        );
        console::set_color(LTGRAY_IDX);
        return;
    };

    let Some(base) = frames.alloc_contiguous(pages) else {
        // Unreachable unless `largest_span` and `alloc_contiguous` disagree
        // about what is available, which would mean one of them is wrong.
        console::set_color(LTRED);
        kprintln!("[boot] heap allocation FAILED -- {} MiB was measured available",
            pages * mem::PAGE_SIZE as usize / 1024 / 1024);
        console::set_color(LTGRAY_IDX);
        return;
    };

    let size = pages * mem::PAGE_SIZE as usize;
    unsafe { mem::heap::HEAP.add_region(base as usize, size) };
    // Anything below the first rung means a model may fail to allocate its
    // state later, with a message far from the cause. Say it here.
    if i > 0 {
        console::set_color(YELLOW);
    }
    kprintln!(
        "[boot] heap {} MiB at {:#x}{}",
        size / 1024 / 1024,
        base,
        if i > 0 { "  (reduced -- no larger contiguous region)" } else { "" }
    );
    console::set_color(LTGRAY_IDX);

    // Then take everything else that is left.
    //
    // `add_region` inserts into one address-sorted free list and coalesces, so
    // extra regions cost nothing and are indistinguishable from the first once
    // they are in. This raises *total* heap without raising the largest single
    // allocation -- the regions are disjoint by definition -- which is exactly
    // what the per-layer KV cache needs: many allocations of a few tens of MiB
    // rather than one of several hundred.
    //
    // Safe to consume the map because `frames` has no users after this point.
    // If SMP ever arrives it will want low memory for AP trampolines and must
    // reserve before this runs, not after.
    let mut extra = 0usize;
    let mut regions = 1usize;
    loop {
        let span = frames.largest_span();
        if span < MIN_EXTRA_PAGES {
            break;
        }
        let Some(base) = frames.alloc_contiguous(span) else {
            break;
        };
        unsafe { mem::heap::HEAP.add_region(base as usize, span * mem::PAGE_SIZE as usize) };
        extra += span;
        regions += 1;
    }
    if extra > 0 {
        kprintln!(
            "[boot] heap +{} MiB across {} more regions ({} MiB total)",
            extra * mem::PAGE_SIZE as usize / 1024 / 1024,
            regions - 1,
            (extra + pages) * mem::PAGE_SIZE as usize / 1024 / 1024,
        );
    }
}

/// Ignore scraps. A region too small to hold anything the model allocates costs
/// a free-list entry on every traversal and buys nothing.
const MIN_EXTRA_PAGES: usize = 256; // 1 MiB

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
                "[boot] paging active  cr3={:#x}  mapped {} MiB  ({} frames, 1 GiB pages {})",
                cpu::read_cr3(),
                limit / (1024 * 1024),
                frames.allocated_frames(),
                if cpu::gib_pages_supported() { "yes" } else { "no" }
            );
            // Both change what a page table entry *means*, so they go on
            // immediately after the map this kernel built becomes the map the
            // processor is using, and before anything has a chance to rely on
            // a permission that was not being enforced.
            //
            // Neither changes anything today. Everything is mapped writable
            // and nothing has ever set bit 63, so the map means exactly what
            // it meant a moment earlier. What they buy is that read-only and
            // no-execute stop being decorative the first time anything asks
            // for them.
            cpu::enable_wp();
            let nx = cpu::enable_nx();
            kprintln!(
                "[boot] page rights  wp={}  nx={}",
                if cpu::wp_on() { 1 } else { 0 },
                if nx { 1 } else { 0 }
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
/// Run one boot selftest under a guard, and decide what its failure means.
///
/// **This is the line between "a thermometer broke" and "the machine is
/// gone".** Before it, any fault inside any selftest halted the boot before
/// the shell existed -- which is exactly what a `#GP` in `dev::power` did on
/// the first bare-metal run, costing storage, the namespace and the model to a
/// register nobody needs.
///
/// `Unguarded` is not treated as a pass and not treated as a failure: it means
/// the closure ran with no landing pad, so nothing was proven either way. It
/// cannot happen here -- `percpu::arm` runs at `init_smp`, one step before the
/// selftests -- and is matched explicitly so that if the boot order ever
/// changes, this reads as the open question it is rather than as success.
///
/// Note the `fn() -> bool` rather than a closure: a check has to be
/// **re-runnable**, because re-running it is how a repair is judged. Every
/// section here captures nothing, so this costs nothing and buys the whole
/// repair loop.
///
/// ### A fault is only half of failing
///
/// This took `fn()` for as long as it had existed, and every subsystem wrapped
/// in it *answers a verdict*: `sysbox::selftest`, `crypto::selftest`,
/// `rng::selftest`, `fmt`, `usbhid` and `code` all return `bool`, and every
/// call site here discarded it -- `|| { sysbox::selftest(); }`, literally. So
/// the boot and repair loop was a **liveness oracle wearing a correctness
/// one's name**: a change that made ChaCha20 return the wrong bytes without
/// faulting was not recorded in `boot_report`, not counted by `outstanding()`,
/// never offered to `repair`, and did not stop the boot *even when `Vital`*.
///
/// The two failures are recorded identically from here on. They are not the
/// same thing and the report says which: a fault is a subsystem that is
/// **gone**, a `false` is one that is **wrong**, and wrong is the more
/// dangerous of the two everywhere a wrong answer still looks like an answer.
fn section(name: &'static str, need: boot_report::Need, f: fn() -> bool) {
    use cpu::recover::Caught;
    // Recorded whether it passes or not, because a repair already applied to
    // this subsystem has to be re-testable: passing with a repair holding it up
    // and passing because the bug was fixed look the same from anywhere else.
    boot_report::note_check(name, f);

    // The verdict comes back through a local because `guarded` takes a closure
    // that answers nothing, and it has to stay that way -- on the path where
    // the closure never finished there is no value for the longjmp to produce.
    // So `false` is what a fault leaves behind, which is the right default and
    // is never what gets reported: the fault arm names the fault instead.
    let mut agreed = false;
    let caught = cpu::recover::guarded(|| agreed = f());

    if let Caught::Unguarded(_) = caught {
        console::set_color(LTRED);
        kprintln!("[selftest] {} ran with no landing pad, so a fault would have been fatal", name);
        console::set_color(LTGRAY_IDX);
    }

    // `Unguarded` is not a pass and not a failure -- but the closure *ran*
    // either way, so whatever verdict it reached still stands. What was not
    // proven there is only that a fault would have been caught.
    let broke = match caught {
        Caught::Faulted(why) => Some((why, cpu::recover::site().unwrap_or(0))),
        // No faulting instruction to point at, so no site. `recover::take_panic`
        // zeroes `LAST_RIP` on the same argument.
        _ if !agreed => Some(("failed its own checks", 0)),
        _ => None,
    };
    let Some((why, rip)) = broke else { return };

    boot_report::record(name, need, why, rip, f);
    console::set_color(LTRED);
    kprintln!(
        "[selftest] {} {} -- {}",
        name,
        why,
        match need {
            boot_report::Need::Vital => "and this machine needs it",
            boot_report::Need::Optional => "this subsystem is unavailable",
        }
    );
    console::set_color(LTGRAY_IDX);
    if need == boot_report::Need::Vital {
        // Nothing after this line can be trusted, so the honest thing is to
        // stop here rather than to boot something that will fail somewhere
        // less legible.
        boot_report::report();
        console::set_color(LTRED);
        kprintln!("\n[boot] {} is vital, so this machine will not continue.", name);
        halt();
    }
}

/// Whether the allocator gives back exactly what it handed out.
///
/// **This was asking the wrong question and had been printing `LEAKED` in red
/// on every boot for as long as anything else allocated before it.** It
/// compared the heap against *zero* after its own objects dropped, which is
/// the same question only while nothing else in the kernel has ever
/// allocated -- and the console's scrollback ring, among others, is long since
/// resident by the time the selftests run. A clean boot read `after drop:
/// 399104 B LEAKED` and nothing consumed the verdict, so the line was
/// computed, coloured red, printed, and never once acted on. It is the first
/// thing `section` learning to read a verdict turned up, which is the whole
/// argument for the change.
///
/// What it means is whether *this block's* allocations came back, so it is a
/// delta against a baseline taken a line earlier.
fn check_heap() -> bool {
    console::set_color(LTGREEN);
    kprintln!("\n[selftest] heap:");
    console::set_color(LTGRAY_IDX);
    let (before, _) = mem::heap::HEAP.stats();
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
    // way this is exactly the baseline; any other number is a per-allocation
    // leak, and the sign says which way.
    let (after, _) = mem::heap::HEAP.stats();
    let ok = after == before;
    console::set_color(if ok { LTGREEN } else { LTRED });
    if ok {
        kprintln!(
            "  after drop: back to {} B -- alloc/dealloc are exact inverses",
            before
        );
    } else {
        kprintln!(
            "  after drop: {} B against {} B before -- alloc and dealloc disagree",
            after, before
        );
    }
    console::set_color(LTGRAY_IDX);
    ok
}

/// Version ordering, because an updater will decide on it.
///
/// The interesting case is 0.10.0 against 0.9.0: compared as strings "0.1"
/// sorts before "0.9", so the naive implementation installs an older image and
/// reports success. Checked here rather than reasoned about, since the failure
/// is silent and the consequence is a downgrade nobody asked for.
fn check_version() -> bool {
    console::set_color(LTGREEN);
    kprintln!("\n[selftest] version:");
    console::set_color(LTGRAY_IDX);
    let vok = version_newer("0.2.0", "0.1.0")
        && version_newer("0.10.0", "0.9.0")
        && !version_newer("0.9.0", "0.10.0")
        && !version_newer(VERSION, VERSION)
        && version_newer("1.0.0", "0.99.99");
    if !vok {
        console::set_color(LTRED);
    }
    kprintln!(
        "  {}  this build is {}, and 0.10.0 is newer than 0.9.0",
        if vok { "ok " } else { "FAIL" },
        VERSION
    );
    console::set_color(LTGRAY_IDX);
    vok
}

/// The timer is the first thing in this kernel that runs without being called.
/// If ticks advance, the LAPIC, the IDT vector, the EOI path and the
/// calibration are all correct at once.
fn check_timer() -> bool {
    console::set_color(LTGREEN);
    kprintln!("\n[selftest] timer:");
    console::set_color(LTGRAY_IDX);
    // Timed against the TSC rather than labelled. This printed "N ticks in
    // ~0.5 s" for a long time, where the 0.5 was a constant in the format
    // string and not a measurement -- so it read identically however fast the
    // counter was really advancing, and could not see that every core's timer
    // ISR was incrementing one global `TICKS`. Two clocks that are supposed to
    // agree do not stay agreeing on their own.
    //
    // **And for a while afterwards it still could not see it**, which is the
    // more interesting half. `time::calibrate` derived the microsecond from
    // `ticks()` itself, so a tick rate wrong by a whole core count scaled both
    // sides of this comparison and divided straight back out. Driven, with the
    // ISR incrementing by two: `tsc` read 1345 MHz against a true 2690, this
    // line answered "the two clocks agree" over a window half as long as it
    // believed, and `uptime` reported 22.54 s at 11.3 s of real time. A check
    // with a hundred per cent false-negative rate for the one bug its own
    // message names.
    //
    // The reference is independent now (`lapic::tsc_per_us_ref`, off the PIT
    // or the PM timer), which is what makes the band below mean anything: the
    // same injection reads 250 ms against a floor of 350 and fails.
    let t0 = time::rdtsc();
    let start = dev::lapic::ticks();
    let want = start + TIMER_HZ as u64 / 2; // half a second, if ticks are honest
    let mut spins: u64 = 0;
    while dev::lapic::ticks() < want {
        spins += 1;
        if spins > 200_000_000 {
            break; // timer is dead; do not hang the boot waiting for it
        }
        core::hint::spin_loop();
    }
    let elapsed = dev::lapic::ticks() - start;
    let mhz = time::tsc_mhz();
    let real_ms = if mhz > 0 {
        (time::rdtsc() - t0) / (mhz * 1000)
    } else {
        0
    };
    let verdict = if elapsed < TIMER_HZ as u64 / 2 {
        console::set_color(LTRED);
        kprintln!("  only {} ticks -- timer is not delivering", elapsed);
        false
    } else if mhz > 0 && !time::reference_is_independent() {
        // Neither the PIT nor the PM timer answered, so the only TSC figure
        // available was derived from this very counter. Reporting agreement
        // would be reporting arithmetic. Firing is still checked above, and
        // that is the half this can honestly answer.
        kprintln!(
            "  {} ticks -- firing, but the TSC was calibrated from this same \
             counter, so the two cannot be compared",
            elapsed
        );
        true
    } else if mhz == 0 {
        kprintln!("  {} ticks -- firing, but the TSC is uncalibrated", elapsed);
        // Firing is the half this check exists for, and an uncalibrated TSC
        // is a fact about the other clock. Refusing here would make a machine
        // whose timer is fine unbootable for the sake of a comparison that
        // could not be made.
        true
    } else {
        // 500 ms expected. The band stays wide on purpose: this is a spin
        // loop on an emulator and the point is to catch a rate wrong by a
        // whole core count, not to measure the crystal. Two cores read 250 ms
        // and three read 167, so the floor has a factor of 1.4 of headroom
        // under the smallest error worth catching, and ten boots of a healthy
        // machine measured 482 to 501.
        let ok = (350..=750).contains(&real_ms);
        console::set_color(if ok { LTGREEN } else { LTRED });
        kprintln!(
            "  {} {} ticks in {} ms of TSC time -- {}",
            if ok { "ok  " } else { "FAIL" },
            elapsed,
            real_ms,
            if ok {
                "the two clocks agree"
            } else {
                "ticks() disagrees with the TSC; is every core incrementing it?"
            }
        );
        ok
    };
    console::set_color(LTGRAY_IDX);
    verdict
}

/// Calendar arithmetic, and what the machine thinks the date is.
///
/// The reading is printed and never judged: a machine with no usable RTC says
/// so and carries on, because "snapshots will record no time" is a limitation
/// and not a broken subsystem.
fn check_clock() -> bool {
    console::set_color(LTGREEN);
    kprintln!("\n[selftest] clock:");
    console::set_color(LTGRAY_IDX);
    let ok = dev::rtc::selftest();
    if ok {
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
    ok
}

/// One line per parser, and **a verdict where there used to be silence.**
///
/// These four were `if json::selftest() { print the ok line }`, so a parser
/// that failed printed *nothing at all* -- the worst shape a check can take,
/// because a missing line is what a check that never ran looks like too.
fn check_parser(name: &'static str, what: &'static str, f: fn() -> bool) -> bool {
    let ok = f();
    console::set_color(if ok { LTGREEN } else { LTRED });
    kprintln!("  {}     {:<9} {}", if ok { "ok" } else { "FAIL" }, name, what);
    console::set_color(LTGRAY_IDX);
    ok
}

fn check_json() -> bool {
    check_parser("json", "parse, escapes, snowflakes, depth bound", json::selftest)
}
fn check_ws() -> bool {
    check_parser("websocket", "RFC 6455 accept, masking, split frames", net::ws::selftest)
}
fn check_html() -> bool {
    check_parser("html", "urls, entities, unclosed tags", net::html::selftest)
}
fn check_css() -> bool {
    check_parser("css", "selectors, at-rules, inline display", net::css::selftest)
}

fn check_text() -> bool {
    kprintln!("\n[selftest] text:");
    let ok = gfx::text_selftest();
    if !ok {
        console::set_color(LTRED);
        kprintln!("[selftest] the console cannot be trusted to draw what it was given");
        console::set_color(LTGRAY_IDX);
    }
    ok
}

/// Cheap, pure, and no network: header assembly, the target arithmetic and the
/// midstate. Every mistake available in that code is silent -- a header with
/// two bytes swapped hashes at full speed and is rejected forever -- so it
/// earns a place in the boot sequence rather than only in `diag`.
fn check_mining() -> bool {
    kprintln!("\n[selftest] mining:");
    let mut bad = 0usize;
    let mut n = 0usize;
    for (what, good) in mine::checks() {
        n += 1;
        if !good {
            bad += 1;
            console::set_color(LTRED);
            kprintln!("  FAIL {}", what);
            console::set_color(LTGRAY_IDX);
        }
    }
    if bad == 0 {
        console::set_color(LTGREEN);
        kprintln!("  ok   {} claim(s), block 125552 reassembles and hashes", n);
        console::set_color(LTGRAY_IDX);
    }
    bad == 0
}

fn selftest(acpi_ref: &Option<acpi::Acpi>) {
    // **Every check here is wrapped now, and thirteen of them were not.**
    // An unwrapped check that faults takes the machine before the shell
    // exists, which is the failure `boot_report` was built for and which
    // seven of twenty checks were still exposed to; one that answers `false`
    // was recorded nowhere at all. Both are `section`'s job.
    //
    // `Optional` on all of the newly wrapped ones, deliberately. What the
    // wrapping buys is that a failure is *recorded and named* rather than
    // fatal or silent; escalating any of these to `Vital` is a separate
    // decision that wants evidence from the GF63 about, for instance, how
    // wide the timer's band really is on hardware -- and a `Vital` false
    // positive is an unbootable machine, which is exactly as bad as the miss
    // it would be protecting against.
    section("heap", boot_report::Need::Optional, check_heap);
    section("version", boot_report::Need::Optional, check_version);
    section("timer", boot_report::Need::Optional, check_timer);
    section("clock", boot_report::Need::Optional, check_clock);

    console::set_color(LTGREEN);
    kprintln!("\n[selftest] sysbox namespace:");
    console::set_color(LTGRAY_IDX);
    section("sysbox", boot_report::Need::Vital, sysbox::selftest);

    // The RFC vectors, at every boot. 25 ms, and it is the only thing standing
    // between a broken field arithmetic and a TLS handshake that fails with
    // nothing to point at -- crypto is the one place where wrong code still
    // produces perfectly plausible output.
    // Vital: a cipher that is quietly wrong produces output that works
    // perfectly and is not secure, which is the failure `crypto` opens by
    // warning about. Absent is safer than subtly broken.
    section("crypto", boot_report::Need::Vital, crypto::selftest);

    // Straight after the ciphers, and deliberately so: the generator is a
    // construction over the ChaCha20 checked one line above, so its claims
    // only mean anything if that one passed.
    console::set_color(LTGREEN);
    kprintln!("\n[selftest] random:");
    console::set_color(LTGRAY_IDX);
    section("rng", boot_report::Need::Vital, || {
        let ok = rng::selftest();
        if !ok {
            console::set_color(LTRED);
            kprintln!("  FAIL -- key material would look fine and be predictable");
            console::set_color(LTGRAY_IDX);
        }
        ok
    });

    section("json", boot_report::Need::Optional, check_json);
    section("websocket", boot_report::Need::Optional, check_ws);
    section("html", boot_report::Need::Optional, check_html);
    section("css", boot_report::Need::Optional, check_css);

    console::set_color(LTGREEN);
    kprintln!("\n[selftest] int3 should report and resume:");
    console::set_color(LTGRAY_IDX);
    unsafe {
        core::arch::asm!("int3", options(nomem, nostack));
    }

    console::set_color(LTGREEN);
    kprintln!("[selftest] survived int3 -- idt is live.");
    console::set_color(LTGRAY_IDX);

    // Beside int3 because it asks the same kind of question -- whether the
    // processor does what this kernel believes it does -- and because it is
    // the first time anything here fetches an instruction from the heap. A
    // wrong answer is a halted machine, so it runs early and under QEMU
    // before it ever runs on the GF63.
    // **The one that has actually faulted on real hardware.** Optional by
    // any reading: nothing downstream needs a temperature, and the first
    // bare-metal boot lost the entire machine to it.
    // **This one answers `true` unconditionally, and that is the honest
    // verdict rather than a leftover.** What this section can observe is
    // whether reading the registers takes the machine down -- which is exactly
    // the GF63's bug and exactly what `Caught::Faulted` reports. Every claim
    // `dev::power` can make about a *value* without hardware is already `diag
    // power`, so a verdict here would be that suite run twice, printed into
    // the boot log, and re-run on every repair judgement.
    section("power", boot_report::Need::Optional, || {
        dev::power::probe();
        kprintln!("
[power]");
        dev::power::report();
        true
    });

    kprintln!("
[selftest] file formats:");
    section("fmt", boot_report::Need::Optional, || {
        let ok = fmt::selftest();
        if !ok {
            console::set_color(LTRED);
            kprintln!("[selftest] file type handling is unsound");
            console::set_color(LTGRAY_IDX);
        }
        ok
    });

    // **The two that are not wrapped, and the reason is a type rather than an
    // oversight.** Both take `acpi_ref`, so a closure around either captures,
    // and `section` wants a `fn()` precisely because a check that cannot be
    // re-run is a check no repair can be judged against. Making these
    // re-runnable means giving `acpi` a handle that outlives this call, which
    // is a change to how the tables are held and belongs in its own argument.
    // Until then a fault in either is fatal, the way every check here used to
    // be.
    kprintln!("
[selftest] acpi tables:");
    if !acpi::selftest(acpi_ref) {
        console::set_color(LTRED);
        kprintln!("[selftest] the tables cannot be trusted, and the namespace comes from them");
        console::set_color(LTGRAY_IDX);
    }

    kprintln!("
[selftest] aml:");
    if !acpi::aml_selftest(acpi_ref) {
        console::set_color(LTRED);
        kprintln!("[selftest] the namespace is not trustworthy, and battery comes from it");
        console::set_color(LTGRAY_IDX);
    }

    kprintln!("
[selftest] usb input:");
    section("usbhid", boot_report::Need::Optional, || {
        let ok = dev::usbhid::selftest();
        if !ok {
            console::set_color(LTRED);
            kprintln!("[selftest] a USB keyboard would type the wrong characters");
            console::set_color(LTGRAY_IDX);
        }
        ok
    });

    section("text", boot_report::Need::Optional, check_text);
    section("mining", boot_report::Need::Optional, check_mining);

    kprintln!("
[selftest] generated code:");
    section("code", boot_report::Need::Optional, || {
        let ok = cpu::code::selftest();
        if !ok {
            console::set_color(LTRED);
            kprintln!("[selftest] the code substrate is not sound -- do not generate any");
            console::set_color(LTGRAY_IDX);
        }
        ok
    });

    // The deliberate null dereference now lives behind the shell's `fault`
    // command. It is fatal by design, so running it during boot would mean the
    // shell never starts.
}

/// What this build is.
///
/// A prerequisite for updating, not a nicety: an updater has to answer "is the
/// staged image newer than the running one", and until now nothing in the
/// binary could say what the running one *was*. The only version strings in
/// the whole image were two hardcoded `User-Agent: glados/0.1` headers.
///
/// From `CARGO_PKG_VERSION` rather than a constant typed here, so the number
/// in `Cargo.toml` and the number the machine reports cannot disagree. There
/// is deliberately no `build.rs`: a git hash would make every build a
/// different version and this tree has no CI to stamp one consistently.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Ordered comparison, for deciding whether an update goes backwards.
///
/// Dotted numbers, compared numerically field by field. String comparison
/// would sort "0.10.0" before "0.9.0", which is the classic way an updater
/// installs an older image and reports success.
pub fn version_newer(candidate: &str, current: &str) -> bool {
    let mut a = candidate.split('.');
    let mut b = current.split('.');
    for _ in 0..4 {
        let x: u32 = a.next().unwrap_or("0").trim().parse().unwrap_or(0);
        let y: u32 = b.next().unwrap_or("0").trim().parse().unwrap_or(0);
        if x != y {
            return x > y;
        }
    }
    false
}

fn banner(boot: &BootInfo, acpi: &Option<acpi::Acpi>) {
    console::set_color(LTCYAN);
    kprintln!("glados {}", VERSION);
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
    // The pixel-format check: if red and blue come out swapped, the firmware
    // reported Rgbx where it meant Bgrx. Drawn only once the boot screen has
    // handed the framebuffer back -- until then there is a panel in the way,
    // and three bars across it prove nothing except that something else is
    // drawing.
    kprintln!("\n[selftest] bars should read RED GREEN BLUE, left to right:");
    if !gfx::splash::active() {
        let y = boot.fb.height().saturating_sub(80);
        let w = boot.fb.width() / 6;
        boot.fb.rect(w, y, w, 40, palette::LTRED);
        boot.fb.rect(w * 2, y, w, 40, palette::LTGREEN);
        boot.fb.rect(w * 3, y, w, 40, palette::LTBLUE);
    } else {
        kprintln!("  deferred: 'video bars' redraws them");
    }

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
    // **A panic during a boot selftest is survivable; everywhere else it is
    // not.** A check that asserts its way out has said its subsystem is
    // broken, which is information rather than a reason to stop the machine --
    // and an `assert!` is how most selftests fail, so catching only hardware
    // exceptions would cover far less than it appears to.
    //
    // The window is opened around the selftest block and closed immediately
    // after, so this consults a pad for one stretch of boot and never again. A
    // panic means a Rust invariant broke, which is a weaker thing to survive
    // than a #GP, and that narrowness is the whole of what makes it
    // defensible.
    //
    // Serial and not the console, because the console lock may be exactly what
    // the panicking code was holding; `guard` releases it after landing, which
    // has not happened yet.
    if crate::cpu::recover::in_selftest() {
        serial_println!("\n*** PANIC (inside a selftest, recovering) *** {}", info);
        if let Some(pad) = crate::cpu::recover::take_panic() {
            unsafe { crate::cpu::recover::land(pad) }
        }
    }
    serial_println!("\n*** PANIC *** {}", info);
    if console::is_ready() {
        // A panic behind a progress bar helps nobody, and on the GF63 the
        // framebuffer is the only place a diagnostic can go.
        gfx::splash::abandon();
        console::set_color(LTRED);
        console::_print(format_args!("\n*** KERNEL PANIC ***\n{}\n", info));
    }
    halt()
}
