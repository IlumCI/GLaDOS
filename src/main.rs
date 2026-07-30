//! sanctum -- a from-scratch ring-0 operating system for the MSI MS-16R8.
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

mod cpu;
mod gfx;
mod mem;
mod serial;
mod sync;
mod uefi;

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
    /// One past the last byte of the framebuffer aperture. Our page tables have
    /// to cover this or the first write after `activate` faults.
    pub fb_end: u64,
}

#[no_mangle]
pub extern "efiapi" fn efi_main(image: Handle, st: *mut SystemTable) -> Status {
    serial::init();
    serial_println!("\n\nsanctum: entered efi_main");

    let st = unsafe { &mut *st };
    let bs = unsafe { &mut *st.boot_services };

    // UEFI arms a 5-minute watchdog for boot applications. We are never going
    // to return, so if we leave it armed the firmware resets the machine
    // mid-boot and it looks like a kernel hang.
    (bs.set_watchdog_timer)(0, 0, 0, ptr::null_mut());
    serial_println!("sanctum: watchdog disarmed");

    // --- Graphics Output Protocol ---
    let mut gop_ptr: *mut c_void = ptr::null_mut();
    let s = (bs.locate_protocol)(
        &GRAPHICS_OUTPUT_PROTOCOL_GUID,
        ptr::null_mut(),
        &mut gop_ptr,
    );
    if is_error(s) || gop_ptr.is_null() {
        con_out(st, "sanctum: no Graphics Output Protocol\r\n");
        serial_println!("sanctum: locate_protocol(GOP) failed: {:#x}", s);
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
            con_out(st, "sanctum: unsupported GOP pixel format\r\n");
            serial_println!("sanctum: pixel_format {} unsupported", other);
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

    let fb_end = mode.frame_buffer_base + mode.frame_buffer_size as u64;

    serial_println!(
        "sanctum: fb base={:#x} {}x{} stride={} format={:?}",
        mode.frame_buffer_base,
        info.horizontal_resolution,
        info.vertical_resolution,
        info.pixels_per_scan_line,
        format
    );

    // --- ACPI RSDP, while the configuration table still exists ---
    let rsdp = find_rsdp(st);
    serial_println!("sanctum: rsdp={:?}", rsdp);

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
        con_out(st, "sanctum: allocate_pool for memory map failed\r\n");
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
            con_out(st, "sanctum: get_memory_map failed\r\n");
            halt();
        }

        if (bs.exit_boot_services)(image, map_key) == SUCCESS {
            break sz;
        }

        attempts += 1;
        if attempts > 8 {
            con_out(st, "sanctum: exit_boot_services kept failing\r\n");
            halt();
        }
    };

    // ---------------------------------------------------------------
    // Past this line the firmware is gone. No boot services, no con_out,
    // no protocols. Serial and the framebuffer are all we have.
    // ---------------------------------------------------------------
    serial_println!("sanctum: exited boot services after {} retries", attempts);

    let boot = BootInfo {
        fb,
        rsdp,
        mmap: buf,
        mmap_size: final_size,
        desc_size,
        fb_end,
    };

    console::init(boot.fb, 2, palette::BLACK);

    // Replace the firmware's descriptor tables with ours. Until the IDT is in,
    // any fault is a triple fault: an instant reboot with no diagnostic.
    cpu::gdt::init();
    cpu::idt::init();
    kprintln!("[boot] gdt + idt installed");

    install_paging(&boot);

    banner(&boot);
    selftest();

    halt();
}

/// Build and install our own identity map, replacing the firmware's.
///
/// Failure here is survivable: UEFI's page tables are still loaded and still
/// correct, so we report and carry on rather than halting. Everything through
/// M4 works fine on the firmware's map -- we just do not own it.
fn install_paging(boot: &BootInfo) {
    let mut frames = unsafe {
        mem::frame::EarlyFrames::new(boot.mmap, boot.mmap_size, boot.desc_size)
    };

    let top = mem::frame::max_physical_address(boot.mmap, boot.mmap_size, boot.desc_size);
    // Always cover the low 4 GiB: the MMIO hole lives there, and on this
    // machine so does the Intel iGPU's framebuffer aperture.
    let limit = top.max(boot.fb_end).max(4 * mem::GIB);

    match mem::paging::build_identity_map(&mut frames, limit) {
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
    kprintln!("\n[selftest] int3 should report and resume:");
    console::set_color(LTGRAY_IDX);
    unsafe {
        core::arch::asm!("int3", options(nomem, nostack));
    }

    console::set_color(LTGREEN);
    kprintln!("[selftest] survived int3 -- idt is live.");
    console::set_color(LTGRAY_IDX);

    // Fatal, and meant to be. If the next thing on screen is a #PF report
    // naming cr2 = 0, the single most important piece of debugging
    // infrastructure in this kernel is working.
    cpu::idt::trigger_page_fault();

    kprintln!("[selftest] UNREACHABLE -- the null read did not fault!");
}

fn banner(boot: &BootInfo) {
    console::set_color(LTCYAN);
    kprintln!("sanctum");
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

    kprintln!("\n  boot services released, running on our own.");

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
    kprintln!("\nhalted.");
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
