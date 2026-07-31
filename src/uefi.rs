//! Hand-written UEFI bindings.
//!
//! Deliberately from scratch -- no `uefi-rs`. Be careful in here: the field
//! order and padding of these structs *are* the ABI. Every entry in
//! `BootServices` must be declared, in the spec's order, or the handful of
//! function pointers we actually call land at the wrong offsets and we jump
//! into the middle of an unrelated firmware routine. That failure looks like a
//! spontaneous reboot with no diagnostic, so it is worth being pedantic here.
//!
//! Reference: UEFI Specification 2.10, sections 4 (Boot Manager) and 7 (Services).

#![allow(dead_code)]

use core::ffi::c_void;

pub type Status = usize;
pub type Handle = *mut c_void;

/// UEFI marks errors by setting the high bit of a native-width word.
pub const STATUS_ERROR_BIT: Status = 1 << (usize::BITS - 1);

pub const SUCCESS: Status = 0;
pub const INVALID_PARAMETER: Status = STATUS_ERROR_BIT | 2;
pub const UNSUPPORTED: Status = STATUS_ERROR_BIT | 3;
pub const BUFFER_TOO_SMALL: Status = STATUS_ERROR_BIT | 5;
pub const NOT_FOUND: Status = STATUS_ERROR_BIT | 14;

#[inline]
pub fn is_error(s: Status) -> bool {
    s & STATUS_ERROR_BIT != 0
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Guid {
    pub d1: u32,
    pub d2: u16,
    pub d3: u16,
    pub d4: [u8; 8],
}

/// EFI_GRAPHICS_OUTPUT_PROTOCOL_GUID -- 9042a9de-23dc-4a38-96fb-7aded080516a
pub const GRAPHICS_OUTPUT_PROTOCOL_GUID: Guid = Guid {
    d1: 0x9042a9de,
    d2: 0x23dc,
    d3: 0x4a38,
    d4: [0x96, 0xfb, 0x7a, 0xde, 0xd0, 0x80, 0x51, 0x6a],
};

/// EFI_LOADED_IMAGE_PROTOCOL_GUID -- 5b1b31a1-9562-11d2-8e3f-00a0c969723b
pub const LOADED_IMAGE_PROTOCOL_GUID: Guid = Guid {
    d1: 0x5b1b31a1,
    d2: 0x9562,
    d3: 0x11d2,
    d4: [0x8e, 0x3f, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b],
};

/// EFI_SIMPLE_FILE_SYSTEM_PROTOCOL_GUID -- 964e5b22-6459-11d2-8e39-00a0c969723b
pub const SIMPLE_FILE_SYSTEM_PROTOCOL_GUID: Guid = Guid {
    d1: 0x964e5b22,
    d2: 0x6459,
    d3: 0x11d2,
    d4: [0x8e, 0x39, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b],
};

/// EFI_ACPI_20_TABLE_GUID -- 8868e871-e4f1-11d3-bc22-0080c73c8881 (RSDP, ACPI 2.0+)
pub const ACPI_20_TABLE_GUID: Guid = Guid {
    d1: 0x8868e871,
    d2: 0xe4f1,
    d3: 0x11d3,
    d4: [0xbc, 0x22, 0x00, 0x80, 0xc7, 0x3c, 0x88, 0x81],
};

/// ACPI_TABLE_GUID -- eb9d2d30-2d88-11d3-9a16-0090273fc14d (RSDP, ACPI 1.0)
pub const ACPI_10_TABLE_GUID: Guid = Guid {
    d1: 0xeb9d2d30,
    d2: 0x2d88,
    d3: 0x11d3,
    d4: [0x9a, 0x16, 0x00, 0x90, 0x27, 0x3f, 0xc1, 0x4d],
};

#[repr(C)]
pub struct TableHeader {
    pub signature: u64,
    pub revision: u32,
    pub header_size: u32,
    pub crc32: u32,
    pub reserved: u32,
}

#[repr(C)]
pub struct ConfigurationTable {
    pub vendor_guid: Guid,
    pub vendor_table: *mut c_void,
}

#[repr(C)]
pub struct SystemTable {
    pub hdr: TableHeader,
    pub firmware_vendor: *const u16,
    pub firmware_revision: u32,
    pub console_in_handle: Handle,
    pub con_in: *mut c_void,
    pub console_out_handle: Handle,
    pub con_out: *mut SimpleTextOutputProtocol,
    pub standard_error_handle: Handle,
    pub std_err: *mut SimpleTextOutputProtocol,
    pub runtime_services: *mut c_void,
    pub boot_services: *mut BootServices,
    pub number_of_table_entries: usize,
    pub configuration_table: *mut ConfigurationTable,
}

#[repr(C)]
pub struct SimpleTextOutputProtocol {
    pub reset: extern "efiapi" fn(*mut Self, bool) -> Status,
    /// Takes a null-terminated UCS-2 string.
    pub output_string: extern "efiapi" fn(*mut Self, *const u16) -> Status,
    pub test_string: usize,
    pub query_mode: usize,
    pub set_mode: usize,
    pub set_attribute: usize,
    pub clear_screen: extern "efiapi" fn(*mut Self) -> Status,
    pub set_cursor_position: usize,
    pub enable_cursor: usize,
    pub mode: usize,
}

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AllocateType {
    AnyPages = 0,
    MaxAddress = 1,
    Address = 2,
}

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MemoryType {
    Reserved = 0,
    LoaderCode = 1,
    LoaderData = 2,
    BootServicesCode = 3,
    BootServicesData = 4,
    RuntimeServicesCode = 5,
    RuntimeServicesData = 6,
    Conventional = 7,
    Unusable = 8,
    AcpiReclaim = 9,
    AcpiNvs = 10,
    MappedIo = 11,
    MappedIoPortSpace = 12,
    PalCode = 13,
    Persistent = 14,
}

/// One entry of the UEFI memory map.
///
/// Never stride an array of these by `size_of::<MemoryDescriptor>()`. The
/// firmware reports its own `descriptor_size` from `GetMemoryMap`, and it is
/// allowed to be larger than this struct. Using the wrong stride is one of the
/// classic ways to walk off into garbage.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MemoryDescriptor {
    pub ty: u32,
    pub phys_start: u64,
    pub virt_start: u64,
    pub num_pages: u64,
    pub attribute: u64,
}

impl MemoryDescriptor {
    /// Memory the OS may take over once boot services are gone.
    ///
    /// `BootServicesCode`/`BootServicesData` become free the moment
    /// `ExitBootServices` returns, and `LoaderCode`/`LoaderData` hold *us*, so
    /// they are deliberately excluded.
    /// Strictly free RAM: memory the firmware never used for anything.
    ///
    /// The early frame allocator uses only this, deliberately. The looser
    /// `is_usable_after_exit` includes `BootServicesData`, and our current
    /// stack is almost certainly sitting in it -- handing our own stack out as
    /// a page table would be a spectacular and very hard-to-find bug.
    pub fn is_conventional(&self) -> bool {
        self.ty == MemoryType::Conventional as u32
    }

    pub fn is_usable_after_exit(&self) -> bool {
        matches!(self.ty, x if x == MemoryType::Conventional as u32
            || x == MemoryType::BootServicesCode as u32
            || x == MemoryType::BootServicesData as u32)
    }
}

#[repr(C)]
pub struct BootServices {
    pub hdr: TableHeader,

    // --- Task priority services ---
    pub raise_tpl: usize,
    pub restore_tpl: usize,

    // --- Memory services ---
    pub allocate_pages:
        extern "efiapi" fn(AllocateType, MemoryType, usize, *mut u64) -> Status,
    pub free_pages: usize,
    /// (map_size, map, map_key, descriptor_size, descriptor_version)
    pub get_memory_map: extern "efiapi" fn(
        *mut usize,
        *mut u8,
        *mut usize,
        *mut usize,
        *mut u32,
    ) -> Status,
    pub allocate_pool: extern "efiapi" fn(MemoryType, usize, *mut *mut u8) -> Status,
    pub free_pool: extern "efiapi" fn(*mut u8) -> Status,

    // --- Event & timer services ---
    pub create_event: usize,
    pub set_timer: usize,
    pub wait_for_event: usize,
    pub signal_event: usize,
    pub close_event: usize,
    pub check_event: usize,

    // --- Protocol handler services ---
    pub install_protocol_interface: usize,
    pub reinstall_protocol_interface: usize,
    pub uninstall_protocol_interface: usize,
    /// Typed rather than left as `usize` because the file loader calls it.
    /// Same width either way, so the struct layout is unchanged.
    pub handle_protocol:
        extern "efiapi" fn(Handle, *const Guid, *mut *mut c_void) -> Status,
    pub reserved: usize,
    pub register_protocol_notify: usize,
    pub locate_handle: usize,
    pub locate_device_path: usize,
    pub install_configuration_table: usize,

    // --- Image services ---
    pub load_image: usize,
    pub start_image: usize,
    pub exit: usize,
    pub unload_image: usize,
    pub exit_boot_services: extern "efiapi" fn(Handle, usize) -> Status,

    // --- Misc services ---
    pub get_next_monotonic_count: usize,
    pub stall: extern "efiapi" fn(usize) -> Status,
    pub set_watchdog_timer:
        extern "efiapi" fn(usize, u64, usize, *mut u16) -> Status,

    // --- Driver support services ---
    pub connect_controller: usize,
    pub disconnect_controller: usize,

    // --- Open/close protocol services ---
    pub open_protocol: usize,
    pub close_protocol: usize,
    pub open_protocol_information: usize,

    // --- Library services ---
    pub protocols_per_handle: usize,
    pub locate_handle_buffer: usize,
    pub locate_protocol:
        extern "efiapi" fn(*const Guid, *mut c_void, *mut *mut c_void) -> Status,
    pub install_multiple_protocol_interfaces: usize,
    pub uninstall_multiple_protocol_interfaces: usize,

    // --- 32-bit CRC services ---
    pub calculate_crc32: usize,

    // --- More misc services ---
    pub copy_mem: usize,
    pub set_mem: usize,
    pub create_event_ex: usize,
}

// --- Graphics Output Protocol ---

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PixelFormat {
    /// 8-bit R, G, B, then a reserved byte.
    RedGreenBlueReserved8 = 0,
    /// 8-bit B, G, R, then a reserved byte. What OVMF and most Intel iGPUs report.
    BlueGreenRedReserved8 = 1,
    /// Channel positions given by `PixelBitmask`.
    BitMask = 2,
    /// No linear framebuffer; Blt() only. We cannot use this mode after
    /// ExitBootServices, because Blt() lives in boot services.
    BltOnly = 3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PixelBitmask {
    pub red: u32,
    pub green: u32,
    pub blue: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ModeInformation {
    pub version: u32,
    pub horizontal_resolution: u32,
    pub vertical_resolution: u32,
    pub pixel_format: u32,
    pub pixel_information: PixelBitmask,
    /// Stride in *pixels*, not bytes, and often larger than the visible width.
    pub pixels_per_scan_line: u32,
}

#[repr(C)]
pub struct GraphicsOutputMode {
    pub max_mode: u32,
    pub mode: u32,
    pub info: *mut ModeInformation,
    pub size_of_info: usize,
    pub frame_buffer_base: u64,
    pub frame_buffer_size: usize,
}

#[repr(C)]
pub struct GraphicsOutputProtocol {
    pub query_mode: extern "efiapi" fn(
        *mut Self,
        u32,
        *mut usize,
        *mut *mut ModeInformation,
    ) -> Status,
    pub set_mode: extern "efiapi" fn(*mut Self, u32) -> Status,
    pub blt: usize,
    pub mode: *mut GraphicsOutputMode,
}

// --- file access --------------------------------------------------------
//
// The firmware has a working FAT driver and we boot off a FAT partition, so
// there is a filesystem available for exactly as long as boot services are.
// Reading what we need before `ExitBootServices` is much less work than
// implementing FAT, and it is how every other kernel gets its initrd.
//
// Everything read this way must be allocated as `LoaderData`: that type is
// ours, it survives ExitBootServices, `is_ram_type` maps it write-back, and
// `is_usable_after_exit` deliberately excludes it so the frame allocator will
// never hand a model's weights out as free pages.

/// Only the leading fields are declared. The protocol continues past
/// `device_handle`, but nothing here reads further and the struct is only ever
/// used behind a pointer, so the tail can stay undescribed.
#[repr(C)]
pub struct LoadedImageProtocol {
    pub revision: u32,
    pub parent_handle: Handle,
    pub system_table: *mut SystemTable,
    pub device_handle: Handle,
}

#[repr(C)]
pub struct SimpleFileSystemProtocol {
    pub revision: u64,
    pub open_volume:
        extern "efiapi" fn(*mut SimpleFileSystemProtocol, *mut *mut FileProtocol) -> Status,
}

#[repr(C)]
pub struct FileProtocol {
    pub revision: u64,
    pub open: extern "efiapi" fn(
        *mut FileProtocol,
        *mut *mut FileProtocol,
        *const u16,
        u64,
        u64,
    ) -> Status,
    pub close: extern "efiapi" fn(*mut FileProtocol) -> Status,
    pub delete: usize,
    pub read: extern "efiapi" fn(*mut FileProtocol, *mut usize, *mut u8) -> Status,
    pub write: usize,
    pub get_position: extern "efiapi" fn(*mut FileProtocol, *mut u64) -> Status,
    pub set_position: extern "efiapi" fn(*mut FileProtocol, u64) -> Status,
    pub get_info: usize,
    pub set_info: usize,
    pub flush: usize,
}

pub const FILE_MODE_READ: u64 = 0x0000_0000_0000_0001;

/// Seeking here and asking where you landed is the documented way to get a
/// file's size without the variable-length buffer dance `GetInfo` requires.
const POSITION_END_OF_FILE: u64 = 0xFFFF_FFFF_FFFF_FFFF;

/// A buffer that outlives boot services.
#[derive(Clone, Copy)]
pub struct Blob {
    pub ptr: *mut u8,
    pub len: usize,
}

impl Blob {
    /// # Safety
    /// Only valid for blobs produced by `read_file`, whose pool allocation is
    /// never freed and stays identity-mapped for the life of the system.
    pub fn as_slice(&self) -> &'static [u8] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Read a whole file from the volume this image was loaded from.
///
/// `path` is an absolute path on that volume using backslashes, e.g.
/// `\GLADOS\model.bin`. ASCII only -- it is widened to UCS-2 by zero
/// extension, which is correct for ASCII and wrong for anything else.
///
/// Returns `None` for every failure, including "not there". A missing model is
/// a normal condition, not an error: the system has to boot without one.
pub fn read_file(bs: &BootServices, image: Handle, path: &str) -> Option<Blob> {
    // image handle -> the device it was loaded from
    let mut iface: *mut c_void = core::ptr::null_mut();
    if is_error((bs.handle_protocol)(image, &LOADED_IMAGE_PROTOCOL_GUID, &mut iface)) {
        return None;
    }
    let device = unsafe { (*(iface as *mut LoadedImageProtocol)).device_handle };

    // that device -> its filesystem -> the root directory
    let mut iface: *mut c_void = core::ptr::null_mut();
    if is_error((bs.handle_protocol)(device, &SIMPLE_FILE_SYSTEM_PROTOCOL_GUID, &mut iface)) {
        return None;
    }
    let sfs = iface as *mut SimpleFileSystemProtocol;
    let mut root: *mut FileProtocol = core::ptr::null_mut();
    if is_error(unsafe { ((*sfs).open_volume)(sfs, &mut root) }) || root.is_null() {
        return None;
    }

    // UCS-2, null terminated. Bounded so a long path truncates rather than
    // writing past the array.
    let mut wide = [0u16; 128];
    if path.len() >= wide.len() {
        unsafe { ((*root).close)(root) };
        return None;
    }
    for (i, b) in path.bytes().enumerate() {
        wide[i] = b as u16;
    }

    let mut file: *mut FileProtocol = core::ptr::null_mut();
    let opened = unsafe { ((*root).open)(root, &mut file, wide.as_ptr(), FILE_MODE_READ, 0) };
    unsafe { ((*root).close)(root) };
    if is_error(opened) || file.is_null() {
        return None;
    }

    let mut size: u64 = 0;
    let sized = unsafe {
        !is_error(((*file).set_position)(file, POSITION_END_OF_FILE))
            && !is_error(((*file).get_position)(file, &mut size))
            && !is_error(((*file).set_position)(file, 0))
    };
    if !sized || size == 0 {
        unsafe { ((*file).close)(file) };
        return None;
    }

    let mut buf: *mut u8 = core::ptr::null_mut();
    if is_error((bs.allocate_pool)(MemoryType::LoaderData, size as usize, &mut buf)) {
        unsafe { ((*file).close)(file) };
        return None;
    }

    // Read() is permitted to return fewer bytes than asked for, so this loops.
    // A short read that returns zero would otherwise spin here forever.
    let total = size as usize;
    let mut done = 0usize;
    while done < total {
        let mut want = total - done;
        let st = unsafe { ((*file).read)(file, &mut want, buf.add(done)) };
        if is_error(st) || want == 0 {
            unsafe { ((*file).close)(file) };
            let _ = (bs.free_pool)(buf);
            return None;
        }
        done += want;
    }
    unsafe { ((*file).close)(file) };

    Some(Blob { ptr: buf, len: done })
}
