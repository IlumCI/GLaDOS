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
    pub handle_protocol: usize,
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
