//! Device drivers.

pub mod power;
pub mod e1000;
pub mod ioapic;
pub mod kbd;
pub mod lapic;
pub mod nvme;
pub mod pci;
pub mod pic;
pub mod rtc;
pub mod mouse;
pub mod rtl8168;
pub mod rtl8188eu;
pub mod rtl8188eu_tables;
pub mod usbhid;
pub mod xhci;

/// Interrupt vector assignments.
///
/// These start at 0x20 because 0x00..0x1F belong to CPU exceptions. The legacy
/// PIC is remapped clear of these (to 0x30) rather than onto them, so that a
/// spurious IRQ7 from the masked PIC can never be mistaken for our timer.
pub const VECTOR_TIMER: u8 = 0x20;
pub const VECTOR_KEYBOARD: u8 = 0x21;
pub const VECTOR_MOUSE: u8 = 0x22;
pub const VECTOR_SERIAL: u8 = 0x23;
pub const VECTOR_SPURIOUS: u8 = 0xFF;
