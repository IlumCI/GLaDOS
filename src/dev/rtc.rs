//! The CMOS real-time clock.
//!
//! Everything in this system has been timeless until now. `snaps` lists a
//! history by sequence number, which tells you the order things happened and
//! nothing about when -- so "roll back to before I broke it" is a question the
//! store cannot answer even though it holds the answer.
//!
//! The RTC is the one clock that survives power loss. `lapic::ticks` counts
//! from boot and the TSC counts from reset; neither knows what year it is.
//!
//! Reading it correctly is fiddlier than it looks. The chip updates itself
//! roughly once a second and the registers are inconsistent while it does, so
//! a naive read can catch 01:59:59 halfway to 02:00:00 and return 01:00:59.
//! The standard remedy, used here, is to read twice and accept only when two
//! consecutive reads agree.

use core::arch::asm;

const INDEX: u16 = 0x70;
const DATA: u16 = 0x71;

const REG_SECOND: u8 = 0x00;
const REG_MINUTE: u8 = 0x02;
const REG_HOUR: u8 = 0x04;
const REG_DAY: u8 = 0x07;
const REG_MONTH: u8 = 0x08;
const REG_YEAR: u8 = 0x09;
const REG_STATUS_A: u8 = 0x0A;
const REG_STATUS_B: u8 = 0x0B;

#[inline]
unsafe fn outb(port: u16, val: u8) {
    unsafe {
        asm!("out dx, al", in("dx") port, in("al") val, options(nomem, nostack, preserves_flags));
    }
}

#[inline]
unsafe fn inb(port: u16) -> u8 {
    let v: u8;
    unsafe {
        asm!("in al, dx", out("al") v, in("dx") port, options(nomem, nostack, preserves_flags));
    }
    v
}

fn read_reg(reg: u8) -> u8 {
    unsafe {
        // Bit 7 of the index port gates NMI. Left clear so this does not
        // silently disable non-maskable interrupts for the rest of the boot,
        // which is a classic way to lose machine-check reports.
        outb(INDEX, reg & 0x7F);
        inb(DATA)
    }
}

fn update_in_progress() -> bool {
    read_reg(REG_STATUS_A) & 0x80 != 0
}

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct DateTime {
    pub year: u16,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
}

fn raw() -> DateTime {
    DateTime {
        second: read_reg(REG_SECOND),
        minute: read_reg(REG_MINUTE),
        hour: read_reg(REG_HOUR),
        day: read_reg(REG_DAY),
        month: read_reg(REG_MONTH),
        year: read_reg(REG_YEAR) as u16,
    }
}

#[inline]
fn from_bcd(v: u8) -> u8 {
    (v & 0x0F) + ((v >> 4) * 10)
}

/// Read the clock, or `None` if it never settles.
///
/// Bounded rather than looping forever: on a machine with no RTC the update
/// flag can read as permanently set, and a boot that hangs waiting for a clock
/// is a worse outcome than one that admits it has none.
pub fn now() -> Option<DateTime> {
    let mut guard = 0u32;
    while update_in_progress() {
        guard += 1;
        if guard > 1_000_000 {
            return None;
        }
    }

    let mut last = raw();
    for _ in 0..16 {
        let next = raw();
        if next == last && !update_in_progress() {
            let status = read_reg(REG_STATUS_B);
            let binary = status & 0x04 != 0;
            let hour24 = status & 0x02 != 0;

            let mut hour_raw = last.hour;
            // In 12-hour mode bit 7 marks PM, and it survives the BCD
            // conversion as garbage unless masked off first.
            let pm = !hour24 && (hour_raw & 0x80) != 0;
            hour_raw &= 0x7F;

            let mut dt = DateTime {
                second: if binary { last.second } else { from_bcd(last.second) },
                minute: if binary { last.minute } else { from_bcd(last.minute) },
                hour: if binary { hour_raw } else { from_bcd(hour_raw) },
                day: if binary { last.day } else { from_bcd(last.day) },
                month: if binary { last.month } else { from_bcd(last.month) },
                year: if binary { last.year } else { from_bcd(last.year as u8) as u16 },
            };

            if pm && dt.hour < 12 {
                dt.hour += 12;
            } else if hour24 || !pm {
                // 12 AM reads as 12 in 12-hour mode and means midnight.
                if !hour24 && dt.hour == 12 {
                    dt.hour = 0;
                }
            }

            // The century register is not reliably present, so the two-digit
            // year is windowed instead. Anything below 70 is this century.
            dt.year += if dt.year < 70 { 2000 } else { 1900 };

            if dt.month == 0 || dt.month > 12 || dt.day == 0 || dt.day > 31 {
                return None;
            }
            return Some(dt);
        }
        last = next;
    }
    None
}

const DAYS_BEFORE_MONTH: [u32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];

fn is_leap(y: u32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// Seconds since 1970, as a u32.
///
/// Chosen over a u64 because the manifest has exactly four spare bytes between
/// its entry count and the entries themselves, and a timestamp that fits there
/// costs no format change and leaves older manifests readable with a zero.
/// It overflows in 2106, which this will not outlive.
pub fn unix_seconds(dt: &DateTime) -> u32 {
    let y = dt.year as u32;
    if y < 1970 {
        return 0;
    }
    let mut days: u32 = 0;
    for year in 1970..y {
        days += if is_leap(year) { 366 } else { 365 };
    }
    days += DAYS_BEFORE_MONTH[(dt.month as usize - 1).min(11)];
    if dt.month > 2 && is_leap(y) {
        days += 1;
    }
    days += dt.day.saturating_sub(1) as u32;

    days * 86_400 + dt.hour as u32 * 3600 + dt.minute as u32 * 60 + dt.second as u32
}

/// Turn a stored timestamp back into a date.
pub fn from_unix(mut secs: u32) -> DateTime {
    let mut year = 1970u32;
    loop {
        let len = if is_leap(year) { 366 } else { 365 } * 86_400;
        if secs < len {
            break;
        }
        secs -= len;
        year += 1;
    }
    let mut day_of_year = secs / 86_400;
    let rem = secs % 86_400;

    let mut month = 1u8;
    for m in (0..12).rev() {
        let mut start = DAYS_BEFORE_MONTH[m];
        if m >= 2 && is_leap(year) {
            start += 1;
        }
        if day_of_year >= start {
            month = m as u8 + 1;
            day_of_year -= start;
            break;
        }
    }

    DateTime {
        year: year as u16,
        month,
        day: day_of_year as u8 + 1,
        hour: (rem / 3600) as u8,
        minute: ((rem % 3600) / 60) as u8,
        second: (rem % 60) as u8,
    }
}

/// Round-trip check, run at boot.
///
/// Calendar arithmetic is easy to get subtly wrong in a way that only shows up
/// in February, or in a leap year, or after a month boundary -- and a
/// timestamp that is wrong by a day looks entirely plausible in a listing.
pub fn selftest() -> bool {
    let cases = [
        DateTime { year: 1970, month: 1, day: 1, hour: 0, minute: 0, second: 0 },
        DateTime { year: 2000, month: 2, day: 29, hour: 12, minute: 0, second: 0 },
        DateTime { year: 2024, month: 2, day: 29, hour: 23, minute: 59, second: 59 },
        DateTime { year: 2026, month: 7, day: 31, hour: 14, minute: 32, second: 5 },
        DateTime { year: 2099, month: 12, day: 31, hour: 23, minute: 59, second: 59 },
    ];
    for c in cases {
        if from_unix(unix_seconds(&c)) != c {
            return false;
        }
    }
    // 2000 was a leap year and 1900 was not, which is the rule most
    // implementations get wrong.
    unix_seconds(&DateTime { year: 2000, month: 3, day: 1, ..Default::default() })
        - unix_seconds(&DateTime { year: 2000, month: 2, day: 28, ..Default::default() })
        == 2 * 86_400
}
