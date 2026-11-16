//! Where an install goes: the disk as it is, plus a choice, gives one exact
//! partition table or one named refusal.
//!
//! Pure, in `update::decide`'s shape. Nothing here reads a device or writes
//! one, so every case below is asserted at boot with no disk attached, and the
//! wizard shows the operator *this* plan -- the same struct that is later
//! encoded and written -- rather than a description of one.
//!
//! **Two modes, and the second is the dangerous one.**
//!
//! - `Erase` gives the whole disk a new table. What was on it is ignored, not
//!   read: the operator chose to lose it, and the wizard says so in those
//!   words before anything is asked for.
//! - `Alongside` uses the largest stretch of unallocated space on a GPT disk
//!   and touches nothing else. Every existing entry is carried into the new
//!   array *in its own slot*, byte for byte, because Windows' BCD and
//!   firmware boot entries name partitions by GUID and some tools by index.
//!   Shrinking a partition is never offered: resizing NTFS is how somebody
//!   loses a thesis, and the space can be made from Windows' own Disk
//!   Management in a minute.
//!
//! **GLaDOS gets its own ESP, always**, including alongside Windows. A Windows
//! ESP is typically 100 to 260 MiB and the model alone is 570; UEFI permits
//! several ESPs, the firmware boots ours by its own `Boot####` entry, and the
//! two systems never write each other's boot volume.
//!
//! **An MBR disk is refused alongside, not converted.** Converting a live
//! MBR disk to GPT in place is possible and is exactly the operation whose
//! failure leaves a machine that boots nothing. Erase still works on one.

use super::gpt::{self, Entry, Geometry};
use crate::store::block::{self, Partition, Scheme};
use alloc::vec::Vec;

/// GLaDOS root slot, an EROFS system image. Text form
/// dc41c180-f425-45aa-b8c6-cabf243f971d; on-disk mixed-endian here.
pub const ROOT_TYPE_GUID: [u8; 16] = [
    0x80, 0xc1, 0x41, 0xdc, 0x25, 0xf4, 0xaa, 0x45, 0xb8, 0xc6, 0xca, 0xbf, 0x24, 0x3f, 0x97, 0x1d,
];

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    Erase,
    Alongside,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Role {
    Esp,
    RootA,
    RootB,
    Store,
}

impl Role {
    pub fn type_guid(self) -> [u8; 16] {
        match self {
            Role::Esp => block::GUID_ESP,
            Role::RootA | Role::RootB => ROOT_TYPE_GUID,
            Role::Store => crate::store::cas::GLADOS_TYPE_GUID,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Role::Esp => "GLaDOS boot",
            Role::RootA => "GLaDOS system A",
            Role::RootB => "GLaDOS system B",
            Role::Store => "GLaDOS store",
        }
    }
}

/// How much each part wants, in bytes.
#[derive(Clone, Copy, Debug)]
pub struct Want {
    /// Kernel, firmware, and the weights the firmware reads before
    /// ExitBootServices -- room for two models, so one can be staged beside
    /// the one in use.
    pub esp: u64,
    /// Each of the two system slots. The full root is 2.4 GB today.
    pub root_slot: u64,
    /// The least the writable store may have; it takes everything left.
    pub store_min: u64,
}

impl Want {
    pub const DEFAULT: Want = Want { esp: 4 * GIB, root_slot: 6 * GIB, store_min: 8 * GIB };
    pub fn total_min(&self) -> u64 {
        self.esp + 2 * self.root_slot + self.store_min
    }
}

/// The disk as `block::scan` found it.
pub struct Disk {
    pub geometry: Geometry,
    pub scheme: Scheme,
    pub partitions: Vec<Partition>,
}

/// A new partition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Part {
    pub role: Role,
    pub first_lba: u64,
    pub blocks: u64,
}

pub struct Plan {
    pub mode: Mode,
    /// Existing GPT entries carried across, each in its original slot.
    pub kept: Vec<Partition>,
    pub new: Vec<Part>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Refusal {
    /// The block size is not one a GPT can be written for here.
    Geometry,
    /// The whole disk is smaller than an install, in bytes.
    DiskTooSmall { need: u64, have: u64 },
    /// Alongside asked of a disk with no GPT.
    NotGpt,
    /// Alongside, and the largest free stretch is too small, in bytes.
    NoRoom { need: u64, largest: u64 },
    /// Alongside, and GLaDOS is already here. Reinstalling is its own path.
    AlreadyInstalled,
    /// Alongside, and the existing array has no four free slots.
    NoSlots,
}

impl Refusal {
    pub fn say(&self) -> alloc::string::String {
        use alloc::format;
        match *self {
            Refusal::Geometry => "this disk's block size is not one GLaDOS can partition".into(),
            Refusal::DiskTooSmall { need, have } => format!(
                "the disk is {} GiB and an install needs at least {} GiB",
                have / GIB,
                need.div_ceil(GIB)
            ),
            Refusal::NotGpt => "the disk uses an MBR table; installing alongside needs GPT, \
                                and converting a disk in use is not offered -- erase it instead, \
                                or convert it from the system that owns it"
                .into(),
            Refusal::NoRoom { need, largest } => format!(
                "the largest unallocated space is {} GiB and GLaDOS needs {} GiB -- \
                 shrink a volume from the system that owns it, then try again",
                largest / GIB,
                need.div_ceil(GIB)
            ),
            Refusal::AlreadyInstalled => "GLaDOS is already installed on this disk".into(),
            Refusal::NoSlots => "the partition table has no room for four more entries".into(),
        }
    }
}

/// Partitions start on a 1 MiB boundary, which every SSD's erase block and
/// every tool since Vista agree on.
fn align(g: &Geometry) -> u64 {
    (MIB / g.block_size as u64).max(1)
}

fn up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

fn blocks_of(g: &Geometry, bytes: u64) -> u64 {
    up(bytes.div_ceil(g.block_size as u64), align(g))
}

/// Lay out the four parts inside `[start, end]` (inclusive), store last and
/// taking the rest. `None` if they do not fit.
fn lay(g: &Geometry, want: &Want, start: u64, end: u64) -> Option<Vec<Part>> {
    let a = align(g);
    let mut at = up(start, a);
    let mut out = Vec::new();
    for (role, bytes) in [
        (Role::Esp, want.esp),
        (Role::RootA, want.root_slot),
        (Role::RootB, want.root_slot),
    ] {
        let n = blocks_of(g, bytes);
        out.push(Part { role, first_lba: at, blocks: n });
        at = at.checked_add(n)?;
    }
    // The store ends on an alignment boundary too, so a later tool that rounds
    // never finds it overhanging the usable area.
    let stop = ((end + 1) / a) * a;
    if stop <= at {
        return None;
    }
    let store = stop - at;
    if store < blocks_of(g, want.store_min) {
        return None;
    }
    out.push(Part { role: Role::Store, first_lba: at, blocks: store });
    Some(out)
}

/// Free stretches between `first_usable` and `last_usable`, inclusive bounds.
pub fn gaps(g: &Geometry, parts: &[Partition]) -> Vec<(u64, u64)> {
    let mut used: Vec<(u64, u64)> = parts.iter().map(|p| (p.start_lba, p.end_lba())).collect();
    used.sort();
    let mut out = Vec::new();
    let mut at = g.first_usable();
    for (s, e) in used {
        if s > at {
            out.push((at, s - 1));
        }
        at = at.max(e);
    }
    if at <= g.last_usable() {
        out.push((at, g.last_usable()));
    }
    out
}

pub fn plan(disk: &Disk, mode: Mode, want: &Want) -> Result<Plan, Refusal> {
    let g = &disk.geometry;
    let bs = g.block_size as u64;
    if bs < 512 || !bs.is_power_of_two() || bs > 4096 {
        return Err(Refusal::Geometry);
    }
    let have = g.blocks.saturating_mul(bs);
    // The tables and the alignment slack on top of the parts.
    let need = want.total_min() + 4 * MIB;
    if have < need || g.blocks < 2 * g.first_usable() + 2 {
        return Err(Refusal::DiskTooSmall { need, have });
    }
    match mode {
        Mode::Erase => {
            let new = lay(g, want, g.first_usable(), g.last_usable())
                .ok_or(Refusal::DiskTooSmall { need, have })?;
            Ok(Plan { mode, kept: Vec::new(), new })
        }
        Mode::Alongside => {
            if disk.scheme != Scheme::Gpt {
                return Err(Refusal::NotGpt);
            }
            if disk.partitions.iter().any(|p| {
                p.type_guid == crate::store::cas::GLADOS_TYPE_GUID || p.type_guid == ROOT_TYPE_GUID
            }) {
                return Err(Refusal::AlreadyInstalled);
            }
            let taken = disk.partitions.iter().filter(|p| p.index as usize <= gpt::ENTRIES).count();
            if taken + 4 > gpt::ENTRIES {
                return Err(Refusal::NoSlots);
            }
            let mut best: Option<(u64, u64)> = None;
            for (s, e) in gaps(g, &disk.partitions) {
                if best.map_or(true, |(bs_, be)| e - s > be - bs_) {
                    best = Some((s, e));
                }
            }
            let largest = best.map_or(0, |(s, e)| (e - s + 1) * bs);
            let need = want.total_min();
            let (s, e) = best.ok_or(Refusal::NoRoom { need, largest })?;
            let new = lay(g, want, s, e).ok_or(Refusal::NoRoom { need, largest })?;
            Ok(Plan { mode, kept: disk.partitions.clone(), new })
        }
    }
}

impl Plan {
    /// The entry array this plan writes, in slot order: kept entries where
    /// they were, new ones in the lowest free slots. `uniq(i)` supplies the
    /// i-th new partition's unique GUID, so the caller decides where those
    /// come from (`rng` on a device, constants in a claim).
    ///
    /// A kept entry is rebuilt from what `block::scan` returned, which is its
    /// type, start and length. **Its unique GUID, attributes and name are not
    /// carried**, because the scan never read them -- so `install::apply`
    /// copies kept slots' raw 128 bytes over these before writing, and this
    /// array is the layout check, never the bytes.
    pub fn entries(&self, uniq: impl Fn(usize) -> [u8; 16]) -> Vec<(usize, Entry)> {
        let mut out: Vec<(usize, Entry)> = self
            .kept
            .iter()
            .map(|p| {
                (
                    p.index as usize - 1,
                    Entry {
                        type_guid: p.type_guid,
                        unique_guid: [0xFF; 16],
                        first_lba: p.start_lba,
                        blocks: p.block_count,
                        name: "",
                    },
                )
            })
            .collect();
        let mut slot = 0usize;
        for (i, n) in self.new.iter().enumerate() {
            while out.iter().any(|(s, _)| *s == slot) {
                slot += 1;
            }
            out.push((
                slot,
                Entry {
                    type_guid: n.role.type_guid(),
                    unique_guid: uniq(i),
                    first_lba: n.first_lba,
                    blocks: n.blocks,
                    name: n.role.name(),
                },
            ));
            slot += 1;
        }
        out.sort_by_key(|(s, _)| *s);
        out
    }

    pub fn part(&self, role: Role) -> Option<Part> {
        self.new.iter().copied().find(|p| p.role == role)
    }
}
