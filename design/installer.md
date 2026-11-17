# GLaDOS 1.4.5 — the installer, persistence, and two install images

## Context

1.4.4 runs from a USB stick: the slim Linux root is read into RAM at boot and
everything `orange` installs vanishes at reboot ("the installer is 1.4.5").
1.4.5 makes GLaDOS installable onto the laptop's NVMe — alone or next to
Windows — with a userland that persists, system updates with rollback, and two
images: **Netinstall** (tiny) and **Base** (model + trimmed desktop, ≤2 GB).
The Hugging Face model picker and on-device conversion are **1.4.6**; 1.4.5
must not block them.

Exploration found the kernel today cannot: format FAT, mkdir or write long
names on FAT, write boot entries, keep its own EFI file, pick a system slot
(it mounts the *first* EROFS partition), or persist a desktop-sized layer (no
metadata, files whole in heap, `cas::dma` leaks per write, no GC, writes only
after a manual `store unlock`, `fsync` is a no-op). Guests have no block
devices; after ExitBootServices the USB stick is unreadable, so install
sources are RAM or network.

## Decisions (user, 2026-11-16)

| | |
|---|---|
| Images | Netinstall: kernel + minimal userland + installer; user picks package sets (Minimal/Desktop/Office/Media); model optional, default yes. Base: ≤2 GB, model + trimmed desktop, heavier sets online |
| Disk | Erase whole disk, or alongside an existing OS in free space; never shrink; MBR refused alongside |
| UI | Native desktop wizard (DeskApp) + `install` shell verb driving the same steps |
| Updates | Image + layer: whole signed system images into the spare A/B slot with rollback; `orange install` persists on top |
| Packages | Each system image pins a dated Arch snapshot; layered installs match it |
| Dual boot | GLaDOS startup menu; picking Windows sets `BootNext` and restarts (BitLocker-safe); direct chainload only as fallback |
| Encryption | Not in 1.4.5; leave room in the format |
| Models | Change takes effect after reboot. HF picker + on-device conversion in 1.4.6 |

## Architecture

**Disk (GPT, already planned in `src/install/plan.rs`, 35 claims):** own ESP
4 GiB · system A / system B EROFS 6 GiB each (`ROOT_TYPE_GUID`) · CAS store
for the rest (`GLADOS_TYPE_GUID`). Raise `store_min` (desktop layer ≈2.2 GB).

**Writable layer = the CAS store evolved, not ext4.** Keeps atomic
dual-superblock commits, snapshots/rollback, and one commit for `/home`, `/etc`
and the kernel namespace; ext4-rw is thousands of lines with jbd2 and no fsck.
- Metadata on directory entries: `Dir(Vec<DirEnt{name, attr: Option<Box<Attr>>, node}>)`,
  `Attr{mode, uid, gid, mtime_ns, xattrs}`. Hashed **conditionally** (new kind
  bytes 2/3 only when `attr` is set) so every existing address is unchanged;
  reserve kind 4 for extent-list blobs. Blobs stay `sha256(bytes)`.
- Sidecars (`/linux/mode`, `/linux/link`) migrate lazily; `chown`,
  `utimensat`, `setxattr` (`user.*`, `security.capability`) become real;
  hardlinks remain copies in 1.4.5.
- Cached directory hashes invalidated along mutated paths (per-transaction
  commits over ~49k files cannot rehash everything).
- `src/store/space.rs`: free-extent map stored as a manifest entry; frees
  deferred two commits (mount may fall back to the older superblock); retained
  roots (current, pre-transaction, pre-flip, last K) → mark/sweep `store gc`.
- Guest file bodies `Resident | Stored(ChunkRef) | Spool(extent)`: ranged reads
  via `cas::read_blocks`; large writes stream to a spool extent, become
  `Node::Away` at close.
- Commit on `fsync`/`sync`/guest exit/shutdown + short dirty debounce; on an
  installed disk the store window unlocks at boot automatically.

**Slots and updates:** manifest gains `system-*` and `arch-snapshot` keys
(signed by the existing update key); `/dev/glados/slot` is a write-only device
armed by `update system fetch` (size + digest + target), fed by guest `curl`,
hashed in-flight, mounted-checked before the slot record moves. Slot record =
two alternating checksummed 512-byte files on the ESP (`GLADOS\SLOT0.REC`,
`SLOT1.REC`). Pure `install::slot::decide` (shape of `update::decide`);
`auto_root` mounts the slot the record names (needs `Partition.unique_guid`),
"first EROFS" stays for live media. Health: desktop reached → `SLOT.OK`, else
roll back. On flip, pure `ns::reconcile` drops layer copies of packages the new
image now owns (`/etc` keeps user copy, `.pacnew` rule), snapshot first.

## Updating an installed system

Three things move independently and one release record ties them:

| Part | Size | Channel today | 1.4.5 |
|---|---|---|---|
| Kernel `BOOTX64.EFI` | ~5 MB | signed manifest → `STAGED.EFI` + health flag + `BOOTX64.OLD` (exists) | unchanged, now staged to *our* ESP path `\EFI\GLADOS\` |
| System image (EROFS) | 0.7–2 GB | none | new: spare slot via `/dev/glados/slot` |
| User layer (`orange`) | varies | pacman | persists; reconciled on flip |
| Model | 0.6–2 GB | none | `[glados]` package → ESP, after reboot (1.4.6 picker) |

**Most releases are kernel-only.** The manifest says per release whether the
system image changed (`system-sha256` same as installed → nothing to fetch), so
a frequent release cadence costs ~5 MB per update; only releases that move the
userland cost an image.

**Policy (user):** download and prepare quietly in the background, then a
desktop notice offers "restart to update"; the user picks when.

**A full release, step by step:**
1. `update check` (or the background check) fetches the signed manifest:
   kernel version + digest, system version + digest + `arch-snapshot`,
   `min-kernel`, `store-format`.
2. Download the system image into the spare slot (verified in flight, mount
   check), then stage the kernel. Nothing the running system uses is touched.
3. Write one *pending release* record naming both. Reboot (user chooses when).
4. Next boot: `update::hook` swaps the kernel (trial), `slot::decide` picks the
   pending slot (trial). Snapshot the layer, run `ns::reconcile`.
5. Desktop reached → both marked healthy together. Any failure before that →
   the next boot restores `BOOTX64.OLD`, the old slot and the pre-flip layer
   snapshot. A kernel and an image never run mismatched across a rollback.

**Format rule (makes rollback real):** a new kernel may *read* every older
store format but must not *write* a format the previous kernel can't read until
the release is marked healthy. Store/layer format carries a version; migration
is lazy and conditional-hash (old trees stay valid), so skipping versions
(1.4.5 → 1.4.9) is one hop: images are whole, migrations are cumulative.

**Coming from 1.4.4 and earlier:** nothing was installed (USB live), so the
path is "boot the 1.4.5 image and install". If the installer finds an existing
GLaDOS store (old unpartitioned-tail store or deploy-to-SSD layout), it offers
to import its namespace (`/ai` memories, `about`, ledger, corpus) into the new
store rather than discarding it.

**Later, not 1.4.5:** delta system updates — the image is chunked by content
hash, so only chunks the current slot lacks need downloading (casync-style).

## Phases (each driven under QEMU on a blank NVMe image, `-smp 4`, kvm)

**P0 — foundations. Done.** `cas::DmaBuf` (owned, freed; not a shared
scratch, since `get` and `put` run on different tasks) + 2 MiB streaming;
one NVMe write window per holder (fixed: `update stage` used to close the
store's window); controller lock already existed (`nvme::with`);
`store wbench`: SHA-256 is ~250 MiB/s under KVM and is the bound on slot
writes (SHA-NI later). Cached dir hashes moved to P3/P4: measure commit
time once bodies are on disk, cache only if it shows.

**P1 — `install` verb, erase mode. Done:** sfdisk verifies the table, slot A
byte-identical to the root, reboot mounts / from slot A and restores the store.
Refuses 4Kn disks until `store::erofs` stops assuming 512-byte sectors.
Was: `src/install/apply.rs`: GPT write (backup
first, then primary, read back through `block::parse_gpt_entries`, confirm by
typed digest), slot A raw from the RAM `root.erofs`, store formatted and
adopted. `auto_root` by type GUID. Verify: 64 GiB blank image, reboot sees
slot A.

**P2 — bootable disk. Done:** fsck.fat clean, kernel on the ESP byte-equal
to the build, reboot with NVRAM kept boots the installed disk through
`Boot####` (`INSTALL.ID` printed), model loads from it, `ask` answers. The
kernel sits at `\EFI\BOOT\BOOTX64.EFI` on our own ESP, not `\EFI\GLADOS\`:
no collision possible, and `update stage` keeps swapping the same path.
Was: Keep `\EFI\BOOT\BOOTX64.EFI` file bytes pre-EBS
(`main.rs`); `src/store/fatfmt.rs` = port of `tools/mkiso.py::build_fat` (pure
layout plan with LFN + directories, then large sequential writes), checked by
formatting into RAM and reading back with `store::fat`; copy kernel, model,
tokenizer, FW, roots, voice, brain; `src/install/bootopt.rs` builds
`EFI_LOAD_OPTION` (HD + File device path), free `Boot####`, prepend
`BootOrder` (vectors checked against efibootmgr output). Verify: second boot
from NVMe only, no ISO; host `fsck.fat -n`.

**P3 — metadata + auto-unlock. Done** (commit-on-fsync moved to P4, where
writes stop being whole-file): real `chmod`/`chown`/`touch -d`/`setcap` on
an installed disk survive a reboot, read back by `stat` and `getcap`; the
store unlocks itself only on a disk with GLaDOS slots. Planted unconditional
attribute rendering fails the golden hash and halts boot (sysbox is vital).
One `diag all` showed `migrate` heap +2 MB; it passes alone twice.
Was: Golden-hash
claim (planted unconditional render must fail it); `chmod/chown/touch`
survive reboot.

**P4 — disk-backed bodies, space map, GC. Done:**
- blobs ≥ 64 KiB become `Node::Away` once stored (also under a cached
  ancestor): 297 MB heap → 88 MB across a snapshot of two 100 MB files;
- large stored files open read-only as range-read `DiskFile`s; `ld.so`
  loaded glibc from the layer, `tail` of 100 MB read the end only;
- mark-and-sweep keeping 8 checkpoints, free runs in memory only (format
  unchanged); a second independent walk must agree or nothing is freed —
  a planted marker bug freed live data *without* the first test noticing
  (first fit missed it), which is why the cross-check exists, and it caught
  the same plant; reclaimed space reused, reboot + fsck clean;
- the Linux layer evicts from 4 KiB (kernel namespace keeps 64 KiB): an
  82 MB copy of Python's stdlib took the heap +89 MB and back to +6.5 MB
  after the snapshot; a full desktop install (244 packages) persisted and
  `pacman -Qkq` is clean after reboot; heap growth after that is the root
  image's bounded caches (`mem` reports them) — a second identical query
  adds 704 B;
- `fsync`/`sync` commit; a power-cut test keeps the fsynced file and loses
  the unsynced one; writes commit at 256 MiB pending, since autosnap never
  runs while a guest holds the shell.
Was: Verify: install the Desktop set on
an installed disk, reboot, `pacman -Qk` clean, heap flat; torn-superblock claim
(planted immediate free must fail).

**P5 — system updates.** Manifest keys, `/dev/glados/slot`, slot records,
`slot::decide`, one pending-release record coupling kernel + slot, the
format rule, import of an old store, reconcile, pinned Arch snapshot in the image's `pacman.conf`
(`tools/rootfs.py`). Verify: two images a version apart flip; a deliberately
broken one rolls back.

**P6 — alongside + startup menu.** Alongside mode in `apply.rs` (kept slots'
raw 128-byte entries overlaid from disk); `src/boot/menu.rs` pre-EBS, shown
when another ESP has `\EFI\Microsoft\Boot\bootmgfw.efi`; Windows via
`BootNext` + reset (first find why a pre-EBS `ResetSystem` went silent),
chainload fallback. Verify: fake Windows ESP whose `bootmgfw.efi` is OVMF's
shell.

**P7 — wizard and images.** DeskApp pages (disk → mode → plan shown exactly
→ typed confirmation → progress → reboot), each page calling the same
`install` steps; Netinstall root (`rootfs.py --net`) + package-set
definitions (`packaging/glados-set-*` meta packages); model as a `[glados]`
package (`glados-model-qwen3-0.6b`) copied store→ESP after install; Base
trimmed to ≤2 GB; `release.yml` builds both. Update `CLAUDE.md`,
`design/installer.md`, release notes.

**1.4.6 groundwork kept in view (not built):** model path on ESP is per-file so
a second model can be staged; store→ESP copy path is the "switch model after
reboot" mechanism; space reserved on the ESP for two models.

## Critical files

`src/install/{plan,gpt,apply,slot,bootopt}.rs` · `src/store/{cas,fatfmt,space,fatw,block}.rs`
· `src/sysbox/{tree,mod,stored}.rs` · `src/linux/{fs,ns,disk,dev,link,syscall}.rs`
· `src/dev/nvme.rs` · `src/update/{mod,manifest,stage}.rs` · `src/main.rs`
· `src/cpu/mod.rs` (efi vars) · `tools/{mkiso,rootfs,payload}.py`
· `.github/workflows/release.yml` · `packaging/`.

Reuse: `install::plan/gpt` (done), `block::parse_gpt_entries`,
`update::decide` shape, `update::stage::put_verified`, `cas::Store::commit`,
`sysbox::stored` ranged reads, `cpu::efi_get/set_variable`,
`mkiso.build_fat` logic, `update::manifest` signing.

## Verification

- Every pure decision (plan, gpt, slot::decide, space alloc/sweep, reconcile,
  bootopt encoding, fatfmt layout) is a `diag` suite asserted at boot, each
  with one planted bug proven to fail it.
- End-to-end per phase via `tools/drive.py --release` builds with
  `--esp-on-nvme` and blank NVMe images; final run: boot Base ISO → install
  (erase) → reboot from disk → `orange install` Desktop set → reboot →
  packages present, `diag all` clean; repeat alongside a fake Windows disk.
- On the GF63 (owed, needs the laptop): MSI firmware keeps our `Boot####`,
  honours `BootNext`, BitLocker untouched, heap and write throughput figures.

## Risks to measure first

GF63 heap during a desktop-set install; NVMe write throughput for multi-GB
slot/ESP writes; MSI firmware behaviour with two ESPs and boot variables;
BitLocker; tree depth cap 32 (raise to 64 or iterate).
