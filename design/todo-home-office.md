# To do: a home and office OS

Asked for on 2026-10-25. The order below is the order to build in. Each part is
a rung that stands on the one before it.

## 0. Applications run without hand-staging (in progress, first)

Every LibreOffice hurdle was the OS failing to provide something a Linux app
takes for granted: a file under `/usr/share`, a dlopen'd library, a writable
cache, a case-preserving name, a symlink, a session service, a protocol. Each
was fixed by hand for one app. The OS has to provide all of it, so any app
just runs.

1. **A real root filesystem for guests.** Build a distribution root on the
   host (Arch packages, resolved with `pacman -Sp` and unpacked, so no root
   is needed) into an **EROFS** image: Linux's read-only filesystem, with
   symlinks, modes and owners, and simple when uncompressed. The kernel gets
   an EROFS reader and mounts it at `/` for guests.
2. **A writable layer over it.** `/tmp`, `/home`, `/etc`, `/var`, the
   profile directories and caches: writes go to an overlay (memory first,
   then persistent on the store), reads fall through to the image.
3. **Session services started by the OS.** A D-Bus session bus,
   `XDG_RUNTIME_DIR` and the XDG directories, writable font caches,
   settings storage (dconf or GSettings' keyfile backend).
4. **Wayland completeness.** `xdg_popup` (no menu can open today), a real
   clipboard, cursor shapes, text-input-v3, subsurfaces as needed; XWayland
   later, for X11-only apps.
5. **System-call coverage by sweep.** Run many apps, not one; implement
   what the `-ENOSYS` trace and the throw/exit shim name.
6. **The `orange` package manager** installs into the writable layer from
   the same package source, which is what makes this a user-facing feature
   rather than a build step.

**State on 2026-10-27, and the next step.** Items 1-3 are done (`b436ded`,
`c7f761a`, `e7233c5`), and 4 is mostly done: popups, subsurfaces, wheel,
decorations, focus (`0743f6a`). LibreOffice, foot and bash, and a GTK 4 app
all start. What stops several GUI apps at once is memory:

- **Demand paging for anonymous `mmap` -- next.** glibc gives each thread a
  malloc arena by mapping 128 MiB `PROT_NONE` and trimming it to 64 MiB;
  Mesa, Qt and every thread stack reserve large and touch little. Here
  `sys_mmap` backs every page up front (the `anon_paged` branch, a page at a
  time under the guest's own tables), so a GTK 4 app with a dozen threads
  asked for gigabytes, `[mmap] out of memory` filled the screen, and the
  kernel heap ran dry with three programs running. Plan:
  1. In the `anon_paged` branch, take the arena address and write each leaf
     as a **non-present entry with a software marker** (an available bit,
     e.g. bit 9, plus the rights in the usual bits) instead of allocating.
     The rights then live in the entry, so `mprotect` keeps working on a
     page nobody has touched -- `paging::protect` must keep the marker and
     rewrite only the rights on a non-present entry.
  2. `glados_fault_entry` returns for a `#PF` (vector 14) whose `cr2`
     walks to a marked entry whose rights allow the access (error code bits
     W and U against the marker's rights): allocate a zeroed page, install
     it with `cmpxchg` on the entry (two threads may fault the same page;
     the loser frees its page), `invlpg`, return. `glados_fault_common`
     ends in `ud2` today; it needs a return path that pops the fifteen
     registers, drops vector and error code, and `iretq`s. From ring 0
     too: the kernel writing a `read` result into an untouched guest page
     is the same fault.
  3. `reachable` treats a marked page as reachable (or faults it in), or
     every syscall into fresh memory answers `EFAULT`.
  4. `munmap`/`free_paged`/teardown free only present frames; `fork`'s copy
     copies a marked entry as a marked entry.
  5. Proof: three GUI programs at once (FeatherPad, `gtk4-widget-factory`,
     LibreOffice) and the heap figure in `mem` before and after.
- Then: run each test app (FeatherPad, `gtk4-widget-factory`, mpv, imv,
  `weston-simple-*`) and fix what each names; inotify; a real clipboard.
- **No GNOME apps.** GTK 4 is tested with its own demos (`gtk4-demos`).

**Iterating faster.** `cargo build --profile quick` (no LTO) rebuilds in
about 45 s against 2.5 min for release, and `drive.py` boots whichever of the
two is newer. A guest that resets now pauses, so `drive.py` reports
`STOPPED` within seconds and writes every vCPU's registers to
`.qemu/regs.txt` instead of sitting out its timeout. Still to do: start
several apps in one boot, and wait for a window rather than a fixed beat.

## 0b. sci-fish, the guest shell

The Linux side's own shell: **sci-fish** ("sci-fi sh"). It should look like
oh-my-zsh, with a themed, informative prompt showing git status, the exit
code of the last command and the time a command took. It should feel and
behave like fish:
- suggestions from history as you type;
- syntax highlighting while typing;
- tab completion that knows commands and their options;
- sane scripting (no word-splitting surprises);
- a web-free `config` for themes.

It is the program a Linux terminal window starts. It ships on the root
image, so nothing has to be installed for it.

Open decision: build it as a configured fish (fish ships in Arch, so it is
a package plus a theme and a prompt), or write our own. Fish first is the
fast way to the feel; a rewrite only if fish's own limits get in the way.
Either way it needs a guest terminal: a pty and a terminal window that
renders its escape codes.

## 1. Office work, with the model helping (in progress)

- **LibreOffice in a Linux guest.** The host's build and its library closure
  (704 MB) sit on a FAT32 disk, `.qemu/office.img`. The builder is
  `out/office/mkoffice.sh` and the runner is `out/office/run.sh`; both are
  local and not in the repo.
  - First rung: `soffice.bin --headless --convert-to pdf`. On the first try it
    loaded all its libraries, made its profile, and then aborted with an
    uncaught `uno::RuntimeException`. That is being traced now.
  - Then the GUI on Skywalker, through LibreOffice's gtk3 or qt6 Wayland plugin.
- **The model doing office tasks.** Headless conversion and filling a document
  from a template are things the agent can call. Two rules:
  - An action that writes is gated like every other mutating applet.
  - The 0.6B gets measured before anything is claimed for it.
- **Trading agents.** These run trading software the user picks, with the model
  summarising and proposing. Two rules:
  - Placing an order is an operator grant, never the model's own decision.
  - Paper trading comes first.

## 1b. A real clock in the taskbar

Replace the tray's `up 46.2s` with the date and time, including seconds and
milliseconds, for example `2026-10-26 15:42:07.318`.
- Read the RTC once at boot for the wall time, then advance it from the TSC,
  which is the same source CLOCK_MONOTONIC uses.
- Timer ticks are not usable here: the uptime counter is built on them, and
  they stall when the boot processor is busy.
- The display has to follow the time zone. Store it in `/etc/desk`, the same
  file as the other desktop settings.

## 2. A package manager: OrangeBox, command `orange`

A wrapper over an existing distribution's packages. Which one is still to
decide:

| Source | For | Against |
|---|---|---|
| Arch and the AUR | glibc, which is what the guests already run; packages are current | the AUR builds from source, so it needs a toolchain in the guest |
| Alpine APK | musl, small, binary packages | another libc |

Both spellings are first-class, following apt and yay:

| apt-style | pacman/yay-style | What it does |
|---|---|---|
| `orange install <pkg>` | `orange -S <pkg>` | install |
| `orange remove <pkg>` | `orange -R <pkg>` | remove |
| `orange update` | `orange -Sy` | refresh the package index |
| `orange upgrade` | `orange -Syu` | refresh, then upgrade everything |
| (none) | `orange -Syyu` | force-refresh, then upgrade |
| `orange search <q>` | `orange -Ss <q>` | search |

Open questions:
- Where installed packages live. The `/mnt` FAT mounts are read-only, so this
  needs a writable store for big files.
- How a package's `.desktop` file becomes an `apps/*.app` icon.
- Signature checking, using the distribution's own keys.

## 3. A live installer ISO, as the default image

- The user boots straight into a usable live OS. They can set up the network and
  run recovery tools.
- Then they choose to go through a permanent install, which partitions the disk,
  writes the ESP and the store, and copies the system across.
- This needs the package manager first, so the installed system can update
  itself.
- It also needs the "a live ISO cannot update itself" limit (see CLAUDE.md,
  Staged updates) answered by the installed copy.
