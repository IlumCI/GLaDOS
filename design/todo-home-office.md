# To do: a home and office OS

Asked for on 2026-10-25. The order below is the order to build in. Each part is
a rung that stands on the one before it.

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
