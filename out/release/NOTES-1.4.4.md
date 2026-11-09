# GLaDOS 1.4.4

GLaDOS is an operating system with a language model living inside its kernel.
Rust, from nothing, on real hardware. No Linux underneath it and no wall
between the model and the machine.

This release gives it a Linux userland it can install software into, and puts
the model to work in an office suite.

## A real userland

- **An Arch Linux root filesystem.** Linux programs run on a real root: bash,
  coreutils, git, Python, fish, GTK and Qt, all unmodified. GLaDOS finds it on
  the disk at boot by itself.
- **It identifies as GLaDOS.** `orange install fastfetch`, run it, and it shows
  the GLaDOS mark and name.
- **Graphical programs.** GTK 3, GTK 4 and Qt applications open in their own
  windows on the GLaDOS desktop: text editors, terminals, LibreOffice.
- **Faster underneath.** Downloads are 19 times faster, and starting a program
  costs about a quarter of what it did.

## Install anything

- **`orange`, the package manager.** apt-style commands over pacman:
  `orange install`, `remove`, `search`, `upgrade`. Installs come straight from
  Arch's repositories over the network.
- **The AUR, like yay.** Packages no repository has are built on the machine
  itself, with signatures checked, and installed as ordinary packages.
- **Debian packages.** `orange install ./something.deb` converts a `.deb` into
  a native package, works out what it needs from the program itself, and
  installs it.
- **Our own signed repository.** GLaDOS's own packages come from the
  `[glados]` repository, signed with its own key.

## The model, at the office

- **It types into LibreOffice.** Ask for a document and the model in the
  kernel writes it into Writer, a keystroke at a time.
- **It builds spreadsheets** with live formulas, and answers questions about a
  sheet by working them out in Calc on screen.
- **It writes the report.** Every answer so far, gathered into a Writer
  document with a summary.

## Desktop

- **A new Settings app.** A grid of categories, pages with real switches and
  sliders, search across every setting, and quick settings in the tray.
- **Mood on the wallpaper can be turned off** in Settings.

## Under the hood

Dozens of Linux interfaces were added or corrected so that large programs run:
threads, shared memory, sockets, process credentials, file permissions,
symlinks, `/proc` and more. One of the fixes closes a way any program could
halt the whole machine with a single call.

## Install

Download an ISO, write it to a USB stick, and boot from it.

The Linux userland is a separate download, because it is larger than one
release file may be. To add it, put it on a partition of the computer's NVMe
SSD; GLaDOS mounts it at boot.

1. Download every `glados-1.4.4-root.erofs.part-*` file and `SHA256SUMS`.
2. On Linux, join and check them:

       cat glados-1.4.4-root.erofs.part-* > root.erofs
       sha256sum -c SHA256SUMS

3. Make an empty partition of at least 3 GB on the NVMe SSD, then write the
   image to it. **Writing to the wrong partition destroys what is on it**,
   so check the name first with `lsblk`:

       sudo dd if=root.erofs of=/dev/nvme0n1pN bs=4M status=progress

An installer that does all of this for you is the next release.
