# GLaDOS 1.4.5

GLaDOS is an operating system with a language model living inside its kernel.
Rust, from nothing, on real hardware. No Linux underneath it and no wall
between the model and the machine.

This release installs it. GLaDOS goes onto your disk, alone or beside
Windows, keeps what you install, and updates itself without ever leaving you
with a machine that does not start.

## Install it

- **An installer.** Start > Install, or `install` in the terminal. It shows
  the disk, exactly what it will do to it, and asks you to type a short code
  back before it writes anything.
- **Erase, or install alongside.** Alongside uses free space and moves
  nothing: every existing partition, Windows' included, is kept byte for
  byte. Make room from Windows' Disk Management first; GLaDOS never shrinks a
  partition.
- **A startup menu.** With another system on the disk, GLaDOS offers it for
  a few seconds at start. Picking Windows restarts straight into it through
  the firmware, the way Windows expects to be started. `boot` does the same
  from the terminal.
- **Everything persists.** Programs you install with `orange`, settings,
  permissions, and the files in your home survive a restart.

## Two images

- **Base** carries the model, a Linux system, the voice and the fly's brain.
  Install it and it works offline.
- **Netinstall** is under 300 MB: the kernel, the installer and a system that
  can run pacman. Choose what you want -- desktop programs, LibreOffice,
  media, the model -- and the installed machine fetches it on its first
  start.

## Updates that cannot leave you stranded

- **Whole system images.** A new system is written beside the running one,
  checked, and tried at the next start. If it does not reach the desktop, the
  previous one comes back by itself -- with the kernel it ran with.
- **Your programs follow.** When a new system ships a program you installed
  yourself, the newer copy wins and nothing you upgraded is downgraded.
- **Pinned packages.** Each system image is built from one dated snapshot of
  Arch Linux, and what you install on top comes from the same day.

## Before installing beside Windows

GLaDOS boots with Secure Boot off. If Windows uses BitLocker, have its
recovery key ready (aka.ms/myrecoverykey): Windows asks for it once after
Secure Boot is changed.
