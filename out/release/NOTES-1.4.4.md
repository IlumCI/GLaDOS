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

## Install

Download the ISO, write it to a USB stick, and boot from it.

The Qwen3-0.6B image carries a slim Linux system with `orange`. For the full
desktop -- LibreOffice, GTK and Qt applications, media players -- connect to
the network and run:

    orange upgrade

Until the installer arrives in the next release, GLaDOS runs from the USB
stick and what you install lasts until you reboot.
