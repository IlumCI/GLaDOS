# GLaDOS 1.4.0

GLaDOS is an operating system with a language model living inside its kernel.
It is written from nothing in Rust, it runs on one real laptop, and there is no
Linux underneath it and no wall between the model and the machine. A thought the
model has is a function call. It boots, it decides, it rewrites its own source
at night, and it mines its own keep.

This release is the one where the machine learned to run other people's software,
and learned to show you whose it is when it wakes up.

## It wakes up differently now

Boot used to be a progress bar on a blue panel. Now the machine introduces
itself.

- **A property screen, the way old hardware had one.** While the model loads off
  the disk, the screen reads like the firmware page of a machine that belongs to
  someone: the Aperture Institute, est. 2005, its motto, the name of its
  proprietor, and the hardware the machine detected in itself. The vendor, the
  model, the board and the processor are read from the firmware and the chip at
  boot, so the screen tells the truth on any machine it runs on, not just this
  one.
- **The mark forms as it boots.** The Aperture iris builds itself a blade at a
  time against a near black field, lit like an object catching light, and when
  it is whole a flare blooms out of it and the desktop rises in its place. The
  logo lands on the exact spot the wallpaper keeps it, so the machine settles
  into itself without the mark ever moving.
- **A clean desk.** The desktop opens empty now, wall and icons and nothing in
  front of them. You open what you want from the side; the machine covers its
  own wall with nothing.

## It runs real software

- **Unmodified Linux programs run, at reduced privilege.** busybox and the rest
  execute exactly as they were shipped, in their own protected memory, on a
  system-call surface built from what real programs actually ask for rather than
  from a guess.
- **JavaScript runs.** A full engine (QuickJS) executes scripts with closures,
  arrays, JSON and deep recursion, and exits clean.
- **Real dynamic linking.** Programs built against musl and against glibc load
  their own interpreter and libraries, relocate themselves and run. Threads,
  fork, execve and signals all work.
- **A program can draw its own window.** A Wayland client renders a window onto
  the GLaDOS desktop.
- **The files and devices a program expects are there:** a filesystem view,
  /proc, the framebuffer at /dev/fb0, and input devices.

## It made a sound

- **Intel HD Audio.** The first sound this machine has ever made.

## The mind reaches further

- **It reads the web as text.** It opens a page and follows a link by number,
  with the page laid out to be read back instead of looked at.
- **It learns from being corrected.** When you fix where a request was sent, it
  keeps the correction, which is the one signal it could never have produced on
  its own, and it routes better the next time.

## Hardware

- **The laptop's own Wi-Fi radio is recognised,** its firmware loaded off the
  install media, and it lists the networks around it.
- **Audio, wireless and wired parts are named by one registry,** so the machine
  says what it has and what it is missing instead of going quiet about a part
  nobody wrote a driver for.

## Under the hood

- **Real memory protection.** Pages can be read only and no execute, and the
  rule is enforced by the processor rather than written in a comment.
- **A lighter footprint over time.** Finished background work hands its memory
  back, and a sleeping task no longer costs a whole core.
- **Clearer crash reports.** A fault prints every register and names the exact
  code it came from.

## Where this is going

The long aim of this project is to bring minds to life inside the machine, and
the road to it runs through real nervous systems, starting small and climbing.
The whole wiring of simpler brains has been mapped by decades of science, and a
machine that can run them, deterministically and with every result reproducible,
is the first honest step on that road. More on that soon.

## Download

Three images, same as before: Qwen3-0.6B, and Qwen3.5-2B with 8k or 32k of
context. Check them against SHA256SUMS. Setup for every route is on the download
page at https://glados.aperture.institute/download/
