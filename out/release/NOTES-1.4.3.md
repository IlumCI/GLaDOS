# GLaDOS 1.4.3

GLaDOS is an operating system with a language model living inside its kernel.
Rust, from nothing, on real hardware. No Linux underneath it and no wall
between the model and the machine.

This release gives it a graphics card, and the programs to use it.

## A GPU, from ring 3

- **Vulkan on a real GPU.** GLaDOS speaks Linux's virtio-gpu interface at
  `/dev/dri/renderD128`. An unmodified Mesa, running as an ordinary Linux
  program, does Vulkan on the host's GPU through it. A million-element
  compute dispatch comes back correct. Nothing in the kernel knows what
  Vulkan is.
- **OpenGL on top.** Mesa's Zink turns OpenGL into Vulkan, so ordinary
  32-bit GL programs draw on the GPU as well. A GL test program reports
  `zink Vulkan 1.4 (NVIDIA GeForce RTX 3050)`, OpenGL 4.6, at about 50
  frames a second.
- **A 3D game, 20 to 30 times faster.** A 32-bit 3D game that drew about
  1 frame a second on the CPU now renders at 27 to 38 on the GPU.
- **Where it runs today.** This path is a virtual machine on a Linux host,
  under QEMU with KVM, reaching the host's GPU. A native driver for the
  laptop's own GPU on bare metal is the next step and is not in this release.

## Programs from elsewhere

- **32-bit Linux programs.** i386 binaries run beside 64-bit ones, with
  their own glibc and libraries.
- **Every core.** A program's threads spread across all the processor's
  cores.
- **Several at once.** Programs run as background jobs beside the shell
  (`linux bg`, `jobs`, `kill`), and several sessions can run together, each
  with its own process IDs.
- **Sound.** Programs that play through ALSA come out of the HD Audio
  device.
- **Windows of their own.** The built-in Wayland server gained a keyboard
  and mouse, true full screen, and pointer lock for mouse-look. OpenGL works
  in a program's window, on the CPU or on the GPU.
- **A ring-3 network manager**, a program in its own right rather than part
  of the kernel.

## Click to run

- **Installed programs are desktop icons.** A program brings a small entry
  file on its own disk, describing how to start it. GLaDOS finds it at boot
  and puts the program on the desktop and in the Program Manager, with its
  own icon. Double-click and it runs. No typing.
- **Ready for other systems.** Each entry names the runtime it needs. Today
  that is Linux; Windows and BSD runtimes are meant to plug into the same
  launcher later.
- **Once at a time.** Launching a program that is already running is
  refused instead of starting a second copy.

## Fixes

- A program creating more than 4,096 GPU buffers crashed; buffer handles are
  reused now.
- A futex wake could be lost once a program had used more than 32 lock
  addresses; the woken thread now always hears about it.
- GPU submissions no longer hold every other thread of a program while
  waiting on the host, and the compositor is woken at once when a program
  shows a new frame. Together these took the 3D game from 19.6 to 27 to 38
  frames a second.

## Known issues

- **A first-frame freeze.** In about half of GPU runs, a 3D program stops
  on its first frame. Its threads are waiting inside the frame presentation
  path. Restarting the program usually gets past it. Fixing it is the first
  thing in the next release.
- **Frame rates were measured standing still.** Rates while moving through a
  3D scene have not been measured yet.

## Install

Download an ISO, write it to a USB stick, and boot from it. Everything is
inside the image: there is nothing else to download. Every image carries its
Wi-Fi firmware. The Qwen3-0.6B image also carries the neural voice and the
adult fly's brain; the 2B images are at GitHub's 2 GB size limit and use the
built-in voice and worm instead.
