# GLaDOS 1.4.1

GLaDOS is an operating system with a language model living inside its kernel.
It is written from nothing in Rust, it runs on one real laptop, and there is no
Linux underneath it and no wall between the model and the machine. A thought the
model has is a function call. It boots, it decides, it rewrites its own source
at night, and it mines its own keep.

This release replaces how the machine draws. The desktop was laggy, sometimes
broken, and looked like 1998. Every pixel of it is new: Windows Vista's glass,
arranged the way Hyprland arranges a modern Linux desktop, and drawn by a
renderer that only touches what changed.

## It is smooth now

- **Only what changed is drawn.** The screen was repainted whole for every
  keystroke and every tick of the clock. Now each frame works out what moved
  and redraws that, so an idle desktop costs almost nothing and a moving window
  costs about what the window does.
- **Windows are kept as pictures.** Each window is drawn once and reused, so
  dragging one over another, opening a menu, or hovering a button no longer
  redraws the windows underneath. Measured: a whole frame in half the time,
  a small change in a third.
- **Motion runs on the clock.** Windows grow in and fade up when they open,
  shrink and fade when they close, and slide when the layout changes, all
  timed in milliseconds rather than in frames, so a busy moment shows the
  motion further along instead of making it drag.
- **It gives up decoration before it gives up smoothness.** When frames run
  slow it sheds effects in a fixed order, blur first and motion last, and
  takes them back when there is room again.
- **The keyboard works while the machine is busy.** Window keys are handled by
  the part that draws, not the shell, so switching windows works in the middle
  of a long command.

## It runs at full speed on the laptop, and the mouse works

- **Full clock.** Nothing at boot asked the processor for a speed, and on the
  laptop it sat at under a quarter of its base clock. It runs at full speed
  now, and `perf` tells you how fast and whether the screen or the processor
  is what limits a frame.
- **USB mice and keyboards work.** Including gaming mice that present a
  keyboard beside the mouse, plugged in before boot or after, and found again
  if they are pulled out and put back. A failing device is reset rather than
  left dead.
- **Pointer speed with acceleration.** Slow movement is exact, a flick
  crosses the screen, and the speed is a setting.
- **A device manager.** `devices` lists everything the machine has, grouped
  by kind, inside or plugged in, with whether it works and why not.
- **The touchpad keeps going.** A missed signal from the touchpad used to
  stop it until reboot; it is collected now.

Scrolling is the next update.

## It looks like now

- **Glass.** Window frames, title bars, the taskbar and the Start menu are
  panes of glass over a blurred copy of the wallpaper, Vista's own formula.
- **Real text.** Noto Sans for the interface and Cascadia Code for the
  terminal, smooth at every size, where everything used to be an 8 by 8
  bitmap doubled.
- **Soft edges and soft shadows.** Every corner, line and circle is smooth,
  shadows are soft and fall the way light does, and gradients no longer band.
- **A new wallpaper.** A new day: the Aperture sun, striped and whole, rising
  behind rolling hills, and the boot logo glides into it as the desktop
  appears.
- **New icons, pointer and Start button.** Icons are drawn as lit objects, the
  pointer is smooth with a soft shadow, and the Start button is a glass orb
  with the Aperture mark in it.
- **The window you are using glows.** The focused window wears an orange
  border with a soft glow, and the others dim a little.
- **Sized for the screen.** The interface is drawn at 125%, the size this
  laptop's display calls for, and can be made larger or smaller.

## It arranges itself

- **Windows tile.** New windows split the space with the one you opened them
  from, with gaps, the way Hyprland does it. Any window can float instead:
  take hold of it, or press Super and V.
- **The Windows key works.** Super with Enter opens a terminal, Q closes, V
  floats, F goes full screen, the arrows move between windows and with Shift
  swap them, and D shows the desktop.
- **Flip 3D.** Super and Tab stands every window up in a stack, leaning away
  from you. Tab turns it, Enter picks the window at the front.
- **Previews on the taskbar.** Hover a window's button and a live picture of
  the window appears above it.
- **Settings has a Desktop page.** Size, motion, glass colour, border colour,
  dimming, gaps, corners, blur and tiling, each a click, and the machine
  remembers what you chose.

## Under the hood

- **Write-combined display memory.** The screen is written the way graphics
  drivers write it, in whole lines, rather than through the processor's cache.
- **One owner for the screen.** A full-screen program leases the display and
  hands it back cleanly, so nothing is left drawn over the desktop when it
  exits.
- **Outside code, for fonts only.** The font rasteriser is the first crate this
  kernel did not write. It is vendored, licence-checked in CI, and credited in
  NOTICE.md, with both typefaces under the SIL Open Font License.

## Download

Three images, same as before: Qwen3-0.6B, and Qwen3.5-2B with 8k or 32k of
context. Check them against SHA256SUMS. Setup for every route is on the download
page at https://glados.aperture.institute/download/
