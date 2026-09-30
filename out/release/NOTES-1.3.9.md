# GLaDOS 1.3.9

The biggest release yet. GLaDOS can now mine $GLaDOS, straight from its own
kernel, and it has its own pool that pays miners automatically.

## Mine $GLaDOS

- **A dedicated miner image.** 33 MB, boots straight into mining, and ships as
  its own release: [GLaDOS miner 1.3.9](https://github.com/IlumCI/GLaDOS/releases/tag/mining-v1.3.9).
  Give it your wallet once and it remembers.
- **Automatic payouts.** The pool turns everything it earns into $GLaDOS and
  sends an equal share to every eligible miner. Every payout is listed on
  [the pool page](https://glados.aperture.institute/pool/).
- **A mining screen worth looking at.** Full screen, live hashrate, shares,
  uptime, and the wallet you're paid to.
- **Set it up before you boot.** Put your wallet in `GLADOS/MINER.TXT` on the
  USB stick and the PC mines with nothing to type.
- **Much faster hashing.** The kernel is built at full optimisation, the
  miner spreads across every core, and its inner loop runs 1.5x faster.

## Networking

- **Secure connections to more of the internet.** Certificate chains that go
  through a cross-signed root now validate, and every image carries its own
  root certificate store.
- **DNS stays reliable** with several tasks online at once.
- **Better randomness on machines with no keyboard or mouse,** using the
  processor's hardware random source.

## Also in this release

- **Boots faster on screen.** The console prints at full speed.
- **More stable on multi-core machines.** A bug that overwrote the extra
  cores' idle tasks at every boot is gone.
- **Cleaner licensing.** The one GPL-2.0 driver in the tree is removed.
- **The machine writes code for itself.** Its self-improvement loop runs every
  night on its own, and writes, tests and proposes small changes to itself.

## Download

Three images, same as before: Qwen3-0.6B, and Qwen3.5-2B with 8k or 32k of
context. Check them against `SHA256SUMS`. Setup for every route is on
[the download page](https://glados.aperture.institute/download/).
