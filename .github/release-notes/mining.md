Turn a PC into a $GLaDOS miner. Boot the ISO, give it your wallet, and it
mines. What the pool earns is converted to $GLaDOS and paid out to every
eligible miner automatically -- no claiming, no gas on your side.

**Download:** `glados-{{VERSION}}-miner.iso` (33 MB), plus `SHA256SUMS` to
check it:

    sha256sum -c --ignore-missing SHA256SUMS

Two ways to run it: in a virtual machine with QEMU, or on a real PC from a
USB stick. Both end at the same screen, mining.

## In a virtual machine (QEMU)

    qemu-system-x86_64 -machine q35 -accel kvm -m 2G \
        -bios /usr/share/ovmf/OVMF.fd \
        -cdrom glados-{{VERSION}}-miner.iso

- **`-machine q35`** gives the VM a PCI Express bus, which is where the miner
  looks for its network card. QEMU's older default machine has none.
- **OVMF** is the UEFI firmware the miner boots on. `apt install ovmf` on
  Debian/Ubuntu (path above); `pacman -S edk2-ovmf` on Arch, path
  `/usr/share/edk2/x64/OVMF.4m.fd`; on Fedora
  `/usr/share/edk2/ovmf/OVMF_CODE.fd`.
- **`-accel kvm`** runs it at full speed on Linux. Leave it out on a machine
  without KVM and it still runs, more slowly.

It asks for your 0x wallet address once. Type it, press Enter, and it mines.

**Why QEMU and not VirtualBox.** The miner is a UEFI program that finds its
network card over PCI Express. A new VirtualBox machine boots legacy BIOS on a
chipset without PCI Express configuration, so out of the box the ISO either
does not boot or boots without a network. QEMU with the two flags above is the
setup this release is built and tested on.

## On a PC (bare metal)

1. **Flash the ISO to a USB stick** with balenaEtcher, Rufus (DD mode) or
   `dd`. The stick then appears on your computer as a drive named **GLADOS**.
2. **Put your wallet on the stick** so the PC never needs a keyboard: open
   `MINER.TXT` on the GLADOS drive, replace `ask` on the `worker` line with
   your 0x address, and save.
3. **Boot from the stick** with your PC's boot menu key -- usually F11, F12
   or Esc, depending on the maker.

The PC needs:

- **x86-64 with UEFI**, and **Secure Boot off**. Roughly anything from 2011
  on.
- **Wired Ethernet** (Intel e1000-family or Realtek RTL8168/8111, which
  covers most desktops and laptops), **or a phone sharing its connection over
  USB** (USB tethering).

## Your wallet

- **It must hold at least 1,000,000 $GLaDOS** to mine on the pool, and still
  hold it when a payout is made. The screen tells you if it falls short.
- **It is remembered.** An address typed at boot is saved in that PC's
  firmware, so every later boot goes straight to mining.
- **To switch wallets,** type a new 0x address on the mining screen and press
  Enter.
- **To forget the saved address,** type `mine forget`. The next boot asks
  again.

The address is public information. The miner never asks for a key, and
nothing it stores could spend anything.

## How you get paid

Your PC's work earns the pool money. The pool converts everything it earns
into $GLaDOS and sends it out -- automatically, with nobody at a keyboard.

- **Equal split.** Every eligible wallet in a payout gets the same amount.
- **Eligible** means: your wallet holds the 1,000,000 $GLaDOS minimum, it is a
  normal wallet (not a contract), and in that round it did at least a quarter
  of the work of the typical miner. Work that misses a round carries to the
  next one; nothing is lost.
- **When.** A payout goes out as soon as the pot is big enough that fees are
  under 10% of it. The more miners, the more often.
- **Sent to you.** $GLaDOS arrives in your wallet on Robinhood Chain. There is
  nothing to claim.
- **Public.** Every payout is listed with its transaction at
  <https://pool.aperture.institute/payouts.json>.

## Also in this release

`glados-pool` and `glados-miner`: static Linux builds of the pool and a
host-side miner, for people who would rather not boot an ISO.
