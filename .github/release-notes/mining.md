Turn a PC into a $GLaDOS miner. Boot the ISO, give it your wallet, and it
mines. Everything it earns is paid out in $GLaDOS.

**Download:** `glados-{{VERSION}}-miner.iso`, plus `SHA256SUMS` to check it:

    sha256sum -c --ignore-missing SHA256SUMS

## Run it in a VM (QEMU)

    qemu-system-x86_64 -machine q35 -accel kvm -m 2G \
        -bios /usr/share/ovmf/OVMF.fd \
        -cdrom glados-{{VERSION}}-miner.iso

- **`-machine q35` is required.** QEMU's default machine has no PCIe, and
  without it the miner finds no network card.
- **OVMF is the UEFI firmware.** Install it with `apt install ovmf` on
  Debian/Ubuntu, or `pacman -S edk2-ovmf` on Arch, where the path is
  `/usr/share/edk2/x64/OVMF.4m.fd`. On Fedora use
  `/usr/share/edk2/ovmf/OVMF_CODE.fd`.
- **Without KVM:** drop `-accel kvm`. It runs, but slowly.

At first boot it asks for your 0x wallet address. Type it and press Enter.

## Run it on a PC (bare metal)

1. **Flash the ISO to a USB stick** with balenaEtcher, Rufus (DD mode) or
   `dd`. The stick then shows up on your computer as a drive named
   **GLADOS**.
2. **Optional: skip typing at boot.** Open `GLADOS/MINER.TXT` on that drive,
   replace `ask` on the `worker` line with your 0x wallet address, and save.
3. **Boot from the stick.** Use your PC's boot menu key: F11, F12 or Esc,
   depending on the maker.

What the PC needs:

- **UEFI, with Secure Boot off.** The image is not signed.
- **x86-64.**
- **A wired connection:** Intel (e1000 family) or Realtek RTL8168 Ethernet,
  or a phone sharing its connection over USB (USB tethering). Wi-Fi is not
  supported yet.

## Your wallet

- **It has to hold the pool's minimum of $GLaDOS**, or the pool turns it away
  and the screen says so.
- **It is remembered.** An address typed at boot is saved in that PC's
  firmware, so every later boot starts mining with nothing to type.
- **To switch wallets,** type a new 0x address while it is mining and press
  Enter. The address is saved and the miner restarts onto it.
- **To forget the saved address,** type `mine forget`. The next boot asks
  again.

The address is public information and the miner never asks for a key.
Nothing it stores could spend anything.

## Also in this release

`glados-pool` and `glados-miner`: static Linux builds of the pool and a
host-side miner, for people who would rather not boot an ISO.
