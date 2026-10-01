Your spare PC can mine $GLaDOS tonight. Boot the miner, give it your wallet,
and walk away. The pool turns everything it earns into $GLaDOS and sends it
straight to your wallet. Nothing to claim, nothing to install.

**Download:** `glados-{{VERSION}}-miner.iso` (33 MB), plus `SHA256SUMS` to
check it:

    sha256sum -c --ignore-missing SHA256SUMS

Run it in a virtual machine with QEMU, or boot a real PC from a USB stick.
Both land on the same screen: mining.

## In a virtual machine (QEMU)

1. **Install QEMU and OVMF,** the UEFI firmware the miner boots on:
   `sudo apt install qemu-system-x86 ovmf` (Debian/Ubuntu),
   `sudo pacman -S qemu-full edk2-ovmf` (Arch),
   `sudo dnf install qemu-kvm edk2-ovmf` (Fedora).

2. **Copy the firmware's settings file** next to the ISO. The VM keeps your
   wallet in it between boots:

       cp /usr/share/OVMF/OVMF_VARS_4M.fd vars.fd           # Debian/Ubuntu
       cp /usr/share/edk2/x64/OVMF_VARS.4m.fd vars.fd       # Arch
       cp /usr/share/edk2/ovmf/OVMF_VARS.fd vars.fd         # Fedora

3. **Boot it,** using the firmware file from the same folder
   (`OVMF_CODE_4M.fd`, `OVMF_CODE.4m.fd` or `OVMF_CODE.fd`):

       qemu-system-x86_64 -machine q35 -accel kvm -m 2G \
           -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd \
           -drive if=pflash,format=raw,file=vars.fd \
           -cdrom glados-{{VERSION}}-miner.iso

4. **Type your 0x wallet address and press Enter.** You're mining. Next time,
   the same command goes straight to mining.

`-machine q35` gives the VM the PCI Express bus the miner finds its network
card on. `-accel kvm` runs it at full speed; on a machine without KVM, drop it
and it runs slower.

**Why QEMU?** The miner boots on UEFI and finds its network card over PCI
Express. A new VirtualBox machine boots legacy BIOS on a chipset with no PCI
Express, so it can't see the network. QEMU with the flags above is what this
release is built and tested on.

## On a real PC (USB stick)

1. **Flash the ISO to a USB stick** with balenaEtcher, Rufus (DD mode) or
   `dd`. The stick shows up on your computer as a drive named **GLADOS**.
2. **Put your wallet on the stick:** on the GLADOS drive, open the `GLADOS`
   folder, then `MINER.TXT`. Replace `ask` on the `worker` line with your 0x
   address and save. The PC won't even need a keyboard.
3. **Boot from the stick** with your PC's boot menu key: usually F11, F12 or
   Esc.

What the PC needs:

- **x86-64 with UEFI**, with **Secure Boot off**. Roughly 2011 and newer.
- **Ethernet** (Intel e1000-family or Realtek RTL8168/8111, which covers most
  desktops and laptops), or a phone sharing its connection over USB.

## Your wallet

- **Hold at least 1,000,000 $GLaDOS** to mine on the pool, and keep holding
  it to get paid. The screen tells you if you're short.
- **It's remembered.** An address typed at boot is saved on that PC, so every
  later boot goes straight to mining.
- **Switch wallets** by typing a new 0x address on the mining screen and
  pressing Enter.
- **Forget the saved address** by typing `mine forget`. The next boot asks
  again.

All the miner ever needs is your address. It never asks for a key or a
signature.

## How you get paid

- **Everyone gets the same.** Each payout is split equally between every
  eligible wallet.
- **Who is eligible:** wallets holding at least 1,000,000 $GLaDOS that did at
  least a quarter of the typical miner's work in that round. Miss a round and
  your work carries over to the next.
- **How often:** as soon as the pot is big enough that fees stay under 10% of
  it. More miners means more frequent payouts.
- **Straight to your wallet** on Robinhood Chain. There is nothing to claim.
- **Every payout is public,** with its transaction:
  <https://pool.aperture.institute/payouts.json>, or on
  <https://glados.aperture.institute/pool/>.

## Also in this release

`glados-pool` and `glados-miner`: static Linux builds of the pool and a
host-side miner, if you'd like to mine without booting an ISO.
