<#
.SYNOPSIS
    Build glados and boot it in QEMU under OVMF.

.DESCRIPTION
    This is where development happens. QEMU boots in about a second, gives us a
    real serial console at COM1 (which the GF63 does not have), and can stop on
    a triple fault instead of silently rebooting. Bare metal is for milestones.

.EXAMPLE
    .\scripts\run.ps1
    .\scripts\run.ps1 -Release
    .\scripts\run.ps1 -Gdb          # then: gdb -ex 'target remote :1234'
    .\scripts\run.ps1 -TraceFaults  # log every exception; finds triple faults
    .\scripts\run.ps1 -Smp 1       # one core; `diag mt` cannot pass there
#>
param(
    [switch]$Release,
    [switch]$Gdb,
    [switch]$TraceFaults,
    [string]$Qemu,
    [string]$Ovmf,
    # Guest RAM. The weights are read into a LoaderData pool whole, before
    # ExitBootServices, so the guest needs room for the model plus the firmware
    # plus the heap. 512M was ample for SmolLM2's 135 MB and cannot load
    # Qwen3-0.6B's 570 MB at all -- the read fails and the system boots into a
    # shell with no model, which reads as a loader bug rather than as memory.
    [string]$Memory = '2048M',
    # Guest cores. QEMU gives one unless told otherwise, and with one there are
    # no application processors for `smp::init` to start -- so `diag mt`'s two
    # multi-core claims are false and the suite fails on every clean boot. A
    # check that always fails is read as one nobody has to look at.
    #
    # Four rather than two, because two leaves a single contender for the chunk
    # cursor and the bug `smp.rs` records there needs several. Measured under
    # WHPX, best of nine: the decode path is flat across one, two and four, and
    # `logits` is bit-identical, which is what `smp.rs` claims for a split
    # matvec.
    [int]$Smp = 4,
    # The accelerator, and it defaults to the fast one.
    #
    # This script ran under TCG for its whole life, silently, because nothing
    # here ever passed `-accel`. WHPX -- the Windows Hypervisor Platform, which
    # is what KVM is on Linux -- measures roughly **160x** faster on this
    # workload: a forward-pass group is 286,370 ms under TCG and 1,795 under
    # WHPX. Everything this project called "too slow to run here" was an
    # untested assumption about the emulator, and the same assumption was built
    # into the one script a person actually watches.
    #
    # `-Accel tcg` falls back where the hypervisor is unavailable, and
    # `-Accel 'tcg,thread=multi'` is MTTCG -- which is worth knowing about and
    # not worth using here: it parallelises the *translation* across vCPUs, and
    # this kernel runs everything on the bootstrap processor, so there is no
    # second thread of guest work for it to spread. It is still TCG.
    [string]$Accel = 'whpx',
    # `-cpu max` alongside it, because WHPX alone reports `avx2=0 fma=0` and
    # the trainer's hardware gate then declines to run at all.
    [string]$Cpu = 'max',
    # A WAD to boot with, staged as `GLADOS\doom.wad`.
    #
    # `drive.py --wad` has done this for headless runs since DOOM existed here;
    # this script could not, so the one path a person uses to *look* at it was
    # the one that could not load a level. FreeDoom is the freely licensed one:
    # `out\freedoom\freedoom-0.13.0\freedoom1.wad` after the recipe in
    # CLAUDE.md.
    [string]$Wad,
    # The checkpoint to stage. SmolLM2 by default, which is what QEMU can
    # actually run: the volume below is capped at 516 MB and Qwen3-0.6B is 570.
    # `drive.py` has defaulted to the same file for the same reason.
    [string]$Model,
    # Its tokenizer, which has to match the checkpoint -- Qwen3.5's vocabulary
    # is 248k against Qwen3's 152k, so the wrong one produces text that still
    # looks like text.
    [string]$Tokenizer,
    # Anything else to hand QEMU, for the things this script should not have an
    # opinion about: `-QemuExtra '-full-screen'`, or
    # `-QemuExtra '-display','gtk,show-menubar=off'` for a window with no
    # chrome in it, which is what you want if you are recording.
    [string[]]$QemuExtra = @(),
    # Keep the NVRAM from the last run instead of resetting it. Only the
    # staged-update tests want this; see where it is used.
    [switch]$KeepNvram
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

# --- toolchain ---
$cargoBin = Join-Path $env:USERPROFILE 'scoop\persist\rustup-msvc\.cargo\bin'
if (Test-Path $cargoBin) { $env:PATH = "$cargoBin;$env:PATH" }

# --- locate qemu ---
if (-not $Qemu) {
    $cmd = Get-Command 'qemu-system-x86_64' -ErrorAction SilentlyContinue
    if ($cmd) {
        $Qemu = $cmd.Source
    } else {
        $candidates = @(
            (Join-Path $env:USERPROFILE 'scoop\apps\qemu\current\qemu-system-x86_64.exe'),
            'C:\Program Files\qemu\qemu-system-x86_64.exe',
            'C:\Program Files (x86)\qemu\qemu-system-x86_64.exe',
            'C:\tools\qemu\qemu-system-x86_64.exe'
        )
        foreach ($c in $candidates) {
            if (Test-Path $c) { $Qemu = $c; break }
        }
    }
}
if (-not $Qemu) {
    Write-Error "qemu-system-x86_64 not found. Install it, e.g. (elevated): choco install qemu -y"
}

# --- locate UEFI firmware for the guest ---
#
# Modern QEMU ships edk2 as a *split pflash pair*: a read-only CODE image and a
# writable VARS image (the NVRAM). On this install they are 3,653,632 and
# 540,672 bytes, i.e. exactly 4 MiB together. Those are flash device images, not
# a ROM, so `-bios` rejects them ("could not load PC BIOS"). They have to be
# attached as two if=pflash drives instead.
$share = Join-Path (Split-Path -Parent $Qemu) 'share'

if (-not $Ovmf) {
    foreach ($n in @('edk2-x86_64-code.fd', 'OVMF_CODE.fd')) {
        $p = Join-Path $share $n
        if (Test-Path $p) { $Ovmf = $p; break }
    }
}

$varsTemplate = $null
foreach ($n in @('edk2-i386-vars.fd', 'OVMF_VARS.fd')) {
    $p = Join-Path $share $n
    if (Test-Path $p) { $varsTemplate = $p; break }
}

# Legacy single-file OVMF.fd (code and vars combined) still works with -bios.
$combined = $null
if (-not $Ovmf) {
    $p = Join-Path $share 'OVMF.fd'
    if (Test-Path $p) { $combined = $p }
}

if (-not $Ovmf -and -not $combined) {
    Write-Error "No UEFI firmware found in $share. Pass -Ovmf <path to edk2-x86_64-code.fd>."
}

# Scratch state for the guest: NVRAM, staged firmware, the NVMe image. Created
# unconditionally -- it used to live inside the `if ($Ovmf)` branch below, which
# left it undefined on the legacy combined-OVMF path.
$qemuDir = Join-Path $root '.qemu'
New-Item -ItemType Directory -Force -Path $qemuDir | Out-Null

# Private, writable NVRAM. Never write the template in Program Files, and never
# share one vars file between runs you want to be reproducible.
$varsFile = $null
if ($Ovmf) {
    if (-not $varsTemplate) {
        Write-Error "Found $Ovmf but no matching vars image (edk2-i386-vars.fd) in $share."
    }
    $varsFile = Join-Path $qemuDir 'vars.fd'
    # Refreshed from the template every run unless asked otherwise.
    #
    # It used to be written once and kept forever, which is right for a machine
    # whose boot entries you are testing and wrong for every other run: a stale
    # entry naming a volume that has moved sends the firmware to the **UEFI
    # shell**, and what that looks like is a kernel that failed to boot. Moving
    # the staged ESP to its own directory triggered exactly that, and the
    # symptom was a `Shell>` prompt with no clue as to why.
    #
    # `drive.py` has reset NVRAM on every run since it was written, and says so
    # for the same reason. `-KeepNvram` is for the update tests, which are the
    # only thing here that wants a boot entry to survive.
    if ($KeepNvram -and (Test-Path $varsFile)) {
        Write-Host "nvram: kept"
    } else {
        Copy-Item $varsTemplate $varsFile -Force
    }

    # QEMU's default install path is "C:\Program Files\qemu", and a space inside
    # a -drive "file=..." value gets split during argument parsing -- QEMU then
    # reports it cannot open 'C:\Program'. Stage the firmware into the project,
    # where the path has no spaces.
    $codeFile = Join-Path $qemuDir 'code.fd'
    if (-not (Test-Path $codeFile)) {
        Copy-Item $Ovmf $codeFile
    }
    $Ovmf = $codeFile
}

# --- build ---
Push-Location $root
try {
    # See deploy.ps1: cargo's progress output goes to stderr, which under
    # 'Stop' becomes a terminating error on any non-cached build.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    if ($Release) {
        cargo build --release
        $profileDir = 'release'
    } else {
        cargo build
        $profileDir = 'debug'
    }
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($code -ne 0) { Write-Error "cargo build failed (exit $code)" }
} finally {
    Pop-Location
}

$efi = Join-Path $root "target\x86_64-unknown-uefi\$profileDir\glados.efi"
if (-not (Test-Path $efi)) { Write-Error "missing build artifact: $efi" }

# --- stage an ESP tree ---
#
# **A tree of its own, and not the repo's `esp/`.** This booted `esp/` directly
# for its whole life, which is the same directory `deploy.ps1` fills for the
# real machine -- and that one holds the *real* checkpoint, 598 MB of
# Qwen3-0.6B. QEMU projects this directory as a synthetic FAT volume with a
# fixed 516 MB geometry, so the moment anybody staged a model for the GF63 this
# script stopped booting at all, with `Directory does not fit in FAT32` and
# nothing to say which file was the problem.
#
# `drive.py` never had the fault because it has always staged its own minimal
# tree under `.qemu/`. This does the same now, and the two agree on the
# default checkpoint for the same reason: SmolLM2 is what fits.
#
# BOOTX64.EFI on the removable-media path is what firmware boots with no NVRAM
# entry configured. Same layout works on the real USB SSD.
$esp = Join-Path $root '.qemu\esp-run'
$bootDir = Join-Path $esp 'EFI\BOOT'
$gladosDir = Join-Path $esp 'GLADOS'
New-Item -ItemType Directory -Force -Path $bootDir | Out-Null
New-Item -ItemType Directory -Force -Path $gladosDir | Out-Null
Copy-Item $efi (Join-Path $bootDir 'BOOTX64.EFI') -Force

# Copied only when it would change anything. A checkpoint is 129 MB and this
# runs on every launch, so comparing first turns most launches from a copy into
# two `stat`s.
function Stage-File($src, $dest, $label) {
    if (-not $src) { return }
    if (-not (Test-Path $src)) {
        Write-Host "$label : $src not found, skipping"
        return
    }
    $s = Get-Item $src
    $d = if (Test-Path $dest) { Get-Item $dest } else { $null }
    if (-not $d -or $d.Length -ne $s.Length -or $d.LastWriteTimeUtc -lt $s.LastWriteTimeUtc) {
        Copy-Item $src $dest -Force
    }
    $mb = [math]::Round($s.Length / 1MB, 1)
    Write-Host "$label : $src ($mb MB)"
}

if (-not $Model)     { $Model     = Join-Path $root 'out\smollm2-135m.bin' }
if (-not $Tokenizer) { $Tokenizer = Join-Path $root 'out\smollm2-tokenizer.bin' }
Stage-File $Model     (Join-Path $gladosDir 'model.bin')     'model'
Stage-File $Tokenizer (Join-Path $gladosDir 'tokenizer.bin') 'tokn '
Stage-File (Join-Path $root 'esp\GLADOS\roots.der') (Join-Path $gladosDir 'roots.der') 'roots'
if ($Wad) {
    if (-not (Test-Path $Wad)) { Write-Error "no such WAD: $Wad" }
    Stage-File $Wad (Join-Path $gladosDir 'doom.wad') 'wad  '
}

# The cap, checked here rather than discovered by QEMU. Its message names the
# capacity and not the offender, which is a long way from telling you that the
# 598 MB checkpoint you staged for the laptop is the reason DOOM will not boot.
$vvfatMb = 516
$stagedMb = [math]::Round(((Get-ChildItem $esp -Recurse -File | Measure-Object Length -Sum).Sum) / 1MB, 1)
if ($stagedMb -ge $vvfatMb) {
    Get-ChildItem $esp -Recurse -File | Sort-Object Length -Descending |
        Select-Object -First 3 |
        ForEach-Object { Write-Host ("  {0,7:N1} MB  {1}" -f ($_.Length / 1MB), $_.Name) }
    Write-Error ("staged $stagedMb MB, and QEMU projects this directory as a " +
        "FAT volume of $vvfatMb MB. The largest files are above; pass a smaller " +
        "-Model, or use tools/drive.py --stage-iso, which has no such cap.")
}
Write-Host "esp  : $esp ($stagedMb MB of $vvfatMb)"

# --- launch ---
$qemuArgs = @(
    '-machine', 'q35',
    '-accel', $Accel,
    '-cpu', $Cpu,
    '-m', $Memory,
    '-smp', $Smp
)

if ($combined) {
    $qemuArgs += @('-bios', $combined)
} else {
    # unit=0 is the firmware code, unit=1 is NVRAM. Order matters.
    $qemuArgs += @(
        '-drive', "if=pflash,format=raw,unit=0,readonly=on,file=$Ovmf",
        '-drive', "if=pflash,format=raw,unit=1,file=$varsFile"
    )
}

# A scratch NVMe namespace, so the store half of the system is exercisable here
# rather than only on the laptop. The image has no partition table, which is the
# case find_store_region falls back to -- unclaimed space on an unpartitioned
# disk. The GF63 takes the other path, a partition tagged with our type GUID, so
# neither branch gets to go untested.
$nvmeImg = Join-Path $qemuDir 'nvme.img'
if (-not (Test-Path $nvmeImg)) {
    $fs = [System.IO.File]::Create($nvmeImg)
    $fs.SetLength(64MB)   # sparse: costs no disk until written
    $fs.Close()
    Write-Host "created $nvmeImg (64 MiB)"
}

$qemuArgs += @(
    # Plain VVFAT, which is FAT16, and **not** `fat:32:`.
    #
    # This said `fat:32:` on the reasoning that FAT16 tops out at 516 MB and
    # Qwen3-0.6B is 570. The reasoning is sound and the conclusion does not
    # work: QEMU says its FAT32 is untested, and what it produces is a volume
    # **the firmware cannot read** -- the guest finds no bootloader and lands
    # in the UEFI shell, which looks exactly like a kernel that failed to boot.
    # So the 516 MB ceiling was never raised, only moved somewhere it could not
    # be seen, and a model larger than it is not testable here at all.
    #
    # `drive.py` has used plain `fat:rw:` all along and says why in the same
    # words. Two files disagreeing about one flag, with the working one
    # documenting the failure, is how this went unnoticed: nothing ever ran
    # `run.ps1` and `drive.py` against the same question.
    '-drive', "format=raw,file=fat:rw:$esp",
    '-drive', "file=$nvmeImg,if=none,id=nvm0,format=raw",
    '-device', 'nvme,serial=GLADOSQEMU0001,drive=nvm0',
    '-serial', 'stdio',
    '-net', 'none',
    # Stop on triple fault instead of rebooting forever. Without this, an early
    # paging bug looks like an infinite boot loop with nothing to read.
    '-no-reboot'
)
if ($QemuExtra)   { $qemuArgs += $QemuExtra }
if ($TraceFaults) { $qemuArgs += @('-d', 'int,cpu_reset') }
if ($Gdb)         { $qemuArgs += @('-s', '-S') }

Write-Host "qemu : $Qemu"
if ($combined) {
    Write-Host "bios : $combined"
} else {
    Write-Host "code : $Ovmf"
    Write-Host "vars : $varsFile"
}
Write-Host "esp  : $esp"
if ($Gdb) { Write-Host "paused for gdb on :1234 -- connect, then 'continue'" -ForegroundColor Yellow }
Write-Host ''

& $Qemu @qemuArgs
