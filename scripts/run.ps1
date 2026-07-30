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
#>
param(
    [switch]$Release,
    [switch]$Gdb,
    [switch]$TraceFaults,
    [string]$Qemu,
    [string]$Ovmf
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
    if (-not (Test-Path $varsFile)) {
        Copy-Item $varsTemplate $varsFile
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
# BOOTX64.EFI on the removable-media path is what firmware boots with no NVRAM
# entry configured. Same layout works on the real USB SSD.
$esp = Join-Path $root 'esp'
$bootDir = Join-Path $esp 'EFI\BOOT'
New-Item -ItemType Directory -Force -Path $bootDir | Out-Null
Copy-Item $efi (Join-Path $bootDir 'BOOTX64.EFI') -Force

# --- launch ---
$qemuArgs = @(
    '-machine', 'q35',
    '-m', '512M'
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
    '-drive', "format=raw,file=fat:rw:$esp",
    '-drive', "file=$nvmeImg,if=none,id=nvm0,format=raw",
    '-device', 'nvme,serial=GLADOSQEMU0001,drive=nvm0',
    '-serial', 'stdio',
    '-net', 'none',
    # Stop on triple fault instead of rebooting forever. Without this, an early
    # paging bug looks like an infinite boot loop with nothing to read.
    '-no-reboot'
)
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
