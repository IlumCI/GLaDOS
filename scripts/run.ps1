<#
.SYNOPSIS
    Build sanctum and boot it in QEMU under OVMF.

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

# --- locate OVMF (UEFI firmware for the guest) ---
if (-not $Ovmf) {
    $share = Join-Path (Split-Path -Parent $Qemu) 'share'
    $names = @('edk2-x86_64-code.fd', 'OVMF.fd', 'OVMF_CODE.fd', 'bios-256k.bin')
    foreach ($n in $names) {
        $p = Join-Path $share $n
        if (Test-Path $p) { $Ovmf = $p; break }
    }
}
if (-not $Ovmf) {
    Write-Error "No OVMF firmware found next to QEMU. Pass -Ovmf <path to edk2-x86_64-code.fd>."
}

# --- build ---
Push-Location $root
try {
    if ($Release) {
        cargo build --release
        $profileDir = 'release'
    } else {
        cargo build
        $profileDir = 'debug'
    }
    if ($LASTEXITCODE -ne 0) { Write-Error "cargo build failed" }
} finally {
    Pop-Location
}

$efi = Join-Path $root "target\x86_64-unknown-uefi\$profileDir\sanctum.efi"
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
    '-m', '512M',
    '-bios', $Ovmf,
    '-drive', "format=raw,file=fat:rw:$esp",
    '-serial', 'stdio',
    '-net', 'none',
    # Stop on triple fault instead of rebooting forever. Without this, an early
    # paging bug looks like an infinite boot loop with nothing to read.
    '-no-reboot'
)
if ($TraceFaults) { $qemuArgs += @('-d', 'int,cpu_reset') }
if ($Gdb)         { $qemuArgs += @('-s', '-S') }

Write-Host "qemu : $Qemu"
Write-Host "ovmf : $Ovmf"
Write-Host "esp  : $esp"
if ($Gdb) { Write-Host "paused for gdb on :1234 -- connect, then 'continue'" -ForegroundColor Yellow }
Write-Host ''

& $Qemu @qemuArgs
