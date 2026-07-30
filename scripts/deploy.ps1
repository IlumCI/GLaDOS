<#
.SYNOPSIS
    Copy sanctum to a real EFI System Partition for bare-metal boot.

.DESCRIPTION
    Writes only EFI\BOOT\BOOTX64.EFI on the target volume. It creates no
    partitions, formats nothing, and touches no other file -- deliberately.
    Repartitioning is a separate, manual, deliberate act.

    On the GF63: reboot and hold F11 for the boot menu, then pick the USB
    device. The internal Windows NVMe is never involved.

.EXAMPLE
    .\scripts\deploy.ps1 -EspDrive S:
    .\scripts\deploy.ps1 -EspDrive S: -Release
#>
param(
    [Parameter(Mandatory = $true)][string]$EspDrive,
    [switch]$Release
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

$cargoBin = Join-Path $env:USERPROFILE 'scoop\persist\rustup-msvc\.cargo\bin'
if (Test-Path $cargoBin) { $env:PATH = "$cargoBin;$env:PATH" }

$EspDrive = $EspDrive.TrimEnd('\')
if ($EspDrive -notmatch '^[A-Za-z]:$') {
    Write-Error "EspDrive should look like 'S:'"
}
if (-not (Test-Path "$EspDrive\")) {
    Write-Error "$EspDrive is not mounted."
}

# Guard rail: refuse to write to the Windows volume by accident.
if ($EspDrive -ieq $env:SystemDrive) {
    Write-Error "Refusing to deploy to the system drive ($EspDrive)."
}

# Win32_LogicalDisk, not Get-Volume: the Storage cmdlets query the
# root\Microsoft\Windows\Storage namespace, which does not answer on this
# trimmed IoT Enterprise LTSC image. CIMv2 works.
$vol = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$EspDrive'" -ErrorAction SilentlyContinue
if ($vol) {
    Write-Host ("target : {0}  fs={1}  size={2:N2} GB  free={3:N2} GB" -f `
        $EspDrive, $vol.FileSystem, ($vol.Size/1GB), ($vol.FreeSpace/1GB))
    if ($vol.FileSystem -and $vol.FileSystem -notmatch 'FAT') {
        Write-Warning "$EspDrive is $($vol.FileSystem), not FAT. UEFI only boots FAT12/16/32."
    }
} else {
    Write-Warning "Could not read volume info for $EspDrive. Continuing anyway."
}

Push-Location $root
try {
    if ($Release) { cargo build --release; $profileDir = 'release' }
    else          { cargo build;           $profileDir = 'debug'   }
    if ($LASTEXITCODE -ne 0) { Write-Error "cargo build failed" }
} finally {
    Pop-Location
}

$efi = Join-Path $root "target\x86_64-unknown-uefi\$profileDir\sanctum.efi"
if (-not (Test-Path $efi)) { Write-Error "missing build artifact: $efi" }

$bootDir = Join-Path "$EspDrive\" 'EFI\BOOT'
New-Item -ItemType Directory -Force -Path $bootDir | Out-Null
$dest = Join-Path $bootDir 'BOOTX64.EFI'
Copy-Item $efi $dest -Force

$info = Get-Item $dest
Write-Host ''
Write-Host ("deployed {0:N1} KB -> {1}" -f ($info.Length/1KB), $dest) -ForegroundColor Green
Write-Host 'reboot and hold F11 to pick the USB device.'
