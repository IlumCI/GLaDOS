<#
.SYNOPSIS
    Copy glados to a real EFI System Partition for bare-metal boot.

.DESCRIPTION
    Writes EFI\BOOT\BOOTX64.EFI and the contents of esp\GLADOS -- the model,
    the tokenizer, and the root certificate bundle, each skipped when already
    byte-identical. It creates no partitions, formats nothing, and touches no
    other file -- deliberately. Repartitioning is a separate, manual,
    deliberate act.

    The payload is not optional garnish. Without model.bin the [ai] section
    reports no checkpoint; without roots.der every certificate fails to
    validate and https encrypts without authenticating.

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
    # cargo writes progress ("Compiling glados ...") to stderr, and under
    # $ErrorActionPreference = 'Stop' PowerShell turns native stderr into a
    # terminating error. That made this script abort on any build that was not
    # already cached -- it only ever appeared to work because the build was
    # usually up to date. Relax the preference around cargo and judge it by its
    # exit code, which is the only thing that actually means failure.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    if ($Release) { cargo build --release; $profileDir = 'release' }
    else          { cargo build;           $profileDir = 'debug'   }
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($code -ne 0) { Write-Error "cargo build failed (exit $code)" }
} finally {
    Pop-Location
}

$efi = Join-Path $root "target\x86_64-unknown-uefi\$profileDir\glados.efi"
if (-not (Test-Path $efi)) { Write-Error "missing build artifact: $efi" }

$bootDir = Join-Path "$EspDrive\" 'EFI\BOOT'
New-Item -ItemType Directory -Force -Path $bootDir | Out-Null
$dest = Join-Path $bootDir 'BOOTX64.EFI'
Copy-Item $efi $dest -Force

$info = Get-Item $dest
Write-Host ''
Write-Host ("deployed {0:N1} KB -> {1}" -f ($info.Length/1KB), $dest) -ForegroundColor Green

# The model is part of the system now, not an optional extra: without it the
# [ai] section reports no checkpoint and `gen` has nothing to run. Kept beside
# the binary on the ESP because the firmware's FAT driver is the only
# filesystem that exists before ExitBootServices, and that is where the
# weights have to be read from.
$payload = Join-Path $root 'esp\GLADOS'
if (Test-Path $payload) {
    $targetDir = Join-Path "$EspDrive\" 'GLADOS'
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

    # Refuse rather than run out of room halfway. Copy-Item on a full volume
    # leaves a truncated file behind, and a truncated model.bin is not a failed
    # deploy -- it is a system that boots, reports a checkpoint, and is wrong.
    # Counted against free space plus whatever the existing copies would
    # release, since most deploys overwrite.
    if ($vol) {
        $need = (Get-ChildItem $payload -File | Measure-Object -Sum Length).Sum
        $reclaim = 0
        foreach ($f in Get-ChildItem $payload -File) {
            $existing = Join-Path $targetDir $f.Name
            if (Test-Path $existing) { $reclaim += (Get-Item $existing).Length }
        }
        $avail = $vol.FreeSpace + $reclaim
        if ($need -gt $avail) {
            Write-Error ("payload is {0:N0} MB but only {1:N0} MB is available on {2}. " -f `
                ($need/1MB), ($avail/1MB), $EspDrive) `
                -ErrorAction Continue
            Write-Error ("Re-lay the stick with a larger ESP, e.g. " +
                ".\scripts\build-layout.ps1 -EspSizeMB 8192 -ReservedSizeMB 4096 -Execute")
        }
    }

    foreach ($f in Get-ChildItem $payload -File) {
        $to = Join-Path $targetDir $f.Name
        # Skip files already byte-identical: the ESP is on a slow USB stick and
        # a model is a lot of bytes to rewrite for nothing.
        $same = (Test-Path $to) -and
                ((Get-FileHash $f.FullName -Algorithm SHA256).Hash -eq
                 (Get-FileHash $to -Algorithm SHA256).Hash)
        if ($same) {
            Write-Host ("  {0,-16} unchanged ({1:N0} B)" -f $f.Name, $f.Length)
        } else {
            Copy-Item $f.FullName $to -Force
            Write-Host ("  {0,-16} copied ({1:N0} B)" -f $f.Name, $f.Length) -ForegroundColor Green
        }
    }
} else {
    Write-Warning "no payload at $payload -- the system will boot without a model."
}

Write-Host 'reboot and hold F11 to pick the USB device.'
