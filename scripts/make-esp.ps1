<#
.SYNOPSIS
    Create an EFI System Partition on the sanctum USB SSD.

.DESCRIPTION
    Shrinks the existing NTFS volume and creates two new partitions:

      1 GB   FAT32, EFI System Partition   -> EFI\BOOT\BOOTX64.EFI
      32 GB  unformatted, reserved         -> the M6 filesystem

    The existing NTFS volume keeps every byte of its data; it is only made
    smaller. Nothing is formatted except the new ESP.

    SAFETY: the target disk is identified by SERIAL NUMBER, not by disk index.
    USB disk numbers are assigned at enumeration and can differ between boots,
    so "disk 2" is not a stable identity and is not trusted here. The script
    additionally refuses any disk that is not USB, or that carries a system or
    boot partition.

    Runs as a DRY RUN by default. Nothing is written without -Execute.

.EXAMPLE
    .\scripts\make-esp.ps1                # show the plan, change nothing
    .\scripts\make-esp.ps1 -Execute       # actually do it
#>
param(
    [switch]$Execute,
    # The sanctum USB SSD, as surveyed. Override only if you know why.
    [string]$SerialNumber = '3212271080340376147',
    [int]$EspSizeMB = 1024,
    [int]$ReservedSizeGB = 32
)

$ErrorActionPreference = 'Stop'

# --- must be elevated ---
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "This must run from an elevated PowerShell. Right-click PowerShell -> Run as administrator."
}

# --- find the disk by serial, never by index ---
$target = Get-CimInstance Win32_DiskDrive | Where-Object {
    $_.SerialNumber -and $_.SerialNumber.Trim() -eq $SerialNumber
}

if (-not $target) {
    Write-Host "Disks currently attached:" -ForegroundColor Yellow
    Get-CimInstance Win32_DiskDrive |
        Select-Object Index, Model, InterfaceType,
            @{n='SizeGB';e={[math]::Round($_.Size/1GB,2)}}, SerialNumber |
        Format-Table -AutoSize
    Write-Error "No disk with serial '$SerialNumber'. Is the USB SSD plugged in?"
}
if ($target -is [array]) {
    Write-Error "More than one disk reports serial '$SerialNumber'. Refusing to guess."
}

$diskNumber = [int]$target.Index

# --- refuse anything that is not the removable target ---
if ($target.InterfaceType -ne 'USB') {
    Write-Error "Disk $diskNumber is $($target.InterfaceType), not USB. Refusing."
}

$sysPart = Get-Partition -DiskNumber $diskNumber |
    Where-Object { $_.IsSystem -or $_.IsBoot }
if ($sysPart) {
    Write-Error "Disk $diskNumber carries a system or boot partition. Refusing."
}

$disk = Get-Disk -Number $diskNumber
if ($disk.PartitionStyle -ne 'GPT') {
    Write-Error "Disk $diskNumber is $($disk.PartitionStyle), not GPT. An ESP needs GPT."
}

# --- locate the NTFS volume to shrink ---
$dataPart = Get-Partition -DiskNumber $diskNumber |
    Where-Object { $_.DriveLetter } |
    Sort-Object Size -Descending |
    Select-Object -First 1
if (-not $dataPart) {
    Write-Error "No lettered partition found on disk $diskNumber."
}

$vol = Get-Volume -DriveLetter $dataPart.DriveLetter
$needBytes = ($EspSizeMB * 1MB) + ($ReservedSizeGB * 1GB)
$supported = Get-PartitionSupportedSize -DiskNumber $diskNumber -PartitionNumber $dataPart.PartitionNumber
$newSize = $dataPart.Size - $needBytes

Write-Host ''
Write-Host "target disk    : $diskNumber  $($target.Model)" -ForegroundColor Cyan
Write-Host "serial         : $($target.SerialNumber.Trim())"
Write-Host "interface      : $($target.InterfaceType)"
Write-Host ''
Write-Host ("shrink {0}: {1:N1} GB -> {2:N1} GB   (min possible {3:N1} GB)" -f `
    "$($dataPart.DriveLetter):", ($dataPart.Size/1GB), ($newSize/1GB), ($supported.SizeMin/1GB))
Write-Host ("  currently {0:N1} GB free of {1:N1} GB, {2} in use" -f `
    ($vol.SizeRemaining/1GB), ($vol.Size/1GB), $vol.FileSystem)
Write-Host ''
Write-Host "create         : $EspSizeMB MB FAT32, EFI System Partition"
Write-Host "create         : $ReservedSizeGB GB unformatted, reserved for the sanctum filesystem"
Write-Host ''

if ($newSize -lt $supported.SizeMin) {
    Write-Error ("Cannot shrink that far. Minimum is {0:N1} GB." -f ($supported.SizeMin/1GB))
}

if (-not $Execute) {
    Write-Host "DRY RUN. Nothing was changed. Re-run with -Execute to apply." -ForegroundColor Yellow
    return
}

Write-Host "Type the disk serial to confirm:" -ForegroundColor Yellow
$typed = Read-Host "serial"
if ($typed.Trim() -ne $target.SerialNumber.Trim()) {
    Write-Error "Serial did not match. Nothing was changed."
}

# --- shrink ---
Write-Host "shrinking $($dataPart.DriveLetter): ..." -ForegroundColor Cyan
Resize-Partition -DiskNumber $diskNumber -PartitionNumber $dataPart.PartitionNumber -Size $newSize

# --- ESP ---
# Created as basic data, formatted, and only then retyped to ESP. Formatting a
# partition that already carries the ESP type GUID is unreliable through
# Format-Volume, so the order here is deliberate.
Write-Host "creating the ESP ..." -ForegroundColor Cyan
$esp = New-Partition -DiskNumber $diskNumber -Size ($EspSizeMB * 1MB) -AssignDriveLetter
Format-Volume -Partition $esp -FileSystem FAT32 -NewFileSystemLabel 'SANCTUM' -Confirm:$false | Out-Null
$espLetter = (Get-Partition -DiskNumber $diskNumber -PartitionNumber $esp.PartitionNumber).DriveLetter

Set-Partition -DiskNumber $diskNumber -PartitionNumber $esp.PartitionNumber `
    -GptType '{c12a7328-f81f-11d2-ba4b-00a0c93ec93b}'

# --- reserved partition for the M6 filesystem, deliberately unformatted ---
Write-Host "creating the reserved partition ..." -ForegroundColor Cyan
New-Partition -DiskNumber $diskNumber -Size ($ReservedSizeGB * 1GB) | Out-Null

Write-Host ''
Write-Host "done." -ForegroundColor Green
Get-Partition -DiskNumber $diskNumber |
    Select-Object PartitionNumber, DriveLetter,
        @{n='SizeGB';e={[math]::Round($_.Size/1GB,2)}}, Type, GptType |
    Format-Table -AutoSize

if ($espLetter) {
    Write-Host "ESP is $($espLetter):  -- deploy with:" -ForegroundColor Green
    Write-Host "  .\scripts\deploy.ps1 -EspDrive $($espLetter): -Release"
    Write-Host ''
    Write-Host "If Windows drops that letter after the type change, remount it with:" -ForegroundColor Yellow
    Write-Host "  mountvol S: /S      (elevated)"
} else {
    Write-Host "The ESP has no drive letter. Mount it elevated with: mountvol S: /S" -ForegroundColor Yellow
}
