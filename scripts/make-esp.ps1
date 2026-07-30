<#
.SYNOPSIS
    Create an EFI System Partition on the sanctum USB SSD.

.DESCRIPTION
    Shrinks the existing NTFS volume and creates two new partitions:

      1 GB   FAT32, EFI System Partition   -> EFI\BOOT\BOOTX64.EFI
      32 GB  unformatted, reserved         -> the M6 filesystem

    The existing NTFS volume keeps every byte of its data; it is only made
    smaller. Nothing is formatted except the new ESP.

    Uses diskpart rather than the Storage cmdlets (Get-Disk, New-Partition,
    Resize-Partition). Those query the root\Microsoft\Windows\Storage WMI
    namespace, which does not answer on this machine -- a trimmed Windows 10
    IoT Enterprise LTSC image. Win32_DiskDrive lives in the older CIMv2
    provider and does work, so it is used for identification; diskpart goes
    through VDS and does not depend on the broken namespace either.

    SAFETY: the target disk is identified by SERIAL NUMBER, not by disk index.
    USB disk numbers are assigned at enumeration and can differ between boots,
    so "disk 2" is not a stable identity and is not trusted here. Before any
    write, diskpart's own `detail disk` output is re-checked against the
    expected model string, so both providers must agree on which disk this is.

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
    [int]$ReservedSizeMB = 32768,
    [string]$EspLetter
)

$ErrorActionPreference = 'Stop'

function Invoke-Diskpart {
    param([string[]]$Commands)
    $file = Join-Path $env:TEMP ("sanctum-dp-{0}.txt" -f [guid]::NewGuid().ToString('N'))
    Set-Content -Path $file -Value ($Commands -join "`r`n") -Encoding ASCII
    try {
        $out = & diskpart.exe /s $file
        return ($out | Out-String)
    } finally {
        if (Test-Path $file) { Remove-Item -LiteralPath $file -Force -ErrorAction SilentlyContinue }
    }
}

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

if ($target.InterfaceType -ne 'USB') {
    Write-Error "Disk $diskNumber is $($target.InterfaceType), not USB. Refusing."
}

# --- partitions, via the CIMv2 provider that actually works here ---
$parts = @(Get-CimInstance Win32_DiskPartition | Where-Object { $_.DiskIndex -eq $diskNumber })
if ($parts.Count -eq 0) {
    Write-Error "No partitions visible on disk $diskNumber."
}
if ($parts | Where-Object { $_.BootPartition }) {
    Write-Error "Disk $diskNumber carries a boot partition. Refusing."
}
if ($parts[0].Type -notmatch 'GPT') {
    Write-Error "Disk $diskNumber partition type is '$($parts[0].Type)', not GPT. An ESP needs GPT."
}

# WMI numbers partitions from 0; diskpart numbers them from 1.
$dataWmi = $parts | Sort-Object Size -Descending | Select-Object -First 1
$dataIndex = [int]($dataWmi.Name -replace '.*Partition\s*#','') + 1

$letterMap = Get-CimInstance Win32_LogicalDiskToPartition | Where-Object {
    $_.Antecedent.DeviceID -eq $dataWmi.DeviceID
}
$dataLetter = if ($letterMap) { $letterMap.Dependent.DeviceID } else { '(none)' }

# --- pick a free drive letter for the ESP ---
if (-not $EspLetter) {
    $used = (Get-CimInstance Win32_LogicalDisk | ForEach-Object { $_.DeviceID.Substring(0,1) })
    $EspLetter = ('S','T','U','V','W','X','Y','Z' | Where-Object { $used -notcontains $_ } | Select-Object -First 1)
    if (-not $EspLetter) { Write-Error "No free drive letter available for the ESP." }
}
$EspLetter = $EspLetter.TrimEnd(':')

$shrinkMB = $EspSizeMB + $ReservedSizeMB

Write-Host ''
Write-Host "target disk    : $diskNumber  $($target.Model)" -ForegroundColor Cyan
Write-Host "serial         : $($target.SerialNumber.Trim())"
Write-Host "interface      : $($target.InterfaceType)"
Write-Host ("size           : {0:N2} GB" -f ($target.Size/1GB))
Write-Host ''
Write-Host ("shrink         : partition $dataIndex ($dataLetter) by {0:N1} GB, from {1:N1} GB" -f `
    ($shrinkMB/1024), ($dataWmi.Size/1GB))
Write-Host "create         : $EspSizeMB MB FAT32 EFI System Partition, letter $EspLetter"
Write-Host "create         : $ReservedSizeMB MB unformatted, reserved for the sanctum filesystem"
Write-Host ''

# --- cross-check: make diskpart agree this is the same disk ---
Write-Host "verifying with diskpart ..." -ForegroundColor Cyan
$detail = Invoke-Diskpart @("select disk $diskNumber", "detail disk", "list partition")
Write-Host $detail

$modelToken = ($target.Model -split '\s+')[0]
if ($detail -notmatch [regex]::Escape($modelToken)) {
    Write-Error "diskpart's disk $diskNumber does not mention '$modelToken'. The two providers disagree. Refusing."
}
if ($detail -match 'Boot Disk\s*:\s*Yes' -or $detail -match 'Pagefile Disk\s*:\s*Yes') {
    Write-Error "diskpart reports disk $diskNumber as a boot or pagefile disk. Refusing."
}
Write-Host "diskpart agrees this is $modelToken and it is not the boot disk." -ForegroundColor Green

$script = @(
    "select disk $diskNumber",
    "select partition $dataIndex",
    "shrink desired=$shrinkMB minimum=$shrinkMB",
    "create partition efi size=$EspSizeMB",
    "format fs=fat32 quick label=SANCTUM",
    "assign letter=$EspLetter",
    "create partition primary size=$ReservedSizeMB",
    "list partition"
)

Write-Host ''
Write-Host "diskpart script to be run:" -ForegroundColor Yellow
$script | ForEach-Object { Write-Host "  $_" }
Write-Host ''

if (-not $Execute) {
    Write-Host "DRY RUN. Nothing was changed. Re-run with -Execute to apply." -ForegroundColor Yellow
    return
}

Write-Host "Type the disk serial to confirm:" -ForegroundColor Yellow
$typed = Read-Host "serial"
if ($typed.Trim() -ne $target.SerialNumber.Trim()) {
    Write-Error "Serial did not match. Nothing was changed."
}

Write-Host "running ..." -ForegroundColor Cyan
$result = Invoke-Diskpart $script
Write-Host $result

if ($result -match 'DiskPart has encountered an error' -or $result -match 'failed') {
    Write-Warning "diskpart reported a problem. Read the output above carefully before continuing."
} else {
    Write-Host "done." -ForegroundColor Green
    Write-Host ''
    Write-Host "deploy with:" -ForegroundColor Green
    Write-Host "  .\scripts\deploy.ps1 -EspDrive ${EspLetter}: -Release"
    Write-Host "then reboot and hold F11."
}
