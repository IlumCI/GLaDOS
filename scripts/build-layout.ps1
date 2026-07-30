<#
.SYNOPSIS
    Lay out the counterfeit USB disk using only the flash that really exists.

.DESCRIPTION
    find-capacity.ps1 measured 14.67 GB of real storage behind a device that
    advertises 976.56 GB. This builds a partition layout that lives entirely
    inside the verified region and never touches the fictional tail.

    Two deliberate choices:

    MBR, not GPT. A GPT disk keeps a backup header and partition array in the
    LAST sectors of the disk -- which here means around 976 GB, inside flash
    that does not exist. The disk would be born with an unwritable backup GPT.
    An MBR keeps its entire partition table in LBA 0, comfortably inside the
    working region. UEFI firmware boots MBR media perfectly well; the ESP is
    simply marked with partition type 0xEF instead of a GPT type GUID.

    ESP first, at 1 MB. The boot files go in the lowest, most thoroughly
    verified flash rather than at some high offset. This is also why the
    earlier shrink-based approach was wrong for this device: it put the ESP at
    966 GB, in flash that was never there.

    Everything past the layout is left UNPARTITIONED, so Windows never offers
    the fake capacity to anything.

    DESTRUCTIVE: runs `clean`. The disk is expected to be empty already.

.EXAMPLE
    .\scripts\build-layout.ps1              # dry run
    .\scripts\build-layout.ps1 -Execute
#>
param(
    [switch]$Execute,
    [string]$SerialNumber = '3212271080340376147',
    [int]$EspSizeMB = 512,
    [int]$ReservedSizeMB = 4096,
    # Measured real capacity, with the 10% margin find-capacity recommends.
    [double]$SafeLimitGB = 13.2,
    [string]$EspLetter,
    # Mark the ESP as MBR type 0xEF, which is what the UEFI spec asks for.
    # Windows may then hide the volume; -NoEspType leaves it as plain FAT32
    # (type 0x0C), which most firmware still boots from removable media.
    [switch]$NoEspType
)

$ErrorActionPreference = 'Stop'

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "This must run from an elevated PowerShell."
}

function Invoke-Diskpart {
    param([string[]]$Commands)
    $file = Join-Path $env:TEMP ("glados-dp-{0}.txt" -f [guid]::NewGuid().ToString('N'))
    Set-Content -Path $file -Value ($Commands -join "`r`n") -Encoding ASCII
    try { return (& diskpart.exe /s $file | Out-String) }
    finally { if (Test-Path $file) { Remove-Item -LiteralPath $file -Force -ErrorAction SilentlyContinue } }
}

$target = Get-CimInstance Win32_DiskDrive | Where-Object {
    $_.SerialNumber -and $_.SerialNumber.Trim() -eq $SerialNumber
}
if (-not $target) { Write-Error "No disk with serial '$SerialNumber'." }
if ($target -is [array]) { Write-Error "Multiple disks share that serial. Refusing." }
if ($target.InterfaceType -ne 'USB') { Write-Error "Not a USB disk. Refusing." }
$diskNumber = [int]$target.Index

$totalMB = $EspSizeMB + $ReservedSizeMB
$safeMB = [int]($SafeLimitGB * 1024)
if ($totalMB -gt $safeMB) {
    Write-Error "Layout wants $totalMB MB but only $safeMB MB is verified good. Reduce the sizes."
}

if (-not $EspLetter) {
    $used = (Get-CimInstance Win32_LogicalDisk | ForEach-Object { $_.DeviceID.Substring(0,1) })
    $EspLetter = ('S','T','U','V','W','X','Y','Z' | Where-Object { $used -notcontains $_ } | Select-Object -First 1)
    if (-not $EspLetter) { Write-Error "No free drive letter available." }
}
$EspLetter = $EspLetter.TrimEnd(':')

Write-Host ""
Write-Host "disk           : $diskNumber  $($target.Model)" -ForegroundColor Cyan
Write-Host ("advertised     : {0:N2} GB" -f ($target.Size/1GB))
Write-Host ("real           : {0:N2} GB verified" -f $SafeLimitGB) -ForegroundColor Yellow
Write-Host ""
Write-Host "scheme         : MBR (no backup table in the nonexistent tail)"
Write-Host "partition 1    : $EspSizeMB MB FAT32 at 1 MB, letter $EspLetter, ESP"
Write-Host "partition 2    : $ReservedSizeMB MB unformatted, reserved for the glados filesystem"
Write-Host ("unpartitioned  : everything above {0:N0} MB, including {1:N0} MB that does not exist" -f `
    $totalMB, (($target.Size/1MB) - $safeMB))
Write-Host ""

if (-not $Execute) {
    Write-Host "DRY RUN. Nothing was changed. Re-run with -Execute." -ForegroundColor Yellow
    return
}

Write-Host "Type the disk serial to confirm the wipe:" -ForegroundColor Yellow
$typed = Read-Host "serial"
if ($typed.Trim() -ne $target.SerialNumber.Trim()) {
    Write-Error "Serial did not match. Nothing was changed."
}

# Partition 1 is created as an ordinary FAT32 data partition and formatted as
# one. Windows refuses to write a filesystem to something already marked as an
# EFI System Partition, so the type is set afterwards, not before.
Write-Host "building ..." -ForegroundColor Cyan
$out = Invoke-Diskpart @(
    "select disk $diskNumber",
    "clean",
    "convert mbr",
    "create partition primary size=$EspSizeMB",
    "format fs=fat32 quick label=GLADOS",
    "assign letter=$EspLetter",
    "active",
    "create partition primary size=$ReservedSizeMB",
    "list partition"
)
Write-Host $out
Start-Sleep -Seconds 2

$vol = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='${EspLetter}:'" -ErrorAction SilentlyContinue
if (-not $vol -or $vol.FileSystem -notmatch 'FAT') {
    Write-Error "No FAT filesystem on ${EspLetter}:. The layout did not complete."
}
Write-Host "formatted ${EspLetter}: as $($vol.FileSystem)" -ForegroundColor Green

if (-not $NoEspType) {
    Write-Host "marking partition 1 as MBR type 0xEF (EFI System) ..." -ForegroundColor Cyan
    Invoke-Diskpart @("select disk $diskNumber", "select partition 1", "set id=ef") | Out-Null
    Start-Sleep -Seconds 2
    # Windows tends to drop the letter when a partition becomes an ESP.
    Invoke-Diskpart @("select disk $diskNumber", "select partition 1", "assign letter=$EspLetter") | Out-Null
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "=== final state ===" -ForegroundColor Cyan
Get-CimInstance Win32_DiskPartition | Where-Object { $_.DiskIndex -eq $diskNumber } |
    Select-Object Name, Type, Bootable,
        @{n='SizeMB';e={[math]::Round($_.Size/1MB,0)}},
        @{n='OffsetMB';e={[math]::Round($_.StartingOffset/1MB,0)}} |
    Sort-Object OffsetMB | Format-Table -AutoSize

$vol = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='${EspLetter}:'" -ErrorAction SilentlyContinue
if ($vol) {
    Write-Host ("ESP: {0}  {1}  {2:N0} MB" -f $vol.DeviceID, $vol.FileSystem, ($vol.Size/1MB)) -ForegroundColor Green
} else {
    Write-Warning "${EspLetter}: is not mounted. Windows hides ESPs; remount with: mountvol ${EspLetter}: /S"
    Write-Warning "Or re-run with -NoEspType to leave it as plain FAT32."
}

Write-Host ""
Write-Host "next:" -ForegroundColor Green
Write-Host "  .\scripts\deploy.ps1 -EspDrive ${EspLetter}: -Release"
Write-Host "  .\scripts\verify-esp.ps1 -EspDrive ${EspLetter}:"
Write-Host ""
Write-Host "verify-esp boots this exact disk under OVMF. If it boots there, the"
Write-Host "layout is right and only the GF63's firmware is left to convince."
