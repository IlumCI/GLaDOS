<#
.SYNOPSIS
    Finish an ESP that make-esp.ps1 created but could not format.

.DESCRIPTION
    diskpart can create an EFI System Partition and then fail to format it in
    the same script run, reporting "The parameter is incorrect" at 0 percent.
    The partition is fine; Windows simply has not surfaced a volume object for
    it yet, and diskpart's `format` needs one. The remaining commands in the
    script are then skipped.

    This picks up from there:

      1. rescan, so Windows re-enumerates the new partition
      2. assign a drive letter to the ESP (identified by its System type,
         parsed from diskpart's own output -- never by a guessed index)
      3. format it FAT32 with format.com, which handles this reliably
      4. create the reserved partition from whatever free space is left

    Step 4 deliberately passes no size. After the shrink, the ESP and the
    Microsoft Reserved partition, the leftover is slightly under a round
    32768 MB, so asking for that exact figure fails. Taking "the rest" is both
    correct and immune to the arithmetic.

    Safe to re-run: every step checks whether it is already done.

.EXAMPLE
    .\scripts\finish-esp.ps1
    .\scripts\finish-esp.ps1 -EspLetter S
#>
param(
    [string]$SerialNumber = '3212271080340376147',
    [string]$EspLetter,
    [switch]$SkipReserved
)

$ErrorActionPreference = 'Stop'

function Invoke-Diskpart {
    param([string[]]$Commands)
    $file = Join-Path $env:TEMP ("sanctum-dp-{0}.txt" -f [guid]::NewGuid().ToString('N'))
    Set-Content -Path $file -Value ($Commands -join "`r`n") -Encoding ASCII
    try { return (& diskpart.exe /s $file | Out-String) }
    finally { if (Test-Path $file) { Remove-Item -LiteralPath $file -Force -ErrorAction SilentlyContinue } }
}

# GPT partition type GUIDs.
$GUID_BASIC = 'ebd0a0a2-b9e5-4433-87c0-68b6b72699c7'
$GUID_ESP   = 'c12a7328-f81f-11d2-ba4b-00a0c93ec93b'

function Set-PartType {
    param([int]$Disk, [int]$Part, [string]$Guid)
    Invoke-Diskpart @("select disk $Disk", "select partition $Part", "set id=$Guid") | Out-Null
    Start-Sleep -Seconds 1
}

function Set-PartLetter {
    param([int]$Disk, [int]$Part, [string]$Letter)
    # Errors here are expected and harmless when the letter is already assigned.
    Invoke-Diskpart @("select disk $Disk", "select partition $Part", "assign letter=$Letter") | Out-Null
    Start-Sleep -Seconds 1
}

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "This must run from an elevated PowerShell."
}

# --- identify the disk by serial ---
$target = Get-CimInstance Win32_DiskDrive | Where-Object {
    $_.SerialNumber -and $_.SerialNumber.Trim() -eq $SerialNumber
}
if (-not $target) { Write-Error "No disk with serial '$SerialNumber'. Is the USB SSD plugged in?" }
if ($target -is [array]) { Write-Error "Multiple disks share that serial. Refusing." }
if ($target.InterfaceType -ne 'USB') { Write-Error "Disk is $($target.InterfaceType), not USB. Refusing." }
$diskNumber = [int]$target.Index

Write-Host ""
Write-Host "disk $diskNumber  $($target.Model)  serial $($target.SerialNumber.Trim())" -ForegroundColor Cyan

# --- confirm the ESP exists ---
$esp = Get-CimInstance Win32_DiskPartition |
    Where-Object { $_.DiskIndex -eq $diskNumber -and $_.Type -match 'System' }
if (-not $esp) {
    Write-Error "No GPT System partition on disk $diskNumber. Run make-esp.ps1 first."
}
Write-Host ("found ESP: {0}  {1:N0} MB" -f $esp.Name, ($esp.Size/1MB)) -ForegroundColor Green

# --- 1. rescan and locate the ESP's diskpart index by TYPE, not by guess ---
Write-Host ""
Write-Host "=== rescan and locate ===" -ForegroundColor Cyan
$list = Invoke-Diskpart @("rescan", "select disk $diskNumber", "list partition")
Write-Host $list

$m = [regex]::Match($list, "(?m)^\s*Partition\s+(\d+)\s+System\s")
if (-not $m.Success) {
    Write-Error "diskpart does not list a System partition on disk $diskNumber."
}
$espIndex = [int]$m.Groups[1].Value
Write-Host "ESP is diskpart partition $espIndex" -ForegroundColor Green

# --- 2. drive letter ---
if (-not $EspLetter) {
    $used = (Get-CimInstance Win32_LogicalDisk | ForEach-Object { $_.DeviceID.Substring(0,1) })
    $EspLetter = ('S','T','U','V','W','X','Y','Z' | Where-Object { $used -notcontains $_ } | Select-Object -First 1)
    if (-not $EspLetter) { Write-Error "No free drive letter available." }
}
$EspLetter = $EspLetter.TrimEnd(':')

Write-Host ""
Write-Host "=== assign letter $EspLetter ===" -ForegroundColor Cyan
$assign = Invoke-Diskpart @(
    "select disk $diskNumber",
    "select partition $espIndex",
    "assign letter=$EspLetter"
)
Write-Host $assign

# --- 3. format with format.com, not diskpart ---
Write-Host ""
Write-Host "=== format ${EspLetter}: as FAT32 ===" -ForegroundColor Cyan
$vol = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='${EspLetter}:'" -ErrorAction SilentlyContinue
if ($vol -and $vol.FileSystem -match 'FAT') {
    Write-Host "already FAT ($($vol.FileSystem)); leaving it alone." -ForegroundColor Green
} else {
    # Windows will not write a filesystem to a partition typed as an EFI System
    # Partition. format.com reports "Invalid media or Track 0 bad - disk
    # unusable", which reads like failing hardware and is actually a protection
    # policy; diskpart's own format fails on the same partition for the same
    # reason, with an equally unhelpful message.
    #
    # So: retype to Basic Data, format as ordinary FAT32, retype back. Only the
    # GPT type field changes -- the filesystem written is byte-identical, and
    # firmware locates the ESP by exactly that field, so the end result is a
    # perfectly ordinary ESP.
    Write-Host "retyping to Basic Data so Windows will format it ..." -ForegroundColor Yellow
    Set-PartType -Disk $diskNumber -Part $espIndex -Guid $GUID_BASIC
    Set-PartLetter -Disk $diskNumber -Part $espIndex -Letter $EspLetter

    # `echo Y|` covers builds whose format.com still prompts despite /Y.
    $fmt = & cmd.exe /c "echo Y| format ${EspLetter}: /FS:FAT32 /Q /V:SANCTUM /Y 2>&1"
    Write-Host ($fmt | Out-String)
    Start-Sleep -Seconds 2

    $vol = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='${EspLetter}:'" -ErrorAction SilentlyContinue
    if (-not $vol -or $vol.FileSystem -notmatch 'FAT') {
        Write-Warning "Format failed even as Basic Data. Restoring the ESP type before giving up."
        Set-PartType -Disk $diskNumber -Part $espIndex -Guid $GUID_ESP
        Write-Error "Could not create a FAT filesystem on ${EspLetter}:."
    }
    Write-Host "formatted: $($vol.FileSystem)" -ForegroundColor Green

    Write-Host "restoring the EFI System Partition type ..." -ForegroundColor Yellow
    Set-PartType -Disk $diskNumber -Part $espIndex -Guid $GUID_ESP
    Set-PartLetter -Disk $diskNumber -Part $espIndex -Letter $EspLetter

    $check = Get-CimInstance Win32_DiskPartition |
        Where-Object { $_.DiskIndex -eq $diskNumber -and $_.Type -match 'System' }
    if ($check) {
        Write-Host "ESP type restored." -ForegroundColor Green
    } else {
        Write-Warning "The partition is no longer typed as System. Firmware finds the ESP by that type."
    }
}

# --- 4. reserved partition from whatever is left ---
if (-not $SkipReserved) {
    Write-Host ""
    Write-Host "=== reserved partition ===" -ForegroundColor Cyan
    $before = @(Get-CimInstance Win32_DiskPartition | Where-Object { $_.DiskIndex -eq $diskNumber }).Count
    # No size given on purpose: the leftover is a little under 32768 MB, so
    # naming that figure fails. "Whatever remains" is correct and robust.
    $res = Invoke-Diskpart @(
        "select disk $diskNumber",
        "create partition primary",
        "list partition"
    )
    Write-Host $res
    $after = @(Get-CimInstance Win32_DiskPartition | Where-Object { $_.DiskIndex -eq $diskNumber }).Count
    if ($after -gt $before) {
        Write-Host "reserved partition created." -ForegroundColor Green
    } else {
        Write-Warning "No new partition appeared. There may be no free space left, which is harmless."
    }
}

# --- summary ---
Write-Host ""
Write-Host "=== final state ===" -ForegroundColor Cyan
Get-CimInstance Win32_DiskPartition | Where-Object { $_.DiskIndex -eq $diskNumber } |
    Select-Object Name, Type, Bootable,
        @{n='SizeMB';e={[math]::Round($_.Size/1MB,0)}},
        @{n='OffsetMB';e={[math]::Round($_.StartingOffset/1MB,0)}} |
    Sort-Object OffsetMB | Format-Table -AutoSize

Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' |
    Select-Object DeviceID, VolumeName, FileSystem,
        @{n='SizeMB';e={[math]::Round($_.Size/1MB,0)}} |
    Format-Table -AutoSize

Write-Host "next:" -ForegroundColor Green
Write-Host "  .\scripts\deploy.ps1 -EspDrive ${EspLetter}: -Release"
Write-Host "  .\scripts\verify-esp.ps1 -EspDrive ${EspLetter}:"
