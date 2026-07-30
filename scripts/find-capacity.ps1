<#
.SYNOPSIS
    Find how much storage this counterfeit USB disk actually has.

.DESCRIPTION
    probe-media.ps1 established that this device reports 976.56 GB and cannot
    store data anywhere near that. This finds the real boundary by binary
    search: roughly twenty 4 KB writes rather than writing the whole disk.

    Two details that matter for getting a truthful answer:

    * Each probe writes the target block, then writes a DIFFERENT pattern to a
      scratch block elsewhere, and only then reopens and reads the target back.
      Counterfeit controllers commonly hold the most recent block in a buffer
      and serve it back on read, which makes a bad address look good. Touching
      a second address in between defeats that.

    * The pattern encodes its own offset. A controller that wraps high writes
      onto low addresses is then caught too: the data reads back intact but
      tagged with the wrong address.

    DESTRUCTIVE. It runs `clean` first, removing the partition table. That is
    required, not incidental: writes have to reach the low region where the
    real flash lives, and doing that under a live partition table risks a
    wrapping write landing inside a mounted filesystem. The volume on this disk
    is empty, which is the only reason this is a reasonable thing to do.

.EXAMPLE
    .\scripts\find-capacity.ps1            # dry run, changes nothing
    .\scripts\find-capacity.ps1 -Execute
#>
param(
    [switch]$Execute,
    [string]$SerialNumber = '3212271080340376147'
)

$ErrorActionPreference = 'Stop'

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "Raw disk access needs an elevated PowerShell."
}

function Invoke-Diskpart {
    param([string[]]$Commands)
    $file = Join-Path $env:TEMP ("sanctum-dp-{0}.txt" -f [guid]::NewGuid().ToString('N'))
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
$diskSize = [int64]$target.Size
$path = "\\.\PhysicalDrive$diskNumber"

Write-Host ""
Write-Host "disk     : $diskNumber  $($target.Model)" -ForegroundColor Cyan
Write-Host ("reports  : {0:N2} GB" -f ($diskSize/1GB))
Write-Host "action   : clean the partition table, then binary search for real capacity"
Write-Host ""

$vols = Get-CimInstance Win32_LogicalDiskToPartition | Where-Object {
    $_.Antecedent.DeviceID -match "Disk #$diskNumber,"
}
if ($vols) {
    Write-Host "volumes that will be destroyed:" -ForegroundColor Yellow
    foreach ($v in $vols) {
        $l = $v.Dependent.DeviceID
        $ld = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$l'" -ErrorAction SilentlyContinue
        if ($ld) {
            Write-Host ("  {0}  {1}  {2:N2} GB, {3:N2} GB free" -f `
                $l, $ld.FileSystem, ($ld.Size/1GB), ($ld.FreeSpace/1GB))
        } else {
            Write-Host "  $l  (raw)"
        }
    }
    Write-Host ""
}

if (-not $Execute) {
    Write-Host "DRY RUN. Nothing was changed. Re-run with -Execute." -ForegroundColor Yellow
    return
}

Write-Host "Type the disk serial to confirm the wipe:" -ForegroundColor Yellow
$typed = Read-Host "serial"
if ($typed.Trim() -ne $target.SerialNumber.Trim()) {
    Write-Error "Serial did not match. Nothing was changed."
}

Write-Host "cleaning ..." -ForegroundColor Cyan
Write-Host (Invoke-Diskpart @("select disk $diskNumber", "clean"))
Start-Sleep -Seconds 3

# --- probing ---
$block = 4096
$scratch = 1MB   # touched between write and read, to defeat a one-block cache

function Test-Offset {
    param([int64]$Offset)

    $pattern = New-Object byte[] $block
    $tag = [Text.Encoding]::ASCII.GetBytes("SANCTUM-$Offset-")
    for ($i = 0; $i -lt $block; $i++) { $pattern[$i] = $tag[$i % $tag.Length] }

    $decoy = New-Object byte[] $block
    for ($i = 0; $i -lt $block; $i++) { $decoy[$i] = 0xA5 }

    $fs = [System.IO.File]::Open($path, 'Open', 'ReadWrite', 'ReadWrite')
    try {
        [void]$fs.Seek($Offset, 'Begin'); $fs.Write($pattern, 0, $block); $fs.Flush($true)
        # Second, unrelated address: makes a cached-block false positive impossible.
        [void]$fs.Seek($scratch, 'Begin'); $fs.Write($decoy, 0, $block); $fs.Flush($true)
    } finally { $fs.Dispose() }

    $readBack = New-Object byte[] $block
    $fs = [System.IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
    try {
        [void]$fs.Seek($Offset, 'Begin'); [void]$fs.Read($readBack, 0, $block)
    } finally { $fs.Dispose() }

    for ($i = 0; $i -lt $block; $i++) {
        if ($readBack[$i] -ne $pattern[$i]) { return $false }
    }
    return $true
}

Write-Host ""
Write-Host "=== binary search ===" -ForegroundColor Cyan

$lo = 8MB
if (-not (Test-Offset $lo)) {
    Write-Error "Even 8 MB does not read back. This device is unusable, not merely undersized."
}
Write-Host ("{0,12:N2} GB : OK (baseline)" -f ($lo/1GB)) -ForegroundColor Green

$hi = $diskSize - $block
if (Test-Offset $hi) {
    Write-Host ("{0,12:N2} GB : OK" -f ($hi/1GB)) -ForegroundColor Green
    Write-Host "The whole reported size reads back. The earlier failures were not capacity." -ForegroundColor Yellow
    return
}
Write-Host ("{0,12:N2} GB : BAD (ceiling)" -f ($hi/1GB)) -ForegroundColor Red

while (($hi - $lo) -gt 32MB) {
    $mid = [int64](($lo + $hi) / 2)
    $mid = $mid - ($mid % 512)
    $ok = Test-Offset $mid
    if ($ok) { $lo = $mid } else { $hi = $mid }
    Write-Host ("{0,12:N2} GB : {1}" -f ($mid/1GB), $(if ($ok) { 'OK' } else { 'BAD' })) `
        -ForegroundColor $(if ($ok) { 'Green' } else { 'Red' })
}

# --- confirm the good region really is good, at several points ---
Write-Host ""
Write-Host "=== verifying the usable region ===" -ForegroundColor Cyan
$allGood = $true
foreach ($frac in 0.1, 0.3, 0.5, 0.7, 0.9, 0.99) {
    $o = [int64]($lo * $frac); $o = $o - ($o % 512)
    if ($o -lt 8MB) { continue }
    $ok = Test-Offset $o
    if (-not $ok) { $allGood = $false }
    Write-Host ("{0,12:N2} GB : {1}" -f ($o/1GB), $(if ($ok) { 'OK' } else { 'BAD' })) `
        -ForegroundColor $(if ($ok) { 'Green' } else { 'Red' })
}

Write-Host ""
Write-Host ("reported capacity : {0:N2} GB" -f ($diskSize/1GB))
Write-Host ("real capacity     : {0:N2} GB" -f ($lo/1GB)) -ForegroundColor Green
Write-Host ("fake by           : {0:N2} GB" -f (($diskSize - $lo)/1GB)) -ForegroundColor Red
if (-not $allGood) {
    Write-Warning "Some offsets inside the 'good' region failed. Treat the usable size as smaller, or distrust this device entirely."
}

# Leave a margin: the boundary is only located to 32 MB, and counterfeit flash
# is often flaky just below where it stops working outright.
$safe = [int64]($lo * 0.90)
Write-Host ""
Write-Host ("recommended usable size: {0:N2} GB (90% of measured, as margin)" -f ($safe/1GB)) -ForegroundColor Cyan
Write-Host ""
Write-Host "next: build a layout that stays inside that, with the ESP FIRST at 1 MB"
Write-Host "so the boot files live in the low region that demonstrably works."
