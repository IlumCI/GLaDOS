<#
.SYNOPSIS
    Test whether this USB disk can actually store data at high offsets.

.DESCRIPTION
    Every write that has failed so far sits at roughly 966 GB into the disk;
    everything that succeeded sits near the start. That is the signature of a
    counterfeit-capacity USB device -- a controller that reports a large size
    but has far less real flash behind it, silently discarding or wrapping
    writes past the real end. Generic bridges branded "VendorCo ProductCode"
    are a common carrier for exactly this.

    It could equally be a Windows formatting quirk. This tells the two apart.

    Method: write a distinctive pattern at several offsets, close the handle,
    reopen, read back, compare. If high offsets read back wrong while low ones
    are fine, the media is lying about its size and no amount of partitioning
    will fix it.

    SAFETY: writes only inside the UNALLOCATED tail of the disk, past every
    partition. It refuses any offset below the end of the last partition, so
    it cannot touch the NTFS volume or the ESP. Nothing you can see in Explorer
    is at risk.

.EXAMPLE
    .\scripts\probe-media.ps1
#>
param(
    [string]$SerialNumber = '3212271080340376147'
)

$ErrorActionPreference = 'Stop'

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "Raw disk access needs an elevated PowerShell."
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

# --- find the end of the last partition: our floor for writing ---
$parts = @(Get-CimInstance Win32_DiskPartition | Where-Object { $_.DiskIndex -eq $diskNumber })
$lastEnd = 0
foreach ($p in $parts) {
    $end = [int64]$p.StartingOffset + [int64]$p.Size
    if ($end -gt $lastEnd) { $lastEnd = $end }
}

# Keep clear of the backup GPT in the final sectors.
$ceiling = $diskSize - (16MB)
$floor = $lastEnd + 1MB

Write-Host ""
Write-Host "disk        : $diskNumber  $($target.Model)" -ForegroundColor Cyan
Write-Host ("reported    : {0:N0} bytes ({1:N2} GB)" -f $diskSize, ($diskSize/1GB))
Write-Host ("last part.  : ends at {0:N2} GB" -f ($lastEnd/1GB))
Write-Host ("safe window : {0:N2} GB .. {1:N2} GB (unallocated)" -f ($floor/1GB), ($ceiling/1GB))
Write-Host ""

if ($floor -ge $ceiling) {
    Write-Error "No unallocated space to test in. Cannot probe safely."
}

# Offsets spread across the unallocated tail.
$offsets = @()
foreach ($frac in 0.0, 0.25, 0.5, 0.75, 0.98) {
    $o = [int64]($floor + ($ceiling - $floor) * $frac)
    $o = $o - ($o % 512)   # sector-aligned
    $offsets += $o
}

$blockSize = 4096
$results = @()

foreach ($off in $offsets) {
    if ($off -lt $floor -or ($off + $blockSize) -gt $ceiling) {
        Write-Error "Computed offset $off is outside the safe window. Refusing."
    }

    # A pattern that encodes its own offset, so a device that wraps writes
    # around to a lower address is caught as well as one that drops them.
    $pattern = New-Object byte[] $blockSize
    $tag = [Text.Encoding]::ASCII.GetBytes("GLADOS-PROBE-$off-")
    for ($i = 0; $i -lt $blockSize; $i++) { $pattern[$i] = $tag[$i % $tag.Length] }

    $fs = [System.IO.File]::Open($path, 'Open', 'ReadWrite', 'ReadWrite')
    try {
        [void]$fs.Seek($off, 'Begin')
        $fs.Write($pattern, 0, $blockSize)
        $fs.Flush($true)
    } finally { $fs.Dispose() }

    # Reopen so nothing can be served from a cached page.
    $readBack = New-Object byte[] $blockSize
    $fs = [System.IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
    try {
        [void]$fs.Seek($off, 'Begin')
        $n = $fs.Read($readBack, 0, $blockSize)
    } finally { $fs.Dispose() }

    $match = $true
    for ($i = 0; $i -lt $blockSize; $i++) {
        if ($readBack[$i] -ne $pattern[$i]) { $match = $false; break }
    }

    $results += [PSCustomObject]@{
        OffsetGB = [math]::Round($off/1GB, 2)
        Match    = $match
        FirstBytes = [Text.Encoding]::ASCII.GetString($readBack[0..31]) -replace '[^\x20-\x7E]','.'
    }

    $colour = if ($match) { 'Green' } else { 'Red' }
    Write-Host ("{0,10:N2} GB : {1}" -f ($off/1GB), $(if ($match) { 'OK' } else { 'MISMATCH' })) -ForegroundColor $colour
}

Write-Host ""
$results | Format-Table -AutoSize

$bad = @($results | Where-Object { -not $_.Match })
Write-Host ""
if ($bad.Count -eq 0) {
    Write-Host "All offsets read back correctly." -ForegroundColor Green
    Write-Host "The media stores data at high offsets, so the format failure is a"
    Write-Host "Windows problem, not a hardware one."
} else {
    Write-Host "$($bad.Count) of $($results.Count) offsets did NOT read back." -ForegroundColor Red
    Write-Host ""
    Write-Host "This disk does not really hold what it claims. It reports" -ForegroundColor Red
    Write-Host ("{0:N2} GB but cannot store data at these addresses." -f ($diskSize/1GB)) -ForegroundColor Red
    Write-Host "No partitioning scheme fixes that -- the ESP has to live somewhere" -ForegroundColor Red
    Write-Host "the flash actually exists, or on different hardware." -ForegroundColor Red
}
