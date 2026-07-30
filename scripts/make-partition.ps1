<#
.SYNOPSIS
    Carve a GLaDOS partition out of C: on the internal NVMe.

.DESCRIPTION
    Shrinks the Windows volume and creates one new partition in the freed
    space, tagged with a GLaDOS-specific GPT type GUID.

    This is the highest-risk script in the repo. Everything before it operated
    on a removable disk that could be unplugged; this one touches the drive
    Windows boots from. It is built accordingly:

      * The disk is identified by SERIAL NUMBER, never by index.
      * It refuses unless that disk is the one holding C:.
      * It refuses unless C: has comfortable headroom left afterwards.
      * diskpart's own view must agree with WMI's before anything is written.
      * DRY RUN by default. Nothing happens without -Execute plus a typed
        confirmation of the serial.

    WHAT IT DOES NOT DO, deliberately:

      * It never deletes or moves a partition. The ESP and the recovery
        partition are not touched, named, or selected.
      * It never formats the new partition. An unformatted partition with an
        unrecognised type GUID is one Windows will not mount or write to, so
        the new space cannot be used by accident.
      * It only ever shrinks. NTFS shrink relocates file data out of the region
        it gives up; it does not discard it.

    On the type GUID: tagging the partition means GLaDOS finds it by identity
    rather than by guessing which space looks unused. That matters here because
    this disk is fully allocated -- the freed space lands between C: and the
    recovery partition, not at the end of the disk.

.EXAMPLE
    .\scripts\make-partition.ps1                 # show the plan, change nothing
    .\scripts\make-partition.ps1 -Execute
    .\scripts\make-partition.ps1 -SizeGB 8 -Execute
#>
param(
    [switch]$Execute,
    [int]$SizeGB = 16,
    # The internal Kingston, as surveyed. Override only if you know why.
    [string]$SerialNumber = '0000_0000_0000_0000_0026_B738_28E0_1435.',
    # Free space to leave on C: afterwards. Windows wants room for updates,
    # the page file and hibernation; running it dry is its own kind of damage.
    [int]$MinRemainingFreeGB = 40
)

$ErrorActionPreference = 'Stop'

# GLaDOS data partition type. In GUID text form:
#   b7e1f4a2-9c3d-4e58-a061-2f8d7c4b93e5
$GLADOS_TYPE = 'b7e1f4a2-9c3d-4e58-a061-2f8d7c4b93e5'

function Invoke-Diskpart {
    param([string[]]$Commands)
    $file = Join-Path $env:TEMP ("glados-dp-{0}.txt" -f [guid]::NewGuid().ToString('N'))
    Set-Content -Path $file -Value ($Commands -join "`r`n") -Encoding ASCII
    try { return (& diskpart.exe /s $file | Out-String) }
    finally { if (Test-Path $file) { Remove-Item -LiteralPath $file -Force -ErrorAction SilentlyContinue } }
}

# --- elevation ---
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "This must run from an elevated PowerShell."
}

if ($SizeGB -lt 4) {
    Write-Error "Below 4 GB the checkpoint history is too short for rollback to be a real safety net. Refusing."
}

# --- identify the disk by serial, and confirm it is the Windows disk ---
$target = Get-CimInstance Win32_DiskDrive | Where-Object {
    $_.SerialNumber -and $_.SerialNumber.Trim() -eq $SerialNumber.Trim()
}
if (-not $target) {
    Write-Host "Disks attached:" -ForegroundColor Yellow
    Get-CimInstance Win32_DiskDrive |
        Select-Object Index, Model, InterfaceType,
            @{n='SizeGB';e={[math]::Round($_.Size/1GB,2)}}, SerialNumber |
        Format-Table -AutoSize
    Write-Error "No disk with serial '$SerialNumber'."
}
if ($target -is [array]) { Write-Error "More than one disk reports that serial. Refusing to guess." }
$diskNumber = [int]$target.Index

# C: must live on this disk. Getting this backwards is the failure that matters.
$sysMap = Get-CimInstance Win32_LogicalDiskToPartition | Where-Object { $_.Dependent.DeviceID -eq 'C:' }
if (-not $sysMap) { Write-Error "Cannot determine which disk holds C:." }
if ($sysMap.Antecedent.DeviceID -notmatch "Disk #$diskNumber,") {
    Write-Error "C: is not on disk $diskNumber. Refusing -- this script only shrinks the Windows volume on its own disk."
}

# --- headroom check ---
$c = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'"
$freeGB = [math]::Round($c.FreeSpace/1GB, 2)
$afterGB = [math]::Round($freeGB - $SizeGB, 2)

Write-Host ""
Write-Host "target disk    : $diskNumber  $($target.Model)" -ForegroundColor Cyan
Write-Host "serial         : $($target.SerialNumber.Trim())"
Write-Host ("C:             : {0:N1} GB total, {1:N1} GB free" -f ($c.Size/1GB), $freeGB)
Write-Host ""
Write-Host "existing partitions (none of these are touched):" -ForegroundColor Yellow
Get-CimInstance Win32_DiskPartition | Where-Object { $_.DiskIndex -eq $diskNumber } |
    Select-Object Name, Type,
        @{n='SizeGB';e={[math]::Round($_.Size/1GB,2)}},
        @{n='OffsetGB';e={[math]::Round($_.StartingOffset/1GB,2)}} |
    Sort-Object OffsetGB | Format-Table -AutoSize

Write-Host "plan" -ForegroundColor Yellow
Write-Host "  shrink C: by $SizeGB GB   -> C: free afterwards: $afterGB GB"
Write-Host "  create a $SizeGB GB partition in the freed space"
Write-Host "  tag it $GLADOS_TYPE"
Write-Host "  leave it UNFORMATTED (Windows will not mount it)"
Write-Host ""

if ($afterGB -lt $MinRemainingFreeGB) {
    Write-Error ("Only {0:N1} GB would remain free on C:, below the {1} GB floor. Free some space or use -SizeGB." -f $afterGB, $MinRemainingFreeGB)
}

# --- diskpart must agree this is the same disk ---
Write-Host "cross-checking with diskpart ..." -ForegroundColor Cyan
$detail = Invoke-Diskpart @("select disk $diskNumber", "detail disk", "list partition")
Write-Host $detail

$modelToken = ($target.Model -split '\s+')[0]
if ($detail -notmatch [regex]::Escape($modelToken)) {
    Write-Error "diskpart's disk $diskNumber does not mention '$modelToken'. The two providers disagree. Refusing."
}
if ($detail -notmatch '(?m)^\s*Volume\s+\d+\s+C\s') {
    Write-Error "diskpart does not show volume C: on disk $diskNumber. Refusing."
}
if ($detail -notmatch 'Boot Disk\s*:\s*Yes') {
    Write-Warning "diskpart does not report this as the boot disk. Check the output above carefully."
}
Write-Host "diskpart agrees: $modelToken, holds C:." -ForegroundColor Green

# --- how much can NTFS actually give up? ---
#
# Free space is not the limit. NTFS shrink is bounded by the position of
# immovable files -- the page file, the hibernation file, the MFT -- so a
# volume with 78 GB free can refuse to give up 16 GB if one of those sits near
# the end. Ask before trying, rather than discovering it from an error.
Write-Host "asking NTFS how far C: can shrink ..." -ForegroundColor Cyan
$qm = Invoke-Diskpart @("select disk $diskNumber", "select volume C", "shrink querymax")
$maxMB = $null
if ($qm -match 'reclaimable bytes is:\s*([\d,]+)\s*MB') {
    $maxMB = [int64]($Matches[1] -replace ',', '')
}
if ($null -eq $maxMB) {
    Write-Host $qm
    Write-Warning "Could not read the shrink limit from diskpart. Proceeding is not safe blind."
    if ($Execute) { Write-Error "Refusing to run without knowing the shrink limit." }
} else {
    Write-Host ("  NTFS will give up at most {0:N0} MB ({1:N1} GB)" -f $maxMB, ($maxMB/1024))
}

$sizeMB = $SizeGB * 1024

if ($null -ne $maxMB -and $sizeMB -gt $maxMB) {
    Write-Host ""
    Write-Host "CANNOT SHRINK BY $SizeGB GB." -ForegroundColor Red
    Write-Host ("The limit is {0:N1} GB, set by immovable files near the end of the volume," -f ($maxMB/1024)) -ForegroundColor Red
    Write-Host "not by free space. Options, in order of effect:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  1. Remove the hibernation file (6.3 GB here), reversible with /h on:"
    Write-Host "       powercfg /h off"
    Write-Host ""
    Write-Host "  2. Move the page file off C: temporarily, reboot, shrink, restore it."
    Write-Host "       System Properties > Advanced > Performance > Advanced > Virtual memory"
    Write-Host ""
    Write-Host "  3. Consolidate free space:"
    Write-Host "       defrag C: /X"
    Write-Host ""
    Write-Host ("  4. Or take what is available now:  .\scripts\make-partition.ps1 -SizeGB {0}" -f [math]::Floor($maxMB/1024)) -ForegroundColor Green
    Write-Host ""
    if ($Execute) { Write-Error "Refusing: requested $SizeGB GB, limit is $([math]::Round($maxMB/1024,1)) GB." }
    return
}

$script = @(
    "select disk $diskNumber",
    "select volume C",
    "shrink desired=$sizeMB minimum=$sizeMB",
    "create partition primary size=$sizeMB",
    "set id=$GLADOS_TYPE",
    "list partition"
)

Write-Host ""
Write-Host "diskpart script to be run:" -ForegroundColor Yellow
$script | ForEach-Object { Write-Host "  $_" }
Write-Host ""
Write-Host "note: no 'format', no 'assign', no 'delete'." -ForegroundColor Green
Write-Host ""

if (-not $Execute) {
    Write-Host "DRY RUN. Nothing was changed. Re-run with -Execute to apply." -ForegroundColor Yellow
    return
}

Write-Host "Back up anything irreplaceable before continuing." -ForegroundColor Red
Write-Host "Type the disk serial to confirm:" -ForegroundColor Yellow
$typed = Read-Host "serial"
if ($typed.Trim() -ne $target.SerialNumber.Trim()) {
    Write-Error "Serial did not match. Nothing was changed."
}

Write-Host "running ..." -ForegroundColor Cyan
$result = Invoke-Diskpart $script
Write-Host $result

Write-Host ""
Write-Host "=== resulting layout ===" -ForegroundColor Cyan
$after = @(Get-CimInstance Win32_DiskPartition | Where-Object { $_.DiskIndex -eq $diskNumber })
$after | Select-Object Name, Type,
        @{n='SizeGB';e={[math]::Round($_.Size/1GB,2)}},
        @{n='OffsetGB';e={[math]::Round($_.StartingOffset/1GB,2)}} |
    Sort-Object OffsetGB | Format-Table -AutoSize

$c2 = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'"
Write-Host ("C: now {0:N1} GB total, {1:N1} GB free" -f ($c2.Size/1GB), ($c2.FreeSpace/1GB))
Write-Host ""

# --- verify by outcome, not by narration ---
#
# An earlier version of this script decided success by grepping diskpart's
# output for one specific phrase. diskpart emitted a different one -- "Virtual
# Disk Service error" -- so the check sailed past a failed shrink and reported
# success while printing a layout that plainly showed no new partition.
# Pattern-matching a tool's prose is not verification. Count the partitions.
$expectedGB = $SizeGB
$new = $after | Where-Object {
    [math]::Abs(($_.Size/1GB) - $expectedGB) -lt 0.5 -and $_.StartingOffset -gt 0
}

$errored = ($result -match 'Virtual Disk Service error') -or
           ($result -match 'DiskPart has encountered an error') -or
           ($result -match '(?m)^\s*Error')

if (-not $new) {
    Write-Host "FAILED: no new $SizeGB GB partition exists." -ForegroundColor Red
    if ($errored) { Write-Host "diskpart reported an error; see its output above." -ForegroundColor Red }
    Write-Host "C: is unchanged and nothing was created or destroyed." -ForegroundColor Yellow
    Write-Host "If the shrink was refused, try: powercfg /h off   (then re-run)" -ForegroundColor Yellow
    exit 1
}

if ($errored) {
    Write-Warning "A partition of the right size exists, but diskpart also reported an error."
    Write-Warning "Check the type GUID before trusting it."
}

Write-Host ("created: {0}  {1:N2} GB at {2:N2} GB" -f `
    $new[0].Name, ($new[0].Size/1GB), ($new[0].StartingOffset/1GB)) -ForegroundColor Green
Write-Host ""
Write-Host "done. GLaDOS will find this partition by its type GUID." -ForegroundColor Green
Write-Host "Boot it and run 'store init' to format the checkpoint store."
