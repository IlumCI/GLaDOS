<#
.SYNOPSIS
    Prove the USB SSD is bootable before rebooting the real machine.

.DESCRIPTION
    Three checks, cheapest first:

      1. The ESP carries the correct GPT type GUID and is FAT32.
      2. BOOTX64.EFI exists on the removable-media path.
      3. QEMU boots the ACTUAL PHYSICAL DISK, attached as a USB mass storage
         device behind an xHCI controller, under OVMF.

    Check 3 is the one that matters. It does not use a synthetic FAT image --
    it reads the real partition table, the real ESP, and the real file, through
    a real UEFI implementation's USB stack. If glados's banner appears on the
    serial log, the on-disk layout is genuinely bootable, and any remaining
    failure on the GF63 is that laptop's firmware declining to enumerate this
    particular USB bridge rather than anything wrong with the disk.

    The physical disk is opened READ-ONLY. QEMU cannot modify it.

.EXAMPLE
    .\scripts\verify-esp.ps1 -EspDrive S:
#>
param(
    [string]$EspDrive = 'S:',
    [string]$SerialNumber = '3212271080340376147',
    [int]$BootSeconds = 25
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "Raw physical-disk access needs an elevated PowerShell."
}

$EspDrive = $EspDrive.TrimEnd('\')
$espLetter = $EspDrive.TrimEnd(':')

# --- locate the disk by serial ---
$target = Get-CimInstance Win32_DiskDrive | Where-Object {
    $_.SerialNumber -and $_.SerialNumber.Trim() -eq $SerialNumber
}
if (-not $target) { Write-Error "No disk with serial '$SerialNumber'." }
if ($target -is [array]) { Write-Error "Multiple disks share that serial. Refusing." }
$diskNumber = [int]$target.Index

Write-Host ""
Write-Host "=== 1. ESP partition type ===" -ForegroundColor Cyan

$dpFile = Join-Path $env:TEMP ("glados-verify-{0}.txt" -f [guid]::NewGuid().ToString('N'))
Set-Content -Path $dpFile -Encoding ASCII -Value @(
    "select disk $diskNumber",
    "select volume $espLetter",
    "detail partition",
    "list partition"
)
$detail = (& diskpart.exe /s $dpFile | Out-String)
if (Test-Path $dpFile) { Remove-Item -LiteralPath $dpFile -Force -ErrorAction SilentlyContinue }
Write-Host $detail

# An ESP is marked one of two ways depending on the partitioning scheme:
#   GPT: type GUID c12a7328-f81f-11d2-ba4b-00a0c93ec93b
#   MBR: partition type byte 0xEF
# This disk uses MBR, because a GPT backup header would land in the counterfeit
# region past 14.67 GB. Accept either.
$espGpt = $detail -match 'c12a7328-f81f-11d2-ba4b-00a0c93ec93b'
$espMbr = $detail -match '(?im)^\s*Type\s*:\s*EF\s*$'
if ($espGpt) {
    Write-Host "GPT EFI System Partition type GUID present." -ForegroundColor Green
} elseif ($espMbr) {
    Write-Host "MBR partition type 0xEF (EFI System) present." -ForegroundColor Green
} else {
    Write-Warning "Neither the GPT ESP GUID nor MBR type 0xEF was found."
    Write-Warning "Some firmware still boots any FAT partition on removable media, so this is not fatal -- check 3 is the real test."
}

$vol = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$EspDrive'" -ErrorAction SilentlyContinue
if ($vol) {
    Write-Host ("filesystem: {0}  size {1:N0} MB" -f $vol.FileSystem, ($vol.Size/1MB))
    if ($vol.FileSystem -notmatch 'FAT') {
        Write-Warning "$EspDrive is $($vol.FileSystem). UEFI only boots FAT12/16/32."
    }
}

Write-Host ""
Write-Host "=== 2. removable-media boot path ===" -ForegroundColor Cyan
$bootFile = Join-Path "$EspDrive\" 'EFI\BOOT\BOOTX64.EFI'
if (Test-Path $bootFile) {
    $f = Get-Item $bootFile
    Write-Host ("found {0}  ({1:N1} KB)" -f $bootFile, ($f.Length/1KB)) -ForegroundColor Green
    # A UEFI application is a PE32+ image with subsystem 10.
    $bytes = [System.IO.File]::ReadAllBytes($bootFile)
    $pe = [BitConverter]::ToInt32($bytes, 0x3C)
    $machine = [BitConverter]::ToUInt16($bytes, $pe + 4)
    $subsys = [BitConverter]::ToUInt16($bytes, $pe + 0x5C)
    Write-Host ("machine 0x{0:X4} (want 0x8664), subsystem {1} (want 10)" -f $machine, $subsys)
    if ($machine -ne 0x8664 -or $subsys -ne 10) {
        Write-Warning "That file is not an x86-64 UEFI application."
    }
} else {
    Write-Error "Missing $bootFile. Run deploy.ps1 first."
}

Write-Host ""
Write-Host "=== 3. boot the real disk in QEMU, as USB, read-only ===" -ForegroundColor Cyan

$qemu = $null
$cmd = Get-Command 'qemu-system-x86_64' -ErrorAction SilentlyContinue
if ($cmd) { $qemu = $cmd.Source } else {
    foreach ($c in @('C:\Program Files\qemu\qemu-system-x86_64.exe',
                     (Join-Path $env:USERPROFILE 'scoop\apps\qemu\current\qemu-system-x86_64.exe'))) {
        if (Test-Path $c) { $qemu = $c; break }
    }
}
if (-not $qemu) { Write-Error "qemu-system-x86_64 not found." }

$qd = Join-Path $root '.qemu'
New-Item -ItemType Directory -Force -Path $qd | Out-Null
$code = Join-Path $qd 'code.fd'
$vars = Join-Path $qd 'vars-verify.fd'
$share = Join-Path (Split-Path -Parent $qemu) 'share'
if (-not (Test-Path $code)) { Copy-Item (Join-Path $share 'edk2-x86_64-code.fd') $code }
# A throwaway NVRAM copy, so this test cannot disturb the normal run.ps1 vars.
Copy-Item (Join-Path $share 'edk2-i386-vars.fd') $vars -Force

$log = Join-Path $qd 'bare-metal-sim.log'
$phys = "\\.\PhysicalDrive$diskNumber"
Write-Host "attaching $phys read-only as a USB mass storage device" -ForegroundColor Yellow

$qargs = @(
    '-machine','q35','-m','512M',
    '-drive',"if=pflash,format=raw,unit=0,readonly=on,file=$code",
    '-drive',"if=pflash,format=raw,unit=1,file=$vars",
    # readonly=on is the safety here: QEMU physically cannot write to the disk.
    '-drive',"file=$phys,format=raw,readonly=on,if=none,id=stick",
    '-device','qemu-xhci,id=xhci',
    '-device','usb-storage,bus=xhci.0,drive=stick',
    '-serial',"file:$log",
    '-net','none','-no-reboot','-display','none'
)

$p = Start-Process $qemu -ArgumentList $qargs -PassThru -NoNewWindow
Start-Sleep -Seconds $BootSeconds
try { if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force } } catch {}
Start-Sleep -Seconds 1

Write-Host ""
if (Test-Path $log) {
    $text = (Get-Content $log -Raw) -replace "`e\[[0-9;=]*[A-Za-z]",""
    Write-Host $text
    if ($text -match 'entered efi_main') {
        Write-Host ""
        Write-Host "BOOTED. The on-disk layout is genuinely bootable." -ForegroundColor Green
        Write-Host "Any failure on the GF63 from here is that firmware not enumerating" -ForegroundColor Green
        Write-Host "this USB bridge, not a problem with the disk." -ForegroundColor Green
    } else {
        Write-Warning "glados did not start. The layout, not the laptop, is the problem."
        Write-Host "Look above for what OVMF did instead -- it usually says." -ForegroundColor Yellow
    }
} else {
    Write-Warning "No serial output at all. QEMU may have failed to open $phys."
}
