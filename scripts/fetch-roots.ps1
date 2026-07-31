<#
.SYNOPSIS
    Build esp\GLADOS\roots.der from the host's root certificate store.

.DESCRIPTION
    GLaDOS loads its trust anchors from a file rather than compiling them in,
    so that a distrusted root can be removed without rebuilding, and so that
    'trust' can list exactly what is trusted with nothing hidden in the binary.

    The file is a plain concatenation of DER certificates. Each begins with a
    SEQUENCE whose length says how long it is, so the boundaries are
    self-describing and no container format is needed.

    Roots come from the current user's Trusted Root store, which is the same
    set the host's browsers use. That makes the decision inheritable rather
    than invented here -- but note it also inherits anything the host trusts,
    including certificates added by corporate management or by interception
    software. Pass -List first if that matters.

    Only self-signed CA certificates are exported, because that is what a
    trust anchor is. Anything expired is skipped: it cannot validate a chain
    and would only pad the file.

.EXAMPLE
    .\scripts\fetch-roots.ps1
    .\scripts\fetch-roots.ps1 -List
    .\scripts\fetch-roots.ps1 -Filter 'ISRG|DigiCert|Google'
#>
param(
    [switch]$List,
    [string]$Filter,
    [string]$Store = 'Cert:\CurrentUser\Root'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$dir = Join-Path $root 'esp\GLADOS'
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$out = Join-Path $dir 'roots.der'

$now = Get-Date
$certs = Get-ChildItem $Store | Where-Object {
    $_.NotAfter -gt $now -and $_.NotBefore -le $now -and $_.Subject -eq $_.Issuer
}

if ($Filter) {
    $certs = $certs | Where-Object { $_.Subject -match $Filter }
}

if ($List) {
    $certs | ForEach-Object {
        $cn = $_.Subject
        if ($cn -match 'CN=([^,]+)') { $cn = $Matches[1] }
        "{0,-52} expires {1:yyyy-MM-dd}" -f $cn.Trim(), $_.NotAfter
    } | Sort-Object
    Write-Host ""
    Write-Host ("{0} usable root(s) in {1}" -f $certs.Count, $Store)
    return
}

$stream = [System.IO.File]::Create($out)
$n = 0
foreach ($c in $certs) {
    # RawData is already DER; the store holds nothing else.
    $bytes = $c.RawData
    $stream.Write($bytes, 0, $bytes.Length)
    $n++
}
$stream.Close()

Write-Host ("  wrote {0} root(s), {1:N0} B -> {2}" -f $n, (Get-Item $out).Length, $out) -ForegroundColor Green
Write-Host ""
Write-Host "GLaDOS reads this at boot. 'trust' lists what it accepted;"
Write-Host "a certificate it cannot parse is skipped rather than fatal."
