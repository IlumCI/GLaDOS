<#
.SYNOPSIS
    Fetch a llama2.c checkpoint into esp\GLADOS so deploy.ps1 can carry it.

.DESCRIPTION
    This is the SECONDARY path. The model GLaDOS actually runs is SmolLM2-135M,
    produced by tools\convert.py from a Hugging Face safetensors checkpoint into
    a quantised GLADOSM2 file. Use this script for karpathy's tinyllamas instead
    -- they are small, they load fast, and they are useful for exercising the
    loader and the sampler without waiting on 129 MB.

    The loader identifies a checkpoint by content, not by filename, so both
    formats live at the same path and either can be dropped in.

    The weights are not in the repository -- they are someone else's artifact,
    they are large, and git is the wrong place for either. This script puts them
    where the loader expects them: esp\GLADOS\model.bin and tokenizer.bin, read
    off the boot volume by uefi::read_file before ExitBootServices.

    stories260K is the default because it is the only one of karpathy's
    checkpoints that ships its own tokenizer. The larger ones share the 32000
    token llama tokenizer.bin, which lives in the llama2.c repository rather
    than on the Hub, so -Model stories15M also needs -TokenizerUrl.

    Sizes are checked after download. A truncated model would otherwise be
    caught much later, by Model::from_bytes reporting a weight count that does
    not match the header.

.EXAMPLE
    .\scripts\fetch-model.ps1
    .\scripts\fetch-model.ps1 -Model stories15M -TokenizerUrl https://.../tokenizer.bin
#>
param(
    [ValidateSet('stories260K', 'stories15M', 'stories42M', 'stories110M')]
    [string]$Model = 'stories260K',
    [string]$TokenizerUrl,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$dir  = Join-Path $root 'esp\GLADOS'
New-Item -ItemType Directory -Force -Path $dir | Out-Null

# Both formats share model.bin, so downloading a 1 MB tinyllama on top of a
# converted 129 MB SmolLM2 is an easy accident with a slow recovery -- the
# conversion needs the safetensors checkpoint fetched again. Refuse unless the
# overwrite was asked for.
$existing = Join-Path $root 'esp\GLADOS\model.bin'
if ((Test-Path $existing) -and -not $Force) {
    $fs = [System.IO.File]::OpenRead($existing)
    $magic = New-Object byte[] 8
    [void]$fs.Read($magic, 0, 8)
    $fs.Close()
    if ([Text.Encoding]::ASCII.GetString($magic) -eq 'GLADOSM2') {
        Write-Error ("model.bin is a converted GLADOSM2 checkpoint ({0:N0} B). " -f (Get-Item $existing).Length +
                     "Pass -Force to replace it with a llama2.c checkpoint.")
    }
}

$repo = 'https://huggingface.co/karpathy/tinyllamas/resolve/main'

if ($Model -eq 'stories260K') {
    $modelUrl = "$repo/stories260K/stories260K.bin"
    if (-not $TokenizerUrl) { $TokenizerUrl = "$repo/stories260K/tok512.bin" }
} else {
    $modelUrl = "$repo/$Model.bin"
    if (-not $TokenizerUrl) {
        Write-Error "$Model uses the 32000-token llama tokenizer, which is not in that repo. Pass -TokenizerUrl."
    }
}

$ProgressPreference = 'SilentlyContinue'

function Fetch($url, $path, $label) {
    if ((Test-Path $path) -and -not $Force) {
        Write-Host ("  {0,-14} present ({1:N0} B) -- use -Force to refetch" -f $label, (Get-Item $path).Length)
        return
    }
    Write-Host "  $label <- $url"
    Invoke-WebRequest $url -OutFile $path
    Write-Host ("  {0,-14} {1:N0} B" -f $label, (Get-Item $path).Length) -ForegroundColor Green
}

Fetch $modelUrl     (Join-Path $dir 'model.bin')     'model.bin'
Fetch $TokenizerUrl (Join-Path $dir 'tokenizer.bin') 'tokenizer.bin'

# Read back the header the loader will read, and confirm the file is the size
# that header implies. This is the same arithmetic Model::from_bytes does.
$m = [System.IO.File]::ReadAllBytes((Join-Path $dir 'model.bin'))
if ($m.Length -lt 28) { Write-Error "model.bin is too short to hold a header." }
$h = 0..6 | ForEach-Object { [BitConverter]::ToInt32($m, $_ * 4) }
$dim, $hidden, $layers, $heads, $kvHeads, $rawVocab, $seq = $h

$vocab  = [Math]::Abs($rawVocab)
$shared = $rawVocab -gt 0
$headSz = $dim / $heads
$kvDim  = $dim * $kvHeads / $heads

$params = $vocab*$dim + $layers*$dim + $layers*$dim*$dim + 2*$layers*$dim*$kvDim +
          $layers*$dim*$dim + $layers*$dim + 3*$layers*$hidden*$dim + $dim
if (-not $shared) { $params += $vocab * $dim }
# The legacy export also carries precomputed RoPE tables the runtime skips.
$expected = 28 + 4 * ($params + $seq * $headSz)

Write-Host ''
Write-Host ("  dim {0}  hidden {1}  layers {2}  heads {3}/{4} kv  vocab {5}  seq {6}" -f `
    $dim, $hidden, $layers, $heads, $kvHeads, $vocab, $seq)
Write-Host ("  classifier {0}" -f $(if ($shared) { 'tied to embedding' } else { 'separate' }))
Write-Host ("  {0:N0} params; expecting {1:N0} B, file is {2:N0} B" -f $params, $expected, $m.Length)

if ($expected -ne $m.Length) {
    Write-Error "size mismatch -- the download is truncated or this is not a legacy llama2.c checkpoint."
}
Write-Host '  header and size agree.' -ForegroundColor Green
Write-Host ''
Write-Host 'now: .\scripts\deploy.ps1 -EspDrive S:'
