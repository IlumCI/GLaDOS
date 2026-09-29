#!/bin/bash
# Mine on the GPU and the CPU at once, each on what pays most for it.
#
#   KRYPTEX_USER=you@example.com tools/rig.sh          # account mode, paid in BTC
#   KRYPTEX_USER=<QTC wallet> XMR_USER=<XMR wallet> tools/rig.sh
#
# **Both devices, always, and judged by the sum.** Measured together on the
# i7-12650H + RTX 3050 Laptop, 2026-09-29, against WhatToMine's live figures:
#
#     GPU  Quantus (QPoW)   78 MH/s at 35 W, 86 C   $0.66/day
#     CPU  RandomX / XMR    2.0 kH/s on 14 threads  $0.08/day
#
# which is 2.6x the equihash-192,7 + yespowerR16 pair it replaced ($0.28). Every
# GPU coin WhatToMine lists for a 3050 and every CPU coin hashrate.no lists for a
# 12700 was ranked first; nothing else came within a third of either.
#
# **Neither coin is on zpool**, so this rig mines at Kryptex directly and not
# through `pool/`. The pool and the mining ISO stay on the yespower family, which
# is what a machine with no CUDA can run; this is the operator's own laptop.
#
# The miners are third-party binaries, fetched at a pinned version and refused
# unless their archive matches the digest recorded here -- the digests of the
# exact archives the figures above were measured with. xmrig's also matches its
# own published SHA256SUMS.
#
# Two CPU threads are left free: krig-miner feeds the GPU from the CPU, and at 16
# RandomX threads the GPU's rate is the thing that drops.
set -euo pipefail

: "${KRYPTEX_USER:?set KRYPTEX_USER to a Kryptex account email, or to a QTC wallet}"
XMR_USER="${XMR_USER:-$KRYPTEX_USER}"
WORKER="${WORKER:-$(hostname)}"
CPU_THREADS="${CPU_THREADS:-$(( $(nproc) - 2 ))}"
DIR="${RIG_DIR:-$HOME/.local/share/glados-rig}"

KRIG_URL=https://github.com/kryptex/krig-miner/releases/download/v1.5.2/krig-miner-1.5.2-linux-x64.tar.gz
KRIG_SHA=53863c153c7fddf711482de21414392f856ed3472692757887e65b1c7583005e
XMRIG_URL=https://github.com/xmrig/xmrig/releases/download/v6.26.0/xmrig-6.26.0-linux-static-x64.tar.gz
XMRIG_SHA=fc6f8ae5f64e4f17481f7e3be29a1c56949f216a998414188003eae1db20c9e5

fetch() { # url sha dest-dir
    local tgz="$3.tgz"
    if [ ! -d "$3" ]; then
        mkdir -p "$DIR"
        curl -fsSL "$1" -o "$tgz"
        echo "$2  $tgz" | sha256sum -c --quiet - || { echo "rig: $tgz does not match its pinned digest; refusing" >&2; rm -f "$tgz"; exit 1; }
        mkdir -p "$3" && tar xzf "$tgz" -C "$3" && rm -f "$tgz"
    fi
}
fetch "$KRIG_URL" "$KRIG_SHA" "$DIR/krig"
fetch "$XMRIG_URL" "$XMRIG_SHA" "$DIR/xmrig"
KRIG=$(find "$DIR/krig" -name krig-miner -type f | head -1)
XMRIG=$(find "$DIR/xmrig" -name xmrig -type f | head -1)

# Kryptex takes `wallet/worker`, or `email/worker` in account mode. Quantus is
# TLS-only there: plain TCP is refused by the miner itself.
"$KRIG" --no-tui --coin quantus --url stratum+ssl://qtc-eu.kryptex.network:8049 \
    --user "$KRYPTEX_USER/$WORKER" --log-file "$DIR/gpu.log" &
GPU=$!
"$XMRIG" --no-color -o xmr-eu.kryptex.network:7029 -u "$XMR_USER/$WORKER" -p x \
    -t "$CPU_THREADS" --log-file="$DIR/cpu.log" &
CPU=$!

trap 'kill $GPU $CPU 2>/dev/null; wait' INT TERM EXIT
echo "rig: GPU pid $GPU (Quantus), CPU pid $CPU (RandomX, $CPU_THREADS threads); logs in $DIR"
wait -n
echo "rig: a miner exited; stopping the other" >&2
