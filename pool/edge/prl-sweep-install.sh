#!/bin/sh
# Install the PRL sweep for this user: the wallet as a service that stays in
# sync, and the sweep on an hourly timer. Run once; safe to run again.
#
#   sh pool/edge/prl-sweep-install.sh [--live]     the timer only reports what it would send, unless --live
#
# Writes ~/.pearl-treasury/oyster.conf and sweep.json (mode 600) with a fresh
# RPC password, and three user units. Nothing is sent by installing.
set -eu
APP="$HOME/.pearl-treasury"
BIN="$HOME/.local/share/pearl-wallet/bin"
HERE="$(cd "$(dirname "$0")" && pwd)"
UNITS="$HOME/.config/systemd/user"
PORT=44200
LIVE=""; [ "${1:-}" = "--live" ] && LIVE=" --live"

[ -x "$BIN/oyster" ] || { echo "no oyster at $BIN/oyster" >&2; exit 1; }
[ -f "$APP/mainnet/wallet.db" ] || { echo "no wallet in $APP" >&2; exit 1; }
umask 077

if [ ! -f "$APP/oyster.conf" ]; then
  PASS="$(head -c 32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 40)"
  printf '[Application Options]\nusespv=1\nrpclisten=127.0.0.1:%s\nusername=sweep\npassword=%s\n' "$PORT" "$PASS" > "$APP/oyster.conf"
else
  PASS="$(sed -n 's/^password=//p' "$APP/oyster.conf")"
fi
if [ ! -f "$APP/sweep.json" ]; then
  # The treasury's EVM key, and the bridge's deposit address for it, pinned now.
  EVM="0xfaE492EA62AA2250E0Ab2d1a6331E3c991443Cc5"
  DEP="prl1pchxq5smsmxsl3l59n8knnrgj386vd7qs8dnw468mwln9ezwjuttqxu67sd"
  printf '{\n  "treasury_evm": "%s",\n  "deposit_pinned": "%s",\n  "threshold_prl": 50,\n  "fee_reserve_prl": 0.05,\n  "rpc_port": %s,\n  "rpc_user": "sweep",\n  "rpc_pass": "%s"\n}\n' "$EVM" "$DEP" "$PORT" "$PASS" > "$APP/sweep.json"
fi
chmod 600 "$APP/oyster.conf" "$APP/sweep.json"

mkdir -p "$UNITS"
cat > "$UNITS/glados-prl-wallet.service" <<EOF
[Unit]
Description=GLaDOS pool: the PRL treasury wallet (oyster, SPV), kept in sync

[Service]
ExecStart=$BIN/oyster --appdata=$APP -C $APP/oyster.conf
Restart=on-failure
RestartSec=30

[Install]
WantedBy=default.target
EOF
cat > "$UNITS/glados-prl-sweep.service" <<EOF
[Unit]
Description=GLaDOS pool: sweep PRL into PearlBridge
After=glados-prl-wallet.service

[Service]
Type=oneshot
ExecStart=/usr/bin/env python3 $HERE/prl-sweep.py$LIVE
EOF
cat > "$UNITS/glados-prl-sweep.timer" <<EOF
[Unit]
Description=GLaDOS pool: sweep PRL into PearlBridge, hourly

[Timer]
OnBootSec=10min
OnUnitActiveSec=1h
Persistent=true

[Install]
WantedBy=timers.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now glados-prl-wallet.service glados-prl-sweep.timer
echo "installed${LIVE:+ LIVE: the sweep sends}${LIVE:- (dry: the sweep only reports; run again with --live to send)}. Logs: journalctl --user -u glados-prl-sweep -u glados-prl-wallet"
echo "the wallet's first sync takes ~40 min; sweeps wait for it."
