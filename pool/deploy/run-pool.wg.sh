#!/bin/sh
# The pool without systemd, for a machine where lingering cannot be enabled.
#
# `systemctl --user` is the better answer and this is not a replacement for it.
# It exists because `loginctl enable-linger` is governed by polkit and some
# systems will not grant it without an administrator -- and a user service
# without lingering stops at logout, which presents as a pool that works
# perfectly while you watch it and is gone in the morning.
#
# Deliberately POSIX `sh` and not bash: this has to run under whatever `cron`
# hands it, which on Debian-family systems is `/bin/sh` and is dash.
#
# Install:
#     mkdir -p ~/.local/bin ~/.local/state/glados-pool
#     install -m 0755 glados-pool  ~/.local/bin/
#     install -m 0755 run-pool.sh  ~/.local/bin/
#     crontab -e
#         @reboot /bin/sh $HOME/.local/bin/run-pool.sh
#     # and start it now without waiting for a reboot:
#     nohup ~/.local/bin/run-pool.sh >/dev/null 2>&1 &

# `-u` and deliberately not `-e`. A supervisor should be hard to kill: `-e`
# turns any unexpected non-zero -- a `wc` on a log that vanished, a `mv` across
# a full disk -- into a pool that silently stops, which is the one behaviour
# this script exists to prevent. `-u` stays, because an unset variable here
# would expand to nothing and quietly point a path at the wrong place.
set -u

# --- why this and not a systemd user service, on this host ---
#
# `systemctl --user` is the better answer where it is available, and here
# it is not. A user manager stops with the last session unless lingering is
# enabled, `loginctl enable-linger` is refused to this user by polkit, and
# nobody is granting root. There is no cron either, so the usual fallback
# is out too.
#
# What makes this work anyway is one logind setting, checked rather than
# assumed: `KillUserProcesses=false`. logind does not reap a user's
# processes when their last session ends, so a detached process outlives
# logout even with `Linger=no`. That is the whole mechanism. If somebody
# ever sets `KillUserProcesses=yes` this stops surviving logout and there
# is no warning -- it just will not be there.
#
# What it does not survive is a reboot: nothing unprivileged starts at
# boot without linger or cron. `~/.profile` starts it on the next login
# instead, guarded by the pidfile below so logging in twice cannot make
# two pools.

BIN="${HOME}/.local/bin/glados-pool"
STATE="${HOME}/.local/state/glados-pool"
LOG="${STATE}/pool.log"
PIDFILE="${STATE}/run-pool.pid"

# The coins, and the only line here that is a decision rather than plumbing.
# `bits` is how many leading zero bits a share must have -- a difficulty said
# in a way that has no float in it. Twelve suits a laptop on yespower; sha256d
# and blake2s are thousands of times cheaper to check and can afford far more.
# `--cpu-percent` is the total validation budget, as a share of *one* core,
# and on a borrowed machine it is the setting that matters most. The
# per-connection rate limit does not sum -- 256 connections were never bounded
# in aggregate -- so this is what decides how much of somebody else's box a
# flood can take. 25% of one core is conservative on a four-thread 2012 i3 that
# is also serving media; raise it on a machine with headroom, or 0 for no cap.
#
# `--window` is the PPLNS payout window in work; a share credits 2^bits, so the
# default 2^32 is one difficulty-1 share's worth, which is a starting point and
# not a setting. The comment above this line said "left default here" while the
# line below passed 268435456, and both halves were wrong: the value is not the
# default, and as a bare integer it looked perfectly large while being **2e-7 of
# one Feathercoin block**. That is not a window that pays smoothly or a window
# that pays soon; it pays the last few shares and nothing else, which is the
# failure a number with no unit produces. 2^50 is about 0.85 of a Feathercoin
# block, taking one block as the dial's midpoint -- Rosenfeld gives reward
# variance as pB^2/N and mean time to payment as pN/2, so their product is fixed
# and there is no optimum, only an end to pick.
#
# **Against Bitcoin no value would have been right.** `pool.rs` records it at
# `window_work`: a `u64` reaches one block only while difficulty is under
# 4.295e9, and Bitcoin misses that by 21,432x, so btc's PPLNS degenerates to
# paying the most recent shares whatever is passed here. Sized for the coin the
# dial can actually serve, and the payout line says what it is worth on each.
# **This script is behind `run-pool.sh` in two ways that are named here rather
# than fixed, because the WireGuard path is not the one being deployed and a
# blind port of the other script's machinery is how a supervisor acquires a bug
# nobody drove.** First, its log rotation is still the version called once per
# loop iteration -- which is once per pool *exit*, so the healthier the pool the
# less often its log is checked and one that never crashes never rotates at all.
# `run-pool.sh` splits that into a rotate-between-runs and a `watch_log` beside
# the live pool, and explains why `mv` is the wrong verb on an open descriptor.
# Second, it has no roster refresher, so the `--roster` below is a file nothing
# here keeps current. Take both from `run-pool.sh` when this path matters.
set -- \
    --listen 10.66.66.1:3334 \
    --ledger "${STATE}/ledger.json" \
    --roster "${STATE}/roster.json" \
    --roster-url https://glados.aperture.institute/pool/ \
    --cpu-percent 25 \
    --share-seconds 45 \
    --max-connections 900 \
    --window 1125899906842624 \
    btc:sha256d:24:bitcoin \
    ftc:neoscrypt:20:feathercoin

# A log this can never let grow without bound. On a borrowed machine, filling
# somebody's disk is a worse way to fail than not running at all -- and cron
# has no journal behind it to rotate this for us.
MAX_LOG_BYTES=4194304

mkdir -p "${STATE}"

# One copy. `@reboot` plus a manual start is the ordinary way to end up with
# two, and two pools on one port means the second dies on bind while the first
# keeps going -- which looks like it worked.
if [ -f "${PIDFILE}" ] && kill -0 "$(cat "${PIDFILE}" 2>/dev/null)" 2>/dev/null; then
    echo "already running as PID $(cat "${PIDFILE}")" >&2
    exit 0
fi
echo $$ > "${PIDFILE}"

# Removed however this exits, so a crash does not leave a pidfile that blocks
# the next start.
trap 'rm -f "${PIDFILE}"' EXIT
# **INT and TERM need their own trap that exits, and this was wrong first.**
# A POSIX trap handler that does not call `exit` *returns to where it was*, so
# one handler shared with EXIT deleted the pidfile and then carried on looping:
# the script could not be stopped with `kill`, only `kill -9`, and having
# removed its own pidfile it would no longer have refused a second instance.
# Found by sending it a TERM rather than by reading it.
trap 'rm -f "${PIDFILE}"; exit 0' INT TERM

rotate() {
    if [ -f "${LOG}" ]; then
        size=$(wc -c < "${LOG}" 2>/dev/null || echo 0)
        if [ "${size}" -gt "${MAX_LOG_BYTES}" ]; then
            mv -f "${LOG}" "${LOG}.old"
        fi
    fi
}

# How long a run has to last before it counts as having worked. Under this and
# the pool is failing at startup -- a taken port, a missing file -- and the
# backoff should keep climbing. Over it and whatever happened was a one-off.
HEALTHY_AFTER=60

backoff=1
while : ; do
    rotate
    echo "[run-pool] starting $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG}"
    started=$(date +%s)
    "${BIN}" "$@" >> "${LOG}" 2>&1 || true
    ran=$(( $(date +%s) - started ))

    # **The backoff resets after a run that worked**, and it did not at first.
    # Without this the delay only ever climbs, so a pool that came up, served
    # for a day and then restarted once would wait a full minute to come back
    # -- carrying a penalty earned by an unrelated failure hours earlier. Seen
    # in a test log climbing 1, 2, 4, 8, 16, 32 with nothing to justify it.
    if [ "${ran}" -ge "${HEALTHY_AFTER}" ]; then
        backoff=1
    fi
    echo "[run-pool] ran ${ran}s, restarting in ${backoff}s" >> "${LOG}"
    sleep "${backoff}"
    # Doubling to a minute. A tight restart loop against something that fails
    # immediately -- a port already taken, a missing binary -- is a busy core
    # and a log growing as fast as the disk allows, which is the one failure
    # mode that costs the machine's owner something.
    backoff=$((backoff * 2))
    # An `if` rather than `[ ... ] && ...`. Whether a false AND-OR list is fatal
    # under `set -e` differs between shells, and this has to survive whatever
    # `cron` hands it; an `if` is the same everywhere.
    if [ "${backoff}" -gt 60 ]; then
        backoff=60
    fi
done
