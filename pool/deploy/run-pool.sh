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
# default 2^32 is one difficulty-1 share's worth -- a starting point, and the
# reason `DEFAULT_WINDOW_WORK` says so. Leaving it default here was the same
# rounding error `run-pool.wg.sh` shipped explicitly: against Bitzeny it is
# 5e-5 of one block, so the window paid the last few shares and nothing else.
# 2^46 is about 0.80 of a Bitzeny block. See the longer note in the WireGuard
# script for why one block is the dial's midpoint and why there is no optimum.
#
# Bitcoin testnet is the other coin on this listener, and its difficulty swings
# by orders of magnitude under the 20-minute minimum-difficulty rule, so one
# window cannot be a considered choice for both. It is chosen for bitzeny, the
# coin this pool is actually pointed at, and the payout line prints what it is
# worth on each.
set -- \
    --listen 0.0.0.0:3334 \
    --ledger "${STATE}/ledger.json" \
    --cpu-percent 25 \
    --window 70368744177664 \
    bitzeny:yespower-10-2048-8:12 \
    testnet:sha256d:24

# A log this can never let grow without bound. On a borrowed machine, filling
# somebody's disk is a worse way to fail than not running at all -- and cron
# has no journal behind it to rotate this for us.
MAX_LOG_BYTES=4194304

# **Descriptors, raised before the pool starts rather than discovered during an
# event.** Every connection is one, and the login default on the host this was
# deployed to is 1024 -- so the connection ceiling cannot usefully go past about
# a thousand however `--max-connections` is set. The hard limit there is 524288,
# and raising the soft limit up to the hard one needs no privilege at all, which
# is the one resource question on this box that root was never the answer to.
#
# `|| true` because a shell that refuses `ulimit` is a shell where the pool
# should still start with whatever it has; the daemon reads the real limit and
# clamps its own ceiling to it, so the failure is a smaller pool rather than a
# broken one.
ulimit -n 65536 2>/dev/null || ulimit -n unlimited 2>/dev/null || true

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

# **Rotation had to stop happening only at restart, and a soak is what showed
# it.** This was called once per loop iteration, which is once per pool exit --
# so the healthier the pool, the less often its log was checked, and a pool
# that simply never crashes never rotates at all. Measured on the deployed
# instance: about a kilobyte a minute with two miners on it, so 1.4 MB a day
# against a 4 MiB cap. Three days to the limit and then nothing, forever, on
# somebody else's disk.
#
# **`mv` is also the wrong verb while the pool is running.** The pool's stdout
# is an open descriptor onto this file; renaming it moves the name and not the
# descriptor, so the pool goes on writing into `pool.log.old` and `pool.log`
# stays empty at zero bytes -- which reads as a pool that stopped logging. It
# is correct only in the loop, between runs, where no descriptor is open.
#
# So there are two of them. `rotate_between_runs` keeps the rename, which is
# atomic and loses nothing. `watch_log` runs beside a live pool and copies then
# truncates: the redirection is `>>`, which is `O_APPEND`, so every subsequent
# write goes to the new end of a file that is now empty. Writes that land
# between the copy and the truncate are lost, which is a line or two of a log
# and is the price of not having a signal handler in the daemon to reopen it.
rotate_between_runs() {
    if [ -f "${LOG}" ]; then
        size=$(wc -c < "${LOG}" 2>/dev/null || echo 0)
        if [ "${size}" -gt "${MAX_LOG_BYTES}" ]; then
            mv -f "${LOG}" "${LOG}.old"
        fi
    fi
}

# Checked every minute rather than every byte: the cost of being a minute late
# is at most a minute of log past the cap, and the cost of checking constantly
# is a `wc` on a growing file forever.
#
# **So the cap is a ceiling plus one interval of growth, not a hard limit**, and
# the difference is only invisible at the ordinary rate. At a kilobyte a minute
# the overshoot is a kilobyte; under a flood writing sixteen a second the test
# run rotated a file of 82 KB against a 20 KB cap on a five-second check, which
# is the arithmetic behaving exactly as stated and looking alarming anyway.
# What bounds the disk is `MAX_LOG_BYTES` times two plus that overshoot, since
# there is one `.old` and it is overwritten.
LOG_CHECK_SECS=60

watch_log() {
    while : ; do
        sleep "${LOG_CHECK_SECS}"
        [ -f "${LOG}" ] || continue
        size=$(wc -c < "${LOG}" 2>/dev/null || echo 0)
        if [ "${size}" -gt "${MAX_LOG_BYTES}" ]; then
            cp -f "${LOG}" "${LOG}.old" 2>/dev/null || continue
            : > "${LOG}"
            echo "[run-pool] rotated at ${size} bytes $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG}"
        fi
    done
}

# How long a run has to last before it counts as having worked. Under this and
# the pool is failing at startup -- a taken port, a missing file -- and the
# backoff should keep climbing. Over it and whatever happened was a one-off.
HEALTHY_AFTER=60

# Started once and killed with the script, so a stop leaves nothing behind.
# It is added to the existing traps rather than replacing them, because the
# pidfile removal is what stops the *next* start refusing.
watch_log &
WATCHER=$!
trap 'rm -f "${PIDFILE}"; kill "${WATCHER}" 2>/dev/null || true' EXIT
trap 'rm -f "${PIDFILE}"; kill "${WATCHER}" 2>/dev/null || true; exit 0' INT TERM

backoff=1
while : ; do
    rotate_between_runs
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
