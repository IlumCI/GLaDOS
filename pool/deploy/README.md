# Putting the pool on a server

The daemon is one statically linked file with no runtime dependencies, so
"deploy" is copy, install a unit, open a port. There is nothing to configure on
the host and no toolchain to install there.

## Build it here, not there

```bash
cd pool
cargo build --release --target x86_64-unknown-linux-musl
```

musl rather than gnu so the result is **static**: no glibc version to match
against whatever the host runs, no shared objects, one artefact. Verified on
the current build: 784,784 bytes, x86-64, no `PT_INTERP`, which is what "needs
no loader" looks like in the file itself.

If the server is ARM rather than x86-64 -- a Pi, an Ampere or Graviton instance
-- the target is `aarch64-unknown-linux-musl` instead, and `uname -m` on the
box settles it (`x86_64` or `aarch64`). The rest of this file is unchanged.

**Check before copying**, because the failure mode of the wrong architecture is
`cannot execute binary file` after everything else has already been set up:

```bash
file target/x86_64-unknown-linux-musl/release/glados-pool
# ELF 64-bit LSB pie executable, x86-64, static-pie linked, ...
```

## First, find out what you are deploying onto

All read-only, and `sudo -n` cannot hang on a password prompt:

```bash
uname -m; uname -r
grep PRETTY /etc/os-release
systemctl --version 2>/dev/null | head -1
sudo -n true 2>/dev/null && echo "sudo: yes" || echo "sudo: no (or wants a password)"
loginctl show-user "$USER" -p Linger 2>/dev/null
nproc; free -m | head -2
ss -ltn 2>/dev/null | grep ':3334' || echo "port 3334: free"
cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/cgroup.controllers 2>/dev/null
```

What each answer changes:

- **`uname -m`** picks the build target and nothing else. `x86_64` or `aarch64`.
- **sudo** picks which unit file. With it, `glados-pool.service`; without,
  `glados-pool.user.service` and the caveats in its header.
- **`Linger=no`** with no sudo is the trap: a user service stops when you log
  out, so the pool works while you watch it and is gone by morning.
  `loginctl enable-linger $USER` fixes it, and needs sudo on some systems.
- **`cgroup.controllers`** says whether the resource limits in a *user* unit
  are enforced or merely written down. If `memory` and `cpu` are absent they
  are documentation.
- **port 3334 in use** means pick another and change it in three places: the
  unit, the firewall, and `mine pool`.

## Install

```bash
# on the server, as root
useradd --system --no-create-home --shell /usr/sbin/nologin glados-pool
install -m 0755 glados-pool /usr/local/bin/glados-pool
install -m 0644 glados-pool.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now glados-pool
systemctl status glados-pool
```

Edit the `ExecStart` line first: the coins and their share targets are the only
thing in the unit that is a decision rather than a measurement.

### Without root, which is the expected case here

The daemon needs no privilege: a port above 1024, one directory to write, and
nothing else read that it did not create. What is lost is *enforcement* of the
resource limits, not function.

**Everything turns on one command.** Try it first, because it decides which of
the two paths below applies:

```bash
loginctl enable-linger "$USER" && echo "lingering: yes" || echo "lingering: no"
```

Lingering is what keeps a user service running after you log out. It is
governed by polkit and some systems grant it to a user for themselves while
others want an administrator, so the only way to know is to ask.

**Measured on a real deployment (Arch, systemd 258, over SSH): denied.**
`Could not enable linger: Access denied`, and the session was `State=active` at
the time, so an active session is not sufficient on its own -- this distribution
wants an administrator for `set-self-linger` regardless. Plan for the ask rather
than for the grant.

**And the fallback is not universally available either.** That same host had no
`crontab` binary and no cron daemon, which is ordinary on Arch -- systemd timers
replaced it, and a systemd timer needs the user manager that lingering is what
keeps alive. So on a box with neither, there is no unprivileged path to
persistence at all and the honest answer is one line from whoever has root:

```bash
sudo loginctl enable-linger <user>      # the whole ask
```

Everything else can be installed, enabled and left correct *before* that
happens: `systemctl --user enable` writes the `default.target.wants` symlink
immediately, so the moment lingering is granted the service starts at boot with
nothing further to do.

#### The trap that hides all of this

A stale session masks it completely. On that deployment the pool ran fine for
an hour and a half with `Linger=no`, because an interactive SSH session was
still open and the user manager lives as long as *any* session does. Every
check said active, enabled, restarting cleanly -- and it would have vanished
the moment that terminal closed. This is the unit file's own "works perfectly
while you are watching it and is gone in the morning", observed rather than
predicted.

Check what is actually holding the manager up, not just that it is up:

```bash
loginctl show-user "$USER" -p Linger -p Sessions
loginctl list-sessions --no-legend | awk -v u="$(id -u)" '$2==u'
```

If `Linger=no` and the only sessions are yours, the service is borrowed time.

#### If lingering worked

```bash
mkdir -p ~/.local/bin ~/.config/systemd/user
install -m 0755 glados-pool ~/.local/bin/glados-pool
install -m 0644 glados-pool.user.service ~/.config/systemd/user/glados-pool.service
systemctl --user daemon-reload
systemctl --user enable --now glados-pool
systemctl --user status glados-pool
```

If the unit refuses to start, read the reason before changing anything: a user
manager *fails* a unit on a directive it cannot apply rather than skipping it,
so the fix is to delete the offending line, and the message names it.

**Check whether the limits are real rather than decorative**, which the unit
file asks for and which is easy to skip:

```bash
cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/cgroup.controllers
systemctl --user show glados-pool -p MemoryMax -p CPUQuotaPerSecUSec
```

On the deployment above that read `cpu memory pids`, and the unit reported
`MemoryMax=67108864` and `CPUQuotaPerSecUSec=500ms` -- so both were genuinely
enforced. Where `cpu` and `memory` are absent from the first line, those
directives are documentation and the daemon is bounded only by its own
behaviour.

What it actually used there, mining two coins over WireGuard: **228 KiB
resident** against the 64 MiB cap, and 11.5 ms of CPU across several minutes.
The limits exist for a bug, not for the steady state.

#### If lingering did not

`run-pool.sh` is the fallback: a restart loop with a capped log and a pidfile,
started from `cron`, with no systemd involved at all.

```bash
mkdir -p ~/.local/bin ~/.local/state/glados-pool
install -m 0755 glados-pool ~/.local/bin/
install -m 0755 run-pool.sh ~/.local/bin/
( crontab -l 2>/dev/null; echo "@reboot /bin/sh \$HOME/.local/bin/run-pool.sh" ) | crontab -
nohup ~/.local/bin/run-pool.sh >/dev/null 2>&1 &
tail -f ~/.local/state/glados-pool/pool.log
```

Edit the coins at the top of the script first -- that is the only line in it
which is a decision rather than plumbing.

Stopping it, and checking on it:

```bash
kill "$(cat ~/.local/state/glados-pool/run-pool.pid)"   # stops cleanly
tail -n 40 ~/.local/state/glados-pool/pool.log
cat ~/.local/state/glados-pool/ledger.json
```

A plain `kill` is enough and that is worth stating, because it was not true of
the first version of the script: a trap handler that does not call `exit`
returns to what it was doing, so it removed its pidfile and carried on. Found
by sending it a signal rather than by reading it.

It is worse than the systemd path in ways worth knowing rather than
discovering: no resource limits at all, `@reboot` needs `cron` to be running
and the user to be allowed a crontab, and the log is capped by the script
itself at 4 MB because nothing else is going to rotate it.

#### Either way, the limits are probably not enforced

A user manager applies `MemoryMax` and `CPUQuota` only where the controllers
are delegated to it, and `cpu` frequently is not:

```bash
cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/cgroup.controllers
```

If `memory` and `cpu` are missing, those lines in the unit are documentation.
The daemon still only uses what it uses -- 3.49 MB resident plus at most one
8,296 KiB working set, measured -- but nothing is holding it there, so watch it
for a day before trusting it unattended. With the cron fallback there are no
limits at all.

#### The two things that may still need him

**A firewall rule**, if one is running. Most server installs have none active
by default, so try connecting from another machine before asking:

```bash
ss -ltn | grep 3334        # on the server: is it listening
# from another machine on the same network:
timeout 5 nc -z <server-ip> 3334 && echo open || echo blocked
```

**Nothing else.** No port forward and no DNS while this stays on a private
network, which is what the home-server section above argues for.

## Check it before opening any port

The daemon can prove itself with no network and no miner:

```bash
/usr/local/bin/glados-pool --selftest
```

It stands up a listener on loopback, greets it, takes two jobs on two different
algorithms, mines them, submits, and requires the pool to have independently
recomputed both hashes and agreed. A pass means the binary is right for this
machine.

```bash
/usr/local/bin/glados-pool --bench
```

Prints what one share costs to validate here, which is the number that decides
whether the share targets in the unit are sane. Compare against `design/pool.md`;
a much slower machine wants harder targets, not a bigger box.

**Read the algorithm names on that table rather than skimming for "yespower".**
The 2 MiB and 8 MiB profiles differ by 3.5x and both are spelled yespower; the
unit configures the 2 MiB one. Sizing `--cpu-percent` against the wrong row is
sizing it for a coin you are not serving.

### And make it refuse things, which is the half a selftest cannot do

`tools/poolabuse.py` goes past every limit in `server.rs` on purpose. Point it
at a **throwaway instance on its own port with its own ledger** -- it submits
garbage by design and the share log it leaves is worthless:

```bash
# a second pool, out of the way of the real one
glados-pool --listen 127.0.0.1:3335 --ledger /tmp/abuse.json \
    --cpu-percent 1 abuse:neoscrypt:16 &

python3 tools/poolabuse.py --port 3335 badshares    # 33 answered, then dropped
python3 tools/poolabuse.py --port 3335 ratelimit    # 20 of 60, still connected
python3 tools/poolabuse.py --port 3335 bigline      # desynchronised, closed
python3 tools/poolabuse.py --port 3335 conns -n 300 --hold 20
python3 tools/poolabuse.py --port 3335 names -n 6000
python3 tools/poolabuse.py --port 3335 flood --conns 4 --seconds 20
```

`--hold` on the connection test is not optional if you want to *measure*
anything: without it the connections open and close inside a sampling interval
and every counter reads idle. Sample the daemon while it holds them:

```bash
watch -n2 'grep -E "VmRSS|Threads" /proc/$(pgrep -f "listen 127.0.0.1:333[5]")/status'
```

What each one should say is in `design/pool.md`. Two of them exist because they
found something: `names` because one connection could re-greet under nineteen
new worker names a second, each a permanent record and none of them costing a
single validation, and `flood` because the per-connection limits do not sum.

### What they answered here, so a future run has something to differ from

All six, against the musl artefact on a **12th Gen i7-12650H, 16 threads** --
which is the development machine and **not** the server, so read these as the
shape being right rather than as the server's numbers. The throwaway instance
was `--cpu-percent 1` with `abuse:neoscrypt:16` on its own port and ledger.

    badshares   cap 32, answered 33, then dropped
    ratelimit   cap 20, offered 60, answered 20, still_open true
    bigline     max_line 131072, sent 131158, server_closed true
    conns       ceiling 256, attempted 300, welcomed 256, refused 44,
                connect_errors 0, accepts_after true
    names       names_offered 1160, renames_refused 1160, 19.0 names/s
    flood       4 threads, 20 s, offered 1600, answered 437 (21.8/s)

Every one matches what this file already predicted, which is the result. Two are
worth reading rather than ticking:

**`names` reports 19.0 a second and refuses all 1,160 of them.** That rate is
the number this file records as the hole -- so what is being seen is the attempt
rate unchanged and the refusal working, which is the only way that drill can
report success.

**`flood` answered 437 of 1,600 offered.** That is the *total* validation budget
doing the thing the per-connection limit could not: 73% of offered work was
never validated, at a 1% budget. Raise `--cpu-percent` and this number rises
with it, which is what makes it the setting that decides how much of somebody
else's machine a flood can take.

**And `names` has a `--seconds` default of 400**, so a shorter `timeout` around
it kills it mid-drill and reports nothing. Pass `--seconds` explicitly if you
are bounding the run; the figures above are from `--seconds 60`.

The throwaway's own ledger is the other half of the evidence, because the drills
are supposed to leave a record and a pool that refused everything correctly
should be able to prove it:

    python3 tools/ledgercheck.py /tmp/abuse.json
      ok    the document is a format this knows (v3)
      ok    the rows hash to the published digest
      ok    <every worker> has no negative counts

33 bad from `badshares`, 20 from `ratelimit`, 437 across four `flood` workers,
and `names-0` carrying 4,580 stale against coin `?` -- a share for a coin that
does not exist, counted as stale rather than accepted.

Then the negative, because a checker that only ever says yes is not a checker.
One unit of work moved from one worker to another with the digest left untouched
-- the exact edit somebody would make to steal a slice of a payout:

    FAIL  the rows hash to the published digest
          [published 55c99664..., recomputed e2041992...]
      1 check(s) failed. The digest is not a signature, so this says the record
      changed

## If it is a home server, read this before the port section

A machine in somebody's house is not a small VPS. Four things change, and the
last one is a decision rather than a detail.

**It probably has no reachable address.** A residential connection has a
dynamic IPv4 at best and is behind carrier-grade NAT at worst, and under CGNAT
no port forward exists to make -- the address on the router is not the address
the world sees. `curl -s ifconfig.me` on the server against the WAN address in
the router's own status page settles it: if they differ, forwarding cannot
work and a tunnel is the only route in.

**A forwarded port is a hole in his house, not in a rented box.** Everything
behind that NAT is his: other machines, whatever is on the LAN, his own
traffic. A VPS that gets compromised is a VPS. This is not that, and it is his
risk rather than ours to accept on his behalf.

**A DNS record pointing at his home publishes where he lives**, to street
level, to anybody who resolves `stratum.aperture.institute`. That is a fact
about him that a mining pool has no business making public, and it does not
become reversible by deleting the record later.

**His ISP may forbid it.** Running a public server on a residential line is
against the terms of most of them, and mining-adjacent traffic is the kind
that gets noticed.

### What to do instead, for now

**Do not expose it to the internet yet**, and the reason is not caution -- it
is that doing so buys nothing. Every coin's `Source` is `Local`: the pool
builds its own headers, so there is no chain, no block, and nothing for a
stranger to mine that is worth anything to them or to us. Opening a port on a
friend's house to serve work with no value is all cost.

What the server is genuinely useful for today is being the always-on end of a
*private* link:

- **Same LAN.** If the GLaDOS machine and the server are in one house, this is
  finished: `mine pool <lan-ip>:3334 glados` and nothing is exposed at all.
- **Different houses**, which is the likely case. Put both machines on a
  WireGuard or Tailscale network and point the miner at the private address.
  The kernel has its own TCP stack and no WireGuard, so **the tunnel cannot
  run inside GLaDOS** -- it has to be the router, or the host if GLaDOS is in
  QEMU, and the kernel just sees an ordinary address it can reach.

Revisit hosting when there is something to mine. By then the two things that
make exposure defensible will exist or will not: an upstream behind at least
one coin, and TLS. Until both are true, a public endpoint is a liability with
no matching asset -- and if it still seems worth it then, a five-dollar VPS
fronting the home box keeps his address and his LAN out of it entirely.

## The port

3334, TCP, inbound. Above 1024, so the daemon binds it without any capability.

```bash
# ufw
ufw allow 3334/tcp
# firewalld
firewall-cmd --permanent --add-port=3334/tcp && firewall-cmd --reload
```

Cloud hosts usually have a second firewall in their control panel that has
nothing to do with the one on the machine. Both have to allow it, and a
security group nobody opened looks exactly like a daemon that is not running.

From somewhere else:

```bash
printf '{"id":1,"method":"glados.hello","params":{"v":1,"worker":"probe","agent":"curl"}}\n' \
  | timeout 5 nc <host> 3334
```

A welcome and then a job or two means the whole path is open. Silence means a
firewall; a refused connection means the daemon.

## Then point the kernel at it

```
mine pool <host>:3334 glados
mine user <worker-name>
mine slices 3
mine on
mine coins
```

### And let somebody else check your arithmetic

The published ledger is a *tally* -- work, accepted, stale, bad per worker -- and
a tally cannot be re-verified. A miner reading it is trusting that your program
counted honestly, which is the thing publishing it was meant to replace.

`--sharelog` writes the half that can be checked: one line per accepted share,
carrying the assembled header, the nonce and the target.

```bash
glados-pool --listen 0.0.0.0:3334 --sharelog <state>/shares.txt <coins...>
glados-pool --verify <state>/shares.txt
```

The verifier needs no pool, no socket and no state -- it reads a line, builds the
hasher the line names, hashes the header with the nonce and compares against the
target, using the kernel's own code by `#[path]`. So it runs on a miner's laptop,
or on a CI runner, or anywhere somebody wants to check you.

Driven: 13 shares from a real session verify and exit 0; change one hex digit of
one nonce and it prints

    line 1: rig-alpha does NOT meet its target, digest a66682ae67fe...
    12 share(s) recomputed and met their target, 1 did not, 0 unreadable

and exits 1.

**The audit runs in its own repository**, [IlumCI/glados-pool][gp], on a schedule
and on demand. It checks out *this* repository and builds `pool/` from it rather
than holding a copy, because `pool/src/lib.rs` reaches the kernel by `#[path]`
and a second yespower over there is exactly the drift that arrangement exists to
prevent. So there is one source of truth and the other repository holds only the
running and the published record -- which also keeps it small enough for a miner
to read.

[gp]: https://github.com/IlumCI/glados-pool

Separating them is about blast radius rather than tidiness: the audit repository
holds no signing keys, needs no `workflows` write, and if its Actions usage were
ever questioned it would not take this repository's releases, ISOs, `loop/main`
or the Gödel machine with it.

**It is the audit and not the pool**, and it cannot be the pool: a runner has no
inbound networking, and share acceptance has to be inline anyway. Which is the
right half to put somewhere free -- accepting a share is cheap and you do it
regardless, while independent re-verification is the one thing you structurally
cannot do for yourself.

Driven on a real runner, both ways. Three shares from a live session verify and
the run is green; one hex digit of one nonce changed and it goes red, naming the
share and printing the digest. **The first attempt found the forgery and reported
success**, because `| tee` makes a pipeline's exit status `tee`'s -- so the fix is
`set -o pipefail`, and the lesson is that only the negative case could ever have
shown it.

**It is unbounded, so it is off by default.** About 200 bytes per accepted share,
forever. Rotate it or publish and truncate per epoch.

### Or hand somebody an ISO that does all of that by itself

A miner-only image: no model, no store, and one activity. It is **32.6 MB**
against the full ISO's 1.90 GB, because almost all of that is weights, and it
boots in seconds rather than the minutes a checkpoint costs to load.

```bash
cargo build --release
mkdir -p iso-payload
cat > iso-payload/MINER.TXT <<'TXT'
pool p.example.com:3334
worker 0xYOUR_PAYOUT_ADDRESS
protocol glados
slices 3
TXT
python3 tools/mkiso.py out/miner.iso \
    --efi target/x86_64-unknown-uefi/release/glados.efi \
    --payload iso-payload --allow MINER.TXT --no-license
```

`--no-license` is correct here and is not a way round the gate: that check exists
because the payload normally carries model weights whose licence has to travel
with them, and this payload carries no licensed content at all. `--allow` is
needed because `mkiso.py` places only files `payload/*.txt` names, an allowlist
rather than a denylist for `eval.rs`'s reason.

Boot it and it says what it is doing before the prompt appears:

    [miner] mining as 0x...beef at 10.0.2.2:3334, 2 slice(s)

**`worker` has no default and the image refuses to mine without one.** At the
account-free venues the worker name *is* the payout address, so an image that
booted with a built-in default would mine to whoever owns that default for as
long as nobody noticed. Without it the machine reaches a prompt and sits there,
which is the failure that is visible rather than the one that is profitable.

**Nothing in `MINER.TXT` is executed.** Keys are parsed into typed fields and the
fields act; a line this parser does not know is counted and reported, never
passed to a shell. That matters because the file is on media anybody can edit
and it names a machine to connect to -- `update::repairs` makes the same argument
about the file beside it, and `diag`'s `miner config` section carries eighteen
claims about it, one of which is that a line reading `rm -rf /` is just one more
thing not understood.

Driven end to end on a Linux host: the ISO booted under KVM, read its config,
connected, and the pool logged
`hello worker=0x...beef agent=glados/1.3.8` followed by accepted shares in two
disjoint nonce ranges -- which is `slices 2`, each slice owning its own quarter
of the space.

**What it does not have**, deliberately: no checkpoint, so `ai::init` returns
early and the eleven-ish model-dependent boot selftest sections do not run. The
tally still reads green because the *suites* all pass, which is the hazard
`.github/actions/verify-boot` grew a section-count check for. A miner image is
the configuration that walks into it by design, so read a green boot on one as
saying less than a green boot on a full image.

## What this does not do yet

**No TLS.** Worker names travel in the clear and anything on the path can
rewrite a job. The kernel has one TLS session and the updater owns it, so this
is a real limitation rather than a setting -- see `design/pool.md`. Until it is
fixed, a miner on an untrusted network is trusting the network.

**The ledger survives a restart, but only if you pass `--ledger`.** Without it
the tally is in memory and a restart begins at zero. With it the file is read
back at start, and a truncated or edited one is refused whole and announced
rather than half-loaded.

**No chain behind any coin.** Every `Source` is `Local`: the pool builds its own
headers, so shares are real proof of work against a target nobody else
recognises. This is a working pool with nothing to mine yet, and the report says
`local` on every row rather than implying otherwise.
