# The pool, and the one protocol it speaks downstream

Status: **it has met the actual internet.** The pool speaks Stratum V1 to a
real upstream, builds headers from its `mining.notify`, serves them to the
kernel over the glados protocol, and sends back the shares good enough to
matter. Driven end to end against a controlled server that verified every
forwarded share independently, and then against `solo.ckpool.org`, whose real
coinbase it rebuilt and named the outputs of. What has *not* happened is a
share going back up to a chain, because at this hashrate none will.

`design/mining.md` is the other half and should be read first: it covers what
the kernel does with the work this pool hands it.

## The problem this exists to solve

The kernel can now work four coins at once, each on its own algorithm, and
three of the four slots are fed by a *fixture* job because there is nowhere
for a real one to come from. Stratum V1 has no field that says which
proof-of-work a job wants -- a pool serves one coin and the miner is simply
assumed to know. So four coins means four connections, and four socket tasks
do not fit in `MAX_TASKS` beside four mining slices.

One connection carrying jobs for several coins needs a field Stratum does not
have. That field is the whole reason this protocol exists.

| | work |
|---|---|
| Kernel | N hash algorithms, one protocol, a supervisor with real budgets |
| Pool | N stratum dialects, N upstream connections, share accounting, conversion, payout |

## The kernel never learns what a coinbase is

The job the pool sends is an **assembled 80-byte header with a zero nonce**,
an algorithm, and a target. The miner substitutes a nonce at offset 76, hashes,
compares, and sends back the nonce. That is the entire downstream contract.

Monero's stratum has this shape and Bitcoin's does not, and the difference is
the point: Bitcoin's `mining.notify` hands over `coinb1`, `coinb2`, an
extranonce and a merkle branch, and the miner assembles the header itself. Do
that for a dozen coins and the kernel carries a dozen chains' worth of
transaction-format knowledge, in ring 0, to compute a number the pool already
knows.

`work::Template` was already this shape before this document existed. It carries
`header: [u8; 80]` rather than a midstate, and the comment saying why is about
yespower rather than about protocols -- but the consequence is that the kernel
side of this needs no new structure, only a new way to fill one.

**The Stratum V1 client in `src/mine/` is not wasted.** It moves to the pool as
its *upstream* client, where it is ordinary host code with a heap and a test
runner, and it is the same source file rather than a second implementation.
See below.

## What that costs, stated rather than discovered

A miner handed an assembled header **cannot check what it is mining for.** In
Stratum V1 the coinbase arrives in two halves and a suspicious miner can
reconstruct it, read the outputs, and see which address the block would pay.
Here the coinbase is inside a merkle root inside a header, and a merkle root is
a hash: there is nothing to inspect.

That is a real reduction in what a miner can verify and it is not waved away by
the pool being ours. So the job carries **optional** `coinb1`, `coinb2`,
`extranonce` and `branch` fields. When they are present the miner can assemble
the header itself and require it to equal the one it was sent, which is a
complete check: a matching header proves the coinbase it was shown is the
coinbase that was committed to.

The kernel keeps `header::coinbase` and `header::merkle_root` for exactly this,
and `ev::coinbase_value` already reads the outputs. None of it is on the hot
path -- a job is verified once and then hashed several billion times -- so the
verification costs approximately nothing and is the difference between a pool
that asks for trust and one that offers evidence.

**Built.** `proto::proves` is the check and lives beside the codec so both ends
run one function. Measured, kernel in QEMU against a pool with a real upstream
and a three-level branch:

    0  chain  1 slice  pool  254139 H/s  sha256d
       checked: the coinbase it committed to pays 50.12345678

That figure is the upstream's own coinbase value to the satoshi, so the kernel
rebuilt the coinbase, folded the branch, matched the header's root and summed
the outputs in ring 0.

**And the refusal is watched rather than assumed.** `glados-pool --bad-proof`
flips one byte of the extranonce before sending -- one byte, because a check
that only caught obvious garbage would pass on every interesting lie. Against
that pool the kernel installs no coin at all and says `a job's proof does not
match its header -- refused`. That a pool can lie on purpose is the
arrangement `diag paging` has when it faults deliberately.

A *local* coin sends no proof and there is a claim that it does not: with no
chain behind it the coinbase would be a fabrication, and a proof verifying
against an invented header is worse than none because it looks like evidence.

### The same opacity buys something, and only the cost was written down

Rosenfeld's analysis of pooled reward systems proposes **oblivious shares** as
the real answer to block withholding: shares a miner cannot recognise as full
blocks without submitting them. Both withholding attacks need that recognition
-- sabotage is discarding a block you know you found, and lie-in-wait is
delaying one while you pile hashrate into the pool holding it -- and he notes
the idea would need a change to Bitcoin itself.

This protocol approximates it for nothing. The job carries
`target_with_leading_zeros(bits)`, which is the miner's own VarDiff **share**
target; `up_target` lives in the pool's `Issued` record and never crosses the
wire. Combined with the assembled header above, a miner running this protocol
has neither the network target nor a coinbase to reason about, and cannot tell
a block from an ordinary share.

**It is obfuscation and not a guarantee, and overselling it would be worse than
not having it.** A miner that knows which coin it is on can read the network
target off a block explorer and do the comparison itself. What it changes is
the cost: withholding stops being free and becomes deliberate. Recorded here
because this section had the cost of the opaque header and not the compensating
benefit, and a design note that lists only one side of a trade is how the trade
gets undone by somebody tidying up.

## JSON lines, for one reason

Newline-delimited JSON objects, the way Stratum is. Not because it is a good
wire format -- it is not, and a binary one would be smaller and faster to
parse.

The reason is that `crate::json` is already in the kernel, is already used by
`src/mine/stratum.rs`, and is already checked at boot. A binary protocol needs
a new parser in ring 0, written for this, and the thing this tree has learned
repeatedly is that a second parser is where the disagreement lives. The bytes
saved are a rounding error against a job that arrives every thirty seconds.

## The methods

Four, and a job notification.

```
-> {"id":1,"method":"glados.hello","params":{"v":1,"worker":"...","agent":"glados/1.3.5"}}
<- {"id":1,"result":{"v":1,"slots":4,"session":"..."},"error":null}

<- {"method":"glados.job","params":{
     "slot":1, "coin":"bitzeny", "job":"a3f1",
     "algo":{"name":"yespower","v10":true,"n":2048,"r":8,"pers":"..."},
     "header":"<160 hex chars>",
     "target":"<64 hex chars>",
     "echo":{"extranonce2":"...","ntime":"..."},
     "clean":true
   }}

-> {"id":7,"method":"glados.submit","params":{
     "job":"a3f1", "nonce":"1f8c04b2", "echo":{...}
   }}
<- {"id":7,"result":{"ok":true,"diff":0.001},"error":null}

-> {"id":9,"method":"glados.work","params":{"slot":1}}
<- a glados.job for that slot, or silence
```

Five decisions in that, each with an alternative that is worse.

**`slot` is the pool's, not the miner's.** The pool says which of its coins a
job belongs to and the miner puts it where it likes; the kernel's own slot
numbering is a local matter. Letting the miner choose would mean the pool
tracking a per-connection mapping to attribute a share, which is state that can
disagree with the miner's.

**`algo` is a structure and never a name.** `"yespower"` alone is not an
algorithm: BitZeny, Yenten and Koto all run yespower and all run it with
different parameters, so a name-only field is a job that hashes a different
function perfectly correctly and has every share rejected. This is the same
refusal `Algo` already makes about a per-coin preset table, moved onto the
wire.

**`glados.work` exists because a job is a finite search.** The nonce is four
bytes at a fixed offset, so a job carries 2^32 hashes and not one more --
eight and a half seconds on an RTX 3050, against a pool that re-issues on a
thirty-second idle timeout. Without a way to say so, a device four times too
fast for its work does not stop: it wraps and rescans the space it has already
searched, and resubmits every share it finds there. Measured on the card before
this existed, over one three-coin run: **23 accepted shares against 35
duplicates**, with every repeated nonce arriving exactly six times.

The `duplicate` verdict had been counting that correctly the whole time and
nobody had read it as a defect. It looked like a retry, which is what the
counter is mostly for, and the abuse limiter deliberately does not count it as
abuse for exactly that reason -- so the one number that knew was also the one
number designed not to complain.

It names **one slot** rather than asking for everything. The reason to ask is
always about one coin, `make_job` advances the extranonce2 for an upstream coin
and the job counter for a local one so the answer is a genuinely different
search either way, and re-issuing every slot on every message is the precise
shape of the churn bug the abuse test found once already. It is bounded at four
a second per connection, which is about the job ring rather than about the CPU:
`KEEP_JOBS` is 64, and a connection allowed to ask freely would evict the jobs
every other miner is working.

Silence is a legal answer. An upstream coin with no template yet yields no job,
and inventing one would put a miner on a search that can never pay.

**`echo` is opaque and is returned verbatim.** Whatever the upstream dialect
needs back at submit time -- an extranonce2, an ntime, a Monero job blob id --
travels as a JSON object the miner stores and hands back without reading. The
kernel does not learn what any of it means, and adding an upstream that needs a
fourth field costs no kernel change at all.

**`target` is 32 bytes of hex rather than a difficulty.** `mining.set_difficulty`
sends a decimal, altcoin pools routinely send fractional ones, and the kernel's
own `stratum::decimal` exists because `Json::as_i64` reads `0.001` as `0` and a
target computed from zero either faults or accepts everything. A target on the
wire has no such conversion in it. The pool does the arithmetic once, host-side,
where a float is available and a mistake is visible.

## Why the pool is Rust, and shares the kernel's source

A pool that accounts shares must **validate** them, and validating a share
means computing the same hash the miner computed. If the pool is TypeScript
beside a Deno function, there are now two implementations of yespower that have
to agree, and this tree's whole recorded experience is that two implementations
which are supposed to agree do not stay agreeing on their own -- it is why
`tokenizer.py` has `--verify`, why `v4.py` is not the writer, and why
`differ.rs` exists before the thing it gates.

So the pool includes the kernel's own files. Not a copy and not a port:

```rust
#[path = "../../src/mine/yespower.rs"] pub mod yespower;
```

Nine modules travel that way (`u256`, `hash`, `header`, `blake2s`, `yespower`,
`algo`, `stratum`, `proto`, `ev`) plus three they rest on (`store::sha256`,
`crypto::hkdf`, `json`). They were already almost free of the kernel: `sha256`,
`blake2s`, `u256`, `header` and `ev` name nothing outside themselves at all,
and the rest need three modules and one macro. `kernel_shim` supplies that
macro and nothing else.

**A divergence is a build failure rather than a wrong answer**, which is the
whole of the argument. Change the kernel's yespower and the pool stops
compiling or stops passing its vectors, in the same commit, on the same source.

**And this is the first `cargo test` in the project.** CLAUDE.md opens its
testing section with "There is no `cargo test`", which is true of the kernel and
was true of everything the kernel contains. The hash core is not kernel code in
any meaningful sense -- it touches no hardware and allocates from a heap -- so
on the host it is ordinary Rust with a test runner in front of it, and the
vectors `diag mine` asserts at boot run in about a second instead of a boot.

The boot claims stay. They check the same things on the machine that will
actually run them, which is what `diag paging` exists to say about permissions.

**CLAUDE.md said this could not be done.** `Cargo.toml`'s note records that a
host cdylib fails with "linker `link.exe` not found", and the conclusion drawn
from it -- anything that has to run on the host at build time is out -- was read
as covering host binaries too. Measured, this session: a plain host binary
compiles, links and runs. Whatever was missing has been installed since. The
proc-macro and `-Zbuild-std` findings were not re-tested and are left standing.

## Where this runs, and the one thing GitHub Pages cannot do

The intended home is `pool.aperture.institute`, on a domain this project owns,
and the intended host was GitHub Pages. **Pages cannot run the daemon**, and
the reason is not a setting: it serves static files over 80 and 443 from a CDN,
with no long-lived process and no listener on a port of its own. The pool is a
TCP server that holds connections open for hours and speaks line-delimited JSON
on :3334. Nothing about that is static.

What Pages *is* right for is the half that matters most, which is the published
record. So the two halves get two names, because one hostname resolves to one
place and it can be GitHub's CDN or a server, never both:

| | where | what |
|---|---|---|
| `pool.aperture.institute` | its own repo, GitHub Pages | the published share log, the docs, how to point a miner |
| `stratum.aperture.institute:3334` | a host that runs a binary | the daemon |

**A second repository is genuinely required for the site**, and not as tidiness:
one repository serves one Pages site with one `CNAME`, and this one's is
already `glados.aperture.institute` in `docs/CNAME`.

**The daemon's source stays in this repository**, though, and that is the whole
argument of this file. `pool/` works by including twelve of the kernel's own
files with `#[path]`; move it out and that becomes a submodule or a vendored
copy, and a vendored copy is precisely the drift the arrangement exists to
prevent. The site repository holds published data and HTML and needs none of
the Rust.

### How big a host, measured

`glados-pool --bench` answers it, and the answer is "almost anything", for a
reason worth stating plainly because it is the opposite of the intuition mining
usually carries:

> **A miner grinds several million nonces to find one share. The pool hashes
> exactly once to agree.** So the pool's load scales with the share *rate*,
> which the operator sets by choosing the target, and not with anybody's
> hashrate. A room full of fast miners costs this server no more than the same
> room of slow ones.

Best of nine, on the development machine:

| algorithm | per share | working set | shares/s on one core |
|---|---|---|---|
| sha256d | 0.74 us | 0 | 1,347,708 |
| blake2s | 0.20 us | 0 | 5,025,125 |
| yespower 2 MiB | 1.91 ms | 2,146 KiB | 524 |
| yespower 8 MiB | 7.49 ms | 8,296 KiB | 133 |

And the process itself: **370 KB of binary, 3.49 MB resident** with four coins
configured and five threads.

The transient peak is one working set on top of that, not one per miner, and
that falls out of something accidental rather than designed: `Pool` sits behind
a `Mutex` and validation happens while it is held, so shares are checked one at
a time. Worth knowing before anybody "fixes" that for throughput -- a parallel
validator would make peak memory `threads * 8 MiB`, which is the one way this
program could become large.

**So 32 MB of RAM and one core is comfortable, and the CPU number is the
generous one.** yespower here is the *reference* implementation, deliberately
unoptimised so the vectors have something plain to check; an optimised
validator is roughly eight times faster, and the pool has no reason to want one.

Sized against real miners: a GF63 slice does about 387 H/s of yespower, so at a
12-bit share target (4,096 hashes) it submits one share every ten seconds or so.
That is 0.1 shares a second, and one core absorbs **thirteen hundred** of those
even at the 8 MiB setting.

**The knob that could hurt is the share target, and it is the operator's.** Set
it too low and a single miner submits constantly; one core saturates at 133
yespower-8 MiB validations a second, so anything above about one share per
second per miner means the target is wrong rather than the server small.

Network and disk are noise: a job is a few hundred bytes per coin per thirty
seconds, a share is under two hundred, and the ledger is a few kilobytes.

**The log is not noise, and rotating it at restart was rotating it never.**
`run-pool.sh` caps `pool.log` at 4 MiB and checked that once per iteration of
its supervisor loop -- which is once per pool *exit*. So the healthier the pool
the less often the check ran, and a pool that never crashes never rotates at
all. Measured on the deployed instance with two miners on it: about a kilobyte
a minute, 1.4 MB a day, three days to the cap and then nothing, forever, on
somebody else's disk. Only a soak long enough for the pool not to restart could
show that, which is why it survived every earlier test of the script.

The fix is a watcher beside the running pool, and it cannot use the same verb.
`mv` renames the file while the pool's stdout descriptor still points at the
inode, so the pool goes on writing into `pool.log.old` and `pool.log` sits at
zero bytes -- which reads as a pool that stopped logging. The redirection is
`>>`, so copy-then-truncate works: every later write lands at the new end of a
file that is now empty. `mv` stays in the loop, between runs, where nothing has
it open.

**Do not build on the small machine.** Cross-compile and copy one file:

```bash
cargo build --release --target x86_64-unknown-linux-musl
```

musl rather than gnu so the result is statically linked -- no glibc version to
match against whatever the host is running, no runtime dependencies, and the
whole daemon is a single artefact to copy. A weak box should not be asked to
hold a Rust toolchain.

### The record goes as files, not as an endpoint

`--ledger PATH` writes the share log as canonical JSON. A static site whose
history is a commit chain is **tamper-evident by construction**: a live JSON
endpoint can be quietly rewritten and a commit chain cannot, without it showing.
For a pool whose only asset is being checkable, that beats freshness. It is the
same trade `site.yml` already makes for the download tables, and the same reason
`godel`'s ledger is a file rather than a query.

Three details in the format, each with a worse alternative:

- **The digest covers the rows and not the file.** `generated_at` moves on
  every write, so hashing the whole document would make an unchanged log look
  edited at every republish -- and a record nobody can tell has changed is not
  evidence.
- **The rows are canonicalised to tab-separated text before hashing**, not to
  JSON. Two JSON writers can agree about a document and disagree about its
  bytes, over spacing or escaping, and the digest would then depend on which
  one ran.
- **The write goes through a temporary and a rename.** A publisher may be
  reading at any moment, and half a document is worse than a stale one: it
  parses up to the truncation and then does not.

Checked both ways -- a claim that the digest does not move when only the clock
does, and does move when a share arrives -- and cross-checked against Python's
`hashlib` on a real file: an empty log digests to `e3b0c442...`, which is
sha256 of nothing, computed here by the kernel's own sha256.

**It is not a Merkle root and does not claim to be.** A distributor needs a
tree whose leaves are per-address payouts and whose proofs a contract can
verify. This is a flat digest over a tally: it fixes the published record to a
value now, and gives the tree something to be checked against when it is built.

### What is not secured yet

The protocol is **plaintext**, and over the public internet that is a real
exposure rather than a theoretical one: worker names travel in the clear, and
anything between the miner and the pool can rewrite a job. The kernel has one
TLS session and the updater owns it -- `mine pool` refuses `stratum+tls://` by
name for that reason -- so a TLS miner needs `tls.rs` to stop being welded to
the single-connection API. Until then a miner on an untrusted network is
trusting the network, and that has to be said before anybody points a machine
at this rather than after.

## What this pool is for, which is not what a pool is normally for

**Miners are paid in $GLADOS, never in what their hashrate produced.** The pool
is an auto-exchange: hashrate arrives, the pool mines whatever is most profitable
at the time, the proceeds are exchanged by the cheapest route available, and that
becomes a **market buy of $GLADOS** which is what reaches the miners. It is more
complex than an ordinary pool for exactly that reason, and every other decision
in this file follows from it.

**So the upstream payout address is the operator's, and that is structural rather
than a convenience.** Proceeds have to arrive in one place to be exchanged once
and spent on one buy; a per-miner address at the upstream would scatter them
across accounts that can never be aggregated. `pool/src/pool.rs`'s `Forward`
therefore carries no worker name and `upstream.rs` submits every share under the
one configured `user` -- **correct for this design**, and not the defect it looks
like to somebody who read the paragraph this one replaced.

**What that paragraph said, and why it is gone.** It read: "Layer 1, mining,
holds nothing. Each miner's *own* address is the payout address at the upstream
pool, so proceeds never pass through anything this server controls. That is the
P2Pool shape." Every clause of that is the opposite of an auto-exchange, and it
cost a session: it was read as the current design, the implementation was
measured against it, and `Forward` lacking a worker name was written up as an
architectural gap rather than as the design working. A stale paragraph that
describes a coherent *different* system is worse than an absent one, because it
survives being checked.

Where a miner's own address does matter is **layer 2**, and only there. The
Supabase `/worker/map` roster maps a worker name to an EVM address and
`GladosDistributor` on chain 4663 is what pays them. Two addresses exist in this
system and they belong to different people at different ends of it: the
operator's, at the upstream, receiving coins; and each miner's, on 4663,
receiving the token.

Layer 2, the GLADOS reward, is the operator's fee revenue, which is the
operator's own money, converted and deposited into a Merkle distributor that
miners claim from. The 1,000,000 GLADOS gate is a `require` in the claim
contract rather than a check on this server, for the reason the token page
already gives about kernel-side gating: a check in software the holder can
rebuild is not a check.

**So the share log is the whole product of Layer 1**, and it has to be
published rather than merely kept. Per-worker share counts, each epoch's
Merkle root, every conversion transaction hash. With no treasury and no track
record, that is the only thing standing in for trust.

## What exists now

- `pool/src/lib.rs`, the shared core: twelve of the kernel's own files reached
  by `#[path]`, and the vectors under `cargo test`. Seventeen tests, 0.12 s.
- `src/mine/proto.rs`, the codec, shared by both ends for the same reason.
- `pool/src/pool.rs`, the coin table and share validation, with no socket
  anywhere in it.
- `pool/src/server.rs`, the listener.
- `glados-pool --selftest`, which is the whole path in one process:

```
[pool] listening on 127.0.0.1:63871
[pool] slot 0  selftest-a  sha256d  (local)
[pool] slot 1  selftest-b  blake2s (RFC 7693)  (local)
[pool] 127.0.0.1:63872 hello  worker=selftest.rig agent=glados-pool/selftest
[pool] selftest.rig share job=00000002 nonce=00002a63 -> accepted
[pool] selftest.rig share job=00000001 nonce=000001f7 -> accepted
ok    greeted, mined 2 coins (blake2s + sha256d), and every share was accepted
```

**Two coins is the test and one would not have been.** A pool serving two
coins down one connection while validating both under one algorithm passes a
single-coin check perfectly, and that is precisely the bug this protocol exists
to make impossible.

## And the kernel actually does it

`mine pool <host>:<port> glados` picks the dialect; the word is trailing rather
than a scheme, because a scheme would have to be invented while
`stratum+tcp://` is one people already paste from a pool's own page. Measured,
QEMU against the pool running on the host:

```
glados> mine coins
  slot  label       slices  source   rate            algorithm
  0     btc-ish     1       pool     248884 H/s      sha256d
        5 share(s) found
  1     verge-ish   1       pool     5506646 H/s     blake2s (RFC 7693)
        140 share(s) found
  2     zeny-ish    1       pool     342 H/s         yespower 1.0 N=2048 r=8
        6 share(s) found
```

```
[ledger] gl4d0s.rig1  btc-ish     5 accepted, 0 stale, 0 bad, 0 dup
[ledger] gl4d0s.rig1  verge-ish 166 accepted, 0 stale, 0 bad, 0 dup
[ledger] gl4d0s.rig1  zeny-ish    9 accepted, 0 stale, 0 bad, 0 dup
```

181 shares, every one recomputed at the other end and agreed with, across three
algorithms. **The three coins came from one TCP connection and one socket
task**, which is the thing four Stratum connections could not have done inside
`MAX_TASKS`.

**One bug, and it cost the first end-to-end run.** `stratum::classify` hands
back the whole message rather than the `result` field -- `subscribe_result`
unwraps it too, and the greeting did not. The pool logged the hello and logged
its answer; the kernel reported that nothing had answered. Two logs that both
look correct and disagree about whether a message arrived is the shape to
remember.

**`mine ev` had to learn to say less.** A glados job carries no nbits and no
coinbase, so the block printed `nbits 00000000` and a failed parse of a
zero-byte coinbase: three lines that read as three things being broken. It
names the absence now, which is `src/linux/proc.rs`'s rule -- a field this
machine does not know means the answer does not exist.

## The upstream, which closes the loop

`pool/src/upstream.rs`, one thread per coin that has one. A coin spec grows an
optional suffix and that is the whole of the configuration:

```
bitzeny:yespower-10-2048-8:16@stratum.example.com:3333,waLLet.rig1,x
```

`@` rather than another `:`, because a `host:port` already contains one and a
positional parser would have to count colons from the right -- which breaks the
first time somebody omits the port.

**It is not a port of the kernel's client. It is the kernel's client.**
`mine::stratum` is included by `#[path]` like everything else, so the bytes
this puts on the wire and the bytes the kernel would put there are built by one
function each. `design/mining.md` predicted the move and it turned out not to
be a move at all.

### Two targets, and the difference is the whole of being a proxy

Ours decides what a miner is **credited** for and is set low enough that a
laptop reports in every few seconds. Upstream's decides what is worth
**sending**, and most accepted shares do not meet it. One number for both would
either flood upstream with work it rejects or leave a miner silent for hours.
So `Coin` carries a share target from the config and `Work` carries upstream's
from `set_difficulty`, and `submit` queues a forward only when a share beats
both.

`set_difficulty` goes through `stratum::decimal` and never `as_i64`, which is
the trap that module exists to document: altcoin pools routinely send a
fractional difficulty, `as_i64` reads `0.001` as `0`, and a target built from
zero accepts everything. The end-to-end run below deliberately used 0.0005 for
that reason.

### Measured, the whole chain

Kernel in QEMU, pool on the host, `tools/stratumstub.py` upstream with a
three-level merkle branch and a fractional difficulty:

```
[up chain] subscribed and authorized, extranonce1 4 bytes, extranonce2 4
[up chain] job job1, 3 merkle level(s), target 000007cf..

  slot  label   slices  source   rate          algorithm
  0     chain   2       pool     652033 H/s    sha256d
        1031 share(s) found

[up chain] forwarding a share for job job1      x5
[up chain] submit accepted                      x5
```

1,031 shares credited at our 14-bit target, of which 5 beat upstream's and were
sent. **The stub verified all five and refused none** -- and that is the result
rather than the share count.

### The stub became an oracle, which is where the value is

`--verify` rebuilds each submitted header *in Python* from what it sent plus
what came back: coinbase from `coinb1 || extranonce1 || extranonce2 || coinb2`,
the merkle fold over its own branch, the prevhash word swap, ntime and nonce
reversed out of their big-endian submit form. Then it hashes and checks its own
difficulty. It shares no code with the Rust.

That is the `tokenizer.py --verify` bargain applied to a wire protocol, and it
covers a stretch nothing else did. **`--branch` matters most.** The stub sent an
empty merkle branch before this, so the fold loop had never executed against
real data outside a one-element synthetic case -- `mine::probe` exists in the
kernel precisely because that gap was known and could not be closed without a
live pool. Three levels closes it here, repeatably, offline.

Also newly exercised end to end: extranonce2 generation and its round-trip
through a submit, and the fractional-difficulty path.

## A difficulty per miner per coin

A fixed share target is wrong for everybody the moment the miners are not
identical, and Pis and phones beside a laptop is the stated point. `vardiff.rs`
moves each connection, **per coin**, which an ordinary pool does not need: one
pool serving one coin needs one difficulty per connection, while here a single
machine works several algorithms and its rate across them differs by three
orders of magnitude.

Measured, both coins started at 14 bits on purpose so any divergence is the
retargeter and nothing else, one connection, one kernel:

    slot 0  fast  sha256d                 158289 H/s   14 -> 17 -> 20 -> 21
    slot 1  slow  yespower 1.0 N=2048 r=8     180 H/s   14 -> 12

Opposite directions from one starting point, driven only by measured rate.

Everything is leading zero bits, so a step is exactly a halving and the
arithmetic is integer throughout. A difficulty as a float is what
`stratum::decimal` exists to survive on the way in; there is no reason to
introduce one on the way out.

**The test found the bug that mattered most.** The window was measured in whole
seconds and returned early on an elapsed zero -- but eight shares inside one
second is the most extreme flood there is, so the fastest miners, the entire
reason the file exists, were the one case that never retargeted.

### And a soak found the other end of it: a connection can be too short

VarDiff needs either eight accepted shares or sixty seconds of idle before it
moves. So **it cannot converge on a connection shorter than sixty seconds**,
and its state was per connection, so a miner that reconnects often restarted at
the operator's guess every single time.

Measured, two profiles of the same Python miner against the same pool for the
same four hours, differing only in how long a connection lasts:

    soak-long   30-minute connections   24 -> 12 -> 10 -> 8 bits, shares credited
    soak-churn  45-second connections   19 consecutive cycles, 0 shares accepted

Nothing about the churn miner was wrong. It hashed, it was answered, it was
asked for 24 bits every time and 24 bits is four thousand times more work than
it could do in a cycle. A flaky link or a phone that sleeps its radio *is* that
miner, which makes this the device class the pool was built for rather than a
corner case.

`Pool::converged` remembers where each `(worker, slot)` settled and `hello`
resumes from there. A worker name is unauthenticated and does not need to be:
credit is denominated in work, so claiming somebody else's easier start pays
proportionally less per share and buys nothing, and claiming a harder one only
hurts the claimant.

### Every bound driven past, and what that turned up

`tools/poolabuse.py` goes past each limit in `server.rs` against a throwaway
instance on its own port. Driven on the host the pool runs on:

    MAX_BAD              33 answered, then dropped        exact
    MAX_SUBMITS_PER_SEC  60 offered in 1 ms, 20 answered, connection still open
    MAX_LINE             131,158 bytes with no newline, stream closed
    MAX_CONNECTIONS      300 attempted, 256 welcomed, 44 refused, 0 errors
    budget at 1%         1,540 offered, 323 answered, 0.01 cores
    budget at 25%        3,620 offered, 1,469 answered, 0.23 cores

Every ceiling held to the share, and a connection turned out to cost one
thread, one descriptor and 28 KiB -- 258 threads and 8,184 KiB at the ceiling,
back to 2 and 908 KiB afterwards.

**What the limits could not catch is the thing worth recording.** The tally is
keyed by `(worker, coin)` and a worker name is whatever a stranger types, so
sixty seconds of ordinary flooding with a fresh name per connection left 248
records, 104 KiB resident and 31 KiB of ledger JSON -- which is re-serialised,
rewritten and published every sixty seconds thereafter. At four connections a
second that is fourteen thousand records an hour, and **every one of the four
connection limits held perfectly while it happened.** It is the same shape as
the hole `budget.rs` was written to close: each bound correct, and the quantity
they were collectively bounding not bounded at all.

**And a fresh name per connection was the slow way in.** `hello` could be
re-sent on a live connection and simply overwrote the worker name, so one
socket could be as many miners as it liked -- measured at **nineteen names a
second**, the per-connection submit cap, from a single connection. Each name
became a permanent tally record, and each cost the pool *no validation at all*:
a submit naming a job the pool does not hold answers `stale` before anything is
hashed, so the validation budget, the one bound that is pool-wide, never sees
any of it. Six thousand names in five minutes from one socket took the process
from 836 KB to 1,824 KB resident and the published ledger to 641 KB, which is
then re-serialised and rewritten every sixty seconds. At the connection ceiling
that is about forty-nine hundred names a second.

One worker per connection now. Re-greeting under the same name stays legal and
is a no-op that re-issues work, because a client retrying its handshake after a
timeout is doing something reasonable.

**Two fixes, and it is worth being exact about what each one buys**, because
the cap alone looked better in the argument than it measured. The same six
thousand names, same host, one connection:

    unbounded   836 -> 1,824 KiB resident   641 KiB of published ledger
    capped      740 -> 1,588 KiB resident   204 KiB

Bounded rather than cheap. `retain` clears the map when it fills and it refills,
so the ledger is capped at 4,096 rows -- about half a megabyte -- however many
names arrive, where before it was linear in them and 60,000 names would have
been 6.4 MB. Resident memory does not come back down either: the allocator
keeps the high-water mark of having held them. The cap turns unbounded growth
into a ceiling, which is the whole of what a cap can do.

What actually removes the cost is the handshake rule, which takes the rate from
nineteen names a second per connection to one *ever*. Two thousand attempts
against the same host: **one distinct worker reached the tally, 1,920 renames
refused, 852 KiB resident** -- against 1,588 with the cap alone and 1,824 with
neither.

The cap is not a plain ceiling, because a plain ceiling is the same denial of
service one step later -- the flood simply arrives first and the real miner is
refused. Room is made by dropping records with **zero credited work**, and
never one with work in it: work required hashes that met a target, so it cannot
be manufactured cheaply, which turns the cap into a proof-of-work admission
rule. A flood of invented names can fill the map and can never displace a miner
who has done something. Both directions are claims, including the branch where
the map is genuinely full of paid workers and the pool refuses rather than
evicting one of them -- a refusal path that has never run is a bound that
panics the first time it matters.

## A restart no longer starts from zero

`--ledger` reads the file back before anything can add to it, which matters
because on a machine that is not ours a reboot and a crash-restart are both
ordinary rather than exceptional -- and `run-pool.sh` restarts on exit by
design.

Only the tallies come back. The coins come from the command line, which is
authoritative: a file that disagreed about which algorithm a label means would
silently change what the pool serves, and the operator would be reading their
own config to find out why.

Measured: four accepted shares, restart, `resumed 1 row(s)`, two more mined,
seven in the file. And a doctored one:

```
[pool] bad.json could not be read (digest ec322564... does not match the rows
       (956602d2...)); starting from zero
```

Loud, and it carries on with an empty tally rather than refusing to start. A
pool that will not run because its history is unreadable helps nobody; one that
starts quietly and silently forgets is what has to be avoided.

**What the digest can and cannot catch.** It is over the rows, so a truncated
or corrupted file is refused whole rather than half-loaded -- half a record is
worse than none, since the counts would be wrong in a way nothing downstream
could detect. It is **not a signature** and proves nothing about who wrote the
file: anybody who can edit it can recompute it. That is acceptable because this
is the operator's own record on the operator's own disk, and it is written down
because a digest is easy to mistake for more than it is.

## Driving it without a kernel

`tools/poolclient.py` speaks the protocol and mines sha256d or blake2s with
`hashlib`. Booting the kernel under QEMU to check a change in the pool costs
three minutes; this costs a second, and the kernel run stays the thing that
settles anything about the kernel.

It is also a **third implementation** of the protocol, which matters more than
the convenience: `src/mine/proto.rs` is shared by the kernel and the pool, so
without something written separately the encoder and the decoder are the same
code agreeing with itself.

It earned that immediately. Every share it found was refused, and the pool was
right: `below_target` reads a digest **little-endian** -- a block hash is a
256-bit integer stored least-significant byte first, which is why a Bitcoin
block id is displayed reversed -- and the client compared big-endian. Not close
to right; it accepts and rejects an unrelated set of shares.

**And that led to a test in this repository that passed for the wrong reason.**
`a_target_comparison_is_a_real_comparison` built its digests big-endian too, so
"equal to the target" was a number vastly below it and "one over" vastly above:
three passing assertions, none of which touched the boundary they were named
for. A comparison that only ever sees values orders of magnitude apart passes
with almost any implementation, including the leading-zero count it exists to
rule out. It is now genuinely at the boundary, plus a pair that are each other
reversed -- 1 and 2^248 -- which no order-ignoring comparison can answer the
same way.

yespower is deliberately absent from the client. A Python transliteration would
be a fourth implementation of the one algorithm this tree is most careful
about, and `tools/yespower.py` already exists and is checked against upstream's
own vectors. A job it cannot compute is skipped and said out loud.

## Whether the coin can be sold at all, said before the listener opens

    glados-pool --prices out/prices.json btc:sha256d:24:bitcoin xvg:blake2s:24:verge

    [pool] prices from out/prices.json (0.1 h old)
    [pool] btc: $78141.29 at $30017587394 of 24h volume
    [pool] xvg: $0.00264568 at $6058540 of 24h volume (the sources are 2.5% apart; the lower is quoted)
    [pool] zeny: cannot be quoted -- last priced 1493 days ago
    [pool] nope: 'nope' is not in the price file, so nothing here knows what it is worth

`tools/prices.py` writes the file from two independent sources and `market.rs`
reads it. **Two parsers over one format**, the bargain `tokenizer.py --verify`
makes: the writer is Python and `json.dumps`, the reader is this tree's own
`Json`, and both ends assert the same invariant -- an unusable coin carries its
reason, a usable one is not also refused. A property one end checks is a
property the other can drift away from.

The coin spec gained a fourth positional field, the traded asset, defaulting to
the label. A label is the operator's shorthand and a price file has to be keyed
by something two sources agree on, so `btc` is findable as `bitcoin` without
anybody having to rename their coins.

**It never refuses to start.** Serving a coin nobody trades is a decision an
operator is allowed to make -- a testnet, a chain they believe in, a market
that has not opened -- and a daemon that would not run without a fresh price
file is one more thing to go wrong at three in the morning on somebody else's
machine. What it will not do is stay quiet about it.

`as_i64` is the wrong reader for a price for exactly the reason
`stratum::decimal` exists one file over: it splits at the `.` and answers the
integer part, so every coin in the table except Bitcoin and Monero reads as
zero. A price of zero is not an error, it is a coin the pool ranks last
forever. `market.rs` takes the `Json::Num` token text, and a claim says so.

**There is no expected value here yet, and the second reason is the
interesting one.** The arithmetic is `price x reward / (2^256 / network
target)`; the pool holds the network target from a live `nbits` and the reward
from `ev::coinbase_value`, so it is a few lines. What it does not hold is how
many base units make a coin -- a coinbase output is in the chain's own unit and
that constant is not on the wire. Writing 1e8 because Bitcoin uses it is the
invented figure `ev.rs` refuses in its own header. Both blockers lift together:
when there is a live upstream there is also a chain to read it from.

## What now does

- **PPLNS.** `pool/src/pool.rs` carries a `Window` per coin, denominated in
  work rather than shares for the reason the tally already is -- VarDiff makes
  a share meaningless as a unit. It is persisted share by share and folded into
  the published digest, so the number a payout comes from is attested rather
  than only the all-time tally. `--window` sets it. The one thing it does not
  do is fairness under load, which is written into the module.
- **A total validation budget.** `pool/src/budget.rs`, `--cpu-percent`. The
  per-connection rate limit never summed -- 256 connections at 20 yespower
  submits a second is ninety-seven cores on a four-thread box -- and this is
  the aggregate bound, in microseconds of measured validation. Found by
  deploying to a borrowed 2012 i3 and reading its own bench: a yespower share
  is 19 ms there, not the 7.5 ms the old comment assumed.
- **It has run on real hardware over WireGuard.** A cross-compiled static musl
  binary, no toolchain on the server, two algorithms mined into it from the
  GF63 across the tunnel: 36 shares accepted, 0 bad, the box unmoved. Not yet
  reachable from the public internet -- that needs a forwarded port -- and not
  yet left running, which needs the operator's `enable-linger`.

## The window had no units, and now that it does the number is wrong

`Coin::network_target` has existed since the struct did. `U256::from_nbits` has
existed, with a claim, since `u256` did. **Nothing ever joined them**, so three
separate things sat blocked on a number that was arriving in every single
`mining.notify`: expected value per coin, the payout window having any unit at
all, and telling this PPLNS from the hopping-proof one. One line in `set_work`.

The unit is Rosenfeld's. Reward variance goes as `pB^2/N` and mean time to
payment as `pN/2`, so their **product is fixed whatever `N` is** -- the window
is a dial between paying smoothly and paying soon rather than an optimisation
with a right answer, and `--window` being the operator's number is correct. What
was missing is any way for them to know which end they had picked. A block is
`2^256 / target` expected hashes and the window is already denominated in
expected hashes, so the ratio is the whole of it.

**And the deployed setting is a rounding error.** `run-pool.sh` passes
`--window 268435456`, which against Bitcoin's own `nbits` from that live
`mining.notify` is under `1e-12` of one block's work. That is not a window that
pays smoothly or a window that pays soon; it is a window that pays the last few
shares and nothing else. The figure looked perfectly large as a bare integer,
which is exactly the failure a missing unit produces, and it went unnoticed for
as long as every coin was local and there was nothing to compare against.

The claim checks both halves: a difficulty-1 target with a `2^32` window is one
block of work by construction, halving the window halves the answer -- which is
what separates a real ratio from a constant that happens to be 1 -- and then
Bitcoin's real `nbits` says what the shipped configuration is worth. A coin with
no chain behind it answers nothing rather than a plausible number.

## It has met a real chain

The largest thing on the list below was "no upstream on the internet has been
asked for work". That is done, and it went further than a connectivity check.

    [up btc] connecting to solo.ckpool.org:3333
    [up btc] subscribed and authorized, extranonce1 4 bytes, extranonce2 8
    [up btc] job 6a72bdc00001adec, 10 merkle level(s), target 00000000..

    00000002 on btc: proof VERIFIES, pays 3.13499712
      3.07229718 to bc1q070w...thvst
      0.06269994 to bc1q28kk...wfzu

solo.ckpool.org because it wants **no account**: the username is a payout
address, which is the only arrangement that can be tested at all without
somebody registering somewhere first. Bounded runs, nothing left connected.

Every line of that is a separate thing working. A real pool's `mining.subscribe`
and `mining.authorize`; a real `mining.notify` with a ten-level merkle branch,
which is the case `mine::probe` exists in the kernel because nothing could
exercise; a downstream job built from it; and `tools/prooftest.py` -- Python,
sharing no code with the Rust -- rebuilding the coinbase from
`coinb1 || extranonce1 || extranonce2 || coinb2`, folding that branch, and
landing on the merkle root inside the 80-byte header it was handed.

**And the second line is the one that matters for a non-custodial pool.** The
value alone never answered the question a miner is actually asking, which is
*whose address*. `prooftest.py` names the outputs now and `--expect-paid` makes
it a check rather than a report. The split is ckpool's own two percent, read off
the chain's data rather than off their website.

The bech32 encoder is written out in that file rather than imported, for the
reason the file exists: it is meant to be runnable by a miner with nothing
installed, before they point hardware at anybody, and a dependency is a reason
not to bother. Checked against BIP-173's own vectors.

**What this still does not prove** is that any of it earns anything. Solo
mining Bitcoin from a laptop is a lottery ticket with no realistic prospect,
and `mine ev` says so on every run. What was under test is the plumbing, and
the plumbing is now the only part that has been demonstrated against something
this project does not control.

## What does not
- **Shares going back upstream.** The forward path exists and is claimed in
  tests, but nothing here has ever produced a share that beat Bitcoin's network
  target, and nothing ever will at this hashrate. That leg is checked against
  `stratumstub.py` and not against a chain, and that is the honest state of it.
- TLS, and therefore any safety on an untrusted network. See above.
- The site repository, the DNS record, and the host. None of them exist yet.
- Worker identity, which is `supabase/functions/link` already and needs joining
  up rather than writing.
- Publishing any of it, which is the whole of B4 and the only thing standing in
  for trust.
- **An expected value per coin**, and therefore weights a miner could derive
  rather than be given. `miner/src/main.rs` shares one device across coins by
  weight and every weight is 1, which is honest and is not a policy. See above
  for the two things missing. `choose`'s doc now says what the weights *should*
  be and cites the first-order condition it comes from, which is a step short
  of setting them.
- **Hopping-proof PPLNS.** What is here is Rosenfeld's simple variant, which is
  hopping-proof only while difficulty is constant. The proof variant needs each
  share's difficulty relative to the network, which is the same missing number
  that blocks expected value. One absent quantity, three consequences.
- **A window sized for a real chain.** The dial has markings now -- see below --
  and what they say about the deployed setting is that it is not a payout
  window at all.
- **A distributed flood.** Every bound here is per connection or per pool and
  none of them is per source. On a home connection the upstream link saturates
  long before the daemon does, and nothing running on the server can help.
