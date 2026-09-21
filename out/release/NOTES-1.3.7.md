# GLaDOS 1.3.7

Forty-two commits. Mostly this release is about giving the model things it did
not have: somewhere to look things up, a radio to talk over, and a language
underneath it that can do exact arithmetic.

## It can look things up now

A searchable knowledge base of concept nodes, indexed in memory, stored on
disk, pulled into the model's context a budget at a time. Small models are
capacity-bound. That is a law rather than a shortcoming, so the answer to "it
does not know enough" is retrieval.

- **Search went from 0.5% to 91.9%** recall@1 over 8,913 nodes. The first
  version used mean-pooled embeddings and scored 0.5%, which is barely above
  the 0.011% chance line. What fixed it was old-fashioned text search: inverse
  document frequency, an inverted index, and a length discount.
- **Two of those came out of `forest why`**, which prints what the scorer
  actually saw. It found that a byte-level BPE spells `what` and ` what` as
  different tokens, so the first word of every query looked rare, and a
  question about roulette outranked one about derivatives for starting with
  "What". It also found that linear IDF cannot suppress a whole set of
  stopwords, and that squaring it can.
- **Bodies live on disk.** `read_blocks` had been finished and unreachable for
  as long as it had existed. An 8,913-node forest that could not reach a prompt
  inside 900 seconds now loads in 39, with 7.7 MB that never enters memory.
- **A fitted router beats the free one.** Cosine 35.5% against a ridge probe at
  42.8% top-1 on 1,782 held-out nodes, and bit-identical across three boots.
- **Retrieval budgets are counted rather than estimated.** Ask for 1500 tokens
  and it renders 1496. Tokenisation is not additive at a boundary, so summing
  per-node counts drifts, always toward admitting one node too many.

## Wireless, all of it except a driver

An 802.11 stack written once for every chip instead of once per chip.
`dev::radio::Radio` is the seam a driver implements. Everything above it is
shared: SNAP, the four address layouts, sequencing, the MLME state machine,
CCMP, and both halves of the four-way handshake.

- **The whole path runs at boot with nothing plugged in**, against a loopback
  radio and a fake access point that authenticates, associates, and runs the
  authenticator's half of the handshake.
- **A remote ring-0 panic, found by fuzzing our own code.** `wpa2::parse` could
  be crashed by anyone in radio range with no credentials, because EAPOL
  crosses an unencrypted link by design. The fuzzer missed it for 9,200 cases
  until it learned to target length fields. It now finds it at case 763.
- **Checked against scapy**, which is an implementation nobody here wrote, over
  104 checks. Self-consistent tests do not earn a bare-metal boot.
- **A network manager window** listing SSID, signal, channel, band, security
  and BSSID.

**There is no driver for the Wi-Fi in this laptop.** The stack is real and
`wlan0` does not exist on hardware. Wired ethernet and USB tethering work.

## Firmware, and waking the discrete GPU

- **The namespace was missing 90 of this laptop's 92 power resources.** The
  parser stepped over a one-argument method call as though it were a leaf, so
  one region's recorded bytes ended a term short.
- **The DSDT turned out to be one table of fifteen.** The GPU's power control
  lives in an SSDT, and the 48 conditionals that looked like the obstacle were
  about WWAN slots.
- **`gpu wake` reads `_PR0` and calls `_ON`**, refusing three ways first:
  locked regions, a foreign namespace, or a method that will not evaluate.
  Untested on hardware.

## Exact arithmetic in the system language

Four rungs: integers, exact rationals, dimensioned quantities, and explicit
approximations. Inexactness is opt-in and every rung is reached through a named
builtin. Seven libraries at `/lib`: probability, geometry, linear algebra,
number theory, polynomials, physics, chemistry.

The headline is `inv(hilb(3))`. The Hilbert matrix is the standard
ill-conditioned example and its inverse is a matrix of whole numbers, which no
floating-point library recovers at any precision. This one does.

Units live in the value, so adding seconds to metres is an error instead of a
wrong answer.

Also: **128-bit division never returns on this target.** One `u128 %` in a gcd
loop stopped boot with no fault and no output, three sections before the shell.

## Benchmark plumbing

The dense evaluation path could not run Qwen3 at all, and three other defects
in the harness meant the arithmetic rail had never once measured a model. All
four are fixed, and `--show` now prints what the model actually returned, which
is what would have made any of them obvious. Numbers from that rail are still
work in progress and none is quoted here.

`tools/fastdense.py` is the runner that makes re-measuring affordable. Prefill
is 92x faster, and the shared few-shot prefix, 658 of a 726-token prompt, is
computed once instead of once per question. It is checked against the numeric
oracle on every run, over a deliberately ragged batch.

A correction to an earlier version of these notes, which claimed the same seed
gives the same answers batched and unbatched. It does, but that was not what
had been measured: the harness never passed the batch size through to the task,
so both sides of that comparison ran one question at a time. The claim is
withdrawn rather than restated, and the runner's own documentation now says the
batching speedup is unmeasured instead of quoting a number for it.

## Fixes worth naming

- The agent loop knew it was going in circles and had never said so.
- A test that poisoned its own canary, so it could only pass once per boot.
  That made `diag all` twice, which is the release gate, report a red line on
  a clean machine.
- Two recorded claims that had stopped being true: a "not fixed" note for a bug
  fixed eleven days earlier, and a withdrawn benchmark row whose blocker no
  longer existed. Documentation is a snapshot. Check the code.

## The download page

Rebuilt around the question people kept asking, which is how to try this
without wiping a laptop. Three routes ranked by what they cost you: VirtualBox,
QEMU, and a USB stick. The two settings every failed first attempt is really
about are called out where they happen: VirtualBox's Enable EFI, and QEMU's
split pflash firmware that `-bios` rejects.

## Not done

- No driver for the wireless chip, so no `wlan0` on real hardware.
- `gpu wake` and the `skip-hwp` repair loop have never run on the target laptop.
- The GPU does nothing for the evaluation runner yet.
- MMLU, NIAH and routing figures are owed a re-run through the fixed harness.
