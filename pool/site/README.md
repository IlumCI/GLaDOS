# The published share log

One HTML file that reads `ledger.json` and **recomputes its digest in the
browser**. Intended for GitHub Pages at `pool.aperture.institute`.

`design/pool.md` calls publishing "the only thing standing in for trust", and
means it literally: Layer 1 never holds a miner's coins, so there is no wallet
to audit. What a miner has instead is this record and whatever can be checked
against it.

## Why it recomputes rather than displays

A page showing the digest the pool wrote beside the rows the pool wrote proves
nothing -- both halves come from one program, and a program whose
canonicalisation was wrong would agree with itself. So the canonical form is
rebuilt here from the rows and hashed with the browser's own SHA-256.

That makes three implementations of the format: the Rust writer
(`Pool::ledger_json`), the Rust reader (`Pool::load_ledger`), and this. The
bargain `tokenizer.py --verify` makes, for the same reason.

Verified both ways rather than assumed. Against a real ledger the page reports
`DIGEST VERIFIED`; with **one unit of work moved from one worker to another and
the digest left untouched** -- the exact edit somebody would make to steal a
slice of a payout -- it reports `DIGEST MISMATCH` and prints both hashes.

## What the digest is not

**Not a signature.** Anybody who can edit the file can recompute it, and
`ledger_json` says so in its own comment. It fixes the record to a value so two
copies can be compared, and so a later Merkle distributor has something to be
checked against. The page says this in its footer rather than letting a green
tick imply more than it means.

## The `ledger.json` beside this file

**A real ledger from a real run, not a hand-written one.** Two `poolclient.py`
miners against the musl daemon on loopback, twelve sha256d shares interleaved,
a 75/25 split -- and `tools/ledgercheck.py` agrees with every figure in it,
digest included. It is here so the page can be opened and looked at with
something in it.

It replaced a sample whose window was **2^28 against a single 2^30 share**, so
the work inside the window was 4.3x the window itself. That is a reachable state
rather than an impossible one -- PPLNS cannot evict below one share, so a share
larger than the whole window overshoots it -- and it is the misconfiguration the
pool warns about *at startup*, in words about paying "the most recent shares and
nothing else". A published ledger carries no startup log, so a reader had no way
to learn it. `ledgercheck.py` prints that as a note now; a note and not a claim,
because refusing the state would call a genuine record invalid.

The old sample also labelled its coin `btc`, which reads as a record of real
Bitcoin mining. The replacement's coin is `probe` and its workers are
`probe-a.rig` and `probe-b.rig`, so nothing about it can be mistaken for an
event.

## Deploying

    cp pool/site/index.html  <pages-repo>/
    cp <state>/ledger.json   <pages-repo>/

**`docs/pool/ledger.json` is deliberately absent**, which is why the published
page reports `Could not read ledger.json: HTTP 404`. `docs/pool/index.html` is
live at <https://glados.aperture.institute/pool/> and byte-identical to the file
beside this one; what is missing is a record worth publishing. Putting the probe
ledger there would put a document on a public page that reads as a live pool with
miners on it, and there is no upstream, no chain and nobody mining. The page's
404 is the honest state and it says so in a sentence rather than looking broken.

The pool writes `ledger.json` through a temporary and renames, so a publisher
copying it never sees half a document.

**`crypto.subtle` needs a secure context**, so verification works over `https`
and on `localhost` and not from a `file://` URL. GitHub Pages is https. Opened
from a file the page says `NOT CHECKED` rather than failing quietly -- "this
page cannot tell" and "this record is wrong" are different answers.

Locally:

    cd pool/site && python3 -m http.server 8731 --bind 127.0.0.1

## No build step

One file, no framework. A build step is a thing that can produce a site which
does not match its source, and the entire product here is that the published
thing can be checked -- so the source you can read is the thing that runs.
