// Who an epoch pays: the equal split, as `tools/distribute.py --split equal`
// decides it, over the work done since the last epoch. Pure, so
// `epoch.test.mjs` asserts it with no network and no storage.
//
// The rules, each one there for a reason `distribute.py` records:
//
// - **Per payout address, never per worker.** `0xabc.rig1` and `0xabc.rig2`
//   are one address and one share.
// - **The holding gate, at build time.** The address must hold `gateMin` GLADOS
//   *now*; one balance moved between addresses between connections passes a
//   connection-time gate, but at one instant it is in one address.
// - **A floor of `minFrac` of the work-weighted median address's work.** Under
//   an equal split a payout rewards addresses, so each extra address has to
//   cost burned compute; weighting by work is what stops a hundred sybils
//   doing slivers from dragging the median down to their sliver.
//
// Work is the ledger's cumulative tally; an epoch pays the delta since the
// snapshot the previous epoch recorded, so nobody is paid twice for one share.

// Ledger JSON with every `work` value read as a BigInt. JSON.parse would turn a
// tally past 2^53 into a nearby double -- silently, and the tally only grows.
export function parseLedger(text) {
  const quoted = text.replace(/"work"\s*:\s*(\d+)/g, '"work":"$1"');
  const doc = JSON.parse(quoted);
  return (doc.shares || []).map((r) => ({ worker: r.worker, coin: r.coin, work: BigInt(r.work) }));
}

// The payout address a worker name carries, lowercased, or null.
export function addressOf(worker) {
  const head = String(worker).split(".")[0];
  return /^0x[0-9a-fA-F]{40}$/.test(head) ? head.toLowerCase() : null;
}

// Whether `eth_getCode`'s answer means the address cannot be paid. Any code
// is refused except an EIP-7702 delegation (0xef0100 + an address), which is an
// ordinary key-held account that happens to have a delegate: a miner using a
// smart wallet that way is still a person, and still receives a transfer.
export function hasCode(code) {
  const c = String(code || "0x").toLowerCase();
  return c !== "0x" && c !== "0x0" && !/^0xef0100[0-9a-f]{40}$/.test(c);
}

// Cumulative work per address across every coin and worker.
export function workByAddress(rows) {
  const m = new Map();
  for (const r of rows) {
    const a = addressOf(r.worker);
    if (a) m.set(a, (m.get(a) || 0n) + r.work);
  }
  return m;
}

// `frac` of the work of the address at which half the epoch's work is reached.
export function medianFloor(work, frac) {
  const ws = [...work.values()].filter((w) => w > 0n).sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  if (!ws.length) return 0n;
  const total = ws.reduce((a, b) => a + b, 0n);
  let run = 0n;
  const scale = BigInt(Math.round(frac * 1_000_000));
  for (const w of ws) {
    run += w;
    if (run * 2n >= total) return (w * scale) / 1_000_000n;
  }
  return (ws[ws.length - 1] * scale) / 1_000_000n;
}

// Work since the last epoch, per address: positive deltas only.
export function deltas(now, prev) {
  const delta = new Map();
  for (const [a, w] of now) {
    const d = w - (prev.get(a) || 0n);
    if (d > 0n) delta.set(a, d);
  }
  return delta;
}

// The floor an address's delta must reach to be a share: `minFrac` of the
// work-weighted median over *everybody's* delta, and at least `minWork`.
//
// Over everybody rather than over the gated, because it has to be known
// before any balance is read: at 100,000 miners the treasury reads balances
// only for the addresses that can be paid this epoch, highest work first, and
// it cannot rank them against a floor that depends on the reads. Work
// weighting is what makes this safe -- a crowd of slivers, gated or not,
// barely moves a work-weighted median.
export function floorOf(delta, minWork = 1n, minFrac = 0.25) {
  const f = medianFloor(delta, minFrac);
  return f > minWork ? f : minWork;
}

// Who this epoch could pay, in the order to read their balances: at or over
// the floor, not known ineligible, most work waiting first (ties by address,
// so a later reader reaches the same order).
export function candidates(delta, floor, ineligible = new Set()) {
  return [...delta].filter(([a, d]) => d >= floor && !ineligible.has(a))
    .sort(([a, x], [b, y]) => (y > x ? 1 : y < x ? -1 : a < b ? -1 : 1))
    .map(([a]) => a);
}

// Build an epoch. `now` and `prev` are cumulative work per address (Maps);
// `balances` answers GLADOS wei per address (a Map; absent means unknown and
// is refused, since a gate that opens on a failed read opens for anybody).
// `ineligible` is a Set of addresses that may never be paid: contracts, and
// above all the WETH/GLADOS pair -- paying it is a taxed sell, the payout
// reverts, and one miner naming it as their address would stop every payout
// for everyone, forever (found by the second review, proven on a fork).
// `considered`, when given, is the set of addresses whose balances were
// asked for; anybody else over the floor was simply not reached this epoch,
// and carries, counted rather than listed.
//
// The snapshot is what the next epoch counts from, and it is chosen per
// address so work is neither paid twice nor lost:
//   paid, under the gate, a contract   -> now   (the work is spent)
//   unreadable, under floor, not reached,
//   or past the recipient cap           -> prev  (it carries, and carried
//                                                 work only grows, so it is
//                                                 nearer the front next time)
// and never below `prev`, so a shard dropped from the sum cannot lower it and
// then pay the same work again when the shard returns.
//
// **At most `maxRecipients` are paid in one epoch, and the rest carry.** One
// transaction holds 400 before it nears 4663's 32M per-transaction gas cap.
//
// `updates` is only the addresses whose snapshot moves -- at most the ones
// considered -- which is what gets stored with a payout: at 100,000 miners the
// whole snapshot is megabytes, and writing it into every payout's record would
// pass what one atomic write can hold.
//
// **And at most `capacity` units of `weigh`, when given.** A wallet paid in a
// ten-stock basket costs ten sends where a $GLADOS wallet costs one, so the
// treasury weighs each wallet by its choice's legs and admits, most work
// waiting first, until one transaction's gas is spent. The rest carry like any
// wallet past the cap, nearer the front next time.
export function build({ now, prev, balances, gateMin, minWork = 1n, minFrac = 0.25, ineligible = new Set(), maxRecipients = Infinity, considered = null, weigh = null, capacity = Infinity }) {
  const delta = deltas(now, prev);
  const floor = floorOf(delta, minWork, minFrac);
  const excluded = {};
  const carry = new Set();
  const gated = new Map();
  let notReached = 0;
  for (const [a, d] of delta) {
    const b = balances.get(a);
    if (ineligible.has(a)) excluded[a] = "a contract or the pool's own plumbing, which cannot be paid";
    else if (d < floor) { excluded[a] = `did ${d} work against a floor of ${floor}; it carries to the next epoch`; carry.add(a); }
    else if (considered && !considered.has(a)) { notReached += 1; carry.add(a); }
    else if (b === undefined) { excluded[a] = "balance unreadable; the work carries to the next epoch"; carry.add(a); }
    else if (b < gateMin) excluded[a] = `holds ${b / 10n ** 18n} of ${gateMin / 10n ** 18n} GLADOS`;
    else gated.set(a, d);
  }
  let recipients = [...gated.keys()];
  if (recipients.length > maxRecipients || weigh) {
    recipients.sort((a, b) => (gated.get(b) > gated.get(a) ? 1 : gated.get(b) < gated.get(a) ? -1 : a < b ? -1 : 1));
    let used = 0;
    let fit = 0;
    for (const a of recipients) {
      const w = weigh ? weigh(a) : 0;
      if (fit >= maxRecipients || used + w > capacity) break;
      used += w;
      fit += 1;
    }
    for (const a of recipients.splice(fit)) {
      excluded[a] = `one of more eligible this epoch than one payout can carry; the work carries and is nearer the front next time`;
      carry.add(a);
    }
  }
  recipients.sort();
  const updates = new Map();
  for (const [a, w] of now) {
    if (carry.has(a)) continue;
    const p = prev.get(a) || 0n;
    if (w > p) updates.set(a, w);
  }
  const snapshot = new Map(prev);
  for (const [a, w] of updates) snapshot.set(a, w);
  const work = {};
  for (const a of [...recipients, ...Object.keys(excluded)]) work[a] = delta.get(a).toString();
  return { recipients, work, floor: floor.toString(), excluded, notReached, updates, snapshot };
}
