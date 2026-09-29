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

// Build an epoch. `now` and `prev` are cumulative work per address (Maps);
// `balances` answers GLADOS wei per address (a Map; absent means unknown and
// is refused, since a gate that opens on a failed read opens for anybody).
export function build({ now, prev, balances, gateMin, minWork = 1n, minFrac = 0.25 }) {
  const delta = new Map();
  for (const [a, w] of now) {
    const d = w - (prev.get(a) || 0n);
    if (d > 0n) delta.set(a, d);
  }
  const excluded = {};
  const gated = new Map();
  for (const [a, d] of delta) {
    const b = balances.get(a);
    if (b === undefined) excluded[a] = "balance unreadable";
    else if (b < gateMin) excluded[a] = `holds ${b / 10n ** 18n} of ${gateMin / 10n ** 18n} GLADOS`;
    else gated.set(a, d);
  }
  const floor = [minWork, medianFloor(gated, minFrac)].reduce((a, b) => (a > b ? a : b));
  const recipients = [];
  for (const [a, d] of gated) {
    if (d >= floor) recipients.push(a);
    else excluded[a] = `did ${d} work against a floor of ${floor}`;
  }
  recipients.sort();
  return { recipients, work: Object.fromEntries([...delta].map(([a, d]) => [a, d.toString()])), floor: floor.toString(), excluded, snapshot: now };
}
