// How one epoch's pot becomes one GladosPayout2.payAll call. Pure, so
// plan.test.mjs asserts it with no network, no storage and no coins.
//
//   plan({ recipients, choices, blocked, fallback, value }) -> { groups, eth }
//
// **Every wallet gets the same share of the pot, whatever it chose.** The pot
// is split equally over the recipients first, and only then grouped by choice:
// a group of 7 gets 7 shares, spent on its legs and divided between its 7.
// Choosing a basket changes what a share buys, never how big it is.
//
// **A choice that cannot be honoured becomes $GLADOS, never a failed payout.**
// Three ways it can't, each a reason a stock leg would make `payAll` revert for
// *everybody* in the epoch:
//   - `blocked`: wallets the stock tokens' beacon blocklists (design/rwa.md);
//     a transfer to one reverts.
//   - `fallback`: choices whose quote failed this epoch (a pool drained, a
//     token paused); a leg with no quote has no safe `minOut`.
//   - anything not on the menu (rewards.js already maps it to $GLADOS).
// So the worst case of every stock problem is that some miners are paid in
// $GLADOS that epoch, which is what they got before choices existed.
import { MENU, DEFAULT, rewardOf } from "./rewards.js";

// How many token legs a choice costs per wallet: what the gas plan counts.
export function legsOf(code) {
  return MENU[code] && MENU[code].tokens.length ? MENU[code].tokens.length : 1;
}

// The choice a wallet is actually paid in this epoch.
export function effective(addr, choices, blocked = new Set(), fallback = new Set()) {
  const c = rewardOf(choices && choices.get ? choices.get(addr) : undefined);
  if (c === DEFAULT || blocked.has(addr) || fallback.has(c)) return DEFAULT;
  return c;
}

export function plan({ recipients, choices = new Map(), blocked = new Set(), fallback = new Set(), value }) {
  const v = BigInt(value);
  const n = BigInt(recipients.length);
  if (n === 0n || v <= 0n) return { groups: [], eth: 0n };
  const per = v / n;

  const by = new Map();
  for (const a of [...recipients].sort()) {
    const c = effective(a, choices, blocked, fallback);
    if (!by.has(c)) by.set(c, []);
    by.get(c).push(a);
  }
  // $GLADOS first, then the menu's own order, so a later reader rebuilds the
  // same call byte for byte.
  const order = Object.keys(MENU).filter((c) => by.has(c));
  const groups = order.map((code) => {
    const to = by.get(code);
    const share = per * BigInt(to.length);
    const syms = code === DEFAULT ? ["GLADOS"] : MENU[code].tokens;
    const k = BigInt(syms.length);
    const legs = syms.map((sym, i) => ({ sym, eth: i < syms.length - 1 ? share / k : share - (share / k) * (k - 1n) }));
    return { code, to, legs };
  });
  // The pot's indivisible remainder rides on the first leg, so the legs spend
  // exactly `value` and payAll's ValueMismatch check holds.
  const spent = groups.reduce((s, g) => s + g.legs.reduce((t, l) => t + l.eth, 0n), 0n);
  groups[0].legs[0].eth += v - spent;
  return { groups, eth: v };
}

// Gas for a plan: a base, plus each leg's swaps, plus a send per wallet per leg.
export function planGas(p, cfg) {
  let g = BigInt(cfg.baseGas);
  for (const grp of p.groups) {
    for (const l of grp.legs) {
      g += BigInt(l.sym === "GLADOS" ? cfg.v2LegGas : cfg.v3LegGas);
      g += BigInt(cfg.perRecipientGas) * BigInt(grp.to.length);
    }
  }
  return g;
}
