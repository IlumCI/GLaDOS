// When an epoch is worth paying out. Pure, so `cadence.test.mjs` asserts it
// with no network, no storage and no money.
//
// **Two clocks, and the slower one decides.** The swap has a floor (ChangeNOW
// will not take less than ~151 RVN), and the payout has a cost: one market buy
// plus one batch send at ~32k gas a recipient (measured, contracts/test/
// batch.mjs). An epoch whose costs are a large slice of what it pays out is
// giving the miners' money to the chain, so it waits until the costs are at
// most `maxOverhead` of the pot.
//
// Under an equal split this is what makes the pool scale: the cost per miner is
// fixed, the pot grows with the number of miners, so a bigger pool pays out
// *more often* at the same overhead -- daily at ten miners, about hourly at a
// thousand -- rather than every miner waiting for a claim worth its own gas.
//
//   potUsd          what the epoch would pay out, in dollars
//   rvn, rvnMin     what the pool holds, and the swap's minimum
//   recipients      addresses in the epoch
//   perRecipientUsd batch gas per recipient, in dollars
//   fixedUsd        the buy and the batch transaction's base cost, in dollars
export function worthPaying({ potUsd, rvn, rvnMin, recipients, perRecipientUsd, fixedUsd, maxOverhead = 0.1 }) {
  if (!(recipients > 0)) return { go: false, why: "nobody qualifies this epoch" };
  if (!(rvn >= rvnMin)) return { go: false, why: `${rvn} RVN is under the swap minimum of ${rvnMin}` };
  const cost = recipients * perRecipientUsd + fixedUsd;
  if (!(potUsd > 0) || cost > maxOverhead * potUsd) {
    const need = cost / maxOverhead;
    return { go: false, why: `costs $${cost.toFixed(4)} would be over ${maxOverhead * 100}% of $${(potUsd || 0).toFixed(4)}; waiting for $${need.toFixed(4)}` };
  }
  return { go: true, why: `$${potUsd.toFixed(4)} to ${recipients} at $${cost.toFixed(4)} cost (${((100 * cost) / potUsd).toFixed(1)}%)` };
}

// How long until an epoch is worth paying, given the pool's earning rate.
export function hoursUntil({ usdPerDay, recipients, perRecipientUsd, fixedUsd, maxOverhead = 0.1 }) {
  const need = (recipients * perRecipientUsd + fixedUsd) / maxOverhead;
  return usdPerDay > 0 ? (24 * need) / usdPerDay : Infinity;
}
