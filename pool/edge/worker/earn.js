// The pure parts of the live revenue estimate, so `earn.test.mjs` checks them
// with no network. The figure is one recipient's equal share of the pool's
// distributable pot, valued in the asset they are paid in -- a display number
// pushed as glados.earn, which moves no money. The asset conversion (the
// GLADOS pair, QuoterV2) needs the chain and stays in runner.js; only the
// arithmetic lives here.

// The ETH one recipient's equal share comes to: the treasury's balance, less
// twice what the payout's gas would cost, split by the recipient count, and
// never negative. Twice the gas is the same headroom the payout forecast keeps
// (runner.js), so the estimate never shows value the payout would spend.
export function perRecipientEth({ ethWei, recipients, baseGas, perRecipientGas, gasPrice }) {
  const n = BigInt(Math.max(1, recipients | 0));
  const gas = (BigInt(baseGas) + BigInt(perRecipientGas) * n) * BigInt(gasPrice) * 2n;
  const net = BigInt(ethWei) > gas ? BigInt(ethWei) - gas : 0n;
  return net / n;
}

// A wallet's effective reward: its choice only while it holds the gate minimum,
// otherwise $GLaDOS -- the same rule the payout applies, so the figure shows
// the asset the wallet would actually be paid in.
export function effectiveReward(choice, heldWei, gateMinWei, fallback = "glados") {
  return BigInt(heldWei || 0) >= BigInt(gateMinWei || 0) ? (choice || fallback) : fallback;
}

// A raw 18-decimal token amount as a display number. $GLaDOS and the Robinhood
// stock tokens are all 18-decimal ERC-20s; the division loses only digits far
// below anything a page would show.
export const fromWei18 = (raw) => Number(BigInt(raw)) / 1e18;
