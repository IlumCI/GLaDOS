// The treasury's next move, as a pure function of what is true right now.
//
//   decide(state, facts, cfg) -> { action, state }
//
// **One action per tick, and the state records it before the world does.**
// Every step that moves money is two-phase: the signed transaction is written
// into `state.pending` *before* it is broadcast, so an object that is evicted
// mid-broadcast wakes, finds the same bytes, and rebroadcasts them -- the same
// transaction, the same hash, never a second one. A payout that could be sent
// twice by a restart is the failure this shape exists to make impossible.
//
// **Nothing here waits for a person.** The operator's rule: an error that
// cannot be fixed is handled automatically, and nobody should come to them
// with an error message. So there is no halt. Every failure is one of:
//
//   retried     the next tick asks again (an answer that did not come)
//   paused      a backoff that ends on its own: 1 h, doubling, at most 24 h
//               (definite failures in a row, which retrying at once repeats)
//   written off recorded as an incident and left behind, and the pipeline
//               carries on with new money (an exchange ChangeNOW failed or
//               sat on for two days; a nonce somebody else used)
//
// and what makes that safe is that none of them can pay twice: the nonce
// discipline in runner.js means a transaction given up on can never be mined
// beside its replacement, and an exchange written off is never funded again.
// Incidents are kept for anybody who looks, at /treasury; they block nothing.
//
// The runner (runner.js) gathers `facts`, calls this, performs the one action,
// and stores the new state. Everything that decides is here and tested in
// `flow.test.mjs` with no network, no storage and no coins.
//
// The loop, once per epoch:
//   RVN accumulates  ->  exchange created  ->  RVN sent to ChangeNOW
//   ->  ETH arrives  ->  (GladosPayout deployed, once)  ->  buyAndPay
import { worthPaying } from "./cadence.js";

export const FRESH = { phase: "idle", exchange: null, pending: null, contract: null, failures: 0, paused: null, backoffs: 0,
                       epochs: 0, incidents: [], orphans: [] };

// Ravencoin's default minimum relay fee is 0.01 RVN per kB; 0.015 leaves margin.
const RVN_SAT_PER_BYTE = 1500n;
const rvnTxBytes = (inputs, outputs) => BigInt(10 + 148 * inputs + 34 * outputs);
export const MAX_RVN_INPUTS = 50;
// Below this a change output is dust the network refuses to relay; it goes to
// the fee instead. 0.01 RVN, well above Ravencoin's dust line at its 0.01 RVN/kB
// relay fee (and the change is a rounding remainder, not money anybody owns).
export const DUST_SATS = 1_000_000n;
// More recipients than this in one payout is a transaction too near 4663's
// per-transaction gas cap; epoch.js pays the 400 with the most work waiting and
// carries the rest, and this trims as a second line if anything ever passes more.
export const MAX_RECIPIENTS = 400;
// Incidents and written-off exchanges kept for anybody who looks.
const KEEP = 20;

export function rvnFee(inputs) {
  return rvnTxBytes(inputs, 1) * RVN_SAT_PER_BYTE;
}

// Pause after `failures` definite failures in a row: an hour, doubling each
// time it happens again without a success between, at most a day.
export function backoffMs(backoffs) {
  return Math.min(24 * 3_600_000, 3_600_000 * 2 ** Math.min(backoffs, 5));
}

export function decide(state, facts, cfg) {
  const s = { ...FRESH, ...state };
  const wait = (why) => ({ action: { kind: "wait", why }, state: s });
  const note = (why) => ({ action: { kind: "note", why }, state: s });
  // Numbered, so the runner can deliver each to the operator exactly once more
  // than zero times: an alert lost to an eviction is sent on the next tick.
  const incident = (why) => {
    s.seq = (s.seq || 0) + 1;
    s.incidents = [...(s.incidents || []), { at: facts.now, seq: s.seq, why }].slice(-KEEP);
  };
  const pause = (why) => {
    const ms = backoffMs(s.backoffs || 0);
    s.paused = { why, until: facts.now + ms };
    s.backoffs = (s.backoffs || 0) + 1;
    s.failures = 0;
    incident(`${why}; paused ${Math.round(ms / 60_000)} min, then it tries again`);
  };

  if (s.paused) {
    if (facts.now < s.paused.until) return wait(`paused until ${new Date(s.paused.until).toISOString()}: ${s.paused.why}`);
    s.paused = null;
  }

  // 1. A transaction in flight settles before anything else happens.
  if (s.pending) {
    const p = s.pending;
    const r = facts.receipt;
    // The chain says this transaction's nonce went to something that is none
    // of ours. It can never be mined now, so it is let go -- nothing it could
    // have done can still happen -- and the next transaction takes the next
    // nonce. What used the nonce is an incident, for whoever reads them.
    if (r && r.lost) {
      s.pending = null;
      incident(r.lost);
      return note(`${p.kind} ${p.hash} can no longer be mined: ${r.lost}`);
    }
    if (r === undefined || r === null) {
      if (facts.now - p.at > cfg.dropAfterMs) {
        return { action: { kind: "rebroadcast", chain: p.chain, raw: p.raw, why: "not mined yet; the same bytes again" }, state: s };
      }
      return wait(`waiting on ${p.kind} ${p.hash}`);
    }
    if (r.ok) {
      s.pending = null;
      s.failures = 0;
      s.backoffs = 0;
      if (p.kind === "deploy") s.contract = p.contract;
      if (p.kind === "rvn") s.exchange = { ...s.exchange, sent: p.hash };
      if (p.kind === "pay") {
        s.epochs += 1;
        s.phase = "idle";
        return { action: { kind: "record_epoch", epoch: p.epoch, hash: p.hash }, state: s };
      }
      return note(`${p.kind} ${p.hash} confirmed`);
    }
    s.pending = null;
    s.failures = (s.failures || 0) + 1;
    if (s.failures >= cfg.maxFailures) pause(`${s.failures} failed transactions in a row, last ${p.kind} ${p.hash}`);
    return note(`${p.kind} ${p.hash} failed (${s.failures || cfg.maxFailures})`);
  }

  // 2. An exchange in progress: send the RVN once, then wait for the ETH.
  if (s.exchange) {
    const x = s.exchange;
    const age = facts.now - (x.created ?? facts.now);
    // Never funded in two hours: ChangeNOW lets an unfunded exchange lapse, and
    // sending to a lapsed deposit address is money with nobody expecting it.
    if (!x.sent && age > cfg.exchangeUnsentMs) {
      s.exchange = null;
      return note(`exchange ${x.id} was never funded in time; dropped, a new one will be made`);
    }
    // Funded and not finished in two days, or failed outright: ChangeNOW is
    // holding it. It is written off -- kept among the orphans, whose status is
    // still read, so if it finishes the ETH simply arrives and if it refunds
    // the RVN simply comes back -- and the next RVN goes into a new exchange.
    const st = facts.exchangeStatus;
    const stuck = x.sent && age > cfg.exchangeStuckMs && st !== "finished" && st !== "refunded";
    if (st === "failed" || stuck) {
      s.orphans = [...(s.orphans || []), { id: x.id, sats: x.sats, sent: x.sent, at: facts.now, status: st || "unknown" }].slice(-KEEP);
      s.exchange = null;
      incident(`exchange ${x.id} ${st === "failed" ? "failed at ChangeNOW" : `sat at ChangeNOW for over ${Math.round(cfg.exchangeStuckMs / 3_600_000)} h`}; written off, still watched, and the treasury carries on`);
      return note(`exchange ${x.id} written off (${st || "unknown"})`);
    }
    if (!x.sent) {
      // **Exactly the coins the exchange was sized from**, recorded when it was
      // made. Choosing again at send time is how a changed wallet -- a new
      // payout, dust somebody sent -- turns into an amount the exchange did not
      // expect.
      const have = new Map((facts.rvnUtxos || []).map((u) => [`${u.txid}:${u.vout}`, u]));
      const utxos = (x.outpoints || []).map((o) => have.get(o));
      if (!x.outpoints || !x.outpoints.length || utxos.some((u) => !u)) {
        // Its coins are gone and nothing here spent them. Nothing was sent to
        // the exchange, so dropping it costs nothing; the coins that remain
        // go into the next one.
        s.exchange = null;
        incident(`exchange ${x.id}'s coins were spent by something that is not the treasury; dropped before anything was sent`);
        return note(`exchange ${x.id} dropped: its coins are gone`);
      }
      const total = utxos.reduce((a, u) => a + BigInt(u.sats), 0n);
      const fee = rvnFee(utxos.length);
      if (total - fee < BigInt(x.sats)) return wait(`the exchange wants ${x.sats} sats and its coins make ${total - fee}`);
      let change = total - fee - BigInt(x.sats);
      let spentFee = fee;
      if (change < DUST_SATS) {
        spentFee += change;
        change = 0n;
      }
      return { action: { kind: "send_rvn", utxos, to: x.payin, sats: BigInt(x.sats), fee: spentFee, change }, state: s };
    }
    if (st === "finished") {
      s.exchange = null;
      s.phase = "funded";
      return note(`exchange ${x.id} finished`);
    }
    if (st === "refunded") {
      s.exchange = null;
      return note(`exchange ${x.id} refunded; the RVN is back in the pool's wallet`);
    }
    return wait(`exchange ${x.id} is ${st || "unknown"}`);
  }

  const gp = BigInt(facts.gasPrice || 0);
  const eth = BigInt(facts.ethWei || 0);

  // 3. The payout contract, deployed once, from the first ETH that arrives.
  if (!s.contract) {
    const cost = gp * BigInt(cfg.deployGas);
    if (eth > cost * 2n) return { action: { kind: "deploy", gas: cfg.deployGas }, state: s };
  }

  // 4. Pay, when there is somebody to pay and it is worth the gas.
  const recipients = (facts.recipients || []).slice(0, MAX_RECIPIENTS);
  const n = recipients.length;
  if (s.contract && n > 0) {
    const gas = BigInt(cfg.baseGas) + BigInt(cfg.perRecipientGas) * BigInt(n);
    const reserve = gas * gp * 2n; // this payout's gas, twice over
    let value = eth - reserve;
    if (value > 0n) {
      const verdict = worthPaying({
        potUsd: Number(value), rvn: 1, rvnMin: 0, recipients: n,
        perRecipientUsd: Number(BigInt(cfg.perRecipientGas) * gp),
        fixedUsd: Number(BigInt(cfg.baseGas) * gp), maxOverhead: cfg.maxOverhead,
      });
      if (verdict.go) {
        // A pot over the cap is paid a cap at a time: the rest stays for the
        // next epoch, where it is simply more money.
        if (cfg.maxPayWei && value > BigInt(cfg.maxPayWei)) value = BigInt(cfg.maxPayWei);
        return { action: { kind: "pay", value, recipients, gas }, state: s };
      }
    }
  }

  // 5. Enough RVN for a swap starts the next one.
  //
  // **Largest coins first, dust ignored, and the amount from exactly those.**
  // Counting every coin toward the threshold and then spending the first fifty
  // stalled forever on a wallet of many small payouts (the second review: a
  // hundred 2.5 RVN outputs sized an exchange under ChangeNOW's minimum, every
  // tick); and anybody can send dust to a public address.
  const coins = (facts.rvnUtxos || []).filter((u) => BigInt(u.sats) >= DUST_SATS)
    .sort((a, b) => (BigInt(b.sats) > BigInt(a.sats) ? 1 : BigInt(b.sats) < BigInt(a.sats) ? -1 : 0))
    .slice(0, MAX_RVN_INPUTS);
  const rvn = coins.reduce((a, u) => a + BigInt(u.sats), 0n);
  const min = BigInt(facts.rvnMinSats || 0);
  const threshold = BigInt(cfg.rvnBatchSats) > min ? BigInt(cfg.rvnBatchSats) : min;
  const sats = rvn - rvnFee(coins.length || 1);
  if (min > 0n && rvn >= threshold && sats >= min) {
    return { action: { kind: "create_exchange", sats, outpoints: coins.map((u) => `${u.txid}:${u.vout}`) }, state: s };
  }
  return wait(`holding ${rvn} of ${threshold} sats RVN; ${eth} wei ETH; ${n} recipient(s)`);
}
