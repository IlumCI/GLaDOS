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
// The runner (worker.js) gathers `facts`, calls this, performs the one action,
// and stores the new state. Everything that decides is here and tested in
// `flow.test.mjs` with no network, no storage and no coins.
//
// The loop, once per epoch:
//   RVN accumulates  ->  exchange created  ->  RVN sent to ChangeNOW
//   ->  ETH arrives  ->  (GladosPayout deployed, once)  ->  buyAndPay
import { worthPaying } from "./cadence.js";

export const FRESH = { phase: "idle", exchange: null, pending: null, contract: null, failures: 0, halted: null, epochs: 0 };

// Ravencoin's default minimum relay fee is 0.01 RVN per kB; 0.015 leaves margin.
const RVN_SAT_PER_BYTE = 1500n;
const rvnTxBytes = (inputs, outputs) => BigInt(10 + 148 * inputs + 34 * outputs);
export const MAX_RVN_INPUTS = 50;

export function rvnFee(inputs) {
  return rvnTxBytes(inputs, 1) * RVN_SAT_PER_BYTE;
}

export function decide(state, facts, cfg) {
  const s = { ...state };
  const wait = (why) => ({ action: { kind: "wait", why }, state: s });
  if (s.halted) return wait(`halted: ${s.halted}`);

  // 1. A transaction in flight settles before anything else happens.
  if (s.pending) {
    const p = s.pending;
    const r = facts.receipt;
    if (r === undefined || r === null) {
      if (facts.now - p.at > cfg.dropAfterMs) {
        return { action: { kind: "rebroadcast", chain: p.chain, raw: p.raw, why: "not mined yet; the same bytes again" }, state: s };
      }
      return wait(`waiting on ${p.kind} ${p.hash}`);
    }
    if (r.ok) {
      s.pending = null;
      s.failures = 0;
      if (p.kind === "deploy") s.contract = p.contract;
      if (p.kind === "rvn") s.exchange = { ...s.exchange, sent: p.hash };
      if (p.kind === "pay") {
        s.epochs += 1;
        s.phase = "idle";
        return { action: { kind: "record_epoch", epoch: p.epoch, hash: p.hash }, state: s };
      }
      return { action: { kind: "note", why: `${p.kind} ${p.hash} confirmed` }, state: s };
    }
    s.pending = null;
    s.failures += 1;
    if (s.failures >= cfg.maxFailures) s.halted = `${s.failures} failed transactions in a row, last ${p.kind} ${p.hash}`;
    return { action: { kind: "note", why: `${p.kind} ${p.hash} failed (${s.failures})` }, state: s };
  }

  // 2. An exchange in progress: send the RVN once, then wait for the ETH.
  if (s.exchange) {
    const x = s.exchange;
    if (!x.sent) {
      const utxos = (facts.rvnUtxos || []).slice(0, MAX_RVN_INPUTS);
      const total = utxos.reduce((a, u) => a + BigInt(u.sats), 0n);
      const fee = rvnFee(utxos.length);
      if (total - fee < BigInt(x.sats)) return wait(`the exchange wants ${x.sats} sats and the wallet can send ${total - fee}`);
      return { action: { kind: "send_rvn", utxos, to: x.payin, sats: BigInt(x.sats), fee, change: total - fee - BigInt(x.sats) }, state: s };
    }
    const st = facts.exchangeStatus;
    if (st === "finished") {
      s.exchange = null;
      s.phase = "funded";
      return { action: { kind: "note", why: `exchange ${x.id} finished` }, state: s };
    }
    if (st === "failed" || st === "refunded" || st === "expired") {
      s.exchange = null;
      return { action: { kind: "note", why: `exchange ${x.id} ${st}; RVN is refunded to the pool's own wallet` }, state: s };
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
  const n = (facts.recipients || []).length;
  if (s.contract && n > 0) {
    const gas = BigInt(cfg.baseGas) + BigInt(cfg.perRecipientGas) * BigInt(n);
    const reserve = gas * gp * 2n; // this payout's gas, twice over
    const value = eth - reserve;
    if (value > 0n) {
      const verdict = worthPaying({
        potUsd: Number(value), rvn: 1, rvnMin: 0, recipients: n,
        perRecipientUsd: Number(BigInt(cfg.perRecipientGas) * gp),
        fixedUsd: Number(BigInt(cfg.baseGas) * gp), maxOverhead: cfg.maxOverhead,
      });
      if (verdict.go) {
        if (cfg.maxPayWei && value > BigInt(cfg.maxPayWei)) {
          s.halted = `a payout of ${value} wei is over the cap of ${cfg.maxPayWei}; an operator should look`;
          return wait(s.halted);
        }
        return { action: { kind: "pay", value, recipients: facts.recipients, gas }, state: s };
      }
    }
  }

  // 5. Enough RVN for a swap starts the next one.
  const rvn = (facts.rvnUtxos || []).reduce((a, u) => a + BigInt(u.sats), 0n);
  const threshold = BigInt(cfg.rvnBatchSats) > BigInt(facts.rvnMinSats || 0) ? BigInt(cfg.rvnBatchSats) : BigInt(facts.rvnMinSats || 0);
  if (rvn >= threshold && threshold > 0n) {
    const inputs = Math.min((facts.rvnUtxos || []).length, MAX_RVN_INPUTS);
    const usable = (facts.rvnUtxos || []).slice(0, inputs).reduce((a, u) => a + BigInt(u.sats), 0n);
    const sats = usable - rvnFee(inputs);
    return { action: { kind: "create_exchange", sats }, state: s };
  }
  return wait(`holding ${rvn} of ${threshold} sats RVN; ${eth} wei ETH; ${n} recipient(s)`);
}
