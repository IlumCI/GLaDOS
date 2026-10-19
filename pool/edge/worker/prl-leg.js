// The Pearl leg of the treasury: WPRL on Ethereum mainnet into ETH on chain
// 4663, where the payout already happens.
//
//   PRL (mined, paid by 2Miners to the operator's wallet)
//     -> the operator's sweep script sends it to PearlBridge's deposit address
//        for the treasury's EVM key (pool/edge/prl-sweep.py)
//     -> WPRL arrives at that key on Ethereum mainnet          <- this file from here
//     -> Uniswap v3 WPRL/USDT (1%) swaps it to USDT
//     -> a ChangeNOW exchange, USDT (ERC-20) to ETH on 4663, paid to the same key
//     -> the ETH is in the payout pot like the RVN's
//
// One key, three chains: the treasury's EVM key is the same address on
// mainnet and on 4663, so WPRL, USDT and the ETH that pays all land on it.
//
// **Balances decide, not memory.** Each tick reads WPRL, USDT and ETH as the
// chain has them and does at most one thing. A swap that mined shows as less
// WPRL; one that did not, as the same. So a lost record costs a tick, never a
// second swap. The one thing that is not visible on the chain is which
// ChangeNOW exchange a USDT transfer funds, so that is recorded before the
// transfer is signed.
//
// **One transaction in flight, and never a new nonce while it is.** A stuck
// one is signed again at the same nonce, higher priced, carrying its earlier
// attempts; whichever the chain keeps is the one recorded.
//
// Off unless PRL_LEG is "live" ("dry" logs what it would do and sends nothing).

import { Endpoints, attempt, Transient, RpcError } from "./rpc.js";
import { signLegacyTx, selector } from "./treasury.js";

export const MAINNET = {
  chainId: 1,
  wprl: "0x07696dcab55e62cfef953666b29fe1970518cb00", // 8 decimals; PearlBridge's, verified on Etherscan
  usdt: "0xdac17f958d2ee523a2206206994597c13d831ec7", // 6 decimals
  pool: "0x89a67c6dee35db9815da2fb9191f0998a8b37c39", // Uniswap v3 WPRL/USDT, 1%
  fee: 10_000,
  router: "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45", // SwapRouter02
  quoter: "0x61ffe014ba17989e743c5f6cb21bf9697530b21e", // QuoterV2
};
export const MAINNET_RPCS = ["https://ethereum-rpc.publicnode.com", "https://eth.llamarpc.com", "https://rpc.ankr.com/eth", "https://cloudflare-eth.com"];
const CHANGENOW = "https://api.changenow.io/v2";

export const PRL_CFG = {
  minSwapUsd: 20,          // a swap is worth its gas from here (gas is cents on mainnet today)
  maxSwapUsd: 2_000,       // one swap at most; more waits for the next tick
  maxImpactBps: 150,       // a swap that moves the price more than 1.5% is halved until it does not
  slipBps: 150,            // the minimum out is the quote less 1.5%
  maxPriceMove: 0.15,      // the price against its recent median before a swap is refused
  priceSamples: 12, minPriceSamples: 3,
  minSendUsdt: 20,         // USDT goes to ChangeNOW from here (its minimum is ~$1.50)
  gasApprove: 80_000, gasSwap: 260_000, gasTransfer: 100_000,
  gasCapWei: 50n * 10n ** 9n,     // 50 gwei: above this, wait
  stuckMs: 30 * 60_000,    // a transaction unmined this long is priced up
  nonceGraceMs: 10 * 60_000,
  exchangeTimeoutMs: 48 * 3_600_000,
  maxFailures: 3, pauseMs: 6 * 3_600_000,
};

// --- calldata ---------------------------------------------------------------------------------
const w = (v) => BigInt(v).toString(16).padStart(64, "0");
const a = (addr) => String(addr).toLowerCase().replace(/^0x/, "").padStart(64, "0");
const SEL = {
  balanceOf: "70a08231", allowance: "dd62ed3e", approve: "095ea7b3", transfer: "a9059cbb",
  exactInputSingle: selector("exactInputSingle((address,address,uint24,address,uint256,uint256,uint160))"),
  quote: selector("quoteExactInputSingle((address,address,uint256,uint24,uint160))"),
};
export const calldata = {
  balanceOf: (owner) => "0x" + SEL.balanceOf + a(owner),
  allowance: (owner, spender) => "0x" + SEL.allowance + a(owner) + a(spender),
  approve: (spender, amount) => "0x" + SEL.approve + a(spender) + w(amount),
  transfer: (to, amount) => "0x" + SEL.transfer + a(to) + w(amount),
  quote: (amountIn) => "0x" + SEL.quote + a(MAINNET.wprl) + a(MAINNET.usdt) + w(amountIn) + w(MAINNET.fee) + w(0),
  swap: (recipient, amountIn, minOut) => "0x" + SEL.exactInputSingle + a(MAINNET.wprl) + a(MAINNET.usdt) + w(MAINNET.fee) + a(recipient) + w(amountIn) + w(minOut) + w(0),
};
const MAX_UINT = (1n << 256n) - 1n;
const big = (x) => BigInt(x ?? 0);
const USD6 = (v) => Number(v) / 1e6; // USDT units to dollars

// --- the decision -----------------------------------------------------------------------------

/**
 * What to do this tick, from the leg's state and what the chain says now.
 * Pure: everything it needs is in `s` and `f`.
 */
export function decidePrl(s, f, cfg = PRL_CFG) {
  if (s.pending) return { kind: "wait", why: `waiting on ${s.pending.kind} ${s.pending.hash}` };
  if (s.paused && s.paused.until > f.now) return { kind: "wait", why: `paused until ${new Date(s.paused.until).toISOString()}: ${s.paused.why}` };
  if (f.gasPrice > cfg.gasCapWei) return { kind: "wait", why: `mainnet gas at ${Number(f.gasPrice) / 1e9} gwei, over the cap` };

  // An exchange created and not yet funded is funded now, with exactly what it
  // was created for. One funded is ChangeNOW's to finish (watched every tick,
  // apart from this); meanwhile WPRL is still swapped -- USDT is a stable place
  // to wait -- and only a second exchange waits for the first.
  if (s.exchange && !s.exchange.sent) {
    if (f.usdt < big(s.exchange.usdt)) return { kind: "drop_exchange", why: `holding ${f.usdt} USDT units, less than the ${s.exchange.usdt} the exchange was made for` };
    return { kind: "fund_exchange", gas: cfg.gasTransfer };
  }
  if (!s.exchange && f.usdt > 0n && USD6(f.usdt) >= Math.max(cfg.minSendUsdt, (f.cnMin || 0) * 1.2)) return { kind: "create_exchange", usdt: f.usdt };

  if (f.wprl > 0n && f.unitUsd > 0 && (Number(f.wprl) / 1e8) * f.unitUsd >= cfg.minSwapUsd) {
    if (f.allowance < f.wprl) return { kind: "approve", gas: cfg.gasApprove };
    return { kind: "swap", gas: cfg.gasSwap };
  }
  return { kind: "idle", why: `holding ${Number(f.wprl) / 1e8} WPRL and ${USD6(f.usdt)} USDT; nothing worth moving` };
}

// The price guard, as the payout's: readings kept in the state, one per tick.
export function priceMoved(s, unitUsd, cfg = PRL_CFG) {
  const hist = (s.px || []).filter((x) => x > 0);
  s.px = [...hist, unitUsd].slice(-cfg.priceSamples);
  if (!(unitUsd > 0)) return "no price for WPRL";
  if (hist.length < cfg.minPriceSamples) return `${hist.length + 1} price reading(s) so far; ${cfg.minPriceSamples + 1} before the first swap`;
  const sorted = [...hist].sort((x, y) => x - y);
  const median = sorted[Math.floor(sorted.length / 2)];
  const move = Math.abs(unitUsd - median) / median;
  return move > cfg.maxPriceMove ? `WPRL is ${(move * 100).toFixed(1)}% off its recent median` : null;
}

// --- the leg ----------------------------------------------------------------------------------

export class PrlLeg {
  // `t` is the Treasury: its transport, clock, log, storage and incidents.
  constructor(t) {
    this.t = t;
    this.cfg = { ...PRL_CFG, ...((t.overrides && t.overrides.prl) || {}) };
    const urls = String(t.pool.env.ETH_RPC_URLS || "").split(/[\s,]+/).filter(Boolean);
    this.rpcs = new Endpoints(urls.length ? urls : MAINNET_RPCS, t.now, "0x1");
  }

  get mode() { return this.t.pool.env.PRL_LEG || "off"; }
  log(line) { this.t.log(`prl: ${line}`); }

  rpc(method, params) {
    return this.t.retry(async () => {
      const v = await attempt(this.rpcs, { jsonrpc: "2.0", id: 1, method, params }, this.t.rpcPost, `mainnet ${method}`);
      if (v.error !== undefined) throw new RpcError(`mainnet ${method}: ${v.error}`, v.code);
      return v.result;
    });
  }
  async call(to, data) {
    const r = await this.rpc("eth_call", [{ to, data }, "latest"]);
    if (!/^0x[0-9a-f]*$/i.test(String(r))) throw new Transient(`mainnet eth_call answered ${String(r).slice(0, 40)}`);
    return r;
  }
  async uint(to, data) {
    const r = await this.call(to, data);
    if (r.length < 66) throw new Transient(`mainnet eth_call to ${to} answered ${r}`);
    return BigInt(r.slice(0, 66));
  }

  async state() { return (await this.t.storage.get("treasury.prl")) || {}; }
  async save(s) {
    if (this.mode !== "live") return;
    await this.t.storage.put("treasury.prl", s);
  }
  async incident(why) {
    const main = await this.t.state();
    this.t.incident(main, `Pearl leg: ${why}`);
    await this.t.save(main);
  }

  // The facts a decision needs, read fresh.
  async facts(keys) {
    const me = keys.evm;
    const [eth, wprl, usdt, allowance, gasPrice] = await Promise.all([
      this.rpc("eth_getBalance", [me, "latest"]).then(big),
      this.uint(MAINNET.wprl, calldata.balanceOf(me)),
      this.uint(MAINNET.usdt, calldata.balanceOf(me)),
      this.uint(MAINNET.wprl, calldata.allowance(me, MAINNET.router)),
      this.rpc("eth_gasPrice", []).then(big),
    ]);
    // One WPRL's worth, for the floor and the price guard; the quote for the
    // actual amount is taken at swap time.
    let unitUsd = 0;
    try { unitUsd = USD6(await this.uint(MAINNET.quoter, calldata.quote(10n ** 8n))); } catch (e) { if (!(e instanceof Transient || e instanceof RpcError)) throw e; }
    return { now: this.t.now(), eth, wprl, usdt, allowance, gasPrice, unitUsd };
  }

  async tick(keys) {
    if (this.mode === "off") return;
    const s = await this.state();
    if (s.pending) {
      const done = await this.settle(s, keys);
      if (!done) return;
    }
    const f = await this.facts(keys);
    if (s.exchange && !s.exchange.sent) {
      try { f.cnMin = await this.cnMin(); } catch {}
    } else if (!s.exchange && f.usdt > 0n) {
      try { f.cnMin = await this.cnMin(); } catch (e) { if (!(e instanceof Transient)) throw e; return this.log(`ChangeNOW unanswered: ${e.message}`); }
    }
    if (s.exchange && s.exchange.sent) await this.watchExchange(s);
    const act = decidePrl(s, f, this.cfg);
    // A price reading every tick, so the guard has a history when a swap comes.
    if (act.kind !== "swap" && f.unitUsd > 0) { priceMoved(s, f.unitUsd, this.cfg); await this.save(s); }
    if (act.kind === "wait" || act.kind === "idle") {
      if (act.why !== this.lastNote) this.log(act.why);
      this.lastNote = act.why;
      return;
    }
    if (act.gas && f.eth < BigInt(act.gas) * (f.gasPrice * 125n / 100n) * 2n) {
      const why = `the treasury's mainnet address ${keys.evm} holds ${Number(f.eth) / 1e18} ETH, too little gas to ${act.kind}; send it about 0.002 ETH on Ethereum`;
      if (!s.gasLow) { s.gasLow = true; await this.save(s); await this.incident(why); }
      if (why !== this.lastNote) this.log(why);
      this.lastNote = why;
      return;
    }
    if (s.gasLow) delete s.gasLow;
    await this.perform(act, s, f, keys);
  }

  // --- acting -------------------------------------------------------------------------------
  async perform(act, s, f, keys) {
    const dry = this.mode !== "live";
    if (act.kind === "drop_exchange") {
      this.log(`exchange ${s.exchange.id} dropped unfunded: ${act.why}`);
      delete s.exchange;
      return this.save(s);
    }
    if (act.kind === "create_exchange") {
      if (dry) return this.log(`dry run: would exchange ${USD6(act.usdt)} USDT for ETH on 4663`);
      const res = await this.t.json(`${CHANGENOW}/exchange`, { method: "POST", headers: this.t.cnHeaders({ "content-type": "application/json" }),
        body: JSON.stringify({ fromCurrency: "usdt", fromNetwork: "eth", toCurrency: "eth", toNetwork: "hood", fromAmount: USD6(act.usdt).toFixed(6),
          address: keys.evm, refundAddress: keys.evm, flow: "standard", type: "direct" }) });
      const b = res.body || {};
      // Checked before anything is stored, as for RVN: the deposit is an
      // address USDT can go to and is not ours, the ETH comes back to us, and
      // no memo is wanted that an ERC-20 transfer cannot carry.
      let refusal = null;
      if (!b.id || typeof b.id !== "string" || !b.payinAddress) refusal = `no exchange: ${JSON.stringify(b).slice(0, 200)}`;
      else if (!/^0x[0-9a-fA-F]{40}$/.test(b.payinAddress)) refusal = `the deposit address ${b.payinAddress} is not an Ethereum address`;
      else if (b.payinAddress.toLowerCase() === keys.evm.toLowerCase()) refusal = "the deposit address is the treasury's own";
      else if (b.payoutAddress && String(b.payoutAddress).toLowerCase() !== keys.evm.toLowerCase()) refusal = `the exchange pays ${b.payoutAddress}, not the pool's ${keys.evm}`;
      else if (b.payinExtraId) refusal = `the deposit wants a memo (${b.payinExtraId}) a transfer cannot attach`;
      if (refusal) return this.log(`exchange not created: ${refusal}`);
      s.exchange = { id: b.id, payin: b.payinAddress.toLowerCase(), usdt: act.usdt.toString(), created: f.now, sent: false };
      await this.save(s);
      return this.log(`exchange ${b.id} created for ${USD6(act.usdt)} USDT`);
    }
    if (act.kind === "approve") {
      if (dry) return this.log("dry run: would let the Uniswap router spend WPRL");
      return this.send(s, keys, f, { kind: "approve", to: MAINNET.wprl, data: calldata.approve(MAINNET.router, MAX_UINT), gas: act.gas });
    }
    if (act.kind === "swap") {
      const moved = priceMoved(s, f.unitUsd, this.cfg);
      if (moved) { await this.save(s); return this.log(`swap waits: ${moved}`); }
      // Sized so it moves the price at most maxImpactBps, and is at most maxSwapUsd.
      let amount = f.wprl;
      const capUnits = BigInt(Math.floor((this.cfg.maxSwapUsd / f.unitUsd) * 1e8));
      if (amount > capUnits) amount = capUnits;
      let out = 0n;
      for (let i = 0; i < 8; i++) {
        out = await this.uint(MAINNET.quoter, calldata.quote(amount));
        const ideal = (Number(amount) / 1e8) * f.unitUsd;
        const impact = ideal > 0 ? 1 - USD6(out) / ideal : 1;
        if (impact * 10_000 <= this.cfg.maxImpactBps) break;
        amount /= 2n;
        if ((Number(amount) / 1e8) * f.unitUsd < this.cfg.minSwapUsd) { await this.save(s); return this.log(`swap waits: even ${this.cfg.minSwapUsd} USD of WPRL moves the price over ${this.cfg.maxImpactBps / 100}%`); }
      }
      const minOut = (out * BigInt(10_000 - this.cfg.slipBps)) / 10_000n;
      if (dry) return this.log(`dry run: would swap ${Number(amount) / 1e8} WPRL for at least ${USD6(minOut)} USDT`);
      return this.send(s, keys, f, { kind: "swap", to: MAINNET.router, data: calldata.swap(keys.evm, amount, minOut), gas: act.gas, wprl: amount.toString(), minOut: minOut.toString() });
    }
    if (act.kind === "fund_exchange") {
      if (dry) return this.log(`dry run: would send ${USD6(big(s.exchange.usdt))} USDT to ChangeNOW`);
      return this.send(s, keys, f, { kind: "fund", to: MAINNET.usdt, data: calldata.transfer(s.exchange.payin, big(s.exchange.usdt)), gas: act.gas, exchange: s.exchange.id });
    }
  }

  // Sign at the next nonce, record, then broadcast: a crash between the two
  // leaves a recorded transaction the next tick finds mined or sends again.
  async send(s, keys, f, tx) {
    const nonce = Number(await this.rpc("eth_getTransactionCount", [keys.evm, "latest"]));
    const gasPrice = (f.gasPrice * 125n) / 100n;
    const signed = signLegacyTx(keys.evmKey, { nonce, gasPrice, gas: BigInt(tx.gas), to: tx.to, value: 0n, data: tx.data, chainId: MAINNET.chainId });
    s.pending = { ...tx, hash: signed.hash, raw: signed.raw, nonce, gasPrice: gasPrice.toString(), at: f.now, prior: [] };
    await this.save(s);
    await this.broadcast(s.pending);
    this.log(`${tx.kind} ${signed.hash} sent at nonce ${nonce}`);
  }

  async broadcast(p) {
    try {
      await this.rpc("eth_sendRawTransaction", [p.raw]);
    } catch (e) {
      if (e instanceof Transient) return; // unanswered: the stored bytes go again next tick
      if (!(e instanceof RpcError)) throw e;
      if (!/already known|known transaction|already imported|nonce too low/i.test(e.message)) this.log(`${p.kind} ${p.hash} refused: ${e.message}`);
    }
  }

  // The transaction in flight. True once it is settled and the tick may go on.
  async settle(s, keys) {
    const p = s.pending;
    for (const t of [p, ...(p.prior || [])]) {
      const r = await this.rpc("eth_getTransactionReceipt", [t.hash]);
      if (!r) continue;
      const ok = r.status === "0x1";
      this.log(`${p.kind} ${t.hash} mined${ok ? "" : ", and reverted"}`);
      if (p.kind === "fund" && s.exchange && s.exchange.id === p.exchange) {
        if (ok) s.exchange.sent = true;
        else delete s.exchange;
      }
      if (ok) s.failures = 0;
      else {
        s.failures = (s.failures || 0) + 1;
        if (s.failures >= this.cfg.maxFailures) {
          s.paused = { until: this.t.now() + this.cfg.pauseMs, why: `${s.failures} transactions in a row reverted` };
          s.failures = 0;
          await this.incident(`${s.paused.why}; the leg waits ${this.cfg.pauseMs / 3_600_000} h`);
        }
      }
      delete s.pending;
      await this.save(s);
      return true;
    }
    // Unmined. Its nonce taken by something that is none of ours: let go and say so.
    const used = Number(await this.rpc("eth_getTransactionCount", [keys.evm, "latest"]));
    const now = this.t.now();
    if (used > p.nonce) {
      if (p.goneAt === undefined) { p.goneAt = now; await this.save(s); return false; }
      if (now - p.goneAt < this.cfg.nonceGraceMs) return false;
      await this.incident(`nonce ${p.nonce} on mainnet was used by a transaction that is none of ours (${[p, ...(p.prior || [])].map((x) => x.hash).join(", ")})`);
      if (p.kind === "fund") delete s.exchange; // never funded by us; balances decide from here
      delete s.pending;
      await this.save(s);
      return true;
    }
    // Stuck: the same action at the same nonce, priced up, its attempts carried.
    if (now - p.at > this.cfg.stuckMs) {
      const gp = big(await this.rpc("eth_gasPrice", []));
      const gasPrice = [gp * 125n / 100n, big(p.gasPrice) * 115n / 100n].reduce((x, y) => (x > y ? x : y));
      if (gasPrice > this.cfg.gasCapWei) { this.log(`${p.kind} ${p.hash} stuck, and pricing it up would pass the gas cap; waiting`); return false; }
      // **A swap signed again is priced again.** Its minimum out was the quote
      // when first signed; hours later, after the price has risen, that floor
      // lets it fill far under the market -- the stress world found one fill
      // 23% under the price it was re-signed at. So the floor is the higher of
      // the old one and today's quote less the slippage: never looser, and a
      // price that has fallen since makes it revert rather than fill badly.
      let data = p.data, minOut = p.minOut;
      if (p.kind === "swap") {
        const q = await this.uint(MAINNET.quoter, calldata.quote(big(p.wprl)));
        const fresh = (q * BigInt(10_000 - this.cfg.slipBps)) / 10_000n;
        if (fresh > big(p.minOut)) minOut = fresh.toString();
        data = calldata.swap(keys.evm, big(p.wprl), big(minOut));
      }
      const signed = signLegacyTx(keys.evmKey, { nonce: p.nonce, gasPrice, gas: BigInt(p.gas), to: p.to, value: 0n, data, chainId: MAINNET.chainId });
      s.pending = { ...p, data, minOut, hash: signed.hash, raw: signed.raw, gasPrice: gasPrice.toString(), at: now, prior: [...(p.prior || []), { hash: p.hash }].slice(-8) };
      await this.save(s);
      await this.broadcast(s.pending);
      this.log(`${p.kind} at nonce ${p.nonce} stuck; sent again at ${Number(gasPrice) / 1e9} gwei as ${signed.hash}`);
      return false;
    }
    await this.broadcast(p);
    return false;
  }

  async watchExchange(s) {
    const x = s.exchange;
    let st;
    try {
      const r = await this.t.json(`${CHANGENOW}/exchange/by-id?id=${encodeURIComponent(x.id)}`, { headers: this.t.cnHeaders() });
      st = r.body && typeof r.body.status === "string" ? r.body.status : undefined;
    } catch (e) { if (!(e instanceof Transient)) throw e; return; }
    if (st === "finished") { this.log(`exchange ${x.id} finished: its ETH is in the payout pot on 4663`); delete s.exchange; return this.save(s); }
    if (st === "failed" || st === "refunded" || st === "expired") {
      await this.incident(`exchange ${x.id} ${st}${st === "refunded" ? "; the USDT comes back to the treasury and is tried again" : ""}`);
      delete s.exchange;
      return this.save(s);
    }
    if (this.t.now() - x.created > this.cfg.exchangeTimeoutMs && !x.late) {
      x.late = true;
      await this.save(s);
      await this.incident(`exchange ${x.id} still '${st}' after ${this.cfg.exchangeTimeoutMs / 3_600_000} h; still watching it`);
    }
  }

  async cnMin() {
    const r = await this.t.json(`${CHANGENOW}/exchange/min-amount?fromCurrency=usdt&fromNetwork=eth&toCurrency=eth&toNetwork=hood&flow=standard`, { headers: this.t.cnHeaders() });
    const m = Number(r.body && r.body.minAmount);
    if (!(m > 0)) throw new Transient(`ChangeNOW min-amount answered ${JSON.stringify(r.body).slice(0, 80)}`);
    return m;
  }
}
