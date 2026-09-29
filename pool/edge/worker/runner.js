// The treasury's hands: gather what is true, ask `decide`, do the one thing.
//
// Everything that *decides* is in flow.js and tested there. This file only
// reads (balances, unspent outputs, receipts, an exchange's status, the pair's
// reserves) and writes (an exchange, a broadcast), and each write follows the
// rule flow.js depends on: **the state records a transaction before the network
// sees it**, so a restart resends those bytes rather than making new ones.
//
// Three modes, from `TREASURY`:
//   off   the default. Nothing runs.
//   dry   reads everything and logs the action it would take; moves nothing.
//   live  does it.
import { decide, FRESH } from "./flow.js";
import { signRvnTx, rvnTxid, signLegacyTx, createdAddress, encodeBuyAndPay, encodeCtor, amountOut } from "./treasury.js";
import { parseLedger, workByAddress, build } from "./epoch.js";
import payoutArtifact from "./GladosPayout.json";

const RPC = "https://rpc.mainnet.chain.robinhood.com";
const CHAIN_ID = 4663;
const BLOCKBOOK = "https://blockbook.ravencoin.org/api/v2";
const CHANGENOW = "https://api.changenow.io/v2";
export const TOKEN = "0x3d609ecafc6aa7dba67dd7ad1d10b49c52d57777";
export const PAIR = "0x93f777932d98d15b351d1bce8c76b34381eede5b";
export const WETH = "0x0bd7d308f8e1639fab988df18a8011f41eacad73";

// Gas figures measured on a fork of the real chain (contracts/test/payout-fork.mjs):
// ~166,600 for the buy and ~50,850 a recipient, rounded up.
const CFG = {
  dropAfterMs: 20 * 60_000, maxFailures: 3,
  exchangeUnsentMs: 2 * 3_600_000, exchangeStuckMs: 48 * 3_600_000, deployGas: 900_000, baseGas: 200_000,
  perRecipientGas: 60_000, maxOverhead: 0.1,
};

const big = (x) => BigInt(x);
const toJSON = (v) => JSON.parse(JSON.stringify(v, (_, x) => (typeof x === "bigint" ? x.toString() : x)));

async function rpc(method, params) {
  const r = await fetch(RPC, { method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ jsonrpc: "2.0", id: 1, method, params }) });
  const j = await r.json();
  if (j.error) throw new Error(`${method}: ${j.error.message}`);
  return j.result;
}

async function json(url, init) {
  const r = await fetch(url, init);
  const t = await r.text();
  try { return { status: r.status, body: JSON.parse(t) }; } catch { return { status: r.status, body: { error: t.slice(0, 200) } }; }
}

export class Treasury {
  constructor(pool) {
    this.pool = pool; // the Pool Durable Object: ctx, env, core, treasury()
    this.lastTick = 0;
    this.lastNote = "";
  }

  get mode() {
    return this.pool.env.TREASURY || "off";
  }

  get cfg() {
    return { ...CFG,
      rvnBatchSats: big(Math.round(Number(this.pool.env.RVN_BATCH || 200) * 1e8)),
      maxPayWei: this.pool.env.TREASURY_MAX_PAY_WEI ? big(this.pool.env.TREASURY_MAX_PAY_WEI) : 5n * 10n ** 16n };
  }

  log(line) {
    if (line !== this.lastNote) this.pool.log(`[treasury] ${line}`);
    this.lastNote = line;
  }

  async state() {
    return (await this.pool.ctx.storage.get("treasury.state")) || { ...FRESH };
  }

  async save(s) {
    await this.pool.ctx.storage.put("treasury.state", toJSON(s));
  }

  // Once every `everyMs`, whatever calls it.
  async tick(everyMs = 5 * 60_000) {
    if (this.mode === "off" || Date.now() - this.lastTick < everyMs) return;
    this.lastTick = Date.now();
    try {
      await this.step();
    } catch (e) {
      this.log(`step failed, nothing sent: ${e && e.message ? e.message : e}`);
    }
  }

  async step() {
    const keys = await this.pool.treasury();
    const state = await this.state();
    // **Resuming a halt is a deploy, never a request.** A halted treasury
    // stays halted until the operator changes TREASURY_RESUME in the config
    // and redeploys; there is no endpoint for it, because an endpoint is
    // something anybody on the internet can call. The value is remembered, so
    // one change resumes one halt and not every halt after it.
    const resume = this.pool.env.TREASURY_RESUME || "";
    if (state.halted && resume && resume !== state.resumedWith) {
      this.log(`resumed by TREASURY_RESUME=${resume} from: ${state.halted}`);
      state.halted = null;
      state.failures = 0;
      state.resumedWith = resume;
      await this.save(state);
    }
    const facts = await this.facts(keys, state);
    const { action, state: next } = decide(state, facts, this.cfg);
    if (action.kind === "wait" || action.kind === "note") {
      this.log(action.why);
      await this.save(next);
      return;
    }
    if (this.mode !== "live") {
      this.log(`dry run: would ${action.kind} ${JSON.stringify(toJSON({ ...action, recipients: action.recipients?.length, utxos: action.utxos?.length }))}`);
      return;
    }
    await this.perform(action, next, keys, facts);
  }

  // --- reading ---------------------------------------------------------------------------

  async facts(keys, state) {
    const f = { now: Date.now() };
    const [utxos, min, gasPrice, ethWei] = await Promise.all([
      json(`${BLOCKBOOK}/utxo/${keys.rvn}?confirmed=true`),
      json(`${CHANGENOW}/exchange/min-amount?fromCurrency=rvn&fromNetwork=rvn&toCurrency=eth&toNetwork=hood&flow=standard`),
      rpc("eth_gasPrice", []),
      rpc("eth_getBalance", [keys.evm, "latest"]),
    ]);
    // Coinbase outputs cannot be spent for 100 blocks; a transaction spending one
    // earlier is refused by every node. zpool pays with ordinary transactions,
    // so this should never bite -- which is exactly when a filter is cheap.
    f.rvnUtxos = Array.isArray(utxos.body)
      ? utxos.body.filter((u) => !(u.coinbase && (u.confirmations || 0) < 100))
          .map((u) => ({ txid: u.txid, vout: u.vout, sats: u.value }))
      : [];
    f.rvnMinSats = min.body && min.body.minAmount ? big(Math.ceil(min.body.minAmount * 1e8)) : 0n;
    f.gasPrice = big(gasPrice);
    f.ethWei = big(ethWei);

    if (state.pending) f.receipt = await this.receipt(state.pending);
    if (state.exchange && state.exchange.sent) {
      const x = await json(`${CHANGENOW}/exchange/by-id?id=${encodeURIComponent(state.exchange.id)}`,
        { headers: { "x-changenow-api-key": this.pool.env.CHANGENOW_KEY || "" } });
      f.exchangeStatus = x.body && x.body.status;
    }
    // The epoch is only worth building when there is a contract and money to pay with.
    if (state.contract && f.ethWei > 0n && !state.pending) {
      f.epoch = await this.epoch();
      f.recipients = f.epoch.recipients;
    }
    return f;
  }

  async receipt(p) {
    if (p.rejected) return { ok: false };
    if (p.chain === "evm") {
      const r = await rpc("eth_getTransactionReceipt", [p.hash]);
      return r ? { ok: r.status === "0x1" } : null;
    }
    const t = await json(`${BLOCKBOOK}/tx/${p.hash}`);
    if (t.status === 200 && t.body && t.body.confirmations >= 1) return { ok: true };
    return null;
  }

  // Every shard's ledger summed by payout address, less the last epoch's
  // snapshot, gated on GLADOS held right now, floored at a quarter of typical.
  async epoch() {
    const env = this.pool.env;
    const n = Math.max(1, Number(env.SHARDS || 1));
    const gpu = env.GPU_COINS ? Math.max(1, Number(env.GPU_SHARDS || 1)) : 0;
    const names = [...[...Array(n).keys()].map((i) => (i === 0 ? "main" : `shard-${i}`)), ...[...Array(gpu).keys()].map((i) => `gpu-${i}`)];
    const docs = await Promise.all(names.map((name) => name === "main"
      ? this.pool.core.ledger(1, Math.floor(Date.now() / 1000))
      : env.POOL.get(env.POOL.idFromName(name)).fetch(new Request("https://pool/ledger.json")).then((r) => r.text())));
    const now = workByAddress(docs.flatMap(parseLedger));
    const snap = (await this.pool.ctx.storage.get("treasury.snapshot")) || {};
    const prev = new Map(Object.entries(snap).map(([a, w]) => [a, big(w)]));
    const addrs = [...now.keys()];
    const balances = new Map();
    if (addrs.length) {
      const res = await fetch(RPC, { method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify(addrs.map((a, i) => ({ jsonrpc: "2.0", id: i, method: "eth_call",
          params: [{ to: TOKEN, data: "0x70a08231" + a.slice(2).padStart(64, "0") }, "latest"] }))) });
      for (const r of await res.json()) if (r.result) balances.set(addrs[r.id], big(r.result));
    }
    const gateMin = big(env.GATE_MIN || "0") * 10n ** 18n;
    return build({ now, prev, balances, gateMin });
  }

  // --- writing ---------------------------------------------------------------------------

  async perform(a, next, keys, facts) {
    if (a.kind === "create_exchange") {
      const res = await json(`${CHANGENOW}/exchange`, { method: "POST",
        headers: { "content-type": "application/json", "x-changenow-api-key": this.pool.env.CHANGENOW_KEY || "" },
        body: JSON.stringify({ fromCurrency: "rvn", fromNetwork: "rvn", toCurrency: "eth", toNetwork: "hood",
          fromAmount: (Number(a.sats) / 1e8).toFixed(8), address: keys.evm, refundAddress: keys.rvn,
          flow: "standard", type: "direct" }) });
      if (!res.body || !res.body.id || !res.body.payinAddress) {
        this.log(`exchange not created: ${JSON.stringify(res.body).slice(0, 200)}`);
        return;
      }
      next.exchange = { id: res.body.id, payin: res.body.payinAddress, sats: a.sats.toString(), created: Date.now() };
      await this.save(next);
      this.log(`exchange ${res.body.id} created for ${Number(a.sats) / 1e8} RVN`);
      return;
    }

    if (a.kind === "send_rvn") {
      const outputs = [{ address: a.to, sats: a.sats }];
      if (a.change > 0n) outputs.push({ address: keys.rvn, sats: a.change });
      const raw = signRvnTx(keys.rvnKey, a.utxos, outputs);
      next.pending = { kind: "rvn", chain: "rvn", hash: rvnTxid(raw), raw, at: Date.now() };
      await this.save(next); // recorded before the network sees it
      await this.broadcast(next);
      return;
    }

    if (a.kind === "rebroadcast") {
      await this.broadcast(next);
      return;
    }

    if (a.kind === "deploy") {
      const nonce = Number(await rpc("eth_getTransactionCount", [keys.evm, "pending"]));
      const gasPrice = (facts.gasPrice * 125n) / 100n;
      const data = "0x" + payoutArtifact.bytecode + encodeCtor(WETH, TOKEN, PAIR);
      const est = await this.estimate({ from: keys.evm, data }, next);
      if (est === null) return;
      const gasLimit = (est * 12n) / 10n;
      if (facts.ethWei < gasLimit * gasPrice) {
        this.log(`deploy waits: ${facts.ethWei} wei cannot cover ${gasLimit * gasPrice}`);
        return;
      }
      const tx = signLegacyTx(keys.evmKey, { nonce, gasPrice, gas: gasLimit, to: null, value: 0n, data, chainId: CHAIN_ID });
      next.pending = { kind: "deploy", chain: "evm", hash: tx.hash, raw: tx.raw, at: Date.now(), contract: createdAddress(keys.evm, nonce) };
      await this.save(next);
      await this.broadcast(next);
      return;
    }

    if (a.kind === "pay") {
      const [r0, r1] = await this.reserves();
      const [rW, rG] = big(WETH) < big(TOKEN) ? [r0, r1] : [r1, r0];
      // The quote, less the token's 1% buy tax, less 3% for movement between
      // now and the block: a sandwich past that reverts the whole payout.
      const quoteMin = (v) => (amountOut(v, rW, rG) * 99n * 97n) / 10_000n;
      const nonce = Number(await rpc("eth_getTransactionCount", [keys.evm, "pending"]));
      const gasPrice = (facts.gasPrice * 125n) / 100n;
      const est = await this.estimate({ from: keys.evm, to: next.contract, data: encodeBuyAndPay(a.recipients, quoteMin(a.value)),
                                        value: "0x" + a.value.toString(16) }, next);
      if (est === null) return;
      const gasLimit = (est * 12n) / 10n;
      // **Value plus the most this transaction can burn must fit the balance**,
      // or the node refuses it as underfunded. The reserve in flow.js is sized
      // from measured gas; this is the check against what was actually signed.
      let value = a.value;
      if (value + gasLimit * gasPrice > facts.ethWei) value = facts.ethWei - gasLimit * gasPrice;
      if (value <= 0n) {
        this.log(`payout waits: gas ${gasLimit * gasPrice} wei would take the whole balance`);
        return;
      }
      const minOut = quoteMin(value);
      const data = encodeBuyAndPay(a.recipients, minOut);
      const tx = signLegacyTx(keys.evmKey, { nonce, gasPrice, gas: gasLimit, to: next.contract, value, data, chainId: CHAIN_ID });
      const e = facts.epoch;
      next.pending = { kind: "pay", chain: "evm", hash: tx.hash, raw: tx.raw, at: Date.now(),
        epoch: { id: (next.epochs || 0) + 1, at: Date.now(), value: value.toString(), minOut: minOut.toString(),
                 recipients: a.recipients, work: e.work, floor: e.floor, excluded: e.excluded,
                 snapshot: Object.fromEntries([...e.snapshot].map(([k, v]) => [k, v.toString()])) } };
      await this.save(next);
      await this.broadcast(next);
      return;
    }

    if (a.kind === "record_epoch") {
      const epochs = (await this.pool.ctx.storage.get("treasury.epochs")) || [];
      const { snapshot, ...rest } = a.epoch;
      epochs.push({ ...rest, tx: a.hash });
      await this.pool.ctx.storage.put("treasury.epochs", epochs);
      await this.pool.ctx.storage.put("treasury.snapshot", snapshot);
      await this.save(next);
      this.log(`epoch ${rest.id} paid ${rest.recipients.length} miner(s) in ${a.hash}`);
    }
  }

  // eth_estimateGas, where a revert is a failure the decision counts: a payout
  // the chain would refuse must reach the halt after three tries, not be
  // re-estimated every five minutes forever with nobody told.
  async estimate(tx, next) {
    try {
      return big(await rpc("eth_estimateGas", [tx]));
    } catch (e) {
      next.failures = (next.failures || 0) + 1;
      if (next.failures >= CFG.maxFailures) next.halted = `${next.failures} estimates refused in a row, last: ${e.message}`;
      await this.save(next);
      this.log(`estimate refused (${next.failures}): ${e.message}`);
      return null;
    }
  }

  async reserves() {
    const ret = await rpc("eth_call", [{ to: PAIR, data: "0x0902f1ac" }, "latest"]);
    return [big("0x" + ret.slice(2, 66)), big("0x" + ret.slice(66, 130))];
  }

  // Send `state.pending` to its chain. A refusal that says the transaction is
  // already known is success; any other refusal marks it rejected, which the
  // next `decide` counts as a failure rather than waiting on it forever.
  async broadcast(s) {
    const p = s.pending;
    let err = null;
    if (p.chain === "rvn") {
      const r = await json(`${BLOCKBOOK}/sendtx/`, { method: "POST", body: p.raw });
      if (r.body && r.body.error && !/already|known|in mempool|exists/i.test(String(r.body.error))) err = String(r.body.error);
    } else {
      try {
        await rpc("eth_sendRawTransaction", [p.raw]);
      } catch (e) {
        if (!/already known|nonce too low|already imported/i.test(e.message)) err = e.message;
      }
    }
    if (err) {
      s.pending = { ...p, rejected: err };
      await this.save(s);
      this.log(`${p.kind} ${p.hash} refused: ${err}`);
    } else {
      this.log(`${p.kind} ${p.hash} broadcast`);
    }
  }

  // What /treasury shows: never a key.
  async summary() {
    const s = await this.state();
    return { mode: this.mode, phase: s.phase, halted: s.halted, epochs: s.epochs, contract: s.contract,
             exchange: s.exchange && { id: s.exchange.id, sent: s.exchange.sent || null },
             pending: s.pending && { kind: s.pending.kind, hash: s.pending.hash, rejected: s.pending.rejected || null } };
  }
}
