// node pool/edge/worker/prl-leg.test.mjs [seeds]
//
// The Pearl leg (prl-leg.js) against a simulated Ethereum mainnet that is out
// to get it: a chain with nonces, a mempool, stalls, late mining, gas spikes
// and replacement rules; WPRL and USDT; a constant-product WPRL/USDT pool with
// a 1% fee, a quoter and a router that pulls by allowance and reverts under
// the minimum out; a ChangeNOW that fails, refunds, sits on funds and lies
// about where money goes; PearlBridge minting WPRL at random; RPCs that drop,
// rate-limit, answer garbage or accept a transaction while saying they did
// not; and the object evicted after a request lands.
//
// It checks the world, after every tick, never what the leg says:
//   WPRL      every unit minted is held or was swapped by the router
//   USDT      every unit from a swap or a refund is held or went to the deposit
//             address of an exchange ChangeNOW made for the pool, paying the pool
//   exchanges each funded at most once, with exactly what it was made for
//   ETH       never sent on mainnet but as gas
//   price     no swap filled far under the pool's price when it was signed
// and, after a calm: everything worth moving has moved, nothing is left in
// flight, and the ETH delivered on 4663 is what the WPRL was worth, less fees.
import * as secp from "@noble/secp256k1";
import { keccak_256 } from "@noble/hashes/sha3";
import { Treasury } from "./runner.js";
import { MAINNET, MAINNET_RPCS, PRL_CFG, decidePrl } from "./prl-leg.js";
import { Transient } from "./rpc.js";
import { newKey, evmAddress, rlp, int, hex, unhex } from "./treasury.js";

let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.log(`${c ? "ok  " : "FAIL"}  ${w}`); };
const TICK = 5 * 60_000, DAY = 288;
const CN = "https://api.changenow.io/v2";

function rng(seed) {
  let a = seed >>> 0;
  const f = () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  f.int = (n) => Math.floor(f() * n); f.chance = (p) => f() < p;
  return f;
}
class Evicted extends Error {}

function rlpDecode(b, i = 0) {
  const x = b[i];
  const num = (s, e) => { let n = 0; for (let k = s; k < e; k++) n = n * 256 + b[k]; return n; };
  if (x < 0x80) return [b.slice(i, i + 1), i + 1];
  if (x < 0xb8) return [b.slice(i + 1, i + 1 + x - 0x80), i + 1 + x - 0x80];
  if (x < 0xc0) { const ll = x - 0xb7, len = num(i + 1, i + 1 + ll), s = i + 1 + ll; return [b.slice(s, s + len), s + len]; }
  let s, end;
  if (x < 0xf8) { s = i + 1; end = s + x - 0xc0; } else { const ll = x - 0xf7; s = i + 1 + ll; end = s + num(i + 1, i + 1 + ll); }
  const out = [];
  while (s < end) { const [it, n] = rlpDecode(b, s); out.push(it); s = n; }
  return [out, end];
}
const toBig = (bytes) => (bytes.length ? BigInt("0x" + hex(bytes)) : 0n);
function decodeTx(raw, chainId) {
  const [f] = rlpDecode(unhex(raw.replace(/^0x/, "")));
  const [nonce, gasPrice, gas, to, value, data, v, r, s] = f;
  const rec = Number(toBig(v) - BigInt(chainId) * 2n - 35n);
  if (rec !== 0 && rec !== 1) throw new Error("invalid sender: wrong chain id");
  const digest = keccak_256(rlp([nonce, gasPrice, gas, to, value, data, int(chainId), int(0), int(0)]));
  const pub = new secp.Signature(toBig(r), toBig(s), rec).recoverPublicKey(digest).toRawBytes(false);
  return { nonce: Number(toBig(nonce)), gasPrice: toBig(gasPrice), gas: toBig(gas), to: to.length ? "0x" + hex(to) : null, value: toBig(value),
           data: "0x" + hex(data), from: "0x" + hex(keccak_256(pub.slice(1)).slice(12)), hash: "0x" + hex(keccak_256(unhex(raw.slice(2)))) };
}
const word = (d, i) => BigInt("0x" + d.slice(10 + i * 64, 10 + (i + 1) * 64));
const addrAt = (d, i) => "0x" + d.slice(10 + i * 64 + 24, 10 + (i + 1) * 64);
const W = (v) => "0x" + BigInt(v).toString(16).padStart(64, "0");
const randAddr = (r) => "0x" + [...Array(20)].map(() => r.int(256).toString(16).padStart(2, "0")).join("");

// --- the world -----------------------------------------------------------------------------
class World {
  constructor(seed, f) {
    this.r = rng(seed); this.f = f; this.now = 1_800_000_000_000; this.start = this.now;
    const k = newKey(); this.keys = { evmKey: k, evm: evmAddress(k).toLowerCase(), rvn: "R-unused" };
    this.me = this.keys.evm;
    this.eth = f.startEth ?? 2_000_000_000_000_000n; // 0.002 ETH of gas to start
    this.wprl = 0n; this.usdt = 0n; this.allowance = 0n;
    this.nonce = 0; this.mempool = new Map(); this.held = new Map(); this.receipts = new Map(); this.block = 1;
    this.baseFee = 50_000_000n; // 0.05 gwei
    this.rW = 225_000n * 10n ** 8n; this.rU = 450_000n * 10n ** 6n; // $2.00, ~$900k deep
    this.minted = 0n; this.swappedIn = 0n; this.fromSwaps = 0n; this.refunded = 0n; this.toPayins = 0n; this.gasSpent = 0n;
    this.xs = new Map(); this.payins = new Map(); this.deliveredWei = 0n; this.violations = []; this.calls = 0;
    this.swaps = [];
  }
  p(k) { return this.f[k] || 0; }
  mid() { return Number(this.rU) / 1e6 / (Number(this.rW) / 1e8); } // USD per WPRL
  out(amountIn) { const inAfter = (amountIn * 9_900n) / 10_000n; return (inAfter * this.rU) / (this.rW + inAfter); }

  advance() {
    this.now += TICK;
    if (this.f.topUpAt !== undefined && this.tick() === this.f.topUpAt) this.eth += 2_000_000_000_000_000n; // the operator sends gas
    // PearlBridge mints now and then.
    if (this.r.chance(this.f.mints ?? 0.08)) { const m = BigInt(10 + this.r.int(400)) * 10n ** 8n; this.wprl += m; this.minted += m; }
    // Others trade: the price drifts, and sometimes lurches.
    if (this.r.chance(this.p("shove"))) this.trade(1 + (this.r() - 0.5) * 0.6);
    else if (this.r.chance(0.3)) this.trade(1 + (this.r() - 0.5) * 0.02);
    for (const c of this.f.conditions || []) if (c.what === "gas" && this.tick() >= c.from && this.tick() < c.to) this.baseFee = 60n * 10n ** 9n;
    if (!(this.f.conditions || []).some((c) => c.what === "gas" && this.tick() >= c.from && this.tick() < c.to)) this.baseFee = BigInt(40_000_000 + this.r.int(30_000_000));
    if (!this.r.chance(this.p("stall"))) this.mine();
    for (const x of this.xs.values()) this.cnAdvance(x);
  }
  tick() { return Math.floor((this.now - this.start) / TICK); }
  // Somebody else moves the pool to `factor` times its price, keeping k.
  trade(factor) {
    const k = this.rW * this.rU, price = this.mid() * factor;
    const rW = BigInt(Math.floor(Math.sqrt(Number(k) / (price * 1e6 / 1e8)))); // price = (rU/1e6)/(rW/1e8)
    if (rW > 0n) { this.rW = rW; this.rU = k / rW; }
  }

  mine() {
    this.block += 1;
    for (;;) {
      const n = this.nonce, h = this.held.get(n);
      if (h && h.until > this.now) break;
      if (h) { this.held.delete(n); this.mempool.set(n, h.tx); }
      const tx = this.mempool.get(n);
      if (!tx || tx.gasPrice < this.baseFee) break;
      if (this.f.sandwich && tx.to === MAINNET.router && this.r.chance(this.f.sandwich)) this.trade(0.96); // front-run by 4%
      const gasUsed = tx.to === MAINNET.router ? 150_000n : 50_000n;
      const fee = gasUsed * tx.gasPrice;
      if (this.eth < fee) break;
      this.eth -= fee; this.gasSpent += fee;
      const okExec = this.exec(tx);
      this.nonce = n + 1;
      this.receipts.set(tx.hash, { status: okExec, block: this.block });
      this.mempool.delete(n);
    }
    for (const k of [...this.mempool.keys()]) if (k < this.nonce) this.mempool.delete(k);
    for (const k of [...this.held.keys()]) if (k < this.nonce) this.held.delete(k);
  }

  exec(tx) {
    if (tx.value > 0n) this.violations.push(`ETH sent on mainnet to ${tx.to}`);
    const sel = tx.data.slice(2, 10);
    if (tx.to === MAINNET.wprl && sel === "095ea7b3") {
      if (addrAt(tx.data, 0) !== MAINNET.router) this.violations.push(`WPRL approved to ${addrAt(tx.data, 0)}`);
      this.allowance = word(tx.data, 1); return true;
    }
    if (tx.to === MAINNET.router && sel === "04e45aaf") {
      const [tin, tout, fee, recip, amountIn, minOut] = [addrAt(tx.data, 0), addrAt(tx.data, 1), word(tx.data, 2), addrAt(tx.data, 3), word(tx.data, 4), word(tx.data, 5)];
      if (tin !== MAINNET.wprl || tout !== MAINNET.usdt || fee !== 10_000n) { this.violations.push("a swap of something else"); return false; }
      if (recip !== this.me) this.violations.push(`a swap paying ${recip}`);
      if (this.allowance < amountIn || this.wprl < amountIn) return false;
      const out = this.out(amountIn);
      if (out < minOut) return false; // the router's revert: too little out
      const signedAt = tx.signedMid || this.mid();
      const price = Number(out) / 1e6 / (Number(amountIn) / 1e8);
      if (price < signedAt * 0.95) this.violations.push(`a swap filled at $${price.toFixed(4)} against $${signedAt.toFixed(4)} when signed`);
      this.allowance -= amountIn; this.wprl -= amountIn; this.rW += amountIn; this.rU -= out; this.usdt += out;
      this.swappedIn += amountIn; this.fromSwaps += out; this.swaps.push({ amountIn, out, price, mid: signedAt });
      return true;
    }
    if (tx.to === MAINNET.usdt && sel === "a9059cbb") {
      const to = addrAt(tx.data, 0), amt = word(tx.data, 1);
      if (this.usdt < amt) return false;
      const x = this.payins.get(to);
      if (!x) this.violations.push(`USDT sent to ${to}, no exchange's deposit`);
      else {
        if (x.lie) this.violations.push(`USDT sent to exchange ${x.id}, whose answer was a lie (${x.lie})`);
        if (x.funded > 0n) this.violations.push(`exchange ${x.id} funded twice`);
        if (amt !== x.amount) this.violations.push(`exchange ${x.id} funded with ${amt}, made for ${x.amount}`);
        x.funded += amt; x.fundedAt = this.now;
      }
      this.usdt -= amt; this.toPayins += amt; return true;
    }
    this.violations.push(`a transaction to ${tx.to} (${sel})`);
    return false;
  }

  // ChangeNOW's side of an exchange.
  cnAdvance(x) {
    if (x.funded === 0n || x.status === "finished" || x.status === "failed" || x.status === "refunded") return;
    if (x.status === "waiting") { x.status = "confirming"; x.step = 0; if (this.r.chance(this.p("cnFail"))) x.doom = "failed"; else if (this.r.chance(this.p("cnRefund"))) x.doom = "refunded"; else if (this.r.chance(this.p("cnStuck"))) x.stuckUntil = this.now + 24 * 3_600_000; return; }
    x.step += 1;
    if (x.stuckUntil && this.now < x.stuckUntil) { x.status = "exchanging"; return; }
    if (x.step < 3) { x.status = ["confirming", "exchanging", "sending"][x.step]; return; }
    if (x.doom === "failed") { x.status = "failed"; this.lost = (this.lost || 0n) + x.funded; return; }
    if (x.doom === "refunded") { x.status = "refunded"; this.usdt += x.funded; this.refunded += x.funded; return; }
    x.status = "finished";
    // ~ the dollar value at $2400/ETH, less ChangeNOW's cut
    this.deliveredWei += (x.funded * 10n ** 12n * 98n) / 100n / 2400n;
  }

  rpcOne(method, params) {
    const ok = (result) => ({ result }), err = (message, code = -32000) => ({ error: { code, message } });
    switch (method) {
      case "eth_chainId": return ok("0x1");
      case "eth_gasPrice": return ok("0x" + this.baseFee.toString(16));
      case "eth_blockNumber": return ok("0x" + this.block.toString(16));
      case "eth_getBalance": return ok("0x" + (params[0].toLowerCase() === this.me ? this.eth : 0n).toString(16));
      case "eth_getTransactionCount": return ok("0x" + this.nonce.toString(16));
      case "eth_getTransactionReceipt": { const r = this.receipts.get(params[0]); return ok(r && this.block - r.block >= (this.f.receiptLag || 0) ? { status: r.status ? "0x1" : "0x0", blockNumber: "0x" + r.block.toString(16) } : null); }
      case "eth_call": {
        const { to, data } = params[0], sel = data.slice(2, 10), who = addrAt(data, 0);
        if (to === MAINNET.wprl && sel === "70a08231") return ok(W(who === this.me ? this.wprl : 0n));
        if (to === MAINNET.usdt && sel === "70a08231") return ok(W(who === this.me ? this.usdt : 0n));
        if (to === MAINNET.wprl && sel === "dd62ed3e") return ok(W(who === this.me && addrAt(data, 1) === MAINNET.router ? this.allowance : 0n));
        if (to === MAINNET.quoter && sel === "c6a5026a") return ok(W(this.out(word(data, 2))) + "0".repeat(192));
        return err("execution reverted");
      }
      case "eth_sendRawTransaction": {
        let tx;
        try { tx = decodeTx(params[0], 1); } catch (e) { return err(e.message); }
        if (tx.from !== this.me) this.violations.push(`a transaction signed by ${tx.from}`);
        if (tx.nonce < this.nonce) return err("nonce too low");
        const old = this.mempool.get(tx.nonce);
        if (old && old.hash === tx.hash) return err("already known");
        if (old && tx.gasPrice * 10n < old.gasPrice * 11n) return err("replacement transaction underpriced");
        if (this.eth < tx.gas * tx.gasPrice) return err("insufficient funds for gas * price + value");
        tx.signedMid = this.mid();
        this.mempool.set(tx.nonce, tx);
        if (this.f.lateMine && !this.held.has(tx.nonce) && this.r.chance(this.f.lateMine)) this.held.set(tx.nonce, { tx, until: this.now + 3 * 3_600_000 });
        if (this.r.chance(this.p("ambiguous"))) return err("internal error", -32603);
        return ok(tx.hash);
      }
    }
    return err(`no method ${method}`, -32601);
  }

  changenow(url, init) {
    const u = new URL(url);
    if (u.pathname.endsWith("/exchange/min-amount")) return [200, { minAmount: 1.5 }];
    if (u.pathname.endsWith("/exchange") && init.method === "POST") {
      const b = JSON.parse(init.body);
      if (b.fromCurrency !== "usdt" || b.fromNetwork !== "eth" || b.toCurrency !== "eth" || b.toNetwork !== "hood") this.violations.push(`an exchange asked for ${b.fromCurrency}/${b.fromNetwork} to ${b.toCurrency}/${b.toNetwork}`);
      if (String(b.address).toLowerCase() !== this.me || String(b.refundAddress).toLowerCase() !== this.me) this.violations.push("an exchange paying or refunding somewhere else");
      const id = "cn" + (this.xs.size + 1);
      let payin = randAddr(this.r), payout = this.me, lie = null, extra;
      if (this.r.chance(this.p("cnLies"))) {
        const how = this.r.int(4);
        if (how === 0) { payout = randAddr(this.r); lie = "pays elsewhere"; }
        if (how === 1) { payin = this.me; lie = "deposit is the pool's own"; }
        if (how === 2) { payin = "0x1234"; lie = "deposit not an address"; }
        if (how === 3) { extra = "memo-42"; lie = "wants a memo"; }
      }
      const amount = BigInt(Math.round(Number(b.fromAmount) * 1e6));
      const x = { id, payin: payin.toLowerCase(), amount, funded: 0n, status: "waiting", lie };
      this.xs.set(id, x);
      if (/^0x[0-9a-f]{40}$/.test(x.payin)) this.payins.set(x.payin, x);
      return [200, { id, payinAddress: payin, payoutAddress: payout, ...(extra ? { payinExtraId: extra } : {}), fromAmount: b.fromAmount }];
    }
    if (u.pathname.endsWith("/exchange/by-id")) { const x = this.xs.get(u.searchParams.get("id")); return x ? [200, { id: x.id, status: x.status }] : [404, { error: "not found" }]; }
    return [404, { error: "?" }];
  }

  async fetch(url, init = {}) {
    this.calls += 1;
    const r = this.r;
    if (r.chance(this.p("net"))) throw new TypeError("fetch failed: connection reset");
    if (r.chance(this.p("http429"))) return new Response("slow down", { status: 429 });
    if (r.chance(this.p("http5xx"))) return new Response("<html>502</html>", { status: 502 });
    let status, body;
    if (MAINNET_RPCS.some((m) => url.startsWith(m))) {
      const req = JSON.parse(init.body);
      if (r.chance(this.p("garbage"))) [status, body] = [200, "<html>cloudflare</html>"];
      else if (r.chance(this.p("rateLimit"))) [status, body] = [200, { jsonrpc: "2.0", id: req.id, error: { code: -32005, message: "Too Many Requests" } }];
      else [status, body] = [200, { jsonrpc: "2.0", id: req.id, ...this.rpcOne(req.method, req.params) }];
    } else if (url.startsWith(CN)) [status, body] = this.changenow(url, init);
    else throw new Error(`the leg reached for ${url}`);
    if (r.chance(this.p("evict"))) { this.evicted = true; throw new Evicted("evicted after the request landed"); }
    return new Response(typeof body === "string" ? body : JSON.stringify(body), { status });
  }
}

// --- one run -------------------------------------------------------------------------------
async function run({ seed, faults = {}, ticks, calm = 400, mode = "live" }) {
  const w = new World(seed, faults);
  const m = new Map();
  const storage = {
    get: async (k) => (Array.isArray(k) ? new Map(k.filter((x) => m.has(x)).map((x) => [x, structuredClone(m.get(x))])) : m.has(k) ? structuredClone(m.get(k)) : undefined),
    put: async (k, v) => { for (const [key, val] of Object.entries(typeof k === "string" ? { [k]: v } : k)) m.set(key, structuredClone(val)); },
    delete: async (k) => { for (const x of Array.isArray(k) ? k : [k]) m.delete(x); },
  };
  const logs = [];
  const pool = { env: { TREASURY: mode, PRL_LEG: mode, CHANGENOW_KEY: "k", RPC_URLS: "https://rpc.4663.example" }, ctx: { storage }, log: (l) => logs.push(`${w.tick()} ${l}`), treasury: async () => w.keys };
  const io = { fetch: (u, i) => w.fetch(u, i), now: () => w.now, sleep: async (ms) => { w.now += Math.min(ms, 60_000); } };
  let t = new Treasury(pool, io), evictions = 0;
  const check = (i) => {
    if (w.minted !== w.wprl + w.swappedIn) w.violations.push(`tick ${i}: WPRL minted ${w.minted}, held ${w.wprl} + swapped ${w.swappedIn}`);
    if (w.fromSwaps + w.refunded !== w.usdt + w.toPayins) w.violations.push(`tick ${i}: USDT in ${w.fromSwaps}+${w.refunded}, held ${w.usdt} + sent ${w.toPayins}`);
    const st = m.get("treasury.prl") || {};
    if (st.paused && st.paused.until - w.now > PRL_CFG.pauseMs + TICK) w.violations.push(`tick ${i}: paused past its limit`);
  };
  for (let i = 0; i < ticks + calm; i++) {
    if (i === ticks) w.f = { mints: 0, topUpAt: w.f.topUpAt }; // the calm: no faults, no new WPRL
    w.advance();
    // As the treasury's tick does: an unanswered read ends the step, nothing else.
    try { await t.prl.tick(w.keys); } catch (e) { if (!(e instanceof Evicted) && !(e instanceof Transient)) throw e; }
    if (w.evicted) { w.evicted = false; t = new Treasury(pool, io); evictions += 1; }
    if (w.calls > 60) w.violations.push(`tick ${i}: ${w.calls} requests in one tick`);
    w.calls = 0;
    if (mode === "live") check(i);
  }
  return { w, st: m.get("treasury.prl") || {}, main: m.get("treasury.state") || {}, logs, evictions, storage: m };
}

const settled = (res) => {
  const { w, st } = res;
  const dustW = (Number(w.wprl) / 1e8) * w.mid(), dustU = Number(w.usdt) / 1e6;
  return { done: !st.pending && !st.exchange && dustW < PRL_CFG.minSwapUsd && dustU < PRL_CFG.minSendUsdt, dustW, dustU };
};
const summary = (res) => {
  const { w, evictions } = res;
  const mintedUsd = (Number(w.minted) / 1e8) * 2, delivered = Number(w.deliveredWei) / 1e18 * 2400;
  return `${(Number(w.minted) / 1e8).toFixed(0)} WPRL minted (~$${mintedUsd.toFixed(0)}), ${w.swaps.length} swap(s), ${w.xs.size} exchange(s), $${delivered.toFixed(0)} delivered on 4663, gas ${(Number(w.gasSpent) / 1e18).toFixed(6)} ETH, ${evictions} eviction(s)`;
};
const clean = (res, name) => ok(res.w.violations.length === 0, `${name}: no invariant broken${res.w.violations.length ? ` -- ${[...new Set(res.w.violations)].slice(0, 4).join(" | ")}` : ""}`);

// --- the decision, alone --------------------------------------------------------------------
{
  const f = { now: 0, eth: 10n ** 18n, wprl: 0n, usdt: 0n, allowance: 0n, gasPrice: 50_000_000n, unitUsd: 2 };
  ok(decidePrl({}, f).kind === "idle", "nothing held: idle");
  ok(decidePrl({}, { ...f, wprl: 5n * 10n ** 8n }).kind === "idle", "$10 of WPRL: under the swap floor, idle");
  ok(decidePrl({}, { ...f, wprl: 50n * 10n ** 8n }).kind === "approve", "$100 of WPRL, no allowance: approve first");
  ok(decidePrl({}, { ...f, wprl: 50n * 10n ** 8n, allowance: 1n << 255n }).kind === "swap", "with the allowance: swap");
  ok(decidePrl({}, { ...f, usdt: 30n * 10n ** 6n, cnMin: 1.5 }).kind === "create_exchange", "$30 of USDT: an exchange");
  ok(decidePrl({ exchange: { id: "x", usdt: "30000000", sent: false } }, { ...f, usdt: 30n * 10n ** 6n }).kind === "fund_exchange", "an exchange made and not funded: fund it");
  ok(decidePrl({ exchange: { id: "x", usdt: "30000000", sent: false } }, { ...f, usdt: 29n * 10n ** 6n }).kind === "drop_exchange", "made for more than is held: dropped, never part-funded");
  ok(decidePrl({ pending: { kind: "swap", hash: "0x1" } }, { ...f, wprl: 50n * 10n ** 8n }).kind === "wait", "a transaction in flight: nothing new");
  ok(decidePrl({}, { ...f, wprl: 50n * 10n ** 8n, gasPrice: 60n * 10n ** 9n }).kind === "wait", "gas over the cap: wait");
}

// --- worlds ---------------------------------------------------------------------------------
{
  const res = await run({ seed: 1, ticks: 4 * DAY });
  clean(res, "a quiet world");
  const s = settled(res);
  ok(s.done && res.w.swaps.length >= 3 && res.w.deliveredWei > 0n, `a quiet world moves every WPRL through to ETH on 4663: ${summary(res)}`);
  // What the leg controls: each swap near the price it saw, and what ChangeNOW
  // was sent arriving (its cut aside). The pool's own drift is the market's.
  const fills = res.w.swaps.map((x) => x.price / x.mid);
  const worst = Math.min(...fills);
  const delivered = Number(res.w.deliveredWei) / 1e18 * 2400, sent = Number(res.w.toPayins) / 1e6;
  ok(worst > 0.97 && delivered > sent * 0.97, `swaps fill within 3% of the price seen (worst ${(worst * 100).toFixed(2)}%), and $${delivered.toFixed(0)} arrives of $${sent.toFixed(0)} sent to ChangeNOW`);
}
{
  const res = await run({ seed: 2, ticks: 2 * DAY, mode: "dry" });
  ok(res.w.mempool.size === 0 && res.w.swaps.length === 0 && res.w.xs.size === 0 && res.storage.size === 0, `dry mode signs nothing, makes no exchange, stores nothing (${res.logs.filter((l) => /dry run/.test(l)).length} would-be actions logged)`);
}
{
  const res = await run({ seed: 3, ticks: 5 * DAY, faults: { cnLies: 0.5 } });
  clean(res, "a ChangeNOW that lies half the time");
  ok(res.logs.some((l) => /exchange not created/.test(l)) && settled(res).done, `lies are refused before a cent moves, and the rest goes through: ${summary(res)}`);
}
{
  const res = await run({ seed: 4, ticks: 6 * DAY, faults: { cnFail: 0.3, cnRefund: 0.3, cnStuck: 0.3 } });
  clean(res, "ChangeNOW failing, refunding and sitting on funds");
  ok(settled(res).done && res.main.incidents && res.main.incidents.some((x) => /Pearl leg/.test(x.why)), `refunds are tried again, failures and delays are said: ${summary(res)}`);
}
{
  const res = await run({ seed: 5, ticks: 5 * DAY, faults: { shove: 0.3, sandwich: 0.3 } });
  clean(res, "a pool pushed around and swaps front-run");
  ok(res.logs.some((l) => /off its recent median|reverted/.test(l)) && settled(res).done, `a moved price waits, a front-run swap reverts instead of filling badly, and it all goes through: ${summary(res)}`);
}
{
  const res = await run({ seed: 6, ticks: 6 * DAY, faults: { stall: 0.5, lateMine: 0.3, ambiguous: 0.4, receiptLag: 2 } });
  clean(res, "a chain that stalls, mines late and says it did not");
  ok(settled(res).done, `stuck and late transactions are adopted, never doubled: ${summary(res)}`);
}
{
  const res = await run({ seed: 7, ticks: 5 * DAY, faults: { conditions: [{ what: "gas", from: DAY, to: 2 * DAY }] } });
  clean(res, "gas over the cap for a day");
  ok(settled(res).done, `gas at 60 gwei for a day: it waits, then goes on: ${summary(res)}`);
}
{
  const res = await run({ seed: 8, ticks: 5 * DAY, faults: { net: 0.1, http429: 0.08, http5xx: 0.06, rateLimit: 0.08, garbage: 0.05, evict: 0.05 } });
  clean(res, "flaky RPCs and evictions");
  ok(settled(res).done, `flaky endpoints and an object evicted at random: ${summary(res)}`);
}
{
  // No mainnet gas at the start: nothing is signed, the operator is told once,
  // and once gas arrives everything waiting goes through by itself.
  const res = await run({ seed: 9, ticks: 3 * DAY, faults: { startEth: 0n, topUpAt: DAY } });
  clean(res, "no gas until the operator sends some");
  const told = (res.main.incidents || []).filter((x) => /too little gas/.test(x.why)).length;
  const before = res.logs.filter((l) => Number(l.split(" ")[0]) < DAY && /sent at nonce/.test(l)).length;
  ok(told === 1 && before === 0 && settled(res).done, `no gas: nothing signed for a day (${before}), the operator told once (${told}), then it all goes through: ${summary(res)}`);
}
{
  // Many storms, every fault at a random strength.
  const seeds = Number(process.argv[2] || 12);
  let bad = 0;
  for (let s = 100; s < 100 + seeds; s++) {
    const r0 = rng(s * 7919);
    const faults = { shove: r0() * 0.3, sandwich: r0() * 0.3, stall: r0() * 0.5, lateMine: r0() * 0.3, ambiguous: r0() * 0.4, receiptLag: r0.int(3),
                     net: r0() * 0.1, http429: r0() * 0.08, http5xx: r0() * 0.06, rateLimit: r0() * 0.08, garbage: r0() * 0.05, evict: r0() * 0.05,
                     cnLies: r0() * 0.3, cnFail: r0() * 0.2, cnRefund: r0() * 0.2, cnStuck: r0() * 0.2 };
    const res = await run({ seed: s, faults, ticks: 4 * DAY });
    const st = settled(res);
    if (res.w.violations.length || !st.done) { bad += 1; console.log(`  seed ${s}: ${[...new Set(res.w.violations)].slice(0, 3).join(" | ") || `not settled: ${JSON.stringify({ ...st, pending: res.st.pending && res.st.pending.kind, exchange: res.st.exchange && res.st.exchange.id })}`}`); }
  }
  ok(bad === 0, `${seeds} random storms: ${bad} broke an invariant or left value unmoved`);
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
