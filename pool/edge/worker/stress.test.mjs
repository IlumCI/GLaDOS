// node pool/edge/worker/stress.test.mjs [seeds]
//
// The treasury against a world that is out to get it. Everything runner.js
// talks to is simulated here -- a Ravencoin chain that checks inputs and fees,
// ChangeNOW with deposits, expiries, refunds and failures, chain 4663 with real
// signature recovery, nonces, a mempool that replaces and underprices, reverts
// and receipts that lag -- and every seam between them can refuse, rate-limit,
// time out, answer garbage, accept-then-error, or evict the object mid-step.
//
// It checks what is true in the *world*, never what the treasury says about
// itself, after every tick:
//
//   money     no coin spent twice, no exchange funded twice, nothing sent to an
//             exchange that lapsed or lied, no ETH anywhere but gas and the
//             payout contract, at most one contract ever deployed
//   payouts   every payout mined is recorded as exactly what it paid, at most
//             one mined payout is ever unrecorded (the one in flight), never a
//             contract, the pair or an address under the gate, and no address
//             credited more work than it did
//   ledger    the snapshot never moves backwards
//   secrets   no private key in any log line
//   dry       dry mode writes nothing, sends nothing, creates nothing
//   liveness  a quiet world pays; a hostile one pays, and after every outage
//             it pays again by itself
//
// And after the storm, a calm: faults off, time to settle, then the books must
// balance. Nothing may ever be left halted waiting for a person, and every
// incident the treasury records must have reached the operator's webhook.
import * as secp from "@noble/secp256k1";
import { keccak_256 } from "@noble/hashes/sha3";
import { Treasury, TOKEN, PAIR, WETH, getBig, CFG } from "./runner.js";
import { newKey, rvnAddress, evmAddress, rvnScript, rvnTxid, rlp, int, createdAddress, amountOut, hex, unhex } from "./treasury.js";
import { hasCode } from "./epoch.js";

const RPC = "https://rpc.mainnet.chain.robinhood.com";
const BLOCKBOOK = "https://blockbook.ravencoin.org/api/v2";
const CHANGENOW = "https://api.changenow.io/v2";
const TICK = 5 * 60_000;
const G = 10n ** 18n;

// --- a seeded world ------------------------------------------------------------------------

function rng(seed) {
  let a = seed >>> 0;
  const f = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  f.int = (n) => Math.floor(f() * n);
  f.pick = (xs) => xs[f.int(xs.length)];
  f.chance = (p) => f() < p;
  return f;
}

class Evicted extends Error {}
const fakeTxid = (r) => [...Array(32)].map(() => r.int(256).toString(16).padStart(2, "0")).join("");
const addrLike = (r) => "0x" + [...Array(20)].map(() => r.int(256).toString(16).padStart(2, "0")).join("");

// --- decoders the world uses to read what the treasury sends -------------------------------

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

function decodeEvmTx(raw, chainId) {
  const [f] = rlpDecode(unhex(raw.replace(/^0x/, "")));
  const [nonce, gasPrice, gas, to, value, data, v, r, s] = f;
  const vv = toBig(v);
  const rec = Number(vv - BigInt(chainId) * 2n - 35n);
  if (rec !== 0 && rec !== 1) throw new Error("invalid sender: wrong chain id");
  const digest = keccak_256(rlp([nonce, gasPrice, gas, to, value, data, int(chainId), int(0), int(0)]));
  const pub = new secp.Signature(toBig(r), toBig(s), rec).recoverPublicKey(digest).toRawBytes(false);
  const from = "0x" + hex(keccak_256(pub.slice(1)).slice(12));
  return { nonce: Number(toBig(nonce)), gasPrice: toBig(gasPrice), gas: toBig(gas), to: to.length ? "0x" + hex(to) : null,
           value: toBig(value), data: "0x" + hex(data), from, hash: "0x" + hex(keccak_256(unhex(raw.slice(2)))) };
}

function decodeRvnTx(rawHex) {
  const b = unhex(rawHex);
  let p = 0;
  const u32 = () => { const v = b[p] | (b[p + 1] << 8) | (b[p + 2] << 16) | (b[p + 3] << 24); p += 4; return v >>> 0; };
  const vi = () => { const x = b[p++]; if (x < 0xfd) return x; if (x === 0xfd) { const v = b[p] | (b[p + 1] << 8); p += 2; return v; } p -= 1; throw new Error("big varint"); };
  u32();
  const ins = [], outs = [];
  for (let n = vi(), i = 0; i < n; i++) {
    const txid = hex(b.slice(p, p + 32).reverse()); p += 32;
    const vout = u32(); const sl = vi(); p += sl; u32();
    ins.push(`${txid}:${vout}`);
  }
  for (let n = vi(), i = 0; i < n; i++) {
    let v = 0n; for (let k = 7; k >= 0; k--) v = v * 256n + BigInt(b[p + k]); p += 8;
    const sl = vi(); outs.push({ sats: v, script: hex(b.slice(p, p + sl)) }); p += sl;
  }
  u32();
  if (p !== b.length) throw new Error("trailing bytes");
  return { ins, outs, bytes: b.length };
}

function decodeBuyAndPay(data) {
  const d = data.slice(10);
  const w = (i) => BigInt("0x" + d.slice(i * 64, i * 64 + 64));
  const minOut = w(1), n = Number(w(2));
  return { minOut, recipients: [...Array(n).keys()].map((i) => "0x" + d.slice((3 + i) * 64 + 24, (4 + i) * 64)) };
}

// --- the world -----------------------------------------------------------------------------

class World {
  constructor(seed, faults) {
    this.r = rng(seed);
    this.f = faults;
    this.now = 1_800_000_000_000;
    this.start = this.now;
    this.alerts = [];
    this.keys = (() => {
      const k = { rvnKey: newKey(), evmKey: newKey() };
      return { ...k, rvn: rvnAddress(k.rvnKey), evm: evmAddress(k.evmKey).toLowerCase() };
    })();
    this.hotScript = hex(rvnScript(this.keys.rvn));
    this.violations = [];
    this.lost = 0n;

    // Ravencoin.
    this.height = 1000;
    this.utxos = new Map(); // outpoint -> {script, sats, height|null, coinbase}
    this.spentBy = new Map(); // outpoint -> txid
    this.rtx = new Map(); // txid -> {height|null, outs}
    this.deposits = new Map(); // payin script -> [txid]
    this.utxoListsMempoolSpent = this.r.chance(0.5);

    // ChangeNOW.
    this.cnMin = 15_170_121_340n;
    this.weiPerRvn = 544_500_000_000n; // 0.0001089 ETH per 200 RVN, measured
    this.xs = new Map();
    this.xn = 0;

    // Chain 4663.
    this.block = 5_000_000;
    this.baseFee = 22_600_000n;
    this.eth = new Map([[this.keys.evm, 0n]]);
    this.nonce = new Map();
    this.mempool = new Map(); // nonce -> tx (our account only)
    this.receipts = new Map(); // hash -> {status, block}
    this.code = new Map([[PAIR, "0x60806040"], [TOKEN, "0x60806040"], [WETH, "0x60806040"]]);
    this.tok = new Map();
    this.rW = 2n * G;
    this.rG = 400_000_000n * G;
    this.tax = 100n;
    this.contracts = [];
    this.payouts = []; // mined buyAndPay: {hash, recipients, each}
    this.valueTo = new Map();

    // Miners: most gated, some not, and the ones that must never be paid.
    this.miners = [];
    for (let i = 0; i < Number(process.env.MINERS || 24); i++) {
      const a = addrLike(this.r);
      this.miners.push({ addr: a, rate: BigInt(1 + this.r.int(40)) * 1000n, shard: this.r.int(2), both: this.r.chance(0.2) });
      this.tok.set(a, this.r.chance(0.8) ? 50_000n * G + BigInt(this.r.int(100_000)) * G : BigInt(this.r.int(49_999)) * G);
    }
    const special = (addr, code, bal) => {
      this.miners.push({ addr, rate: 30_000n, shard: this.r.int(2), special: true });
      if (code) this.code.set(addr, code);
      this.tok.set(addr, bal);
    };
    special(PAIR, null, 10n ** 27n); // the pair holds plenty and must never be paid
    const safe = addrLike(this.r); special(safe, "0x6080604052", 60_000n * G); // a contract wallet
    const d7702 = addrLike(this.r); special(d7702, "0xef0100" + addrLike(this.r).slice(2), 60_000n * G); // 7702: an ordinary account
    this.eligible7702 = d7702;
    this.work = new Map(this.miners.map((m) => [m.addr, 0n]));
    this.shardDown = false;
  }

  // Faults, per request.
  p(name) { return this.f[name] || 0; }

  // --- time passes ---
  advance() {
    this.now += TICK;
    const r = this.r;
    // Mining pays the pool's RVN wallet about hourly; hostile worlds add dust,
    // swarms of small outputs, and coinbase outputs that are not yet spendable.
    // Income scales with the crowd: more miners is more hashrate is more RVN.
    const crowd = BigInt(Math.max(1, Math.round(this.miners.length / 27)));
    if (r.chance(1 / 12)) this.pay(this.hotScript, BigInt(20 + r.int(60)) * 10n ** 8n * crowd);
    if (r.chance(this.p("dust"))) for (let i = 0; i < 5; i++) this.pay(this.hotScript, 546n + BigInt(r.int(1000)));
    if (r.chance(this.p("smallSwarm"))) for (let i = 0; i < 40; i++) this.pay(this.hotScript, 250_000_000n);
    if (r.chance(this.p("coinbase"))) this.pay(this.hotScript, 100n * 10n ** 8n, true);
    // Work grows.
    for (const m of this.miners) if (r.chance(0.7)) this.work.set(m.addr, this.work.get(m.addr) + m.rate * BigInt(1 + r.int(3)));
    // A block on each chain, sometimes none.
    if (!r.chance(this.p("rvnStall"))) this.mineRvn();
    this.cnAdvance();
    this.baseFee = this.r.chance(this.p("gasSpike")) ? this.baseFee * 3n : (this.baseFee * BigInt(90 + r.int(21))) / 100n;
    if (this.baseFee < 10_000_000n) this.baseFee = 10_000_000n;
    if (this.baseFee > 200_000_000n && !(this.f.conditions || []).some((c) => c.what === "gas")) this.baseFee = 200_000_000n;
    if (!r.chance(this.p("evmStall"))) this.mineEvm();
    // The pair's price wanders, and sometimes somebody shoves it for a block.
    if (this.shoved) { [this.rW, this.rG] = this.shoved; this.shoved = null; }
    if (r.chance(this.p("shove"))) { this.shoved = [this.rW, this.rG]; this.rW = (this.rW * 3n) / 2n; this.rG = (this.rG * 2n) / 3n; }
    if (r.chance(0.05)) { this.rG = (this.rG * BigInt(97 + r.int(7))) / 100n; }
    if (r.chance(this.p("taxHike"))) this.tax = 2000n;
    const tick = Math.floor((this.now - this.start) / TICK);
    for (const c of this.f.conditions || []) {
      const on = tick >= c.from && tick < c.to;
      if (c.what === "tax") this.tax = on ? c.bps : (this.tax === c.bps ? 100n : this.tax);
      if (c.what === "gas" && on) this.baseFee = 2_260_000_000n; // x100
      if (c.what === "rate") this.weiPerRvn = on ? 5_445_000_000n : 544_500_000_000n; // a 99% crash
    }
    this.shardDown = r.chance(this.p("shardDown"));
  }

  pay(script, sats, coinbase = false) {
    const txid = fakeTxid(this.r);
    this.utxos.set(`${txid}:0`, { script, sats, height: this.height, coinbase });
    this.rtx.set(txid, { height: this.height, outs: [{ script, sats }] });
  }

  mineRvn() {
    this.height += 1;
    for (const [txid, t] of this.rtx) if (t.height === null) {
      t.height = this.height;
      t.outs.forEach((_, i) => { const u = this.utxos.get(`${txid}:${i}`); if (u) u.height = this.height; });
    }
  }

  // --- Blockbook ---
  blockbook(url, init) {
    const path = url.slice(BLOCKBOOK.length);
    if (path.startsWith("/utxo/")) {
      if (this.r.chance(this.p("garbage"))) return [200, { error: "backend unavailable" }];
      const out = [];
      for (const [op, u] of this.utxos) {
        if (u.script !== this.hotScript || u.height === null) continue;
        const spender = this.spentBy.get(op);
        if (spender && (this.rtx.get(spender).height !== null || !this.utxoListsMempoolSpent)) continue;
        const [txid, vout] = op.split(":");
        out.push({ txid, vout: Number(vout), value: u.sats.toString(), confirmations: this.height - u.height + 1, coinbase: u.coinbase || undefined,
                   ...(u.coinbase ? { confirmations: 3 } : {}) });
      }
      return [200, out];
    }
    if (path.startsWith("/tx/")) {
      const t = this.rtx.get(path.slice(4));
      if (!t) return [400, { error: "txid not found" }];
      return [200, { txid: path.slice(4), confirmations: t.height === null ? 0 : this.height - t.height + 1 }];
    }
    if (path.startsWith("/sendtx/")) {
      const raw = init.body;
      let tx;
      try { tx = decodeRvnTx(raw); } catch (e) { return [200, { error: { message: `TX decode failed: ${e.message}` } }]; }
      const txid = rvnTxid(raw);
      if (this.rtx.has(txid)) return [200, { error: { message: this.rtx.get(txid).height === null ? "txn-already-in-mempool" : "transaction already in block chain" } }];
      let inSum = 0n;
      for (const op of tx.ins) {
        const u = this.utxos.get(op);
        if (!u || this.spentBy.has(op)) return [200, { error: { message: "bad-txns-inputs-missingorspent" } }];
        if (u.script !== this.hotScript) this.violations.push(`spent a coin that is not the pool's: ${op}`);
        if (u.coinbase) return [200, { error: { message: "bad-txns-premature-spend-of-coinbase" } }];
        inSum += u.sats;
      }
      const outSum = tx.outs.reduce((a, o) => a + o.sats, 0n);
      if (outSum > inSum) return [200, { error: { message: "bad-txns-in-belowout" } }];
      if (inSum - outSum < BigInt(tx.bytes) * 1000n) return [200, { error: { message: "min relay fee not met" } }];
      if (tx.outs.some((o) => o.sats < 546n)) return [200, { error: { message: "dust" } }];
      for (const op of tx.ins) this.spentBy.set(op, txid);
      this.rtx.set(txid, { height: null, outs: tx.outs });
      tx.outs.forEach((o, i) => {
        this.utxos.set(`${txid}:${i}`, { script: o.script, sats: o.sats, height: null });
        if (o.script !== this.hotScript) {
          const d = this.deposits.get(o.script) || [];
          d.push(txid);
          this.deposits.set(o.script, d);
          this.cnDeposit(o.script, o.sats, txid);
        }
      });
      if (this.r.chance(this.p("ambiguous"))) return [500, "upstream timeout"];
      return [200, { result: txid }];
    }
    return [404, { error: "no such route" }];
  }

  // --- ChangeNOW ---
  changenow(url, init) {
    const path = url.slice(CHANGENOW.length);
    if (path.startsWith("/exchange/min-amount")) return [200, this.r.chance(this.p("garbage")) ? { error: "maintenance" } : { minAmount: Number(this.cnMin) / 1e8 }];
    if (path.startsWith("/exchange/estimated-amount")) return [200, { toAmount: 0.0001089 }];
    if (path.startsWith("/exchange/by-id")) {
      const x = this.xs.get(decodeURIComponent(path.split("id=")[1]));
      return x ? [200, { id: x.id, status: x.status, payinAddress: x.payin }] : [400, { error: "not_found" }];
    }
    if (path === "/exchange" && init.method === "POST") {
      const b = JSON.parse(init.body);
      const id = `cn${++this.xn}`;
      const k = newKey();
      let payin = this.r.chance(0.5) ? rvnAddress(k) : rvnAddress(k); // P2PKH is what ChangeNOW hands out for RVN
      let payout = b.address;
      if (this.r.chance(this.p("cnLies"))) payin = "0xnot-a-ravencoin-address";
      if (this.r.chance(this.p("cnLies"))) payout = addrLike(this.r);
      const fate = this.r.chance(this.p("cnFail")) ? "failed" : this.r.chance(this.p("cnRefund")) ? "refunded" : this.r.chance(this.p("cnStuck")) ? "stuck" : "ok";
      let script = null;
      try { script = hex(rvnScript(payin)); } catch {}
      const x = { id, payin, script, sats: BigInt(Math.round(Number(b.fromAmount) * 1e8)), payout: payout.toLowerCase(), refund: b.refundAddress,
                  created: this.now, status: "waiting", fate, got: 0n, at: 0 };
      this.xs.set(id, x);
      return [200, { id, payinAddress: payin, payoutAddress: payout, fromAmount: Number(b.fromAmount), toAmount: 0.0001 }];
    }
    return [404, { error: "no such route" }];
  }

  cnDeposit(script, sats, txid) {
    const x = [...this.xs.values()].find((e) => e.script === script);
    if (!x) { this.lost += sats; this.violations.push(`${sats} sats sent to an address no exchange owns (${txid})`); return; }
    if (x.got > 0n) this.violations.push(`exchange ${x.id} funded twice`);
    if (x.status === "expired") { this.lost += sats; this.violations.push(`exchange ${x.id} funded after it lapsed`); }
    if (x.payout !== this.keys.evm) this.violations.push(`exchange ${x.id} funded while paying ${x.payout}`);
    x.got += sats;
    if (x.status === "waiting") { x.status = "confirming"; x.at = this.now; }
  }

  cnAdvance() {
    for (const x of this.xs.values()) {
      const age = this.now - x.at;
      if (x.status === "waiting" && this.now - x.created > 3 * 3_600_000) x.status = "expired";
      else if (x.status === "confirming" && age > 10 * 60_000) { x.status = "exchanging"; x.at = this.now; }
      else if (x.status === "exchanging" && age > 15 * 60_000) {
        if (x.fate === "stuck") continue;
        if (x.fate === "failed") { x.status = "failed"; continue; }
        if (x.fate === "refunded" || x.got < this.cnMin) { x.status = "refunded"; this.pay(this.hotScript, x.got - 100_000n); continue; }
        x.status = "sending"; x.at = this.now;
      } else if (x.status === "sending" && age > 5 * 60_000) {
        x.status = "finished";
        const wei = x.got * this.weiPerRvn / 10n ** 8n;
        this.eth.set(x.payout, (this.eth.get(x.payout) || 0n) + wei);
      }
    }
  }

  // --- chain 4663 ---
  rpcOne(m, params) {
    const me = this.keys.evm;
    const err = (message, code = -32000) => ({ error: { code, message } });
    switch (m) {
      case "eth_blockNumber": return { result: "0x" + this.block.toString(16) };
      case "eth_gasPrice": return { result: "0x" + this.baseFee.toString(16) };
      case "eth_getBalance": return { result: "0x" + (this.eth.get(params[0].toLowerCase()) || 0n).toString(16) };
      case "eth_getTransactionCount": return { result: "0x" + (this.nonce.get(params[0].toLowerCase()) || 0).toString(16) };
      case "eth_getTransactionReceipt": {
        const rc = this.receipts.get(params[0]);
        if (!rc || rc.block > this.block - (this.f.receiptLag || 0)) return { result: null };
        return { result: { status: rc.status ? "0x1" : "0x0", blockNumber: "0x" + rc.block.toString(16) } };
      }
      case "eth_getCode": return { result: this.code.get(params[0].toLowerCase()) || "0x" };
      case "eth_call": {
        const { to, data } = params[0];
        if (to === PAIR && data === "0x0902f1ac") return { result: "0x" + this.rW.toString(16).padStart(64, "0") + this.rG.toString(16).padStart(64, "0") + "0".repeat(64) };
        if (to === TOKEN && data === "0x691f224f") return { result: "0x" + this.tax.toString(16).padStart(64, "0") };
        if (to === TOKEN && data.startsWith("0x70a08231")) return { result: "0x" + (this.tok.get("0x" + data.slice(-40)) || 0n).toString(16).padStart(64, "0") };
        return err("execution reverted", 3);
      }
      case "eth_estimateGas": {
        const tx = params[0];
        const r = this.exec({ from: tx.from.toLowerCase(), to: tx.to ? tx.to.toLowerCase() : null, value: tx.value ? BigInt(tx.value) : 0n,
                              data: tx.data, gas: 30_000_000n, nonce: this.nonce.get(me) || 0 }, true);
        return r.ok ? { result: "0x" + r.gas.toString(16) } : err(`execution reverted: ${r.why}`, 3);
      }
      case "eth_sendRawTransaction": {
        let tx;
        try { tx = decodeEvmTx(params[0], 4663); } catch (e) { return err(e.message); }
        if (tx.from !== me) this.violations.push(`a transaction from ${tx.from}, not the pool`);
        const n = this.nonce.get(me) || 0;
        if (this.f.stealNonce && this.r.chance(this.f.stealNonce) && tx.nonce === n) {
          // Somebody with the key gets there first.
          this.nonce.set(me, n + 1);
          this.receipts.set("0xevil" + n, { status: true, block: this.block });
          this.stolen = (this.stolen || 0) + 1;
        }
        if (tx.nonce < (this.nonce.get(me) || 0)) return err("nonce too low");
        const old = this.mempool.get(tx.nonce);
        if (old && old.hash === tx.hash) return err("already known");
        if (old && tx.gasPrice * 10n < old.gasPrice * 11n) return err("replacement transaction underpriced");
        if ((this.eth.get(me) || 0n) < tx.value + tx.gas * tx.gasPrice) return err("insufficient funds for gas * price + value");
        if (tx.to && tx.to !== (this.contracts[0] || "").toLowerCase() && tx.value > 0n) this.violations.push(`ETH sent to ${tx.to}`);
        this.mempool.set(tx.nonce, tx);
        if (this.r.chance(this.p("ambiguous"))) return this.r.chance(0.5) ? err("internal error: upstream timed out", -32603) : err("internal error", -32603);
        return { result: tx.hash };
      }
    }
    return err(`method ${m} not supported`, -32601);
  }

  // Runs one call against the chain. `dry` for estimates.
  exec(tx, dry) {
    const me = this.keys.evm;
    const bal = this.eth.get(me) || 0n;
    if (!tx.to) {
      const gas = 743_219n;
      if (tx.gas < gas) return { ok: false, gas: tx.gas, why: "out of gas" };
      if (!dry) { const a = createdAddress(me, tx.nonce).toLowerCase(); this.code.set(a, "0xpayout"); this.contracts.push(a); }
      return { ok: true, gas };
    }
    if (!this.contracts.includes(tx.to)) return { ok: true, gas: 21_000n };
    if (bal < tx.value) return { ok: false, gas: 30_000n, why: "insufficient balance" };
    const { recipients, minOut } = decodeBuyAndPay(tx.data);
    const gas = 166_600n + 50_850n * BigInt(recipients.length);
    if (!recipients.length) return { ok: false, gas: 30_000n, why: "NoRecipients" };
    if (minOut === 0n) return { ok: false, gas: 30_000n, why: "NoSlippageBound" };
    if (tx.gas < gas) return { ok: false, gas: tx.gas, why: "out of gas" };
    const out = amountOut(tx.value, this.rW, this.rG);
    const got = (out * (10_000n - this.tax)) / 10_000n;
    if (got < minOut) return { ok: false, gas: 120_000n, why: "Slippage" };
    const each = got / BigInt(recipients.length);
    for (const [i, a] of recipients.entries()) {
      if (a === PAIR) return { ok: false, gas, why: `ShortPaid(${i})` };
    }
    if (!dry) {
      this.rW += tx.value; this.rG -= out;
      for (const a of recipients) this.tok.set(a, (this.tok.get(a) || 0n) + each);
    }
    return { ok: true, gas, each, recipients };
  }

  mineEvm() {
    this.block += 1;
    const me = this.keys.evm;
    for (;;) {
      const n = this.nonce.get(me) || 0;
      const tx = this.mempool.get(n);
      if (!tx || tx.gasPrice < this.baseFee) break;
      const r = this.exec(tx, false);
      const cost = r.gas * tx.gasPrice + (r.ok ? tx.value : 0n);
      this.eth.set(me, (this.eth.get(me) || 0n) - cost);
      if ((this.eth.get(me) || 0n) < 0n) this.violations.push("the hot wallet went negative");
      if (r.ok && tx.to && tx.value > 0n) this.valueTo.set(tx.to, (this.valueTo.get(tx.to) || 0n) + tx.value);
      this.nonce.set(me, n + 1);
      this.receipts.set(tx.hash, { status: r.ok, block: this.block });
      if (r.ok && r.recipients) this.payouts.push({ hash: tx.hash, recipients: r.recipients, each: r.each });
      this.mempool.delete(n);
    }
    for (const k of [...this.mempool.keys()]) if (k < (this.nonce.get(me) || 0)) this.mempool.delete(k);
  }

  rpc(body) {
    if (Array.isArray(body)) {
      if (this.r.chance(this.p("garbage"))) return [200, { jsonrpc: "2.0", error: { code: -32005, message: "batch limit exceeded, slow down" } }];
      return [200, body.map((c) => ({ jsonrpc: "2.0", id: c.id, ...(this.r.chance(this.p("itemErr")) ? { error: { code: -32000, message: "header not found" } } : this.rpcOne(c.method, c.params)) }))];
    }
    if (this.r.chance(this.p("rateLimit"))) return [200, { jsonrpc: "2.0", id: body.id, error: { code: -32005, message: "Too Many Requests" } }];
    return [200, { jsonrpc: "2.0", id: body.id, ...this.rpcOne(body.method, body.params) }];
  }

  // --- the one door ---
  async fetch(url, init = {}) {
    const r = this.r;
    if (url.startsWith("https://alerts.example/")) { this.alerts.push(JSON.parse(init.body).content); return new Response("ok"); }
    // Timed outages: a dependency gone for a stretch of days, in the way it
    // actually goes -- refusing, erroring, or unreachable.
    const dep = url.startsWith(RPC) ? "rpc" : url.startsWith(BLOCKBOOK) ? "blockbook" : url.startsWith(CHANGENOW) ? "changenow" : "?";
    const tick = Math.floor((this.now - this.start) / TICK);
    for (const o of this.f.outages || []) {
      if (o.dep !== dep || tick < o.from || tick >= o.to) continue;
      if (o.how === "unreachable") throw new TypeError("fetch failed: getaddrinfo ENOTFOUND");
      if (o.how === "401") return new Response(JSON.stringify({ error: "not_valid_api_key", message: "Unauthorized" }), { status: 401 });
      return new Response("<html>503 Service Unavailable</html>", { status: 503 });
    }
    if (r.chance(this.p("net"))) throw new TypeError("fetch failed: connection reset");
    if (r.chance(this.p("http429"))) return new Response("slow down", { status: 429 });
    if (r.chance(this.p("http5xx"))) return new Response("<html>502 Bad Gateway</html>", { status: 502 });
    let status, body;
    if (url.startsWith(RPC)) [status, body] = this.rpc(JSON.parse(init.body));
    else if (url.startsWith(BLOCKBOOK)) [status, body] = this.blockbook(url, init);
    else if (url.startsWith(CHANGENOW)) [status, body] = this.changenow(url, init);
    else throw new Error(`the treasury reached for ${url}`);
    // The object can be evicted the instant after the world acted on a request.
    if (r.chance(this.p("evict"))) { this.evicted = true; throw new Evicted("evicted after the request landed"); }
    return new Response(typeof body === "string" ? body : JSON.stringify(body), { status });
  }

  ledgerDoc(shard) {
    const part = (m) => { const w = this.work.get(m.addr); return m.both ? (shard === 0 ? w / 2n : w - w / 2n) : w; };
    const rows = this.miners.filter((m) => m.both || m.shard === shard)
      .map((m) => `{"worker":"${m.addr}.rig${shard}","coin":"yespowerR16","work":${part(m)}}`);
    // A miner on both shards has its work split between them; the sum is the whole.
    return `{"shares":[${rows.join(",")}]}`;
  }
}

// Storage with the Durable Object's shape, serialised like the real one, and
// able to die: before a write (nothing lands) or after it (the rest of the step
// never runs).
class Storage {
  constructor(w) { this.m = new Map(); this.w = w; this.writes = 0; }
  async get(k) {
    if (Array.isArray(k)) return new Map(k.filter((x) => this.m.has(x)).map((x) => [x, structuredClone(this.m.get(x))]));
    return this.m.has(k) ? structuredClone(this.m.get(k)) : undefined;
  }
  async put(k, v) {
    if (this.w.r.chance(this.w.p("evictBeforeWrite"))) { this.w.evicted = true; throw new Evicted("evicted before a write"); }
    const entries = typeof k === "string" ? { [k]: v } : k;
    for (const [key, val] of Object.entries(entries)) {
      const size = JSON.stringify(val).length;
      if (size > 131_072) throw new Error(`value for ${key} is ${size} bytes, over the 128 KiB limit`);
    }
    for (const [key, val] of Object.entries(entries)) this.m.set(key, structuredClone(val));
    this.writes += 1;
    if (this.w.r.chance(this.w.p("evictAfterWrite"))) { this.w.evicted = true; throw new Evicted("evicted after a write"); }
  }
  async delete(k) { this.m.delete(k); }
}

// --- one run -------------------------------------------------------------------------------

async function run({ seed, faults, ticks, calm = 600, mode = "live", env = {} }) {
  const w = new World(seed, faults);
  const storage = new Storage(w);
  const logs = [];
  const pool = {
    env: { TREASURY: mode, RVN_BATCH: "200", GATE_MIN: "50000", SHARDS: "2", TREASURY_RESUME: "", CHANGENOW_KEY: "k",
           ALERT_WEBHOOK: "https://alerts.example/hook", ...env,
           POOL: { idFromName: (n) => n, get: (n) => ({ fetch: async () => (w.shardDown ? new Response("down", { status: 503 }) : new Response(w.ledgerDoc(1))) }) } },
    ctx: { storage },
    core: { ledger: () => w.ledgerDoc(0) },
    log: (l) => logs.push(`${w.now} ${l}`),
    treasury: async () => w.keys,
  };
  // Blocks here are five minutes apart, not 0.1 s, so the nonce depth is
  // scaled to the simulated lag (at most two blocks) rather than to 4663's.
  const io = { fetch: (u, i) => w.fetch(u, i), now: () => w.now, sleep: async (ms) => { w.now += ms; }, cfg: { nonceDepth: 6 } };
  let t = new Treasury(pool, io);
  let evictions = 0;
  const seenSnap = new Map();
  const check = async (tickNo) => {
    // Recorded epochs against what the chain actually paid.
    const st = storage.m.get("treasury.state") || {};
    const recorded = [];
    for (let i = 1; i <= (st.epochs || 0); i++) recorded.push(await getBig(storage, `treasury.epoch.${i}`));
    const mined = w.payouts;
    if (mined.length - recorded.length > 1) w.violations.push(`tick ${tickNo}: ${mined.length} payouts mined, ${recorded.length} recorded`);
    if (mined.length - recorded.length === 1) {
      const last = mined[mined.length - 1];
      const p = st.pending;
      // Tracked is in flight: pending, carried by the pending one, or kept
      // among earlier attempts to be adopted before anything else is signed.
      const inFlight = (p && (p.hash === last.hash || (p.prior || []).some((x) => x.hash === last.hash))) ||
                       (st.evmPrior && st.evmPrior.txs.some((x) => x.hash === last.hash));
      if (!inFlight) w.violations.push(`tick ${tickNo}: payout ${last.hash} mined, unrecorded and not in flight`);
    }
    for (const rec of recorded) {
      if (!rec) { w.violations.push(`tick ${tickNo}: an epoch record is missing`); continue; }
      const m = mined.find((x) => x.hash === rec.tx);
      if (!m) w.violations.push(`tick ${tickNo}: epoch ${rec.id} names ${rec.tx}, which paid nothing`);
      else if (m.recipients.join() !== rec.recipients.join()) w.violations.push(`tick ${tickNo}: epoch ${rec.id} records recipients the chain did not pay`);
    }
    if (new Set(recorded.map((r) => r && r.tx)).size !== recorded.length) w.violations.push(`tick ${tickNo}: one transaction recorded as two epochs`);
    // Nobody who must not be paid ever is.
    for (const m of mined) for (const a of m.recipients) {
      if (a === PAIR || hasCode(w.code.get(a) || "0x")) w.violations.push(`tick ${tickNo}: paid ${a}, which holds code`);
      const miner = w.miners.find((x) => x.addr === a);
      if (!miner) w.violations.push(`tick ${tickNo}: paid ${a}, which never mined`);
    }
    // Work credited never exceeds work done.
    const credited = new Map();
    for (const rec of recorded) if (rec) for (const a of rec.recipients) credited.set(a, (credited.get(a) || 0n) + BigInt(rec.work[a]));
    for (const [a, c] of credited) if (c > w.work.get(a)) w.violations.push(`tick ${tickNo}: ${a} credited ${c} work of ${w.work.get(a)}`);
    // The snapshot only rises.
    const snap = (await getBig(storage, "treasury.snapshot")) || {};
    for (const [a, v] of Object.entries(snap)) {
      if (seenSnap.has(a) && BigInt(v) < seenSnap.get(a)) w.violations.push(`tick ${tickNo}: snapshot for ${a} fell`);
      seenSnap.set(a, BigInt(v));
    }
    if (st.halted) w.violations.push(`tick ${tickNo}: halted, waiting for a person: ${st.halted}`);
    if (st.paused && st.paused.until - w.now > 24 * 3_600_000 + TICK) w.violations.push(`tick ${tickNo}: paused for more than a day`);
    if (w.contracts.length > 1) w.violations.push(`tick ${tickNo}: ${w.contracts.length} contracts deployed`);
    for (const [to] of w.valueTo) if (to !== w.contracts[0]) w.violations.push(`tick ${tickNo}: ETH paid to ${to}`);
  };
  const total = ticks + calm;
  for (let i = 0; i < total; i++) {
    if (i === ticks) { w.f = { receiptLag: 0, conditions: (w.f.conditions || []).filter((c) => c.to > ticks) }; } // the calm
    w.advance();
    try {
      await t.tick(0);
    } catch (e) {
      if (!(e instanceof Evicted)) throw e;
    }
    // Any eviction during the step -- the tick swallows it as a failed step --
    // takes the object's memory with it.
    if (w.evicted) { w.evicted = false; t = new Treasury(pool, io); evictions += 1; }
    if (mode === "live") await check(i);
  }
  const st = storage.m.get("treasury.state") || {};
  const privs = [w.keys.rvnKey, w.keys.evmKey];
  for (const l of logs) if (privs.some((k) => l.includes(k))) w.violations.push("a private key reached a log line");
  if (mode === "live") {
    if (w.payouts.length !== (st.epochs || 0)) w.violations.push(`after the calm: ${w.payouts.length} mined, ${st.epochs || 0} recorded`);
    // The operator hears about every incident from the treasury itself.
    for (const i of st.incidents || []) if (!w.alerts.some((a) => a.includes(i.why))) w.violations.push(`an incident never reached the webhook: ${i.why}`);
  }
  return { w, st, logs, storage, evictions };
}

// --- scenarios -----------------------------------------------------------------------------

let passed = 0, failed = 0;
const ok = (c, what) => { c ? passed++ : failed++; console.log(`${c ? "ok  " : "FAIL"}  ${what}`); };
const DAY = 288;
const summary = (res) => `${res.st.epochs || 0} epoch(s), ${res.w.payouts.length} mined, ${res.w.xs.size} exchange(s), ${res.evictions} eviction(s), ${(res.st.incidents || []).length} incident(s), ${res.w.alerts.length} alert(s)`;
// Epochs paid after a given tick: recovery is paying again, not merely not crashing.
const paidAfter = async (res, tick) => {
  let n = 0;
  for (let i = 1; i <= (res.st.epochs || 0); i++) {
    const e = await getBig(res.storage, `treasury.epoch.${i}`);
    if (e && e.at >= res.w.start + tick * 5 * 60_000) n += 1;
  }
  return n;
};
const clean = (res, name) => {
  ok(res.w.violations.length === 0, `${name}: no invariant broken${res.w.violations.length ? ` -- ${[...new Set(res.w.violations)].slice(0, 4).join(" | ")}` : ""}`);
};

const HARSH = { net: 0.08, http429: 0.08, http5xx: 0.06, rateLimit: 0.08, garbage: 0.05, itemErr: 0.05, ambiguous: 0.15,
                evict: 0.02, evictBeforeWrite: 0.05, evictAfterWrite: 0.08, receiptLag: 2, rvnStall: 0.3, evmStall: 0.3,
                gasSpike: 0.02, shove: 0.03, dust: 0.1, coinbase: 0.02, shardDown: 0.1 };

// One storm, its faults and its log: STORM=<seed> node stress.test.mjs
function stormFaults(s) {
  const r0 = rng(s * 7919);
  const faults = Object.fromEntries(Object.entries(HARSH).map(([k, v]) => [k, typeof v === "number" && k !== "receiptLag" ? v * r0() * 2 : r0.int(3)]));
  for (const k of ["cnFail", "cnStuck", "cnRefund", "cnLies", "taxHike", "stealNonce", "smallSwarm"]) if (r0.chance(0.2)) faults[k] = r0() * 0.2;
  return faults;
}
// A crowd: LOAD=<miners> node stress.test.mjs -- a quiet world, timed.
if (process.env.LOAD) {
  process.env.MINERS = process.env.LOAD;
  const t0 = performance.now();
  const res = await run({ seed: 1, faults: {}, ticks: Number(process.env.DAYS || 3) * 288, calm: 0 });
  const ms = performance.now() - t0;
  const sizes = [...res.storage.m.entries()].map(([k, v]) => [k, JSON.stringify(v).length]).sort((a, b) => b[1] - a[1]);
  const epochs = [];
  for (let i = 1; i <= (res.st.epochs || 0); i++) epochs.push(await getBig(res.storage, `treasury.epoch.${i}`));
  console.log(`${process.env.LOAD} miners: ${summary(res)}; ${(ms / 1000).toFixed(1)} s wall`);
  console.log(`recipients per epoch: ${epochs.map((e) => e.recipients.length).join(", ") || "none"}`);
  console.log(`largest stored value: ${sizes[0][0]} ${sizes[0][1]} B; ${res.storage.m.size} keys`);
  console.log(`gas per payout: ${res.w.payouts.map((p) => 166_600 + 50_850 * p.recipients.length).join(", ")}`);
  console.log([...new Set(res.w.violations)].slice(0, 3).join("\n"));
  process.exit(0);
}
if (process.env.STORM) {
  const s = Number(process.env.STORM);
  const res = await run({ seed: s, faults: stormFaults(s), ticks: 4 * 288, calm: 400 });
  console.log(JSON.stringify(stormFaults(s)));
  console.log(res.logs.filter((l) => !/holding|waiting on|exchange cn\d+ is/.test(l)).slice(0, Number(process.env.LINES || 400)).join("\n"));
  console.log([...new Set(res.w.violations)].slice(0, 5).join("\n"));
  process.exit(0);
}

{
  const res = await run({ seed: 1, faults: {}, ticks: 6 * DAY });
  clean(res, "a quiet world");
  ok((res.st.epochs || 0) >= 3 && !(res.st.incidents || []).length, `a quiet world pays, with no incidents: ${summary(res)}`);
  ok(res.w.payouts.some((p) => p.recipients.includes(res.w.eligible7702)), "an EIP-7702 account is paid like anyone (the pair and contract wallets never are: an invariant)");
}
{
  const res = await run({ seed: 2, faults: {}, ticks: 3 * DAY, mode: "dry" });
  ok(res.storage.writes === 0 && res.w.xs.size === 0 && [...res.w.rtx.values()].every((x) => x.outs.every((o) => o.script === res.w.hotScript)) && !res.w.alerts.length,
     `dry mode writes nothing, creates no exchange, sends nothing, alerts nobody (${res.logs.filter((l) => /dry run/.test(l)).length} would-be actions logged)`);
}
{
  const res = await run({ seed: 3, faults: HARSH, ticks: 8 * DAY });
  clean(res, "everything failing at once");
  ok((res.st.epochs || 0) >= 1, `everything failing at once still pays: ${summary(res)}`);
}
{
  const res = await run({ seed: 4, faults: { smallSwarm: 0.05, dust: 0.5, coinbase: 0.1 }, ticks: 5 * DAY });
  clean(res, "a wallet buried in dust and small outputs");
  ok((res.st.epochs || 0) >= 1, `dust and swarms of small outputs never stall the swap: ${summary(res)}`);
}
{
  const res = await run({ seed: 5, faults: { cnLies: 0.3 }, ticks: 5 * DAY });
  clean(res, "an exchange that lies about where money goes");
  ok(res.logs.some((l) => /not a Ravencoin address|not the pool's/.test(l)) && (res.st.epochs || 0) >= 1, `a lying exchange is refused before anything is stored, and payouts go on: ${summary(res)}`);
}

// --- the third parties, failing in every way they can ---
{
  // Every exchange ChangeNOW takes, it fails: each is written off and alerted,
  // and nothing waits for a person -- the next RVN goes into the next one.
  const res = await run({ seed: 6, faults: { cnFail: 1 }, ticks: 3 * DAY });
  clean(res, "ChangeNOW failing every exchange");
  ok(!res.st.paused && res.st.orphans.length >= 2 && res.w.alerts.some((a) => /failed at ChangeNOW/.test(a)),
     `ChangeNOW failing every exchange: each written off and alerted, the treasury never stops trying (${res.st.orphans.length} written off)`);
}
{
  // Half of them fail: the other half still pay.
  const res = await run({ seed: 14, faults: { cnFail: 0.5 }, ticks: 6 * DAY });
  clean(res, "ChangeNOW failing half its exchanges");
  ok((res.st.epochs || 0) >= 1 && res.st.orphans.length >= 1, `ChangeNOW failing half: written off, and the rest pay: ${summary(res)}`);
}
{
  const res = await run({ seed: 7, faults: { cnRefund: 1 }, ticks: 3 * DAY, calm: 0 });
  clean(res, "an exchange that refunds");
  ok(res.w.xs.size >= 2 && res.logs.some((l) => /refunded/.test(l)), `a refunded exchange is let go and the RVN tried again: ${summary(res)}`);
}
{
  // Sitting on funds: written off at 48 h, watched, and when ChangeNOW finally
  // finishes, the late ETH is noticed and simply paid out.
  const res = await run({ seed: 8, faults: { cnStuck: 0.5 }, ticks: 6 * DAY });
  clean(res, "ChangeNOW sitting on funds");
  ok(res.st.orphans.length >= 1 && (res.st.epochs || 0) >= 1, `a funded exchange stuck two days is written off and payouts carry on: ${summary(res)}`);
}
{
  // ChangeNOW entirely gone for two days, then back.
  const res = await run({ seed: 15, faults: { outages: [{ dep: "changenow", from: DAY, to: 3 * DAY, how: "503" }] }, ticks: 6 * DAY });
  clean(res, "ChangeNOW down for two days");
  ok((await paidAfter(res, 3 * DAY)) >= 1, `ChangeNOW down two days: RVN waits, and payouts resume by themselves after: ${summary(res)}`);
}
{
  // The API key revoked for two days (401s), then restored.
  const res = await run({ seed: 16, faults: { outages: [{ dep: "changenow", from: DAY, to: 3 * DAY, how: "401" }] }, ticks: 6 * DAY });
  clean(res, "ChangeNOW refusing the key");
  ok((await paidAfter(res, 3 * DAY)) >= 1, `ChangeNOW refusing the key two days: nothing sent, and it resumes when the key works: ${summary(res)}`);
}
{
  // The Ravencoin explorer unreachable for a day, mid-exchange as likely as not.
  const res = await run({ seed: 17, faults: { outages: [{ dep: "blockbook", from: DAY, to: 2 * DAY, how: "unreachable" }] }, ticks: 5 * DAY });
  clean(res, "the Ravencoin explorer gone for a day");
  ok((await paidAfter(res, 2 * DAY)) >= 1, `blockbook unreachable a day: nothing guessed, payouts resume after: ${summary(res)}`);
}
{
  // Chain 4663's RPC down for a day, while transactions are in flight.
  const res = await run({ seed: 18, faults: { outages: [{ dep: "rpc", from: DAY, to: 2 * DAY, how: "503" }] }, ticks: 5 * DAY });
  clean(res, "the 4663 RPC down for a day");
  ok((await paidAfter(res, 2 * DAY)) >= 1, `the RPC down a day: in-flight transactions settle after, payouts resume: ${summary(res)}`);
}
{
  // Every dependency down at once for a day.
  const all = ["rpc", "blockbook", "changenow"].map((dep) => ({ dep, from: 2 * DAY, to: 3 * DAY, how: "unreachable" }));
  const res = await run({ seed: 19, faults: { outages: all }, ticks: 6 * DAY });
  clean(res, "every third party down at once");
  ok((await paidAfter(res, 3 * DAY)) >= 1, `every third party down a day: it all resumes by itself: ${summary(res)}`);
}
{
  // The RVN price crashes 99% for two days: pots too small to be worth the
  // gas wait, and pay once it recovers.
  const res = await run({ seed: 20, faults: { conditions: [{ what: "rate", from: DAY, to: 3 * DAY }] }, ticks: 6 * DAY });
  clean(res, "RVN crashing 99%");
  ok((await paidAfter(res, 3 * DAY)) >= 1, `a 99% price crash: small pots wait, nothing is wasted on gas, payouts resume: ${summary(res)}`);
}
{
  // Gas at 100x for a day.
  const res = await run({ seed: 21, faults: { conditions: [{ what: "gas", from: DAY, to: 2 * DAY }] }, ticks: 5 * DAY });
  clean(res, "gas at 100x");
  ok((await paidAfter(res, 2 * DAY)) >= 1, `gas at 100x for a day: stuck transactions are replaced, payouts resume: ${summary(res)}`);
}
{
  // The token's tax at 40% for a day, then back to 1%: payouts wait it out.
  const res = await run({ seed: 9, faults: { conditions: [{ what: "tax", bps: 4000n, from: DAY, to: 2 * DAY }] }, ticks: 5 * DAY });
  clean(res, "a 40% tax for a day");
  ok(res.w.alerts.some((a) => /buy tax reads 40%/.test(a)) && (await paidAfter(res, 2 * DAY)) >= 1, `a 40% buy tax waits, alerts, and pays when it drops: ${summary(res)}`);
}
{
  // A 20% tax that stays: under the 25% ceiling, so payouts go on, priced in.
  const res = await run({ seed: 22, faults: { conditions: [{ what: "tax", bps: 2000n, from: 0, to: 99 * DAY }] }, ticks: 5 * DAY });
  clean(res, "a 20% tax that stays");
  ok((res.st.epochs || 0) >= 1, `a lasting 20% tax is priced into the bound and paid through: ${summary(res)}`);
}
{
  const res = await run({ seed: 10, faults: { shove: 0.3 }, ticks: 6 * DAY });
  clean(res, "a pair shoved off its price");
  ok(res.logs.some((l) => /off its recent median/.test(l)) && (res.st.epochs || 0) >= 1, `a buy waits out a price pushed off its median: ${summary(res)}`);
}
{
  const res = await run({ seed: 11, faults: { stealNonce: 0.3 }, ticks: 6 * DAY });
  clean(res, "somebody else holding the key");
  ok(res.w.alerts.some((a) => /none of ours|none of the attempts/.test(a)), `a nonce taken by somebody else is let go and alerted, never guessed at: ${summary(res)}`);
}
{
  const res = await run({ seed: 12, faults: { ambiguous: 0.6, receiptLag: 2, evmStall: 0.5, gasSpike: 0.1 }, ticks: 8 * DAY });
  clean(res, "a chain that accepts and says it did not");
  ok((res.st.epochs || 0) >= 1, `ambiguous broadcasts never become two payouts: ${summary(res)}`);
}
{
  const res = await run({ seed: 13, faults: { evictAfterWrite: 0.4, evictBeforeWrite: 0.2, evict: 0.1 }, ticks: 6 * DAY });
  clean(res, "an object evicted at every turn");
  ok((res.st.epochs || 0) >= 1, `eviction mid-step never sends twice and still pays: ${summary(res)}`);
}
{
  // Many seeds, every fault at a random strength.
  const seeds = Number(process.argv[2] || 24);
  let bad = 0, paid = 0;
  const reasons = new Map();
  for (let s = 100; s < 100 + seeds; s++) {
    const faults = stormFaults(s);
    const res = await run({ seed: s, faults, ticks: 4 * DAY, calm: 400 });
    if (res.w.violations.length) {
      bad += 1;
      console.log(`  seed ${s}: ${[...new Set(res.w.violations)].slice(0, 3).join(" | ")}`);
    }
    if (res.st.epochs) paid += 1;
    for (const i of res.st.incidents || []) { const k = i.why.replace(/0x[0-9a-f]+|cn\d+|\d+/g, "#").slice(0, 70); reasons.set(k, (reasons.get(k) || 0) + 1); }
  }
  ok(bad === 0, `${seeds} random storms: ${bad} broke an invariant, ${paid} paid at least once, none left waiting for a person`);
  for (const [k, n] of [...reasons].sort((x, y) => y[1] - x[1])) console.log(`      incident ${n}x: ${k}`);
}

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
