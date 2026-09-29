// The pool's hot wallets, and the transactions that move proceeds out of them.
//
// **The keys are generated here and never leave.** The operator agreed to the
// automation holding keys on the condition that they are never the operator's,
// never pass through a conversation, and are never shown to anybody. So the
// Durable Object generates both on first use, keeps them in its own storage,
// and only ever exposes the addresses. There is no endpoint that returns a
// private key, and nothing in this file logs one.
//
// **Only proceeds sit here.** zpool pays the RVN wallet; the swap pays the
// 4663 wallet; the distributor is funded from that. Nobody deposits savings
// into either, and a balance that has not moved on to an epoch within a day is
// a fault somebody should look at, not a float.
//
// This module is pure -- keys in, bytes out -- so `treasury.test.mjs` checks it
// against published vectors with no network, no storage and no money.

import * as secp from "@noble/secp256k1";
import { sha256 } from "@noble/hashes/sha256";
import { ripemd160 } from "@noble/hashes/ripemd160";
import { keccak_256 } from "@noble/hashes/sha3";
import { hmac } from "@noble/hashes/hmac";

// noble signs synchronously once it has an HMAC; RFC 6979 nonces, so a
// signature is a function of key and message and never of a random source.
secp.etc.hmacSha256Sync = (k, ...m) => hmac(sha256, k, secp.etc.concatBytes(...m));

const hex = (b) => Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
const unhex = (s) => Uint8Array.from(s.match(/../g) || [], (h) => parseInt(h, 16));
const cat = (...a) => secp.etc.concatBytes(...a);
const dsha = (b) => sha256(sha256(b));

// --- keys -----------------------------------------------------------------------

export function newKey() {
  return hex(secp.utils.randomPrivateKey());
}

// --- Ravencoin --------------------------------------------------------------------

// Ravencoin mainnet P2PKH version byte: addresses begin with `R`.
export const RVN_P2PKH = 60;
const B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

function base58check(payload) {
  const b = cat(payload, dsha(payload).slice(0, 4));
  let n = 0n;
  for (const x of b) n = n * 256n + BigInt(x);
  let s = "";
  while (n > 0n) {
    s = B58[Number(n % 58n)] + s;
    n /= 58n;
  }
  for (const x of b) {
    if (x !== 0) break;
    s = "1" + s;
  }
  return s;
}

export function decodeBase58check(s) {
  let n = 0n;
  for (const c of s) {
    const d = B58.indexOf(c);
    if (d < 0) throw new Error("not base58");
    n = n * 58n + BigInt(d);
  }
  const bytes = [];
  while (n > 0n) {
    bytes.unshift(Number(n & 255n));
    n >>= 8n;
  }
  for (const c of s) {
    if (c !== "1") break;
    bytes.unshift(0);
  }
  const b = Uint8Array.from(bytes);
  const body = b.slice(0, -4);
  if (hex(dsha(body).slice(0, 4)) !== hex(b.slice(-4))) throw new Error("checksum fails");
  return body;
}

export function rvnAddress(privHex) {
  const pub = secp.getPublicKey(privHex, true);
  return base58check(cat(Uint8Array.of(RVN_P2PKH), ripemd160(sha256(pub))));
}

// The 20-byte hash inside an RVN P2PKH address, refusing anything else.
function rvnHash160(addr) {
  const b = decodeBase58check(addr);
  if (b.length !== 21 || b[0] !== RVN_P2PKH) throw new Error(`${addr} is not a Ravencoin P2PKH address`);
  return b.slice(1);
}

// DER, which noble-secp256k1 v2 no longer produces: SEQUENCE { INTEGER r,
// INTEGER s }, each integer minimal and with a zero byte prefixed when its top
// bit is set (else it reads as negative). Bitcoin-family nodes reject any other
// encoding (BIP 66), so this is written out rather than left to chance.
export function derSig(r, s) {
  const int = (x) => {
    let h = x.toString(16);
    if (h.length % 2) h = "0" + h;
    let b = unhex(h);
    if (b[0] & 0x80) b = cat(Uint8Array.of(0), b);
    return cat(Uint8Array.of(0x02, b.length), b);
  };
  const body = cat(int(r), int(s));
  return cat(Uint8Array.of(0x30, body.length), body);
}

const p2pkh = (h160) => cat(Uint8Array.of(0x76, 0xa9, 0x14), h160, Uint8Array.of(0x88, 0xac));

function varint(n) {
  if (n < 0xfd) return Uint8Array.of(n);
  if (n <= 0xffff) return Uint8Array.of(0xfd, n & 255, n >> 8);
  return Uint8Array.of(0xfe, n & 255, (n >> 8) & 255, (n >> 16) & 255, (n >>> 24) & 255);
}
const u32 = (n) => Uint8Array.of(n & 255, (n >> 8) & 255, (n >> 16) & 255, (n >>> 24) & 255);
function u64(sats) {
  const b = new Uint8Array(8);
  let v = BigInt(sats);
  for (let i = 0; i < 8; i++) {
    b[i] = Number(v & 255n);
    v >>= 8n;
  }
  return b;
}

// Serialise a legacy transaction. `scripts[i]` is input i's scriptSig.
function serialise(inputs, outputs, scripts) {
  const parts = [u32(1), varint(inputs.length)];
  inputs.forEach((inp, i) => {
    parts.push(unhex(inp.txid).reverse(), u32(inp.vout), varint(scripts[i].length), scripts[i], u32(0xffffffff));
  });
  parts.push(varint(outputs.length));
  for (const o of outputs) {
    const s = p2pkh(rvnHash160(o.address));
    parts.push(u64(o.sats), varint(s.length), s);
  }
  parts.push(u32(0));
  return cat(...parts);
}

// Sign a P2PKH-only transaction with one key, SIGHASH_ALL.
//
// **Legacy sighash, and that is Ravencoin's rule rather than a shortcut.** It
// is a Bitcoin fork from before SegWit and never adopted BIP 143 or a fork id,
// so the digest for input i is double-SHA256 of the transaction with input i's
// script replaced by the spent output's script, every other input's emptied,
// and the hash type appended as four bytes.
//
// `inputs`: [{txid (display hex), vout, sats}], all paying this key.
// `outputs`: [{address, sats}]. Returns the raw transaction as hex.
export function signRvnTx(privHex, inputs, outputs) {
  const pub = secp.getPublicKey(privHex, true);
  const mine = p2pkh(ripemd160(sha256(pub)));
  const empty = new Uint8Array(0);
  const scripts = inputs.map((_, i) => {
    const pre = serialise(inputs, outputs, inputs.map((_, j) => (j === i ? mine : empty)));
    const digest = dsha(cat(pre, u32(1)));
    const sg = secp.sign(digest, privHex, { lowS: true });
    const sig = derSig(sg.r, sg.s);
    const s = cat(sig, Uint8Array.of(0x01));
    return cat(Uint8Array.of(s.length), s, Uint8Array.of(pub.length), pub);
  });
  return hex(serialise(inputs, outputs, scripts));
}

// The digest input i is signed over, exposed so the test can check it against
// an implementation that is not this one.
export function rvnSighash(privHex, inputs, outputs, i) {
  const pub = secp.getPublicKey(privHex, true);
  const mine = p2pkh(ripemd160(sha256(pub)));
  const pre = serialise(inputs, outputs, inputs.map((_, j) => (j === i ? mine : new Uint8Array(0))));
  return hex(dsha(cat(pre, u32(1))));
}

// --- EVM (chain 4663) ---------------------------------------------------------------

export function evmAddress(privHex) {
  const pub = secp.getPublicKey(privHex, false).slice(1);
  const a = hex(keccak_256(pub).slice(12));
  const h = hex(keccak_256(new TextEncoder().encode(a)));
  let out = "0x";
  for (let i = 0; i < 40; i++) out += parseInt(h[i], 16) >= 8 ? a[i].toUpperCase() : a[i];
  return out;
}

// --- EVM transactions --------------------------------------------------------------

// RLP, the encoding every Ethereum transaction is signed and sent in. Items are
// byte strings or lists; integers are big-endian with no leading zeros, and
// zero is the empty string -- the rule a hand-rolled encoder most often gets
// wrong, which is why the test vector below includes a zero.
function rlpBytes(b) {
  if (b.length === 1 && b[0] < 0x80) return b;
  return cat(rlpLen(b.length, 0x80), b);
}
function rlpLen(n, offset) {
  if (n < 56) return Uint8Array.of(offset + n);
  const h = unhex(n.toString(16).padStart(Math.ceil(n.toString(16).length / 2) * 2, "0"));
  return cat(Uint8Array.of(offset + 55 + h.length), h);
}
export function rlp(item) {
  if (Array.isArray(item)) {
    const body = cat(...item.map(rlp));
    return cat(rlpLen(body.length, 0xc0), body);
  }
  return rlpBytes(item);
}
export function int(v) {
  let n = BigInt(v);
  if (n === 0n) return new Uint8Array(0);
  let h = n.toString(16);
  if (h.length % 2) h = "0" + h;
  return unhex(h);
}
const addrBytes = (a) => (a ? unhex(a.slice(2).toLowerCase()) : new Uint8Array(0));

// A legacy transaction with EIP-155 replay protection: the chain id is inside
// what is signed, so a signature for chain 4663 means nothing on any other.
// Legacy rather than EIP-1559 because every EVM chain accepts it and it has one
// fee field to get wrong instead of three. `to` null is a contract creation.
export function signLegacyTx(privHex, { nonce, gasPrice, gas, to, value, data, chainId }) {
  const fields = [int(nonce), int(gasPrice), int(gas), addrBytes(to), int(value), data ? unhex(data.replace(/^0x/, "")) : new Uint8Array(0)];
  const digest = keccak_256(rlp([...fields, int(chainId), int(0), int(0)]));
  const sg = secp.sign(digest, privHex, { lowS: true });
  const v = BigInt(chainId) * 2n + 35n + BigInt(sg.recovery);
  const raw = rlp([...fields, int(v), int(sg.r), int(sg.s)]);
  return { raw: "0x" + hex(raw), hash: "0x" + hex(keccak_256(raw)) };
}

// The address a contract created by `sender` at `nonce` will have.
export function createdAddress(sender, nonce) {
  const h = keccak_256(rlp([addrBytes(sender), int(nonce)]));
  return "0x" + hex(h.slice(12));
}

export { hex, unhex };
