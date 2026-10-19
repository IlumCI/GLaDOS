// A Pearl share as it travels, and the pool's check of one.
//
// On the wire a share is `plain_proof`: base64 of Pearl's `PlainProof` encoded
// with bincode 1.x (fixint, little-endian -- what `PlainProof::to_base64` does in
// zk-pow/src/ffi/plain_proof.rs). It carries the matrix sizes, the noise rank,
// and for each of A and B^t the tile's rows as BLAKE3 chunks plus a Merkle path
// from them to the committed root.
//
// `verifyShare` is Pearl's `verify_plain_proof` (zk-pow/src/api/verify.rs) for
// the shares this pool's own miners produce, and stricter than it on purpose:
//
//   * Tiles are 16 x 16, contiguous and aligned -- the one shape our kernels
//     mine. Pearl allows any periodic pattern; accepting only ours keeps the
//     part of Pearl reimplemented here small, and a smaller verifier is the one
//     with fewer ways to be wrong. Everything else is refused by name.
//   * No MoE proofs, and the rank must divide k.
//
// Within that it checks what Pearl checks, in Pearl's order: the declared
// dimensions against both trees' leaf counts (so m and n, which salt the noise
// seeds, are bound to what was committed), the consensus sanity bounds, both
// Merkle paths against their roots under the job key, the [-64, 64] signal
// range, then the noise from the roots, the jackpot, its hash, and the hash
// against the bound. It never trusts a hash the miner claims.
//
// Checked against proofs Pearl's own miner produced and Pearl's own verifier
// accepted, and against broken copies Pearl rejected (pearlproof.test.mjs).

import { blake3, seedsFromRoots, computeNoise, computeJackpot, jackpotHash, leafCV, parentOf, rootOf } from "./pearl.js";

const CHUNK = 1024;
const TILE = 16;
// Bounds on what one share may declare, well past anything honest, so a
// malformed share costs a parse and not a gigabyte.
const MAX_LEAVES = 4096, MAX_SIBLINGS = 1024, MAX_ROWS = 64;

// ------------------------------------------------------------ bincode -------
class Reader {
  constructor(u8) { this.b = u8; this.p = 0; this.v = new DataView(u8.buffer, u8.byteOffset, u8.byteLength); }
  need(n) { if (this.p + n > this.b.length) throw new Error("share ends early"); }
  u8() { this.need(1); return this.b[this.p++]; }
  u64() {
    this.need(8);
    const lo = this.v.getUint32(this.p, true), hi = this.v.getUint32(this.p + 4, true);
    this.p += 8;
    if (hi > 0x1fffff) throw new Error("a number in the share is too large");
    return hi * 2 ** 32 + lo;
  }
  bytes(n) { this.need(n); const s = this.b.subarray(this.p, this.p + n); this.p += n; return s; }
  count(max, what) { const n = this.u64(); if (n > max) throw new Error(`too many ${what} (${n})`); return n; }
}

class Writer {
  constructor() { this.parts = []; this.n = 0; }
  push(u8) { this.parts.push(u8); this.n += u8.length; }
  u8(x) { this.push(Uint8Array.of(x)); }
  u64(x) { const b = new Uint8Array(8); const v = new DataView(b.buffer); v.setUint32(0, x >>> 0, true); v.setUint32(4, Math.floor(x / 2 ** 32), true); this.push(b); }
  bytes(u8) { this.push(u8); }
  done() { const o = new Uint8Array(this.n); let p = 0; for (const x of this.parts) { o.set(x, p); p += x.length; } return o; }
}

function readMatrixProof(r) {
  const nLeaves = r.count(MAX_LEAVES, "leaves");
  const leafData = [];
  for (let i = 0; i < nLeaves; i++) {
    if (r.u64() !== CHUNK) throw new Error("a leaf is not 1024 bytes");
    leafData.push(r.bytes(CHUNK));
  }
  const nIdx = r.count(MAX_LEAVES, "leaf indices");
  const leafIndices = Array.from({ length: nIdx }, () => r.u64());
  const totalLeaves = r.u64();
  const root = r.bytes(32);
  const nSib = r.count(MAX_SIBLINGS, "siblings");
  const siblings = Array.from({ length: nSib }, () => r.bytes(32));
  const nRows = r.count(MAX_ROWS, "rows");
  const rowIndices = Array.from({ length: nRows }, () => r.u64());
  return { leafData, leafIndices, totalLeaves, root, siblings, rowIndices };
}

function writeMatrixProof(w, p) {
  w.u64(p.leafData.length);
  for (const d of p.leafData) { w.u64(d.length); w.bytes(d); }
  w.u64(p.leafIndices.length); for (const i of p.leafIndices) w.u64(i);
  w.u64(p.totalLeaves);
  w.bytes(p.root);
  w.u64(p.siblings.length); for (const s of p.siblings) w.bytes(s);
  w.u64(p.rowIndices.length); for (const i of p.rowIndices) w.u64(i);
}

/** bincode bytes of a PlainProof -> { m, n, k, rank, a, bt, moe } */
export function decodeProof(bytes) {
  const r = new Reader(bytes);
  const m = r.u64(), n = r.u64(), k = r.u64(), rank = r.u64();
  const a = readMatrixProof(r), bt = readMatrixProof(r);
  // The trailing Option<MoE>: 0 is None. A legacy (pre-MoE) proof stops here.
  let moe = false;
  if (r.p < bytes.length) moe = r.u8() !== 0;
  if (!moe && r.p !== bytes.length) throw new Error("bytes left over after the share");
  return { m, n, k, rank, a, bt, moe };
}

/** The inverse, for a miner building a share. */
export function encodeProof(p) {
  const w = new Writer();
  w.u64(p.m); w.u64(p.n); w.u64(p.k); w.u64(p.rank);
  writeMatrixProof(w, p.a); writeMatrixProof(w, p.bt);
  w.u8(0); // moe: None
  return w.done();
}

export const fromBase64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
export const toBase64 = (u8) => { let s = ""; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode(...u8.subarray(i, i + 0x8000)); return btoa(s); };

// ------------------------------------------------------------ Merkle --------
// Pearl's MerkleProof::compute_root (pearl-blake3/src/merkle.rs): pair
// neighbours level by level, promote a lone last node, take missing nodes from
// the siblings in order, and hash the final pair with the ROOT flag. Not BLAKE3's
// own split-at-the-largest-power-of-two tree; the two agree only on
// power-of-two leaf counts.
export function computeRoot(p, key) {
  if (!p.leafIndices.length) return null;
  let cur = new Map(p.leafIndices.map((idx, j) => [idx, leafCV(p.leafData[j], idx, key)]));
  let level = p.totalLeaves, s = 0;
  const sib = () => (s < p.siblings.length ? p.siblings[s++] : null);
  if (level === 1) return p.siblings.length === 0 ? cur.get(0) ?? null : null;
  while (level > 2) {
    const next = new Map();
    for (const i of [...cur.keys()].sort((x, y) => x - y)) {
      const cv = cur.get(i);
      if (i % 2 === 0) {
        let right = cur.get(i + 1) ?? null;
        if (!right && i + 1 < level) { right = sib(); if (!right) return null; }
        next.set(i / 2, right ? parentOf(cv, right, key) : cv);
      } else if (!cur.has(i - 1)) {
        const left = sib(); if (!left) return null;
        next.set((i - 1) / 2, parentOf(left, cv, key));
      }
    }
    cur = next;
    level = Math.ceil(level / 2);
  }
  const left = cur.get(0) ?? sib(), right = cur.get(1) ?? sib();
  if (!left || !right || s !== p.siblings.length) return null;
  return rootOf(left, right, key);
}

// MerkleProof::extract_bytes: `length` bytes from `start` of the committed
// matrix, out of the leaves the proof carries, or null if they do not cover it.
function extract(p, start, length) {
  const out = new Int8Array(length), end = start + length;
  let copied = 0;
  p.leafIndices.forEach((idx, j) => {
    const ls = idx * CHUNK, le = ls + CHUNK;
    if (ls < end && start < le) {
      const cs = Math.max(start, ls), ce = Math.min(end, le);
      out.set(new Int8Array(p.leafData[j].buffer, p.leafData[j].byteOffset + (cs - ls), ce - cs), cs - start);
      copied += ce - cs;
    }
  });
  return copied === length ? out : null;
}

// ------------------------------------------------------------ targets -------
const MAX256 = (1n << 256n) - 1n;
/** Bitcoin compact bits to a 256-bit target. */
export function nbitsToTarget(nbits) {
  const exp = nbits >>> 24, mant = BigInt(nbits & 0x007fffff);
  return exp <= 3 ? mant >> BigInt(8 * (3 - exp)) : mant << BigInt(8 * (exp - 3));
}
/** Pearl's scale(): target x factor, saturating at 2^256 - 1. */
export const scaleTarget = (target, factor) => { const t = target * BigInt(factor); return t > MAX256 ? MAX256 : t; };
/** The rank-penalised adjustment factor: tile x (k / rank) x 128. */
export const adjustmentFactor = (k, rank) => TILE * TILE * (k / rank) * 128;
const leU256 = (u8) => { let x = 0n; for (let i = 31; i >= 0; i--) x = (x << 8n) | BigInt(u8[i]); return x; };
const hex = (u8) => Array.from(u8, (b) => b.toString(16).padStart(2, "0")).join("");

// MiningConfiguration::to_bytes for a 16 x 16 contiguous tile: common_dim,
// rank, mma type 0, then each pattern as (factor-1, length-1) per dimension --
// one dimension of length 16 is 00 0f, the rest 00 00 -- then 32 zero bytes.
export function configBytes(k, rank) {
  const b = new Uint8Array(52), v = new DataView(b.buffer);
  v.setUint32(0, k, true); v.setUint16(4, rank, true); v.setUint16(6, 0, true);
  b[9] = TILE - 1; b[15] = TILE - 1; // rows pattern at 8..13, cols at 14..19
  return b;
}

const isPow2 = (x) => x > 0 && (x & (x - 1)) === 0;

/**
 * Check one share. `header` is the job's 76-byte incomplete header; `bound` a
 * BigInt the jackpot hash must not exceed (already scaled), or `nbits` to have
 * it computed the way Pearl's consensus does. Answers
 * { ok: true, jackpotHash, adjustmentFactor, tile: [row, col], m, n, k, rank }
 * or { ok: false, why }.
 */
export function verifyShare(proofBytes, header, { bound, nbits, salted = true } = {}) {
  let p;
  try { p = decodeProof(proofBytes); } catch (e) { return { ok: false, why: `unreadable share: ${e.message}` }; }
  const { m, n, k, rank, a, bt } = p;
  if (p.moe) return { ok: false, why: "MoE proofs are not accepted" };
  if (!(header instanceof Uint8Array) || header.length !== 76) return { ok: false, why: "the job header is not 76 bytes" };

  // This pool's tile: 16 contiguous rows and columns, at a multiple of 16.
  const tileOf = (idx, lim) => idx.length === TILE && idx[0] % TILE === 0 && idx.every((x, i) => x === idx[0] + i) && idx[0] + TILE - 1 < lim;
  if (!tileOf(a.rowIndices, m) || !tileOf(bt.rowIndices, n)) return { ok: false, why: "the tile is not a 16 x 16 aligned block of the matrix" };
  const tr = a.rowIndices[0], tc = bt.rowIndices[0];

  // public_params_sanity_check (zk-pow/src/api/sanity_checks.rs), for this shape.
  if (!(isPow2(rank) && rank >= 128 && rank <= 1024)) return { ok: false, why: `rank ${rank} is not a power of two from 128 to 1024` };
  if (k % rank) return { ok: false, why: "the rank does not divide k" };
  if (!(k >= 1024 && k <= 65536 && k % 64 === 0 && k <= 4 * rank * rank && k >= 16 * rank)) return { ok: false, why: `k ${k} is outside Pearl's bounds for rank ${rank}` };
  if (!(m <= 2 ** 24 && n <= 2 ** 24)) return { ok: false, why: "a dimension past 2^24" };
  if (2 * TILE * k > 2 ** 22) return { ok: false, why: "the tile is larger than Pearl allows" };

  // check_declared_tree_sizes: m and n salt the noise seeds, so they must be
  // the dimensions that were actually committed.
  if (a.totalLeaves !== Math.ceil((m * k) / CHUNK) || bt.totalLeaves !== Math.ceil((n * k) / CHUNK)) return { ok: false, why: "a tree's size does not match the declared dimensions" };

  for (const t of [a, bt]) {
    if (t.leafIndices.length !== t.leafData.length || !t.leafIndices.length) return { ok: false, why: "a Merkle proof is malformed" };
    for (let i = 1; i < t.leafIndices.length; i++) if (t.leafIndices[i] <= t.leafIndices[i - 1]) return { ok: false, why: "Merkle leaves out of order" };
    if (t.leafIndices[t.leafIndices.length - 1] >= t.totalLeaves) return { ok: false, why: "a Merkle leaf past the end of its tree" };
  }

  const jobKey = blake3(new Uint8Array([...header, ...configBytes(k, rank)]), null);
  for (const [t, name] of [[a, "A"], [bt, "B"]]) {
    const root = computeRoot(t, jobKey);
    if (!root || hex(root) !== hex(t.root)) return { ok: false, why: `${name}'s Merkle proof does not reach its root` };
  }

  const stripsA = [], stripsB = [];
  for (const [t, out] of [[a, stripsA], [bt, stripsB]]) {
    for (const row of t.rowIndices) {
      const s = extract(t, row * k, k);
      if (!s) return { ok: false, why: "the proof does not carry the whole tile" };
      for (const v of s) if (v < -64 || v > 64) return { ok: false, why: `a matrix value ${v} is outside [-64, 64]` };
      out.push(s);
    }
  }

  const { bNoiseSeed, aNoiseSeed } = seedsFromRoots(jobKey, a.root, bt.root, m, n, salted);
  const noise = computeNoise(k, rank, bNoiseSeed, aNoiseSeed, a.rowIndices, bt.rowIndices);
  const jh = jackpotHash(computeJackpot(stripsA, stripsB, noise, k, rank), aNoiseSeed);

  const factor = adjustmentFactor(k, rank);
  const limit = bound !== undefined ? bound : nbits !== undefined ? scaleTarget(nbitsToTarget(nbits), factor) : null;
  if (limit === null) return { ok: false, why: "no bound to check against" };
  if (leU256(jh) > limit) return { ok: false, why: "the jackpot hash is above the target" };
  return { ok: true, jackpotHash: hex(jh), adjustmentFactor: factor, tile: [tr / TILE, tc / TILE], m, n, k, rank };
}
