// Pearl (PRL) proof-of-work, the plain (no-ZK) mining path, in JavaScript.
//
// Transliterated from pearl-research-labs/pearl:
//   zk-pow/src/ffi/mine.rs            (commitment, the mining loop)
//   zk-pow/src/circuit/pearl_noise.rs (the four noise matrices)
//   zk-pow/src/api/seed.rs            (v3 salted seed derivation)
//   zk-pow/src/api/proof_utils.rs     (job_key, jackpot hash, serialization)
//   zk-pow/src/circuit/chip/jackpot/helper.rs (compute_jackpot)
//   pearl-blake3/src/hasher.rs        (keyed BLAKE3; the Merkle root of a
//                                      matrix is standard BLAKE3 of its bytes)
//
// Checked against vectors that Pearl's own code produced and its own verifier
// accepted (pearl.test.mjs). Pure functions, so the same file runs under node
// for the reference and feeds the WGSL kernel its test inputs.

// ============================================================ BLAKE3 =========
const IV = Uint32Array.of(0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19);
const MSG_PERM = Uint8Array.of(2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8);
const CHUNK_START = 1, CHUNK_END = 2, PARENT = 4, ROOT = 8, KEYED = 16;
const CHUNK_LEN = 1024;
const rotr = (x, n) => (x >>> n) | (x << (32 - n));

// One compression. `cv` 8 words, `m` 16 words, returns the 8-word chaining
// value (which is also the first 32 bytes of the root output).
function compress(cv, m, counter, blockLen, flags) {
  const v = new Uint32Array(16);
  v.set(cv, 0); v[8] = IV[0]; v[9] = IV[1]; v[10] = IV[2]; v[11] = IV[3];
  v[12] = counter >>> 0; v[13] = Math.floor(counter / 4294967296) >>> 0; v[14] = blockLen; v[15] = flags;
  let w = m;
  const g = (a, b, c, d, x, y) => {
    v[a] = v[a] + v[b] + x; v[d] = rotr(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];     v[b] = rotr(v[b] ^ v[c], 12);
    v[a] = v[a] + v[b] + y; v[d] = rotr(v[d] ^ v[a], 8);
    v[c] = v[c] + v[d];     v[b] = rotr(v[b] ^ v[c], 7);
  };
  for (let r = 0; r < 7; r++) {
    g(0, 4, 8, 12, w[0], w[1]); g(1, 5, 9, 13, w[2], w[3]); g(2, 6, 10, 14, w[4], w[5]); g(3, 7, 11, 15, w[6], w[7]);
    g(0, 5, 10, 15, w[8], w[9]); g(1, 6, 11, 12, w[10], w[11]); g(2, 7, 8, 13, w[12], w[13]); g(3, 4, 9, 14, w[14], w[15]);
    if (r < 6) { const n = new Uint32Array(16); for (let i = 0; i < 16; i++) n[i] = w[MSG_PERM[i]]; w = n; }
  }
  const out = new Uint32Array(8);
  for (let i = 0; i < 8; i++) out[i] = v[i] ^ v[i + 8];
  return out;
}

const blockWords = (data, start, end) => {
  const m = new Uint32Array(16);
  for (let i = 0; i < end - start; i++) m[i >> 2] |= data[start + i] << ((i & 3) * 8);
  return m;
};

// One chunk (<= 1024 bytes), returning its 8-word CV; ROOT marks the whole
// input being a single chunk.
function chunkCV(data, start, end, chunkIdx, keyWords, baseFlags, isRoot) {
  const n = Math.max(1, Math.ceil((end - start) / 64));
  let cv = keyWords;
  for (let b = 0, pos = start; b < n; b++) {
    const bend = Math.min(pos + 64, end), blen = bend - pos;
    let flags = baseFlags;
    if (b === 0) flags |= CHUNK_START;
    if (b === n - 1) { flags |= CHUNK_END; if (isRoot) flags |= ROOT; }
    cv = compress(cv, blockWords(data, pos, bend), chunkIdx, blen, flags);
    pos = bend;
  }
  return cv;
}

function parentCV(left, right, keyWords, baseFlags, isRoot) {
  const m = new Uint32Array(16);
  m.set(left, 0); m.set(right, 8);
  return compress(keyWords, m, 0, 64, baseFlags | PARENT | (isRoot ? ROOT : 0));
}

const lpotLeq = (n) => { let p = 1; while (p * 2 <= n) p *= 2; return p; };

// BLAKE3's own chunk tree, so the root equals standard BLAKE3 of the bytes.
function subtree(data, start, end, chunkStart, keyWords, baseFlags, isRoot) {
  const len = end - start;
  if (len <= CHUNK_LEN) return chunkCV(data, start, end, chunkStart, keyWords, baseFlags, isRoot);
  const leftChunks = lpotLeq(Math.floor((len - 1) / CHUNK_LEN));
  const ll = leftChunks * CHUNK_LEN;
  const left = subtree(data, start, start + ll, chunkStart, keyWords, baseFlags, false);
  const right = subtree(data, start + ll, end, chunkStart + leftChunks, keyWords, baseFlags, false);
  return parentCV(left, right, keyWords, baseFlags, isRoot);
}

const u32le = (bytes) => { const w = new Uint32Array(8); for (let i = 0; i < 8; i++) w[i] = (bytes[i * 4] | (bytes[i * 4 + 1] << 8) | (bytes[i * 4 + 2] << 16) | (bytes[i * 4 + 3] << 24)) >>> 0; return w; };
const wToBytes = (w) => { const b = new Uint8Array(32); for (let i = 0; i < 8; i++) { b[i * 4] = w[i]; b[i * 4 + 1] = w[i] >>> 8; b[i * 4 + 2] = w[i] >>> 16; b[i * 4 + 3] = w[i] >>> 24; } return b; };

// blake3_digest: keyed when `key` (32 bytes) is given, else plain.
export function blake3(data, key) {
  const keyWords = key ? u32le(key) : IV;
  const flags = key ? KEYED : 0;
  return wToBytes(subtree(data, 0, data.length, 0, keyWords, flags, true));
}

// ============================================================ Pearl ==========
const cat = (...arrs) => { let n = 0; for (const a of arrs) n += a.length; const o = new Uint8Array(n); let p = 0; for (const a of arrs) { o.set(a, p); p += a.length; } return o; };
const u32bytes = (v) => Uint8Array.of(v & 255, (v >>> 8) & 255, (v >>> 16) & 255, (v >>> 24) & 255);
const label = (s) => { const b = new Uint8Array(32); for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i); return b; };

const SEED_LABEL_A = label("A_tensor");
const SEED_LABEL_B = label("B_tensor");
// blake3("pearl/cert-v3/noise-seed/A" | "/B"); the hardcoded consensus salts.
const SEED_SALT_A = Uint8Array.of(0x82, 0x49, 0x40, 0x6c, 0xa0, 0xed, 0x15, 0x16, 0x96, 0x16, 0xf6, 0x92, 0xfc, 0xf0, 0x76, 0xf8, 0x92, 0xdb, 0xdb, 0x2a, 0x70, 0x23, 0xb8, 0x52, 0xf0, 0xd4, 0x77, 0x19, 0xc3, 0x90, 0x01, 0x7b);
const SEED_SALT_B = Uint8Array.of(0x11, 0x30, 0x06, 0x32, 0xec, 0x63, 0x01, 0xca, 0x2b, 0xe2, 0xaf, 0x71, 0x8b, 0x3f, 0x4d, 0x4f, 0x1a, 0xe9, 0xc6, 0x39, 0x88, 0xe8, 0xcc, 0x04, 0x48, 0x44, 0x30, 0x1d, 0x71, 0xb8, 0x9a, 0xa9);

const bindRoot = (root, dim, salt) => blake3(cat(root, u32bytes(dim), new Uint8Array(28)), salt);

// (b_noise_seed, a_noise_seed) from the committed matrix hashes. Salted = the
// v3 chain, which binds the matrix dimensions; Legacy = pre-v3.
export function commitmentSeeds(jobKey, aRowMajor, bColMajor, m, n, salted = true) {
  return seedsFromRoots(jobKey, blake3(aRowMajor, jobKey), blake3(bColMajor, jobKey), m, n, salted);
}

// The same from the two Merkle roots alone, which is all a pool has: a share
// carries the roots and the tile, not the matrices.
export function seedsFromRoots(jobKey, rootA, rootB, m, n, salted = true) {
  let hashA = rootA, hashB = rootB;
  if (salted) { hashA = bindRoot(hashA, m, SEED_SALT_A); hashB = bindRoot(hashB, n, SEED_SALT_B); }
  const bNoiseSeed = blake3(cat(jobKey, hashB), null);
  const aNoiseSeed = blake3(cat(bNoiseSeed, hashA), null); // hash_activations = hash_a (no MoE)
  return { bNoiseSeed, aNoiseSeed };
}

// Pieces of the keyed tree, for checking a Merkle proof: a leaf's chaining
// value (never the root), a parent's, and the root's. All take and give 32
// bytes; the key is the job key.
export const leafCV = (chunk1024, index, key) => wToBytes(chunkCV(chunk1024, 0, chunk1024.length, index, u32le(key), KEYED, false));
export const parentOf = (left, right, key) => wToBytes(parentCV(u32le(left), u32le(right), u32le(key), KEYED, false));
export const rootOf = (left, right, key) => wToBytes(parentCV(u32le(left), u32le(right), u32le(key), KEYED, true));

const RANGE_MASK = 63, ZERO_POINT = 32;
const randomHash = (index, seed, key, prependIdx) => { const msg = new Uint8Array(64); const pv = (1 + index) | 0; msg.set(u32bytes(pv), prependIdx * 4); msg.set(seed, 32); return blake3(msg, key); };
const mulHi = (a, b) => Number((BigInt(a >>> 0) * BigInt(b >>> 0)) >> 32n) >>> 0;

// A uniform matrix row by row, each value a byte of a BLAKE3 stream folded to
// [-32, 31]. `rowIndices` selects which rows to compute; `numCols` = noise_rank.
function uniformMatrix(seed, key, rowIndices, numCols) {
  return rowIndices.map((rowIdx) => {
    const start = rowIdx * numCols, out = new Int8Array(numCols);
    for (let block = Math.floor(start / 32); block < Math.ceil((start + numCols) / 32); block++) {
      const h = randomHash(block, seed, key, 0);
      for (let k = 0; k < 32; k++) { const idx = block * 32 + k; if (idx >= start && idx < start + numCols) out[idx - start] = (h[k] & RANGE_MASK) - ZERO_POINT; }
    }
    return out;
  });
}

// A sparse permutation: k rows, each a (+1 at first, -1 at second) index pair.
function permMatrix(seed, key, k, noiseRank) {
  const mask = (noiseRank - 1) >>> 0, res = new Array(k);
  for (let i = 0; i < Math.ceil(k / 8); i++) {
    const h = randomHash(i, seed, key, 1);
    for (let j = 0; j < 8 && i * 8 + j < k; j++) {
      const r = (h[j * 4] | (h[j * 4 + 1] << 8) | (h[j * 4 + 2] << 16) | (h[j * 4 + 3] << 24)) >>> 0;
      const first = r & mask, second = (first ^ (1 + mulHi(noiseRank - 1, r))) >>> 0;
      res[i * 8 + j] = [first, second];
    }
  }
  return res;
}

const toI8 = (x) => (x << 24) >> 24;
const matvecSparse = (perm, vec) => { const out = new Int8Array(perm.length); for (let i = 0; i < perm.length; i++) out[i] = toI8(vec[perm[i][0]] - vec[perm[i][1]]); return out; };

// noise.a: the `aRows` rows of NOISE_A = E_AL * E_AR (each length k).
// noise.b: the `bCols` cols of NOISE_B = E_BL * E_BR (each length k).
export function computeNoise(k, noiseRank, bNoiseSeed, aNoiseSeed, aRows, bCols) {
  const eAL = uniformMatrix(SEED_LABEL_A, aNoiseSeed, aRows, noiseRank);
  const eARt = permMatrix(SEED_LABEL_A, aNoiseSeed, k, noiseRank);
  const eBL = permMatrix(SEED_LABEL_B, bNoiseSeed, k, noiseRank);
  const eBRt = uniformMatrix(SEED_LABEL_B, bNoiseSeed, bCols, noiseRank);
  return { a: eAL.map((row) => matvecSparse(eARt, row)), b: eBRt.map((col) => matvecSparse(eBL, col)) };
}

const rotl = (x, n) => ((x << n) | (x >>> (32 - n))) >>> 0;

// The jackpot: noised tile GEMM, XOR-folded into 16 words. `secretA` h rows of
// A, `secretB` w rows of B^t, each length k; noise as computeNoise returns.
export function computeJackpot(secretA, secretB, noise, k, r) {
  const h = secretA.length, w = secretB.length, jackpot = Array.from({ length: h }, () => new Int32Array(w)), msg = new Uint32Array(16);
  for (let ll = r; ll <= k; ll += r) {
    for (let u = 0; u < h; u++) for (let v = 0; v < w; v++) { let acc = jackpot[u][v]; for (let l = ll - r; l < ll; l++) acc += (secretA[u][l] + noise.a[u][l]) * (secretB[v][l] + noise.b[v][l]); jackpot[u][v] = acc; }
    let xored = 0; for (let u = 0; u < h; u++) for (let v = 0; v < w; v++) xored = (xored ^ jackpot[u][v]) >>> 0;
    const tid = (ll / r - 1) % 16; msg[tid] = (rotl(msg[tid], 13) ^ xored) >>> 0;
  }
  return msg;
}

export const jackpotHash = (jackpot, aNoiseSeed) => { const msg = new Uint8Array(64); for (let i = 0; i < 16; i++) msg.set(u32bytes(jackpot[i]), i * 4); return blake3(msg, aNoiseSeed); };

// header.to_bytes(): version | prev_block reversed | merkle_root reversed | timestamp | nbits.
export const headerBytes = (h) => cat(u32bytes(h.version), Uint8Array.from(h.prevBlock).reverse(), Uint8Array.from(h.merkleRoot).reverse(), u32bytes(h.timestamp), u32bytes(h.nbits));
export const jobKey = (headerB, configB) => blake3(cat(headerB, configB), null);

// pad a byte array up to a multiple of the BLAKE3 chunk length.
export const padChunk = (data) => { const n = Math.ceil(data.length / CHUNK_LEN) * CHUNK_LEN; if (n === data.length) return data; const o = new Uint8Array(n); o.set(data); return o; };
