// The verdict anchor, and the arithmetic that checks a signature against it.
//
// Plain JS with no imports beyond gladosig.js, so `node` can run it -- the
// bargain envelope.js documents. The point pinned here must equal
// `VERDICT_KEY` in `src/update/mod.rs`; ci.yml asserts that with
// `sign.anchor` on every push, because a pinned value copied by hand is a
// pinned value that drifts, and this one decides which verdicts get stored.
//
// **Why the ingest function verifies before storing at all**: the POST side
// authenticates with a shared token, and a leaked token that could insert
// arbitrary rows would let somebody feed a device a "verdict" the kernel
// would then also refuse -- no harm done, but a poisoned queue. Verifying
// here means a leaked ingest token can insert only verdicts the verdict key
// actually signed, which is the same class of statement `propose.yml` makes
// about its own signing step.

import { G, N, add, mul, sha256, toBigInt } from "./gladosig.js";

// src/update/mod.rs VERDICT_KEY, uncompressed 0x04 || X || Y.
export const VERDICT_KEY_HEX =
  "0413cb9d149f65da55c411561a13ad081e2357e01d37ab0593d3bec8314ecb6e" +
  "e7ae8d9b772a2a2b7e8ef35da0225b9b27ff8366ce6c6634ab3fc68f86c1f37e" +
  "d0";

export const SIG_LEN = 80;

/** Modular inverse by extended Euclid; BigInt has no modpow to lean on. */
function invMod(a, m) {
  let [old_r, r] = [((a % m) + m) % m, m];
  let [old_s, s] = [1n, 0n];
  while (r !== 0n) {
    const q = old_r / r;
    [old_r, r] = [r, old_r - q * r];
    [old_s, s] = [s, old_s - q * s];
  }
  if (old_r !== 1n) throw new Error("not invertible");
  return ((old_s % m) + m) % m;
}

/** The 80-byte GLADOSIG tail, refused rather than guessed at. */
export function parseSig(bytes) {
  if (bytes.length !== SIG_LEN) return { ok: false, why: "not 80 bytes" };
  const magic = new TextDecoder().decode(bytes.slice(0, 8));
  if (magic !== "GLADOSIG") return { ok: false, why: "not a GLADOSIG signature" };
  const version = bytes[8] | (bytes[9] << 8) | (bytes[10] << 16) | (bytes[11] << 24);
  const curve = bytes[12] | (bytes[13] << 8) | (bytes[14] << 16) | (bytes[15] << 24);
  if (version !== 1 || curve !== 0) {
    return { ok: false, why: "a signature format this does not implement" };
  }
  const r = toBigInt(bytes.slice(16, 48));
  const s = toBigInt(bytes.slice(48, 80));
  if (!(0n < r && r < N && 0n < s && s < N)) {
    return { ok: false, why: "r or s is out of range" };
  }
  return { ok: true, r, s };
}

function pubPoint(hex) {
  if (hex.length !== 130 || !hex.startsWith("04")) {
    throw new Error("the public key is not uncompressed 04||X||Y");
  }
  const x = BigInt("0x" + hex.slice(2, 66));
  const y = BigInt("0x" + hex.slice(66, 130));
  return [x, y];
}

/** ECDSA verify of `sig` over sha256(data), against an uncompressed hex key. */
export async function verifyRaw(pubHex, data, sig) {
  const p = parseSig(sig);
  if (!p.ok) return p;
  const digest = await sha256(data);
  const z = toBigInt(digest);
  const q = pubPoint(pubHex);
  const w = invMod(p.s, N);
  const u1 = (z * w) % N;
  const u2 = (p.r * w) % N;
  const pt = add(mul(u1, G), mul(u2, q));
  if (pt === null) return { ok: false, why: "not a signature over these bytes by this key" };
  if (pt[0] % N !== p.r) {
    return { ok: false, why: "not a signature over these bytes by this key" };
  }
  return { ok: true };
}

/**
 * A verdict blob is its text followed by exactly 80 bytes of GLADOSIG --
 * one object, the manifest.rs argument. Answers the verified text, or why
 * not. The text itself is NOT parsed here beyond the split: the kernel's
 * `parse_verdict` is the authority on the format, and this side's only
 * question is whether the verdict key signed these bytes.
 */
export async function verifyVerdict(bin) {
  if (bin.length <= SIG_LEN) {
    return { ok: false, why: `${bin.length} B is too short to be a signed verdict` };
  }
  const text = bin.slice(0, bin.length - SIG_LEN);
  const sig = bin.slice(bin.length - SIG_LEN);
  const v = await verifyRaw(VERDICT_KEY_HEX, text, sig);
  if (!v.ok) return v;
  return { ok: true, text };
}
