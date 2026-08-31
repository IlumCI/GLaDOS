// P-256 ECDSA, and the 80-byte container the kernel reads.
//
// Plain JavaScript rather than TypeScript, and no Deno APIs, for one reason:
// it can then be run by node and cross-checked against `tools/manifest.py`'s
// verifier without a toolchain. A signer nobody can test is a signer that
// produces 80 plausible bytes and is discovered to be wrong by a machine that
// has already refused to boot.
//
// WebCrypto would give a correct ECDSA signature in a container this kernel
// does not read, so the curve arithmetic is written out. It is checked against
// the Python implementation, which is checked against the kernel's verifier,
// which is checked at every boot against published ECDSA vectors.

export const SIG_LEN = 80;

const P = 0xffffffff00000001000000000000000000000000ffffffffffffffffffffffffn;
export const N =
  0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551n;
const A = P - 3n;
const GX = 0x6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296n;
const GY = 0x4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5n;
export const G = [GX, GY];

function mod(a, m) {
  const r = a % m;
  return r < 0n ? r + m : r;
}

// Fermat, since both moduli here are prime. Short and obviously right, which
// matters more than speed at one signature per request.
function inv(a, m) {
  let result = 1n;
  let base = mod(a, m);
  let e = m - 2n;
  while (e > 0n) {
    if (e & 1n) result = (result * base) % m;
    base = (base * base) % m;
    e >>= 1n;
  }
  return result;
}

export function add(p, q) {
  if (p === null) return q;
  if (q === null) return p;
  const [x1, y1] = p;
  const [x2, y2] = q;
  if (x1 === x2 && mod(y1 + y2, P) === 0n) return null;
  const lam = x1 === x2 && y1 === y2
    ? mod((3n * x1 * x1 + A) * inv(2n * y1, P), P)
    : mod((y2 - y1) * inv(x2 - x1, P), P);
  const x3 = mod(lam * lam - x1 - x2, P);
  return [x3, mod(lam * (x1 - x3) - y1, P)];
}

export function mul(k, p) {
  let r = null;
  let acc = p;
  let n = k;
  while (n > 0n) {
    if (n & 1n) r = add(r, acc);
    acc = add(acc, acc);
    n >>= 1n;
  }
  return r;
}

export function toBigInt(bytes) {
  let n = 0n;
  for (const b of bytes) n = (n << 8n) | BigInt(b);
  return n;
}

export function beBytes(v, len) {
  const out = new Uint8Array(len);
  let n = v;
  for (let i = len - 1; i >= 0; i--) {
    out[i] = Number(n & 0xffn);
    n >>= 8n;
  }
  return out;
}

export async function sha256(data) {
  return new Uint8Array(await crypto.subtle.digest("SHA-256", data));
}

export function publicOf(d) {
  const q = mul(d, G);
  const out = new Uint8Array(65);
  out[0] = 0x04;
  out.set(beBytes(q[0], 32), 1);
  out.set(beBytes(q[1], 32), 33);
  return out;
}

/// An 80-byte GLADOSIG over `data`: "GLADOSIG", u32 version, u32 curve, r, s.
///
/// The nonce comes from the platform CSPRNG and never from anything derived
/// from the message. Reusing a nonce across two signatures reveals the private
/// key outright -- not weakens it, reveals it -- which is the one way to get
/// this wrong that cannot be noticed by testing the output.
export async function sign(privHex, data) {
  const d = BigInt("0x" + String(privHex).trim());
  const z = toBigInt(await sha256(data));

  let r = 0n;
  let s = 0n;
  for (;;) {
    const kb = new Uint8Array(32);
    crypto.getRandomValues(kb);
    const k = mod(toBigInt(kb), N);
    if (k === 0n) continue;
    const point = mul(k, G);
    if (point === null) continue;
    r = mod(point[0], N);
    if (r === 0n) continue;
    s = mod(inv(k, N) * (z + r * d), N);
    if (s === 0n) continue;
    break;
  }

  const out = new Uint8Array(SIG_LEN);
  out.set(new TextEncoder().encode("GLADOSIG"), 0);
  out[8] = 1; // version 1, little-endian u32
  // curve 0 (P-256) is the zeroes already there.
  out.set(beBytes(r, 32), 16);
  out.set(beBytes(s, 32), 48);
  return out;
}

/// Text plus its signature, which is what a signed manifest is.
export async function signedManifest(privHex, text) {
  const body = new TextEncoder().encode(text);
  const sig = await sign(privHex, body);
  const out = new Uint8Array(body.length + sig.length);
  out.set(body, 0);
  out.set(sig, body.length);
  return out;
}
