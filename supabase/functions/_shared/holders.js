// What a holder is, decided without a network, a database or a wallet.
//
// Plain JS with no `npm:` imports, for the reason `envelope.js` is: node can
// load it, so `holders.test.mjs` can check the arithmetic that decides who is
// let into a chat without an Edge Function runtime to host it.
//
// ### A share is of `totalSupply()`, read at the same block as the balance
//
// "Holds 2%" has three readings -- of the minted billion, of the supply the
// contract reports, or of what is not sitting in the pool and the dead address.
// The second is the only one the chain answers in one call and the only one a
// holder can check for themselves, so it is the one used. Tokens sent to
// `0x...dEaD` still count toward `totalSupply()` on a contract that never burns
// for real, which makes the bar slightly *higher* than a circulating-supply
// reading would; that is the conservative direction to be wrong in for a gate.
//
// ### The comparison is exact, never a rounded percentage
//
// `balance * 10000 >= bps * supply` in BigInt. A float share rounded to four
// places admits a holder one wei short of the line, and a gate that can be
// argued with about its last digit is not a gate.

export const BPS = 10000n;

/// Tiers from `HOLDER_TIERS`: `bps[=chat],bps[=chat],...`, e.g.
/// `200=-1001234567890,400=-1009876543210`.
///
/// A tier without a chat is still a tier -- the page reports it -- it simply
/// has nowhere to admit anybody. Sorted ascending and refused outright on
/// anything malformed, because a typo that silently dropped the 4% chat would
/// leave it open to every 2% holder.
export function parseTiers(spec) {
  const s = (spec ?? "").trim();
  if (!s) return [{ bps: 200, chat: null }, { bps: 400, chat: null }];
  const out = [];
  for (const part of s.split(",")) {
    const m = /^\s*(\d{1,5})\s*(?:=\s*(-?\d{1,20}|@[A-Za-z][A-Za-z0-9_]{4,31}))?\s*$/.exec(part);
    if (!m) throw new Error(`HOLDER_TIERS: cannot read '${part}'`);
    const bps = Number(m[1]);
    if (bps < 1 || bps > 10000) throw new Error(`HOLDER_TIERS: ${bps} bps is outside 1..10000`);
    if (out.some((t) => t.bps === bps)) throw new Error(`HOLDER_TIERS: ${bps} bps appears twice`);
    const chat = m[2] ?? null;
    if (chat && out.some((t) => t.chat === chat))
      throw new Error(`HOLDER_TIERS: chat ${chat} is named by two tiers`);
    out.push({ bps, chat });
  }
  return out.sort((a, b) => a.bps - b.bps);
}

/// Whether `balance` is at least `bps` basis points of `supply`. Exact.
export function meets(balance, supply, bps) {
  if (supply <= 0n) return false;
  return balance * BPS >= BigInt(bps) * supply;
}

/// The highest tier met, as its bps, or 0 for none.
export function tierOf(balance, supply, tiers) {
  let best = 0;
  for (const t of tiers) if (meets(balance, supply, t.bps)) best = Math.max(best, t.bps);
  return best;
}

/// The chats a member at `tier` bps may be in: every chat whose bar is at or
/// below it. A 4% holder is also a 2% holder.
export function chatsFor(tier, tiers) {
  return tiers.filter((t) => t.chat && t.bps <= tier).map((t) => t.chat);
}

/// The bps a chat requires, or null if no tier names it.
export function barOf(chat, tiers) {
  const t = tiers.find((t) => t.chat !== null && String(t.chat) === String(chat));
  return t ? t.bps : null;
}

/// A share as a decimal percentage string, truncated (never rounded up) to
/// `places`, for display only. Nothing decides on this string.
export function percent(balance, supply, places = 4) {
  if (supply <= 0n) return "0";
  const scale = 10n ** BigInt(places);
  const v = (balance * 100n * scale) / supply;
  const whole = v / scale;
  const frac = (v % scale).toString().padStart(places, "0");
  return places ? `${whole}.${frac}` : `${whole}`;
}

/// A raw 18-decimal amount as a whole-token string with thousands separators,
/// truncated. Display only.
export function tokens(raw, decimals = 18) {
  const w = (raw / 10n ** BigInt(decimals)).toString();
  return w.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/// The EIP-4361 message a holder signs.
///
/// The Telegram user id is in the statement *and* in a resource line, so the
/// signature names the account it admits: a signature produced for one
/// Telegram user cannot be replayed to admit another, because the text that
/// was signed says which. The statement is one line, as EIP-4361's grammar
/// requires -- a newline inside it is a different message to every parser
/// that follows the spec.
export function holderMessage({ domain, address, tgUser, chainId, nonce, issued, expires }) {
  if (!/^\d{1,20}$/.test(String(tgUser))) throw new Error("tgUser must be a numeric id");
  return [
    `${domain} wants you to sign in with your Ethereum account:`,
    address,
    "",
    `Link this wallet to Telegram user ${tgUser} for the GLaDOS holder chats. ` +
      "This signature costs nothing, moves nothing, and approves no transaction.",
    "",
    `URI: https://${domain}/holders/`,
    "Version: 1",
    `Chain ID: ${chainId}`,
    `Nonce: ${nonce}`,
    `Issued At: ${issued.toISOString()}`,
    `Expiration Time: ${expires.toISOString()}`,
    "Resources:",
    `- tg://user?id=${tgUser}`,
  ].join("\n");
}

// ---- contract wallets -----------------------------------------------------
//
// `ecrecover` answers for a key, and a growing share of wallets are not a key:
// Safe, Coinbase Smart Wallet, passkey accounts and every ERC-4337 account are
// contracts, and they sign through EIP-1271 -- the account itself is asked
// whether a signature is valid. "Any wallet" therefore means asking the chain
// when recovery does not name the address. ERC-6492 wraps a signature from an
// account not yet deployed; it is unwrapped here and honoured only once the
// account exists on chain 4663, because validating an undeployed one means
// simulating its factory, and the factory is code this server has never read.

export const EIP1271_MAGIC = "0x1626ba7e";
export const ERC6492_SUFFIX = "6492649264926492649264926492649264926492649264926492649264926492";

const strip = (h) => (h.startsWith("0x") ? h.slice(2) : h).toLowerCase();
const word = (n) => BigInt(n).toString(16).padStart(64, "0");

/// Calldata for `isValidSignature(bytes32 hash, bytes signature)`.
export function encodeIsValidSignature(hashHex, sigHex) {
  const h = strip(hashHex);
  const s = strip(sigHex);
  if (!/^[0-9a-f]{64}$/.test(h)) throw new Error("hash must be 32 bytes");
  if (!/^[0-9a-f]*$/.test(s) || s.length % 2) throw new Error("signature is not hex");
  const len = s.length / 2;
  const padded = s.padEnd(Math.ceil(len / 32) * 64, "0");
  return EIP1271_MAGIC + h + word(0x40) + word(len) + padded;
}

/// Whether an `eth_call` answer to `isValidSignature` is the magic value. The
/// return is `bytes4` left-aligned in a word; anything shorter, longer or
/// different is a no.
export function isMagic(resultHex) {
  const r = strip(resultHex ?? "");
  return r.length === 64 && r.startsWith(strip(EIP1271_MAGIC)) && /^0*$/.test(r.slice(8));
}

/// The inner signature of an ERC-6492 wrapper, or null when `sigHex` is not
/// one. The wrapper is `abi.encode(address factory, bytes calldata, bytes sig)`
/// followed by the 32-byte magic suffix.
export function unwrap6492(sigHex) {
  const s = strip(sigHex);
  if (!s.endsWith(ERC6492_SUFFIX)) return null;
  const body = s.slice(0, -ERC6492_SUFFIX.length);
  const at = (byte) => {
    const w = body.slice(byte * 2, byte * 2 + 64);
    if (w.length !== 64) throw new Error("ERC-6492 wrapper is truncated");
    return BigInt("0x" + w);
  };
  const off = Number(at(64));
  const len = Number(at(off));
  const start = (off + 32) * 2;
  if (start + len * 2 > body.length) throw new Error("ERC-6492 inner signature runs past the end");
  return "0x" + body.slice(start, start + len * 2);
}

/// A bearer token for a Telegram session, url-safe, 32 random bytes.
export function sessionToken() {
  const b = crypto.getRandomValues(new Uint8Array(32));
  let s = "";
  for (const x of b) s += String.fromCharCode(x);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export async function sha256Hex(s) {
  const d = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return Array.from(new Uint8Array(d), (b) => b.toString(16).padStart(2, "0")).join("");
}
