// Recovering an address from a signature, and reading a balance off a chain.
//
// Plain JS and imported rather than inlined, for the reason `gladosig.js` is:
// it is the half node can run, so it can be cross-checked outside the
// function that uses it.
//
// ### What is actually being trusted here
//
// `personal_sign` proves one thing: whoever produced the signature holds the
// key for the recovered address. It says nothing about a chain, a balance, or
// a token -- which is useful, because it means the wallet never has to be on
// Robinhood Chain for the proof to work. The signature establishes the
// address; a separate `eth_call` answers for the balance. Keeping those two
// apart is why a holder with MetaMask that has never added chain 4663 can
// still link.
//
// ### The independent check
//
// The recovery below has no published vector in this repository, and one that
// this file generated for itself would prove nothing. The real check is the
// first use: the wallet page shows the address the *wallet* reports and the
// address the *server* recovered, and if the recovery were wrong they would
// differ. MetaMask and Phantom are the independent implementations, and they
// disagree loudly rather than silently.

import { secp256k1 } from "npm:@noble/curves@1.4.0/secp256k1";
import { keccak_256 } from "npm:@noble/hashes@1.4.0/sha3";
import { encodeIsValidSignature, isMagic, unwrap6492 } from "./holders.js";

const hex = (b) => "0x" + Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");

function unhex(s) {
  const t = s.startsWith("0x") ? s.slice(2) : s;
  if (t.length % 2) throw new Error("odd-length hex");
  const out = new Uint8Array(t.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(t.slice(2 * i, 2 * i + 2), 16);
  return out;
}

/// EIP-55: the mixed-case checksum an address is normally written in.
///
/// Comparison happens on the lowercase form everywhere in this codebase, and
/// this exists only for display. Comparing checksummed strings is a bug that
/// shows up as "the same address does not match itself".
export function toChecksum(addr) {
  const lower = addr.toLowerCase().replace(/^0x/, "");
  const h = hex(keccak_256(new TextEncoder().encode(lower))).slice(2);
  let out = "0x";
  for (let i = 0; i < lower.length; i++)
    out += parseInt(h[i], 16) >= 8 ? lower[i].toUpperCase() : lower[i];
  return out;
}

/// The EIP-191 prefixed hash `personal_sign` actually signs.
///
/// The prefix is what stops a signature captured from one context being
/// replayed as a transaction: a prefixed message can never be valid RLP, so a
/// wallet signing a login cannot be tricked into signing a transfer.
function personalHash(message) {
  const body = new TextEncoder().encode(message);
  const prefix = new TextEncoder().encode("\x19Ethereum Signed Message:\n" + body.length);
  const all = new Uint8Array(prefix.length + body.length);
  all.set(prefix, 0);
  all.set(body, prefix.length);
  return keccak_256(all);
}

/// The address that signed `message`, lowercase, or null.
///
/// Returns null rather than throwing on anything malformed. A caller deciding
/// whether to hand out a credential should not have to tell a bad signature
/// from a bad hex string, and both answers are "no".
export function recoverAddress(message, signature) {
  try {
    const sig = unhex(signature);
    if (sig.length !== 65) return null;
    // The last byte is v. Wallets send 27/28; some send 0/1.
    let v = sig[64];
    if (v >= 27) v -= 27;
    if (v !== 0 && v !== 1) return null;

    const rs = secp256k1.Signature.fromCompact(sig.slice(0, 64)).addRecoveryBit(v);
    const pub = rs.recoverPublicKey(personalHash(message)).toRawBytes(false);
    // Drop the 0x04 tag; the address is the last 20 bytes of the hash of the
    // remaining 64.
    return hex(keccak_256(pub.slice(1)).slice(-20));
  } catch {
    return null;
  }
}

/// One `eth_call`, answering the raw hex result.
///
/// Throws on a bad RPC answer rather than returning zero. Zero is a real
/// balance and a legitimate refusal; an RPC that is down is a different thing
/// and must not be silently converted into "this person holds nothing".
async function rpcCall(rpc, method, params) {
  const res = await fetch(rpc, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ jsonrpc: "2.0", id: 1, method, params }),
  });
  if (!res.ok) throw new Error("rpc http " + res.status);
  const j = await res.json();
  if (j.error) throw new Error("rpc: " + (j.error.message || "error"));
  if (typeof j.result !== "string" || !j.result.startsWith("0x")) throw new Error("rpc: no result");
  return j.result;
}

const ethCall = (rpc, to, data, block = "latest") =>
  rpcCall(rpc, "eth_call", [{ to, data }, block]);

/// The latest block number, so a balance and a supply can be read at one height.
export async function blockNumber(rpc) {
  return await rpcCall(rpc, "eth_blockNumber", []);
}

/// `balanceOf(address)` against an ERC-20, as a BigInt.
export async function balanceOf(rpc, token, address, block = "latest") {
  const data = "0x70a08231" + address.toLowerCase().replace(/^0x/, "").padStart(64, "0");
  return BigInt(await ethCall(rpc, token, data, block));
}

/// `totalSupply()` against an ERC-20, as a BigInt. Read at the same block as
/// the balances it divides, or a mint or burn between the two reads moves the
/// share by an amount nobody can account for afterwards.
export async function totalSupply(rpc, token, block = "latest") {
  return BigInt(await ethCall(rpc, token, "0x18160ddd", block));
}

/// Whether `address` has code on the chain `rpc` serves.
export async function hasCode(rpc, address) {
  const c = await rpcCall(rpc, "eth_getCode", [address, "latest"]);
  return c !== "0x" && !/^0x0*$/.test(c);
}

/// Who signed: the address when it signed `message`, or null.
///
/// An externally owned key first, by recovery. When recovery does not name the
/// address -- which is what a contract wallet's signature looks like -- the
/// account is asked through EIP-1271 on the token's own chain, with an ERC-6492
/// wrapper unwrapped first. Answers `"eoa"`, `"1271"` or null, so the caller
/// can say which kind of proof it accepted. An RPC failure throws; it is not a
/// refusal of the signature.
export async function verifySigner(rpc, message, address, signature) {
  const addr = address.toLowerCase();
  if (/^0x[0-9a-fA-F]{130}$/.test(signature) && recoverAddress(message, signature) === addr)
    return "eoa";
  let sig = signature;
  try {
    sig = unwrap6492(signature) ?? signature;
  } catch {
    return null;
  }
  if (!(await hasCode(rpc, addr))) return null;
  const data = encodeIsValidSignature(hex(personalHash(message)), sig);
  try {
    return isMagic(await ethCall(rpc, addr, data)) ? "1271" : null;
  } catch {
    // A reverting `isValidSignature` is the account saying no.
    return null;
  }
}
