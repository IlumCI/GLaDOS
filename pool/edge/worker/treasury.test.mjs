// node pool/edge/worker/treasury.test.mjs
//
// The treasury's pure half against values nothing in this file computed:
// Python's hashlib for the Ravencoin address and the legacy sighash (a
// hand-written serialiser, independent of treasury.js), and the address of
// private key 1, which every Ethereum tool agrees on. No network, no storage,
// no money: a signer that is wrong here is wrong before any coin moves.
import * as secp from "@noble/secp256k1";
import { evmAddress, rvnAddress, rvnSighash, signRvnTx, newKey, decodeBase58check, unhex } from "./treasury.js";

// DER back to (r, s), written independently of treasury.js's encoder.
function fromDER(d) {
  if (d[0] !== 0x30 || d[1] !== d.length - 2) throw new Error("not a DER sequence");
  const read = (at) => {
    if (d[at] !== 0x02) throw new Error("not an INTEGER");
    const len = d[at + 1];
    const b = d.slice(at + 2, at + 2 + len);
    if (b[0] === 0 && !(b[1] & 0x80)) throw new Error("non-minimal INTEGER");
    return [BigInt("0x" + Buffer.from(b).toString("hex")), at + 2 + len];
  };
  const [r, next] = read(2);
  const [s] = read(next);
  return new secp.Signature(r, s);
}

let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.log(`${c ? "ok  " : "FAIL"}  ${w}`); };

const KEY1 = "00".repeat(31) + "01";

ok(evmAddress(KEY1) === "0x7E5F4552091A69125d5DfCb7b8C2659029395Bdf",
   "private key 1 is Ethereum's 0x7E5F...5Bdf, with its EIP-55 casing");
ok(rvnAddress(KEY1) === "RKxTdfmtxtfLDKZBgx6SvNkBtNu9jRYnLh",
   "and Ravencoin's RKxTdfmt...YnLh (version 60), matching hashlib");

const inputs = [
  { txid: "aa".repeat(32), vout: 0, sats: 50_000_000 },
  { txid: "bb".repeat(31) + "cc", vout: 3, sats: 20_000_000 },
];
const outputs = [{ address: "RKxTdfmtxtfLDKZBgx6SvNkBtNu9jRYnLh", sats: 12_345_678 }];
const want = [
  "8054930cf45b85a0ff5a72787d716c380b41a31828682bb07b45802af2d5c219",
  "87cb6f8b37d1af1b72c69cf735ee4c0dc56f69ab6ddb740c70faab718013afac",
];
ok(rvnSighash(KEY1, inputs, outputs, 0) === want[0], "input 0's legacy sighash matches the independent serialiser");
ok(rvnSighash(KEY1, inputs, outputs, 1) === want[1], "and input 1's, whose outpoint differs only in its last byte");

// The signed transaction: every signature must verify against the digest the
// independent serialiser produced, under the key's public point.
const raw = unhex(signRvnTx(KEY1, inputs, outputs));
const pub = secp.getPublicKey(KEY1, true);
let at = 4 + 1; // version, input count
const sigs = [];
for (let i = 0; i < 2; i++) {
  at += 32 + 4;                     // outpoint
  const len = raw[at]; at += 1;
  const script = raw.slice(at, at + len); at += len + 4;
  const sl = script[0];
  const der = script.slice(1, sl);  // drop the trailing hashtype byte
  ok(script[sl] === 0x01, `input ${i} is signed SIGHASH_ALL`);
  ok(script[sl + 1] === 33 && script.slice(sl + 2).every((b, k) => b === pub[k]), `input ${i} carries the compressed public key`);
  sigs.push(der);
}
for (let i = 0; i < 2; i++) {
  ok(secp.verify(fromDER(sigs[i]), unhex(want[i]), pub, { lowS: true }),
     `input ${i}'s signature verifies against hashlib's digest`);
}
ok(!secp.verify(fromDER(sigs[0]), unhex(want[1]), pub), "and not against the other input's digest");

// Refusals: an output that is not a Ravencoin P2PKH address never gets serialised.
let threw = false;
try { signRvnTx(KEY1, inputs, [{ address: "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", sats: 1 }]); } catch { threw = true; }
ok(threw, "a Bitcoin address as an output is refused, not paid");
threw = false;
try { decodeBase58check("RKxTdfmtxtfLDKZBgx6SvNkBtNu9jRYnLi"); } catch { threw = true; }
ok(threw, "an address with a broken checksum is refused");

const a = newKey(), b = newKey();
ok(a !== b && /^[0-9a-f]{64}$/.test(a), "generated keys are 32 random bytes and differ");

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
