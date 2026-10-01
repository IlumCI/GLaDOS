// node pool/edge/worker/treasury.test.mjs
//
// The treasury's pure half against values nothing in this file computed:
// Python's hashlib for the Ravencoin address and the legacy sighash (a
// hand-written serialiser, independent of treasury.js), and the address of
// private key 1, which every Ethereum tool agrees on. No network, no storage,
// no money: a signer that is wrong here is wrong before any coin moves.
import * as secp from "@noble/secp256k1";
import { evmAddress, rvnAddress, rvnSighash, signRvnTx, newKey, decodeBase58check, unhex, signLegacyTx, createdAddress,
         rvnScript, rvnTxid, encodeBuyAndPay, encodeCtor, amountOut, hex } from "./treasury.js";
import { ethers } from "../../../contracts/node_modules/ethers/lib.esm/index.js";

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

// EIP-155's own example, from the EIP text: key 0x4646..46, nonce 9, 20 gwei,
// 21000 gas, to 0x3535..35, 1 ETH, no data, chain 1. RFC 6979 nonces make the
// signature deterministic, so the whole signed transaction must match byte
// for byte -- a wrong RLP length, a zero encoded as 0x00, or v off by the
// chain-id arithmetic all show up here.
const eip155 = signLegacyTx("46".repeat(32), {
  nonce: 9, gasPrice: 20_000_000_000n, gas: 21000, to: "0x" + "35".repeat(20),
  value: 10n ** 18n, data: "", chainId: 1,
});
ok(eip155.raw === "0xf86c098504a817c800825208943535353535353535353535353535353535353535880de0b6b3a76400008025a028ef61340bd939bc2195fe537567866003e1a15d3c71ff63e1590620aa636276a067cbe9d8997f761aecb703304b3800ccf555c9f3dc64214b297fb1966a3b6d83",
   "a legacy EIP-155 transaction signs byte-for-byte as the EIP's own example");
ok(eip155.hash === "0x33469b22e9f636356c4160a87eb19df52b7412e8eac32a4a55ffe88ea8350788",
   "and hashes to the EIP example's transaction hash");
// The CREATE address rule, against a value every tool agrees on: the first
// contract deployed by 0x6ac7..6ac7 at nonce 0.
ok(createdAddress("0x6ac7ea33f8831ea9dcc53393aaa88b25a785dbf0", 0) === "0xcd234a471b72ba2f1ccf0a70fcaba648a5eecd8d",
   "a created contract's address follows the sender and nonce");

// ABI encodings against ethers, which is not this file.
const pay = new ethers.Interface(["function buyAndPay(address[] to, uint256 minOut)"]);
const rcpts = ["0x00000000000000000000000000000000000000aa", "0x" + "bb".repeat(20)];
ok(encodeBuyAndPay(rcpts, 12345n) === pay.encodeFunctionData("buyAndPay", [rcpts, 12345n]),
   "buyAndPay calldata is byte-identical to ethers' encoding");
const W = "0x0bd7d308f8e1639fab988df18a8011f41eacad73", T = "0x3d609ecafc6aa7dba67dd7ad1d10b49c52d57777", P = "0x93f777932d98d15b351d1bce8c76b34381eede5b";
ok(encodeCtor(W, T, P) === ethers.AbiCoder.defaultAbiCoder().encode(["address", "address", "address"], [W, T, P]).slice(2),
   "and so are the constructor arguments");
ok(amountOut(10n ** 15n, 6556353309807986866n, 263_900_000n * 10n ** 18n) > 0n, "the swap quote is computed");

// A P2SH output is written as OP_HASH160 <20> OP_EQUAL, and the txid is the
// reversed double SHA-256 of the raw bytes.
const shAddr = (() => {
  const body = new Uint8Array(21); body[0] = 122; body.fill(7, 1);
  return ethers.encodeBase58(ethers.concat([body, ethers.dataSlice(ethers.sha256(ethers.sha256(body)), 0, 4)]));
})();
ok(hex(rvnScript(shAddr)) === "a914" + "07".repeat(20) + "87", "a Ravencoin P2SH address pays OP_HASH160 <hash> OP_EQUAL");
const rawHex = signRvnTx(KEY1, inputs, outputs);
ok(rvnTxid(rawHex) === ethers.sha256(ethers.sha256("0x" + rawHex)).slice(2).match(/../g).reverse().join(""), "a txid is the reversed double SHA-256");

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
