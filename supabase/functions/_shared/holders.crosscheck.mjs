// The holder gate against an implementation this tree did not write.
//
//     deno run supabase/functions/_shared/holders.crosscheck.mjs
//
// `holders.test.mjs` checks the module against itself, which proves the claims
// are consistent and not that they are right. Four things here have a
// published meaning somebody else has implemented, and viem is that somebody:
//
//   - the message is EIP-4361, so viem's SIWE parser must read back every field;
//   - `isValidSignature` calldata is ABI, so viem's encoder must produce the
//     same bytes;
//   - an ERC-6492 wrapper viem serialises must unwrap to the signature inside;
//   - and a signature viem's account makes must recover, through `evm.js`, to
//     that account -- the check the wallet page otherwise leaves to the first
//     real wallet.
//
// Deno and not node because `evm.js` imports `npm:` specifiers.

import { encodeFunctionData, getAddress, hashMessage } from "npm:viem@2.57.10";
import { parseSiweMessage, validateSiweMessage } from "npm:viem@2.57.10/siwe";
import { serializeErc6492Signature } from "npm:viem@2.57.10/utils";
import { privateKeyToAccount } from "npm:viem@2.57.10/accounts";
import { encodeIsValidSignature, holderMessage, unwrap6492 } from "./holders.js";
import { recoverAddress, verifySigner } from "./evm.js";

let ok = true;
function claim(good, what) {
  console.log(`  ${good ? "ok " : "FAIL"}   ${what}`);
  if (!good) ok = false;
}

// A throwaway key, worthless by being published here.
const acct = privateKeyToAccount("0x" + "42".repeat(32));
const issued = new Date("2026-10-27T00:00:00.000Z");
const expires = new Date("2026-10-27T00:10:00.000Z");
const msg = holderMessage({
  domain: "glados.aperture.institute", address: acct.address, tgUser: 987654321,
  chainId: 4663, nonce: "0123456789abcdef0123456789abcdef", issued, expires,
});

console.log("[crosscheck] EIP-4361");
const p = parseSiweMessage(msg);
claim(p.domain === "glados.aperture.institute", "viem reads the domain");
claim(p.address === acct.address, "and the address");
claim(p.chainId === 4663 && p.version === "1", "and the chain and version");
claim(p.nonce === "0123456789abcdef0123456789abcdef", "and the nonce");
claim(p.uri === "https://glados.aperture.institute/holders/", "and the URI");
claim(p.statement?.includes("Telegram user 987654321"), "and a statement naming the Telegram user");
claim(p.resources?.[0] === "tg://user?id=987654321", "and the resource line");
claim(+p.expirationTime === +expires && +p.issuedAt === +issued, "and both times");
claim(validateSiweMessage({ message: p, domain: "glados.aperture.institute", address: acct.address,
  time: new Date("2026-10-27T00:05:00Z") }), "and validates it inside its window");
claim(!validateSiweMessage({ message: p, domain: "glados.aperture.institute",
  time: new Date("2026-10-27T00:11:00Z") }), "and refuses it after expiry");

console.log("[crosscheck] signatures");
const sig = await acct.signMessage({ message: msg });
claim(recoverAddress(msg, sig) === acct.address.toLowerCase(), "evm.js recovers viem's signer");
claim(await verifySigner("http://127.0.0.1:9", msg, acct.address, sig) === "eoa",
  "verifySigner accepts it as a key, without asking any chain");
claim(recoverAddress(msg.replace("987654321", "987654322"), sig) !== acct.address.toLowerCase(),
  "the same signature over a message naming another Telegram user recovers to someone else");

console.log("[crosscheck] ABI");
const hash = hashMessage(msg);
const theirs = encodeFunctionData({
  abi: [{ type: "function", name: "isValidSignature", stateMutability: "view",
    inputs: [{ type: "bytes32" }, { type: "bytes" }], outputs: [{ type: "bytes4" }] }],
  functionName: "isValidSignature", args: [hash, sig],
});
claim(encodeIsValidSignature(hash, sig) === theirs.toLowerCase(), "isValidSignature calldata is byte-identical to viem's");
const odd = "0x" + "cd".repeat(97);
claim(encodeIsValidSignature(hash, odd) === encodeFunctionData({
  abi: [{ type: "function", name: "isValidSignature", stateMutability: "view",
    inputs: [{ type: "bytes32" }, { type: "bytes" }], outputs: [{ type: "bytes4" }] }],
  functionName: "isValidSignature", args: [hash, odd] }).toLowerCase(),
  "including for a signature that is not a whole number of words");

console.log("[crosscheck] ERC-6492");
const wrapped = serializeErc6492Signature({
  address: getAddress("0x" + "ee".repeat(20)), data: "0xdeadbeef" + "00".repeat(33), signature: odd,
});
claim(unwrap6492(wrapped) === odd, "a wrapper viem serialises unwraps to the signature inside it");

console.log(ok ? "crosscheck: all claims hold" : "crosscheck: FAILED");
Deno.exit(ok ? 0 : 1);
