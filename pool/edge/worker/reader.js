// Balances and code for many addresses in one eth_call: GladosReader's
// creation code, sent with no `to`, never deployed (contracts/src/
// GladosReader.sol says why -- the public 4663 RPC refuses a 100-call batch).
import reader from "./GladosReader.json" with { type: "json" };

// At most this many addresses per call: ~8k gas each, so 200 is ~1.6M gas.
export const READ_MAX = 200;

const word = (v) => BigInt(v).toString(16).padStart(64, "0");

// The call's data: creation code, then abi.encode(token, who).
export function readerData(token, who) {
  return "0x" + reader.bytecode + word(token) + word(0x40) + word(who.length) + who.map((a) => word(a)).join("");
}

// What came back, per address: {balance (BigInt, or undefined if balanceOf
// failed), code: "none" | "delegated" | "contract"}. Three words an address;
// anything else is a malformed answer and throws.
export function readerDecode(who, hex) {
  const d = String(hex).replace(/^0x/, "");
  if (d.length !== who.length * 192) throw new Error(`reader answered ${d.length / 2} bytes for ${who.length} address(es)`);
  const MAX = (1n << 256n) - 1n;
  return new Map(who.map((a, i) => {
    const w = (k) => BigInt("0x" + d.slice(i * 192 + k * 64, i * 192 + k * 64 + 64));
    const bal = w(0), size = w(1), head = d.slice(i * 192 + 128, i * 192 + 192);
    // EIP-7702: an account whose code is 0xef0100 followed by an address, 23
    // bytes, is a key-held account with a delegate, and is paid like any other.
    const code = size === 0n ? "none" : size === 23n && head.startsWith("ef0100") ? "delegated" : "contract";
    return [a, { balance: bal === MAX ? undefined : bal, code }];
  }));
}
