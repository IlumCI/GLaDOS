// GladosReader, run the way the pool runs it: creation code as a call with no
// `to`, never deployed. Against a real EVM, with a real token, a contract, an
// EIP-7702-shaped account and plain accounts.
import { makeVm, fund, deploy, addr } from "./evm.mjs";
import { compile } from "./build.mjs";
import { hexToBytes, bytesToHex, Account } from "@ethereumjs/util";
import { readerData, readerDecode, READ_MAX } from "../../pool/edge/worker/reader.js";

let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.log(`${c ? "ok  " : "FAIL"}  ${w}`); };
const c = compile(["TestToken.sol"]);
const tok = c["TestToken.sol"].TestToken;
const vm = await makeVm();
const OWNER = "0x" + "11".repeat(20);
await fund(vm, OWNER);
const token = await deploy(vm, OWNER, { abi: tok.abi, bytecode: tok.evm.bytecode.object }, [10n ** 27n]);
const { ethers } = await import("ethers");
const iface = new ethers.Interface(tok.abi);
const mint = iface.fragments.find((f) => f.type === "function" && /^(mint|transfer)$/.test(f.name));
const holder = "0x" + "22".repeat(20), plain = "0x" + "33".repeat(20), delegated = "0x" + "44".repeat(20);
// Give `holder` tokens: the deployer holds the supply in TestToken.
const tx = await vm.evm.runCall({ caller: addr(OWNER), origin: addr(OWNER), to: addr(token), gasLimit: 1_000_000n,
  data: hexToBytes(iface.encodeFunctionData("transfer", [holder, 12345n])) });
ok(!tx.execResult.exceptionError, `setup: ${mint ? mint.name : "?"} ran`);
// An EIP-7702-shaped account: 0xef0100 + an address as its code.
await vm.stateManager.putContractCode(addr(delegated), hexToBytes("0xef0100" + "55".repeat(20)));

async function read(who) {
  const res = await vm.evm.runCall({ caller: addr(OWNER), origin: addr(OWNER), to: undefined, gasLimit: 30_000_000n,
    data: hexToBytes(readerData(token, who)) });
  if (res.execResult.exceptionError) throw new Error(res.execResult.exceptionError.error);
  return { m: readerDecode(who, bytesToHex(res.execResult.returnValue)), gas: res.execResult.executionGasUsed };
}
const { m } = await read([holder, plain, token, delegated]);
ok(m.get(holder).balance === 12345n && m.get(holder).code === "none", "a holder's balance is read exactly, and it is an ordinary account");
ok(m.get(plain).balance === 0n && m.get(plain).code === "none", "an empty account reads zero");
ok(m.get(token).code === "contract", "a contract is seen as one");
ok(m.get(delegated).code === "delegated", "an EIP-7702 delegation is told apart from a contract");
// A token whose balanceOf reverts for everyone: unreadable, not zero.
const { m: bad } = await (async () => {
  const who = [holder];
  const res = await vm.evm.runCall({ caller: addr(OWNER), origin: addr(OWNER), to: undefined, gasLimit: 30_000_000n,
    data: hexToBytes(readerData(plain, who)) });
  return { m: readerDecode(who, bytesToHex(res.execResult.returnValue)) };
})();
ok(bad.get(holder).balance === undefined, "a balanceOf that fails is unreadable, never zero");
const many = [...Array(READ_MAX).keys()].map((i) => "0x" + (i + 1000).toString(16).padStart(40, "0"));
const full = await read(many);
ok(full.m.size === READ_MAX, `${READ_MAX} addresses in one call, ${full.gas} gas`);
let threw = false; try { readerDecode([holder, plain], "0x" + "00".repeat(96)); } catch { threw = true; }
ok(threw, "an answer the wrong length is refused, not half-read");
console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
