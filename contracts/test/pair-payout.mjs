// Review: a miner whose payout address is the pair (which holds far more than
// the 1,000,000 GLADOS gate) makes every buyAndPay revert. Fork of the real chain.
import { VM } from "@ethereumjs/vm";
import { Chain, Common, Hardfork } from "@ethereumjs/common";
import { RPCStateManager } from "@ethereumjs/statemanager";
import { Account, Address, hexToBytes, bytesToHex } from "@ethereumjs/util";
import { ethers } from "ethers";
import { compile } from "./build.mjs";
const RPC = "https://rpc.mainnet.chain.robinhood.com";
const TOKEN = "0x3d609ecafc6aa7dba67dd7ad1d10b49c52d57777", PAIR = "0x93f777932d98d15b351d1bce8c76b34381eede5b", WETH = "0x0bd7d308f8e1639fab988df18a8011f41eacad73";
const OP = "0x1111111111111111111111111111111111111111";
const coder = ethers.AbiCoder.defaultAbiCoder();
const addr = (h) => new Address(hexToBytes(h.toLowerCase()));
async function run(vm, from, to, data, value = 0n) {
  const res = await vm.evm.runCall({ caller: addr(from), origin: addr(from), to: to ? addr(to) : undefined, data: hexToBytes(data), gasLimit: 30_000_000n, value });
  return { ok: !res.execResult.exceptionError, ret: bytesToHex(res.execResult.returnValue), created: res.createdAddress };
}
const r = await fetch(RPC, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "eth_blockNumber", params: [] }) });
const block = parseInt((await r.json()).result, 16);
const vm = await VM.create({ common: new Common({ chain: Chain.Mainnet, hardfork: Hardfork.Shanghai }), stateManager: new RPCStateManager({ provider: RPC, blockTag: BigInt(block) }) });
const acc = new Account(); acc.balance = 10n ** 20n; await vm.stateManager.putAccount(addr(OP), acc);
const art = compile(["GladosPayout.sol"])["GladosPayout.sol"].GladosPayout;
const iface = new ethers.Interface(art.abi);
const dep = await run(vm, OP, null, "0x" + art.evm.bytecode.object + coder.encode(["address", "address", "address"], [WETH, TOKEN, PAIR]).slice(2));
const payout = bytesToHex(dep.created.bytes);
const bal = await run(vm, OP, TOKEN, "0x70a08231" + PAIR.slice(2).padStart(64, "0"));
console.log("pair GLADOS balance", Number(coder.decode(["uint256"], bal.ret)[0]) / 1e18, "(gate is 1000000)");
const miners = ["0x000000000000000000000000000000000000c0a1", PAIR];
const call = await run(vm, OP, payout, iface.encodeFunctionData("buyAndPay", [miners, 1n]), 10n ** 15n);
let why = "ok"; try { const e = iface.parseError(call.ret); why = `${e.name}(${e.args.join(",")})`; } catch {}
console.log(call.ok ? "FAIL: payout with the pair as recipient succeeded" : `ok: payout with the pair as a recipient reverts: ${why}`);
