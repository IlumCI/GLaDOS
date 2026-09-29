// A very small harness: deploy to a real EVM, call functions, read reverts.
//
// `@ethereumjs/vm` executes the same bytecode a chain would, in-process, with
// no node and no network. That matters more than convenience: the alternative
// to running the contract is reading it, and this repository's whole position
// is that reading is where confident mistakes come from.
import { VM } from "@ethereumjs/vm";
import { Chain, Common, Hardfork } from "@ethereumjs/common";
import { Account, Address, hexToBytes, bytesToHex } from "@ethereumjs/util";
import { ethers } from "ethers";

const common = new Common({ chain: Chain.Mainnet, hardfork: Hardfork.Shanghai });

export async function makeVm() {
  return await VM.create({ common });
}

export function addr(hex) {
  return new Address(hexToBytes(hex));
}

/// Give an account a balance so it can pay for gas.
export async function fund(vm, address, wei = 10n ** 20n) {
  const a = addr(address);
  const acc = (await vm.stateManager.getAccount(a)) ?? new Account();
  acc.balance = wei;
  await vm.stateManager.putAccount(a, acc);
}

/// Deploy and return the created address.
export async function deploy(vm, from, artifact, args = []) {
  const iface = new ethers.Interface(artifact.abi);
  const ctor = iface.deploy;
  const encoded = ctor.inputs.length
    ? ethers.AbiCoder.defaultAbiCoder().encode(ctor.inputs.map((i) => i.type), args)
    : "0x";
  const data = hexToBytes("0x" + artifact.bytecode + encoded.slice(2));
  const res = await vm.evm.runCall({
    caller: addr(from),
    origin: addr(from),
    to: undefined,
    data,
    gasLimit: 30_000_000n,
    value: 0n,
  });
  if (res.execResult.exceptionError) {
    throw new Error(`deploy reverted: ${res.execResult.exceptionError.error}`);
  }
  return bytesToHex(res.createdAddress.bytes);
}

/// Call a function. Returns `{ ok, result, reason }` rather than throwing, so a
/// test can assert on a refusal as readily as on a success -- which is most of
/// what there is to check in a contract that mostly says no.
export async function call(vm, from, to, artifact, fn, args = [], opts = {}) {
  const iface = new ethers.Interface(artifact.abi);
  const data = hexToBytes(iface.encodeFunctionData(fn, args));
  const res = await vm.evm.runCall({
    caller: addr(from),
    origin: addr(from),
    to: addr(to),
    data,
    gasLimit: 30_000_000n,
    value: opts.value ?? 0n,
    block: opts.block,
  });
  const ret = bytesToHex(res.execResult.returnValue);
  if (res.execResult.exceptionError) {
    return { ok: false, reason: decodeRevert(iface, ret), raw: ret, gas: res.execResult.executionGasUsed };
  }
  let result = null;
  try {
    const decoded = iface.decodeFunctionResult(fn, ret);
    result = decoded.length === 1 ? decoded[0] : decoded;
  } catch {
    result = ret;
  }
  return { ok: true, result, logs: res.execResult.logs ?? [], gas: res.execResult.executionGasUsed };
}

/// Turn revert data into the name of the custom error, so a test can say which
/// refusal it expected rather than merely that one happened. Getting the wrong
/// refusal is a real failure and a bare "it reverted" hides it.
export function decodeRevert(iface, ret) {
  if (!ret || ret === "0x") return "(no data)";
  try {
    const e = iface.parseError(ret);
    if (e) return e.args.length ? `${e.name}(${e.args.join(",")})` : e.name;
  } catch {
    /* fall through */
  }
  try {
    return "Error: " + ethers.AbiCoder.defaultAbiCoder().decode(["string"], "0x" + ret.slice(10))[0];
  } catch {
    return ret.slice(0, 20);
  }
}
