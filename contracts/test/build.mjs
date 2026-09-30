// Compile the contracts with solc's own JS binding.
//
// No hardhat and no foundry: what is needed here is a compiler and an EVM, and
// both are one npm package each. A framework would bring a config file, a
// plugin system and a directory layout, none of which this repository would
// then be able to explain the way it explains everything else.
import solc from "solc";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, "..");

export function compile(files) {
  const sources = {};
  for (const f of files) {
    sources[f] = { content: fs.readFileSync(path.join(root, "src", f), "utf8") };
  }
  const input = {
    language: "Solidity",
    sources,
    settings: {
      // Optimised, because the deployed bytecode is what gets verified on the
      // explorer and testing an unoptimised build then shipping an optimised
      // one is testing a different program.
      optimizer: { enabled: true, runs: 200 },
      evmVersion: "paris",
      outputSelection: { "*": { "*": ["abi", "evm.bytecode.object", "evm.deployedBytecode.object"] } },
    },
  };
  const out = JSON.parse(solc.compile(JSON.stringify(input)));
  const errors = (out.errors || []).filter((e) => e.severity === "error");
  if (errors.length) {
    for (const e of errors) console.error(e.formattedMessage);
    throw new Error(`${errors.length} compile error(s)`);
  }
  for (const w of (out.errors || []).filter((e) => e.severity === "warning")) {
    console.error("warning:", w.formattedMessage.trim().split("\n")[0]);
  }
  return out.contracts;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const c = compile(["GladosDistributor.sol", "TestToken.sol", "GladosPayout.sol", "GladosReader.sol"]);
  const dir = path.join(root, "out");
  fs.mkdirSync(dir, { recursive: true });
  for (const [file, entries] of Object.entries(c)) {
    for (const [name, art] of Object.entries(entries)) {
      fs.writeFileSync(
        path.join(dir, `${name}.json`),
        JSON.stringify({ abi: art.abi, bytecode: art.evm.bytecode.object }, null, 2),
      );
      console.log(`${file}:${name}  ${art.evm.deployedBytecode.object.length / 2} bytes deployed`);
    }
  }
  // The pool's treasury deploys GladosPayout itself, so it carries the exact
  // bytecode this build produced -- committed beside the Worker, and checked
  // against a fresh compile by test/payout.mjs so the two cannot drift.
  const p = c["GladosPayout.sol"].GladosPayout;
  fs.writeFileSync(path.join(root, "..", "pool", "edge", "worker", "GladosPayout.json"),
    JSON.stringify({ abi: p.abi, bytecode: p.evm.bytecode.object }, null, 2) + "\n");
  // And the reader it runs as an eth_call, never deployed (test/reader.mjs).
  const rd = c["GladosReader.sol"].GladosReader;
  fs.writeFileSync(path.join(root, "..", "pool", "edge", "worker", "GladosReader.json"),
    JSON.stringify({ bytecode: rd.evm.bytecode.object }) + "\n");
}
