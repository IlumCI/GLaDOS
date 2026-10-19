// `cloudflare:sockets` under node, for edge-sim.mjs. A test sets
// `globalThis.__edgeConnect` to stand up an upstream; without one, every
// connect fails the way an unreachable host does.
export function connect(address) {
  if (typeof globalThis.__edgeConnect === "function") return globalThis.__edgeConnect(address);
  throw new Error(`proxy request failed, cannot connect to ${address}`);
}
