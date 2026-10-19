// Module hooks that let node load worker.js as workerd would: the two imports
// only Cloudflare's bundler understands are answered here. Registered by
// edge-sim.mjs; nothing else uses it.
//
//   cloudflare:sockets  -> edge-fake-sockets.mjs, whose connect() the test steers
//   *.wasm              -> a module whose default export is the compiled
//                          WebAssembly.Module, which is what wrangler hands a Worker
export async function resolve(spec, ctx, next) {
  if (spec === "cloudflare:sockets") return { url: new URL("./edge-fake-sockets.mjs", import.meta.url).href, shortCircuit: true };
  return next(spec, ctx);
}

export async function load(url, ctx, next) {
  if (url.endsWith(".wasm")) {
    return {
      format: "module",
      shortCircuit: true,
      source: `import { readFileSync } from "node:fs";\nexport default new WebAssembly.Module(readFileSync(new URL(${JSON.stringify(url)})));\n`,
    };
  }
  return next(url, ctx);
}
