// Sign something with the Edge Function's signer, so Python can check it.
//
//     node supabase/functions/_shared/crosscheck.mjs <out-dir>
//
// Writes a private key, a public key, a signed manifest and a detached
// signature. `tools/manifest.py --verify` then reads all of it. Two
// implementations that are supposed to agree do not stay agreeing, and the one
// that matters here has no test runner of its own.

import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { publicOf, sign, signedManifest } from "./gladosig.js";

const out = process.argv[2] ?? ".";

// A fixed key, so a failure is reproducible. Not a secret and never used for
// anything: the whole point is that it appears in a file in a public repo.
const priv = "c9af a685 1f2b 3d47 8e10 5c62 9b04 7ae3 51d8 26fc 40b9 1e73 8a5d 62c1 0f94 3b28"
  .replace(/\s+/g, "");

const text = [
  "glados-update 1",
  "channel experimental",
  "version 9.9.9",
  "image https://example.invalid/glados-9.9.9.efi",
  "sig https://example.invalid/glados-9.9.9.efi.sig",
  "size 4",
  "sha256 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
  "notes a manifest that exists only in this test",
  "",
].join("\n");

const blob = await signedManifest(priv, text);
const detached = await sign(priv, new TextEncoder().encode(text));

writeFileSync(join(out, "xcheck.priv"), priv);
writeFileSync(join(out, "xcheck.pub"), Buffer.from(publicOf(BigInt("0x" + priv))).toString("hex"));
writeFileSync(join(out, "xcheck.manifest"), Buffer.from(blob));
writeFileSync(join(out, "xcheck.sig"), Buffer.from(detached));

console.log(`  signed ${text.length} B of manifest -> ${blob.length} B`);
