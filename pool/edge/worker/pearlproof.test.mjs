// node pool/edge/worker/pearlproof.test.mjs
//
// The Pearl verifier's own refusals, each pinned to its reason. A proof broken
// one way is usually also broken another way further down, so "refused" alone
// would pass with any one check deleted or inverted -- mutation testing found
// exactly that. Here each broken proof must be refused *for its own reason*,
// and the share bound is pinned to the bit.
import { readFileSync } from "node:fs";
import { verifyShare, decodeProof, encodeProof, fromBase64 } from "./pearlproof.js";

let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.log(`${c ? "ok  " : "FAIL"}  ${w}`); };

const fx = JSON.parse(readFileSync(new URL("./pearl-fixtures.json", import.meta.url), "utf8"));
const v = fx.find((t) => t.name === "square-16x16-salted");
const header = fromBase64(Buffer.from(v.header.bytes, "hex").toString("base64"));
const proof = fromBase64(v.plain_proof_b64);
const MAX = (1n << 256n) - 1n;
const jh = BigInt("0x" + Buffer.from(v.jackpot_hash, "hex").reverse().toString("hex")); // the hash, read little-endian as the bound is

const good = verifyShare(proof, header, { bound: MAX });
ok(good.ok && good.jackpotHash === v.jackpot_hash, "the fixture verifies, with Pearl's own jackpot hash");

// The bound, to the unit: a hash equal to the bound passes, one over it is low.
ok(verifyShare(proof, header, { bound: jh }).ok, "a hash exactly at the bound is a share");
const low = verifyShare(proof, header, { bound: jh - 1n });
ok(!low.ok && /above the target/.test(low.why), `one under it is 'above the target' (${low.why})`);

// A copy of the decoded proof, broken one way, re-encoded.
const broken = (f) => { const p = decodeProof(proof); f(p); return encodeProof(p); };
const why = (bytes) => verifyShare(bytes, header, { bound: MAX });

const shiftA = why(broken((p) => { p.a.rowIndices = p.a.rowIndices.map((x) => x + 1); }));
ok(!shiftA.ok && /not a 16 x 16 aligned block/.test(shiftA.why), `A's rows off the 16-grid: refused as a bad tile (${shiftA.why})`);
const shiftB = why(broken((p) => { p.bt.rowIndices = p.bt.rowIndices.map((x) => x + 1); }));
ok(!shiftB.ok && /not a 16 x 16 aligned block/.test(shiftB.why), `B's rows off the 16-grid, A's fine: still a bad tile (${shiftB.why})`);
const gap = why(broken((p) => { p.a.rowIndices[5] += 1; }));
ok(!gap.ok && /not a 16 x 16 aligned block/.test(gap.why), "rows that are not contiguous: a bad tile");

const past = why(broken((p) => { const t = p.a; t.leafIndices[t.leafIndices.length - 1] = t.totalLeaves; }));
ok(!past.ok && /past the end of its tree/.test(past.why), `a Merkle leaf index equal to the leaf count is past the end (${past.why})`);
const order = why(broken((p) => { const t = p.a; if (t.leafIndices.length > 1) t.leafIndices[1] = t.leafIndices[0]; }));
ok(decodeProof(proof).a.leafIndices.length < 2 || (!order.ok && /out of order/.test(order.why)), "a repeated Merkle leaf is out of order");

for (const rank of [0, 96, 127, 129, 2048]) {
  const r = why(broken((p) => { p.rank = rank; }));
  ok(!r.ok && /rank/.test(r.why), `rank ${rank}: refused on the rank (${r.why})`);
}

const junk = verifyShare(new Uint8Array([1, 2, 3, 4, 5]), header, { bound: MAX });
ok(!junk.ok && /unreadable share/.test(junk.why), `five bytes of junk: unreadable, not ok (${junk.why})`);
const empty = verifyShare(new Uint8Array(0), header, { bound: MAX });
ok(!empty.ok, "no bytes at all: refused");
const short = verifyShare(proof, header.subarray(0, 75), { bound: MAX });
ok(!short.ok && /76 bytes/.test(short.why), "a 75-byte header: refused");

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
