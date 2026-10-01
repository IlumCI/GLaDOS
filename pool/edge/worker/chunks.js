// Durable Object storage caps one value at 128 KiB and one atomic put at 128
// keys. A ledger, a snapshot of every address that ever mined, or an epoch that
// lists everybody it left out all outgrow the first; so a long string is stored
// as ordered chunks under one head, and every chunk goes in the same put as the
// head (and as whatever state refers to it), which is what keeps a reader from
// ever seeing half of one.
export const CHUNK = 100_000;

// The entries to put: `key` -> {chunks: n}, `key#i` -> the i-th slice.
export function textChunks(key, text) {
  const n = Math.max(1, Math.ceil(text.length / CHUNK));
  if (n > 127) throw new Error(`${key} is ${text.length} bytes, past what one atomic write can hold`);
  const out = { [key]: { chunks: n } };
  for (let i = 0; i < n; i++) out[`${key}#${i}`] = text.slice(i * CHUNK, (i + 1) * CHUNK);
  return out;
}

// The string back, or undefined. A plain string under `key` is the layout
// before chunking and is returned as it is. A missing chunk throws: a partial
// ledger loaded and then saved over the whole one is the log erased.
export async function loadText(storage, key) {
  const head = await storage.get(key);
  if (head === undefined || head === null || typeof head === "string") return head ?? undefined;
  const keys = [...Array(head.chunks).keys()].map((i) => `${key}#${i}`);
  const m = await storage.get(keys);
  const parts = keys.map((k) => m.get(k));
  if (parts.some((x) => typeof x !== "string")) throw new Error(`${key} is missing a chunk; refusing to read a partial value`);
  return parts.join("");
}
