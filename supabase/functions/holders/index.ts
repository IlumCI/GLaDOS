// Holder chats: a Telegram account let into a chat on a share of $GLADOS.
//
// One function, two kinds of caller, told apart by the last path segment:
//
//   Telegram, as the bot's webhook
//     POST /holders/telegram       an Update; authenticated by the secret
//                                  header Telegram echoes back
//
//   The page at /holders/, which a holder opens from the bot's link
//     POST /holders/session  {session}                       -> who the link is for
//     POST /holders/check    {address}                       -> a share, read-only
//     POST /holders/nonce    {session, address}              -> the message to sign
//     POST /holders/verify   {session, address, signature}   -> linked, and the tier
//
//   A schedule, to take away what a sold holding no longer earns
//     POST /holders/recheck  header x-recheck-secret
//
// ### Who is proved to be whom
//
// Two identities have to be joined and neither side can vouch for the other.
// Telegram vouches for the account: the bot hands a session token to whoever
// sent it `/verify`, so a session names a Telegram user. The wallet vouches for
// the address: an EIP-4361 signature over a message that *names that Telegram
// user*. A signature produced for one account therefore cannot admit another,
// and a session link that leaks admits nobody without the key.
//
// What this cannot stop, and nothing could: a holder signing on behalf of a
// friend's session. That is the holder lending their holding, which they could
// equally do by sending the tokens. What it does stop is one holding lending
// itself to many accounts -- a wallet belongs to one Telegram user at most.
//
// ### Where the share is read, and how often
//
// Here, server-side, balance and `totalSupply()` at one block height. At link
// time, again on every join request, and on the recheck schedule -- never on
// the client, and never remembered as a fact about a person. `tier_bps` is a
// cache with one reader, the chat-member handler, and the recheck is what keeps
// it honest.

import { createClient } from "jsr:@supabase/supabase-js@2";
import { balanceOf, blockNumber, toChecksum, totalSupply, verifySigner } from "../_shared/evm.js";
import {
  barOf, holderMessage, parseTiers, percent, sessionToken, sha256Hex, tierOf, tokens,
} from "../_shared/holders.js";
import { bot } from "../_shared/telegram.js";

const db = createClient(Deno.env.get("SUPABASE_URL")!, Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!);

// The token and chain are the same three `link` and `channel` read, from the
// same place, so the doors cannot disagree about which contract counts.
const TOKEN = (Deno.env.get("TOKEN_CONTRACT") ??
  "0x3d609ecafc6aa7dba67dd7ad1d10b49c52d57777").toLowerCase();
const RPC = Deno.env.get("TOKEN_RPC") ?? "https://rpc.mainnet.chain.robinhood.com";
const CHAIN_ID = Number(Deno.env.get("TOKEN_CHAIN_ID") ?? 4663);
const DOMAIN = Deno.env.get("LINK_DOMAIN") ?? "glados.aperture.institute";
const SITE = `https://${DOMAIN}/holders/`;

// Parsed at start-up and allowed to throw there. A malformed tier list that
// started anyway would be a gate with a tier missing, and a function that
// refuses to boot is the louder of the two failures.
const TIERS = parseTiers(Deno.env.get("HOLDER_TIERS"));

const WEBHOOK_SECRET = Deno.env.get("TG_WEBHOOK_SECRET") ?? "";
const RECHECK_SECRET = Deno.env.get("HOLDER_RECHECK_SECRET") ?? "";
const tg = () => bot(Deno.env.get("TG_BOT_TOKEN") ?? "");

const SESSION_TTL_MS = 15 * 60 * 1000;
const NONCE_TTL_MS = 10 * 60 * 1000;
const JOIN_TTL_S = 60 * 60;
const MAX_WALLETS = 5;

const cors = {
  "access-control-allow-origin": "*",
  "access-control-allow-headers": "content-type",
  "access-control-allow-methods": "POST, OPTIONS",
};
const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...cors } });
const refuse = (s: number, why: string) => json(s, { error: why });

const isAddress = (s: unknown): s is string => typeof s === "string" && /^0x[0-9a-fA-F]{40}$/.test(s);

/// Constant time over equal lengths. A secret compared with `===` leaks its
/// prefix one byte at a time to anybody patient enough to time it.
function same(a: string, b: string) {
  if (!a || a.length !== b.length) return false;
  let d = 0;
  for (let i = 0; i < a.length; i++) d |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return d === 0;
}

const tierLabel = (bps: number) => (bps ? `${(bps / 100).toFixed(2)}%` : "none");

// ---- reading a holding ------------------------------------------------------

type Reading = { block: string; supply: bigint };

async function reading(): Promise<Reading> {
  const block = await blockNumber(RPC);
  return { block, supply: await totalSupply(RPC, TOKEN, block) };
}

/// Every wallet a Telegram user has linked, read at one block, and the tier
/// their sum reaches. Writes the readings back. Throws on an RPC failure, so
/// a chain that cannot be read is never recorded as a holding of zero.
async function measure(tgUser: number, at?: Reading, extra?: string) {
  const r = at ?? await reading();
  const { data: rows } = await db.from("holder_wallets").select("address").eq("tg_user", tgUser);
  const addrs = (rows ?? []).map((w: { address: string }) => w.address);
  if (extra && !addrs.includes(extra)) addrs.push(extra);
  const each: Record<string, bigint> = {};
  let total = 0n;
  for (const a of addrs) {
    each[a] = await balanceOf(RPC, TOKEN, a, r.block);
    total += each[a];
  }
  return { ...r, each, total, tier: tierOf(total, r.supply, TIERS) };
}

async function record(tgUser: number, m: Awaited<ReturnType<typeof measure>>) {
  const now = new Date().toISOString();
  for (const [address, balance] of Object.entries(m.each))
    await db.from("holder_wallets").update({ balance: balance.toString(), last_checked: now })
      .eq("address", address);
  await db.from("holder_members").update({
    tier_bps: m.tier, total_balance: m.total.toString(), supply: m.supply.toString(), last_checked: now,
  }).eq("tg_user", tgUser);
}

// ---- membership -------------------------------------------------------------

const IN_CHAT = new Set(["member", "restricted", "administrator", "creator"]);

async function statusIn(chat: string, user: number): Promise<string | null> {
  try {
    return (await tg().member(chat, user)).status;
  } catch {
    return null;
  }
}

/// Bring a user's chats into line with their tier: remove them from every chat
/// whose bar they no longer meet, and send a join link for every chat they now
/// qualify for and are not in. Administrators are never removed -- an admin
/// below the bar is the operator's decision, not the bot's.
async function reconcile(tgUser: number, tier: number, invite: boolean) {
  const removed: string[] = [];
  const links: { chat: string; url: string; bps: number }[] = [];
  for (const t of TIERS) {
    if (!t.chat) continue;
    const status = await statusIn(t.chat, tgUser);
    if (t.bps > tier) {
      if (status === "member" || status === "restricted") {
        try {
          await tg().kick(t.chat, tgUser);
          removed.push(tierLabel(t.bps));
        } catch (e) {
          console.error(`kick ${tgUser} from ${t.chat}:`, (e as Error).message);
        }
      }
    } else if (invite && !(status && IN_CHAT.has(status))) {
      try {
        const l = await tg().joinLink(t.chat, `holder ${tgUser}`, JOIN_TTL_S);
        links.push({ chat: t.chat, url: l.invite_link, bps: t.bps });
      } catch (e) {
        console.error(`join link for ${t.chat}:`, (e as Error).message);
      }
    }
  }
  return { removed, links };
}

function summary(m: Awaited<ReturnType<typeof measure>>) {
  return `${tokens(m.total)} GLADOS across ${Object.keys(m.each).length} wallet(s), ` +
    `${percent(m.total, m.supply)}% of supply -- tier ${tierLabel(m.tier)}`;
}

async function tell(tgUser: number, text: string, links: { url: string; bps: number }[] = []) {
  const extra = links.length
    ? { reply_markup: { inline_keyboard: links.map((l) => [{ text: `Request to join (${tierLabel(l.bps)})`, url: l.url }]) } }
    : {};
  try {
    await tg().send(tgUser, text, extra);
    return true;
  } catch (e) {
    console.error(`dm ${tgUser}:`, (e as Error).message);
    return false;
  }
}

// ---- Telegram ---------------------------------------------------------------

// The fields of an Update this file reads, and no more.
type TgUser = { id: number; is_bot?: boolean; username?: string; first_name?: string };
type TgChat = { id: number; type?: string };
type TgMessage = { message_id: number; text?: string; from?: TgUser; chat: TgChat };
type TgJoinRequest = { chat: TgChat; from: TgUser; user_chat_id?: number };
type TgChatMember = {
  chat: TgChat;
  from?: TgUser;
  old_chat_member?: { status: string };
  new_chat_member: { status: string; user: TgUser };
};

async function newSession(user: TgUser) {
  const token = sessionToken();
  const now = Date.now();
  const { error } = await db.from("holder_sessions").insert({
    token_hash: await sha256Hex(token), tg_user: user.id,
    tg_name: user.username ? `@${user.username}` : (user.first_name ?? null),
    issued_at: new Date(now).toISOString(), expires_at: new Date(now + SESSION_TTL_MS).toISOString(),
  });
  // A link to a session that was never stored would fail on the page with
  // "no such session", which reads as the user's mistake. Say it here instead.
  return error ? null : token;
}

const HELP = [
  "<b>GLaDOS holder chats</b>",
  "",
  `Holders of at least ${TIERS.map((t) => tierLabel(t.bps)).join(" / ")} of the $GLADOS supply are let in.`,
  "",
  "/verify -- link a wallet (any wallet; the page asks for a signature, never a transaction)",
  "/status -- what is linked and the tier it reaches now",
  "/unlink &lt;address|all&gt; -- remove a wallet",
  "",
  "Nothing here will ever ask for a seed phrase or a private key, and the bot never DMs first.",
].join("\n");

async function onMessage(msg: TgMessage) {
  const text: string = msg.text ?? "";
  if (!text.startsWith("/")) return;
  const [head, ...args] = text.trim().split(/\s+/);
  const cmd = head.replace(/@\w+$/, "").toLowerCase();
  const user = msg.from;
  if (!user || user.is_bot) return;

  if (msg.chat?.type !== "private") {
    if (cmd === "/verify")
      await tg().send(msg.chat.id, "Send me /verify in a private message -- a link to your wallet does not belong in a group.", {
        reply_to_message_id: msg.message_id,
      }).catch(() => {});
    return;
  }

  if (cmd === "/start" || cmd === "/verify") {
    if (cmd === "/start" && args[0] !== "verify") return void await tell(user.id, HELP);
    const token = await newSession(user);
    if (!token) return void await tell(user.id, "Could not start a session just now; try /verify again shortly.");
    await tg().send(user.id, [
      "Open this page and connect any wallet that holds $GLADOS.",
      "It asks you to <b>sign a message</b> naming your Telegram account. It never asks for a transaction, an approval, a seed phrase or a key.",
      "",
      `The link is yours alone and works once, for ${SESSION_TTL_MS / 60000} minutes.`,
    ].join("\n"), {
      reply_markup: { inline_keyboard: [[{ text: "Link a wallet", url: `${SITE}?s=${token}` }]] },
    });
    return;
  }

  if (cmd === "/status") {
    const { data: ws } = await db.from("holder_wallets").select("address").eq("tg_user", user.id);
    if (!ws?.length) return void await tell(user.id, "No wallet is linked. Send /verify.");
    try {
      const m = await measure(user.id);
      await record(user.id, m);
      const { removed, links } = await reconcile(user.id, m.tier, true);
      const lines = Object.entries(m.each).map(([a, b]) => `<code>${toChecksum(a)}</code>  ${tokens(b)}`);
      if (removed.length) lines.push("", `Removed from: ${removed.join(", ")}`);
      await tell(user.id, [summary(m), "", ...lines].join("\n"), links);
    } catch {
      await tell(user.id, "Could not read the chain just now; try again shortly.");
    }
    return;
  }

  if (cmd === "/unlink") {
    const which = (args[0] ?? "").toLowerCase();
    if (which !== "all" && !isAddress(which))
      return void await tell(user.id, "/unlink &lt;address&gt; or /unlink all");
    let q = db.from("holder_wallets").delete().eq("tg_user", user.id);
    if (which !== "all") q = q.eq("address", which);
    await q;
    try {
      const m = await measure(user.id);
      await record(user.id, m);
      const { removed } = await reconcile(user.id, m.tier, false);
      await tell(user.id, `Unlinked. Now ${summary(m)}.` + (removed.length ? `\nRemoved from: ${removed.join(", ")}` : ""));
    } catch {
      await tell(user.id, "Unlinked. The chain could not be read; the next recheck settles your chats.");
    }
    return;
  }

  await tell(user.id, HELP);
}

/// A join request is the gate. It is answered on a live reading, not the
/// cached tier, because this is the one moment somebody is asking to be let
/// in. If the chain cannot be read the request is left pending rather than
/// declined: a holder refused because an RPC was down has been told something
/// false about their wallet.
async function onJoinRequest(r: TgJoinRequest) {
  const bar = barOf(r.chat.id, TIERS);
  if (bar === null) return;
  const user = r.from.id;
  const { data: mem } = await db.from("holder_members").select("tg_user").eq("tg_user", user).maybeSingle();
  if (!mem) {
    await tg().decline(r.chat.id, user).catch(() => {});
    await tell(r.user_chat_id ?? user, "Link a wallet first: send me /verify.");
    return;
  }
  let m;
  try {
    m = await measure(user);
  } catch {
    await tell(r.user_chat_id ?? user, "Could not read the chain to check your request; it is still pending. Send /status in a minute to retry.");
    return;
  }
  await record(user, m);
  if (m.tier >= bar) {
    await tg().approve(r.chat.id, user).catch((e) => console.error("approve:", e.message));
  } else {
    await tg().decline(r.chat.id, user).catch(() => {});
    await tell(user, `That chat needs ${tierLabel(bar)}. You hold ${summary(m)}.`);
  }
}

/// Somebody arrived without going through a join request -- added by a member,
/// or through an old ordinary invite link. Let them stay if their cached tier
/// meets the bar or an administrator added them; remove them otherwise.
async function onChatMember(cm: TgChatMember) {
  const bar = barOf(cm.chat.id, TIERS);
  if (bar === null) return;
  const was = cm.old_chat_member?.status, now = cm.new_chat_member?.status;
  if (!(now === "member" || now === "restricted") || (was && IN_CHAT.has(was))) return;
  const user = cm.new_chat_member.user;
  if (user.is_bot) return;
  const { data: mem } = await db.from("holder_members").select("tier_bps").eq("tg_user", user.id).maybeSingle();
  if ((mem?.tier_bps ?? 0) >= bar) return;
  if (cm.from && cm.from.id !== user.id) {
    const by = await statusIn(String(cm.chat.id), cm.from.id);
    if (by === "administrator" || by === "creator") return;
  }
  await tg().kick(cm.chat.id, user.id).catch((e) => console.error("kick:", e.message));
  await tell(user.id, `That chat is for holders of ${tierLabel(bar)} of $GLADOS. Send /verify to link a wallet.`);
}

// ---- the page ---------------------------------------------------------------

async function sessionOf(token: unknown) {
  if (typeof token !== "string" || !/^[A-Za-z0-9_-]{43}$/.test(token))
    return { error: "that link is not a session link -- send /verify to the bot for a new one" };
  const hash = await sha256Hex(token);
  const { data } = await db.from("holder_sessions")
    .select("token_hash, tg_user, tg_name, expires_at, used_at").eq("token_hash", hash).maybeSingle();
  if (!data) return { error: "no such session -- send /verify to the bot for a new link" };
  if (data.used_at) return { error: "that link has been used -- send /verify again to link another wallet" };
  if (new Date(data.expires_at) < new Date()) return { error: "that link has expired -- send /verify again" };
  return { session: data };
}

async function page(step: string, body: Record<string, unknown>) {
  const address = typeof body.address === "string" ? body.address.toLowerCase() : "";

  if (step === "check") {
    if (!isAddress(address)) return refuse(400, "address must be 0x and 40 hex digits");
    try {
      const r = await reading();
      const b = await balanceOf(RPC, TOKEN, address, r.block);
      return json(200, {
        address: toChecksum(address), balance: b.toString(), supply: r.supply.toString(),
        percent: percent(b, r.supply), tier: tierOf(b, r.supply, TIERS), block: BigInt(r.block).toString(),
        tiers: TIERS.map((t) => t.bps),
      });
    } catch {
      return refuse(503, "could not read the chain; try again shortly");
    }
  }

  const s = await sessionOf(body.session);
  if (!s.session) return refuse(400, s.error!);
  const sess = s.session;

  if (step === "session")
    return json(200, {
      tg_user: sess.tg_user, tg_name: sess.tg_name, expires_at: sess.expires_at,
      tiers: TIERS.map((t) => t.bps), chain_id: CHAIN_ID, token: toChecksum(TOKEN),
    });

  if (!isAddress(address)) return refuse(400, "address must be 0x and 40 hex digits");

  if (step === "nonce") {
    const nonce = crypto.randomUUID().replace(/-/g, "");
    const issued = new Date();
    const expires = new Date(issued.getTime() + NONCE_TTL_MS);
    const { error } = await db.from("holder_nonces").insert({
      nonce, session_hash: sess.token_hash, address,
      issued_at: issued.toISOString(), expires_at: expires.toISOString(),
    });
    if (error) return refuse(500, "could not issue a nonce");
    const message = holderMessage({
      domain: DOMAIN, address: toChecksum(address), tgUser: sess.tg_user, chainId: CHAIN_ID, nonce, issued, expires,
    });
    // `issued_at` travels because the signed text contains it; `link` records why.
    return json(200, { nonce, message, issued_at: issued.toISOString(), expires_at: expires.toISOString() });
  }

  if (step !== "verify") return refuse(404, "use /session, /check, /nonce or /verify");

  // Hex of any even length: an EOA signature is 65 bytes, a contract wallet's
  // is whatever its account decides, and an ERC-6492 wrapper is longer still.
  const signature = typeof body.signature === "string" ? body.signature.trim() : "";
  if (!/^0x([0-9a-fA-F]{2}){65,8192}$/.test(signature))
    return refuse(400, "signature must be 0x and hex, at least 65 bytes");

  const { data: row } = await db.from("holder_nonces")
    .select("nonce, issued_at, expires_at").eq("session_hash", sess.token_hash).eq("address", address)
    .is("used_at", null).order("issued_at", { ascending: false }).limit(1).maybeSingle();
  if (!row) return refuse(400, "no unused nonce for that address in this session -- ask for one first");
  if (new Date(row.expires_at) < new Date()) return refuse(400, "that nonce has expired");

  const message = holderMessage({
    domain: DOMAIN, address: toChecksum(address), tgUser: sess.tg_user, chainId: CHAIN_ID,
    nonce: row.nonce, issued: new Date(row.issued_at), expires: new Date(row.expires_at),
  });
  let kind: string | null;
  try {
    kind = await verifySigner(RPC, message, address, signature);
  } catch {
    return refuse(503, "could not reach the chain to check a contract-wallet signature; try again shortly");
  }
  if (!kind) return refuse(401, "that signature does not belong to that address");

  // Spend the nonce only if still unspent: two requests racing one nonce, the
  // second update matches no row.
  const { data: spent } = await db.from("holder_nonces").update({ used_at: new Date().toISOString() })
    .eq("nonce", row.nonce).is("used_at", null).select("nonce").maybeSingle();
  if (!spent) return refuse(409, "that nonce was already used");

  const { data: owner } = await db.from("holder_wallets").select("tg_user").eq("address", address).maybeSingle();
  if (owner && Number(owner.tg_user) !== Number(sess.tg_user))
    return refuse(409, "that wallet is linked to another Telegram account -- unlink it there first (/unlink)");
  if (!owner) {
    const { count } = await db.from("holder_wallets")
      .select("address", { count: "exact", head: true }).eq("tg_user", sess.tg_user);
    if ((count ?? 0) >= MAX_WALLETS)
      return refuse(409, `this Telegram account already has ${MAX_WALLETS} wallets -- /unlink one first`);
  }

  // Read before writing anything, so a chain that cannot be read leaves no
  // half-linked state behind.
  let m;
  try {
    m = await measure(sess.tg_user, undefined, address);
  } catch {
    return refuse(503, "could not read the balance from the chain; try again shortly");
  }

  await db.from("holder_members").upsert({ tg_user: sess.tg_user, tg_name: sess.tg_name });
  await db.from("holder_wallets").upsert({ address, tg_user: sess.tg_user, kind });
  await record(sess.tg_user, m);
  await db.from("holder_sessions").update({ used_at: new Date().toISOString() }).eq("token_hash", sess.token_hash);

  const { links } = await reconcile(sess.tg_user, m.tier, true);
  const told = await tell(sess.tg_user,
    `Linked <code>${toChecksum(address)}</code>.\n${summary(m)}` +
      (m.tier ? (links.length ? "\n\nYour join links, valid for an hour:" : "\n\nYou are already in every chat you qualify for.")
              : `\n\nThat is below the ${tierLabel(TIERS[0].bps)} bar, so no chat is open to you yet.`),
    links);

  return json(200, {
    address: toChecksum(address), kind, balance: m.each[address].toString(), total: m.total.toString(),
    supply: m.supply.toString(), percent: percent(m.total, m.supply), tier: m.tier,
    tiers: TIERS.map((t) => t.bps), wallets: Object.keys(m.each).length, invites: links.length, told,
    block: BigInt(m.block).toString(),
  });
}

// ---- the schedule -------------------------------------------------------------

async function recheck() {
  const r = await reading();
  let checked = 0, removed = 0, failed = 0;
  for (let from = 0; ; from += 500) {
    const { data: mems } = await db.from("holder_members").select("tg_user, tier_bps")
      .order("tg_user").range(from, from + 499);
    if (!mems?.length) break;
    for (const mem of mems) {
      try {
        const m = await measure(Number(mem.tg_user), r);
        await record(Number(mem.tg_user), m);
        const rose = m.tier > mem.tier_bps;
        const out = await reconcile(Number(mem.tg_user), m.tier, rose);
        removed += out.removed.length;
        if (out.removed.length)
          await tell(Number(mem.tg_user), `Your holding is now ${summary(m)}, so you were removed from: ${out.removed.join(", ")}.`);
        else if (rose && out.links.length)
          await tell(Number(mem.tg_user), `Your holding is now ${summary(m)}. New chats are open to you:`, out.links);
        checked++;
      } catch (e) {
        // One member's failed reading is not a reason to stop removing the
        // rest, and is never recorded as a holding of zero.
        failed++;
        console.error(`recheck ${mem.tg_user}:`, (e as Error).message);
      }
    }
    if (mems.length < 500) break;
  }
  // Spent sessions and nonces are evidence of nothing after a day.
  const day = new Date(Date.now() - 86400_000).toISOString();
  await db.from("holder_sessions").delete().lt("expires_at", day);
  return { block: BigInt(r.block).toString(), supply: r.supply.toString(), checked, removed, failed };
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: cors });
  if (req.method !== "POST") return refuse(405, "POST only");
  const step = new URL(req.url).pathname.split("/").filter(Boolean).pop() ?? "";

  if (step === "telegram") {
    if (!same(WEBHOOK_SECRET, req.headers.get("x-telegram-bot-api-secret-token") ?? ""))
      return new Response("forbidden", { status: 403 });
    // Always 200 once authenticated: Telegram redelivers anything else, and a
    // handler that failed once will fail the same way on every redelivery.
    try {
      const u = await req.json();
      if (u.message) await onMessage(u.message);
      else if (u.chat_join_request) await onJoinRequest(u.chat_join_request);
      else if (u.chat_member) await onChatMember(u.chat_member);
    } catch (e) {
      console.error("update:", (e as Error).message);
    }
    return new Response("ok");
  }

  if (step === "recheck") {
    if (!same(RECHECK_SECRET, req.headers.get("x-recheck-secret") ?? "")) return refuse(403, "forbidden");
    try {
      return json(200, await recheck());
    } catch {
      return refuse(503, "could not read the chain; nothing was changed");
    }
  }

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return refuse(400, "body must be JSON");
  }
  return page(step, body);
});
