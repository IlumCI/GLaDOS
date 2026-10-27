// The Bot API, as the few calls the holder chats need.
//
// Plain fetch and no library: the surface is eight methods, every one is a
// POST with a JSON body, and a dependency for that would be the largest thing
// in the function.
//
// `ok: false` throws with Telegram's own description, because the difference
// between "bot is not an admin there" and "user has blocked the bot" is the
// whole of what an operator needs when a member was not let in.

export function bot(token) {
  if (!token) throw new Error("TG_BOT_TOKEN is not set");
  const base = `https://api.telegram.org/bot${token}/`;
  async function call(method, body) {
    const r = await fetch(base + method, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body ?? {}),
    });
    const j = await r.json().catch(() => null);
    if (!j || !j.ok) throw new Error(`${method}: ${(j && j.description) || "HTTP " + r.status}`);
    return j.result;
  }
  return {
    call,
    send: (chat_id, text, extra = {}) =>
      call("sendMessage", { chat_id, text, parse_mode: "HTML", disable_web_page_preview: true, ...extra }),
    // A link that creates a join *request* rather than admitting on click, so
    // whoever ends up holding it is checked again when they use it. A plain
    // invite link forwarded to a friend admits the friend.
    joinLink: (chat_id, name, ttlSeconds) =>
      call("createChatInviteLink", {
        chat_id, name: name.slice(0, 32), creates_join_request: true,
        expire_date: Math.floor(Date.now() / 1000) + ttlSeconds,
      }),
    approve: (chat_id, user_id) => call("approveChatJoinRequest", { chat_id, user_id }),
    decline: (chat_id, user_id) => call("declineChatJoinRequest", { chat_id, user_id }),
    member: (chat_id, user_id) => call("getChatMember", { chat_id, user_id }),
    // Telegram has no "remove"; a kick is a ban lifted at once, which leaves the
    // user free to rejoin through a join request once they qualify again.
    kick: async (chat_id, user_id) => {
      await call("banChatMember", { chat_id, user_id, revoke_messages: false });
      await call("unbanChatMember", { chat_id, user_id, only_if_banned: true });
    },
  };
}

/// Text made safe for `parse_mode: HTML`.
export const esc = (s) =>
  String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
