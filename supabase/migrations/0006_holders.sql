-- Holder chats: a Telegram account admitted on a share of $GLADOS supply.
--
-- Separate from `wallets`/`devices` deliberately. Those answer "may this
-- machine reach the experimental channel" at a 1,000,000-token bar; these
-- answer "may this person be in a chat" at a share of supply. One wallet can be
-- both, and neither table's rows mean anything to the other's question.
--
-- RLS on and no policy on any of them, as everywhere in this schema: the only
-- door is the `holders` function holding the service role.

-- A session the bot handed to one Telegram user, good once.
--
-- The token travels in the link the bot sends and only its hash is kept, for
-- `devices`'s reason: a bearer token the server can read back is a credential a
-- database leak hands over. Short-lived, because a link in a chat history is a
-- link anybody scrolling that history can open.
create table if not exists holder_sessions (
  token_hash  text primary key,
  tg_user     bigint      not null,
  tg_name     text,
  issued_at   timestamptz not null default now(),
  expires_at  timestamptz not null,
  used_at     timestamptz
);
create index if not exists holder_sessions_user on holder_sessions (tg_user);
alter table holder_sessions enable row level security;

-- A nonce for one address inside one session. Bound to the session so a
-- signature collected under one Telegram user's link cannot be submitted under
-- another's -- the signed text names the user as well, so this is the second
-- of two checks rather than the only one.
create table if not exists holder_nonces (
  nonce         text primary key,
  session_hash  text        not null references holder_sessions (token_hash) on delete cascade,
  address       text        not null,
  issued_at     timestamptz not null,
  expires_at    timestamptz not null,
  used_at       timestamptz
);
create index if not exists holder_nonces_session on holder_nonces (session_hash, address);
alter table holder_nonces enable row level security;

-- A Telegram user and the tier their wallets reached at the last reading.
--
-- `tier_bps` is a cache for the join-request handler, which must answer inside
-- Telegram's webhook window and cannot afford an RPC round trip per request.
-- It is rewritten at every link and every recheck, and the recheck is what
-- removes somebody whose holding fell -- so a stale tier is at most one recheck
-- interval stale, and that interval is the operator's choice.
create table if not exists holder_members (
  tg_user       bigint primary key,
  tg_name       text,
  tier_bps      integer     not null default 0,
  total_balance numeric,
  supply        numeric,
  last_checked  timestamptz
);
alter table holder_members enable row level security;

-- Which wallets belong to which Telegram user. A wallet belongs to one user at
-- most -- the primary key is the address -- or one holding would admit as many
-- accounts as its owner cared to link. A user may hold several wallets and is
-- measured on their sum, because splitting a holding across wallets is
-- ordinary and is not a reason to be refused.
create table if not exists holder_wallets (
  address       text primary key,
  tg_user       bigint      not null references holder_members (tg_user) on delete cascade,
  kind          text        not null default 'eoa',
  linked_at     timestamptz not null default now(),
  balance       numeric,
  last_checked  timestamptz
);
create index if not exists holder_wallets_user on holder_wallets (tg_user);
alter table holder_wallets enable row level security;
