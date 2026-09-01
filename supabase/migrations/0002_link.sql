-- Wallet linking: the half `0001` left for when custody was settled.
--
-- It is settled. The token is an ERC-20 on Robinhood Chain, so a holder has a
-- key and an address of their own and can sign. `wallets` stops being an empty
-- table.

-- A nonce issued to one address, good once.
--
-- Server-issued and single-use, which is the whole security of the flow. A
-- client-chosen nonce lets somebody replay a signature they captured
-- elsewhere; a reusable one lets them replay it here. `used_at` is set inside
-- the same statement that claims it, so two requests racing the same nonce
-- cannot both win.
create table if not exists nonces (
  nonce       text primary key,
  address     text        not null,
  issued_at   timestamptz not null default now(),
  expires_at  timestamptz not null,
  used_at     timestamptz
);

create index if not exists nonces_expiry on nonces (expires_at);

-- Expired and spent nonces are not evidence of anything. Kept for an hour so a
-- confused client gets "expired" rather than "no such nonce", which are
-- different problems and should read differently.
create index if not exists nonces_address on nonces (address);

alter table nonces enable row level security;

-- How many devices one wallet may hold at once.
--
-- Without a cap a single holder mints codes forever and hands them out, which
-- turns a per-holder gate into a public one. Five is arbitrary and generous;
-- what matters is that a number exists.
alter table wallets add column if not exists device_cap integer not null default 5;

-- The balance that was read when the wallet linked, and when.
--
-- Recorded for support rather than for entitlement. Entitlement re-reads the
-- chain at every download, because a holding is not a permanent fact about a
-- person and a cached one is a gate that stays open after the holding is gone.
alter table wallets add column if not exists linked_balance numeric;
alter table wallets add column if not exists last_checked timestamptz;
