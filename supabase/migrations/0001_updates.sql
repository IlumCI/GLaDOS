-- The update service's tables.
--
-- Row-level security is enabled on every one of them with no policy attached,
-- which denies all access through the anon and authenticated keys. Everything
-- reaches these tables through an Edge Function holding the service role, so
-- there is one door and it is a function whose code is in this repo. A policy
-- would be a second door that has to be reasoned about separately.
--
-- The buckets these describe:
--   stable        public.  One object, `manifest`, plus the images it names.
--   experimental  private. Reached only through a signed URL the channel
--                 function mints after checking entitlement.

-- Every experimental build the pipeline has published.
create table if not exists builds (
  id           bigint generated always as identity primary key,
  channel      text        not null default 'experimental',
  slug         text        not null unique,
  version      text        not null,
  sha256       text        not null,
  size         bigint      not null,
  -- Path within the bucket, so the function can sign a URL for it without
  -- reconstructing a layout the pipeline chose.
  object_path  text        not null,
  notes        text        not null default '',
  created_at   timestamptz not null default now()
);

create index if not exists builds_channel_created
  on builds (channel, created_at desc);

-- A wallet somebody proved they hold, once the link function exists.
--
-- Nothing writes this yet. Custody is unsettled: if the token is held inside a
-- custodial app rather than a self-custody wallet, holders have no key to sign
-- with and this table stays empty while `allowlist` does the work. That is why
-- entitlement is one function and not a join.
create table if not exists wallets (
  address     text primary key,
  chain       text        not null,
  linked_at   timestamptz not null default now()
);

-- A device code a machine sends to reach the experimental channel.
--
-- The code is stored HASHED and never in plaintext. It is a bearer token: a
-- database leak that included them would be a leak of working credentials, and
-- there is no reason for the server to be able to read one back -- it only
-- ever needs to recognise one.
create table if not exists devices (
  code_hash   text primary key,
  wallet      text references wallets (address) on delete cascade,
  created_at  timestamptz not null default now(),
  last_seen   timestamptz,
  revoked     boolean     not null default false
);

-- Entitlement, before there is a chain to read.
--
-- This is the first implementation of `entitled()` and is meant to stay
-- afterwards: a way to hand somebody access without a wallet, for a tester, a
-- contributor, or anyone holding custodially. Rows are added by hand.
create table if not exists allowlist (
  code_hash   text primary key,
  note        text        not null default '',
  expires_at  timestamptz,
  created_at  timestamptz not null default now()
);

alter table builds    enable row level security;
alter table wallets   enable row level security;
alter table devices   enable row level security;
alter table allowlist enable row level security;
