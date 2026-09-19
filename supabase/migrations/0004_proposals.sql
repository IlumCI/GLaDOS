-- What a machine asked to have judged about itself.
--
-- `godel push` sends an envelope to the `proposal` function; the function
-- checks it, records it here, and dispatches `propose.yml`. This table is what
-- makes the outward leg accountable: `workflow_dispatch` answers 204 with an
-- empty body and does not say which run it started, so without a row on this
-- side there is no way to ask afterwards what was asked for and whether
-- anything came of it.
--
-- ### Why the whole envelope is stored beside its parsed fields
--
-- The columns are for looking things up; the text is for re-deriving. A
-- verdict names a `point`, and `godel verdict` on the machine files it against
-- a lineage -- so the question "which patch was that point" has to have an
-- answer years later, from bytes rather than from a reconstruction. The parsed
-- columns could be rebuilt from `envelope` and the reverse is not true, which
-- is the direction that decides what is authoritative.
--
-- ### What a compromise of this table buys an attacker
--
-- A record of which constants a machine wanted changed, and the ability to
-- lose or forge that record. It cannot cause a build: the dispatch goes out
-- with a credential the table does not hold, and the workflow re-reads the
-- envelope it was handed rather than anything here. It cannot cause an
-- adoption either -- a verdict is signed by the update key, and the kernel
-- refuses an unsigned one. Same division `0003_workers.sql` draws.

create table if not exists proposals (
  id          bigint generated always as identity primary key,

  -- The device that asked, by the same hash `channel` and `link` key on. Never
  -- the code itself: a code in a table is a code somebody else has.
  code_hash   text        not null,

  -- The proposal's content address, as the kernel renders it. This is the
  -- field a returning verdict is matched on, so it is indexed and it is not
  -- unique: the same point may legitimately be proposed again against a
  -- different corpus, and refusing that would make a re-measurement after the
  -- evidence changed impossible.
  point       text        not null,

  version     text        not null,
  head        text        not null,
  corpus      text        not null,

  -- How many tests the corpus in force had already paid for when this was
  -- sent. The family-wise budget's counter, carried so a reader can tell a
  -- proposal made on fresh evidence from one made on a corpus that was nearly
  -- spent.
  tests       integer     not null default 0,

  file        text        not null,
  symbol      text        not null,
  was         text        not null,
  -- `now` is reserved in SQL, and a column that has to be quoted everywhere is
  -- a column somebody will eventually fail to quote.
  now_value   text        not null,
  rail        text        not null,

  -- The canonical envelope, as the function rendered it and as the workflow
  -- received it. Authoritative; the columns above are derived from it.
  envelope    text        not null,

  -- pending -> dispatched, or -> refused / unconfigured. Written before the
  -- dispatch and updated after, so a crash in between leaves a row that shows
  -- as pending rather than a run nobody can account for.
  status      text        not null default 'pending',
  detail      text,

  created_at  timestamptz not null default now()
);

create index if not exists proposals_point on proposals (point);
-- The rate limit reads this every request: one device, one day.
create index if not exists proposals_device_day on proposals (code_hash, created_at desc);

alter table proposals enable row level security;
