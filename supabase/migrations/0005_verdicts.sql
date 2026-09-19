-- The verdict store: the return leg the proposal function's own 202 has
-- been promising ("the verdict comes back signed, through the channel")
-- since the day it shipped, without anything behind it.
--
-- A row is one signed verdict blob, stored ONLY after the `verdict` edge
-- function has checked its GLADOSIG against the pinned verdict public
-- point -- so a leaked ingest token can insert nothing the verdict key did
-- not sign. `body` is the whole bin, base64: text plus exactly 80 bytes of
-- signature, one object, because the kernel verifies before it parses and
-- splitting them here would be splitting what travels as one.
--
-- `point` is indexed and NOT unique, like proposals.point: a re-judged
-- point produces a second verdict and both are the record.
--
-- RLS is enabled with no policies, like every table here: the only door is
-- an edge function holding the service role.

create table if not exists verdicts (
  id           bigint generated always as identity primary key,
  proposal_id  bigint references proposals(id),
  point        text not null,
  -- Denormalized from the proposal row at ingest: which device may fetch
  -- this. Empty means nobody -- a loop-originated or drill verdict, kept
  -- for audit and served to no device.
  code_hash    text not null default '',
  body         text not null,
  run_id       text not null,
  run_attempt  text not null,
  -- The served-once cursor. A device GET sets it; `?again=1` ignores it,
  -- because a verdict fetched and then lost to a crashed boot must be
  -- re-fetchable without an operator in the path.
  claimed_at   timestamptz,
  created_at   timestamptz not null default now()
);

create index if not exists verdicts_point on verdicts (point);
create index if not exists verdicts_device
  on verdicts (code_hash, claimed_at, created_at);

alter table verdicts enable row level security;

-- proposals.status gains 'answered' as a value; status is already free
-- text, so this is documentation rather than DDL: pending -> dispatched ->
-- answered (a verdict arrived and was linked), alongside refused and
-- unconfigured.
