# The protections the loop assumes, applied by the operator

Everything in code refuses; these are the walls GitHub itself holds up.
Apply after the first push, before the first cron is uncommented. Settings
-> Rules -> Rulesets (or `gh api /repos/:owner/:repo/rulesets`).

## Ruleset `main` (branch: main)

- Require a pull request before merging.
- Required status checks: `ci / check`, `ci / build-boot`.
- Block force pushes. Restrict deletions.

The loop's token never pushes main anyway -- the audit PR is its only path
-- but a wall that exists only in a token's habits is a comment.

## Ruleset `loop` (branch: loop/**)

- Block force pushes. Restrict deletions.

Fast-forward-only is constructed by the `loop-main` concurrency group plus
adopt's plumbing; this ruleset is the backstop against a human hand doing
history surgery on the ledger by accident.

## Ruleset `boundary` (branch: boundary/**)

- Block force pushes. Restrict deletions.

## Push protection for the evaluator (if the plan supports file paths)

- Restrict file paths: `.github/workflows/**` for all non-admin actors.

Belt to the braces: the loop's `GITHUB_TOKEN` already has no `workflows`
write, so GitHub rejects such pushes at the token layer; this covers every
other non-admin credential too.

## Environment `evaluator`

- Required reviewer: the operator.
- Secret `EVALUATOR_TOKEN`: a fine-grained PAT, this repository only,
  permissions `contents: write` + `workflows: write` + `pull requests:
  write`. It exists nowhere else, and boundary.yml's write job cannot start
  -- the secret does not resolve -- until a human approves the run.

## Secrets and variables recap (the loop's whole surface)

| where | name | held by |
|---|---|---|
| repo secret | UPDATE_SIGNING_KEY | release.yml, experimental.yml only |
| repo secret | VERDICT_SIGNING_KEY | propose.yml only |
| repo secret | VERDICT_INGEST_TOKEN | propose.yml (POST); mirrors a Supabase function secret |
| environment `evaluator` | EVALUATOR_TOKEN | boundary.yml's write job only |
| repo variable | SUPABASE_URL | publish + verdict POSTs |

The loop's own jobs (loop-night, loop-judge) hold **no repository
secrets**: `contents: write` on its own branches and `models: read` for the
author, both from the ephemeral `GITHUB_TOKEN`.
