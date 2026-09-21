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

## Actions may open a pull request

Settings -> Actions -> General -> **Allow GitHub Actions to create and
approve pull requests**.

Off by default, and the adopt job's last step fails on it with

    GraphQL: GitHub Actions is not permitted to create or approve pull
    requests (createPullRequest)

after everything else in the night has already succeeded -- the certificate
filed, `loop/main` pushed, the candidate ref tidied. So the symptom is a red
square on a night that worked, which is the shape that gets ignored.

**It does not widen what the loop may do**, which is why it is safe to grant.
The rolling audit PR is the machine's *only* path to `main`, and a PR is a
request: the token still has no `workflows` write, `main` is still protected,
and merging is still a human decision. Withholding this does not contain the
loop, it silences it -- the work keeps happening on `loop/main` and stops
being offered to anybody.

The "approve" half of the setting's name is GitHub's, not a grant to this
loop: nothing here approves anything, and `boundary.yml`'s write job still
sits behind the `evaluator` environment and a required reviewer.

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
