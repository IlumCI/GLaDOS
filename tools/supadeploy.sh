#!/usr/bin/env bash
# Deploy the Supabase half, in the one order that works.
#
# **The order is the whole reason this is a script.** `supabase/README.md` lists
# the commands and a reader runs them in the order they appear; two of the orderings
# that look fine are wrong:
#
#   - migrations before functions, because `verdict` inserts into a table
#     `0005_verdicts.sql` creates and a function deployed first answers 500 on
#     every call until the migration lands;
#   - and the image before the manifest, which is `release.yml`'s rule rather than
#     this script's, recorded here because somebody reading a deploy script looks
#     for the whole order in it.
#
# **The project reference comes out of the kernel.** `src/update/channel.rs` pins
# the origin every shipped image asks, so that constant is the authority and this
# script derives from it rather than carrying a second copy. Deploying to a project
# the kernel does not name is the failure `tools/origin.py` exists to catch, and
# doing it from a script that could not make the mistake is better than catching it.
set -euo pipefail

cd "$(dirname "$0")/.."

REF=$(python3 - <<'PY'
import re, sys
s = open("src/update/channel.rs", encoding="utf-8").read()
m = re.search(r'pub\s+const\s+DEFAULT_SOURCE\s*:\s*&\s*str\s*=\s*"([^"]*)"\s*;', s)
if not m:
    sys.exit("no DEFAULT_SOURCE in src/update/channel.rs")
host = m.group(1).split("://", 1)[-1].split("/", 1)[0]
if not host.endswith(".supabase.co"):
    sys.exit(f"{host} is not a Supabase project host; nothing to deploy to")
print(host.split(".", 1)[0])
PY
)

echo "kernel pins project:  $REF"

# Refuse before spending anything if the project is not there. A deploy against a
# missing project fails somewhere in the middle, and half a deploy is worse than
# none: the migrations may have landed and the functions not.
if ! python3 tools/origin.py --resolve >/dev/null 2>&1; then
    echo "  that project does not resolve -- create it, re-pin channel.rs, rebuild"
    echo "  tools/origin.py --resolve says so in one line"
    exit 1
fi

command -v supabase >/dev/null || {
    echo "  the supabase CLI is not installed"
    echo "  npm i -g supabase   (or see supabase.com/docs/guides/cli)"
    exit 1
}

echo "==> link"
supabase link --project-ref "$REF"

echo "==> migrations, before the functions that read the tables they make"
supabase db push

# Settings come from supabase/config.toml, so no --no-verify-jwt here: a flag
# typed per deploy is one somebody forgets, and forgetting it deploys a function
# every caller gets a 401 from.
for f in channel link worker proposal verdict; do
    echo "==> function $f"
    supabase functions deploy "$f"
done

echo
echo "deployed. What is still owed, and none of it is in this script:"
echo "  - the secrets: VERDICT_INGEST_TOKEN, and whatever channel/link need"
echo "  - the GitHub variable SUPABASE_URL, which must equal what the kernel pins"
echo "    (python3 tools/origin.py --expect \"\$SUPABASE_URL\")"
echo "  - a rebuilt kernel, if the ref changed: the old one asks the old host"
