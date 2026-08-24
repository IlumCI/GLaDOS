#!/usr/bin/env python3
"""Phase 3 spike harness. Runs the query plan against the live Shovels API with
a hard call budget and a per-call transcript, so the run is reproducible and
self-reports how much of the 250-call trial key it spent.

Usage:  SHOVELS_KEY=... python3 spike.py [ceiling]
The key comes from the environment; it is never written to disk or committed.
"""
import os, sys, json, urllib.parse, urllib.request

BASE = "https://api.shovels.ai/v2"
KEY = os.environ.get("SHOVELS_KEY", "")
CEIL = int(sys.argv[1]) if len(sys.argv) > 1 else 120   # stay well under 250
TRANSCRIPT = os.path.join(os.path.dirname(__file__), "spike-transcript.jsonl")

calls = 0
log = open(TRANSCRIPT, "a")


def get(path, **params):
    """One API call. Counts against budget whether it succeeds or errors, exactly
    as the trial key does. Hard-stops at the ceiling."""
    global calls
    if calls >= CEIL:
        raise SystemExit(f"budget ceiling {CEIL} reached; stopping before spending more")
    calls += 1
    q = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None}, doseq=True)
    url = f"{BASE}{path}" + (f"?{q}" if q else "")
    req = urllib.request.Request(url, headers={"X-API-Key": KEY, "Accept": "application/json"})
    rec = {"n": calls, "path": path, "params": params}
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            body = json.load(r)
            rec["status"] = r.status
    except urllib.error.HTTPError as e:
        body = {"error": e.read().decode("utf-8", "replace")[:300]}
        rec["status"] = e.code
    except Exception as e:
        body = {"error": str(e)[:200]}
        rec["status"] = "exc"
    rec["ok"] = str(rec["status"]).startswith("2")
    log.write(json.dumps(rec) + "\n"); log.flush()
    print(f"[{calls:3}] {rec['status']} {path} {q[:70]}")
    return body


def main():
    if not KEY:
        print("No SHOVELS_KEY in environment. Create a trial key at "
              "https://app.shovels.ai and export SHOVELS_KEY before running.")
        return
    # Tier 0, orient
    rel = get("/meta/release")
    print("  release:", json.dumps(rel)[:120])
    tags = get("/list/tags", size=200)
    names = [t.get("name", t) if isinstance(t, dict) else t
             for t in (tags.get("items", tags) if isinstance(tags, dict) else tags)]
    print("  tags:", names[:40])
    # The rest of the plan (Tier 1-3) is filled in live once we see which tags
    # and states are actually covered, so we do not burn calls on empty filters.
    # Deliberately stops here on the first run: orient, then decide with Arron.
    print(f"\nSpent {calls} calls. Transcript at {TRANSCRIPT}")


if __name__ == "__main__":
    main()
