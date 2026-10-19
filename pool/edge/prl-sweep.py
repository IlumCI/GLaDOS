#!/usr/bin/env python3
"""Sweep the pool's PRL into PearlBridge, so it arrives as WPRL at the treasury.

The semi-automatic half of the Pearl payout path. 2Miners pays the pool's PRL
to the operator's own wallet (oyster, on this machine). Run hourly, this sends
whatever has gathered there -- once it is worth the bridge's fee -- to the
PearlBridge deposit address tied to the treasury's EVM key. From there the
treasury does the rest by itself (worker/prl-leg.js): WPRL to USDT on Uniswap,
USDT to ETH on 4663, ETH to the miners.

    prl-sweep.py            check, and say what it would send (the default)
    prl-sweep.py --live     check, and send if there is enough
    prl-sweep.py --status   what earlier sweeps became at the bridge

**Dry unless --live.** Nothing leaves the wallet without that flag.

Refuses rather than guesses, every time:
  - the wallet must have caught up with the network (2Miners' height)
  - the bridge's deposit address must equal the one pinned in the config: an
    API that starts answering another address is an attack or a change, and
    either way a person decides
  - the bridge must not be paused, must say mainnet, and the amount must be
    at least ten times its minimum fee
  - one sweep at a time: a new one waits until the last is seen at the bridge

Config: ~/.pearl-treasury/sweep.json (mode 600), written by prl-sweep-install.sh.
The wallet's passphrase is read from ~/.pearl-treasury/SECRETS.txt only for a
live send, and is never printed or logged.
"""
import base64, json, os, ssl, sys, time, urllib.request

HOME = os.path.expanduser("~/.pearl-treasury")
CONF = os.path.join(HOME, "sweep.json")
LOG = os.path.join(HOME, "sweeps.jsonl")
BRIDGE = "https://pearlbridge.xyz"
BRIDGE_API = "https://api.pearlbridge.xyz"
POOL_STATS = "https://prl.2miners.com/api/stats"
GRAIN = 10**8


def say(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def get_json(url, timeout=20):
    req = urllib.request.Request(url, headers={"user-agent": "glados-prl-sweep/1", "accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class Wallet:
    def __init__(self, conf):
        self.url = f"https://127.0.0.1:{conf['rpc_port']}/"
        self.auth = "Basic " + base64.b64encode(f"{conf['rpc_user']}:{conf['rpc_pass']}".encode()).decode()
        self.ctx = ssl.create_default_context(cafile=os.path.join(HOME, "rpc.cert"))

    def call(self, method, *params):
        body = json.dumps({"jsonrpc": "1.0", "id": 1, "method": method, "params": list(params)}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"content-type": "application/json", "authorization": self.auth})
        with urllib.request.urlopen(req, timeout=60, context=self.ctx) as r:
            out = json.loads(r.read().decode())
        if out.get("error"):
            raise RuntimeError(f"{method}: {out['error'].get('message', out['error'])}")
        return out["result"]


def passphrase():
    path = os.path.join(HOME, "SECRETS.txt")
    if os.stat(path).st_mode & 0o077:
        raise SystemExit(f"{path} is readable by others; fix its mode (chmod 600) before anything is sent")
    with open(path) as f:
        for line in f:
            if line.startswith("private passphrase: "):
                return line.split(": ", 1)[1].strip()
    raise SystemExit("no passphrase line in SECRETS.txt")


def sweeps():
    if not os.path.exists(LOG):
        return []
    with open(LOG) as f:
        return [json.loads(l) for l in f if l.strip()]


def mint_status(txid):
    try:
        return get_json(f"{BRIDGE}/api/mint-status?txid={txid}")
    except Exception as e:  # unanswered is not an answer
        return {"state": None, "error": str(e)}


def main():
    live = "--live" in sys.argv
    with open(CONF) as f:
        conf = json.load(f)
    if os.stat(CONF).st_mode & 0o077:
        raise SystemExit(f"{CONF} is readable by others (it holds the RPC password); chmod 600 it")

    if "--status" in sys.argv:
        for s in sweeps():
            st = mint_status(s["txid"])
            say(f"{s['at']}  {s['prl']} PRL  {s['txid']}  bridge: {st.get('state')}{' mint ' + st['mintTxHash'] if st.get('mintTxHash') else ''}")
        return

    # One at a time: the last sweep must be seen by the bridge before another goes.
    past = sweeps()
    if past:
        last = past[-1]
        st = mint_status(last["txid"])
        if not st.get("state"):
            age = time.time() - last["ts"]
            if age < 6 * 3600:
                return say(f"the last sweep ({last['txid']}, {last['prl']} PRL) is not at the bridge yet; waiting ({int(age / 60)} min)")
            return say(f"WARNING: the last sweep ({last['txid']}) is still unknown to the bridge after {int(age / 3600)} h; "
                       f"nothing more is sent until it is checked by hand: {BRIDGE}/status/{last['txid']}")

    w = Wallet(conf)
    try:
        best = w.call("getbestblock")
    except Exception as e:
        return say(f"the wallet is not answering ({e}); is glados-prl-wallet.service running?")
    try:
        net = int(get_json(POOL_STATS)["nodes"][0]["height"])
    except Exception as e:
        return say(f"cannot read the network height ({e}); not sending blind")
    if best["height"] < net - 3:
        return say(f"the wallet is at block {best['height']} of {net}; waiting for it to catch up")

    bal = float(w.call("getbalance", "*", 6))  # six confirmations, as the bridge wants
    grains = int(round(bal * GRAIN))
    threshold = int(conf.get("threshold_prl", 50) * GRAIN)
    if grains < threshold:
        return say(f"{bal:.8f} PRL confirmed; the sweep waits for {threshold / GRAIN:g}")

    dest = get_json(f"{BRIDGE}/api/deposit-address?ethAddress={conf['treasury_evm']}")
    if str(dest.get("ethAddress", "")).lower() != conf["treasury_evm"].lower() or dest.get("pearlAddress") != conf["deposit_pinned"]:
        raise SystemExit(f"REFUSED: the bridge now answers {dest.get('pearlAddress')} for {dest.get('ethAddress')}, "
                         f"not the pinned {conf['deposit_pinned']}. Nothing was sent. If the bridge really changed it, "
                         f"check on {BRIDGE} with your own eyes and update deposit_pinned in {CONF}.")
    status = get_json(f"{BRIDGE_API}/v1/status")
    if status.get("paused"):
        return say("the bridge is paused; waiting")
    if status.get("contracts", {}).get("network") != "mainnet":
        raise SystemExit(f"REFUSED: the bridge reports network {status.get('contracts', {}).get('network')!r}, not mainnet")
    fee_min = int(status["fees"]["depositFeeMinGrains"])
    send = grains - int(conf.get("fee_reserve_prl", 0.05) * GRAIN)  # the Pearl network fee comes from what is left
    if send < 10 * fee_min:
        return say(f"{send / GRAIN:.8f} PRL is under ten times the bridge's minimum fee ({fee_min / GRAIN:g}); waiting for more")
    fast = int(status["limits"]["fastMintWindowRemainingGrains"])
    lane = "fast lane (~20 min)" if send <= fast else "slow lane (~24 h): more than the bridge's fast window has left today"

    amount = f"{send / GRAIN:.8f}"
    if not live:
        return say(f"dry run: would send {amount} PRL to {conf['deposit_pinned']} ({lane}); run with --live to send")

    w.call("walletpassphrase", passphrase(), 60)
    try:
        txid = w.call("sendtoaddress", conf["deposit_pinned"], float(amount))
    finally:
        try:
            w.call("walletlock")
        except Exception:
            pass
    rec = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "ts": time.time(), "txid": txid, "prl": amount, "dest": conf["deposit_pinned"], "lane": lane}
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    os.chmod(LOG, 0o600)
    say(f"sent {amount} PRL to PearlBridge as {txid} ({lane}); it arrives as WPRL at {conf['treasury_evm']}")


if __name__ == "__main__":
    main()
