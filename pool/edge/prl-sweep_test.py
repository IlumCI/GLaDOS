# python3 pool/edge/prl-sweep_test.py -- the sweep's decisions against a faked wallet, bridge and
# network: never the real wallet, never the real passphrase (a dummy in a temp dir).
import importlib.util, io, json, os, sys, tempfile, contextlib
spec = importlib.util.spec_from_file_location("sweep", os.path.join(os.path.dirname(os.path.abspath(__file__)), "prl-sweep.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
home = tempfile.mkdtemp(); os.chmod(home, 0o700)
S.HOME, S.CONF, S.LOG = home, os.path.join(home, "sweep.json"), os.path.join(home, "sweeps.jsonl")
PIN = "prl1pchxq5smsmxsl3l59n8knnrgj386vd7qs8dnw468mwln9ezwjuttqxu67sd"; EVM = "0xfaE492EA62AA2250E0Ab2d1a6331E3c991443Cc5"
with open(S.CONF, "w") as f: json.dump({"treasury_evm": EVM, "deposit_pinned": PIN, "threshold_prl": 50, "fee_reserve_prl": 0.05, "rpc_port": 1, "rpc_user": "u", "rpc_pass": "p"}, f)
os.chmod(S.CONF, 0o600)
with open(os.path.join(home, "SECRETS.txt"), "w") as f: f.write("x\nprivate passphrase: hunter2-not-real\n")
os.chmod(os.path.join(home, "SECRETS.txt"), 0o600)
sent = []
def world(height=130453, net=130453, bal=120.5, dep=PIN, deveth=EVM, paused=False, network="mainnet", mint=None):
    class W:
        def __init__(self, conf): pass
        def call(self, m, *p):
            if m == "getbestblock": return {"hash": "h", "height": height}
            if m == "getbalance": return bal
            if m == "walletpassphrase": assert p[0] == "hunter2-not-real"; return None
            if m == "sendtoaddress": sent.append(p); return "ab" * 32
            if m == "walletlock": return None
            raise Exception(m)
    S.Wallet = W
    def gj(url, timeout=20):
        if "2miners" in url: return {"nodes": [{"height": str(net)}]}
        if "deposit-address" in url: return {"pearlAddress": dep, "ethAddress": deveth.lower()}
        if "/v1/status" in url: return {"paused": paused, "contracts": {"network": network}, "fees": {"depositFeeMinGrains": "400000000"}, "limits": {"fastMintWindowRemainingGrains": "38972269834526"}}
        if "mint-status" in url: return {"state": mint}
        raise Exception(url)
    S.get_json = gj
def run(*args):
    sys.argv = ["prl-sweep.py", *args]; out = io.StringIO(); code = 0
    with contextlib.redirect_stdout(out):
        try: S.main()
        except SystemExit as e: code = 1; print(str(e))
    return out.getvalue().strip(), code
passed = failed = 0
def ok(c, w):
    global passed, failed
    passed += c; failed += (not c); print(("ok  " if c else "FAIL") + "  " + w)
world(height=100000); o, _ = run(); ok("catch up" in o and not sent, "wallet behind the network: waits, sends nothing")
world(bal=30); o, _ = run(); ok("waits for 50" in o and not sent, "30 PRL under a 50 threshold: waits")
world(); o, _ = run(); ok("dry run: would send 120.45000000 PRL" in o and not sent and "hunter2" not in o, "120.5 PRL, no flag: a dry run that says what it would send, and sends nothing")
world(dep="prl1pATTACKER"); o, c = run("--live"); ok("REFUSED" in o and c == 1 and not sent, "the bridge answering another deposit address: refused, even with --live")
world(deveth="0x" + "11" * 20); o, c = run("--live"); ok("REFUSED" in o and not sent, "the bridge answering for another EVM address: refused")
world(paused=True); o, _ = run("--live"); ok("paused" in o and not sent, "the bridge paused: waits")
world(network="sepolia"); o, c = run("--live"); ok("REFUSED" in o and not sent, "the bridge on a test network: refused")
world(bal=40.04, ); o, _ = run("--live")
S_conf = json.load(open(S.CONF)); S_conf["threshold_prl"] = 10; json.dump(S_conf, open(S.CONF, "w")); os.chmod(S.CONF, 0o600)
world(bal=30); o, _ = run("--live"); ok("ten times" in o and not sent, "30 PRL is under ten times the 4 PRL minimum fee: waits")
S_conf["threshold_prl"] = 50; json.dump(S_conf, open(S.CONF, "w")); os.chmod(S.CONF, 0o600)
world(); o, _ = run("--live"); ok(len(sent) == 1 and sent[0][0] == PIN and abs(sent[0][1] - 120.45) < 1e-9 and "hunter2" not in o, f"--live: one send of 120.45 PRL to the pinned address ({o[-80:]})")
log = [json.loads(l) for l in open(S.LOG)]; ok(len(log) == 1 and oct(os.stat(S.LOG).st_mode & 0o777) == "0o600", "recorded in sweeps.jsonl, mode 600")
world(mint=None); o, _ = run("--live"); ok(len(sent) == 1 and "not at the bridge yet" in o, "the last sweep not yet seen by the bridge: no second send")
world(mint="minted"); o, _ = run("--live"); ok(len(sent) == 2, "once the bridge has it, the next sweep may go")
os.chmod(os.path.join(home, "SECRETS.txt"), 0o644); open(S.LOG, "w").close(); world(mint="minted")
o, c = run("--live"); ok(c == 1 and "readable by others" in o and len(sent) == 2, "a SECRETS.txt others can read: refused before unlocking")
print(f"\n{passed} passed, {failed} failed")
