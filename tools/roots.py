#!/usr/bin/env python3
"""Build a `roots.der` trust store from this host's CA bundle, on any platform.

`scripts/fetch-roots.ps1` is the original and exports the Windows store. That
was the only way to make one, so a machine without Windows had no trust store at
all -- and with no roots file, **every** certificate fails to validate, which is
correct and makes TLS useless. A mining image built on Linux had exactly that.

The rules are the PowerShell script's, deliberately, so the two cannot disagree
about what a trust anchor is: only self-signed certificates, and only ones valid
now. Output is a plain concatenation of DER, which is what `net::trust` reads.

**Which store this reads is a decision about who you trust, and it is printed.**
The kernel once validated a Cloudflare chain only because the Windows store still
carried a legacy root the chain did not need; an Arch store that had dropped it
exposed the bug. So the source path, the count and the sha256 of the output are
all printed, and two machines that disagree about `roots.der` can see that they
do before either of them debugs a handshake.

    python3 tools/roots.py                      # write esp/GLADOS/roots.der
    python3 tools/roots.py --list
    python3 tools/roots.py --filter 'ISRG|GTS|DigiCert'
    python3 tools/roots.py --bundle /path/to/ca.pem --out out/roots.der
    python3 tools/roots.py --selftest

No third-party packages and no `openssl` binary: the few fields needed are read
straight out of the DER, which also makes this a second reader of the fields
`src/net/x509.rs` parses.
"""
import argparse
import base64
import calendar
import hashlib
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where distributions keep the bundle. Checked in order; the first that exists
# wins, and which one it was is printed.
BUNDLES = (
    "/etc/ssl/certs/ca-certificates.crt",   # Debian, Ubuntu, Arch, Alpine
    "/etc/pki/tls/certs/ca-bundle.crt",     # Fedora, RHEL
    "/etc/ssl/ca-bundle.pem",               # openSUSE
    "/etc/ssl/cert.pem",                     # macOS, some BSDs
)


def _read(der, at):
    """One DER element at `at`: (tag, start of contents, end of element)."""
    tag = der[at]
    n = der[at + 1]
    at += 2
    if n & 0x80:
        k = n & 0x7F
        if k == 0 or k > 4:
            raise ValueError("unsupported DER length")
        n = int.from_bytes(der[at:at + k], "big")
        at += k
    if at + n > len(der):
        raise ValueError("DER element runs past the end")
    return tag, at, at + n


def _children(der, start, end):
    at = start
    while at < end:
        tag, s, e = _read(der, at)
        yield tag, s, e, at
        at = e


def _time(tag, v):
    s = v.decode("ascii")
    if tag == 0x17:                         # UTCTime, YYMMDDHHMMSSZ
        yy = int(s[:2])
        s = ("19" if yy >= 50 else "20") + s
    return calendar.timegm(time.strptime(s.rstrip("Z"), "%Y%m%d%H%M%S"))


def fields(der):
    """(issuer DER, subject DER, notBefore, notAfter) of one certificate."""
    _, s, e = _read(der, 0)                  # Certificate
    _, ts, te = _read(der, s)                # TBSCertificate
    kids = list(_children(der, ts, te))
    i = 1 if kids and kids[0][0] == 0xA0 else 0   # optional [0] version
    # serial, signature algorithm, issuer, validity, subject
    _, _, _, _ = kids[i]
    issuer = der[kids[i + 2][3]:kids[i + 2][2]]
    vt, vs, ve, _ = kids[i + 3]
    times = [(t, der[a:b]) for t, a, b, _ in _children(der, vs, ve)]
    subject = der[kids[i + 4][3]:kids[i + 4][2]]
    return issuer, subject, _time(*times[0]), _time(*times[1])


def cn_of(name_der):
    """The CN in a Name, for display only. Never compared."""
    m = re.search(rb"\x06\x03\x55\x04\x03[\x0c\x13\x14\x16](.)", name_der, re.S)
    if not m:
        return "(no CN)"
    n = m.group(1)[0]
    start = m.end()
    return name_der[start:start + n].decode("utf-8", "replace")


def pems(text):
    for b in re.findall(r"-----BEGIN CERTIFICATE-----(.*?)-----END CERTIFICATE-----", text, re.S):
        yield base64.b64decode("".join(b.split()))


def anchors(ders, now, pattern=None):
    """The PowerShell script's rule: self-signed, and valid at `now`."""
    out, skipped = [], {"not self-signed": 0, "not valid now": 0, "unreadable": 0, "filtered": 0}
    for der in ders:
        try:
            issuer, subject, nb, na = fields(der)
        except Exception:
            skipped["unreadable"] += 1
            continue
        if issuer != subject:
            skipped["not self-signed"] += 1
            continue
        if not (nb <= now <= na):
            skipped["not valid now"] += 1
            continue
        if pattern and not re.search(pattern, cn_of(subject)):
            skipped["filtered"] += 1
            continue
        out.append((cn_of(subject), na, der))
    return out, skipped


def selftest():
    """The four certificates `diag x509` uses, read by the other reader."""
    ok = True

    def claim(good, what):
        nonlocal ok
        ok = ok and good
        print(f"  {'ok  ' if good else 'FAIL'}  {what}")

    fx = os.path.join(ROOT, "src", "net", "fixtures")
    read = lambda n: open(os.path.join(fx, n), "rb").read()
    leaf, we1, cross, root = (read(n) for n in
                              ("gts-leaf.der", "gts-we1.der", "gts-r4-cross.der", "gts-r4-root.der"))
    now = 1_790_596_800  # 2026-09-28 12:00 UTC, the instant x509's claims use

    got, _ = anchors([leaf, we1, cross, root], now)
    claim([cn for cn, _, _ in got] == ["GTS Root R4"], "of the four, only the self-signed root is an anchor")
    claim(fields(cross)[1] == fields(root)[1],
          "the cross-signed copy has the root's own subject -- which is why it misled the walk")
    claim(fields(cross)[0] != fields(root)[0], "and a different issuer, which is why it is not an anchor")
    claim(fields(leaf)[3] == 1795522545, "the leaf's notAfter reads as 2026-11-24 12:15:45 UTC")
    none, skip = anchors([root], 2_200_000_000)
    claim(not none and skip["not valid now"] == 1, "a root read after it expires is not exported")
    claim(anchors([root], now, "ISRG")[0] == [], "a filter that matches nothing exports nothing")
    try:
        fields(b"\x30\x05\x01\x02")
        claim(False, "a truncated certificate is refused")
    except Exception:
        claim(True, "a truncated certificate is refused rather than read past")
    print("\nselftest passed" if ok else "\nselftest FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", help="PEM bundle to read (default: the first system bundle found)")
    ap.add_argument("--out", default=os.path.join(ROOT, "esp", "GLADOS", "roots.der"))
    ap.add_argument("--list", action="store_true", help="print what would be exported, write nothing")
    ap.add_argument("--filter", help="only roots whose CN matches this regex")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    src = a.bundle or next((p for p in BUNDLES if os.path.exists(p)), None)
    if not src:
        print("no CA bundle found; pass --bundle", file=sys.stderr)
        return 1
    got, skipped = anchors(pems(open(src, encoding="utf-8").read()), int(time.time()), a.filter)
    print(f"source   {os.path.realpath(src)}")
    print(f"anchors  {len(got)}   skipped " + ", ".join(f"{v} {k}" for k, v in skipped.items() if v))
    if a.list:
        for cn, na, _ in sorted(got):
            print(f"  {cn:52} expires {time.strftime('%Y-%m-%d', time.gmtime(na))}")
        return 0
    blob = b"".join(der for _, _, der in got)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(blob)
    print(f"wrote    {a.out}  {len(blob)} bytes  sha256 {hashlib.sha256(blob).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
