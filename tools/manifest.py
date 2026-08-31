#!/usr/bin/env python3
"""Emit and sign the manifest an in-OS updater reads.

    manifest.py <image> --version V --base URL --key-file FILE
                [--channel C] [--notes TEXT] [--out DIR]
    manifest.py --verify <manifest> [--image <file>] [--key <hex-public>]
    manifest.py --selftest

The field names, and the rule that unknown ones are ignored, are the ones
`src/update/manifest.rs` parses. The two are checked against each other by
`--verify`, which reimplements the kernel's parser rather than calling this
file's own -- a writer and a reader that share code agree by construction and
prove nothing, which is the same bargain `tools/tokenizer.py --verify` makes.

A signed manifest is its text followed by the 80 bytes of GLADOSIG over that
text -- one object, because the kernel has one TCP connection and no
pipelining, and because two objects can be served out of step with each other
and produce a signature failure that is really a deployment race.

The manifest is signed because the image signature does not cover the choice
of image. A host that picks which correctly signed version you install can
pick an old one with a known hole in it, and every byte of that verifies.

Signing reuses `sign.py`, so a manifest this produces is checked by the same
80-byte GLADOSIG the kernel already verifies with `crypto::p256::verify` --
itself checked at boot against published ECDSA vectors.
"""
import hashlib
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sign  # noqa: E402

def private_key(argv):
    """The signing key, from a file by preference.

    `--key <hex>` puts the private half in shell history, in the process list,
    and in any CI log that echoes its command. `--key-file` is the one to use;
    the other stays because CI passes it from a secret, where the shell is not
    a person's.
    """
    if "--key-file" in argv:
        return pathlib.Path(argv[argv.index("--key-file") + 1]).read_text().strip()
    if "--key" in argv:
        return argv[argv.index("--key") + 1].strip()
    raise ValueError("no signing key -- pass --key-file FILE")


HEADER = "glados-update 1"
REQUIRED = ("channel", "version", "image", "sig", "size", "sha256")


def render(channel, version, base, size, digest, notes):
    """The manifest bytes, exactly as they will be signed and served."""
    base = base.rstrip("/")
    stem = f"{base}/glados-{version}.efi"
    lines = [
        HEADER,
        f"channel {channel}",
        f"version {version}",
        f"image {stem}",
        f"sig {stem}.sig",
        f"size {size}",
        f"sha256 {digest}",
    ]
    if notes:
        # One line: the kernel takes the rest of the line verbatim, and a
        # newline here would silently become a key it does not know.
        lines.append("notes " + " ".join(notes.split()))
    return ("\n".join(lines) + "\n").encode("utf-8")


def parse(text):
    """The kernel's parser, reimplemented. Raises ValueError the way it refuses.

    Deliberately not sharing code with `render`. This is the half that says
    whether what we published can be read by what we shipped.
    """
    try:
        text = text.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("not a manifest: not utf-8")
    lines = text.split("\n")
    if not lines or lines[0].strip() != HEADER:
        raise ValueError(f"not a manifest: first line is not {HEADER!r}")

    out = {"notes": ""}
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        key = parts[0]
        value = parts[1].strip() if len(parts) > 1 else ""
        if key in ("channel", "version", "notes"):
            out[key] = value
        elif key in ("image", "sig"):
            if not value.startswith("https://"):
                raise ValueError(f"the {key!r} line is not an https URL")
            out[key] = value
        elif key == "size":
            if not value.isdigit():
                raise ValueError("the 'size' line does not parse")
            out[key] = int(value)
        elif key == "sha256":
            if len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
                raise ValueError("the 'sha256' line is not a 64-character digest")
            out[key] = value.lower()
        # Anything else is ignored, as the kernel ignores it, so a later field
        # does not make an older kernel reject the whole file.

    missing = [k for k in REQUIRED if k not in out]
    if missing:
        raise ValueError("the manifest has no " + ", ".join(repr(m) for m in missing))
    return out


SIG_LEN = 80


def split(blob):
    """Text and signature, at a fixed offset from the end."""
    if len(blob) <= SIG_LEN:
        raise ValueError(f"{len(blob)} B is too short to be a signed manifest")
    return blob[:-SIG_LEN], blob[-SIG_LEN:]


def verify_sig(pub, data, blob):
    """Check an 80-byte GLADOSIG against an uncompressed public key."""
    if len(blob) != 80 or blob[:8] != b"GLADOSIG":
        raise ValueError("not a GLADOSIG signature")
    version = int.from_bytes(blob[8:12], "little")
    curve = int.from_bytes(blob[12:16], "little")
    if version != 1 or curve != 0:
        raise ValueError("a signature format this does not implement")
    r = int.from_bytes(blob[16:48], "big")
    s = int.from_bytes(blob[48:80], "big")
    if not (0 < r < sign.N and 0 < s < sign.N):
        raise ValueError("r or s is out of range")

    if len(pub) != 65 or pub[0] != 0x04:
        raise ValueError("the public key is not uncompressed 0x04||X||Y")
    q = (int.from_bytes(pub[1:33], "big"), int.from_bytes(pub[33:65], "big"))

    z = int.from_bytes(hashlib.sha256(data).digest(), "big")
    w = sign.inv(s, sign.N)
    p = sign.add(
        sign.mul(z * w % sign.N, (sign.GX, sign.GY)),
        sign.mul(r * w % sign.N, q),
    )
    if p is None:
        raise ValueError("not a signature over these bytes by this key")
    if p[0] % sign.N != r:
        raise ValueError("not a signature over these bytes by this key")


def public_of(d):
    q = sign.mul(d, (sign.GX, sign.GY))
    return b"\x04" + q[0].to_bytes(32, "big") + q[1].to_bytes(32, "big")


def selftest():
    """Round-trip a manifest through both halves, then break it four ways."""
    ok = True

    def claim(what, good):
        nonlocal ok
        if not good:
            ok = False
        print(f"  {'ok ' if good else 'FAIL'}  {what}")

    d, _ = sign.keygen()
    pub = public_of(d)
    image = os.urandom(4096)
    digest = hashlib.sha256(image).hexdigest()
    text = render("stable", "9.9.9", "https://example.invalid/stable", len(image), digest, "a test")
    blob = sign.pack(*sign.sign(d, hashlib.sha256(text).digest()))

    m = parse(text)
    claim("what render writes, parse reads", m["version"] == "9.9.9" and m["size"] == len(image))
    claim("the image URL is derived from the version", m["image"].endswith("/glados-9.9.9.efi"))
    claim("and the signature URL from the image", m["sig"] == m["image"] + ".sig")

    try:
        verify_sig(pub, text, blob)
        claim("a manifest we signed verifies", True)
    except ValueError as e:
        claim(f"a manifest we signed verifies ({e})", False)

    # Every way this is supposed to say no.
    def refuses(what, fn):
        try:
            fn()
            claim(what, False)
        except ValueError:
            claim(what, True)

    refuses("a tampered manifest does not verify",
            lambda: verify_sig(pub, text.replace(b"9.9.9", b"9.9.8"), blob))
    refuses("a manifest signed by somebody else does not verify",
            lambda: verify_sig(public_of(sign.keygen()[0]), text, blob))
    refuses("a truncated signature is refused",
            lambda: verify_sig(pub, text, blob[:79]))
    refuses("a future format is refused rather than guessed at",
            lambda: parse(b"glados-update 2\nchannel stable\n"))
    refuses("a plain-http image URL is refused",
            lambda: parse(b"glados-update 1\nimage http://example.invalid/x.efi\n"))
    refuses("a manifest missing a field it needs is refused",
            lambda: parse(b"glados-update 1\nchannel stable\n"))

    t, sg = split(text + blob)
    claim("a signed manifest splits back into its halves", t == text and sg == blob)
    refuses("and something shorter than a signature is not one",
            lambda: split(b"short"))

    # An unknown key must be ignored rather than fatal, or no field can ever
    # be added without every older kernel refusing the whole file.
    extended = text[:-1] + b"\nsomething-later yes\n"
    try:
        parse(extended)
        claim("an unknown field is ignored, not refused", True)
    except ValueError:
        claim("an unknown field is ignored, not refused", False)

    return ok


def main():
    argv = sys.argv[1:]

    if "--selftest" in argv:
        raise SystemExit(0 if selftest() else 1)

    if "--verify" in argv:
        man = pathlib.Path(argv[argv.index("--verify") + 1])
        blob = man.read_bytes()
        # A detached pair is still accepted, since that is what a hand-built
        # test case looks like before anything has published one.
        if "--sig" in argv:
            text = blob
            detached = pathlib.Path(argv[argv.index("--sig") + 1]).read_bytes()
        else:
            text, detached = split(blob)
        m = parse(text)
        print(f"  {man}")
        for k in REQUIRED:
            print(f"    {k:8} {m[k]}")
        if m["notes"]:
            print(f"    {'notes':8} {m['notes']}")

        if True:
            key = argv[argv.index("--key") + 1] if "--key" in argv else None
            if key is None:
                # Read the pinned key out of the kernel so a manifest is
                # checked against what will actually be running, not against
                # whatever was pasted on the command line.
                src = pathlib.Path(__file__).resolve().parent.parent / "src/update/mod.rs"
                body = src.read_text(encoding="utf-8")
                at = body.index("pub const UPDATE_KEY")
                end = body.index("];", at)
                # Every delimiter becomes whitespace, brackets included. A
                # key pasted on one line reads as "[0x04" for its first
                # element, which a scan for a "0x" prefix drops -- 64 bytes,
                # a key that looks unprovisioned, and a release that fails
                # its own verify step for a key that is perfectly good.
                seg = body[at:end]
                for ch in ",[];":
                    seg = seg.replace(ch, " ")
                nums = [int(t, 16) for t in seg.split() if t.startswith("0x")]
                if len(nums) != 65 or nums[0] != 0x04:
                    if not nums or all(n == 0 for n in nums):
                        raise SystemExit(
                            "  UPDATE_KEY is not provisioned in src/update/mod.rs -- "
                            "pass --key <hex-public> to check against another"
                        )
                    raise SystemExit(
                        f"  UPDATE_KEY parsed as {len(nums)} bytes starting "
                        f"{nums[0]:#04x}; it must be 65 starting 0x04"
                    )
                pub = bytes(nums)
            else:
                pub = bytes.fromhex(key)
            verify_sig(pub, text, detached)
            print("    signature ok")

        if "--image" in argv:
            data = pathlib.Path(argv[argv.index("--image") + 1]).read_bytes()
            if len(data) != m["size"]:
                raise SystemExit(f"  image is {len(data)} B, the manifest says {m['size']}")
            got = hashlib.sha256(data).hexdigest()
            if got != m["sha256"]:
                raise SystemExit(f"  image digest is {got}, the manifest says {m['sha256']}")
            print("    image matches")
        return

    if not argv or argv[0].startswith("-"):
        raise SystemExit(__doc__)

    image = pathlib.Path(argv[0])

    def opt(name, default=None):
        if name in argv:
            return argv[argv.index(name) + 1]
        if default is None:
            raise SystemExit(f"  {name} is required\n{__doc__}")
        return default

    version = opt("--version")
    channel = opt("--channel", "stable")
    base = opt("--base")
    notes = opt("--notes", "")
    key = private_key(argv)
    out = pathlib.Path(opt("--out", str(image.parent)))
    out.mkdir(parents=True, exist_ok=True)

    data = image.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    text = render(channel, version, base, len(data), digest, notes)
    blob = sign.pack(*sign.sign(int(key, 16), hashlib.sha256(text).digest()))

    # Refuse to publish something we cannot read back. A manifest that the
    # kernel would reject is worse than no manifest: it fails on the machines
    # that already trust us, at the moment they are trying to update.
    parse(text)
    verify_sig(public_of(int(key, 16)), text, blob)

    (out / "manifest").write_bytes(text + blob)
    print(f"  {channel} {version}: {len(data)} B, sha256 {digest}")
    print(f"  wrote {out / 'manifest'} ({len(text)} B of text + {SIG_LEN} B of signature)")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as e:
        # A refusal is an answer, not a crash. Every error raised here names
        # what was wrong -- a bad manifest or a file that is not there -- and a
        # traceback buries that under a stack nobody needs.
        raise SystemExit(f"  {e}")
