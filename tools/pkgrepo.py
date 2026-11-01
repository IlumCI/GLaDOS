#!/usr/bin/env python3
"""The [glados] package repository: build it, sign it, check it, publish it.

    python3 tools/pkgrepo.py build              every packaging/*/PKGBUILD -> out/repo/
    python3 tools/pkgrepo.py check              every signature in out/repo/, against the shipped key
    python3 tools/pkgrepo.py publish            upload out/repo/ to the release (needs --yes)
    python3 tools/pkgrepo.py serve [port]       out/repo/ over HTTP, for a guest under QEMU

A package is a directory under packaging/ with an ordinary PKGBUILD, built by
the host's own makepkg, so what is in the repository is exactly what Arch's
tools say it is. Each package is signed, and so is the database, with the
repository's own key -- not the update key and not the verdict key, which
sign different things and must not be answerable by one signature. The
private half lives outside the tree (--key-home, by default
~/.local/share/glados/repo-gnupg); the public half is packaging/glados-keyring,
which is how a machine comes to trust it.

The repository is a GitHub release, `repo-x86_64`, on IlumCI/GLaDOS: a
release is a flat directory of files at a stable URL, which is all pacman
asks of a server. Release assets cannot be symlinks, so the `glados.db` and
`glados.files` that repo-add makes as links are written as copies. And a
release asset's name cannot hold a colon, so a package with an epoch is
refused here rather than published as a 404.
"""

import argparse
import hashlib
import http.server
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGING = ROOT / "packaging"
OUT = ROOT / "out" / "repo"
REPO = "glados"
GITHUB = "IlumCI/GLaDOS"
TAG = "repo-x86_64"
KEY_HOME = Path.home() / ".local/share/glados/repo-gnupg"
# Pinned, so a key that is not this one cannot sign the repository by
# accident: the shipped keyring and the signer must name the same key.
FINGERPRINT = "83E75C5B98F2579892AF3019AB32E9629000D4EC"


def run(cmd, **kw):
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        sys.exit(f"pkgrepo: {' '.join(map(str, cmd))} failed ({r.returncode})")
    return r


def signer_env(home):
    env = dict(os.environ, GNUPGHOME=str(home))
    out = subprocess.run(["gpg", "--list-secret-keys", "--with-colons"], env=env,
                         capture_output=True, text=True).stdout
    if FINGERPRINT not in out:
        sys.exit(f"pkgrepo: no secret key {FINGERPRINT} in {home}; the repository is signed with that key only")
    return env


def shipped_key_matches():
    """The key in packaging/glados-keyring is the one this signs with -- or
    every machine that installs it trusts a key that signs nothing."""
    with tempfile.TemporaryDirectory() as t:
        env = dict(os.environ, GNUPGHOME=t)
        run(["gpg", "--batch", "--quiet", "--import", str(PACKAGING / "glados-keyring/glados.gpg")],
            env=env, capture_output=True)
        out = subprocess.run(["gpg", "--list-keys", "--with-colons"], env=env,
                             capture_output=True, text=True).stdout
    trusted = (PACKAGING / "glados-keyring/glados-trusted").read_text().split(":")[0]
    return FINGERPRINT in out and trusted == FINGERPRINT


def build(home):
    if not shipped_key_matches():
        sys.exit("pkgrepo: packaging/glados-keyring does not carry the signing key")
    env = signer_env(home)
    arch = OUT / "x86_64"
    if arch.exists():
        shutil.rmtree(arch)
    arch.mkdir(parents=True)
    pkgs = []
    for d in sorted(p for p in PACKAGING.iterdir() if (p / "PKGBUILD").exists()):
        # Built in a scratch copy, so makepkg's src/ and pkg/ never land in
        # the tree, and signed by makepkg itself with the repository's key.
        with tempfile.TemporaryDirectory() as t:
            work = Path(t) / d.name
            shutil.copytree(d, work)
            benv = dict(env, PKGDEST=str(arch), GPGKEY=FINGERPRINT, SRCDEST=t, BUILDDIR=t)
            run(["makepkg", "--force", "--clean", "--sign", "--nodeps", "--noconfirm"],
                cwd=work, env=benv, capture_output=True)
    for p in sorted(arch.glob("*.pkg.tar.zst")):
        if ":" in p.name:
            sys.exit(f"pkgrepo: {p.name} has an epoch; a release asset cannot be named with a colon")
        pkgs.append(p)
    run(["repo-add", "--sign", "--key", FINGERPRINT, "--quiet", str(arch / f"{REPO}.db.tar.gz"),
         *map(str, pkgs)], env=env, capture_output=True)
    for name in ("db", "files"):
        for suffix in ("", ".sig"):
            link = arch / f"{REPO}.{name}{suffix}"
            target = arch / f"{REPO}.{name}.tar.gz{suffix}"
            if link.is_symlink() or link.exists():
                link.unlink()
            shutil.copyfile(target, link)
    print(f"pkgrepo: {len(pkgs)} package(s) in {arch}")
    for p in pkgs:
        print(f"  {p.name}  {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}")


def check():
    """Every package and the database verify against the shipped public key,
    in a keyring holding nothing else -- the check a machine will make."""
    arch = OUT / "x86_64"
    files = sorted(arch.glob("*.pkg.tar.zst")) + [arch / f"{REPO}.db", arch / f"{REPO}.files"]
    with tempfile.TemporaryDirectory() as t:
        env = dict(os.environ, GNUPGHOME=t)
        run(["gpg", "--batch", "--quiet", "--import", str(PACKAGING / "glados-keyring/glados.gpg")],
            env=env, capture_output=True)
        bad = 0
        for f in files:
            sig = f.with_name(f.name + ".sig")
            r = subprocess.run(["gpg", "--batch", "--status-fd", "1", "--verify", str(sig), str(f)],
                               env=env, capture_output=True, text=True)
            ok = r.returncode == 0 and f"VALIDSIG {FINGERPRINT}" in r.stdout
            print(f"  {'ok  ' if ok else 'FAIL'}  {f.name}")
            bad += not ok
    if bad:
        sys.exit(f"pkgrepo: {bad} file(s) do not verify")
    print(f"pkgrepo: {len(files)} file(s) verify against {FINGERPRINT[-16:]}")


def publish(yes):
    arch = OUT / "x86_64"
    files = [p for p in sorted(arch.iterdir()) if p.is_file() and not p.name.endswith(".tar.gz")
             and not p.name.endswith(".tar.gz.sig")]
    print(f"pkgrepo: would upload {len(files)} file(s) to {GITHUB} release {TAG}:")
    for f in files:
        print(f"  {f.name}")
    if not yes:
        print("pkgrepo: nothing sent; --yes publishes")
        return
    have = subprocess.run(["gh", "release", "view", TAG, "--repo", GITHUB], capture_output=True)
    if have.returncode != 0:
        run(["gh", "release", "create", TAG, "--repo", GITHUB, "--title", "[glados] package repository",
             "--notes", "The [glados] pacman repository. Rebuilt from packaging/ by tools/pkgrepo.py.",
             "--latest=false"])
    run(["gh", "release", "upload", TAG, "--repo", GITHUB, "--clobber", *map(str, files)])
    print(f"pkgrepo: Server = https://github.com/{GITHUB}/releases/download/{TAG}")


def serve(port):
    os.chdir(OUT / "x86_64")
    print(f"pkgrepo: serving {OUT / 'x86_64'} on :{port} (a guest under QEMU reaches it as 10.0.2.2)")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), http.server.SimpleHTTPRequestHandler).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["build", "check", "publish", "serve"])
    ap.add_argument("port", nargs="?", type=int, default=8070)
    ap.add_argument("--key-home", type=Path, default=KEY_HOME)
    ap.add_argument("--yes", action="store_true", help="publish: actually upload")
    a = ap.parse_args()
    if a.what == "build":
        build(a.key_home)
        check()
    elif a.what == "check":
        check()
    elif a.what == "publish":
        publish(a.yes)
    else:
        serve(a.port)


if __name__ == "__main__":
    main()
