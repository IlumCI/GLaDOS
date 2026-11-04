#!/usr/bin/env python3
"""orange's AUR half: find, review, build and install packages from the AUR.

The AUR is unreviewed: every package there is a build script somebody
uploaded, and nothing in GLaDOS -- not the [glados] key, not anybody here --
vouches for any of it. So this never builds anything it was not explicitly
asked to build, shows the PKGBUILD before it does, and says on every path
that what is being built is unreviewed.

How a build runs here, and why each part is the shape it is:

* makepkg refuses to run as root, and every guest process starts as root, so
  the build drops to a builder user with `setpriv` (the kernel keeps a real
  per-process uid for exactly this).
* makepkg packages under fakeroot, whose job is to make the files look
  root-owned. This kernel has no file ownership -- every file reports uid 0
  -- so that is already true, and fakeroot is a shim that sets the variable
  makepkg looks for and runs the command (`/usr/lib/orange/fakeroot`).
* Dependencies are installed by pacman as root before the build, and makepkg
  runs with --nodeps: it cannot install anything itself without sudo.
* The built packages are installed with `pacman -U`, so pacman owns every
  file and `orange remove` and `pacman -Qm` work on them as on any package.

    orange aur search <words>        search the AUR
    orange aur info <pkg>            what the AUR says about it
    orange aur show <pkg>            print its PKGBUILD, to read before building
    orange aur install <pkg>... [-- makepkg flags]
    orange aur upgrade               rebuild installed AUR packages that have newer versions
    orange aur list                  installed packages that came from the AUR
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request

AUR = "https://aur.archlinux.org"
CACHE = "/var/cache/orange/aur"
SHIM = "/usr/lib/orange/shim"
BUILDER = ("orange-build", 967)


def say(msg):
    print(f"orange: {msg}", flush=True)


def die(msg, code=1):
    print(f"orange: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


# ---- the AUR's own interface ---------------------------------------------------

def rpc(params):
    url = f"{AUR}/rpc/v5/" + params
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.load(r)
    except Exception as e:
        die(f"could not reach the AUR ({e})")
    if data.get("type") == "error":
        die(f"the AUR answered: {data.get('error')}")
    return data.get("results", [])


def info(names):
    """name -> the AUR's record, for every name it knows."""
    out = {}
    names = list(dict.fromkeys(names))
    for i in range(0, len(names), 100):
        q = "&".join("arg[]=" + urllib.parse.quote(n) for n in names[i:i + 100])
        for r in rpc("info?" + q):
            out[r["Name"]] = r
    return out


def providers(dep):
    """AUR packages that provide `dep` when none is named it."""
    return rpc(f"search/{urllib.parse.quote(dep)}?by=provides")


# ---- pacman, asked rather than reimplemented ------------------------------------

def run(cmd, **kw):
    return subprocess.run(cmd, **kw)


def quiet(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)


def satisfied(dep):
    """Is something installed that satisfies `dep` (versions included)?"""
    return quiet(["pacman", "-T", dep]).returncode == 0


def in_repos(dep):
    """Can pacman install `dep` from a repository -- a package, a provider or a group?"""
    r = quiet(["pacman", "-Sp", "--print-format", "%n", dep])
    if r.returncode == 0 and r.stdout.strip():
        return True
    return bool(quiet(["pacman", "-Sg", dep]).stdout.strip())


def bare(dep):
    """`foo>=1.2` -> `foo`: the AUR is asked by name, pacman by the whole thing."""
    for op in (">=", "<=", "=", ">", "<"):
        if op in dep:
            return dep.split(op, 1)[0]
    return dep


def vercmp(a, b):
    return int(quiet(["vercmp", a, b]).stdout.strip() or "0")


# ---- resolving -----------------------------------------------------------------

def resolve(targets, with_check=False):
    """Everything needed to build `targets`: AUR records in build order, and
    repository packages to install first. Refuses, naming it, a dependency
    nothing provides."""
    order, seen, repo = [], set(), []
    recs = info(targets)
    missing = [t for t in targets if t not in recs]
    if missing:
        die(f"not in the AUR: {', '.join(missing)}")

    def visit(rec, chain):
        name = rec["Name"]
        if name in seen:
            return
        if name in chain:
            die(f"a dependency cycle: {' -> '.join(chain + [name])}")
        deps = rec.get("Depends", []) + rec.get("MakeDepends", [])
        if with_check:
            deps += rec.get("CheckDepends", [])
        aur_needed = []
        for d in deps:
            if satisfied(d):
                continue
            if in_repos(d):
                if d not in repo:
                    repo.append(d)
                continue
            aur_needed.append(d)
        found = info([bare(d) for d in aur_needed])
        for d in aur_needed:
            r = found.get(bare(d))
            if r is None:
                prov = providers(bare(d))
                if not prov:
                    die(f"{name} needs {d}, which no repository and no AUR package provides")
                # The most voted provider, said aloud: a choice made for the
                # person should be visible to them.
                r = max(prov, key=lambda p: p.get("NumVotes", 0))
                r = info([r["Name"]])[r["Name"]]
                say(f"{d} is provided by the AUR package {r['Name']}")
            visit(r, chain + [name])
        seen.add(name)
        order.append(rec)

    for t in targets:
        visit(recs[t], [])
    return order, repo


# ---- building ------------------------------------------------------------------

def ensure_builder():
    name, uid = BUILDER
    for path, line in (("/etc/passwd", f"{name}:x:{uid}:{uid}:orange AUR builds:{CACHE}:/usr/bin/nologin\n"),
                       ("/etc/group", f"{name}:x:{uid}:\n")):
        try:
            text = open(path).read()
        except FileNotFoundError:
            text = ""
        if not any(l.split(":")[0] == name for l in text.splitlines()):
            with open(path, "a") as f:
                f.write(line)
    os.makedirs(CACHE, exist_ok=True)
    conf = os.path.join(CACHE, ".makepkg.conf")
    # Debug packages need debugedit and nothing here installs them; LTO is
    # left to the PKGBUILD. Everything else is the distribution's default.
    with open(conf, "w") as f:
        f.write("OPTIONS=(strip docs !libtool !staticlibs emptydirs zipman purge !debug)\n")
    # git refuses a repository whose owner is not the user running it, and
    # GLaDOS has no file ownership: every file reports root while the build
    # runs as the builder. The check has nothing to protect here, so the
    # builder's own git trusts the build directories.
    with open(os.path.join(CACHE, ".gitconfig"), "w") as f:
        f.write(f"[safe]\n\tdirectory = *\n")


def fetch(rec):
    """The package's build files, from the AUR snapshot, into a fresh directory."""
    base = rec["PackageBase"]
    dest = os.path.join(CACHE, base)
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    tgz = os.path.join(CACHE, base + ".tar.gz")
    url = AUR + rec["URLPath"]
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tgz, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception as e:
        die(f"could not download {base} ({e})")
    with tarfile.open(tgz) as t:
        for m in t.getmembers():
            # A snapshot is one directory named for the package; anything that
            # would land outside it is refused rather than extracted.
            if m.name.startswith("/") or ".." in m.name.split("/") or not (m.name == base or m.name.startswith(base + "/")):
                die(f"{base}'s snapshot holds a path outside itself: {m.name}")
        t.extractall(CACHE)
    os.remove(tgz)
    return dest


def as_builder(cmd, cwd, quiet=False):
    name, uid = BUILDER
    env = dict(os.environ, HOME=CACHE, USER=name, LOGNAME=name, GNUPGHOME=os.path.join(CACHE, ".gnupg"),
               PATH=SHIM + ":" + os.environ.get("PATH", "/usr/bin:/bin"),
               # The same trust as the builder's .gitconfig, in the form git
               # applies to every git a build starts whatever it reads first.
               GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="safe.directory", GIT_CONFIG_VALUE_0="*")
    os.makedirs(env["GNUPGHOME"], mode=0o700, exist_ok=True)
    out = subprocess.DEVNULL if quiet else None
    return run(["setpriv", f"--reuid={uid}", f"--regid={uid}", "--clear-groups", "--"] + cmd,
               cwd=cwd, env=env, stdout=out, stderr=out)


def signers(d):
    """The keys a PKGBUILD says may sign its sources, from its .SRCINFO."""
    keys = []
    try:
        for line in open(os.path.join(d, ".SRCINFO")):
            k, _, v = line.strip().partition(" = ")
            if k == "validpgpkeys" and v:
                keys.append(v)
    except FileNotFoundError:
        pass
    return keys


def import_keys(d, base):
    """Fetch the signers' keys into the builder's keyring, as yay does, so
    makepkg can verify what it downloads. Refuses rather than skipping: a
    signature nobody checked is the one thing a signed source exists to stop."""
    keys = signers(d)
    if not keys:
        return
    missing = [k for k in keys if as_builder(["gpg", "--batch", "--no-permission-warning", "--list-keys", k], d, quiet=True).returncode != 0]
    if not missing:
        return
    say(f"{base}'s sources are signed; fetching the signers' keys: {', '.join(missing)}")
    # Over HTTPS from here rather than `gpg --recv-keys`: gpg's dirmngr wants a
    # socket this kernel does not serve. What is imported is checked by full
    # fingerprint afterwards, so a keyserver cannot substitute another key.
    servers = ("https://keyserver.ubuntu.com/pks/lookup?op=get&options=mr&search=0x{}",
               "https://keys.openpgp.org/vks/v1/by-fingerprint/{}")
    for k in missing:
        for url in servers:
            try:
                with urllib.request.urlopen(url.format(k.upper()), timeout=60) as r:
                    armored = r.read()
            except Exception:
                continue
            path = os.path.join(CACHE, k + ".asc")
            with open(path, "wb") as f:
                f.write(armored)
            as_builder(["gpg", "--batch", "--no-permission-warning", "--import", path], d, quiet=True)
            os.remove(path)
            if as_builder(["gpg", "--batch", "--no-permission-warning", "--list-keys", k], d, quiet=True).returncode == 0:
                say(f"key {k} imported")
                break
    if all(as_builder(["gpg", "--batch", "--no-permission-warning", "--list-keys", k], d, quiet=True).returncode == 0 for k in missing):
        return
    die(f"could not fetch the keys that sign {base}'s sources, so they cannot be verified. "
        f"To build without that check: 'orange aur install --yes {base} -- --skippgpcheck'")


def build(rec, extra):
    d = fetch(rec)
    if "--skippgpcheck" not in extra:
        import_keys(d, rec["PackageBase"])
    say(f"building {rec['PackageBase']} {rec['Version']} -- unreviewed AUR code, running as {BUILDER[0]}")
    r = as_builder(["makepkg", "--nodeps", "--noconfirm", "--force", "--cleanbuild"] + extra, d)
    if r.returncode != 0:
        die(f"{rec['PackageBase']} did not build (makepkg exited {r.returncode}); its files are in {d}")
    listed = subprocess.run(["setpriv", f"--reuid={BUILDER[1]}", f"--regid={BUILDER[1]}", "--clear-groups", "--",
                             "makepkg", "--packagelist"], cwd=d, env=dict(os.environ, HOME=CACHE),
                            stdout=subprocess.PIPE, text=True).stdout.split()
    built = [p for p in listed if os.path.exists(p)]
    if not built:
        die(f"{rec['PackageBase']} built nothing makepkg can name")
    return built


def install(targets, extra, review_ok):
    order, repo = resolve(targets)
    # Every PKGBUILD assumes base-devel without naming it -- makepkg's own
    # rule -- so make, gcc and patch are never in depends and a build fails
    # on the first one it reaches. Installed explicitly rather than as a
    # dependency: it is a toolchain the operator now has, and an orphan
    # sweep taking it away would break the next AUR build silently.
    toolchain = [] if quiet(["pacman", "-Qq", "base-devel"]).returncode == 0 else ["base-devel"]
    if toolchain:
        say("the AUR builds with base-devel, which is not installed; it will be")
    say("from the AUR (unreviewed): " + ", ".join(f"{r['Name']} {r['Version']}" for r in order))
    if repo:
        say("from the repositories first: " + ", ".join(repo))
    if not review_ok:
        if not sys.stdin.isatty():
            die("nothing is built without a yes, and nothing here can ask -- read each PKGBUILD with "
                "'orange aur show <pkg>', then 'orange aur install --yes <pkg>'")
        for r in order:
            show(r["Name"])
        if input("orange: build and install these? [y/N] ").strip().lower() not in ("y", "yes"):
            die("nothing built", 0)
    ensure_builder()
    if toolchain:
        r = run(["pacman", "-S", "--needed", "--noconfirm"] + toolchain)
        if r.returncode != 0:
            die("base-devel did not install, and nothing in the AUR builds without it")
    if repo:
        r = run(["pacman", "-S", "--needed", "--asdeps", "--noconfirm"] + repo)
        if r.returncode != 0:
            die("the repository dependencies did not install")
    for rec in order:
        files = build(rec, extra)
        wanted = [f for f in files if os.path.basename(f).rsplit("-", 3)[0] == rec["Name"]] or files
        as_dep = rec["Name"] not in targets
        r = run(["pacman", "-U", "--noconfirm"] + (["--asdeps"] if as_dep else []) + wanted)
        if r.returncode != 0:
            die(f"{rec['Name']} built but did not install")
    say("installed: " + ", ".join(targets))


def show(name):
    rec = info([name]).get(name) or die(f"{name} is not in the AUR")
    d = fetch(rec)
    print(f"---- {rec['PackageBase']} {rec['Version']}: PKGBUILD (unreviewed) ----")
    print(open(os.path.join(d, "PKGBUILD")).read())
    others = [f for f in sorted(os.listdir(d)) if f not in ("PKGBUILD", ".SRCINFO")]
    if others:
        print("---- also in it: " + ", ".join(others))


def foreign():
    """Installed packages no repository has, which the AUR knows by name.

    `pacman -Qm` alone also lists packages that came from neither -- the
    image's own `glados-system`, a package somebody built by hand -- and
    calling those AUR packages would offer to "upgrade" them from a stranger's
    PKGBUILD that happens to share the name."""
    out = quiet(["pacman", "-Qm"]).stdout.split("\n")
    local = dict(l.split(" ", 1) for l in out if " " in l)
    known = info(list(local)) if local else {}
    return {n: v for n, v in local.items() if n in known}


def upgrade(extra, review_ok):
    have = foreign()
    if not have:
        say("no AUR packages are installed")
        return
    recs = info(list(have))
    newer = [n for n, v in have.items() if n in recs and vercmp(recs[n]["Version"], v) > 0]
    if not newer:
        say("every AUR package is current")
        return
    for n in newer:
        say(f"{n} {have[n]} -> {recs[n]['Version']}")
    install(newer, extra, review_ok)


def main(argv):
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    yes = False
    for flag in ("--yes", "-y"):
        if flag in argv:
            argv.remove(flag)
            yes = True
    if not argv or argv[0] in ("help", "-h", "--help"):
        print(__doc__.split("\n\n")[-1])
        return
    verb, args = argv[0], argv[1:]
    if verb == "search":
        if not args:
            die("search for what?")
        res = rpc("search/" + urllib.parse.quote(" ".join(args)) + "?by=name-desc")
        for r in sorted(res, key=lambda r: -r.get("Popularity", 0))[:40]:
            print(f"aur/{r['Name']} {r['Version']}  ({r.get('NumVotes', 0)} votes)\n    {r.get('Description') or ''}")
        if not res:
            say("nothing in the AUR matches")
    elif verb == "info":
        for n in args:
            r = info([n]).get(n)
            if not r:
                say(f"{n} is not in the AUR")
                continue
            for k in ("Name", "Version", "Description", "URL", "Maintainer", "NumVotes", "Popularity",
                      "Depends", "MakeDepends", "OutOfDate"):
                if r.get(k) not in (None, [], ""):
                    v = r[k]
                    print(f"{k:<14}: {', '.join(v) if isinstance(v, list) else v}")
            print(f"{'Reviewed':<14}: no -- the AUR is unreviewed")
            print()
    elif verb == "show":
        for n in args:
            show(n)
    elif verb == "install":
        if not args:
            die("install what?")
        install(args, extra, yes)
    elif verb == "upgrade":
        upgrade(extra, yes)
    elif verb == "list":
        for n, v in foreign().items():
            print(f"{n} {v}")
    elif verb == "offer":
        # For `orange install`, about names no repository has.
        recs = info(args)
        missing = False
        for n in args:
            r = recs.get(n)
            if r:
                say(f"{n} is not in the repositories; the AUR has {n} {r['Version']} (unreviewed) -- "
                    f"'orange aur show {n}' to read it, 'orange aur install --yes {n}' to build it")
            else:
                say(f"{n} is in neither the repositories nor the AUR")
                missing = True
        sys.exit(1 if missing else 0)
    elif verb == "pending":
        # For `orange upgrade`: say what the AUR has newer, build nothing.
        have = foreign()
        if have:
            recs = info(list(have))
            newer = [n for n, v in have.items() if n in recs and vercmp(recs[n]["Version"], v) > 0]
            if newer:
                say(f"newer in the AUR: {', '.join(newer)} -- 'orange aur upgrade --yes' rebuilds them")
    elif verb == "split":
        # For `orange install`: which of these names only the AUR has.
        print(" ".join(a for a in args if not a.startswith("-") and not in_repos(a)))
    else:
        die(f"'{verb}' is not an aur command -- 'orange aur help'")


if __name__ == "__main__":
    main(sys.argv[1:])
