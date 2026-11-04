#!/usr/bin/env python3
"""orange's .deb half: turn a Debian package into a pacman one and install it.

GLaDOS's userland is Arch's, and pacman is what knows what is on the
machine. Running dpkg beside it would give a second database that knows
none of pacman's packages -- every .deb with a dependency refused unless
forced, and nothing pacman can remove afterwards. So a .deb is *converted*:
its files become a pacman package, its dependencies become Arch ones, and
pacman installs it, owns every file, and `orange remove` takes it away.

    orange install ./foo.deb          convert and install (orange deb install)
    orange deb show <file.deb>        what it holds, and what its dependencies became
    orange deb list                   installed packages that came from a .deb

**What a dependency is, is read off the binaries.** A Debian dependency
names a Debian package, and Debian's names are not Arch's -- libgtk-3-0 is
gtk3, libc6 is glibc, libxss1 is libxss. A table of those would be a guess
that goes stale. What the program actually needs to start is in its ELF
headers: every DT_NEEDED soname. Those are resolved against what is already
installed, then against the sonames Arch packages declare they provide
(`libncursesw.so=6-64`), then against the repositories' file lists. An
unresolved soname refuses the install by name; `--force` installs anyway.
Debian's own dependency names are then tried for what the binaries cannot
say -- an interpreter, a tool a script calls -- and anything that maps to
nothing is reported and does not block.

**Maintainer scripts are not run.** postinst and its relatives are written
against dpkg, debconf and update-alternatives; on this system they would
fail, or half-work, which is worse. They are kept beside the package's
control file under /usr/share/orange/deb/<name>/ to read, and what they
usually do -- refresh the desktop database, icon caches, ldconfig -- Arch's
own pacman hooks already do for every package.

**The layout is Arch's.** /bin, /sbin, /lib and /lib64 are symlinks here,
so a package holding them as directories cannot install; their contents
move under /usr. Debian's multiarch library directory has no meaning to
Arch's loader, so a package with libraries there carries an ld.so.conf.d
entry naming it.
"""

import os
import re
import shutil
import struct
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aur  # noqa: E402  (the builder, the shim, and makepkg as somebody else)

CACHE = "/var/cache/orange/deb"
MARK = "/usr/share/orange/deb"
MULTIARCH = "usr/lib/x86_64-linux-gnu"

say, die, quiet, run = aur.say, aur.die, aur.quiet, aur.run

# Debian names whose Arch name differs and that a binary cannot reveal: an
# interpreter or a tool a script runs. Library packages are not here on
# purpose -- the sonames answer for those, and answer correctly.
NAMES = {
    "python3": "python", "python3-minimal": "python", "perl-base": "perl",
    "libc-bin": "glibc", "libc6": "glibc", "libgcc-s1": "gcc-libs", "libstdc++6": "gcc-libs",
    "zlib1g": "zlib", "bsdutils": "util-linux", "mount": "util-linux", "procps": "procps-ng",
    "debianutils": "which", "dpkg": None, "debconf": None, "lsb-base": None, "init-system-helpers": None,
    "dbus-x11": "dbus", "dbus-user-session": "dbus", "x11-utils": "xorg-xprop",
    "x11-xserver-utils": "xorg-xrandr", "libgtk-3-0": "gtk3", "libgtk-3-0t64": "gtk3",
    "libgtk2.0-0": "gtk2", "libnss3": "nss", "libnspr4": "nspr", "libasound2": "alsa-lib",
    "libasound2t64": "alsa-lib", "libxss1": "libxss", "libxtst6": "libxtst", "libnotify4": "libnotify",
    "libsecret-1-0": "libsecret", "libgbm1": "mesa", "libvulkan1": "vulkan-icd-loader",
    "libatomic1": "gcc-libs", "fonts-liberation": "ttf-liberation", "gnupg": "gnupg",
    "apt": None, "apt-utils": None, "software-properties-common": None, "adduser": None,
}


# ---- the archive ----------------------------------------------------------------

def unpack(deb, into):
    """A .deb is an ar archive of control.tar.* and data.tar.*; bsdtar reads
    ar and every compression Debian uses, and comes with pacman."""
    if os.path.isdir(into):
        shutil.rmtree(into)
    outer = os.path.join(into, "ar")
    os.makedirs(outer)
    if run(["bsdtar", "-xf", os.path.abspath(deb), "-C", outer]).returncode != 0:
        die(f"{deb} is not an archive bsdtar can read")
    parts = os.listdir(outer)
    if "debian-binary" not in parts:
        die(f"{deb} is an archive but not a Debian package (no debian-binary)")
    ctl = next((p for p in parts if p.startswith("control.tar")), None)
    data = next((p for p in parts if p.startswith("data.tar")), None)
    if not ctl or not data:
        die(f"{deb} has no {'control' if not ctl else 'data'} archive")
    for name, sub in ((ctl, "control"), (data, "root")):
        d = os.path.join(into, sub)
        os.makedirs(d)
        # Members are checked before extraction: a data archive is paths a
        # stranger chose, and one climbing out of the tree is refused whole.
        names = quiet(["bsdtar", "-tf", os.path.join(outer, name)]).stdout.split("\n")
        for n in names:
            if n.startswith("/") or ".." in n.split("/"):
                die(f"{deb} holds a path outside itself: {n}")
        if run(["bsdtar", "-xpf", os.path.join(outer, name), "-C", d]).returncode != 0:
            die(f"{deb}'s {sub} archive did not extract")
    shutil.rmtree(outer)
    return os.path.join(into, "control"), os.path.join(into, "root")


def control(ctl_dir):
    """The control file's fields, continuation lines joined."""
    fields, key = {}, None
    for line in open(os.path.join(ctl_dir, "control"), encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if line[:1] in (" ", "\t") and key:
            fields[key] += "\n" + line.strip()
        elif ":" in line:
            key, _, v = line.partition(":")
            key = key.strip()
            fields[key] = v.strip()
    return fields


def relations(s):
    """`a (>= 1) | b, c:any` -> [[a, b], [c]]: names only, every alternative."""
    out = []
    for group in filter(None, (g.strip() for g in s.split(","))):
        alts = []
        for alt in group.split("|"):
            name = re.split(r"[\s(\[<]", alt.strip(), maxsplit=1)[0].split(":")[0]
            if name:
                alts.append(name)
        if alts:
            out.append(alts)
    return out


# ---- what the binaries need -----------------------------------------------------

def dynamic(path):
    """An x86-64 ELF's dynamic section, as the loader reads it: the sonames it
    needs, the symbol versions it requires of each (DT_VERNEED), and the
    versions it defines itself (DT_VERDEF). None if it is not such an ELF.
    Read here rather than through readelf, which is binutils and not always
    installed."""
    try:
        with open(path, "rb") as f:
            head = f.read(64)
            if len(head) < 64 or head[:4] != b"\x7fELF" or head[4] != 2 or head[5] != 1:
                return None
            if struct.unpack_from("<H", head, 18)[0] != 62:  # EM_X86_64
                return None
            phoff, = struct.unpack_from("<Q", head, 32)
            phentsize, phnum = struct.unpack_from("<HH", head, 54)
            f.seek(phoff)
            phdrs = [struct.unpack_from("<IIQQQQQQ", f.read(phentsize)) for _ in range(phnum)]
            loads = [(p[3], p[2], p[5]) for p in phdrs if p[0] == 1]  # vaddr, offset, filesz
            out = {"needed": [], "verneed": {}, "verdef": set()}

            def at(vaddr):
                for va, off, sz in loads:
                    if va <= vaddr < va + sz:
                        return off + vaddr - va
                return None

            dyn = next((p for p in phdrs if p[0] == 2), None)
            if not dyn:
                return out
            f.seek(dyn[2])
            raw = f.read(dyn[5])
            entries = []
            for i in range(0, len(raw) - 15, 16):
                t, v = struct.unpack_from("<qQ", raw, i)
                if t == 0:
                    break
                entries.append((t, v))
            tags = dict(entries)
            if 5 not in tags or at(tags[5]) is None:
                return out
            strtab = at(tags[5])

            def string(off):
                f.seek(strtab + off)
                return f.read(256).split(b"\0", 1)[0].decode("utf-8", "replace")

            out["needed"] = [string(v) for t, v in entries if t == 1]
            # Elf64_Verneed {u16 version, u16 cnt, u32 file, u32 aux, u32 next}
            # and Elf64_Vernaux {u32 hash, u16 flags, u16 other, u32 name, u32 next}.
            vn = at(tags[0x6FFFFFFE]) if 0x6FFFFFFE in tags else None
            for _ in range(tags.get(0x6FFFFFFF, 0) if vn is not None else 0):
                f.seek(vn)
                _, cnt, file_, aux, nxt = struct.unpack("<HHIII", f.read(16))
                vers, a = set(), vn + aux
                for _ in range(cnt):
                    f.seek(a)
                    _, _, _, name, anext = struct.unpack("<IHHII", f.read(16))
                    vers.add(string(name))
                    if not anext:
                        break
                    a += anext
                out["verneed"].setdefault(string(file_), set()).update(vers)
                if not nxt:
                    break
                vn += nxt
            # Elf64_Verdef {u16 version, u16 flags, u16 ndx, u16 cnt, u32 hash,
            # u32 aux, u32 next}; its first Verdaux {u32 name, u32 next} names it.
            vd = at(tags[0x6FFFFFFC]) if 0x6FFFFFFC in tags else None
            for _ in range(tags.get(0x6FFFFFFD, 0) if vd is not None else 0):
                f.seek(vd)
                _, flags, _, _, _, aux, nxt = struct.unpack("<HHHHIII", f.read(20))
                f.seek(vd + aux)
                name, = struct.unpack("<I", f.read(4))
                if not flags & 1:  # VER_FLG_BASE names the file, not a version
                    out["verdef"].add(string(name))
                if not nxt:
                    break
                vd += nxt
            return out
    except (OSError, struct.error):
        return None


def needed(path):
    """DT_NEEDED of an x86-64 ELF, or None if it is not one."""
    d = dynamic(path)
    return None if d is None else d["needed"]


def version_gaps(root, have):
    """Symbol versions the package's binaries require of libraries already
    installed that those libraries do not define -- the loader's own check,
    made before installing rather than at the first run. A soname matching
    is not enough: Debian's libncurses.so.6 and Arch's libncursesw.so.6 share
    an ABI and not their version names, and a binary built against a newer
    glibc than this one names the same libc.so.6 and still does not start."""
    gaps, defs = [], {}
    for d, _, files in os.walk(root):
        for fn in files:
            p = os.path.join(d, fn)
            if os.path.islink(p):
                continue
            dy = dynamic(p)
            if not dy:
                continue
            for soname, vers in dy["verneed"].items():
                lib = have.get(soname)
                if not lib:
                    continue
                if lib not in defs:
                    ld = dynamic(lib)
                    defs[lib] = ld["verdef"] if ld else set()
                missing = sorted(v for v in vers if v not in defs[lib])
                if missing:
                    gaps.append((soname, missing, os.path.relpath(p, root)))
    return gaps


def sonames_needed(root):
    """Every soname the package's binaries ask for that it does not ship itself."""
    want, own = {}, set()
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            if os.path.islink(p) or not os.path.isfile(p):
                if os.path.islink(p):
                    own.add(f)
                continue
            own.add(f)
            n = needed(p)
            for s in n or ():
                want.setdefault(s, os.path.relpath(p, root))
    return {s: by for s, by in want.items() if s not in own}


def installed_sonames():
    """What the loader can already find, from its own cache: soname -> path,
    x86-64 entries only (the cache lists 32-bit libraries under one name)."""
    out = {}
    for line in quiet(["ldconfig", "-p"]).stdout.split("\n")[1:]:
        line = line.strip()
        if not line or "x86-64" not in line or " => " not in line:
            continue
        out.setdefault(line.split(" ", 1)[0], line.rsplit(" => ", 1)[1])
    return out


def owners(paths):
    """The packages that own these files, as pacman records them."""
    paths = [os.path.realpath(p) for p in paths]
    if not paths:
        return set()
    return {o for o in quiet(["pacman", "-Qqo"] + paths).stdout.split() if o}


_files_synced = False


def provider_of(soname):
    """Which repository package ships `soname`, or None."""
    global _files_synced
    m = re.match(r"(.+\.so)\.(\d+)", soname)
    if m:
        r = quiet(["pacman", "-Sp", "--print-format", "%n", f"{m.group(1)}={m.group(2)}-64"])
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().split("\n")[-1]
    # The repositories' file lists are the last resort: they answer every
    # soname, and they are a 50 MB download the first time.
    if not _files_synced and not os.path.exists("/var/lib/pacman/sync/extra.files"):
        say("fetching the repositories' file lists, to find which packages ship the libraries it needs")
        run(["pacman", "-Fy"], stdout=subprocess.DEVNULL)
    _files_synced = True
    r = quiet(["pacman", "-F", "--machinereadable", f"usr/lib/{soname}"])
    for line in r.stdout.split("\n"):
        parts = line.split("\0")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return None


def arch_name(debname):
    """A Debian package name as an Arch one, when that is knowable."""
    if debname in NAMES:
        return NAMES[debname]
    tries = [debname]
    # The two naming rules the distributions share in spirit: Perl and
    # Python modules, spelled from the other end.
    m = re.fullmatch(r"lib(.+)-perl", debname)
    if m:
        tries.append("perl-" + m.group(1))
    m = re.fullmatch(r"python3-(.+)", debname)
    if m:
        tries.append("python-" + m.group(1))
    for t in tries:
        if aur.satisfied(t) or aur.in_repos(t):
            return t
    return False


# ---- the plan, and the package ---------------------------------------------------

def plan(deb):
    base = re.sub(r"[^A-Za-z0-9]+", "-", os.path.basename(deb))
    work = os.path.join(CACHE, base)
    ctl_dir, root = unpack(deb, work)
    c = control(ctl_dir)
    name = c.get("Package") or die(f"{deb} has no Package field")
    arch = c.get("Architecture", "")
    if arch not in ("amd64", "all"):
        die(f"{name} is built for {arch or 'nothing it names'}; this machine runs amd64 (and 'all')")
    deps, unmapped = set(), []
    missing, from_libs = [], {}
    have = installed_sonames()
    want = sonames_needed(root)
    on_disk = []
    binaries = any(needed(os.path.join(d, f)) is not None
                   for d, _, fs in os.walk(root) for f in fs if not os.path.islink(os.path.join(d, f)))
    for s, by in sorted(want.items()):
        if s not in have and os.path.exists("/usr/lib/" + s):
            # The loader's cache files a library under its own soname, so an
            # alias like Arch's libtinfo.so.6 is not in it; the loader still
            # finds it, by filename, in /usr/lib -- and so does this.
            have[s] = "/usr/lib/" + s
        if s in have:
            # Already here -- but still a dependency, through whichever
            # package owns it, or removing that package later would break
            # this one with nothing saying so.
            on_disk.append(have[s])
            continue
        p = provider_of(s)
        if p:
            deps.add(p)
            from_libs[s] = p
        else:
            missing.append((s, by))
    deps |= owners(on_disk)
    for soname, vers, by in version_gaps(root, have):
        missing.append((f"{soname} with {', '.join(vers)}", by))
    for group in relations(c.get("Pre-Depends", "") + "," + c.get("Depends", "")):
        got = None
        for alt in group:
            a = arch_name(alt)
            if a is None:  # Debian machinery with no counterpart, and no need of one
                got = ""
                break
            if a:
                got = a
                break
        if got is None:
            # A library package the binaries already answered for is not a
            # gap: the sonames are what it was there to supply.
            if want is not None and binaries and all(a.startswith("lib") and not a.endswith("-perl") for a in group):
                continue
            unmapped.append(" | ".join(group))
        elif got:
            deps.add(got)
    scripts = [s for s in ("preinst", "postinst", "prerm", "postrm", "config", "triggers") if os.path.exists(os.path.join(ctl_dir, s))]
    return dict(name=name, control=c, work=work, ctl=ctl_dir, root=root, deps=sorted(deps),
                unmapped=unmapped, missing=missing, from_libs=from_libs, scripts=scripts)


def version(v):
    """Debian `[epoch:]upstream[-revision]` as pacman's epoch and pkgver;
    pkgver may hold neither ':' nor '-', and '~' sorts differently."""
    epoch, _, rest = v.rpartition(":") if ":" in v else ("", "", v)
    up, _, rev = rest.rpartition("-") if "-" in rest else (rest, "", "")
    pv = re.sub(r"[^A-Za-z0-9._+]", "_", up + ("_" + rev if rev else ""))
    return (epoch if epoch.isdigit() else ""), pv or "0"


def rearrange(root):
    """Debian's layout into Arch's: the merged directories under /usr."""
    for top, dst in (("bin", "usr/bin"), ("sbin", "usr/bin"), ("usr/sbin", "usr/bin"),
                     ("lib", "usr/lib"), ("lib64", "usr/lib")):
        src = os.path.join(root, top)
        if os.path.isdir(src) and not os.path.islink(src):
            target = os.path.join(root, dst)
            os.makedirs(target, exist_ok=True)
            for e in os.listdir(src):
                if os.path.lexists(os.path.join(target, e)):
                    die(f"the package has both {top}/{e} and {dst}/{e}, which are one path here")
                shutil.move(os.path.join(src, e), target)
            os.rmdir(src)
    multi = os.path.join(root, MULTIARCH)
    return os.path.isdir(multi) and any(f.endswith(".so") or ".so." in f for f in os.listdir(multi))


def package(p):
    c = p["control"]
    epoch, pkgver = version(c.get("Version", "0"))
    name = re.sub(r"[^a-z0-9@._+-]", "-", p["name"].lower())
    multi = rearrange(p["root"])
    if multi:
        conf = os.path.join(p["root"], "etc/ld.so.conf.d")
        os.makedirs(conf, exist_ok=True)
        with open(os.path.join(conf, f"deb-{name}.conf"), "w") as f:
            f.write(f"/{MULTIARCH}\n")
    keep = os.path.join(p["root"], MARK.lstrip("/"), name)
    os.makedirs(keep, exist_ok=True)
    for s in ["control"] + p["scripts"]:
        shutil.copy(os.path.join(p["ctl"], s), keep)
    desc = (c.get("Description", "").split("\n")[0] or name).replace("'", "")
    home = c.get("Homepage", "").replace("'", "")
    build = os.path.join(p["work"], "build")
    os.makedirs(build, exist_ok=True)
    q = lambda xs: " ".join("'" + x.replace("'", "") + "'" for x in xs)
    with open(os.path.join(build, "PKGBUILD"), "w") as f:
        f.write(f"# Converted by orange from {c.get('Package')} {c.get('Version')} (a .deb).\n")
        f.write(f"pkgname={name}\npkgver={pkgver}\npkgrel=1\n")
        if epoch:
            f.write(f"epoch={epoch}\n")
        f.write(f"pkgdesc='{desc} (from a .deb)'\narch=('x86_64')\nurl='{home}'\nlicense=('custom')\n")
        f.write(f"depends=({q(p['deps'])})\n")
        # A .deb is already built and stripped by whoever made it, and its
        # files are what they shipped: makepkg must not strip, compress or
        # second-guess them.
        f.write("options=('!strip' '!debug' '!zipman' 'emptydirs' '!lto')\n")
        f.write(f"package() {{\n  cp -a '{p['root']}/.' \"$pkgdir/\"\n}}\n")
    os.chmod(p["work"], 0o755)
    for d, dirs, files in os.walk(p["work"]):
        for x in dirs:
            os.chmod(os.path.join(d, x), os.stat(os.path.join(d, x)).st_mode | 0o755)
        for x in files:
            fp = os.path.join(d, x)
            if not os.path.islink(fp):
                os.chmod(fp, os.stat(fp).st_mode | 0o444)
    os.chmod(build, 0o777)
    aur.ensure_builder()
    r = aur.as_builder(["makepkg", "--nodeps", "--noconfirm", "--force", "--nocheck"], build)
    if r.returncode != 0:
        die(f"{name} did not package (makepkg exited {r.returncode}); the converted tree is in {p['work']}")
    built = [os.path.join(build, f) for f in os.listdir(build) if f.endswith(".pkg.tar.zst")]
    if not built:
        die(f"makepkg made no package for {name}")
    return built[0]


def report(p):
    c = p["control"]
    say(f"{p['name']} {c.get('Version', '?')} ({c.get('Architecture')}) -- {c.get('Description', '').split(chr(10))[0]}")
    if c.get("Maintainer"):
        say(f"packaged by {c['Maintainer']}; nothing here vouches for it")
    if p["deps"]:
        say("needs, from the repositories: " + " ".join(p["deps"]))
    for s, by in p["missing"]:
        if " with " in s:
            lib, _, vers = s.partition(" with ")
            say(f"{by} needs {lib} at symbol version {vers}, and the {lib} here does not define it")
        else:
            say(f"no package found that ships {s} (wanted by {by})")
    if p["unmapped"]:
        say("Debian dependencies with no Arch counterpart found, not required: " + ", ".join(p["unmapped"]))
    if p["scripts"]:
        say(f"its maintainer scripts ({', '.join(p['scripts'])}) are not run; they are kept in {MARK}/{p['name']}/")


def install(debs, force):
    for deb in debs:
        if not os.path.isfile(deb):
            die(f"{deb} is not a file")
        p = plan(deb)
        report(p)
        if p["missing"] and not force:
            die(f"{p['name']} needs libraries this system does not have, or not in the versions it was "
                f"built against, so it would not start -- '--force' installs it anyway")
        repo = [d for d in p["deps"] if not aur.satisfied(d)]
        if repo:
            r = run(["pacman", "-S", "--needed", "--asdeps", "--noconfirm"] + repo)
            if r.returncode != 0:
                die("its dependencies did not install")
        pkg = package(p)
        r = run(["pacman", "-U", "--noconfirm"] + (["--nodeps"] if force and p["missing"] else []) + [pkg])
        if r.returncode != 0:
            die(f"{p['name']} converted but did not install -- pacman says why above")
        shutil.rmtree(p["work"], ignore_errors=True)
        say(f"installed {p['name']}; 'orange remove {p['name']}' takes it away")


def main(argv):
    force = "--force" in argv
    argv = [a for a in argv if a not in ("--force", "--yes", "-y")]
    if not argv or argv[0] in ("help", "-h", "--help"):
        print(__doc__.split("\n\n")[2])
        return
    verb, args = argv[0], argv[1:]
    if verb == "install":
        install(args or die("install which .deb?"), force)
    elif verb == "show":
        for deb in args or die("show which .deb?"):
            p = plan(deb)
            report(p)
            for s, pk in sorted(p["from_libs"].items()):
                print(f"    {s:<32} {pk}")
            shutil.rmtree(p["work"], ignore_errors=True)
    elif verb == "list":
        if os.path.isdir(MARK):
            for n in sorted(os.listdir(MARK)):
                r = quiet(["pacman", "-Q", n])
                if r.returncode == 0:
                    print(r.stdout.strip())
    else:
        die(f"'{verb}' is not a deb command -- 'orange deb help'")


if __name__ == "__main__":
    main(sys.argv[1:])
