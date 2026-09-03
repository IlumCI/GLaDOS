"""Generate `src/doom/info.rs` from a room4doom checkout.

DOOM's `info.c` is the game's content: every sprite frame, how long it shows,
what it becomes next, and for each kind of thing where its animation starts and
how much of it there is. Nine hundred and sixty-seven states and a hundred and
thirty-seven thing types, which upstream carries as **341 KB** of Rust.

**Generated rather than vendored, and rather than hand-written.** Three
reasons, in the order they matter:

  * A table that size cannot be written by hand without errors that look
    exactly like correct entries. A wrong `next_state` is a monster whose
    walk cycle skips a frame; a wrong `tics` is one that limps. Neither
    raises anything.
  * It is *derived*, so it can be re-derived. `tools/rtlconv.py` makes the
    same argument about the RTL8188EU tables and `tools/font.py` about the
    glyphs -- this repository already prefers a generator and a provenance
    line to a large opaque file.
  * Upstream generates it too, from DOOM's own `multigen` input. Arriving at
    the same conclusion independently is worth something.

What is carried and what is dropped:

  * Sounds are dropped. There is no audio driver on this machine and none is
    planned, so `SfxName` fields would be a table nothing could read.
  * `misc1`/`misc2` are dropped. They are used by the weapon states this port
    does not have yet, and carrying a field nothing reads invites somebody to
    trust it.
  * Actions are kept as a **numbered id**, not a function pointer. Nothing
    dispatches them yet; the id is what a later `A_Chase` will match on, and
    keeping the number now means the table does not have to be regenerated
    when it does.

    .\\tools\\venv\\Scripts\\python.exe tools\\doominfo.py out\\room4doom\\room4doom-main
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def enum_names(src, name):
    """The variants of a bare enum, in declaration order.

    Order *is* the value: every index in the tables below is a position in
    these lists, so a variant read out of order would silently renumber
    everything after it.
    """
    m = re.search(r"pub enum %s \{(.*?)\n\}" % name, src, re.S)
    if not m:
        raise SystemExit("no enum %s" % name)
    body = re.sub(r"//.*", "", m.group(1))
    out = []
    for part in body.split(","):
        part = part.strip()
        if not part or "=" in part:
            continue
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part):
            continue
        out.append(part)
    return out


def field(block, name, default=None):
    m = re.search(r"\b%s:\s*([^,\n]+)," % name, block)
    if not m:
        if default is not None:
            return default
        raise SystemExit("field %s missing in %s" % (name, block[:80]))
    return m.group(1).strip()


def enum_ref(value):
    """`StateNum::POSS_RUN2` -> `POSS_RUN2`."""
    return value.split("::")[-1].strip()


def parse_states(src, states, sprites, actions):
    """The state table: one row per animation frame."""
    idx = {n: i for i, n in enumerate(states)}
    sidx = {n: i for i, n in enumerate(sprites)}
    aidx = {n: i for i, n in enumerate(actions)}
    m = re.search(r"pub static STATES: \[StateData; NUM_STATES\] = \[(.*?)\n\];", src, re.S)
    if not m:
        raise SystemExit("no STATES array")
    rows = []
    for block in re.findall(r"StateData \{(.*?)\n    \},", m.group(1), re.S):
        sprite = enum_ref(field(block, "sprite"))
        nxt = enum_ref(field(block, "next_state"))
        act = enum_ref(field(block, "action"))
        rows.append(
            (
                sidx[sprite],
                int(field(block, "frame")),
                int(field(block, "tics")),
                aidx.get(act, 0),
                idx[nxt],
            )
        )
    return rows


def parse_kinds(src, states):
    """The thing table: one row per kind of object a map can place."""
    idx = {n: i for i, n in enumerate(states)}
    m = re.search(r"pub const MOBJINFO: \[MapObjInfo; NUM_CATEGORIES\] = \[(.*?)\n\];", src, re.S)
    if not m:
        raise SystemExit("no MOBJINFO array")
    rows = []
    for block in re.findall(r"MapObjInfo \{(.*?)\n    \},", m.group(1), re.S):
        def st(name):
            return idx[enum_ref(field(block, name))]

        # `flags` is a bitflags expression; the individual names are what
        # matter and they are recovered by scanning rather than evaluated.
        flags = re.findall(r"MapObjFlag::([A-Za-z0-9]+)\.bits\(\)", field(block, "flags"))
        rows.append(
            {
                "doomednum": int(field(block, "doomednum")),
                "spawn": st("spawnstate"),
                "see": st("seestate"),
                "pain": st("painstate"),
                "melee": st("meleestate"),
                "missile": st("missilestate"),
                "death": st("deathstate"),
                "xdeath": st("xdeathstate"),
                "raise": st("raisestate"),
                "health": int(field(block, "spawnhealth")),
                "reaction": int(field(block, "reactiontime")),
                "painchance": int(field(block, "painchance")),
                "speed": int(field(block, "speed")),
                "radius": int(float(field(block, "radius"))),
                "height": int(float(field(block, "height"))),
                "mass": int(field(block, "mass")),
                "damage": int(field(block, "damage")),
                "flags": flags,
            }
        )
    return rows


def parse_flags(src):
    """The flag names and values, read out of upstream's `bitflags` block.

    Hand-written first, and wrong twice in ways that are worth recording
    because they are the same mistake in two costumes. `NotDeathmatch` is
    spelled `Notdmatch` upstream, so the constant emitted here and the
    constant referred to by `MOBJINFO` were different identifiers and the
    table would not compile -- which is the *good* failure. `Translation` is
    `0xC000000`, a two-bit colour field and not a bit at all, so a positional
    list assigning it bit 26 produced a constant that compiled perfectly and
    meant something else.

    Reading name and value from the declaration makes both impossible.
    """
    m = re.search(r"pub struct MapObjFlag: u32 \{(.*?)\n    \}", src, re.S)
    if not m:
        raise SystemExit("no MapObjFlag bitflags block")
    out = []
    for name, value in re.findall(r"const ([A-Za-z0-9_]+) = (0x[0-9A-Fa-f]+|\d+);", m.group(1)):
        out.append((name, int(value, 0)))
    if not out:
        raise SystemExit("MapObjFlag block has no constants")
    return out


def action_const(name):
    """`ABfgspray` -> `A_BFGSPRAY`.

    Upstream spells the variants in Rust case; DOOM spells them `A_BFGSpray`.
    Neither is a good constant name, so both are flattened the same way and
    the flattening is total -- there is no per-name exception list, for the
    reason `eval.rs` gives about builtin naming: one exception means every
    name has to be checked against a list again.
    """
    return "A_" + name[1:].upper()


def emit(sprites, states, kinds, flags, actions, origin):
    f = []
    w = f.append
    w("// GENERATED by tools/doominfo.py from room4doom (MIT, Luke Jones).")
    w("//   https://github.com/flukejones/room4doom")
    w("//")
    w("// **Do not edit this file.** Regenerate it:")
    w("//")
    w("//     tools\\\\venv\\\\Scripts\\\\python.exe tools\\\\doominfo.py <room4doom checkout>")
    w("//")
    w("// This is DOOM's content rather than its code: every sprite frame, how")
    w("// long it shows, what it becomes next, and per kind of thing where its")
    w("// animation starts. Nine hundred states and a hundred and thirty-seven")
    w("// kinds is not a table anybody writes by hand correctly -- a wrong")
    w("// `next` is a walk cycle that skips a frame and a wrong `tics` is a")
    w("// monster that limps, and neither raises anything.")
    w("//")
    w("// Sounds are absent because this machine has no audio driver and none is")
    w("// planned. `misc1`/`misc2` are absent because only the weapon states use")
    w("// them and carrying a field nothing reads invites somebody to trust it.")
    w("")
    w("// Most of this is not read yet, and the allow is the point rather than a")
    w("// concession. A generated table is complete on purpose: emitting only the")
    w("// fields something consumes today would mean regenerating it every time a")
    w("// consumer arrives, which is the opposite of why it is generated. Left")
    w("// alone it produces about 140 dead-code warnings, and a build with 140")
    w("// warnings in it is a build where the one that matters is not read.")
    w("//")
    w("// Note this is the *whole* file, so a genuinely unused item added here by")
    w("// hand would also be silent -- which is another reason not to edit it by")
    w("// hand.")
    w("#![allow(dead_code)]")
    w("")
    w("/// What each action id is called upstream, indexed by `State::action`.")
    w("///")
    w("/// The ids alone are unreadable, and a dispatcher written against bare")
    w("/// numbers is a dispatcher nobody can check. Emitting the names beside")
    w("/// them costs a kilobyte and means a `match` can be read against DOOM's")
    w("/// own source.")
    w("pub const ACTIONS: [&str; %d] = [" % len(actions))
    for i in range(0, len(actions), 8):
        w("    " + " ".join('"%s",' % a for a in actions[i:i + 8]))
    w("];")
    w("")
    w("// One constant per action, so a dispatcher names what it is dispatching")
    w("// and a renumbering upstream cannot silently point it at another.")
    for i, a in enumerate(actions):
        if a == "None":
            continue
        w("pub const %s: u8 = %d;" % (action_const(a), i))
    w("")
    w("/// The four-character sprite names, indexed by `State::sprite`.")
    w("pub const SPRITES: [&str; %d] = [" % len(sprites))
    for i in range(0, len(sprites), 10):
        w("    " + " ".join('"%s",' % s for s in sprites[i : i + 10]))
    w("];")
    w("")
    w("/// One frame of an animation.")
    w("pub struct State {")
    w("    /// Index into `SPRITES`.")
    w("    pub sprite: u16,")
    w("    /// Which letter of that sprite, `A` being 0. The top bit is DOOM's")
    w("    /// fullbright flag, which says the frame ignores sector light.")
    w("    pub frame: u16,")
    w("    /// How many tics it shows for. **-1 means forever** -- a barrel is")
    w("    /// one state with -1 tics, which is how a thing that never animates")
    w("    /// and a thing mid-animation are the same mechanism.")
    w("    pub tics: i16,")
    w("    /// What it does on arriving here. Nothing dispatches these yet; the")
    w("    /// number is what a later `A_Chase` matches on.")
    w("    pub action: u8,")
    w("    /// The state after this one. Chains loop -- a walk cycle is four")
    w("    /// states whose last names the first.")
    w("    pub next: u16,")
    w("}")
    w("")
    w("pub const STATES: [State; %d] = [" % len(states))
    for sprite, frame, tics, action, nxt in states:
        w(
            "    State { sprite: %d, frame: %d, tics: %d, action: %d, next: %d },"
            % (sprite, frame, tics, action, nxt)
        )
    w("];")
    w("")
    w("// The flags, name and value both taken from upstream's own `bitflags`")
    w("// block. `MF_TRANSLATION` is a two-bit colour field rather than a bit,")
    w("// which is the reason these are not `1 << position`.")
    for name, value in flags:
        w("/// DOOM's `MF_%s`." % name.upper())
        w("pub const MF_%s: u32 = 0x%X;" % (name.upper(), value))
    w("")
    w("/// One kind of thing a map can place.")
    w("pub struct Kind {")
    w("    /// What a THING record calls it, or -1 for something only spawned")
    w("    /// by the game -- a rocket has no number because no map places one.")
    w("    pub doomednum: i16,")
    w("    pub spawn: u16,")
    w("    pub see: u16,")
    w("    pub pain: u16,")
    w("    pub melee: u16,")
    w("    pub missile: u16,")
    w("    pub death: u16,")
    w("    pub xdeath: u16,")
    w("    pub raise: u16,")
    w("    pub health: i16,")
    w("    pub reaction: i16,")
    w("    pub painchance: i16,")
    w("    /// Units per tic for anything that walks, and **fixed point** for")
    w("    /// anything that flies.")
    w("    ///")
    w("    /// One field with two units in it, which is DOOM's own and not an")
    w("    /// artefact of this generator: a monster's speed is an integer")
    w("    /// number of units it tries to move, while a missile's is a")
    w("    /// `fixed_t` added to its position directly, so a rocket reads")
    w("    /// 655360 where an imp reads 8. That is 10 << 16, and the reason")
    w("    /// this is `i32` -- 655360 does not fit in the `i16` every other")
    w("    /// number here fits in, which is the only reason anybody notices.")
    w("    /// A caller that reads a missile's speed as units per tic gets a")
    w("    /// rocket travelling ten thousand times too fast.")
    w("    pub speed: i32,")
    w("    pub radius: i16,")
    w("    pub height: i16,")
    w("    /// What damage has to shift to knock this back.")
    w("    ///")
    w("    /// A divisor rather than a weight, so ten million is not an outlier")
    w("    /// to be clamped -- it is how DOOM spells *immovable*, and Commander")
    w("    /// Keen and the boss brain both carry it because they hang in place")
    w("    /// and being shot must not move them. It is the second field here")
    w("    /// that will not fit in an `i16`, and for a completely different")
    w("    /// reason from the first.")
    w("    pub mass: i32,")
    w("    pub damage: i16,")
    w("    pub flags: u32,")
    w("}")
    w("")
    w("pub const KINDS: [Kind; %d] = [" % len(kinds))
    for k in kinds:
        bits = " | ".join("MF_%s" % n.upper() for n in k["flags"]) or "0"
        w("    Kind {")
        w("        doomednum: %d," % k["doomednum"])
        for name in ("spawn", "see", "pain", "melee", "missile", "death", "xdeath", "raise"):
            w("        %s: %d," % (name, k[name]))
        for name in ("health", "reaction", "painchance", "speed", "radius", "height", "mass", "damage"):
            w("        %s: %d," % (name, k[name]))
        w("        flags: %s," % bits)
        w("    },")
    w("];")
    w("")
    w("/// Which kind carries a doomednum, or nothing.")
    w("///")
    w("/// A linear scan, because it runs once per *distinct* thing kind when a")
    w("/// level loads and never per frame. DOOM does the same.")
    w("pub fn kind_of(doomednum: i16) -> Option<&'static Kind> {")
    w("    KINDS.iter().find(|k| k.doomednum == doomednum)")
    w("}")
    w("")
    w("/// Which row a doomednum is, for a caller that wants to remember the")
    w("/// answer rather than the number.")
    w("///")
    w("/// An object stores its row and not its doomednum: the row is what")
    w("/// indexes this table, and several kinds share a doomednum of -1 because")
    w("/// no map places them -- so a doomednum is not an identity and a row is.")
    w("pub fn row_of(doomednum: i16) -> Option<u16> {")
    w("    KINDS.iter().position(|k| k.doomednum == doomednum).map(|i| i as u16)")
    w("}")
    w("")
    w("/// The sprite name and frame letter a state shows, or nothing.")
    w("///")
    w("/// **State 0 is  and draws nothing.** It is DOOM's no-state, and")
    w("/// several kinds sit in it permanently -- a teleport destination is a")
    w("/// real thing with a real doomednum whose whole job is to be a marker.")
    w("/// The row is not blank though:  carries ")
    w("/// because the array needs *something* there, so a reader that trusted")
    w("/// the fields would draw an imp on every teleport pad in the game.")
    w("pub fn frame_of(state: u16) -> Option<(&'static str, u8)> {")
    w("    if state == 0 {")
    w("        return None;")
    w("    }")
    w("    let s = STATES.get(state as usize)?;")
    w("    // The top bit is fullbright and is not part of the letter. What is")
    w("    // left is a frame *number*, and the lump is named by the letter at")
    w("    // that offset from `A` -- which past 25 leaves the alphabet and")
    w("    // carries on into `[`, `\\` and `]`, exactly as id's own names do.")
    w("    let letter = b'A' + (s.frame & 0x7FFF) as u8;")
    w("    Some((SPRITES.get(s.sprite as usize)?, letter))")
    w("}")
    w("")
    w("/// Whether a state draws at full brightness whatever the room is lit to.")
    w("///")
    w("/// A muzzle flash, a plasma bolt, an explosion. Nothing reads it yet --")
    w("/// it is one bit of the field `frame_of` masks off, and dropping it here")
    w("/// would mean regenerating the table when a projectile first needs it.")
    w("pub fn fullbright(state: u16) -> bool {")
    w("    STATES.get(state as usize).is_some_and(|s| s.frame & 0x8000 != 0)")
    w("}")
    w("")
    w("/// Where this generated table came from.")
    w("pub const ORIGIN: &str = \"%s\";" % origin)
    return "\n".join(f) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkout", help="a room4doom checkout")
    ap.add_argument("--out", default=str(ROOT / "src" / "doom" / "info.rs"))
    args = ap.parse_args()

    src_path = Path(args.checkout) / "gameplay" / "src" / "info.rs"
    if not src_path.exists():
        raise SystemExit("no %s" % src_path)
    src = src_path.read_text(encoding="utf-8", errors="replace")

    sprites = enum_names(src, "SpriteNum")
    state_names = enum_names(src, "StateNum")
    actions = enum_names(src, "ActionId")
    # Both enums end with a count sentinel that is not a real entry.
    for lst in (sprites, state_names):
        if lst and lst[-1] in ("Count", "NUMSPRITES", "NUMSTATES"):
            lst.pop()

    states = parse_states(src, state_names, sprites, actions)
    kinds = parse_kinds(src, state_names)

    # The flags live in a different file from the table that uses them.
    flags_path = Path(args.checkout) / "gameplay" / "src" / "thing" / "mod.rs"
    if not flags_path.exists():
        raise SystemExit("no %s" % flags_path)
    flags = parse_flags(flags_path.read_text(encoding="utf-8", errors="replace"))

    # Every flag `MOBJINFO` names must be one the declaration declares. This is
    # the check that would have caught the hand-written list, and it costs two
    # lines: a name in the table with no constant behind it is a file that does
    # not compile, which is fine, but a name that *nearly* matches one is a
    # constant with the wrong value in it, which is not.
    declared = {n for n, _ in flags}
    for i, k in enumerate(kinds):
        for f in k["flags"]:
            if f not in declared:
                raise SystemExit("kind %d has flag %s, which is not declared" % (i, f))

    # The tables index each other, so a miscount is not a smaller table -- it
    # is a table where every row after the mistake means something else.
    if len(states) != len(state_names):
        raise SystemExit(
            "%d states parsed but %d names: the array and the enum disagree"
            % (len(states), len(state_names))
        )
    for i, (sprite, _f, _t, _a, nxt) in enumerate(states):
        if sprite >= len(sprites):
            raise SystemExit("state %d names sprite %d of %d" % (i, sprite, len(sprites)))
        if nxt >= len(states):
            raise SystemExit("state %d chains to %d of %d" % (i, nxt, len(states)))
    for i, k in enumerate(kinds):
        if k["spawn"] >= len(states):
            raise SystemExit("kind %d spawns at state %d of %d" % (i, k["spawn"], len(states)))

    out = Path(args.out)
    out.write_text(
        emit(sprites, states, kinds, flags, actions, "room4doom gameplay/src/info.rs"),
        encoding="utf-8",
    )
    print(
        "%s  %d sprites, %d states, %d kinds, %d actions"
        % (out, len(sprites), len(states), len(kinds), len(actions))
    )


if __name__ == "__main__":
    sys.exit(main())
