#!/usr/bin/env python3
"""Generate a tool-selection dataset for the sysbox applets.

The seed corpus was 40 examples across 21 classes -- under two per class, which
is why every gradient step made generalisation worse. This produces a few
thousand, but the count is the least interesting property of it.

What matters is the split. Each applet gets several *template families*, and
families are held out whole rather than instances being sampled at random. If
the split were by instance, a test item would be a near-duplicate of something
seen in training -- "list the files" against "list the file" -- and the score
would measure memorisation while looking like generalisation. Holding out
families means the test phrasings are structurally unlike anything trained on,
which is the only version of this number worth reporting.

Confusable pairs are included deliberately. snap and snaps differ by one
character; stat and ls both describe things; same and diff both compare. A
dataset that avoided those would score well and teach nothing.
"""

import argparse
import json
import random
from pathlib import Path

# Each family is (template, [slot values]). Slots are filled with {x}.
# Families are ordered roughly from most to least literal, so the held-out
# tail tends to be the harder, more oblique phrasings.
#
# Widened from measurement rather than by guesswork. tools/analyse.py fitted
# the probe and reported: the learning curve still climbing steeply at 174
# examples (26% -> 34% -> 47% -> 57% across quarters, nowhere near flat), so
# more data was the lever rather than better features, which bought under two
# points. And six classes at zero recall -- cat, cp, diff, mv, snaps, sysbox --
# with a clear pattern in the confusions:
#
#   snaps -> ls x3, sysbox -> ls x3, pwd -> ls
#     `ls` was a magnet. It had 13 training examples where others had 6, and
#     "list"/"show" phrasing that generalises over everything. Hence balancing,
#     and hence the weak classes below leaning on vocabulary `ls` never uses.
#   cp -> mv x2
#     Genuinely close. The distinguishing idea is whether the original survives.
#   cat -> write, diff -> back
#     Same objects, opposite direction: reading against creating, inspecting a
#     change against undoing one.
#
FAMILIES = {
    # Leans on capability words -- tools, commands, applets, able -- and never
    # on "files", which is what collapsed it into `ls`.
    "sysbox": [
        ("what tools are available", []),
        ("list the {x}", ["commands", "applets", "available tools", "verbs"]),
        ("what can you {x}", ["do", "run", "execute", "help with"]),
        ("show me the {x}", ["command list", "tool list", "menu of options"]),
        ("what {x} do you have", ["capabilities", "commands", "features"]),
        ("which {x} can i use", ["tools", "commands", "operations"]),
        ("print the list of {x}", ["applets", "supported commands"]),
        ("what are you {x} of", ["capable", "able to do"]),
        ("give me the {x} of everything you can do", ["rundown", "summary", "index"]),
    ],
    "ls": [
        ("list the files", []),
        ("what is in {x}", ["this directory", "the folder", "here", "/ai"]),
        ("show me the contents of {x}", ["the directory", "this folder", "/tmp"]),
        ("{x} the directory listing", ["print", "display", "give me"]),
        ("what files are {x}", ["here", "in this folder", "present"]),
        ("what is {x} in this folder", ["stored", "sitting", "kept"]),
        ("show the {x} of this directory", ["entries", "immediate contents", "members"]),
        ("which files {x} here", ["live", "exist"]),
        ("enumerate the {x} in this folder", ["files", "entries"]),
    ],
    # Against `back`: moves through the namespace, never through history.
    "cd": [
        ("change directory to {x}", ["/ai", "/tmp", "the parent", "/sys"]),
        ("go to {x}", ["the other folder", "/ai/train", "the root", "the parent directory"]),
        ("move into {x}", ["the subdirectory", "/tmp", "that folder"]),
        ("switch to {x}", ["the /sys directory", "another folder"]),
        ("navigate to {x}", ["/ai", "the top level"]),
        ("take me {x}", ["into that folder", "up one level", "to the root"]),
        ("enter {x}", ["the subdirectory", "that folder"]),
        ("set the working directory to {x}", ["/tmp", "that path"]),
    ],
    # Against `ls`: about location, not contents. Never asks what is there.
    "pwd": [
        ("where am i", []),
        ("what directory am i {x}", ["in", "currently in", "sitting in"]),
        ("print the {x}", ["working directory", "current path", "current location"]),
        ("show my {x}", ["current directory", "location", "position in the tree"]),
        ("which folder am i {x} right now", ["in", "standing in"]),
        ("what is my {x}", ["current path", "present location", "cwd"]),
        ("tell me where i {x}", ["am", "have ended up"]),
        ("what path am i {x}", ["at", "on"]),
    ],
    # Against `ls`: every phrasing carries depth -- recursive, nested, all the
    # way down -- because without it this is indistinguishable from a listing.
    "tree": [
        ("list everything recursively", []),
        ("show the {x}", ["whole hierarchy", "entire tree", "full structure"]),
        ("what is under {x} including subdirectories", ["/ai", "this folder", "the root"]),
        ("give me a {x} view", ["recursive", "nested", "full tree"]),
        ("show everything {x}", ["all the way down", "including subfolders", "at every depth"]),
        ("list the files and {x} too", ["all subdirectories", "everything nested"]),
        ("walk the {x}", ["whole tree", "entire hierarchy", "directory tree"]),
        ("expand {x} completely", ["every folder", "the whole structure"]),
    ],
    # Against `write`: this is reading something that already exists. Every
    # phrasing is about getting text *out*, never about producing it.
    "cat": [
        ("read {x}", ["the readme", "the file", "/sys/readme", "that note"]),
        ("print the contents of {x}", ["a file", "the readme", "/ai/notes"]),
        ("show me what {x} says", ["that file", "the readme", "this document"]),
        ("{x} the text in that file", ["display", "dump", "output"]),
        ("what does {x} contain", ["that file", "the readme", "this note"]),
        ("let me see {x} of that file", ["the text", "the body", "the contents"]),
        ("open {x} and show it to me", ["the file", "that note"]),
        ("{x} that file to the screen", ["print", "echo", "spill"]),
        ("i want to read {x}", ["what is in there", "that document"]),
    ],
    # Against `ls`: one named object, not a folder's worth.
    "stat": [
        ("how big is {x}", ["that file", "this object", "/ai/notes"]),
        ("give me details about {x}", ["a path", "this file", "that directory"]),
        ("what are the {x} of this file", ["properties", "attributes", "metadata"]),
        ("tell me about {x}", ["that object", "this path"]),
        ("what {x} does that file have", ["size", "kind", "attributes"]),
        ("inspect {x}", ["this path", "that entry", "the object"]),
        ("describe {x} to me", ["this file", "that directory"]),
        ("is {x} a file or a directory", ["that", "this path"]),
    ],
    "hash": [
        ("what is the {x} of that object", ["address", "hash", "content hash", "digest"]),
        ("show me the content address", []),
        ("print the {x} for this file", ["hash", "checksum", "digest"]),
        ("identify that object by {x}", ["hash", "address"]),
        ("what {x} does this resolve to", ["address", "hash"]),
        ("give me the {x} string", ["sha", "digest", "hash"]),
        ("how is that object {x}", ["addressed", "identified", "fingerprinted"]),
        ("fingerprint {x}", ["this file", "that subtree"]),
    ],
    "same": [
        ("are these two {x}", ["identical", "the same", "equal"]),
        ("compare {x}", ["two subtrees", "these files", "the two directories"]),
        ("check whether {x} match", ["these paths", "the two objects", "both files"]),
        ("do {x} have the same contents", ["these", "both files", "the two folders"]),
        ("is {x} a duplicate of the other", ["this one", "that file"]),
        ("tell me if {x} are equal", ["these two", "both paths"]),
        ("verify that {x} are the same object", ["these", "both of them"]),
        ("did {x} end up identical", ["they", "both copies"]),
    ],
    "du": [
        ("how much space is {x}", ["used", "taken up", "consumed"]),
        ("report {x}", ["disk usage", "space used", "storage consumption"]),
        ("what is the {x} of everything here", ["total size", "footprint", "byte count"]),
        ("show {x} statistics", ["usage", "size", "storage"]),
        ("how many {x} does this take", ["bytes", "megabytes", "blocks"]),
        ("how much is {x} to store all this", ["needed", "required"]),
        ("what is my {x}", ["storage footprint", "total usage"]),
        ("how much of the {x} is in use", ["disk", "store", "space"]),
    ],
    "find": [
        ("search for {x}", ["some text", "a word", "the phrase hello"]),
        ("look for {x} in the files", ["a string", "that word", "some text"]),
        ("which files {x} this word", ["contain", "mention", "include"]),
        ("{x} the tree for that phrase", ["grep", "scan", "search"]),
        ("where does the word {x} appear", ["hello", "model", "that"]),
        ("hunt for {x} anywhere in here", ["that string", "the phrase"]),
        ("locate {x} by its content", ["files", "anything"]),
        ("track down {x} mentioning that", ["files", "anything"]),
    ],
    # Against `back`: inspecting a change, not undoing one. Nothing here asks
    # for the old state to be reinstated.
    "diff": [
        ("compare {x}", ["two snapshots", "snapshot 2 and 3", "the versions"]),
        ("what changed between {x}", ["versions", "the two snapshots", "then and now"]),
        ("show me the {x} between snapshots", ["difference", "delta", "changes"]),
        ("what is different {x}", ["since the last snapshot", "between these two states"]),
        ("which files {x} since then", ["changed", "were modified", "differ"]),
        ("tell me what {x} between those checkpoints", ["moved", "differs", "is not the same"]),
        ("summarise the {x} from 2 to 3", ["changes", "edits", "differences"]),
        ("did anything {x} between the snapshots", ["change", "differ"]),
    ],
    # Against `ls`: the objects are snapshots and checkpoints, never files. The
    # word "list" is deliberately rare here.
    "snaps": [
        ("list the snapshots", []),
        ("show {x}", ["snapshot history", "the checkpoints", "all the saved states"]),
        ("what {x} exist", ["snapshots", "checkpoints", "saved versions"]),
        ("give me the {x} of commits", ["history", "log"]),
        ("what {x} have i taken", ["snapshots", "checkpoints"]),
        ("how many {x} are there", ["snapshots", "checkpoints", "saved versions"]),
        ("print the {x} log", ["snapshot", "checkpoint", "commit"]),
        # "what versions can i go back to" used to live here, and it produced
        # snaps -> back three times out of three. It asks to enumerate, but
        # every content word in it belongs to `back`. Naming the enumeration
        # explicitly is the fix.
        ("enumerate the {x} available to restore", ["versions", "snapshots"]),
        ("index the {x} taken so far", ["checkpoints", "snapshots"]),
    ],
    "fsck": [
        ("verify the disk", []),
        ("check that everything is {x}", ["intact", "consistent", "not corrupted"]),
        ("{x} every stored object", ["verify", "validate", "check"]),
        ("is the store {x}", ["healthy", "consistent", "undamaged"]),
        ("make sure nothing is {x}", ["corrupt", "damaged", "broken"]),
        ("audit the {x} for damage", ["store", "disk", "objects"]),
        ("does everything still {x} its address", ["match", "verify against"]),
        ("run an {x} check on the store", ["integrity", "consistency"]),
    ],
    # Against `write`: makes a container, never content.
    "mkdir": [
        ("make a directory", []),
        ("create a {x}", ["new folder", "directory called work", "subdirectory"]),
        ("add a {x} named notes", ["folder", "directory"]),
        ("set up a {x} for this", ["new directory", "folder"]),
        ("i need a {x} to put things in", ["folder", "directory"]),
        ("start a new {x} here", ["folder", "directory"]),
        ("make me somewhere to {x} these", ["keep", "store", "put"]),
        ("establish a {x} at that path", ["directory", "folder"]),
    ],
    # Against `mkdir`: makes content, never a container.
    "write": [
        ("save some text to a file", []),
        ("create a file {x}", ["with this content", "containing hello", "and put text in it"]),
        ("write {x} into a new file", ["this note", "some text", "a message"]),
        ("store this {x} as a file", ["text", "content", "message"]),
        ("put the words {x} into a file", ["hello there", "this note"]),
        ("record this {x} somewhere on disk", ["text", "message", "line"]),
        ("commit this {x} to a file", ["string", "note", "content"]),
        ("make a file whose contents are {x}", ["this text", "that message"]),
    ],
    "rm": [
        ("delete {x}", ["that name", "the file", "this entry"]),
        ("remove {x}", ["the file", "that path", "this directory entry"]),
        ("get rid of {x}", ["that file", "this name"]),
        ("unlink {x}", ["the entry", "that path"]),
        ("i do not want {x} any more", ["that file", "this entry"]),
        ("drop {x} from the directory", ["that name", "this file"]),
        ("erase {x} from the listing", ["that entry", "this name"]),
        ("take {x} out of the tree", ["that file", "this path"]),
    ],
    # cp against mv is the closest pair in the table. The distinguishing idea
    # is whether the original survives, so mv phrasings stress that the old
    # name goes away and cp phrasings stress that a second one appears.
    "mv": [
        ("rename {x}", ["that file", "this directory", "the note"]),
        ("move {x} somewhere else", ["that file", "this folder"]),
        ("change the {x} of that file", ["name", "path", "location"]),
        ("relocate {x}", ["this entry", "that object"]),
        ("give {x} a different name", ["that file", "this folder"]),
        ("put {x} under another name instead", ["it", "that entry"]),
        ("{x} it to the new location", ["move", "shift", "transfer"]),
        ("call {x} something else from now on", ["that file", "this directory"]),
    ],
    "cp": [
        ("copy {x}", ["that file", "this directory", "the whole folder"]),
        ("duplicate {x}", ["this object", "that subtree", "the file"]),
        ("make a copy of {x}", ["this", "that directory", "the notes"]),
        ("clone {x} to another name", ["this folder", "that file"]),
        ("make a {x} of that file", ["second copy", "backup", "spare"]),
        ("i want {x} of this, keeping the original", ["another one", "a copy"]),
        ("replicate {x} elsewhere", ["that directory", "this file"]),
        ("leave the original and {x} it", ["copy", "duplicate"]),
    ],
    # Against `snaps`: creates one, never enumerates them.
    "snap": [
        ("take a snapshot", []),
        ("commit the {x}", ["current state", "working tree", "changes"]),
        ("save {x} as a checkpoint", ["everything", "the current state"]),
        ("record the {x} now", ["state", "current tree"]),
        ("checkpoint {x} where it is", ["everything", "the tree"]),
        ("freeze the {x} as it stands", ["current state", "tree"]),
        ("pin {x} so i can return to it", ["this moment", "the current state"]),
        ("make a {x} of right now", ["restore point", "checkpoint"]),
    ],
    # Against `diff` and `snaps`: reinstates an old state rather than
    # inspecting or listing one.
    "back": [
        ("go back to an earlier snapshot", []),
        ("restore {x}", ["the previous version", "snapshot 2", "an older state"]),
        ("roll back to {x}", ["the last checkpoint", "snapshot 3", "how it was before"]),
        ("revert to {x}", ["an earlier state", "the previous snapshot"]),
        ("undo {x} back to that checkpoint", ["everything", "the changes"]),
        ("put the tree back the way it was at {x}", ["snapshot 2", "the last commit"]),
        ("return the working tree to {x}", ["an older snapshot", "how it used to be"]),
        ("rewind to {x}", ["the earlier state", "snapshot 3"]),
    ],
}

# Applied to a fraction of examples so the model is not only ever asked in one
# register. Real requests are not all bare imperatives.
DRESSING = [
    "{x}",
    "{x}",
    "{x}",
    "please {x}",
    "can you {x}",
    "i want to {x}",
    "{x} for me",
    "could you {x} please",
    "i need to {x}",
]


def expand(family):
    template, slots = family
    if not slots:
        return [template]
    return [template.format(x=s) for s in slots]


# Which family indices are held out, for every applet.
#
# Fixed positions rather than "the last N". Holding out the tail meant that
# appending new families silently moved the test set onto them -- and since new
# families are written last, they are the most oblique phrasings, so the test
# got harder at the same moment the training set grew. The measured effect was
# 77.6% falling to 75.0% while the corpus more than doubled, which reads as a
# regression and was actually a different exam. These indices stay put as long
# as families are only appended, so numbers remain comparable across revisions.
HELD_OUT = frozenset((1, 3))


def build(seed, test_families, balance=True):
    """Return (train, test). Families are split, not instances."""
    rng = random.Random(seed)
    by_class_train, by_class_test = {}, {}

    for applet, families in FAMILIES.items():
        if len(families) <= max(HELD_OUT):
            raise SystemExit(f"{applet} has too few families to hold any out")
        tr, te = [], []
        for i, fam in enumerate(families):
            target = te if i in HELD_OUT else tr
            for text in expand(fam):
                target.append(rng.choice(DRESSING).format(x=text))
        by_class_train[applet] = tr
        by_class_test[applet] = te

    if balance:
        # Equal examples per class. Unbalanced, the largest class becomes a
        # magnet: `ls` had 13 training examples against 6 for others and
        # swallowed snaps, sysbox and pwd in the measured confusions. Ridge
        # regression has no class prior to correct for that, so the fix has to
        # be in the data.
        n = min(len(v) for v in by_class_train.values())
        for applet, items in by_class_train.items():
            rng.shuffle(items)
            by_class_train[applet] = items[:n]

    train = [{"applet": a, "task": t} for a, v in by_class_train.items() for t in v]
    test = [{"applet": a, "task": t} for a, v in by_class_test.items() for t in v]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def emit_rust(path, train, test):
    """Emit the corpus as a Rust table.

    Compiled in rather than loaded from the ESP because the corpus has to exist
    before any file can be read -- the probe is fitted from it at boot, and a
    system that cannot route until someone mounts something is worse than one
    that ships knowing its own tools. `teach` still appends to the namespace on
    top of this.

    Both splits are emitted, and which is which is preserved: `fit` holds out
    every fourth example, so shuffling them together would leak test items into
    training and turn the reported held-out number into a lie.
    """
    lines = [
        "// Generated by tools/dataset.py -- do not edit by hand.",
        "//",
        "// Template families, with whole families held out per applet rather than",
        "// instances sampled at random, so a held-out item is structurally unlike",
        "// anything trained on.",
        "",
        "/// (applet, task). The tail is the held-out split.",
        "pub const SEED: &[(&str, &str)] = &[",
    ]
    for e in train:
        lines.append(f'    ("{e["applet"]}", "{e["task"]}"),')
    lines.append("    // --- held out below this line ---")
    for e in test:
        lines.append(f'    ("{e["applet"]}", "{e["task"]}"),')
    lines.append("];")
    lines.append("")
    lines.append(f"/// How many of `SEED` are training examples.")
    lines.append(f"pub const SEED_TRAIN: usize = {len(train)};")
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--test-families", type=int, default=1)
    ap.add_argument("--rust", type=Path, default=None)
    args = ap.parse_args()

    train, test = build(args.seed, args.test_families)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"train": train, "test": test}, indent=1), encoding="utf-8"
    )
    if args.rust:
        emit_rust(args.rust, train, test)

    classes = sorted(FAMILIES)
    print(f"  {len(classes)} applets")
    print(f"  train {len(train)} examples, test {len(test)} examples")
    print(f"  held out family indices {sorted(HELD_OUT)} per applet (fixed, so "
          f"appending does not move the test set)")
    per = {c: sum(1 for e in train if e['applet'] == c) for c in classes}
    lo = min(per.values())
    hi = max(per.values())
    print(f"  train per applet: {lo}-{hi}")
    print(f"  chance accuracy: {100 / len(classes):.1f}%")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
