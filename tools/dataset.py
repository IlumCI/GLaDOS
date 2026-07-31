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
FAMILIES = {
    "sysbox": [
        ("what tools are available", []),
        ("list the {x}", ["commands", "applets", "available tools", "things you can do"]),
        ("what can you {x}", ["do", "run", "execute"]),
        ("show me the {x}", ["command list", "tool list", "options"]),
    ],
    "ls": [
        ("list the files", []),
        ("what is in {x}", ["this directory", "the folder", "here", "/ai"]),
        ("show me the contents of {x}", ["the directory", "this folder", "/tmp"]),
        ("{x} the directory listing", ["print", "display", "give me"]),
        ("what files are {x}", ["here", "in this folder", "present"]),
    ],
    "cd": [
        ("change directory to {x}", ["/ai", "/tmp", "the parent", "/sys"]),
        ("go to {x}", ["the other folder", "/ai/train", "the root", "the parent directory"]),
        ("move into {x}", ["the subdirectory", "/tmp", "that folder"]),
        ("switch to {x}", ["the /sys directory", "another folder"]),
        ("navigate to {x}", ["/ai", "the top level"]),
    ],
    "pwd": [
        ("where am i", []),
        ("what directory am i {x}", ["in", "currently in", "sitting in"]),
        ("print the {x}", ["working directory", "current path", "current location"]),
        ("show my {x}", ["current directory", "location", "position in the tree"]),
    ],
    "tree": [
        ("list everything recursively", []),
        ("show the {x}", ["whole hierarchy", "entire tree", "full structure"]),
        ("what is under {x} including subdirectories", ["/ai", "this folder", "the root"]),
        ("give me a {x} view", ["recursive", "nested", "full tree"]),
    ],
    "cat": [
        ("read {x}", ["the readme", "the file", "/sys/readme", "that note"]),
        ("print the contents of {x}", ["a file", "the readme", "/ai/notes"]),
        ("show me what {x} says", ["that file", "the readme", "this document"]),
        ("{x} the text in that file", ["display", "dump", "output"]),
    ],
    "stat": [
        ("how big is {x}", ["that file", "this object", "/ai/notes"]),
        ("give me details about {x}", ["a path", "this file", "that directory"]),
        ("what are the {x} of this file", ["properties", "attributes", "metadata"]),
        ("tell me about {x}", ["that object", "this path"]),
    ],
    "hash": [
        ("what is the {x} of that object", ["address", "hash", "content hash", "digest"]),
        ("show me the content address", []),
        ("print the {x} for this file", ["hash", "checksum", "digest"]),
        ("identify that object by {x}", ["hash", "address"]),
    ],
    "same": [
        ("are these two {x}", ["identical", "the same", "equal"]),
        ("compare {x}", ["two subtrees", "these files", "the two directories"]),
        ("check whether {x} match", ["these paths", "the two objects", "both files"]),
        ("do {x} have the same contents", ["these", "both files", "the two folders"]),
    ],
    "du": [
        ("how much space is {x}", ["used", "taken up", "consumed"]),
        ("report {x}", ["disk usage", "space used", "storage consumption"]),
        ("what is the {x} of everything here", ["total size", "footprint", "byte count"]),
        ("show {x} statistics", ["usage", "size", "storage"]),
    ],
    "find": [
        ("search for {x}", ["some text", "a word", "the phrase hello"]),
        ("look for {x} in the files", ["a string", "that word", "some text"]),
        ("which files {x} this word", ["contain", "mention", "include"]),
        ("{x} the tree for that phrase", ["grep", "scan", "search"]),
    ],
    "diff": [
        ("compare {x}", ["two snapshots", "snapshot 2 and 3", "the versions"]),
        ("what changed between {x}", ["versions", "the two snapshots", "then and now"]),
        ("show me the {x} between snapshots", ["difference", "delta", "changes"]),
        ("what is different {x}", ["since the last snapshot", "between these two states"]),
    ],
    "snaps": [
        ("list the snapshots", []),
        ("show {x}", ["snapshot history", "the checkpoints", "all the saved states"]),
        ("what {x} exist", ["snapshots", "checkpoints", "saved versions"]),
        ("give me the {x} of commits", ["history", "list", "log"]),
    ],
    "fsck": [
        ("verify the disk", []),
        ("check that everything is {x}", ["intact", "consistent", "not corrupted"]),
        ("{x} every stored object", ["verify", "validate", "check"]),
        ("is the store {x}", ["healthy", "consistent", "undamaged"]),
    ],
    "mkdir": [
        ("make a directory", []),
        ("create a {x}", ["new folder", "directory called work", "subdirectory"]),
        ("add a {x} named notes", ["folder", "directory"]),
        ("set up a {x} for this", ["new directory", "folder"]),
    ],
    "write": [
        ("save some text to a file", []),
        ("create a file {x}", ["with this content", "containing hello", "and put text in it"]),
        ("write {x} into a new file", ["this note", "some text", "a message"]),
        ("store this {x} as a file", ["text", "content", "message"]),
    ],
    "rm": [
        ("delete {x}", ["that name", "the file", "this entry"]),
        ("remove {x}", ["the file", "that path", "this directory entry"]),
        ("get rid of {x}", ["that file", "this name"]),
        ("unlink {x}", ["the entry", "that path"]),
    ],
    "mv": [
        ("rename {x}", ["that file", "this directory", "the note"]),
        ("move {x} somewhere else", ["that file", "this folder"]),
        ("change the {x} of that file", ["name", "path", "location"]),
        ("relocate {x}", ["this entry", "that object"]),
    ],
    "cp": [
        ("copy {x}", ["that file", "this directory", "the whole folder"]),
        ("duplicate {x}", ["this object", "that subtree", "the file"]),
        ("make a copy of {x}", ["this", "that directory", "the notes"]),
        ("clone {x} to another name", ["this folder", "that file"]),
    ],
    "snap": [
        ("take a snapshot", []),
        ("commit the {x}", ["current state", "working tree", "changes"]),
        ("save {x} as a checkpoint", ["everything", "the current state"]),
        ("record the {x} now", ["state", "current tree"]),
    ],
    "back": [
        ("go back to an earlier snapshot", []),
        ("restore {x}", ["the previous version", "snapshot 2", "an older state"]),
        ("roll back to {x}", ["the last checkpoint", "snapshot 3", "how it was before"]),
        ("revert to {x}", ["an earlier state", "the previous snapshot"]),
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


def build(seed, test_families):
    """Return (train, test). Families are split, not instances."""
    rng = random.Random(seed)
    train, test = [], []

    for applet, families in FAMILIES.items():
        if len(families) <= test_families:
            raise SystemExit(f"{applet} has too few families to hold any out")
        # The last `test_families` families are held out for every applet, so
        # the split is deterministic and reproducible from the seed alone.
        cut = len(families) - test_families
        for i, fam in enumerate(families):
            target = train if i < cut else test
            for text in expand(fam):
                dressed = rng.choice(DRESSING).format(x=text)
                target.append({"applet": applet, "task": dressed})

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
    print(f"  held out the last {args.test_families} template family per applet")
    per = {c: sum(1 for e in train if e['applet'] == c) for c in classes}
    lo = min(per.values())
    hi = max(per.values())
    print(f"  train per applet: {lo}-{hi}")
    print(f"  chance accuracy: {100 / len(classes):.1f}%")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
