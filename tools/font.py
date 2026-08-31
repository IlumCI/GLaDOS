#!/usr/bin/env python3
"""Build the extended glyph table in src/gfx/font.rs.

The ASCII half of that file is hand-drawn and stays hand-drawn. This builds
what sits above it, and exists because most of those glyphs are not new
drawings at all: an accented letter is a letter this font already has with a
mark over it, and drawing it a second time by hand would let the two versions
drift.

    python tools/font.py --emit src/gfx/font.rs     rewrite the table in place
    python tools/font.py --proof                    print every glyph to look at
    python tools/font.py --proof 0xE9 0x2500        print only these

Read the proof sheet before committing. A bitmap font is one of the few things
in this tree where the only real test is a person looking at it.

Bit convention, matching font.rs: one byte per row, top row first, bit 7
(0x80) leftmost. Hand-drawn glyphs sit on a 5-wide grid in bits 6..2, so a
full row is 0x7C and the cell keeps its side bearing. Box drawing deliberately
breaks that rule and uses all eight bits, because a line that stops short of
the cell edge does not join the line in the next cell.
"""

import argparse
import io
import re
import sys

# --- the base font, read back rather than copied ------------------------
#
# Composition only works if the accent lands on the same 'e' the console
# draws. Parsing the file is what guarantees that; a second copy of the ASCII
# table here would be correct on the day it was written.


def read_ascii(path):
    src = io.open(path, encoding="utf-8").read()
    body = src.split("static GLYPHS", 1)[1].split(chr(10) + "];", 1)[0]
    rows = re.findall(r"\[((?:0x[0-9A-Fa-f]{2},){7}0x[0-9A-Fa-f]{2})\]", body)
    table = {}
    for i, row in enumerate(rows):
        table[0x20 + i] = [int(b, 16) for b in row.split(",")]
    assert len(table) == 0x7E - 0x20 + 1, "expected 95 ASCII glyphs, got %d" % len(table)
    return table


# --- drawing helpers ----------------------------------------------------


def art(*lines):
    """Eight rows of eight characters. '#' is ink, anything else is not."""
    assert len(lines) == 8, "a glyph is 8 rows, got %d" % len(lines)
    out = []
    for line in lines:
        assert len(line) == 8, "a row is 8 columns, got %d: %r" % (len(line), line)
        bits = 0
        for x, c in enumerate(line):
            if c == "#":
                bits |= 0x80 >> x
        out.append(bits)
    return out


def blank():
    return [0] * 8


def hspan(g, row, x0, x1):
    for x in range(x0, x1 + 1):
        g[row] |= 0x80 >> x


def vspan(g, col, y0, y1):
    for y in range(y0, y1 + 1):
        g[y] |= 0x80 >> col


# --- accent composition -------------------------------------------------
#
# One rule for every accented letter: the mark occupies rows 0 and 1, and the
# letter occupies rows 2 to 7. Lowercase needs no work, since every letter that
# takes a mark in Latin-1 (a c e i n o u y) already draws inside rows 2..6.
# Uppercase does not fit, so it gets a six-row form below -- squashed by one
# row, which every 8x8 font that carries these has had to do.

ACCENT = {
    "grave":  [0x20, 0x10],
    "acute":  [0x08, 0x10],
    "circ":   [0x10, 0x28],
    "tilde":  [0x32, 0x4C],
    "uml":    [0x28, 0x00],
    "ring":   [0x18, 0x18],
}

# Six-row uppercase, for the letters Latin-1 puts a mark over. Each is the
# seven-row original with one interior row removed rather than a fresh
# drawing, so a capital with a mark still looks like the capital without one.
SQUASHED = {
    "A": [0x38, 0x44, 0x7C, 0x44, 0x44, 0x44],
    "E": [0x7C, 0x40, 0x78, 0x40, 0x7C, 0x00],
    "I": [0x38, 0x10, 0x10, 0x10, 0x38, 0x00],
    "O": [0x38, 0x44, 0x44, 0x44, 0x44, 0x38],
    "U": [0x44, 0x44, 0x44, 0x44, 0x44, 0x38],
    "N": [0x44, 0x64, 0x54, 0x4C, 0x44, 0x00],
    "C": [0x38, 0x44, 0x40, 0x40, 0x44, 0x38],
    "Y": [0x44, 0x28, 0x10, 0x10, 0x10, 0x00],
}


def upper(letter, mark):
    return ACCENT[mark] + SQUASHED[letter]


def lower(base, letter, mark):
    g = list(base[ord(letter)])
    # 'i' carries a dot at row 0 that the mark replaces. Nothing else in the
    # set draws above row 2, so this is the only glyph that has to be cleared.
    if letter == "i":
        g[0] = 0x00
    rows = ACCENT[mark] + g[2:]
    assert len(rows) == 8
    return rows


def cedilla(base, letter, squash):
    """A tail hanging below the letter, in the row the letter left empty."""
    if squash:
        g = [0x00] + SQUASHED[letter] + [0x00]
    else:
        g = list(base[ord(letter)])
    assert g[7] == 0x00, "%r has no room for a cedilla" % letter
    g[7] = 0x10
    return g


def build(base):
    """Every glyph above ASCII, as {codepoint: [8 rows]}."""
    g = {}

    # --- Latin-1 supplement, 0xA0..0xBF: the symbols ---------------------
    g[0x00A0] = blank()                                    # no-break space
    g[0x00A1] = art("........",  # inverted exclamation
                    "...#....",
                    "........",
                    "...#....",
                    "...#....",
                    "...#....",
                    "...#....",
                    "........")
    g[0x00A2] = art("........",  # cent
                    "...#....",
                    "..####..",
                    ".#.#....",
                    ".#.#....",
                    "..####..",
                    "...#....",
                    "........")
    g[0x00A3] = art("..###...",  # pound
                    ".#...#..",
                    ".#......",
                    "####....",
                    ".#......",
                    ".#......",
                    "#####...",
                    "........")
    g[0x00A4] = art("........",  # currency
                    ".#...#..",
                    "..###...",
                    "..#.#...",
                    "..###...",
                    ".#...#..",
                    "........",
                    "........")
    g[0x00A5] = art(".#...#..",  # yen
                    ".#...#..",
                    "..#.#...",
                    ".#####..",
                    "...#....",
                    ".#####..",
                    "...#....",
                    "........")
    g[0x00A6] = art("...#....",  # broken bar
                    "...#....",
                    "...#....",
                    "........",
                    "...#....",
                    "...#....",
                    "...#....",
                    "........")
    g[0x00A7] = art("..###...",  # section
                    ".#......",
                    "..##....",
                    ".#..#...",
                    "..##....",
                    "....#...",
                    ".###....",
                    "........")
    g[0x00A8] = art("..#.#...",  # diaeresis
                    "........",
                    "........",
                    "........",
                    "........",
                    "........",
                    "........",
                    "........")
    g[0x00A9] = art("..###...",  # copyright
                    ".#...#..",
                    "#.##..#.",
                    "#.#...#.",
                    "#.##..#.",
                    ".#...#..",
                    "..###...",
                    "........")
    g[0x00AA] = art("..###...",  # feminine ordinal
                    ".....#..",
                    "..####..",
                    ".#...#..",
                    "..####..",
                    "........",
                    ".#####..",
                    "........")
    g[0x00AB] = art("........",  # left guillemet
                    "..#..#..",
                    ".#..#...",
                    "#..#....",
                    ".#..#...",
                    "..#..#..",
                    "........",
                    "........")
    g[0x00AC] = art("........",  # not sign
                    "........",
                    ".#####..",
                    ".....#..",
                    "........",
                    "........",
                    "........",
                    "........")
    g[0x00AD] = art("........",  # soft hyphen: drawn, since a cell was spent
                    "........",
                    "........",
                    "..###...",
                    "........",
                    "........",
                    "........",
                    "........")
    g[0x00AE] = art("..###...",  # registered
                    ".#...#..",
                    "#.##..#.",
                    "#.#.#.#.",
                    "#.##..#.",
                    ".#...#..",
                    "..###...",
                    "........")
    g[0x00AF] = art(".#####..",  # macron
                    "........",
                    "........",
                    "........",
                    "........",
                    "........",
                    "........",
                    "........")
    g[0x00B0] = art("..##....",  # degree
                    ".#..#...",
                    ".#..#...",
                    "..##....",
                    "........",
                    "........",
                    "........",
                    "........")
    g[0x00B1] = art("........",  # plus-minus
                    "...#....",
                    "...#....",
                    ".#####..",
                    "...#....",
                    "...#....",
                    ".#####..",
                    "........")
    g[0x00B2] = art("..##....",  # superscript two
                    ".#..#...",
                    "....#...",
                    "...#....",
                    ".####...",
                    "........",
                    "........",
                    "........")
    g[0x00B3] = art(".###....",  # superscript three
                    "....#...",
                    "..##....",
                    "....#...",
                    ".###....",
                    "........",
                    "........",
                    "........")
    g[0x00B4] = art("....#...",  # acute accent
                    "...#....",
                    "........",
                    "........",
                    "........",
                    "........",
                    "........",
                    "........")
    g[0x00B5] = art("........",  # micro
                    "........",
                    ".#...#..",
                    ".#...#..",
                    ".#...#..",
                    ".#####..",
                    ".#......",
                    ".#......")
    g[0x00B6] = art("..#####.",  # pilcrow
                    ".###..#.",
                    ".###..#.",
                    "..##..#.",
                    "...#..#.",
                    "...#..#.",
                    "...#..#.",
                    "........")
    g[0x00B7] = art("........",  # middle dot
                    "........",
                    "........",
                    "...##...",
                    "...##...",
                    "........",
                    "........",
                    "........")
    g[0x00B8] = art("........",  # cedilla
                    "........",
                    "........",
                    "........",
                    "........",
                    "........",
                    "...#....",
                    "..##....")
    g[0x00B9] = art("...#....",  # superscript one
                    "..##....",
                    "...#....",
                    "...#....",
                    "..###...",
                    "........",
                    "........",
                    "........")
    g[0x00BA] = art("..###...",  # masculine ordinal
                    ".#...#..",
                    ".#...#..",
                    "..###...",
                    "........",
                    ".#####..",
                    "........",
                    "........")
    g[0x00BB] = art("........",  # right guillemet
                    "#..#....",
                    ".#..#...",
                    "..#..#..",
                    ".#..#...",
                    "#..#....",
                    "........",
                    "........")
    g[0x00BC] = art(".#...#..",  # one quarter
                    ".#..#...",
                    ".#.#....",
                    "..#.##..",
                    ".#.#.#..",
                    "#..####.",
                    "#....#..",
                    "........")
    g[0x00BD] = art(".#...#..",  # one half
                    ".#..#...",
                    ".#.#....",
                    "..#.##..",
                    ".#....#.",
                    "#...##..",
                    "#..####.",
                    "........")
    g[0x00BE] = art("##...#..",  # three quarters
                    "..#.#...",
                    "##.#....",
                    "..#.##..",
                    ".#.#.#..",
                    "#..####.",
                    "#....#..",
                    "........")
    g[0x00BF] = art("........",  # inverted question
                    "...#....",
                    "........",
                    "...#....",
                    "..#.....",
                    ".#...#..",
                    "..###...",
                    "........")

    # --- Latin-1 letters -------------------------------------------------
    for cp, letter, mark in [
        (0x00C0, "A", "grave"), (0x00C1, "A", "acute"), (0x00C2, "A", "circ"),
        (0x00C3, "A", "tilde"), (0x00C4, "A", "uml"),   (0x00C5, "A", "ring"),
        (0x00C8, "E", "grave"), (0x00C9, "E", "acute"), (0x00CA, "E", "circ"),
        (0x00CB, "E", "uml"),
        (0x00CC, "I", "grave"), (0x00CD, "I", "acute"), (0x00CE, "I", "circ"),
        (0x00CF, "I", "uml"),
        (0x00D1, "N", "tilde"),
        (0x00D2, "O", "grave"), (0x00D3, "O", "acute"), (0x00D4, "O", "circ"),
        (0x00D5, "O", "tilde"), (0x00D6, "O", "uml"),
        (0x00D9, "U", "grave"), (0x00DA, "U", "acute"), (0x00DB, "U", "circ"),
        (0x00DC, "U", "uml"),
        (0x00DD, "Y", "acute"),
    ]:
        g[cp] = upper(letter, mark)

    for cp, letter, mark in [
        (0x00E0, "a", "grave"), (0x00E1, "a", "acute"), (0x00E2, "a", "circ"),
        (0x00E3, "a", "tilde"), (0x00E4, "a", "uml"),   (0x00E5, "a", "ring"),
        (0x00E8, "e", "grave"), (0x00E9, "e", "acute"), (0x00EA, "e", "circ"),
        (0x00EB, "e", "uml"),
        (0x00EC, "i", "grave"), (0x00ED, "i", "acute"), (0x00EE, "i", "circ"),
        (0x00EF, "i", "uml"),
        (0x00F1, "n", "tilde"),
        (0x00F2, "o", "grave"), (0x00F3, "o", "acute"), (0x00F4, "o", "circ"),
        (0x00F5, "o", "tilde"), (0x00F6, "o", "uml"),
        (0x00F9, "u", "grave"), (0x00FA, "u", "acute"), (0x00FB, "u", "circ"),
        (0x00FC, "u", "uml"),
        (0x00FD, "y", "acute"), (0x00FF, "y", "uml"),
    ]:
        g[cp] = lower(base, letter, mark)

    g[0x00C7] = cedilla(base, "C", squash=True)
    g[0x00E7] = cedilla(base, "c", squash=False)

    g[0x00C6] = art("..#####.",  # AE
                    ".##.....",
                    ".##.....",
                    "##.####.",
                    "###.....",
                    "##......",
                    "##.#####",
                    "........")
    g[0x00E6] = art("........",
                    "........",
                    ".##..##.",
                    "...####.",
                    ".#######",
                    "##..#...",
                    ".#####..",
                    "........")
    g[0x00D0] = art(".####...",  # eth, capital
                    ".#...#..",
                    ".#...#..",
                    "####.#..",
                    ".#...#..",
                    ".#...#..",
                    ".####...",
                    "........")
    g[0x00F0] = art("..#.#...",
                    "...#....",
                    "..#.##..",
                    ".....#..",
                    "..####..",
                    ".#...#..",
                    "..###...",
                    "........")
    g[0x00D7] = art("........",  # multiplication
                    "........",
                    ".#...#..",
                    "..#.#...",
                    "...#....",
                    "..#.#...",
                    ".#...#..",
                    "........")
    g[0x00F7] = art("........",  # division
                    "...#....",
                    "........",
                    ".#####..",
                    "........",
                    "...#....",
                    "........",
                    "........")
    g[0x00D8] = art("..####..",  # O slash
                    ".#...##.",
                    ".#..#.#.",
                    ".#.#..#.",
                    ".##...#.",
                    "##...#..",
                    ".####...",
                    "........")
    g[0x00F8] = art("........",
                    "........",
                    "..####..",
                    ".#..##..",
                    ".#.#.#..",
                    ".##..#..",
                    "####....",
                    "........")
    g[0x00DE] = art(".#......",  # thorn, capital
                    ".####...",
                    ".#...#..",
                    ".#...#..",
                    ".####...",
                    ".#......",
                    ".#......",
                    "........")
    g[0x00FE] = art(".#......",
                    ".#......",
                    ".####...",
                    ".#...#..",
                    ".#...#..",
                    ".####...",
                    ".#......",
                    ".#......")
    g[0x00DF] = art("..###...",  # sharp s
                    ".#...#..",
                    ".#..#...",
                    ".#.#....",
                    ".#..#...",
                    ".#...#..",
                    ".#.##...",
                    "........")

    # --- punctuation the model actually writes ---------------------------
    g[0x2010] = base[ord("-")]                             # hyphen
    g[0x2013] = art("........",  # en dash
                    "........",
                    "........",
                    "..####..",
                    "........",
                    "........",
                    "........",
                    "........")
    g[0x2014] = art("........",  # em dash
                    "........",
                    "........",
                    "#######.",
                    "........",
                    "........",
                    "........",
                    "........")
    g[0x2018] = art("...#....",  # left single quote
                    "..#.....",
                    "........",
                    "........",
                    "........",
                    "........",
                    "........",
                    "........")
    g[0x2019] = art("...#....",  # right single quote
                    "....#...",
                    "........",
                    "........",
                    "........",
                    "........",
                    "........",
                    "........")
    g[0x201A] = art("........",  # single low quote
                    "........",
                    "........",
                    "........",
                    "........",
                    "...#....",
                    "..#.....",
                    "........")
    g[0x201C] = art(".#.#....",  # left double quote
                    "#.#.....",
                    "........",
                    "........",
                    "........",
                    "........",
                    "........",
                    "........")
    g[0x201D] = art(".#.#....",  # right double quote
                    "..#.#...",
                    "........",
                    "........",
                    "........",
                    "........",
                    "........",
                    "........")
    g[0x201E] = art("........",  # double low quote
                    "........",
                    "........",
                    "........",
                    "........",
                    ".#.#....",
                    "#.#.....",
                    "........")
    g[0x2020] = art("...#....",  # dagger
                    "...#....",
                    ".#####..",
                    "...#....",
                    "...#....",
                    "...#....",
                    "........",
                    "........")
    g[0x2021] = art("...#....",  # double dagger
                    ".#####..",
                    "...#....",
                    "...#....",
                    ".#####..",
                    "...#....",
                    "........",
                    "........")
    g[0x2022] = art("........",  # bullet
                    "........",
                    "..###...",
                    "..###...",
                    "..###...",
                    "........",
                    "........",
                    "........")
    g[0x2026] = art("........",  # ellipsis
                    "........",
                    "........",
                    "........",
                    "........",
                    "........",
                    "#.#.#...",
                    "........")
    g[0x2030] = art("##...#..",  # per mille
                    "##..#...",
                    "...#....",
                    "..#.##..",
                    ".#..##..",
                    "#.##..##",
                    "..##..##",
                    "........")
    g[0x2039] = art("........",  # left angle quote
                    "...#....",
                    "..#.....",
                    ".#......",
                    "..#.....",
                    "...#....",
                    "........",
                    "........")
    g[0x203A] = art("........",  # right angle quote
                    ".#......",
                    "..#.....",
                    "...#....",
                    "..#.....",
                    ".#......",
                    "........",
                    "........")
    g[0x20AC] = art("...###..",  # euro
                    "..#.....",
                    ".#####..",
                    "..#.....",
                    ".#####..",
                    "..#.....",
                    "...###..",
                    "........")
    g[0x2122] = art("#####.##",  # trade mark
                    "..#..#.#",
                    "..#..#.#",
                    "........",
                    "........",
                    "........",
                    "........",
                    "........")

    # --- arrows ----------------------------------------------------------
    g[0x2190] = art("........",  # left
                    "...#....",
                    "..#.....",
                    ".#######",
                    "..#.....",
                    "...#....",
                    "........",
                    "........")
    g[0x2191] = art("...#....",  # up
                    "..###...",
                    ".#.#.#..",
                    "...#....",
                    "...#....",
                    "...#....",
                    "...#....",
                    "........")
    g[0x2192] = art("........",  # right
                    "....#...",
                    ".....#..",
                    "#######.",
                    ".....#..",
                    "....#...",
                    "........",
                    "........")
    g[0x2193] = art("...#....",  # down
                    "...#....",
                    "...#....",
                    "...#....",
                    ".#.#.#..",
                    "..###...",
                    "...#....",
                    "........")
    g[0x2194] = art("........",  # left right
                    "..#..#..",
                    ".#....#.",
                    "########",
                    ".#....#.",
                    "..#..#..",
                    "........",
                    "........")

    # --- mathematics -----------------------------------------------------
    g[0x2202] = art("..####..",  # partial
                    ".....#..",
                    "..####..",
                    ".#...#..",
                    ".#...#..",
                    ".#...#..",
                    "..###...",
                    "........")
    g[0x2206] = art("........",  # increment
                    "...#....",
                    "...#....",
                    "..#.#...",
                    "..#.#...",
                    ".#...#..",
                    ".#####..",
                    "........")
    g[0x220F] = art("........",  # product
                    ".#####..",
                    ".#...#..",
                    ".#...#..",
                    ".#...#..",
                    ".#...#..",
                    ".#...#..",
                    "........")
    g[0x2211] = art("........",  # sum
                    ".#####..",
                    ".#......",
                    "..#.....",
                    "...#....",
                    "..#.....",
                    ".#......",
                    ".#####..")
    g[0x2212] = art("........",  # minus
                    "........",
                    "........",
                    ".#####..",
                    "........",
                    "........",
                    "........",
                    "........")
    g[0x221A] = art("....###.",  # square root
                    "....#...",
                    "....#...",
                    "....#...",
                    "#...#...",
                    ".#..#...",
                    "..###...",
                    "........")
    g[0x221E] = art("........",  # infinity
                    "........",
                    ".##.##..",
                    "#..#..#.",
                    "#..#..#.",
                    ".##.##..",
                    "........",
                    "........")
    g[0x2229] = art("........",  # intersection
                    "..###...",
                    ".#...#..",
                    ".#...#..",
                    ".#...#..",
                    ".#...#..",
                    ".#...#..",
                    "........")
    g[0x222B] = art("...##...",  # integral
                    "..#..#..",
                    "..#.....",
                    "..#.....",
                    "..#.....",
                    "..#.....",
                    ".#..#...",
                    ".##.....")
    g[0x2248] = art("........",  # approximately
                    "........",
                    "..##..#.",
                    ".#..##..",
                    "........",
                    "..##..#.",
                    ".#..##..",
                    "........")
    g[0x2260] = art("........",  # not equal
                    "....#...",
                    ".#####..",
                    "...#....",
                    ".#####..",
                    "..#.....",
                    "........",
                    "........")
    g[0x2261] = art("........",  # identical
                    ".#####..",
                    "........",
                    ".#####..",
                    "........",
                    ".#####..",
                    "........",
                    "........")
    g[0x2264] = art("....##..",  # less or equal
                    "..##....",
                    "##......",
                    "..##....",
                    "....##..",
                    "........",
                    "######..",
                    "........")
    g[0x2265] = art("##......",  # greater or equal
                    "..##....",
                    "....##..",
                    "..##....",
                    "##......",
                    "........",
                    "######..",
                    "........")

    # --- box drawing -----------------------------------------------------
    #
    # Full cell width and height, so a run of them joins up. The single-line
    # forms cross at column 3 and row 3; the double-line forms use columns
    # 2 and 4 and rows 2 and 4.
    def boxed(left, right, up, down, double):
        gl = blank()
        if double:
            cols, rows = (2, 4), (2, 4)
        else:
            cols, rows = (3,), (3,)
        for r in rows:
            if left:
                hspan(gl, r, 0, max(cols))
            if right:
                hspan(gl, r, min(cols), 7)
        for c in cols:
            if up:
                vspan(gl, c, 0, max(rows))
            if down:
                vspan(gl, c, min(rows), 7)
        # A double corner leaves an interior stub that a real corner does not
        # have. Clear the quadrant the corner turns away from.
        if double:
            if not left:
                for r in rows:
                    for x in range(0, 2):
                        gl[r] &= ~(0x80 >> x) & 0xFF
            if not right:
                for r in rows:
                    for x in range(5, 8):
                        gl[r] &= ~(0x80 >> x) & 0xFF
        return gl

    single = {
        0x2500: (1, 1, 0, 0), 0x2502: (0, 0, 1, 1),
        0x250C: (0, 1, 0, 1), 0x2510: (1, 0, 0, 1),
        0x2514: (0, 1, 1, 0), 0x2518: (1, 0, 1, 0),
        0x251C: (0, 1, 1, 1), 0x2524: (1, 0, 1, 1),
        0x252C: (1, 1, 0, 1), 0x2534: (1, 1, 1, 0),
        0x253C: (1, 1, 1, 1),
    }
    for cp, (l, r, u, d) in single.items():
        g[cp] = boxed(l, r, u, d, False)

    # A double line is four L-corners, not two crossing lines: at every
    # junction the two strokes stop short of each other so the corner reads
    # as a corner. Drawing them as full spans and clearing afterwards closed
    # the elbow of every corner into a small box. Spans are inclusive.
    DOUBLE = {
        0x2550: ([(0, 7)], [(0, 7)], [], []),
        0x2551: ([], [], [(0, 7)], [(0, 7)]),
        0x2554: ([(2, 7)], [(4, 7)], [(2, 7)], [(4, 7)]),
        0x2557: ([(0, 4)], [(0, 2)], [(4, 7)], [(2, 7)]),
        0x255A: ([(4, 7)], [(2, 7)], [(0, 4)], [(0, 2)]),
        0x255D: ([(0, 2)], [(0, 4)], [(0, 2)], [(0, 4)]),
        0x2560: ([(4, 7)], [(4, 7)], [(0, 7)], [(0, 2), (4, 7)]),
        0x2563: ([(0, 2)], [(0, 2)], [(0, 2), (4, 7)], [(0, 7)]),
        0x2566: ([(0, 7)], [(0, 2), (4, 7)], [(4, 7)], [(4, 7)]),
        0x2569: ([(0, 2), (4, 7)], [(0, 7)], [(0, 2)], [(0, 2)]),
        0x256C: ([(0, 2), (4, 7)], [(0, 2), (4, 7)], [(0, 2), (4, 7)], [(0, 2), (4, 7)]),
    }
    for cp, (top, bot, left, right) in DOUBLE.items():
        gl = blank()
        for row, spans in ((2, top), (4, bot)):
            for x0, x1 in spans:
                hspan(gl, row, x0, x1)
        for col, spans in ((2, left), (4, right)):
            for y0, y1 in spans:
                vspan(gl, col, y0, y1)
        g[cp] = gl

    # --- blocks and shades ----------------------------------------------
    g[0x2580] = [0xFF] * 4 + [0x00] * 4                    # upper half
    g[0x2584] = [0x00] * 4 + [0xFF] * 4                    # lower half
    g[0x2588] = [0xFF] * 8                                 # full
    g[0x258C] = [0xF0] * 8                                 # left half
    g[0x2590] = [0x0F] * 8                                 # right half
    g[0x2591] = [0x88, 0x00, 0x22, 0x00] * 2               # light shade
    g[0x2592] = [0xAA, 0x55] * 4                           # medium shade
    g[0x2593] = [0x77, 0xDD] * 4                           # dark shade

    # --- geometric -------------------------------------------------------
    g[0x25A0] = [0x00, 0x7E, 0x7E, 0x7E, 0x7E, 0x7E, 0x7E, 0x00]   # filled square
    g[0x25A1] = [0x00, 0x7E, 0x42, 0x42, 0x42, 0x42, 0x7E, 0x00]   # hollow square
    g[0x25AA] = [0x00, 0x00, 0x3C, 0x3C, 0x3C, 0x3C, 0x00, 0x00]   # small square
    g[0x25B2] = art("........",  # up triangle
                    "...#....",
                    "..###...",
                    "..###...",
                    ".#####..",
                    ".#####..",
                    "#######.",
                    "........")
    g[0x25B6] = art("........",  # right triangle
                    ".#......",
                    ".###....",
                    ".#####..",
                    ".###....",
                    ".#......",
                    "........",
                    "........")
    g[0x25BC] = art("........",  # down triangle
                    "#######.",
                    ".#####..",
                    ".#####..",
                    "..###...",
                    "..###...",
                    "...#....",
                    "........")
    g[0x25C0] = art("........",  # left triangle
                    "......#.",
                    "....###.",
                    "..#####.",
                    "....###.",
                    "......#.",
                    "........",
                    "........")
    g[0x25CB] = art("........",  # hollow circle
                    "..###...",
                    ".#...#..",
                    ".#...#..",
                    ".#...#..",
                    "..###...",
                    "........",
                    "........")
    g[0x25CF] = art("........",  # filled circle
                    "..###...",
                    ".#####..",
                    ".#####..",
                    ".#####..",
                    "..###...",
                    "........",
                    "........")

    # --- symbols the desktop and the model both reach for ---------------
    g[0x2605] = art("...#....",  # filled star
                    "...#....",
                    "#######.",
                    "..###...",
                    ".##.##..",
                    ".#...#..",
                    "........",
                    "........")
    g[0x2606] = art("...#....",  # hollow star
                    "..#.#...",
                    "##...##.",
                    "..#.#...",
                    ".##.##..",
                    ".#...#..",
                    "........",
                    "........")
    g[0x2660] = art("...#....",  # spade
                    "..###...",
                    ".#####..",
                    "#######.",
                    "#######.",
                    "...#....",
                    "..###...",
                    "........")
    g[0x2663] = art("..###...",  # club
                    "..###...",
                    "##.#.##.",
                    "#######.",
                    "##.#.##.",
                    "...#....",
                    "..###...",
                    "........")
    g[0x2665] = art(".##.##..",  # heart
                    "#######.",
                    "#######.",
                    "#######.",
                    ".#####..",
                    "..###...",
                    "...#....",
                    "........")
    g[0x2666] = art("...#....",  # diamond
                    "..###...",
                    ".#####..",
                    "#######.",
                    ".#####..",
                    "..###...",
                    "...#....",
                    "........")
    g[0x2713] = art("........",  # check
                    "......#.",
                    ".....#..",
                    "#...#...",
                    ".#.#....",
                    "..#.....",
                    "........",
                    "........")
    g[0x2717] = art("........",  # ballot x
                    "#.....#.",
                    ".#...#..",
                    "..#.#...",
                    "...#....",
                    "..#.#...",
                    ".#...#..",
                    "#.....#.")

    # --- Greek -----------------------------------------------------------
    #
    # Fourteen capitals are the Latin letter and are aliased to it rather than
    # drawn twice: a Greek Alpha that did not look exactly like an A would be
    # a bug nobody could see. Only the ten with their own shape are drawn.
    for cp, latin in [
        (0x0391, "A"), (0x0392, "B"), (0x0395, "E"), (0x0396, "Z"),
        (0x0397, "H"), (0x0399, "I"), (0x039A, "K"), (0x039C, "M"),
        (0x039D, "N"), (0x039F, "O"), (0x03A1, "P"), (0x03A4, "T"),
        (0x03A5, "Y"), (0x03A7, "X"),
    ]:
        g[cp] = base[ord(latin)]

    g[0x0393] = art(".#####..",  # Gamma
                    ".#......",
                    ".#......",
                    ".#......",
                    ".#......",
                    ".#......",
                    ".#......",
                    "........")
    g[0x0394] = g[0x2206]                                  # Delta
    g[0x0398] = art("..###...",  # Theta
                    ".#...#..",
                    ".#...#..",
                    ".#####..",
                    ".#...#..",
                    ".#...#..",
                    "..###...",
                    "........")
    g[0x039B] = art("...#....",  # Lambda
                    "...#....",
                    "..#.#...",
                    "..#.#...",
                    ".#...#..",
                    ".#...#..",
                    "#.....#.",
                    "........")
    g[0x039E] = art(".#####..",  # Xi
                    "........",
                    "........",
                    "..###...",
                    "........",
                    "........",
                    ".#####..",
                    "........")
    g[0x03A0] = g[0x220F]                                  # Pi
    g[0x03A3] = art(".#####..",  # Sigma
                    ".#......",
                    "..#.....",
                    "...#....",
                    "..#.....",
                    ".#......",
                    ".#####..",
                    "........")
    g[0x03A6] = art("...#....",  # Phi
                    "..###...",
                    ".#.#.#..",
                    ".#.#.#..",
                    ".#.#.#..",
                    "..###...",
                    "...#....",
                    "........")
    g[0x03A8] = art(".#.#.#..",  # Psi
                    ".#.#.#..",
                    ".#.#.#..",
                    "..###...",
                    "...#....",
                    "...#....",
                    "..###...",
                    "........")
    g[0x03A9] = art("..###...",  # Omega
                    ".#...#..",
                    ".#...#..",
                    ".#...#..",
                    "..#.#...",
                    "..#.#...",
                    ".##.##..",
                    "........")

    g[0x03B1] = art("........",  # alpha
                    "........",
                    "..##.#..",
                    ".#..##..",
                    ".#...#..",
                    ".#..##..",
                    "..##.#..",
                    "........")
    g[0x03B2] = art("..##....",  # beta
                    ".#..#...",
                    ".#..#...",
                    ".###....",
                    ".#..#...",
                    ".#..#...",
                    ".###....",
                    ".#......")
    g[0x03B3] = art("........",  # gamma
                    "........",
                    ".#...#..",
                    ".#...#..",
                    "..#.#...",
                    "...#....",
                    "..#.....",
                    ".#......")
    g[0x03B4] = art("..###...",  # delta
                    ".#......",
                    "..#.....",
                    "..###...",
                    ".#...#..",
                    ".#...#..",
                    "..###...",
                    "........")
    g[0x03B5] = art("........",  # epsilon
                    "........",
                    "..####..",
                    ".#......",
                    "..###...",
                    ".#......",
                    "..####..",
                    "........")
    g[0x03B6] = art("..####..",  # zeta
                    "..#.....",
                    "...#....",
                    "...#....",
                    "..#.....",
                    "..#.....",
                    "...##...",
                    "....#...")
    g[0x03B7] = art("........",  # eta
                    "........",
                    ".####...",
                    ".#...#..",
                    ".#...#..",
                    ".#...#..",
                    ".....#..",
                    ".....#..")
    g[0x03B8] = art("..###...",  # theta
                    ".#...#..",
                    ".#...#..",
                    ".#####..",
                    ".#...#..",
                    ".#...#..",
                    "..###...",
                    "........")
    g[0x03B9] = art("........",  # iota
                    "........",
                    "..#.....",
                    "..#.....",
                    "..#.....",
                    "..#..#..",
                    "...##...",
                    "........")
    g[0x03BA] = art("........",  # kappa
                    "........",
                    ".#...#..",
                    ".#..#...",
                    ".###....",
                    ".#..#...",
                    ".#...#..",
                    "........")
    g[0x03BB] = art(".#......",  # lambda
                    "..#.....",
                    "..##....",
                    "..#.#...",
                    ".#...#..",
                    ".#....#.",
                    "#.....#.",
                    "........")
    g[0x03BC] = g[0x00B5]                                  # mu
    g[0x03BD] = art("........",  # nu
                    "........",
                    ".#...#..",
                    ".#..#...",
                    ".#.#....",
                    ".###....",
                    "..#.....",
                    "........")
    g[0x03BE] = art("..####..",  # xi
                    "..#.....",
                    "...#....",
                    "..###...",
                    "..#.....",
                    "...#....",
                    "...##...",
                    "....#...")
    g[0x03BF] = base[ord("o")]                             # omicron
    g[0x03C0] = art("........",  # pi
                    "........",
                    ".#####..",
                    "..#.#...",
                    "..#.#...",
                    "..#.#...",
                    "..#.#...",
                    "........")
    g[0x03C1] = art("........",  # rho
                    "........",
                    "..###...",
                    ".#...#..",
                    ".#...#..",
                    ".####...",
                    ".#......",
                    ".#......")
    g[0x03C2] = art("........",  # final sigma
                    "........",
                    "..####..",
                    ".#......",
                    ".#......",
                    "..###...",
                    "....#...",
                    "...#....")
    g[0x03C3] = art("........",  # sigma
                    "........",
                    "..#####.",
                    ".#..#...",
                    ".#...#..",
                    ".#...#..",
                    "..###...",
                    "........")
    g[0x03C4] = art("........",  # tau
                    "........",
                    ".#####..",
                    "...#....",
                    "...#....",
                    "...#....",
                    "....#...",
                    "........")
    g[0x03C5] = art("........",  # upsilon
                    "........",
                    ".#...#..",
                    ".#...#..",
                    ".#...#..",
                    "..###...",
                    "........",
                    "........")
    g[0x03C6] = art("........",  # phi
                    "...#....",
                    "..###...",
                    ".#.#.#..",
                    ".#.#.#..",
                    "..###...",
                    "...#....",
                    "...#....")
    g[0x03C7] = art("........",  # chi
                    "........",
                    ".#...#..",
                    "..#.#...",
                    "...#....",
                    "..#.#...",
                    ".#...#..",
                    "#.......")
    g[0x03C8] = art("........",  # psi
                    ".#.#.#..",
                    ".#.#.#..",
                    ".#.#.#..",
                    "..###...",
                    "...#....",
                    "...#....",
                    "...#....")
    g[0x03C9] = art("........",  # omega
                    "........",
                    ".#.#.#..",
                    ".#.#.#..",
                    ".#.#.#..",
                    ".#.#.#..",
                    "..#.#...",
                    "........")

    return g


# --- output -------------------------------------------------------------

NAMES = {}


def render(rows):
    return ["".join("#" if r & (0x80 >> x) else "." for x in range(8)) for r in rows]


def proof(glyphs, only):
    """Print glyphs side by side, eight to a line, to be looked at."""
    keys = sorted(glyphs)
    if only:
        keys = [k for k in keys if k in only]
    for i in range(0, len(keys), 8):
        chunk = keys[i:i + 8]
        print("  ".join("U+%04X %s" % (k, chr(k) if k > 0xA0 else " ") for k in chunk))
        art_rows = [render(glyphs[k]) for k in chunk]
        for r in range(8):
            print("  ".join(a[r] + "  " for a in art_rows))
        print()


def emit(glyphs, path):
    lines = []
    lines.append("#[rustfmt::skip]")
    lines.append("static EXTRA: [(u32, [u8; 8]); %d] = [" % len(glyphs))
    for cp in sorted(glyphs):
        rows = ",".join("0x%02X" % b for b in glyphs[cp])
        ch = chr(cp)
        label = ch if cp > 0xA0 and ch.isprintable() else ""
        lines.append("    (0x%04X, [%s]), // %s" % (cp, rows, label))
    lines.append("];")
    block = "\n".join(lines) + "\n"

    src = io.open(path, encoding="utf-8").read()
    start = src.index("#[rustfmt::skip]\nstatic EXTRA")
    end = src.index(chr(10) + "];", start) + 4
    src = src[:start] + block + src[end:]
    io.open(path, "w", encoding="utf-8", newline="\n").write(src)
    print("wrote %d glyphs to %s" % (len(glyphs), path))


def main():
    # The proof sheet prints the character beside its bitmap, and a Windows
    # console defaults to cp1252, which cannot encode most of what this file
    # exists to draw.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", default="src/gfx/font.rs")
    ap.add_argument("--emit", nargs="?", const="src/gfx/font.rs")
    ap.add_argument("--proof", nargs="*", default=None)
    args = ap.parse_args()

    base = read_ascii(args.font)
    glyphs = build(base)

    for cp, rows in glyphs.items():
        assert len(rows) == 8, "U+%04X has %d rows" % (cp, len(rows))
        assert all(0 <= b <= 0xFF for b in rows), "U+%04X has a bad row" % cp

    if args.proof is not None:
        only = set(int(a, 0) for a in args.proof) if args.proof else None
        proof(glyphs, only)
    if args.emit:
        emit(glyphs, args.emit)
    if args.proof is None and not args.emit:
        print("%d glyphs; pass --proof to look at them or --emit to write them" % len(glyphs))


if __name__ == "__main__":
    sys.exit(main())
