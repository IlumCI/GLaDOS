"""Rung 3: declared transform families the CI loop may author from.

**A template is an enumerable family, not a generator of ideas.** Each module
here exposes `emit(root) -> [Candidate]`, where a Candidate carries the kind
it should be filed under, a one-line reason, and a unified diff. The loop
judges the diff through exactly the gate every other rung goes through; what
a template buys is that the *space* is closed and mechanical, so a candidate
is valid Rust by construction rather than by a model's good behaviour.

**The admission rule for a family is that it can win.** A template whose
every output is a regression on its own claimed rail is a template that only
ever spends runners -- `#[inline(never)]` was the first family considered and
was dropped for exactly that: it costs image bytes, buys symbolication, and
symbolication has no rail, so the judge could only ever refuse it. A family
belongs here when the rail it claims can go the right way.

Families today:

- `unused_import` -- the compiler names them, so the space is enumerated by
  rustc rather than guessed at. Claims `cost.warnings`, which such a patch
  moves down by construction, and is a pure deletion, which is what the
  `cleanup` kind asks for.
"""

from . import unused_import

FAMILIES = (unused_import,)


def emit_all(root):
    """Every family's candidates, in a fixed order so a later run reproduces
    the same sequence -- `frontier`'s rule about walking a declared grid."""
    out = []
    for mod in FAMILIES:
        out.extend(mod.emit(root))
    return out


def selftest(claim):
    """Every family's claims. Takes the caller's `claim(what, good)` so
    these land in `godel.py --selftest`: the loop has one selftest, and a
    second one is a second thing to remember to run."""
    claim("at least one template family is declared", len(FAMILIES) > 0)
    for mod in FAMILIES:
        mod.selftest(claim)

    # A family is admitted on being able to WIN, so the rail it claims has
    # to be one a judge counts and the kind it files under has to be one
    # the table enables. A family claiming a rail nobody reads would spend
    # nights on a comparison that cannot come back better, which is the
    # argument that kept `#[inline(never)]` out of this package -- it costs
    # image bytes, buys symbolication, and symbolication has no rail.
    import rails
    import godel
    for mod in FAMILIES:
        name = mod.__name__.rsplit(".", 1)[-1]
        claim(f"{name} claims a rail rails.py declares a floor for",
              any(mod.RAIL.startswith(g) for g in rails.NOISE))
        k = godel.KINDS.get(mod.KIND)
        claim(f"{name} files under an enabled kind", k is not None and k.enabled)
