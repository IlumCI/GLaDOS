"""The host-side knob table: constants in tools/ the CI loop may propose over.

**It ships empty, and that is the honest size.** Every numeric constant in
tools/ examined while this table was created is one of two things: evaluator
machinery (rails.py's floors, knob.py's rules, drive.py's timeouts, godel.py
itself), which changes only through the boundary lane because a loop that can
tune its own judge converges on a judge that says yes; or a kernel mirror
(forest_retrieve.py's scoring constants), which must track the kernel rather
than be tuned apart from it. A row invented to make the table non-empty
would be search without a subject.

Rung 2 (`godel.py discover`) is the mechanism that grows this with evidence:
a discovered constant arrives as a judged change adding a row here, and the
row's values then walk as ordinary grid points.

Same shape as `knob.rs::KNOBS`, deliberately: `file, symbol, now, values,
rail, about`, with `now` and every value as the literal string that appears
in the source. Same invariants too, enforced by `godel.py --selftest`: no
row offers its own current value, no duplicate values, no two rows naming
one constant, every field non-empty, nothing under an UNJUDGEABLE or
EVALUATOR prefix.

Deliberately NOT on godel.py's EVALUATOR list. This file is data the loop
may grow through the ordinary judged lane; splitting it out of godel.py is
what makes that possible without the boundary lane, and its safety is the
structural validation plus the gate every proposal passes anyway.
"""

#: (file, symbol, now, values, rail, about)
ROWS = []
