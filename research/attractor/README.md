# The fly brain's ignition core: scripts and frozen predictions

Run from `out/attractor/` (they read `../fly.bin` and `../fly-src/`, fetched
by `tools/connectome.py fetch fly out/fly-src`). Findings are written up in
`design/attractor.md`.

- `core.py` -- the self-sustaining core, from wiring alone
- `measure.py` -- which neurons persist after each sense's poke
- `types.py` -- cell types of the state and the core
- `lesion.py` -- causal tests: silence a pathway, or the core, or matched random sets
- `ignite.py` -- the forward mean-field bootstrap (refuted)
- `seeds.py` -- seed count and seed output into the core, against 21 cases
- `reduced.py` -- the core alone, against the full brain (21/21)
- `atlas.py` -- every reachable cell type through the reduced model
- `receptor.py` -- full-brain check of one cell type
- `predictions.json`, `predictions-atlas.json` -- predictions frozen before their runs
