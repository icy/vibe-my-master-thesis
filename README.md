## Description

I need to revise my master thesis. The thesis was heavily relying
on some Matlab computing and I have been always worrying if something has gone wrong.
It's in 2006. Basically 20 years ago.

## Files

- `vibe-maths.tex` — the thesis write-up/revision (LaTeX source).
- `output/vibe-maths.pdf` — compiled PDF of the above.
- `verify/` — independent Python/SymPy re-verification of the thesis's Maple appendix and related identities (see `verify/README.md` for the full method/results table).
  - `lorentz.py` — Herm(2) ↔ L⁴ correspondence, Lorentzian wedge/inner products (shared library).
  - `test_wedge_identities.py` — sanity-checks `lorentz.py` against `eq:wpbase`/`eq:wpa`/`eq:wpb`.
  - `verify_HK.py` — H, K curvature formulas and the BLW cancellation.
  - `verify_U.py` — from-scratch derivation of `eq:U` via Wirtinger calculus + Liouville PDE.
  - `verify_case1.py` — the `a+b=1` wedge/normal identity, vs. the thesis's own Maple computation.
  - `verify_case2.py` — the mirror `a+b=-1` case.
  - `verify_init_prop.py` — the initial-data identity `eq:init-prop`.
- `origin/` — original 2006 source material, kept locally only (not tracked in git).
  - `origin/mymaster-final.pdf` — the original master thesis PDF (43 pages).
  - `origin/article.pdf` — related article (6 pages).
