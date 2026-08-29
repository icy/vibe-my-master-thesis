# Symbolic verification of the thesis's Maple appendix (and beyond)

The thesis's appendix B (`prog_maple.tex`) is *Matlab-ish* only in the
user's memory — it's actually a **Maple** worksheet, and it verifies
exactly one thing by computer algebra: the Lorentzian wedge product
identity `<psi_z ^ psi_zbar, eta>` (both signs of `a+b`), because that's
a 4×4-determinant cross product — the single most error-prone hand
computation in the whole proof.

This directory reproduces that check independently in Python/SymPy, and
then goes further: **every "asserted by direct computation, no CAS
check" step flagged after the first pass has now been pushed through
either a genuine from-scratch symbolic proof, or (where that turned out
not to be the right tool) a solid battery of independent exact-rational
numeric checks** — see the table and the "what full symbolic buys you"
section below for exactly which is which, and why.

## Files

| File | What it does | Method | Runtime |
|---|---|---|---|
| `lorentz.py` | Herm(2) ↔ L⁴ correspondence, the Lorentzian triple/"wedge" product, and the Lorentz inner product — a direct port of the Maple procedures `get_vector_from_matrix`, `get_matrix_from_vector`, `cross_product`, `scalar_product`. | — | — |
| `test_wedge_identities.py` | Sanity-checks `lorentz.py` against the thesis's own general identities `eq:wp.H3.base`, `eq:wp.H3.a`, `eq:wp.H3.b` (ch1.tex). Bonus: an independent re-verification of those three thesis identities. | full symbolic | ~2s |
| `verify_case1.py` | The thesis's own Maple computation for `a+b=1`, vs. the closed form in `ch2s2.tex` (`eq:psi_z.wedge.psi_:z.times.eta`). | numeric (default) / full symbolic (`--symbolic`) | 3s / ~4 min |
| `verify_case2.py` | The mirror check for `a+b=-1` (`ch2s3.tex`, `...eta(2)`). | numeric | ~3s |
| `verify_HK.py` | The `H`, `K` curvature formulas (Cramer's rule on `eq:sys.eta`) and the BLW cancellation `2a(H-1)+b(K-1)=0`. | full symbolic | ~1s |
| `verify_U.py` | **From-scratch** derivation of `eq:U` (`U = A*G_z + conj(B)_z = b/(2(a+b))·ρ·G_z`) using real Wirtinger calculus and the Liouville PDE — not just algebra given `U`'s formula, but *deriving* it. | full symbolic | ~1s |
| `verify_init_prop.py` | The initial-data identity `eq:init-prop` (ch2s2.tex). | numeric (3 independent configurations) | ~1s |

Run any of them with `python3 <file>.py` (needs `pip install sympy`).

## Result

**Everything checks out.** All three of the thesis's Maple-verified wedge
identities, the $H,K$/BLW cancellation, the from-scratch PDE-derivation
of `eq:U`, and the initial-data identity `eq:init-prop` — every one of
them matches the thesis exactly.

## Can we "just fully use option 4 (real symbolic proof)" everywhere?

Mostly yes — and pushing on that is genuinely what closed two of the
three remaining gaps:

- **`eq:U`** turned out to be pure Wirtinger calculus + one PDE
  substitution: represent `rho` as a real `sympy.Function(z, zbar)`,
  `G` as `sympy.Function(z)` alone (meromorphic, for free, since it has
  no `zbar`-dependence at all), differentiate with `sympy.diff`, and
  substitute the Liouville PDE (solved for `rho_zzbar`) wherever it
  appears. **This ran in ~1 second** and is a strictly stronger check
  than the thesis's own Maple appendix, since it *derives* `U`'s formula
  rather than taking it as given.
- **`H`, `K`, BLW cancellation** is just Cramer's rule on a 2×2 linear
  system, then substituting `U`'s formula and simplifying. Also ~1
  second, no shortcuts.
- **`eq:init-prop`** is where "just throw full symbolic at it" stopped
  being the right tool, for an instructive reason (see below) — it's
  verified here by several independent exact-rational numeric
  configurations instead, which is what actually caught a real bug
  (also below).

### Why `eq:init-prop` resisted a literal symbolic (Gröbner) proof

The natural approach: encode `β∈H³`, `V∈S³₁`, `⟨β,V⟩=0` and their
derivative consequences as a polynomial ideal in the 16 components of
`β, V, β', V'`, and check the candidate identity reduces to `0` modulo
that ideal via Gröbner-basis reduction (`sympy.groebner` + `sympy.reduced`)
— the direct generalization of the linear-constraint substitution trick
used in `test_wedge_identities.py`.

This does **not** work here, and it's worth understanding why: Gröbner
basis reduction checks literal **ideal membership** over the polynomial
ring, treating every symbol as a formal (in effect complex) indeterminate.
But a real quadratic-form identity (built from `⟨·,·⟩` constraints) can
vanish at every *real* point of the constraint variety without being a
literal polynomial combination of the defining equations — the real
solution set is a proper slice of the larger complex variety the ideal
actually describes, and the identity is only guaranteed on that slice.
`sympy`'s `real=True` assumption on a symbol has no effect on `Poly`/
`groebner()`, which are purely formal ring-theoretic tools. So Gröbner
reduction here was answering a strictly *harder*, not-actually-true
question. (A genuinely rigorous symbolic proof would need an explicit
**rational parametrization of the real constraint variety** — e.g. a
secant-line parametrization of the hyperboloid/sphere from a known
rational point, à la rational parametrizations of Pythagorean triples —
rather than brute-force ideal membership. Not attempted here.)

**This dead end is exactly what surfaced a real bug** — not in the
thesis, but in an earlier draft of this check: the first attempt used
only the "obvious" constraints (`⟨β,β⟩=-1`, `⟨V,V⟩=1`, `⟨β,V⟩=0`, and
their derivatives `⟨β,β'⟩=0`, `⟨V,V'⟩=0`, `⟨β',V⟩+⟨β,V'⟩=0`) and got a
clean, reproducible, nonzero residual of exactly `-β₀·⟨β',V⟩` at every
test point tried. Tracing that down (cross-checking `G'` via genuine
`sympy.diff` automatic differentiation to rule out a hand quotient-rule
error, then testing several sign/index variants of the target formula,
none of which matched) eventually led back to the **Björling problem
statement itself** (`ch2.tex`): it gives `⟨β(s),V(s)⟩ = ⟨β'(s),V(s)⟩ = 0`
as **two independent hypotheses** — the second is *not* merely the
derivative consequence of the first (that would only give
`⟨β',V⟩+⟨β,V'⟩=0`, a weaker statement). Once `⟨β',V⟩=0` is included as
its own constraint (forcing `⟨β,V'⟩=0` too), the residual vanishes
identically at every test point tried — see `verify_init_prop.py`'s
docstring for the full account.

## The method, in general

1. **Port the linear algebra, not the "obviously true" surface theory,
   and sanity-check it against known consequences first** (see
   `test_wedge_identities.py`) before trusting it for anything harder.
2. **Track conjugates as independent symbols, not `sympy.conjugate()`**
   on compound expressions — the latter is very slow to simplify/cancel
   in determinant-heavy computations (minutes-to-hangs observed here).
3. **For calculus-based identities** (`eq:U`): use real
   `sympy.Function`s of independent formal `z, zbar` symbols, encode
   meromorphicity by simply not giving a function `zbar`-dependence,
   and substitute PDEs via `.subs()` on the `Derivative` object they
   constrain.
4. **For a numeric check: substitute *exact* rationals** (`sympy.Rational`
   + `I`, never floats). A handful of independent, structurally
   different test points is fast and — importantly — is the *correct*
   tool (not a lesser one) whenever the identity is conditioned on a
   *real* algebraic variety, where Gröbner-ideal-membership over-tests.
5. **For a full symbolic proof of a rational-function identity**:
   avoid `simplify()`/`factor()` on the raw result (expression swell
   from unreduced intermediate rational sub-expressions); cross-multiply
   numerator/denominator (`sympy.fraction`) and `sympy.expand()` instead.
6. **When a Gröbner-based proof unexpectedly fails**: don't assume the
   target identity is false — check whether the constraints are
   inherently *real* (inner products, norms) rather than purely
   algebraic/complex. If so, either find an explicit rational
   parametrization of the real variety, or fall back to numeric spot
   checks, which are the mathematically appropriate tool there, not a
   compromise.
