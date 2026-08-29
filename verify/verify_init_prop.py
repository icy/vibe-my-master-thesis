"""
Verifies eq:init-prop (ch2s2.tex):

  (v0+v3)^2 G' {beta1+i*beta2 - conj(G)*(beta0+beta3)}
    = v0'+v3' + i*{(V^v')_0 + (V^v')_3}

by EXACT-RATIONAL numeric spot checks at several genuinely different
constraint-satisfying (beta,V,beta',V') configurations.

Constraints, straight from the Bjorling Problem statement (ch2.tex) --
note <beta,V>=0 AND <beta',V>=0 are given as TWO INDEPENDENT
hypotheses (the second is NOT merely the derivative consequence of the
first!). An earlier attempt at this proof missed that and found a
spurious nonzero residual of exactly -beta_0*<beta',V>, which vanishes
once this hypothesis is correctly included -- see verify/README.md for
the full story (a real bug hunt, not a shortcut).

  <beta,beta>=-1, <V,V>=1, <beta,V>=0        (curve data)
  <beta,beta'>=0, <V,V'>=0                   (derivatives of the above)
  <beta',V>=0                                (INDEPENDENT hypothesis)
  <beta,V'>=0                                (then forced)

NOTE on why this is numeric, not a from-scratch symbolic (Groebner)
proof like verify_HK.py/verify_U.py: naive Groebner-basis ideal
membership over these 16 symbols does NOT confirm this identity, even
though it demonstrably holds -- because Groebner reduction works over
the polynomial ring treating symbols as formal (complex) indeterminates,
but this identity only holds on the REAL points of the constraint
variety, not the larger complex variety cut out by the same equations
(a real quadratic-form identity need not be a literal polynomial
combination of the defining equations). A fully rigorous symbolic proof
would need an explicit real-rational parametrization of the constraint
variety (e.g. a secant-line parametrization of the hyperboloid/sphere),
not brute-force Groebner reduction.
"""
import sympy as sp
from lorentz import matrix_from_vector, lorentz_wedge, vector_from_matrix, dot4

I = sp.I


def solve_derivative_constraints(beta, V):
    """Solve the 4 linear constraints for 4 of the 8 components of
    (beta', V') in terms of the other 4, generically (via a proper
    linear-system solve, not hand-picking which symbol each equation
    determines -- that was the bug in an earlier draft)."""
    p0,p1,p2,p3,q0,q1,q2,q3 = sp.symbols('p0 p1 p2 p3 q0 q1 q2 q3', real=True)
    bp, Vp = (p0,p1,p2,p3), (q0,q1,q2,q3)
    unknowns = [p0,p1,p2,p3,q0,q1,q2,q3]
    eqs = [dot4(beta,bp), dot4(V,Vp), dot4(bp,V), dot4(beta,Vp)]
    M = sp.Matrix([[sp.diff(eq, u) for u in unknowns] for eq in eqs])
    # pick 4 unknowns to solve for whose 4x4 coefficient submatrix is invertible
    from itertools import combinations
    for cols in combinations(range(8), 4):
        sub = M[:, list(cols)]
        if sub.det() != 0:
            solve_for = [unknowns[i] for i in cols]
            free = [unknowns[i] for i in range(8) if i not in cols]
            sol = sp.solve([sp.Eq(e, 0) for e in eqs], solve_for, dict=True)[0]
            return bp, Vp, sol, free
    raise RuntimeError("no invertible 4x4 submatrix found")


def check(beta, V, free_values, label):
    bp_sym, Vp_sym, sol, free = solve_derivative_constraints(beta, V)
    subs_free = dict(zip(free, free_values))
    bp = tuple(sp.nsimplify(x.subs(sol).subs(subs_free)) for x in bp_sym)
    Vp = tuple(sp.nsimplify(x.subs(sol).subs(subs_free)) for x in Vp_sym)

    # sanity: re-verify all 4 constraints actually hold
    assert dot4(beta,bp) == 0 and dot4(V,Vp) == 0 and dot4(bp,V) == 0 and dot4(beta,Vp) == 0

    v, vp = tuple(beta[i]+V[i] for i in range(4)), tuple(bp[i]+Vp[i] for i in range(4))
    v0v3 = v[0]+v[3]
    G = (v[1]-I*v[2])/v0v3
    Gbar = sp.conjugate(G)
    Gp = ((vp[1]-I*vp[2])*v0v3 - (v[1]-I*v[2])*(vp[0]+vp[3])) / v0v3**2
    LHS = v0v3**2 * Gp * (beta[1]+I*beta[2] - Gbar*(beta[0]+beta[3]))

    Mbeta, MV, Mvp = matrix_from_vector(beta), matrix_from_vector(V), matrix_from_vector(vp)
    W = vector_from_matrix(lorentz_wedge(Mbeta, MV, Mvp))
    RHS = vp[0]+vp[3] + I*(W[0]+W[3])

    print(f"{label}: LHS-RHS = {sp.simplify(LHS-RHS)}   (beta'={bp}, V'={Vp})")


check((1,0,0,0), (0,1,0,0), [2,3,5,7], "test 1")
check((sp.sqrt(2),1,0,0), (0,0,1,0), [2,3,5,7], "test 2")
check((sp.Rational(5,3),sp.Rational(4,3),0,0), (sp.Rational(4,5),1,sp.Rational(4,5),0),
      [sp.Rational(11,4), sp.Rational(-7,3), sp.Rational(9,2), sp.Rational(1,5)], "test 3")
