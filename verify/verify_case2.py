"""
Same check as verify_case1.py, for the a+b=-1 (anti-meromorphic G) case
-- ch2s3.tex, eq:psi_z.wedge.psi_:z.times.eta(2):

    <psi_z ^ psi_zbar, eta> = (i/2)(|U|^2 - |B_z-bar|^2)

Roles of "the nonzero G-derivative" and "the nonzero conjugated-B
derivative" swap relative to case 1: G is anti-meromorphic here, so the
generically-nonzero quantity is G_{zbar} (called Gb_z below), not G_z.

Two modes, mirroring verify_case1.py:
  --numeric (default): 5 random EXACT-rational trials, ~2s.
  --symbolic: a genuine symbolic proof via cross-multiplying
    numerator/denominator and expanding -- copies verify_case1.py's
    --symbolic path with G_z<->Gb_z, B_z<->(conj B2)_z swapped. ~4-6 min.
"""
import sys
import random
import sympy as sp
from lorentz import lorentz_wedge, lorentz_ip


def build_result(a, b, rho, Gb_z, Gb_zb, G, Gg, A2, A2b, Bx_z, Bx_zb):
    U2, U2b = (b / (2*a + 2*b)) * rho * Gb_z, (b / (2*a + 2*b)) * rho * Gb_zb
    C2 = 2 / rho

    F = sp.Matrix([[1, 0], [G, 1]])
    Fx = sp.Matrix([[1, Gg], [0, 1]])
    # A2,C2 (and B2 itself) drop out of this identity, as in case 1 -- only
    # the matrix SHAPE of psi2_z / psi2_zx (ch2s3.tex) matters here.
    Omega2 = sp.Matrix([[A2, sp.Symbol('B2')], [sp.Symbol('B2x'), C2]])
    psi2 = F * Omega2 * Fx
    psi2_z = F * sp.Matrix([[sp.Symbol('A2_z'), U2], [Bx_zb, 0]]) * Fx
    psi2_zx = F * sp.Matrix([[sp.Symbol('A2_zb'), Bx_z], [U2b, 0]]) * Fx
    N2 = F * sp.Matrix([[rho, 0], [0, 0]]) * Fx
    eta2 = N2 - psi2

    wedge = lorentz_wedge(psi2, psi2_z, psi2_zx)
    return lorentz_ip(wedge, eta2), U2, U2b


def target_form(U2, U2b, Bx_z, Bx_zb):
    return sp.I / 2 * (U2 * U2b - Bx_z * Bx_zb)


def run_numeric(trials=5, seed=1):
    rng = random.Random(seed)

    def rnd_real():
        return sp.Rational(rng.randint(-9, 9), rng.randint(1, 9))

    def rnd_complex():
        return rnd_real() + sp.I * rnd_real()

    for t in range(trials):
        a, b, rho = rnd_real(), rnd_real(), rnd_real()
        while a + b == 0 or rho == 0:
            a, b, rho = rnd_real(), rnd_real(), rnd_real()

        Gb_z = rnd_complex()                 # "G_{zbar}", nonzero for anti-meromorphic G
        while Gb_z == 0:
            Gb_z = rnd_complex()
        Gb_zb = sp.conjugate(Gb_z)
        G = rnd_complex(); Gg = sp.conjugate(G)
        A2 = rnd_complex(); A2b = sp.conjugate(A2)
        Bx_z = rnd_complex(); Bx_zb = sp.conjugate(Bx_z)   # (conj B2)_z and its conjugate

        result, U2, U2b = build_result(a, b, rho, Gb_z, Gb_zb, G, Gg, A2, A2b, Bx_z, Bx_zb)
        target = target_form(U2, U2b, Bx_z, Bx_zb)
        diff = sp.simplify(result - target)
        print(f"trial {t}: diff = {diff}")
        assert diff == 0


def run_symbolic():
    a, b, rho = sp.symbols('a b rho', real=True)
    Gb_z, Gb_zb, G, Gg, A2, A2b, Bx_z, Bx_zb = sp.symbols(
        'Gb_z Gb_zb G Gg A2 A2b Bx_z Bx_zb')

    result, U2, U2b = build_result(a, b, rho, Gb_z, Gb_zb, G, Gg, A2, A2b, Bx_z, Bx_zb)
    target = target_form(U2, U2b, Bx_z, Bx_zb)

    rn, rd = sp.fraction(sp.together(result))
    tn, td = sp.fraction(sp.together(target))
    cross = sp.expand(rn * td - tn * rd)
    print("cross-multiplied numerator difference (want 0):")
    print(cross)
    print("is exactly zero:", cross == 0)
    assert cross == 0


if __name__ == "__main__":
    if "--symbolic" in sys.argv:
        run_symbolic()
    else:
        run_numeric()
