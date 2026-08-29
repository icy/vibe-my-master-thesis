"""
Verifies the thesis's own Maple computation for the a+b=1 case
(prog_maple.tex, section "Truong hop a+b=1"):

    <psi_z ^ psi_zbar, eta>                        (eq:psi_z.wedge.psi_:z.times.eta)

against the closed form quoted in ch2s2.tex,
    i * (4(a+b)^2 |B_z|^2 - b^2 rho^2 |G_z|^2) / (8(a+b)^2)
  = (i/2) (|B_z|^2 - |U|^2),   U = b/(2(a+b)) * rho * G_z.

Every complex atom X carries its own independent conjugate symbol Xb
(tracked by hand), exactly like Maple's own `conjugate(rho_z)`,
`conjugate(G_z)` on atoms -- this avoids sympy's much slower automatic
conjugate() distribution over compound expressions.

Two modes:
  --numeric (default): 5 random EXACT-rational trials, ~2s, and
    already very strong evidence (a nonzero rational-function identity
    essentially never vanishes at several independent random rational
    points).
  --symbolic: a genuine symbolic proof (no substitution at all) via
    cross-multiplying numerator/denominator and expanding -- true
    proof, but ~4 minutes on a modern laptop; sympy's naive determinant
    expansion doesn't reduce intermediate rational expressions, so
    avoid `simplify()`/`factor()` on the raw result (they are much
    slower here) and use cross-multiplication + expand() instead.
"""
import sys
import random
import sympy as sp
from lorentz import lorentz_wedge, lorentz_ip


def build_result(a, b, rho, rho_z, rho_zb, G_z, G_zb, A, Ab, G, Gb,
                  A_z, A_zb, B_z, B_zb):
    U, Ub = (b / (2*a + 2*b)) * rho * G_z, (b / (2*a + 2*b)) * rho * G_zb
    C = 2 / rho
    B = (2 * rho_z) / (rho**2 * G_z)
    Bx = (2 * rho_zb) / (rho**2 * G_zb)          # = conjugate(B)

    F = sp.Matrix([[1, 0], [G, 1]])
    Fx = sp.Matrix([[1, Gb], [0, 1]])            # = F*

    Omega = sp.Matrix([[A, B], [Bx, C]])
    psi = F * Omega * Fx
    psi_z = F * sp.Matrix([[A_z, B_z], [U, 0]]) * Fx
    psi_zx = F * sp.Matrix([[A_zb, Ub], [B_zb, 0]]) * Fx    # conjugate of psi_z

    N = F * sp.Matrix([[rho, 0], [0, 0]]) * Fx
    eta = N - psi

    wedge = lorentz_wedge(psi, psi_z, psi_zx)
    return lorentz_ip(wedge, eta), U, Ub


def target_forms(a, b, rho, G_z, G_zb, B_z, B_zb, U, Ub):
    t1 = sp.I * (4*(a+b)**2 * B_z*B_zb - b**2 * rho**2 * G_z*G_zb) / (8*(a+b)**2)
    t2 = sp.I / 2 * (B_z*B_zb - U*Ub)
    return t1, t2


def run_numeric(trials=5, seed=0):
    random.seed(seed)

    def rnd_real():
        return sp.Rational(random.randint(-9, 9), random.randint(1, 9))

    def rnd_complex():
        return rnd_real() + sp.I * rnd_real()

    for t in range(trials):
        a, b, rho = rnd_real(), rnd_real(), rnd_real()
        while a + b == 0 or rho == 0:
            a, b, rho = rnd_real(), rnd_real(), rnd_real()
        rho_z = rnd_complex(); rho_zb = sp.conjugate(rho_z)
        G_z = rnd_complex()
        while G_z == 0:
            G_z = rnd_complex()
        G_zb = sp.conjugate(G_z)
        A = rnd_complex(); Ab = sp.conjugate(A)
        G = rnd_complex(); Gb = sp.conjugate(G)
        A_z = rnd_complex(); A_zb = sp.conjugate(A_z)
        B_z = rnd_complex(); B_zb = sp.conjugate(B_z)

        result, U, Ub = build_result(a, b, rho, rho_z, rho_zb, G_z, G_zb,
                                      A, Ab, G, Gb, A_z, A_zb, B_z, B_zb)
        t1, t2 = target_forms(a, b, rho, G_z, G_zb, B_z, B_zb, U, Ub)
        d1 = sp.nsimplify(sp.simplify(result - t1))
        d2 = sp.nsimplify(sp.simplify(result - t2))
        print(f"trial {t}: diff-vs-closed-form-#1 = {d1}   diff-vs-(i/2)(|B_z|^2-|U|^2) = {d2}")


def run_symbolic():
    a, b, rho = sp.symbols('a b rho', real=True)
    (rho_z, rho_zb, G_z, G_zb, A, Ab, G, Gb,
     A_z, A_zb, B_z, B_zb) = sp.symbols(
        'rho_z rho_zb G_z G_zb A Ab G Gb A_z A_zb B_z B_zb')

    result, U, Ub = build_result(a, b, rho, rho_z, rho_zb, G_z, G_zb,
                                  A, Ab, G, Gb, A_z, A_zb, B_z, B_zb)
    t1, _ = target_forms(a, b, rho, G_z, G_zb, B_z, B_zb, U, Ub)

    rn, rd = sp.fraction(sp.together(result))
    tn, td = sp.fraction(sp.together(t1))
    cross = sp.expand(rn * td - tn * rd)
    print("cross-multiplied numerator difference (want 0):")
    print(cross)
    print("is exactly zero:", cross == 0)


if __name__ == "__main__":
    if "--symbolic" in sys.argv:
        run_symbolic()
    else:
        run_numeric()
