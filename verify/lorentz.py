"""
Lorentz L^4 <-> Herm(2) linear algebra, in SymPy.

A direct, faithful port of the Maple procedures in the thesis's own
`prog_maple.tex` appendix (get_vector_from_matrix, get_matrix_from_vector,
cross_product, scalar_product), built from:
  - the Herm(2) identification            (eq:random.14.10 in prog_maple.tex)
  - the Lorentzian triple product/"wedge"  (eq:wedge.def in ch1.tex)
  - <m,m> = -det(m) for m in Herm(2), polarized to a bilinear form.

Independently sanity-checked (see test_wedge_identities.py) against the
thesis's own general identities eq:wp.H3.base, eq:wp.H3.a, eq:wp.H3.b
before being trusted for anything else.
"""
import sympy as sp

I = sp.I


def matrix_from_vector(v):
    """(x0,x1,x2,x3) in L^4  ->  Hermitian 2x2 matrix."""
    x0, x1, x2, x3 = v
    return sp.Matrix([[x0 + x3, x1 + I * x2],
                       [x1 - I * x2, x0 - x3]])


def vector_from_matrix(M):
    """Inverse of matrix_from_vector; returns a length-4 list [x0,x1,x2,x3]."""
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    return [(a + d) / 2, (b + c) / 2, (b - c) / (2 * I), (a - d) / 2]


def lorentz_wedge(S, M, N):
    """S x M x N = M ^ N (at basepoint S), all as Herm(2) matrices."""
    vS, vM, vN = vector_from_matrix(S), vector_from_matrix(M), vector_from_matrix(N)
    Big = sp.Matrix([vS, vM, vN])              # 3x4

    def cof(cols):
        return Big[:, cols].det()

    A, B = cof([1, 2, 3]), cof([0, 2, 3])
    C, D = -cof([0, 1, 3]), cof([0, 1, 2])
    return matrix_from_vector((A, B, C, D))


def lorentz_ip(M, N):
    """<M,N> = -1/2 { det(M+N) - det(M) - det(N) }."""
    return sp.together(-sp.Rational(1, 2) * ((M + N).det() - M.det() - N.det()))


def dot4(u, v):
    """Ordinary Lorentz inner product on plain 4-tuples/lists."""
    u0, u1, u2, u3 = u
    v0, v1, v2, v3 = v
    return -u0 * v0 + u1 * v1 + u2 * v2 + u3 * v3
