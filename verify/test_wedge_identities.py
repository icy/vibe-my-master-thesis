"""
Sanity check of lorentz.py (and, as a side effect, an independent
verification of the thesis's own small original results eq:wp.H3.base,
eq:wp.H3.a, eq:wp.H3.b in ch1.tex) BEFORE trusting the machinery for
anything harder.

Run:  python3 test_wedge_identities.py
Expect: three lines, each printing 0 / a zero vector.
Runtime: ~2s.
"""
import sympy as sp
from lorentz import matrix_from_vector, vector_from_matrix, lorentz_wedge, lorentz_ip, dot4

p0, p1, p2, p3 = sp.symbols('p0 p1 p2 p3', real=True)
u0, u1, u2, u3 = sp.symbols('u0 u1 u2 u3', real=True)
v0, v1, v2, v3 = sp.symbols('v0 v1 v2 v3', real=True)
w0, w1, w2, w3 = sp.symbols('w0 w1 w2 w3', real=True)
p, u, v, w = (p0, p1, p2, p3), (u0, u1, u2, u3), (v0, v1, v2, v3), (w0, w1, w2, w3)
Mp, Mu, Mv, Mw = map(matrix_from_vector, (p, u, v, w))

# ---- eq:wp.H3.base: fully general p,u,v with only <u,v>=0 imposed ----
v0_sol = sp.solve(sp.Eq(dot4(u, v), 0), v0)[0]
subs0 = {v0: v0_sol}
W = lorentz_wedge(Mp, Mu, Mv)
lhs = lorentz_ip(W, W)
alpha, beta = dot4(u, u), dot4(v, v)
rhs = -alpha * beta * dot4(p, p) + alpha * dot4(p, v) ** 2 + beta * dot4(p, u) ** 2
base_diff = sp.simplify((lhs - rhs).subs(subs0))
print("eq:wp.H3.base check (want 0):", base_diff)
assert base_diff == 0

# ---- eq:wp.H3.a / eq:wp.H3.b: p in H^3 (<p,p>=-1), u,v,w tangent at p ----
Mp2 = matrix_from_vector((1, 0, 0, 0))          # p = (1,0,0,0) in H^3
u_t, v_t, w_t = (0, u1, u2, u3), (0, v1, v2, v3), (0, w1, w2, w3)
Mu2, Mv2, Mw2 = map(matrix_from_vector, (u_t, v_t, w_t))
v1_sol = sp.solve(sp.Eq(dot4(u_t, v_t), 0), v1)[0]
subs_t = {v1: v1_sol}

UwedgeV = lorentz_wedge(Mp2, Mu2, Mv2)
lhs_a = lorentz_ip(UwedgeV, UwedgeV)
rhs_a = dot4(u_t, u_t) * dot4(v_t, v_t) - dot4(u_t, v_t) ** 2
a_diff = sp.simplify((lhs_a - rhs_a).subs(subs_t))
print("eq:wp.H3.a check (want 0):", a_diff)
assert a_diff == 0

Wwedge_UV = lorentz_wedge(Mp2, Mw2, UwedgeV)
vec = vector_from_matrix(Wwedge_UV)
rhs_b = [u_t[i] * dot4(w_t, v_t) - v_t[i] * dot4(w_t, u_t) for i in range(4)]
diff_b = [sp.simplify((vec[i] - rhs_b[i]).subs(subs_t)) for i in range(4)]
print("eq:wp.H3.b check (want zero vector):", diff_b)
assert all(d == 0 for d in diff_b)
