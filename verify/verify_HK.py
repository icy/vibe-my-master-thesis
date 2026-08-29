"""
Full symbolic proof (Cramer's rule, no shortcuts) of the H, K formulas
and the BLW cancellation 2a(H-1)+b(K-1)=0, from the linear system
eq:sys.eta in ch2s2.tex:

    -H*B_z + Upsilon*conj(U)     = -B_z
    -H*U   + Upsilon*conj(B_z)   = rho*G_z - U

solved for H (real) and Upsilon (complex), with K := H^2 - |Upsilon|^2,
THEN substituting U's own formula eq:U (U = b/(2(a+b)) rho G_z) --
exactly the two-step process the thesis itself does (solve the system
generically, *then* substitute U's formula to get the clean closed form).
"""
import sympy as sp

a, b, rho = sp.symbols('a b rho', real=True)
G_z, G_zb, B_z, B_zb, U, Ub = sp.symbols('G_z G_zb B_z B_zb U Ub')

U_expr, Ub_expr = (b/(2*a+2*b))*rho*G_z, (b/(2*a+2*b))*rho*G_zb
subs_U = {U: U_expr, Ub: Ub_expr}

M = sp.Matrix([[-B_z, Ub], [-U, B_zb]])
rhs = sp.Matrix([-B_z, rho*G_z - U])
sol = M.solve(rhs)
H_sol, Ups_sol = sol[0], sol[1]
Upsb_sol = Ups_sol.subs({U: Ub, Ub: U, B_z: B_zb, B_zb: B_z, G_z: G_zb, G_zb: G_z}, simultaneous=True)
K_sol = H_sol**2 - Ups_sol*Upsb_sol

# -- NOW substitute U's own formula (the step the thesis's 2nd equality does) --
H_sol_c = sp.cancel(H_sol.subs(subs_U))
K_sol_c = sp.cancel(K_sol.subs(subs_U))

D = sp.cancel((-B_z*B_zb + U*Ub).subs(subs_U))    # D = |U|^2-|B_z|^2 after substitution
H_target = 1 - b/(2*(a+b)) * rho**2*G_z*G_zb / D
K_target = 1 + a/(a+b) * rho**2*G_z*G_zb / D

print("H_sol - H_target  (want 0):", sp.simplify(sp.cancel(H_sol_c - H_target)))
print("K_sol - K_target  (want 0):", sp.simplify(sp.cancel(K_sol_c - K_target)))

blw = sp.cancel(2*a*(H_sol_c - 1) + b*(K_sol_c - 1))
print("2a(H-1)+b(K-1), fully substituted (want 0):", sp.simplify(blw))
