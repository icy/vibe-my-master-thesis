"""
Full symbolic derivation of eq:U (ch2s2.tex):

    U := A*G_z + conj(B)_z  =?=  b/(2(a+b)) * rho * G_z

given A, B from the closed-form Omega (eq:F.Omega) and the modified
Liouville PDE that rho satisfies (eq:L.mod.completed):

    4*(log rho)_{z zbar} = -a/(a+b) * rho^2 * |G_z|^2.

Genuine Wirtinger calculus: z, zbar are independent formal symbols;
rho = rho(z,zbar) is a real sympy Function of both; G = G(z) is a
sympy Function of z ALONE (encoding meromorphicity: d/dzbar G == 0
automatically, since G doesn't depend on the symbol zbar at all);
Gbar = Gbar(zbar), a separate Function of zbar alone, represents
conj(G) -- for a holomorphic G, conj(G) is anti-holomorphic, i.e. a
function of zbar alone, and d/dzbar(Gbar) = conj(G_z) (a standard,
elementary Wirtinger-calculus fact, needs no extra assumption).
"""
import sympy as sp

z, zbar = sp.symbols('z zbar')
a, b = sp.symbols('a b', real=True)

rho = sp.Function('rho', real=True)(z, zbar)
G = sp.Function('G')(z)
Gbar = sp.Function('Gbar')(zbar)          # = conj(G), a function of zbar alone

rho_z, rho_zb = sp.diff(rho, z), sp.diff(rho, zbar)
rho_zzb = sp.diff(rho, z, zbar)
G_z = sp.diff(G, z)
Gbar_z = sp.diff(Gbar, zbar)              # = conj(G_z)

modG2 = G_z * Gbar_z                       # = |G_z|^2

# --- the Liouville PDE, solved for rho_zzbar ---
# 4*(log rho)_zzbar = -a/(a+b) rho^2 |G_z|^2
logrho_zzb = sp.diff(sp.log(rho), z, zbar).doit()
pde_lhs = 4 * logrho_zzb
rho_zzb_solution = sp.solve(sp.Eq(pde_lhs, -a/(a+b) * rho**2 * modG2), rho_zzb)[0]

def apply_pde(expr):
    return expr.subs(rho_zzb, rho_zzb_solution)

# --- Omega's entries, eq:F.Omega ---
A = (2*a+b)/(2*(a+b)) * rho + 2*rho_zzb / (rho**2 * modG2)
B = (2*rho_z) / (rho**2 * G_z)
Bbar = (2*rho_zb) / (rho**2 * Gbar_z)      # = conj(B)

Bbar_z = sp.diff(Bbar, z)                  # differentiate the CONJUGATE quantity w.r.t. z

U = A * G_z + Bbar_z
U_after_pde = apply_pde(sp.expand(U))
U_after_pde = sp.simplify(U_after_pde)

target = b/(2*a+2*b) * rho * G_z

print("U (raw, before PDE substitution):")
sp.pprint(sp.simplify(U))
print("\nU after substituting the Liouville PDE:")
sp.pprint(U_after_pde)
print("\ndifference from target b/(2(a+b)) * rho * G_z  (want 0):")
print(sp.simplify(U_after_pde - target))
