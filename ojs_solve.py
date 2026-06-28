r"""
Problem Set 2 -- numerical companion.

Q3(g): Solve the on-the-job-search value function U(w) and the value of
       unemployment V by successive approximation on the contraction

           U(w) = (w-c)/(r+d) + d*U(b+c)/(r+d)
                  + lambda*(1-d)/(r+d) * \int_w^wbar (U(x)-U(w)) f(x) dx .

       Reservation wage w* = b + c.  Parameters:
       b=800, c=0, lambda=0.2, r=0.02, d(=delta)=0.25, F=N(1200,400), wbar=2500.

Also produces the Q3 value-function figure and the two Q4 monopsony figures,
and reports the Q2 elasticity of substitution between Hippies and Nerds.
"""
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Q3(g): on-the-job search
# ----------------------------------------------------------------------
b, c, lam, r, delta = 800.0, 0.0, 0.2, 0.02, 0.25
wbar = 2500.0
wstar = b + c                       # reservation wage = 800

N = 2501                            # grid 0,1,...,2500 (step 1, includes 800)
w = np.linspace(0.0, wbar, N)
dw = w[1] - w[0]
istar = int(round(wstar / dw))      # index of the reservation wage
assert abs(w[istar] - wstar) < 1e-9

f = norm.pdf(w, loc=1200.0, scale=400.0)   # offer density on the grid

def integral_above(g):
    r"""Return vector I[i] = \int_{w_i}^{wbar} g(x) f(x) dx via trapezoid,
    computed as reverse-cumulative trapezoid sums."""
    h = g * f                              # integrand
    # trapezoid contribution of each cell [w_i, w_{i+1}]
    cell = 0.5 * (h[1:] + h[:-1]) * dw     # length N-1
    tail = np.zeros(N)
    tail[:-1] = np.cumsum(cell[::-1])[::-1]
    return tail                            # tail[i] = int_{w_i}^{wbar}

def T(U):
    """One application of the contraction operator (eq. base_f)."""
    Vguess = U[istar]                      # V = U(b+c)
    int_U = integral_above(U)              # \int_w^wbar U(x) f(x) dx
    int_f = integral_above(np.ones(N))     # \int_w^wbar f(x) dx = 1-F(w)
    # \int_w^wbar (U(x)-U(w)) f(x) dx = int_U - U(w)*int_f
    bracket = int_U - U * int_f
    return (w - c) / (r + delta) + delta * Vguess / (r + delta) \
        + lam * (1 - delta) / (r + delta) * bracket

# successive approximation
U = (w - c) / (r + delta)                  # initial guess
for it in range(200000):
    Un = T(U)
    diff = np.max(np.abs(Un - U))
    U = Un
    if diff < 1e-9:
        break
V = U[istar]
print(f"Q3(g) converged in {it} iterations, sup-norm change = {diff:.2e}")
print(f"  reservation wage w*        = {wstar:.1f}")
print(f"  V = U(w*) = U({wstar:.0f})        = {V:,.2f}")
print(f"  U(0)                       = {U[0]:,.2f}")
print(f"  U(wbar)=U(2500)            = {U[-1]:,.2f}")

# ---- cross-check 1: closed form U(wbar) = (wbar - c + delta V)/(r+delta) ----
U_wbar_cf = (wbar - c + delta * V) / (r + delta)
print(f"  U(2500) closed form        = {U_wbar_cf:,.2f}  (check)")

# ---- cross-check 2: iterate the ORIGINAL Bellman (eq. A), modulus 1/(1+r) ----
def T_bellman(U):
    Vg = U[istar]
    int_U = integral_above(U)                       # \int_w^wbar U f
    int_f = integral_above(np.ones(N))              # 1-F(w)
    cont = lam * int_U + (1 - lam * int_f) * U       # lam∫Uf + (1-lam(1-F))U
    return (w - c) / (1 + r) + delta * Vg / (1 + r) + (1 - delta) / (1 + r) * cont
U2 = (w - c) / (r + delta)
for it2 in range(500000):
    Un = T_bellman(U2)
    d2 = np.max(np.abs(Un - U2)); U2 = Un
    if d2 < 1e-9:
        break
print(f"  Bellman-form V             = {U2[istar]:,.2f}  (check, {it2} its)")
print(f"  max|U_basef - U_bellman|   = {np.max(np.abs(U-U2)):.3e}")

# ---- cross-check 3: U'(wbar) should equal 1/(r+delta) ----
slope_num = (U[-1] - U[-2]) / dw
print(f"  U'(2500) numeric           = {slope_num:.4f},  1/(r+delta) = {1/(r+delta):.4f}")

# ----------------------------------------------------------------------
# Q3(d): unemployment duration and expected exit wage
# ----------------------------------------------------------------------
Fwstar = norm.cdf(wstar, 1200, 400)
exit_rate = lam * (1 - Fwstar)
dur = 1.0 / exit_rate
# E[w | w >= w*]  using the grid density (renormalised on the truncation)
mask = w >= wstar
exit_wage = np.trapz(w[mask] * f[mask], w[mask]) / np.trapz(f[mask], w[mask])
print(f"\nQ3(d) 1-F(w*) = {1-Fwstar:.4f}, exit hazard = {exit_rate:.4f}")
print(f"  expected unemployment duration = {dur:.2f} months")
print(f"  expected exit wage E[w|w>=w*]  = {exit_wage:,.2f}")

# ----------------------------------------------------------------------
# Figure Q3: value function U(w) and V
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(w, U, lw=2, label=r"$U(w)$")
ax.axhline(V, color="C3", ls="--", lw=1.5, label=r"$V=U(w^{*})$")
ax.axvline(wstar, color="grey", ls=":", lw=1)
ax.plot([wstar], [V], "ko", ms=5)
ax.annotate(r"$w^{*}=b+c=800$", xy=(wstar, V), xytext=(950, V - 7000),
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xlabel("wage $w$ (CAD/month)")
ax.set_ylabel("value")
ax.set_title("On-the-job search: $U(w)$ is increasing and convex; $V$ horizontal")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("q3_value_function.pdf")
print("\nsaved q3_value_function.pdf")

# ----------------------------------------------------------------------
# Q4: monopsony with profit-insensitive supply  L(w)=K/(P-w),  w(L)=P-K/L
# ----------------------------------------------------------------------
P, K = 10.0, 4.0
L = np.linspace(0.5, 8.0, 400)
w_supply = P - K / L                       # inverse supply
# Fig Q4(d): inverse supply, MFC=P, inverse demand=P (multiple equilibria)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(L, w_supply, lw=2, label=r"inverse supply $w(L)=P-K/L$")
ax.axhline(P, color="C1", lw=2, ls="--", label=r"MFC $=P$ = inverse demand (MRPL$=P$)")
ax.set_ylim(0, P * 1.15)
ax.set_xlabel("employment $L$ (firm size)")
ax.set_ylabel("wage $w$")
ax.set_title("Q4(d): MFC coincides with labour demand $\\Rightarrow$ continuum of equilibria")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig("q4_multiple_equilibria.pdf")
print("saved q4_multiple_equilibria.pdf")

# Fig Q4(e): minimum wage at P/2 raises employment
wlow = 0.3 * P                             # some w* < P/2
Llow = K / (P - wlow)
wmin = 0.5 * P
Lmin = K / (P - wmin)
print(f"\nQ4(e) P={P}, K={K}: start w*={wlow:.2f} -> L*={Llow:.3f};"
      f"  min wage={wmin:.2f} -> L={Lmin:.3f}  (employment rises)")
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(L, w_supply, lw=2, label=r"inverse supply $w(L)=P-K/L$")
ax.axhline(P, color="C1", lw=1.2, ls="--", label="MRPL $=P$")
ax.axhline(wmin, color="C2", lw=1.5, ls="-.", label=r"minimum wage $=P/2$")
for (Lp, wp, txt, col) in [(Llow, wlow, r"$(L^{*},w^{*})$", "k"),
                           (Lmin, wmin, r"$(L',P/2)$", "C2")]:
    ax.plot([Lp], [wp], "o", color=col, ms=6)
ax.annotate("", xy=(Lmin, wmin), xytext=(Llow, wlow),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="C3"))
ax.text(0.55*(Llow+Lmin), 0.5*(wlow+wmin)-0.6, "employment\n rises", color="C3")
ax.set_ylim(0, P * 1.15)
ax.set_xlabel("employment $L$ (firm size)")
ax.set_ylabel("wage $w$")
ax.set_title("Q4(e): a minimum wage at $P/2$ raises employment under monopsony")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig("q4_minimum_wage.pdf")
print("saved q4_minimum_wage.pdf")

# ----------------------------------------------------------------------
# Q2: elasticity of substitution between Hippies (A=2,M=1) and Nerds (A=1,M=3)
#     Q = (2H+N)^a (H+3N)^b (2H+3N)^(1-a-b);  sigma = F_H F_N /(F F_HN)
# ----------------------------------------------------------------------
def sigma_HN(H, Nn, a, bb):
    g = 1 - a - bb
    Pp, Rr, Ss = 2*H + Nn, H + 3*Nn, 2*H + 3*Nn
    thH = 2*a/Pp + bb/Rr + 2*g/Ss
    thN = a/Pp + 3*bb/Rr + 3*g/Ss
    D = 2*a/Pp**2 + 3*bb/Rr**2 + 6*g/Ss**2
    return thH*thN / (thH*thN - D)
print("\nQ2 elasticity of substitution sigma_HN (a=b=1/3):")
for (H, Nn) in [(1,1),(1,2),(2,1),(1,5),(5,1)]:
    print(f"  H={H}, N={Nn}: sigma = {sigma_HN(H,Nn,1/3,1/3):.4f}")
print("  (a+b=1, g=0) at H=N=1:", f"{sigma_HN(1,1,0.5,0.5):.4f}",
      " at H=1,N=3:", f"{sigma_HN(1,3,0.5,0.5):.4f}")
