"""
Problem Set 1 -- Question 2.5
Maximum-likelihood estimation of a simplified two-period Keane-Wolpin (1997) model.

Model
-----
Ages a in {1,2}. At each age the agent chooses m in {1,2,3}:
    1 = white collar, 2 = blue collar, 3 = school.
Flow utilities:
    R_1(a) = alpha_1 + beta_1 G(a) + gamma_1 X_1(a) + eps_1(a)
    R_2(a) = alpha_2 + beta_2 G(a) + gamma_2 X_2(a) + eps_2(a)
    R_3(a) =                                         eps_3(a)
The eps are iid Type-I Extreme Value across alternatives and over time. The
period-2 shocks are unknown in period 1. Discount factor delta in (0,1).

The agent starts period 1 with G = X_1 = X_2 = 0, so the only deterministic
parts in period 1 are the intercepts (v_1 = alpha_1, v_2 = alpha_2, v_3 = 0).
After the period-1 choice exactly one state variable becomes 1:
    chose 1 -> X_1(2)=1 ; chose 2 -> X_2(2)=1 ; chose 3 -> G(2)=1.

Estimation
----------
theta = (alpha_1, beta_1, gamma_1, alpha_2, beta_2, gamma_2, delta).
We optimise over rho = log(delta) so that delta = exp(rho) > 0, and minimise the
negative sample log-likelihood with scipy's BFGS (the analogue of Matlab fminunc).
Standard errors come from the inverse numerical Hessian of the negative
log-likelihood; the SE for delta is obtained by the delta method from rho.

Run:  python3 kw_estimate.py
"""

import numpy as np
from scipy.optimize import minimize

DATA_FILE = "kw_example.csv"


# ----------------------------------------------------------------------------
# Data loading.  The file uses carriage-return ('\r') line endings and each
# record is "C1,C2".  We parse it robustly without assuming a delimiter style.
# ----------------------------------------------------------------------------
def load_data(path=DATA_FILE):
    with open(path, "r") as f:
        raw = f.read()
    # Split into tokens on any of \r \n , (and drop empties).
    tokens = [t for t in raw.replace("\r", "\n").replace(",", "\n").split("\n") if t.strip()]
    # First two tokens are the header "C(1)" and "C(2)".
    assert tokens[0].startswith("C(1)") and tokens[1].startswith("C(2)"), tokens[:2]
    nums = np.array([int(t) for t in tokens[2:]], dtype=int)
    assert nums.size % 2 == 0, "odd number of data entries"
    data = nums.reshape(-1, 2)            # columns: C(1), C(2)
    assert set(np.unique(data)).issubset({1, 2, 3})
    return data[:, 0], data[:, 1]


def logsumexp3(a, b, c):
    """Numerically stable log(exp a + exp b + exp c) for scalars/arrays."""
    m = np.maximum(np.maximum(a, b), c)
    return m + np.log(np.exp(a - m) + np.exp(b - m) + np.exp(c - m))


def model_log_probs(params):
    """
    Given params = (alpha_1, beta_1, gamma_1, alpha_2, beta_2, gamma_2, rho),
    return:
      logP1  : length-3 array, log P(C1 = m), m = 1,2,3
      logP2  : 3x3 array, logP2[s, m] = log P(C2 = m+1 | state set by C1 = s+1)
    States after period 1 (rows of logP2):
      s=0 (C1=1, white collar): X_1(2)=1, G=0, X_2=0
      s=1 (C1=2, blue  collar): X_2(2)=1, G=0, X_1=0
      s=2 (C1=3, school):       G(2)=1,  X_1=0, X_2=0
    """
    a1, b1, g1, a2, b2, g2, rho = params
    delta = np.exp(rho)

    # ---- Period 2 deterministic utilities v_m(2) in each of the 3 states. ----
    # state s gives (G, X1, X2):
    G  = np.array([0.0, 0.0, 1.0])   # education = 1 only after schooling
    X1 = np.array([1.0, 0.0, 0.0])   # white-collar experience
    X2 = np.array([0.0, 1.0, 0.0])   # blue-collar experience

    v1_2 = a1 + b1 * G + g1 * X1     # white collar, by state
    v2_2 = a2 + b2 * G + g2 * X2     # blue collar, by state
    v3_2 = np.zeros(3)               # school

    # E[V_2(state)] = log( e^{v1} + e^{v2} + e^{v3} )  (Euler constant dropped).
    EV2 = logsumexp3(v1_2, v2_2, v3_2)            # length-3, one per state

    denom2 = EV2                                   # the log-denominator per state
    logP2 = np.column_stack((v1_2 - denom2,
                             v2_2 - denom2,
                             v3_2 - denom2))        # 3x3 : [state, choice]

    # ---- Period 1 choice-specific values.  Starting state is empty, so the
    # period-1 flow deterministic parts are the intercepts only. The agent
    # adds the discounted expected continuation value of the state its choice
    # leads to: choice 1 -> state 0, choice 2 -> state 1, choice 3 -> state 2.
    w1 = a1 + delta * EV2[0]
    w2 = a2 + delta * EV2[1]
    w3 = 0.0 + delta * EV2[2]

    denom1 = logsumexp3(w1, w2, w3)
    logP1 = np.array([w1 - denom1, w2 - denom1, w3 - denom1])

    return logP1, logP2


def neg_loglik(params, C1, C2):
    logP1, logP2 = model_log_probs(params)
    # Vectorised contribution: period-1 term + period-2 term conditional on the
    # state implied by the period-1 choice.
    ll1 = logP1[C1 - 1]
    ll2 = logP2[C1 - 1, C2 - 1]
    return -(ll1 + ll2).sum()


def numerical_hessian(fun, x, eps=1e-4):
    """Symmetric finite-difference Hessian of a scalar function."""
    n = x.size
    H = np.zeros((n, n))
    f0 = fun(x)
    for i in range(n):
        for j in range(i, n):
            xi = x.copy(); xi[i] += eps; xi[j] += eps
            xj = x.copy(); xj[i] += eps; xj[j] -= eps
            xk = x.copy(); xk[i] -= eps; xk[j] += eps
            xl = x.copy(); xl[i] -= eps; xl[j] -= eps
            H[i, j] = (fun(xi) - fun(xj) - fun(xk) + fun(xl)) / (4 * eps * eps)
            H[j, i] = H[i, j]
    return H


def main():
    C1, C2 = load_data()
    n = C1.size
    print(f"Loaded {n} observations from {DATA_FILE}.\n")

    # Empirical distribution of the 9 (C1,C2) cells (used as a sanity check).
    emp = np.zeros((3, 3))
    for s in (1, 2, 3):
        for m in (1, 2, 3):
            emp[s - 1, m - 1] = np.mean((C1 == s) & (C2 == m))

    # ---- Optimise.  Search over (alpha1,beta1,gamma1,alpha2,beta2,gamma2,rho). ----
    # The log-likelihood is flat in delta (delta is only weakly identified -- see
    # the large SE below), so a single optimiser run can stall in that direction.
    # We therefore run BFGS from several starting values and keep the global best.
    rng = np.random.default_rng(0)
    starts = [np.zeros(7)]
    for d0 in (0.5, 0.8, 0.95):
        s = np.zeros(7); s[6] = np.log(d0); starts.append(s)
    for _ in range(6):
        starts.append(np.concatenate([rng.normal(0, 0.5, 6), [np.log(rng.uniform(0.3, 0.99))]]))

    res = None
    for s0 in starts:
        r = minimize(neg_loglik, s0, args=(C1, C2), method="BFGS",
                     options={"maxiter": 2000, "gtol": 1e-8})
        if res is None or r.fun < res.fun:
            res = r
    grad_norm = np.linalg.norm(res.jac)
    print(f"Best of {len(starts)} starts. BFGS flag: {res.success} ({res.message})")
    print(f"Gradient norm at optimum: {grad_norm:.2e}")
    print(f"Maximised log-likelihood: {-res.fun:.4f}\n")

    est = res.x.copy()
    # ---- Standard errors from the inverse numerical Hessian of the NLL. ----
    H = numerical_hessian(lambda p: neg_loglik(p, C1, C2), est)
    cov = np.linalg.inv(H)
    se_raw = np.sqrt(np.diag(cov))

    # Convert rho = log(delta) back to delta with the delta method:
    # delta = exp(rho)  ->  SE(delta) = exp(rho) * SE(rho).
    a1, b1, g1, a2, b2, g2, rho = est
    delta = np.exp(rho)
    se_delta = delta * se_raw[6]

    names   = ["alpha_1", "beta_1", "gamma_1", "alpha_2", "beta_2", "gamma_2", "delta"]
    values  = [a1, b1, g1, a2, b2, g2, delta]
    ses     = [se_raw[0], se_raw[1], se_raw[2], se_raw[3], se_raw[4], se_raw[5], se_delta]

    print(f"{'parameter':>10}  {'estimate':>10}  {'std. error':>10}")
    print("-" * 36)
    for nm, v, s in zip(names, values, ses):
        print(f"{nm:>10}  {v:>10.4f}  {s:>10.4f}")
    print(f"\n(log delta = {rho:.4f}, SE = {se_raw[6]:.4f})\n")

    # ---- Sanity check: model-implied vs empirical (C1,C2) cell probabilities. ----
    logP1, logP2 = model_log_probs(est)
    P1 = np.exp(logP1)
    P2 = np.exp(logP2)
    model_cells = P1[:, None] * P2          # [state s = C1, choice m = C2]
    print("Cell probabilities  P(C1=s, C2=m)   [rows C1=1,2,3 ; cols C2=1,2,3]")
    print("  empirical:")
    print(np.array2string(emp, precision=4, floatmode="fixed"))
    print("  model-implied at estimates:")
    print(np.array2string(model_cells, precision=4, floatmode="fixed"))
    print(f"  max abs difference: {np.max(np.abs(emp - model_cells)):.4f}\n")

    # ---- Emit a LaTeX tabular snippet for the writeup. ----
    print("% ---- LaTeX table (paste into ps1_solutions.tex) ----")
    print(r"\begin{tabular}{lcc}")
    print(r"\hline\hline")
    print(r"Parameter & Estimate & Std.\ Error \\ \hline")
    tex_names = [r"$\alpha_1$ (white collar intercept)",
                 r"$\beta_1$ (return to education, WC)",
                 r"$\gamma_1$ (return to WC experience)",
                 r"$\alpha_2$ (blue collar intercept)",
                 r"$\beta_2$ (return to education, BC)",
                 r"$\gamma_2$ (return to BC experience)",
                 r"$\delta$ (discount factor)"]
    for nm, v, s in zip(tex_names, values, ses):
        print(f"{nm} & {v:.4f} & {s:.4f} " + r"\\")
    print(r"\hline")
    print(f"\\multicolumn{{3}}{{l}}{{Log-likelihood $= {-res.fun:.2f}$, $N = {n}$}} \\\\")
    print(r"\hline\hline")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
