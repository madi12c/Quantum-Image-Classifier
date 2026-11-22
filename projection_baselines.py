"""Single-copy projective baselines for the P / A task (Section 3.2 and Table 1).

1. Verifies numerically that the proposed classifier (PCM followed by the Grover
   diffusion D and a computational-basis measurement, outcome |00000>) is the
   rank-one projective measurement onto  w = PCM^T D |0>,  i.e.
   P(|00000> | v) = (w . v)^2  for every input v.
2. Compares, on the two class representatives and for k independent copies,
     - the proposed scheme with three null-space completions
       (Mathematica values reported in the paper, NumPy SVD, gap-optimal),
     - projection onto a template (the baseline requested by Reviewer 3),
     - the single-copy Helstrom projector repeated on every copy,
     - the collective Helstrom bound.
   All count-based rules use the maximum-likelihood threshold (equal priors).
Produces baseline_comparison.png and prints the rows of Table 1.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qic_common import (paper_states, directions, w_from_pcm, w_kl_grover_opt, w_kl_grover_zl,
                        w_template, w_dark, w_helstrom, template_error, collective_helstrom,
                        copies_needed, ml_threshold, pcm_from_free_direction, zero_leakage_exists)

P, A = paper_states()
N = 32
c = float(P @ A)
print(f"overlap <P|A> = {c}  (13/16 = {13/16})")

# --- NumPy-SVD completion, exactly as in quantum_kl_grover.py / noise_simulation.py
mu = 0.5 * (P + A)
U, s, Vt = np.linalg.svd(np.column_stack([P - mu, A - mu]), full_matrices=True)
if U[:, 0] @ (P - A) < 0:
    U[:, 0] = -U[:, 0]
PCM_np = U.T.copy()
D = 2.0 / N * np.ones((N, N)) - np.eye(N)

# --- 1. the classifier is a projection onto w
w_np = w_from_pcm(PCM_np)
rng = np.random.default_rng(1)
for _ in range(1000):
    v = rng.normal(size=N); v /= np.linalg.norm(v)
    assert abs((D @ PCM_np @ v)[0] ** 2 - (w_np @ v) ** 2) < 1e-12
assert abs(np.linalg.norm(w_np) - 1) < 1e-12
e1, es = directions(P, A)
print(f"identity P(|0>|v) = (w.v)^2 verified on 1000 random states; |w| = 1; "
      f"w.e1 = {w_np @ e1:+.4f} (= -(1-2/N) = {-(1-2/N):+.4f})")

# --- zero-leakage completion: explicit PCM, checked against the closed form
assert zero_leakage_exists(P, A)
w_zl = w_kl_grover_zl(P, A)
f = (w_zl + (1 - 2 / N) * e1) * N / (2 * np.sqrt(N - 1))
PCM_zl = pcm_from_free_direction(e1, f)
assert np.allclose(w_from_pcm(PCM_zl), w_zl) and np.allclose(PCM_zl[0], e1)
delta2 = float(np.sum((P - A) ** 2))
print(f"zero-leakage completion: explicit orthogonal PCM built; P(|0>|P) = {(w_zl @ P)**2:.2e}, "
      f"P(|0>|A) = {(w_zl @ A)**2:.6f}  (closed form ((N-2)/N)^2 delta^2 = {((N-2)/N)**2*delta2:.6f}); "
      f"existence bound delta^2 = {delta2:.4f} <= 16(N-1)/N^2 = {16*(N-1)/N**2:.4f}")

# structured completion (3-CNOT PCM of circuit_resources.py):  PCM = W^dagger, W|0> = e1
def w_structured():
    v = np.zeros(N); v[0] = 1.0                      # W D |0> computed with explicit gates
    v = D @ v
    psi = v.reshape([2] * 5)                         # axes: x1 x0 y1 y0 c
    Hm = np.array([[1, 1], [1, -1]]) / np.sqrt(2); X = np.array([[0, 1], [1, 0]]); Z = np.diag([1., -1.])
    th = 2 * np.arccos(1 / np.sqrt(3)); RY = np.array([[np.cos(th/2), -np.sin(th/2)], [np.sin(th/2), np.cos(th/2)]])
    def one(psi, g, ax): return np.moveaxis(np.tensordot(g, psi, axes=([1], [ax])), 0, ax)
    def ctrl(psi, g, c, t):
        out = psi.copy(); sl = [slice(None)] * 5; sl[c] = 1
        sub = psi[tuple(sl)]; tt = t if t < c else t - 1
        out[tuple(sl)] = np.moveaxis(np.tensordot(g, sub, axes=([1], [tt])), 0, tt)
        return out
    psi = one(psi, RY, 0); psi = one(psi, Z, 0)
    psi = ctrl(psi, X, 0, 3); psi = ctrl(psi, X, 0, 2); psi = ctrl(psi, Hm, 0, 1)
    psi = one(psi, X, 4); psi = one(psi, Hm, 4)
    return psi.ravel()
w_st = w_structured()

schemes = [
    ("KL+Grover, Mathematica completion (paper)", 0.00195, 0.28088),
    ("KL+Grover, NumPy-SVD completion", (w_np @ P) ** 2, (w_np @ A) ** 2),
    ("KL+Grover, structured completion", (w_st @ P) ** 2, (w_st @ A) ** 2),
    ("KL+Grover, gap-optimal completion", *[(w_kl_grover_opt(P, A) @ x) ** 2 for x in (P, A)]),
    ("KL+Grover, zero-leakage completion", *[(w_zl @ x) ** 2 for x in (P, A)]),
    ("Template projection onto |P> (bright)", *[(w_template(P, A) @ x) ** 2 for x in (P, A)]),
    ("Template projection, dark port", *[(w_dark(P, A) @ x) ** 2 for x in (P, A)]),
    ("Single-copy Helstrom projector", *[(w_helstrom(P, A) @ x) ** 2 for x in (P, A)]),
]

print("\nTable 1 (all values in %; error = average over the two templates, ML count rule)")
print(f"{'scheme':44s} {'p_P':>7s} {'p_A':>7s} {'gap':>6s} {'k=1':>7s} {'k=5':>7s} {'k=12':>7s} {'k(1%)':>6s}")
for name, pP, pA in schemes:
    errs = [100 * template_error(pP, pA, k) for k in (1, 5, 12)]
    k1 = copies_needed(pP, pA, 0.01)
    print(f"{name:44s} {100*pP:7.2f} {100*pA:7.2f} {100*abs(pP-pA):6.2f} "
          f"{errs[0]:7.2f} {errs[1]:7.2f} {errs[2]:7.3f} {k1:6d}")
hel = [100 * collective_helstrom(c, k) for k in (1, 5, 12)]
k1 = next(k for k in range(1, 100) if collective_helstrom(c, k) <= 0.01)
print(f"{'Collective Helstrom bound (joint meas.)':44s} {'':7s} {'':7s} {'':6s} "
      f"{hel[0]:7.2f} {hel[1]:7.2f} {hel[2]:7.3f} {k1:6d}")

print("\nML thresholds at k = 12 (decide the high-p class iff m >= t):")
for name, pP, pA in schemes:
    t, hi = ml_threshold(pP, pA, 12)
    print(f"  {name:44s} t = {t:2d}  high-p class = {'P' if hi == 1 else 'A'}")

# --- figure
ks = np.arange(1, 21)
styles = ['o-', 'v-', '<-', '^-', '*-', 's-', 'x:', 'D-']
colors = ['#6baed6', '#9ecae1', '#c6dbef', '#2171b5', '#08306b', '#d62728', '#ff9896', '#2ca02c']
plt.figure(figsize=(7.6, 4.8))
for (name, pP, pA), st, col in zip(schemes, styles, colors):
    plt.semilogy(ks, [100 * template_error(pP, pA, k) for k in ks], st, color=col, ms=4, lw=1.2, label=name)
plt.semilogy(ks, [100 * collective_helstrom(c, k) for k in ks], 'k--', lw=1.2,
             label='Collective Helstrom bound (joint measurement)')
plt.xlabel('Number of copies  $k$'); plt.ylabel('Average misclassification rate [%]')
plt.title('Discriminating the two class representatives with single-copy projections')
plt.xticks(ks[::2]); plt.grid(True, which='both', alpha=0.3); plt.legend(fontsize=7.5)
plt.tight_layout(); plt.savefig('baseline_comparison.png', dpi=200)
print("\nwrote baseline_comparison.png")
