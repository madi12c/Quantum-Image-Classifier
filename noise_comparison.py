"""Depolarizing-noise comparison of six implementations of the P / A measurement (Section 4.2).

Same noise model and settings as noise_simulation.py (one-qubit depolarizing channel of
strength p2/10 after every single-qubit gate, two-qubit depolarizing channel of strength p2
after every CNOT; noise-free input state), but evaluated exactly with the density-matrix
simulator (no shot noise) and for six circuits:

  A. PCM (NumPy completion, generic unitary synthesis) + Grover diffusion
  B. PCM (structured completion, W^dagger) + Grover diffusion
  C. projection onto w of the gap-optimal completion      (inverse state preparation)
  D. projection onto w of the zero-leakage completion     (inverse state preparation)
  E. projection onto the template |P>, "bright"           (inverse state preparation)
  F. projection onto the part of |A> orthogonal to |P>, "dark port"

For each circuit and noise level it reports the probabilities of the distinguished outcome
|00000> for the two class representatives, their gap, and the average error with k = 12
copies under the maximum-likelihood count rule calibrated on the (noisy) template statistics.
Produces noise_comparison.png.  Tested with qiskit 2.6.2, qiskit-aer 0.18.4.
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import qiskit_aer.library  # noqa: F401  (enables set_statevector / save_probabilities)
from qic_common import paper_states, w_kl_grover_opt, w_kl_grover_zl, w_dark, template_error

P, A = paper_states()
N, n = 32, 5
BASIS = ['cx', 'rz', 'sx', 'x']

def tr(qc):
    return transpile(qc, basis_gates=BASIS, optimization_level=3, seed_transpiler=11,
                     qubits_initially_zero=False)

def diffusion(qc):
    qc.h(range(n)); qc.x(range(n)); qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)
    qc.x(range(n)); qc.h(range(n))

# A: NumPy completion, generic synthesis
mu = 0.5 * (P + A)
U, s, Vt = np.linalg.svd(np.column_stack([P - mu, A - mu]), full_matrices=True)
if U[:, 0] @ (P - A) < 0:
    U[:, 0] = -U[:, 0]
cA = QuantumCircuit(n); cA.unitary(U.T.copy(), range(n)); diffusion(cA)
# B: structured completion
cB = QuantumCircuit(n)
cB.h(0); cB.x(0); cB.ch(4, 3); cB.cx(4, 2); cB.cx(4, 1); cB.z(4)
cB.ry(-2 * math.acos(1 / math.sqrt(3)), 4); diffusion(cB)
# C, D: direct projections
def projection(w):
    qc = QuantumCircuit(n); qc.append(StatePreparation(w / np.linalg.norm(w)).inverse(), range(n))
    return qc
circuits = [("A: PCM (generic synthesis) + diffusion", tr(cA)),
            ("B: structured PCM + diffusion", tr(cB)),
            ("C: KL+Grover, gap-optimal $w$ (projection)", tr(projection(w_kl_grover_opt(P, A)))),
            ("D: KL+Grover, zero-leakage $w$ (projection)", tr(projection(w_kl_grover_zl(P, A)))),
            ("E: template projection, bright", tr(projection(P))),
            ("F: template projection, dark port", tr(projection(w_dark(P, A))))]

def noise_model(p2):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ['sx', 'x', 'rz'])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ['cx'])
    return nm

def p0(circ, v, p2):
    qc = QuantumCircuit(n)
    qc.set_statevector(v)
    qc.compose(circ, inplace=True)
    qc.save_probabilities()
    backend = AerSimulator(method='density_matrix', noise_model=noise_model(p2) if p2 > 0 else None)
    return float(np.clip(backend.run(qc).result().data()['probabilities'][0], 0.0, 1.0))

p2s = [0.0, 1e-3, 5e-3, 1e-2]
results = {}
print(f"{'circuit':46s} {'CNOT':>5s} {'p2':>7s} {'p_P':>7s} {'p_A':>7s} {'gap':>6s} {'err k=12':>9s}")
for name, circ in circuits:
    cx = circ.count_ops().get('cx', 0)
    rows = []
    for p2 in p2s:
        pP, pA = p0(circ, P, p2), p0(circ, A, p2)
        err = template_error(pP, pA, 12)
        rows.append((pP, pA, err))
        print(f"{name:46s} {cx:5d} {p2:7.3f} {100*pP:7.2f} {100*pA:7.2f} {100*abs(pP-pA):6.2f} {100*err:9.2f}")
    results[name] = (cx, rows)

labels = ["ideal", "$p_2=10^{-3}$", "$p_2=5\\cdot10^{-3}$", "$p_2=10^{-2}$"]
x = np.arange(len(labels)); wbar = 0.14
colors = ['#c6dbef', '#9ecae1', '#2171b5', '#08306b', '#d62728', '#ff9896']
fig, ax = plt.subplots(1, 2, figsize=(10.5, 5.2))
for i, (name, (cx, rows)) in enumerate(results.items()):
    lab = f"{name}  [{cx} CNOTs]"
    ax[0].bar(x + (i - 2.5) * wbar, [100 * abs(r[0] - r[1]) for r in rows], wbar, color=colors[i], label=lab)
    ax[1].bar(x + (i - 2.5) * wbar, [100 * r[2] for r in rows], wbar, color=colors[i], label=lab)
ax[0].set_ylabel('Discrimination gap  $|p_P - p_A|$  [points]')
ax[1].set_ylabel('Average error with $k=12$ copies [%]'); ax[1].set_yscale('log')
ax[1].axhline(50, color='gray', ls='--', lw=1)
for a in ax:
    a.set_xticks(x); a.set_xticklabels(labels); a.grid(axis='y', alpha=0.3)
fig.suptitle('Class representatives under depolarizing noise (exact density-matrix simulation)')
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=2, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig('noise_comparison.png', dpi=200)
print("wrote noise_comparison.png")
