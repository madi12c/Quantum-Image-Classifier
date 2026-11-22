"""Verified circuit resources for the P / A classifier and its baselines (Section 4.1, Table 2).

Every circuit is (i) checked against its target (unitary equivalence up to a global
phase, or the prepared state), (ii) transpiled with ONE setting, and (iii) reported
with ONE set of definitions:

  basis gates        : {CNOT, R_z, sqrt(X), X}, all-to-all connectivity, no ancillas
  transpiler         : qiskit.transpile(optimization_level=3, seed_transpiler=11,
                                        qubits_initially_zero=False)
  CNOT count         : number of CNOT gates
  1q count           : number of single-qubit gates
  depth              : QuantumCircuit.depth() over all gates (longest path)
  CNOT depth         : longest path counting CNOT gates only

For an n-qubit circuit at most floor(n/2) CNOTs fit in one layer, hence
CNOT depth >= ceil(#CNOT / floor(n/2)); this sanity check is asserted for every row.

Qubit convention (Qiskit little-endian): q0 = colour c, q1 = y0, q2 = y1, q3 = x0, q4 = x1,
so that basis index = int('x1 x0 y1 y0 c', 2) as in the paper.

Produces pcm_structured_circuit.png and prints the rows of Table 2.
Tested with qiskit 2.6.2.
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Operator, Statevector
from qic_common import (paper_states, directions, w_from_pcm, w_helstrom, w_kl_grover_zl, w_dark,
                        template_error)

P, A = paper_states()
N, n = 32, 5
e1, es = directions(P, A)
D = 2.0 / N * np.ones((N, N)) - np.eye(N)
BASIS = ['cx', 'rz', 'sx', 'x']

def tr(qc):
    # qubits_initially_zero=False: the classifier acts on an arbitrary input state, so the
    # transpiler must not borrow untouched qubits as clean |0> ancillas for multi-controlled gates.
    return transpile(qc, basis_gates=BASIS, optimization_level=3, seed_transpiler=11,
                     qubits_initially_zero=False)

def resources(qc):
    t = tr(qc)
    # from_circuit() accounts for the output-qubit permutation the transpiler may elide
    assert Operator.from_circuit(t).equiv(Operator(qc)), "transpiled circuit differs from its source"
    ops = t.count_ops()
    cx = ops.get('cx', 0)
    oneq = sum(v for g, v in ops.items() if g != 'cx')
    depth = t.depth()
    cxdepth = t.depth(lambda inst: inst.operation.num_qubits == 2)
    assert cxdepth >= math.ceil(cx / (n // 2)) and depth >= cxdepth
    return cx, oneq, depth, cxdepth

def gap_of(PCM):
    w = w_from_pcm(PCM)
    return (w @ P) ** 2, (w @ A) ** 2

# ----------------------------------------------------------------------------- Grover diffusion
def diffusion():
    qc = QuantumCircuit(n, name='InvAvg')
    qc.h(range(n)); qc.x(range(n))
    qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)
    qc.x(range(n)); qc.h(range(n))
    return qc
qD = diffusion()
assert Operator(qD).equiv(Operator(D))

# ----------------------------------------------------------------------------- PCM, NumPy-SVD completion (generic synthesis)
mu = 0.5 * (P + A)
U, s, Vt = np.linalg.svd(np.column_stack([P - mu, A - mu]), full_matrices=True)
if U[:, 0] @ (P - A) < 0:
    U[:, 0] = -U[:, 0]
PCM_np = U.T.copy()
q_np = QuantumCircuit(n); q_np.unitary(PCM_np, range(n), label='PCM')
rank_np = np.linalg.matrix_rank(PCM_np - np.eye(N), tol=1e-9)
supp_np = int(np.sum(np.abs(PCM_np - np.eye(N)).sum(axis=0) > 1e-9))

# ----------------------------------------------------------------------------- PCM, gap-optimal completion (generic synthesis)
def pcm_optimal():
    M = np.zeros((N, N)); M[0] = e1
    # orthonormal basis G of the complement of span{e1, es}
    Q, _ = np.linalg.qr(np.column_stack([e1, es, np.random.default_rng(0).normal(size=(N, N - 2))]))
    G = Q[:, 2:]
    H = np.eye(N - 1) - np.ones((N - 1, N - 1)) / (N - 1)
    Us, ss, _ = np.linalg.svd(H)
    S = Us[:, :N - 2]                       # rows: simplex vertices sigma_k
    for k in range(N - 1):
        M[k + 1] = es / np.sqrt(N - 1) + G @ S[k]
    assert np.allclose(M @ M.T, np.eye(N), atol=1e-10)
    return M
PCM_opt = pcm_optimal()
q_opt = QuantumCircuit(n); q_opt.unitary(PCM_opt, range(n), label='PCM_opt')

# ----------------------------------------------------------------------------- PCM, structured completion
# e1 = (1/sqrt 3)(|0000> - |1011> - |1111>)_{x1 x0 y1 y0} (x) |->_c  is a product of a three-term
# position state and |-> on the colour qubit.  W prepares e1 from |00000>; PCM := W^dagger.
def w_prep():
    qc = QuantumCircuit(n, name='W')
    qc.ry(2 * math.acos(1 / math.sqrt(3)), 4)   # x1: (1/sqrt3)|0> + sqrt(2/3)|1>
    qc.z(4)                                     #     relative sign of the |1> branch
    qc.cx(4, 1); qc.cx(4, 2)                    # y0 = y1 = x1
    qc.ch(4, 3)                                 # x0 -> |+> when x1 = 1
    qc.x(0); qc.h(0)                            # colour qubit -> |->
    return qc
W = w_prep()
sv = Statevector(W).data.real
assert np.allclose(sv, e1) or np.allclose(sv, -e1), "W does not prepare the principal direction"
q_st = W.inverse(); q_st.name = 'PCM'
PCM_st = Operator(q_st).data.real
assert np.allclose(np.abs(PCM_st[0]), np.abs(e1)) and np.allclose(PCM_st @ PCM_st.T, np.eye(N))

# ----------------------------------------------------------------------------- template projection
def anf(truth):
    """Algebraic normal form (Moebius transform) of a Boolean function given as a truth table."""
    a = list(truth)
    m = len(a).bit_length() - 1
    for i in range(m):
        for j in range(len(a)):
            if j >> i & 1:
                a[j] ^= a[j ^ (1 << i)]
    return [j for j, v in enumerate(a) if v]

def template_unprep(psi):
    """Oracle-style template projection: V with V|psi> = |00000>.

    psi = (1/4) sum_{x,y} |x, y>|f(x, y)>, so V uncomputes the colour bit through the
    algebraic normal form of f (X / CNOT / multi-controlled-X gates, all self-inverse and
    mutually commuting) and then applies H to the four position qubits.
    """
    truth = [0] * 16
    for idx in np.nonzero(psi)[0]:
        truth[idx >> 1] = idx & 1                 # position word (q4 q3 q2 q1) -> colour bit
    qc = QuantumCircuit(n, name='V_T')
    for mono in anf(truth):
        ctrl = [b + 1 for b in range(4) if mono >> b & 1]
        if not ctrl:
            qc.x(0)
        elif len(ctrl) == 1:
            qc.cx(ctrl[0], 0)
        else:
            qc.mcx(ctrl, 0)
    qc.h([1, 2, 3, 4])
    out = Operator(qc).data @ psi
    assert abs(abs(out[0]) - 1) < 1e-9
    return qc
VP, VA = template_unprep(P), template_unprep(A)

def projection(w):
    """Generic synthesis of the rank-one projection onto w: V with V|w> = |00000>.

    Only the single vector w matters for the classification statistics, so the
    measurement can be synthesised as an inverse state preparation instead of a
    full 32 x 32 unitary followed by the diffusion operator.
    """
    qc = QuantumCircuit(n, name='V_w')
    qc.append(StatePreparation(w / np.linalg.norm(w)).inverse(), range(n))
    out = Operator(qc).data @ w
    assert abs(abs(out[0]) - 1) < 1e-9, "V does not map w to |00000>"
    return qc

def classifier(pcm_circ):
    qc = QuantumCircuit(n); qc.compose(pcm_circ, inplace=True); qc.compose(qD, inplace=True)
    return qc

rows = [
    ("Grover diffusion D alone",                       qD, None),
    ("PCM, NumPy-SVD completion (generic synthesis)",  q_np, PCM_np),
    ("PCM, gap-optimal completion (generic synth.)",   q_opt, PCM_opt),
    ("PCM, structured completion (W^dagger)",          q_st, PCM_st),
    ("Classifier: PCM(NumPy) + D",                     classifier(q_np), PCM_np),
    ("Classifier: PCM(gap-optimal) + D",               classifier(q_opt), PCM_opt),
    ("Classifier: PCM(structured) + D",                classifier(q_st), PCM_st),
    ("Template projection onto |P> (ANF oracle)",      VP, None),
    ("Template projection onto |A> (ANF oracle)",      VA, None),
]
proj_rows = [
    ("w of PCM(structured) + D",      w_from_pcm(PCM_st)),
    ("w of PCM(NumPy) + D",           w_from_pcm(PCM_np)),
    ("w of PCM(gap-optimal) + D",     w_from_pcm(PCM_opt)),
    ("w of PCM(zero-leakage) + D",    w_kl_grover_zl(P, A)),
    ("template |P> (bright)",         P),
    ("template |A> (bright)",         A),
    ("dark port (A-part orthogonal to P)", w_dark(P, A)),
    ("single-copy Helstrom vector",   w_helstrom(P, A)),
]
print(f"NumPy-SVD PCM: rank(PCM - I) = {rank_np}, columns touched = {supp_np}")
print(f"\n{'circuit':50s} {'CNOT':>5s} {'1q':>5s} {'depth':>6s} {'CNOT depth':>10s}   p_P     p_A    gap")
for name, qc, M in rows:
    cx, oneq, depth, cxd = resources(qc)
    extra = ""
    if M is not None:
        pP, pA = gap_of(M)
        extra = f"  {100*pP:6.2f}  {100*pA:6.2f}  {100*abs(pP-pA):5.2f}"
    elif name.startswith("Template"):
        w = P if "|P>" in name else A
        extra = f"  {100*(w@P)**2:6.2f}  {100*(w@A)**2:6.2f}  {100*abs((w@P)**2-(w@A)**2):5.2f}"
    print(f"{name:50s} {cx:5d} {oneq:5d} {depth:6d} {cxd:10d} {extra}")

print("\nThe same measurements synthesised directly as rank-one projections (inverse state preparation):")
print(f"{'projection onto':50s} {'CNOT':>5s} {'1q':>5s} {'depth':>6s} {'CNOT depth':>10s}   p_P     p_A    gap   err k=5  err k=12")
for name, w in proj_rows:
    cx, oneq, depth, cxd = resources(projection(w))
    pP, pA = (w @ P) ** 2, (w @ A) ** 2
    print(f"{name:50s} {cx:5d} {oneq:5d} {depth:6d} {cxd:10d}   {100*pP:6.2f}  {100*pA:6.2f}  {100*abs(pP-pA):5.2f}"
          f"   {100*template_error(pP, pA, 5):6.2f}   {100*template_error(pP, pA, 12):6.3f}")

pP, pA = gap_of(PCM_st)
print(f"\nstructured completion: |00000> labels {'A' if pA > pP else 'P'} "
      f"(p_P = {100*pP:.3f}%, p_A = {100*pA:.3f}%)")

for qc, fn in ((q_st, 'pcm_structured_circuit.png'),):
    fig = qc.draw('mpl', reverse_bits=True, fold=-1)
    fig.savefig(fn, dpi=200, bbox_inches='tight')
    print("wrote", fn)
np.save('pcm_structured.npy', PCM_st)
