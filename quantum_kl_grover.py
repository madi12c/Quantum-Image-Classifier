"""
quantum_kl_grover.py
=============================================================================
Qiskit implementation of the Karhunen-Loeve (KL) Transform (PCM) followed
by the Grover Diffusion Operator (InvAvg) on quantum states |LetP> and |LetA>.

Source: Mathematica notebook  OriginalLetters.nb  (Mathematica 11)

Classical pipeline (from the notebook):
    PCM = KarhunenLoeveDecomposition[ Transpose[{LetP, LetA}] ][[2]]
    V1  = InvAvg . PCM . LetP
    V2  = InvAvg . PCM . LetA

Bug fixes applied vs. the first version:
    Bug 1 — Transpose convention:
        Mathematica's KarhunenLoeveDecomposition returns PCM with KL basis
        vectors as ROWS.  numpy's SVD returns U with basis vectors as COLUMNS.
        Correct mapping:  PCM = U.T  (not U).

    Bug 2 — Sign of the principal eigenvector:
        The KL covariance (rank-1 here) has a unique principal direction
        d/||d||  (d = LetP - LetA) but its sign is arbitrary.  numpy chose
        the negative convention; Mathematica uses the positive convention.
        Fix: if  U[:,0] . d < 0,  flip  U[:,0] -> -U[:,0]  before taking U.T.

    Residual (Bug 3, not fixed here):
        31 null-space eigenvectors (covariance is rank-1 with 2 data points
        in R^32) are mathematically arbitrary.  numpy and Mathematica produce
        different completions, leaving a small residual gap:
            This script:    V1 |00000> ~ 0.37%,   V2 |00000> ~ 26.3%
            Mathematica:    V1 |00000> ~ 0.2%,    V2 |00000> ~ 28%
        To reproduce Mathematica exactly, the 32x32 PCM matrix would need to
        be extracted numerically from the notebook and hard-coded here.

Quantum circuit (per input state):
    |0>^(x5) --[INIT |psi>]--[   PCM   ]--[  InvAvg  ]-- <measure>

Measurement statistics reported as:
    * '00000'   count
    * all-other count
    for 1024 shots of each circuit.

Install dependencies:
    pip install qiskit qiskit-aer numpy
=============================================================================
"""

import numpy as np
from numpy.linalg import svd, norm

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator


# =============================================================================
# SECTION 1 — CLASSICAL INPUT VECTORS
# =============================================================================
# LetP and LetA are 32-dimensional binary pattern vectors scaled by 1/4.
# Each contains exactly 16 ones:
#     ||LetP|| = ||LetA|| = sqrt(16 * (1/4)^2) = 1  -> valid quantum states.
#
# The 32-dimensional Hilbert space is spanned by 5 qubits (2^5 = 32).
# Basis index k = q0*1 + q1*2 + q2*4 + q3*8 + q4*16  (Qiskit little-endian).

print("=" * 65)
print("  QUANTUM KL TRANSFORM + GROVER DIFFUSION  (OriginalLetters.nb)")
print("  (Bugs 1 & 2 fixed)")
print("=" * 65)

_raw_P = [1, 0, 1, 0, 1, 0, 0, 1,  1, 0, 0, 1, 0, 1, 1, 0,
          1, 0, 1, 0, 1, 0, 0, 1,  1, 0, 0, 1, 0, 1, 0, 1]

_raw_A = [0, 1, 1, 0, 1, 0, 0, 1,  1, 0, 0, 1, 0, 1, 1, 0,
          1, 0, 1, 0, 1, 0, 1, 0,  1, 0, 0, 1, 0, 1, 1, 0]

LetP = np.array(_raw_P, dtype=float) / 4.0   # shape (32,)
LetA = np.array(_raw_A, dtype=float) / 4.0   # shape (32,)

assert abs(norm(LetP) - 1.0) < 1e-10, "LetP is not a unit vector!"
assert abs(norm(LetA) - 1.0) < 1e-10, "LetA is not a unit vector!"

print("\n[1] Input Vectors  (binary pattern, each element divided by 4)")
print(f"    LetP (x4) : {_raw_P}")
print(f"    LetA (x4) : {_raw_A}")
print(f"    ||LetP|| = {norm(LetP):.4f}   ||LetA|| = {norm(LetA):.4f}")


# =============================================================================
# SECTION 2 — PCM: KARHUNEN-LOEVE TRANSFORM MATRIX  (32 x 32 unitary)
# =============================================================================
# Mathematica:  PCM = KarhunenLoeveDecomposition[ Transpose[{LetP,LetA}] ][[2]]
#
# KL decomposition on two 32-dim observations:
#     mu      = (LetP + LetA) / 2                     [mean vector]
#     X_c     = [LetP - mu | LetA - mu]               [32x2 centered matrix]
#     X_c     = U S V^T                               [full SVD]
#
# BUG 1 FIX — Transpose convention:
#     Mathematica returns PCM with KL basis vectors as ROWS.
#     numpy SVD returns U with KL basis vectors as COLUMNS.
#     Correct:  PCM = U.T
#
# BUG 2 FIX — Sign of principal eigenvector:
#     The sole non-zero singular value corresponds to direction d/||d||
#     where d = LetP - LetA.  numpy may choose -d/||d||.
#     Fix: enforce  U[:,0] . d >= 0  before transposing.

print(f"\n[2] Computing PCM  (KL Transform — corrected)")

mu  = (LetP + LetA) / 2.0
X_c = np.column_stack([LetP - mu, LetA - mu])   # 32x2, rank 1

U, singular_vals, Vt = svd(X_c, full_matrices=True)

# --- Bug 2 fix: align principal eigenvector sign with Mathematica convention
d = LetP - LetA
if np.dot(U[:, 0], d) < 0:
    U[:, 0] = -U[:, 0]          # flip sign of principal column

# --- Bug 1 fix: rows of PCM are the KL basis vectors  ->  PCM = U.T
PCM = U.T                        # 32x32 real orthogonal

assert np.allclose(PCM @ PCM.T, np.eye(32), atol=1e-10), "PCM must be unitary!"

print(f"    PCM shape              : {PCM.shape}")
print(f"    Unitary (PCM.PCM^T=I)  : {np.allclose(PCM @ PCM.T, np.eye(32))}")
print(f"    Principal singular val : {singular_vals[singular_vals > 1e-12]}")
print(f"    Bug 1 fix applied      : PCM = U.T  (rows are KL basis vectors)")
print(f"    Bug 2 fix applied      : principal eigenvector aligned to +d direction")


# =============================================================================
# SECTION 3 — InvAvg: GROVER DIFFUSION OPERATOR  (32 x 32 unitary)
# =============================================================================
# From the notebook:  InvAvg[i,j] = 1/16 (i!=j),  -15/16 (i==j)
#
# This is the standard Grover diffusion matrix:
#     InvAvg = 2|psi><psi| - I,   |psi> = H^(x5)|00000>
#            = H^(x5) (2|0><0| - I) H^(x5)
#
# Verification:  diagonal = 2/32 - 1 = -15/16  (correct)
#                off-diag = 2/32     =   1/16   (correct)
#
# Circuit realisation (5-layer decomposition):
#   Layer 1:  H on all 5 qubits
#   Layer 2:  X on all 5 qubits          (maps |0...0> -> |1...1>)
#   Layer 3:  MCZ = H . C^4X . H on q4  (phase-flips |1...1> only)
#   Layer 4:  X on all 5 qubits          (maps back)
#   Layer 5:  H on all 5 qubits
#
# The resulting unitary equals InvAvg up to a global phase of -1,
# which is physically unobservable.

print(f"\n[3] Building InvAvg (Grover Diffusion) Circuit")

N_QUBITS = 5
N_DIM    = 2**N_QUBITS   # 32

InvAvg_matrix = (2.0 / N_DIM) * np.ones((N_DIM, N_DIM)) - np.eye(N_DIM)

assert np.allclose(InvAvg_matrix @ InvAvg_matrix.T, np.eye(N_DIM), atol=1e-10)

print(f"    InvAvg diagonal  : {InvAvg_matrix[0, 0]:.6f}  (expected {-15/16:.6f})")
print(f"    InvAvg off-diag  : {InvAvg_matrix[0, 1]:.6f}  (expected  {1/16:.6f})")
print(f"    Unitary          : "
      f"{np.allclose(InvAvg_matrix @ InvAvg_matrix.T, np.eye(N_DIM))}")


def build_invavg_circuit(n: int) -> QuantumCircuit:
    """
    Construct the n-qubit Grover diffusion operator as a QuantumCircuit.

    Implements  G = H^(xn) (2|0><0| - I) H^(xn)  via:

      Layer 1  H on all n qubits
      Layer 2  X on all n qubits           (maps |0...0> to |1...1>)
      Layer 3  MCZ:
                 H on qubit n-1
                 C^(n-1)-X  (controls 0..n-2, target n-1)
                 H on qubit n-1
      Layer 4  X on all n qubits           (undo layer 2)
      Layer 5  H on all n qubits           (undo layer 1)

    Output unitary = InvAvg up to a global phase of -1.
    """
    qc = QuantumCircuit(n, name='InvAvg')

    qc.h(range(n))

    qc.x(range(n))

    controls = list(range(n - 1))
    target   = n - 1
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)

    qc.x(range(n))

    qc.h(range(n))

    return qc


invavg_circ = build_invavg_circuit(N_QUBITS)

# Verify circuit unitary matches InvAvg_matrix up to global phase
circ_unitary = Operator(invavg_circ).data
mask         = np.abs(InvAvg_matrix) > 1e-12
ratio        = circ_unitary[mask] / InvAvg_matrix[mask]
phase_ok     = np.allclose(np.abs(ratio - ratio[0]), 0.0, atol=1e-8)

print(f"    Circuit matches InvAvg (up to global phase) : {phase_ok}")
print(f"    Global phase factor : {ratio[0]:.6f}")
print(f"\n    InvAvg gate circuit diagram (5 qubits):")
print(invavg_circ.draw('text', fold=90))


# =============================================================================
# SECTION 4 — FULL QUANTUM CIRCUIT
# =============================================================================
# Structure (same for both |LetP> and |LetA>):
#
#   q[0] --[INIT]--[PCM]--[InvAvg]-- measure -> c[0]
#   q[1] --[INIT]--[PCM]--[InvAvg]-- measure -> c[1]
#   q[2] --[INIT]--[PCM]--[InvAvg]-- measure -> c[2]
#   q[3] --[INIT]--[PCM]--[InvAvg]-- measure -> c[3]
#   q[4] --[INIT]--[PCM]--[InvAvg]-- measure -> c[4]
#
# INIT     : QuantumCircuit.initialize(state_vec, q)
#            Resets all qubits to |0> then prepares the exact 32-amplitude
#            superposition via Isometry decomposition.
#
# PCM      : UnitaryGate(PCM_matrix, label='PCM')
#            32x32 real orthogonal KL rotation (with Bug 1 & 2 fixes).
#            Transpiler decomposes into 2-qubit gate chains.
#
# InvAvg   : invavg_circ.to_gate(label='InvAvg')
#            5-layer Grover diffusion circuit as a composite gate.
#
# Measure  : All 5 qubits -> 5 classical bits.
#            Bitstring convention: c[4]c[3]c[2]c[1]c[0]  (MSB left in Qiskit).

print(f"\n[4] Constructing Full Quantum Circuits")


def build_full_circuit(state_vec:     np.ndarray,
                       pcm_matrix:    np.ndarray,
                       invavg_subckt: QuantumCircuit,
                       label:         str) -> QuantumCircuit:
    """
    Assemble the 5-qubit circuit:
        INIT -> PCM -> InvAvg -> MEASURE

    Parameters
    ----------
    state_vec     : Normalised 32-element statevector (LetP or LetA).
    pcm_matrix    : 32x32 unitary KL-rotation matrix (Bug 1 & 2 corrected).
    invavg_subckt : Grover diffusion QuantumCircuit.
    label         : Human-readable circuit name.
    """
    n  = int(np.round(np.log2(len(state_vec))))   # 5
    q  = QuantumRegister(n, name='q')
    c  = ClassicalRegister(n, name='c')
    qc = QuantumCircuit(q, c, name=label)

    # Stage 1 : State preparation
    qc.initialize(state_vec.tolist(), q)
    qc.barrier(label='init')

    # Stage 2 : PCM Gate  (Bug 1 & 2 corrected 32x32 unitary)
    pcm_gate = UnitaryGate(pcm_matrix, label='PCM')
    qc.append(pcm_gate, q)
    qc.barrier(label='PCM')

    # Stage 3 : InvAvg Gate  (Grover Diffusion)
    invavg_gate = invavg_subckt.to_gate(label='InvAvg')
    qc.append(invavg_gate, q)
    qc.barrier(label='InvAvg')

    # Stage 4 : Measurement
    qc.measure(q, c)

    return qc


qc_LetP = build_full_circuit(LetP, PCM, invavg_circ, label='InvAvg.PCM.|LetP>')
qc_LetA = build_full_circuit(LetA, PCM, invavg_circ, label='InvAvg.PCM.|LetA>')

print(f"    LetP circuit — qubits: {qc_LetP.num_qubits},  "
      f"classical bits: {qc_LetP.num_clbits}")
print(f"    LetA circuit — qubits: {qc_LetA.num_qubits},  "
      f"classical bits: {qc_LetA.num_clbits}")
print(f"\n    LetP circuit (high-level / un-decomposed):")
print(qc_LetP.draw('text', fold=90))


# =============================================================================
# SECTION 5 — CLASSICAL (IDEAL) VERIFICATION
# =============================================================================
print(f"\n[5] Classical (Ideal) Verification")

V1 = InvAvg_matrix @ PCM @ LetP   # ideal output statevector for |LetP>
V2 = InvAvg_matrix @ PCM @ LetA   # ideal output statevector for |LetA>

assert abs(norm(V1) - 1.0) < 1e-9, "V1 not normalised!"
assert abs(norm(V2) - 1.0) < 1e-9, "V2 not normalised!"

prob_V1 = V1 ** 2    # measurement probabilities (amplitudes are real)
prob_V2 = V2 ** 2

SHOTS = 1024

print(f"    ||V1|| = {norm(V1):.8f}   ||V2|| = {norm(V2):.8f}   (both must = 1.0)")
print(f"\n    Ideal probabilities  (x{SHOTS} = expected counts):")
print(f"    {'Outcome':<12}  {'p(V1)':>9}  {'E[V1]':>8}  {'p(V2)':>9}  {'E[V2]':>8}")
print(f"    {'-'*55}")
print(f"    {'|00000>':<12}  {prob_V1[0]:>9.6f}  {prob_V1[0]*SHOTS:>8.1f}  "
      f"{prob_V2[0]:>9.6f}  {prob_V2[0]*SHOTS:>8.1f}")
print(f"    {'other':<12}  {1-prob_V1[0]:>9.6f}  {(1-prob_V1[0])*SHOTS:>8.1f}  "
      f"{1-prob_V2[0]:>9.6f}  {(1-prob_V2[0])*SHOTS:>8.1f}")

print(f"\n    Top-5 most probable outcomes:")
for v_lbl, prob in [("V1  (LetP input)", prob_V1), ("V2  (LetA input)", prob_V2)]:
    top5 = np.argsort(prob)[::-1][:5]
    print(f"\n      {v_lbl}:")
    for i in top5:
        bar = '|' * int(prob[i] * 50)
        print(f"        |{i:05b}>  p = {prob[i]:.6f}  "
              f"(~{prob[i]*SHOTS:5.1f}/{SHOTS})  {bar}")


# =============================================================================
# SECTION 6 — QUANTUM SIMULATION  (AerSimulator, 1024 shots)
# =============================================================================
print(f"\n[6] Running Quantum Simulation  (AerSimulator, {SHOTS} shots each)")

simulator = AerSimulator()

qc_LetP_t = transpile(qc_LetP, simulator, optimization_level=1)
qc_LetA_t = transpile(qc_LetA, simulator, optimization_level=1)

job_P = simulator.run(qc_LetP_t, shots=SHOTS)
job_A = simulator.run(qc_LetA_t, shots=SHOTS)

counts_P = job_P.result().get_counts()
counts_A = job_A.result().get_counts()

print(f"    Simulation complete.")
print(f"    Distinct outcomes — LetP: {len(counts_P)},  LetA: {len(counts_A)}")


# =============================================================================
# SECTION 7 — MEASUREMENT STATISTICS
# =============================================================================

def report_stats(counts: dict, ideal_probs: np.ndarray,
                 label: str, shots: int = 1024) -> None:
    """
    Print measurement statistics: '00000' vs everything else, plus full table.
    """
    c0    = counts.get('00000', 0)
    c_oth = shots - c0
    pct0  = c0    / shots * 100.0
    pct_o = c_oth / shots * 100.0
    bar0  = '#' * int(pct0  / 2)
    bar_o = '#' * int(pct_o / 2)

    print(f"\n  +-----------------------------------------------------------+")
    print(f"  |  {label:<57}|")
    print(f"  |  {shots} shots{' ' * (57 - len(str(shots)) - 6)}|")
    print(f"  +-----------------------------------------------------------+")
    print(f"  |  Outcome   Observed     Ideal (expected)                  |")
    print(f"  +-----------------------------------------------------------+")
    print(f"  |  '00000'   {c0:5d}        "
          f"~{ideal_probs[0]*shots:5.1f}  ({ideal_probs[0]*100:.2f}%)"
          f"  {bar0:<20}|")
    print(f"  |  other     {c_oth:5d}        "
          f"~{(1-ideal_probs[0])*shots:5.1f}  ({(1-ideal_probs[0])*100:.2f}%)"
          f"  {bar_o:<20}|")
    print(f"  +-----------------------------------------------------------+")
    print(f"  |  '00000' = {pct0:5.1f}%   |   other = {pct_o:5.1f}%"
          + " " * 23 + "|")
    print(f"  +-----------------------------------------------------------+")
    print(f"  |  Full counts (sorted by frequency):                       |")
    for bs, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        idx   = int(bs, 2)
        p_obs = cnt / shots
        p_id  = ideal_probs[idx]
        bar   = '|' * int(p_obs * 25)
        print(f"  |    |{bs}>  obs={cnt:4d} ({p_obs*100:5.1f}%)  "
              f"ideal~{p_id*100:5.2f}%  {bar:<15}|")
    print(f"  +-----------------------------------------------------------+")


print("\n\n" + "=" * 65)
print("  QUANTUM MEASUREMENT RESULTS  (Bugs 1 & 2 fixed)")
print("  Circuit  :  InvAvg . PCM . |LetP/LetA>")
print(f"  Shots    :  {SHOTS} per circuit")
print("=" * 65)

report_stats(counts_P, prob_V1, label="InvAvg . PCM . |LetP>", shots=SHOTS)
report_stats(counts_A, prob_V2, label="InvAvg . PCM . |LetA>", shots=SHOTS)

# ── Compact summary ────────────────────────────────────────────────────────
print("\n" + "-" * 65)
print("  COMPACT SUMMARY  --  '00000'  vs.  everything else")
print("-" * 65)
for counts, lbl, prob in [(counts_P, "InvAvg.PCM.|LetP>", prob_V1),
                          (counts_A, "InvAvg.PCM.|LetA>", prob_V2)]:
    c0    = counts.get('00000', 0)
    c_oth = SHOTS - c0
    print(f"\n  {lbl}  (ideal p(|00000>) = {prob[0]*100:.2f}%):")
    print(f"      '00000'  :  {c0:4d}  ({c0/SHOTS*100:.1f}%)")
    print(f"      other    :  {c_oth:4d}  ({c_oth/SHOTS*100:.1f}%)")
print("\n" + "-" * 65)
print("  Done.\n")
