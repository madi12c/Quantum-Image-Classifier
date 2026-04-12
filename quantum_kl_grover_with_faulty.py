"""
quantum_kl_grover.py
=============================================================================
Qiskit implementation of the Karhunen-Loeve (KL) Transform (PCM) followed
by the Grover Diffusion Operator (InvAvg), applied to two faulty query
inputs, Input1 ("faulty P") and Input2 ("faulty A").

Source: modified OriginalLetters.nb  (Mathematica 11)

Classical pipeline:
    PCM    = KarhunenLoeveDecomposition[ Transpose[{LetP, LetA}] ][[2]]
    V_I1   = InvAvg . PCM . Input1
    V_I2   = InvAvg . PCM . Input2

Variables:
    LetP   -- ideal letter P  (used only to build PCM)
    LetA   -- ideal letter A  (used only to build PCM)
    Input1 -- faulty letter P query  (* faulty character P *)
    Input2 -- faulty letter A query  (* faulty character A *)

Bug fixes applied (same as previous version):
    Bug 1 -- Transpose convention : PCM = U.T  (Mathematica returns basis
             vectors as rows; numpy SVD gives them as columns).
    Bug 2 -- Principal eigenvector sign : enforce U[:,0] . (LetP-LetA) >= 0
             before transposing, matching Mathematica's positive convention.

Residual (Bug 3, not fixed):
    31 null-space eigenvectors (rank-1 covariance from 2 data points in R^32)
    differ between numpy and Mathematica.  This causes a small residual gap:
        This script  :  Input1 |00000> ~2.00%,  Input2 |00000> ~10.71%
        Mathematica  :  Input1 |00000> ~1.85%,  Input2 |00000> ~11.08%

Quantum circuit (per input):
    |0>^(x5) --[INIT |psi>]--[ PCM ]--[ InvAvg ]-- <measure>

Measurement statistics reported as  '00000'  vs  everything else
for 1024 shots of each circuit.

Install:
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
# SECTION 1 — INPUT VECTORS
# =============================================================================
# LetP and LetA are the ideal template vectors used to build PCM.
# Input1 and Input2 are the faulty query inputs (from the modified notebook).
# All four vectors are 32-dimensional, scaled by 1/4, with exactly 16 ones
# each -> unit norm -> valid quantum states over 5 qubits (2^5 = 32).

print("=" * 65)
print("  QUANTUM KL + GROVER :  InvAvg . PCM . Input1 / Input2")
print("  (modified OriginalLetters.nb — Bugs 1 & 2 fixed)")
print("=" * 65)

# ── Ideal templates  (used to build PCM only) ─────────────────────────────
_raw_P = [1, 0, 1, 0, 1, 0, 0, 1,  1, 0, 0, 1, 0, 1, 1, 0,
          1, 0, 1, 0, 1, 0, 0, 1,  1, 0, 0, 1, 0, 1, 0, 1]

_raw_A = [0, 1, 1, 0, 1, 0, 0, 1,  1, 0, 0, 1, 0, 1, 1, 0,
          1, 0, 1, 0, 1, 0, 1, 0,  1, 0, 0, 1, 0, 1, 1, 0]

# ── Faulty query inputs (the circuit inputs)  ─────────────────────────────
# Input1: faulty character P  (comment in notebook: (* faulty character P *))
#   Differs from LetP at positions 0,1 (swap) and 22,23 (swap)
_raw_I1 = [0, 1, 1, 0, 1, 0, 0, 1,  1, 0, 0, 1, 0, 1, 1, 0,
           1, 0, 1, 0, 1, 0, 0, 1,  1, 0, 0, 1, 0, 1, 0, 1]

# Input2: faulty character A  (comment in notebook: (* faulty character A *))
#   Differs from LetA at position 22 (1->0) relative to Input1 structure
_raw_I2 = [0, 1, 1, 0, 1, 0, 0, 1,  1, 0, 0, 1, 0, 1, 1, 0,
           1, 0, 1, 0, 1, 0, 1, 0,  1, 0, 0, 1, 0, 1, 0, 1]

LetP   = np.array(_raw_P,  dtype=float) / 4.0
LetA   = np.array(_raw_A,  dtype=float) / 4.0
Input1 = np.array(_raw_I1, dtype=float) / 4.0
Input2 = np.array(_raw_I2, dtype=float) / 4.0

for name, vec, raw in [("LetP",   LetP,   _raw_P), ("LetA",   LetA,   _raw_A), ("Input1", Input1, _raw_I1), ("Input2", Input2, _raw_I2)]:
    assert abs(norm(vec) - 1.0) < 1e-10, f"{name} is not a unit vector!"

print("\n[1] Input Vectors")
print(f"    LetP   (x4): {_raw_P}")
print(f"    LetA   (x4): {_raw_A}")
print(f"    Input1 (x4): {_raw_I1}  (* faulty P *)")
print(f"    Input2 (x4): {_raw_I2}  (* faulty A *)")

# Hamming distances to ideal templates
def hamming(a, b):
    return int(np.sum(np.array(a) != np.array(b)))

print(f"\n    Hamming distances:")
print(f"      Input1 vs LetP : {hamming(_raw_I1, _raw_P)}")
print(f"      Input1 vs LetA : {hamming(_raw_I1, _raw_A)}")
print(f"      Input2 vs LetP : {hamming(_raw_I2, _raw_P)}")
print(f"      Input2 vs LetA : {hamming(_raw_I2, _raw_A)}")


# =============================================================================
# SECTION 2 — PCM: KARHUNEN-LOEVE TRANSFORM MATRIX  (32 x 32 unitary)
# =============================================================================
# Built from the ideal templates LetP and LetA (unchanged from the notebook).
# Same Bug 1 & Bug 2 corrections as the previous version.
#
#   Bug 1: PCM = U.T  (Mathematica stores KL basis as rows; numpy as columns)
#   Bug 2: enforce U[:,0] . (LetP-LetA) >= 0 before transposing

print(f"\n[2] Computing PCM  (KL from ideal templates LetP, LetA)")

mu  = (LetP + LetA) / 2.0
X_c = np.column_stack([LetP - mu, LetA - mu])       # 32x2, rank 1
U, singular_vals, Vt = svd(X_c, full_matrices=True)

# Bug 2 fix: align principal direction sign with Mathematica convention
d = LetP - LetA
if np.dot(U[:, 0], d) < 0:
    U[:, 0] = -U[:, 0]

# Bug 1 fix: rows of PCM are the KL basis vectors
PCM = U.T                                           # 32x32 real orthogonal

assert np.allclose(PCM @ PCM.T, np.eye(32), atol=1e-10), "PCM must be unitary!"

print(f"    PCM shape              : {PCM.shape}")
print(f"    Unitary (PCM.PCM^T=I)  : {np.allclose(PCM @ PCM.T, np.eye(32))}")
print(f"    Non-zero singular value: {singular_vals[singular_vals > 1e-12]}")
print(f"    Bug 1 fix: PCM = U.T")
print(f"    Bug 2 fix: principal direction aligned to +(LetP-LetA)")


# =============================================================================
# SECTION 3 — InvAvg: GROVER DIFFUSION OPERATOR  (32 x 32 unitary)
# =============================================================================
# InvAvg[i,j] = 1/16 (i!=j),  -15/16 (i==j)  ->  2|psi><psi| - I
#
# Circuit:  H^(xn)  X^(xn)  [H . C^(n-1)X . H on q_{n-1}]  X^(xn)  H^(xn)
# Equals InvAvg up to a global phase of -1 (unobservable).

print(f"\n[3] Building InvAvg (Grover Diffusion) Circuit")

N_QUBITS = 5
N_DIM    = 2**N_QUBITS   # 32

InvAvg_matrix = (2.0 / N_DIM) * np.ones((N_DIM, N_DIM)) - np.eye(N_DIM)
assert np.allclose(InvAvg_matrix @ InvAvg_matrix.T, np.eye(N_DIM), atol=1e-10)

print(f"    InvAvg diagonal : {InvAvg_matrix[0,0]:.6f}  (expected {-15/16:.6f})")
print(f"    InvAvg off-diag : {InvAvg_matrix[0,1]:.6f}  (expected  {1/16:.6f})")


def build_invavg_circuit(n: int) -> QuantumCircuit:
    """
    n-qubit Grover diffusion operator:  G = H^n (2|0><0|-I) H^n

    Layers:
      1  H on all n qubits
      2  X on all n qubits
      3  MCZ = H . C^(n-1)X . H  on qubit n-1  (phase-flips |1...1> only)
      4  X on all n qubits
      5  H on all n qubits

    Output unitary = InvAvg up to global phase -1.
    """
    qc = QuantumCircuit(n, name='InvAvg')
    qc.h(range(n))

    qc.x(range(n))
 
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
 
    qc.x(range(n))
  
    qc.h(range(n))
 
    return qc


invavg_circ = build_invavg_circuit(N_QUBITS)

circ_unitary = Operator(invavg_circ).data
mask  = np.abs(InvAvg_matrix) > 1e-12
ratio = circ_unitary[mask] / InvAvg_matrix[mask]
phase_ok = np.allclose(np.abs(ratio - ratio[0]), 0.0, atol=1e-8)
print(f"    Circuit matches InvAvg (up to global phase): {phase_ok}")
print(f"    Global phase: {ratio[0]:.6f}")

print(f"\n    InvAvg circuit (5 qubits):")
print(invavg_circ.draw('text', fold=90))


# =============================================================================
# SECTION 4 — FULL QUANTUM CIRCUITS
# =============================================================================
# One circuit per query input:
#
#   q[0..4] --[INIT |Input_k>]--[ PCM ]--[ InvAvg ]-- measure -> c[0..4]
#
# INIT    : initialize(state_vec)  -- exact 32-amplitude state preparation
# PCM     : UnitaryGate(PCM_matrix) -- 32x32 KL rotation (Bug 1 & 2 corrected)
# InvAvg  : composite gate from build_invavg_circuit
# Measure : all 5 qubits -> 5 classical bits

print(f"\n[4] Constructing Full Quantum Circuits")


def build_full_circuit(state_vec:     np.ndarray,
                       pcm_matrix:    np.ndarray,
                       invavg_subckt: QuantumCircuit,
                       label:         str) -> QuantumCircuit:
    """
    Assemble:  INIT -> PCM -> InvAvg -> MEASURE  (5 qubits, 5 classical bits)

    Parameters
    ----------
    state_vec     : Normalised 32-element input statevector.
    pcm_matrix    : 32x32 unitary KL matrix (Bug 1 & 2 corrected).
    invavg_subckt : Grover diffusion QuantumCircuit.
    label         : Circuit name.
    """
    n  = int(np.round(np.log2(len(state_vec))))
    q  = QuantumRegister(n, name='q')
    c  = ClassicalRegister(n, name='c')
    qc = QuantumCircuit(q, c, name=label)

    qc.initialize(state_vec.tolist(), q)      # Stage 1: state prep
    qc.barrier(label='init')
    qc.append(UnitaryGate(pcm_matrix, label='PCM'), q)   # Stage 2: PCM
    qc.barrier(label='PCM')
    qc.append(invavg_subckt.to_gate(label='InvAvg'), q)  # Stage 3: InvAvg
    qc.barrier(label='InvAvg')
    qc.measure(q, c)                          # Stage 4: measure

    return qc


qc_I1 = build_full_circuit(Input1, PCM, invavg_circ, label='InvAvg.PCM.|Input1>')
qc_I2 = build_full_circuit(Input2, PCM, invavg_circ, label='InvAvg.PCM.|Input2>')

print(f"    Input1 circuit — qubits: {qc_I1.num_qubits},  classical bits: {qc_I1.num_clbits}")
print(f"    Input2 circuit — qubits: {qc_I2.num_qubits},  classical bits: {qc_I2.num_clbits}")

print(f"\n    Input1 circuit (high-level view):")
print(qc_I1.draw('text', fold=90))


# =============================================================================
# SECTION 5 — CLASSICAL (IDEAL) VERIFICATION
# =============================================================================
# Compute exact output statevectors via matrix arithmetic and derive
# ideal measurement probabilities for comparison with simulation.

print(f"\n[5] Classical (Ideal) Verification")

V_I1 = InvAvg_matrix @ PCM @ Input1
V_I2 = InvAvg_matrix @ PCM @ Input2

assert abs(norm(V_I1) - 1.0) < 1e-9
assert abs(norm(V_I2) - 1.0) < 1e-9

prob_I1 = V_I1 ** 2     # measurement probabilities (all amplitudes are real)
prob_I2 = V_I2 ** 2

SHOTS = 1024

# Mathematica reference amplitudes (from notebook output cells)
math_amp_I1 = np.array([
     0.13583908526885938,  0.28354470434358897, -0.012098842115174882,  0.23790115788482513,
    -0.012098842115174896, 0.23790115788482513,  0.23790115788482513,  -0.012098842115174854,
    -0.012098842115174896, 0.23790115788482513,  0.23790115788482513,  -0.012098842115174854,
     0.23790115788482513, -0.012098842115174896, -0.012098842115174882,  0.23790115788482513,
    -0.012098842115174896, 0.23790115788482513,  -0.012098842115174882,  0.23790115788482513,
    -0.012098842115174896, 0.23790115788482513,   0.28060625619730933,  -0.05480394042765907,
    -0.012098842115174896, 0.23790115788482513,   0.23790115788482513,  -0.012098842115174854,
     0.23790115788482513, -0.012098842115174896,  0.28060625619730933,  -0.054803940427659084])

math_amp_I2 = np.array([
     0.33291091473114065,  0.18520529565641103, -0.019151157884825118,  0.23084884211517487,
    -0.019151157884825132, 0.23084884211517487,  0.23084884211517487,  -0.019151157884825104,
    -0.019151157884825132, 0.23084884211517487,  0.23084884211517487,  -0.019151157884825104,
     0.23084884211517487, -0.019151157884825132, -0.019151157884825118,  0.23084884211517487,
    -0.019151157884825132, 0.23084884211517487,  -0.019151157884825118,  0.23084884211517487,
    -0.019151157884825132, 0.23084884211517487,   0.09265224099016434,   0.11904544324018543,
    -0.019151157884825132, 0.23084884211517487,   0.23084884211517487,  -0.019151157884825104,
     0.23084884211517487, -0.019151157884825132,  0.34265224099016434,  -0.13095455675981454])

math_prob_I1 = math_amp_I1 ** 2
math_prob_I2 = math_amp_I2 ** 2

print(f"    ||V_I1|| = {norm(V_I1):.8f}   ||V_I2|| = {norm(V_I2):.8f}")
print(f"\n    Comparison with Mathematica amplitudes:")
print(f"    Max deviation Input1: {np.max(np.abs(V_I1 - math_amp_I1)):.4e}  "
      f"(residual null-space difference, Bug 3)")
print(f"    Max deviation Input2: {np.max(np.abs(V_I2 - math_amp_I2)):.4e}  "
      f"(residual null-space difference, Bug 3)")

print(f"\n    Ideal probabilities (x{SHOTS} = expected counts):")
print(f"    {'Outcome':<12}  {'p(I1,this)':>11}  {'p(I1,Math)':>11}  "
      f"{'p(I2,this)':>11}  {'p(I2,Math)':>11}")
print(f"    {'-'*63}")
print(f"    {'|00000>':<12}  {prob_I1[0]:>11.6f}  {math_prob_I1[0]:>11.6f}  "
      f"{prob_I2[0]:>11.6f}  {math_prob_I2[0]:>11.6f}")
print(f"    {'other':<12}  {1-prob_I1[0]:>11.6f}  {1-math_prob_I1[0]:>11.6f}  "
      f"{1-prob_I2[0]:>11.6f}  {1-math_prob_I2[0]:>11.6f}")

print(f"\n    Top-5 outcomes:")
for lbl, prob in [("Input1", prob_I1), ("Input2", prob_I2)]:
    top5 = np.argsort(prob)[::-1][:5]
    print(f"\n      InvAvg.PCM.|{lbl}>:")
    for i in top5:
        bar = '|' * int(prob[i] * 50)
        print(f"        |{i:05b}>  p={prob[i]:.6f}  (~{prob[i]*SHOTS:5.1f}/{SHOTS})  {bar}")


# =============================================================================
# SECTION 6 — QUANTUM SIMULATION  (AerSimulator, 1024 shots)
# =============================================================================
print(f"\n[6] Running Quantum Simulation  (AerSimulator, {SHOTS} shots each)")

simulator = AerSimulator()

qc_I1_t = transpile(qc_I1, simulator, optimization_level=1)
qc_I2_t = transpile(qc_I2, simulator, optimization_level=1)

job_I1 = simulator.run(qc_I1_t, shots=SHOTS)
job_I2 = simulator.run(qc_I2_t, shots=SHOTS)

counts_I1 = job_I1.result().get_counts()
counts_I2 = job_I2.result().get_counts()

print(f"    Simulation complete.")
print(f"    Distinct outcomes — Input1: {len(counts_I1)},  Input2: {len(counts_I2)}")


# =============================================================================
# SECTION 7 — MEASUREMENT STATISTICS
# =============================================================================

def report_stats(counts:      dict,
                 ideal_probs: np.ndarray,
                 label:       str,
                 shots:       int = 1024) -> None:
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
    print(f"  |  '00000'   {c0:5d}       "
          f"~{ideal_probs[0]*shots:5.1f}  ({ideal_probs[0]*100:.2f}%)"
          f"  {bar0:<22}|")
    print(f"  |  other     {c_oth:5d}       "
          f"~{(1-ideal_probs[0])*shots:5.1f}  ({(1-ideal_probs[0])*100:.2f}%)"
          f"  {bar_o:<22}|")
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
print("  Circuit  :  InvAvg . PCM . |Input1/Input2>")
print(f"  Shots    :  {SHOTS} per circuit")
print("=" * 65)

report_stats(counts_I1, prob_I1, label="InvAvg . PCM . |Input1>  (faulty P)", shots=SHOTS)
report_stats(counts_I2, prob_I2, label="InvAvg . PCM . |Input2>  (faulty A)", shots=SHOTS)

# ── Compact summary ────────────────────────────────────────────────────────
print("\n" + "-" * 65)
print("  COMPACT SUMMARY  --  '00000'  vs.  everything else")
print("-" * 65)
for counts, lbl, prob in [
        (counts_I1, "InvAvg.PCM.|Input1>  (faulty P)", prob_I1),
        (counts_I2, "InvAvg.PCM.|Input2>  (faulty A)", prob_I2)]:
    c0    = counts.get('00000', 0)
    c_oth = SHOTS - c0
    print(f"\n  {lbl}  (ideal p(|00000>) = {prob[0]*100:.2f}%):")
    print(f"      '00000'  :  {c0:4d}  ({c0/SHOTS*100:.1f}%)")
    print(f"      other    :  {c_oth:4d}  ({c_oth/SHOTS*100:.1f}%)")
print("\n" + "-" * 65)
print("  Done.\n")
