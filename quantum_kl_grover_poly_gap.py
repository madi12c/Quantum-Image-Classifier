"""
quantum_kl_grover_poly_gap.py
=============================================================================
Qiskit implementation with a POLYNOMIAL-GATE PCM that achieves a meaningful
20.23 percentage-point gap between P(|00000>, LetP) and P(|00000>, LetA).

BACKGROUND
----------
The amplitude at |00000> after InvAvg @ PCM @ v decomposes as:

    amp(v) = (2/N - 1)*(e1 . v)  +  (2/N)*(v_free . v)

where  v_free = col_sums(PCM) - e1  with  ||v_free|| = sqrt(N-1) = sqrt(31).
The gap  |P_LetP - P_LetA|  is maximised when  v_free  has a large component
along  e_s = (LetP + LetA) / ||LetP + LetA||  (the mean/sum direction).

WHY THE ORIGINAL 7-GATE CIRCUIT GAVE ~0 GAP
--------------------------------------------
The original circuit used the gate sequence:
    X(q0), H(q0), RY(q4), CX(q4,q1), CX(q4,q2), CZ(q4,q1), CH(q4,q3)

Tracking the uniform state |+> through this circuit reveals that:
  - |+> starts with e_s-projection = 0.743 (55% of the maximum)
  - The X(q0) gate immediately drops it to 0.023
  - The CZ(q4,q1) gate then destroys the remaining projection (0.387 → 0.021)
  - Final col_sums . e_s ≈ -0.034  (nearly zero → tiny gap)

THE FIX: REPLACE CZ(q4,q1) WITH Z(q4)
---------------------------------------
Mathematical insight: after CX(q4,q1) and CX(q4,q2), the state |00000>
has its q4=1 amplitudes at positions {22, 23} which require a sign flip for
the correct e1 structure. This sign flip can be achieved with a SINGLE-QUBIT
Z gate on q4 (flips all q4=1 states) instead of the two-qubit CZ(q4,q1).

Simultaneously, replacing X(q0) + H(q0) with H(q0) + Z(q0) avoids the
drastic drop in e_s projection caused by the X gate:
  - X(q0) on |+> → drops e_s from 0.743 to 0.023
  - H(q0) on |+> keeps the projection at 0.624 (much better)

NEW 7-GATE CIRCUIT W (W = PCM^T):
    H(q0), RY(q4,θ), CX(q4,q1), CX(q4,q2), Z(q4), CH(q4,q3), Z(q0)

This still prepares W|00000> = e1 exactly, but now:
    col_sums . e_s = -2.962  (vs -0.034 before)
    gap = 20.23 pp           (vs  0.23 pp before)

This is 53% of the theoretical maximum (38.04 pp).

CIRCUIT PROPERTIES (W and PCM = W^T both)
------------------------------------------
  Gates:   4 single-qubit (H, RY, Z, Z) + 3 two-qubit (CX, CX, CH) = 7 gates
  Depth:   4 layers
  Scaling: O(n)  —  polynomial, same count as the original 7-gate circuit

MEASUREMENT STATISTICS (ideal probabilities)
--------------------------------------------
  InvAvg . PCM_poly . |LetP>   : P(|00000>) = 21.463%
  InvAvg . PCM_poly . |LetA>   : P(|00000>) =  1.228%
  |P_LetP - P_LetA|            = 20.23 pp   (53% of global max 38.04 pp)
  InvAvg . PCM_poly . |Input1> : P(|00000>) =  6.445%  (faulty P)
  InvAvg . PCM_poly . |Input2> : P(|00000>) =  0.871%  (faulty A)

Note on sign convention: P_LetP > P_LetA (reversed relative to the full
optimal PCM_opt, which gave P_LetA > P_LetP). This is because col_sums.e_s
is negative here (= -2.962 vs +sqrt(31) ≈ +5.57 for optimal). The absolute
gap is what matters for discrimination.

Install:
    pip install qiskit qiskit-aer numpy
=============================================================================
"""

import numpy as np
from numpy.linalg import norm

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator


# =============================================================================
# SECTION 1 — INPUT VECTORS
# =============================================================================
print("=" * 70)
print("  QUANTUM KL + GROVER  —  Polynomial PCM  (7 gates, 20.23 pp gap)")
print("=" * 70)

_raw_P  = [1, 0, 1, 0, 1, 0, 0, 1,  1, 0, 0, 1, 0, 1, 1, 0,
           1, 0, 1, 0, 1, 0, 0, 1,  1, 0, 0, 1, 0, 1, 0, 1]
_raw_A  = [0, 1, 1, 0, 1, 0, 0, 1,  1, 0, 0, 1, 0, 1, 1, 0,
           1, 0, 1, 0, 1, 0, 1, 0,  1, 0, 0, 1, 0, 1, 1, 0]
_raw_I1 = [0, 1, 1, 0, 1, 0, 0, 1,  1, 0, 0, 1, 0, 1, 1, 0,
           1, 0, 1, 0, 1, 0, 0, 1,  1, 0, 0, 1, 0, 1, 0, 1]  # faulty P
_raw_I2 = [0, 1, 1, 0, 1, 0, 0, 1,  1, 0, 0, 1, 0, 1, 1, 0,
           1, 0, 1, 0, 1, 0, 1, 0,  1, 0, 0, 1, 0, 1, 0, 1]  # faulty A

N = 32
LetP   = np.array(_raw_P,  dtype=float) / 4
LetA   = np.array(_raw_A,  dtype=float) / 4
Input1 = np.array(_raw_I1, dtype=float) / 4
Input2 = np.array(_raw_I2, dtype=float) / 4

for name, v in [("LetP",LetP),("LetA",LetA),("Input1",Input1),("Input2",Input2)]:
    assert abs(norm(v) - 1.0) < 1e-10, f"{name} not unit"

d  = LetP - LetA;  e1  = d / norm(d)
s  = LetP + LetA;  e_s = s / norm(s)
InvAvg = (2/N) * np.ones((N, N)) - np.eye(N)

print("\n[1] Input vectors loaded.")
print(f"    e1  (LetP-LetA normalised): 6 non-zero entries, all ±1/sqrt(6)")
print(f"    e_s (LetP+LetA normalised): 19 non-zero entries")
print(f"    e1 . e_s = {np.dot(e1,e_s):.2e}  (orthogonal)")


# =============================================================================
# SECTION 2 — POLYNOMIAL PCM CIRCUIT
# =============================================================================
# The PCM = W^T where W is the state-preparation circuit.
# W is built so that:
#   (A)  W|00000> = e1               (KL principal direction — required)
#   (B)  col_sums(PCM) . e_s = -2.96 (large magnitude — gives 20pp gap)
#
# 7-gate circuit W:
#
#  q[0]: ─[H]──────────────────────────────────────────────[Z]─
#  q[1]: ─────────────────────[CX]──────────────────────────────
#  q[2]: ──────────────────────────────[CX]─────────────────────
#  q[3]: ──────────────────────────────────────────[CH]─────────
#  q[4]: ─[RY(θ)]──────[●]─────────────[●]──────[Z]─[●]───────
#
#  Layer 1 (depth 1): H(q0) ‖ RY(θ_ry, q4)
#  Layer 2 (depth 2): CX(q4→q1) ‖ CX(q4→q2)
#  Layer 3 (depth 3): Z(q4)
#  Layer 4 (depth 4): CH(q4→q3) ‖ Z(q0)
#
# θ_ry = 2·arccos(1/√3) ≈ 109.47°
#
# PCM = W^T  (reverse gate order, each gate is its own inverse for H,X,Z,CX,CZ,CH):
#  Layer 1 (depth 1): Z(q0) ‖ CH(q4→q3)
#  Layer 2 (depth 2): Z(q4)
#  Layer 3 (depth 3): CX(q4→q2) ‖ CX(q4→q1)
#  Layer 4 (depth 4): RY(-θ_ry, q4) ‖ H(q0)
#
# Why this works:
#   CZ(q4,q1) in the original circuit dropped col_sums.e_s from 0.387 to 0.021.
#   Z(q4) here keeps col_sums.e_s at -0.524 (large magnitude).
#   The negative sign means P_LetP > P_LetA (opposite ordering vs PCM_opt,
#   but the gap |P_LetP - P_LetA| = 20.23 pp is meaningful).

THETA_RY = 2.0 * np.arccos(1.0 / np.sqrt(3))   # ≈ 109.47°

N_QUBITS = 5

print(f"\n[2] Polynomial PCM Circuit  (7 gates, depth 4)")
print(f"    θ_ry = 2·arccos(1/√3) = {np.degrees(THETA_RY):.4f}°")


def build_W_circuit() -> QuantumCircuit:
    """
    7-gate circuit W (state preparation W|00000> = e1):

      Layer 1: H(q0)  ‖  RY(θ_ry, q4)
      Layer 2: CX(q4→q1)  ‖  CX(q4→q2)
      Layer 3: Z(q4)
      Layer 4: CH(q4→q3)  ‖  Z(q0)

    Properties:
      - W|00000> = e1  (KL constraint satisfied)
      - col_sums(W^T) . e_s = -2.962  (large: gives 20.23pp gap)
      - W is real orthogonal (unitary)
      - 4 single-qubit + 3 two-qubit = 7 gates, depth 4, O(n) scaling
    """
    qc = QuantumCircuit(N_QUBITS, name='W')

    # ── Layer 1 ───────────────────────────────────────────────────────────
    qc.h(0)                    # H on q[0]
    qc.ry(THETA_RY, 4)         # RY(θ_ry) on q[4]  — creates (1/√3)|0>+√(2/3)|1> on q4
 

    # ── Layer 2 ───────────────────────────────────────────────────────────
    qc.cx(4, 1)                # CX(q[4], q[1])  — sets q1=1 when q4=1
    qc.cx(4, 2)                # CX(q[4], q[2])  — sets q2=1 when q4=1


    # ── Layer 3 ───────────────────────────────────────────────────────────
    qc.z(4)                    # Z(q[4])  — flips sign of ALL q4=1 amplitudes
    #   This replaces CZ(q4,q1) from the original circuit.
    #   After CX(q4,q1), both q4=1,q1=1 and q4=1,q1=0 states exist in 1_vec.
    #   Z(q4) flips ALL q4=1 states, which for the |00000> input correctly
    #   produces the negative sign on the |1011> and later |1111> groups.
    #   Crucially, Z(q4) preserves col_sums.e_s much better than CZ(q4,q1).
 

    # ── Layer 4 ───────────────────────────────────────────────────────────
    qc.ch(4, 3)                # CH(q[4], q[3])  — H on q3 when q4=1
    #   Splits |q4=1,q3=0> into (|q3=0>+|q3=1>)/√2, creating |1011> and |1111>
    qc.z(0)                    # Z(q[0])  — creates the (|0>-|1>) sign pattern on q0
    #   Acting on (|0>+|1>)/√2 (from H in layer 1), Z|+> = |->  = (|0>-|1>)/√2


    return qc


def build_PCM_circuit() -> QuantumCircuit:
    """
    PCM_poly = W^T  (adjoint/transpose of W).
    Reverse gate order; all gates here are self-adjoint (H†=H, Z†=Z, CX†=CX, CH†=CH).
    RY is replaced by RY(-θ_ry) since RY(θ)†=RY(-θ).

    Layer 1: Z(q0)  ‖  CH(q4→q3)
    Layer 2: Z(q4)
    Layer 3: CX(q4→q2)  ‖  CX(q4→q1)
    Layer 4: RY(-θ_ry, q4)  ‖  H(q0)
    """
    qc = QuantumCircuit(N_QUBITS, name='PCM_poly')

    # Reverse of W, layer 4 first:
    qc.z(0)                     # Z†(q0) = Z(q0)
    qc.ch(4, 3)                 # CH†(q4,q3) = CH(q4,q3)
 

    qc.z(4)                     # Z†(q4) = Z(q4)
 

    qc.cx(4, 2)                 # CX†(q4,q2) = CX(q4,q2)
    qc.cx(4, 1)                 # CX†(q4,q1) = CX(q4,q1)


    qc.ry(-THETA_RY, 4)         # RY†(θ_ry, q4) = RY(-θ_ry, q4)
    qc.h(0)                     # H†(q0) = H(q0)


    return qc


W_circ   = build_W_circuit()
PCM_circ = build_PCM_circuit()

print(f"\n    W circuit (state preparation  |00000> → e1):")
print(W_circ.draw('text', fold=90))
print(f"\n    PCM_poly = W† circuit:")
print(PCM_circ.draw('text', fold=90))

# Numerical verification via matrix simulation
def simulate_circuit_as_matrix(qc: QuantumCircuit) -> np.ndarray:
    """Build the 32x32 unitary matrix by acting on each basis state."""
    M = np.zeros((N, N))
    def _apply(state: np.ndarray) -> np.ndarray:
        """Apply the circuit gates to a statevector."""
        v = state.copy()
        sq2 = np.sqrt(2)
        for instr in qc.data:
            name = instr.operation.name
            qargs = [q._index for q in instr.qubits]
            if name == 'h':
                q = qargs[0]
                ns = np.zeros(N)
                for k in range(N):
                    if (k >> q) & 1 == 0:
                        ns[k] += v[k] / sq2
                        ns[k | (1 << q)] += v[k] / sq2
                    else:
                        ns[k ^ (1 << q)] += v[k] / sq2
                        ns[k] -= v[k] / sq2
                v = ns
            elif name == 'z':
                q = qargs[0]
                for k in range(N):
                    if (k >> q) & 1:
                        v[k] = -v[k]
            elif name == 'x':
                q = qargs[0]
                ns = np.zeros(N)
                for k in range(N):
                    ns[k ^ (1 << q)] = v[k]
                v = ns
            elif name == 'ry':
                q = qargs[0]
                theta = float(instr.operation.params[0])
                c, sv = np.cos(theta/2), np.sin(theta/2)
                ns = np.zeros(N)
                for k in range(N):
                    if (k >> q) & 1 == 0:
                        ns[k] += c * v[k]
                        ns[k | (1 << q)] += sv * v[k]
                    else:
                        ns[k ^ (1 << q)] -= sv * v[k]
                        ns[k] += c * v[k]
                v = ns
            elif name == 'cx':
                ctrl, tgt = qargs
                ns = np.zeros(N)
                for k in range(N):
                    if (k >> ctrl) & 1:
                        ns[k ^ (1 << tgt)] = v[k]
                    else:
                        ns[k] = v[k]
                v = ns
            elif name == 'cz':
                ctrl, tgt = qargs
                for k in range(N):
                    if ((k >> ctrl) & 1) and ((k >> tgt) & 1):
                        v[k] = -v[k]
            elif name == 'ch':
                ctrl, tgt = qargs
                ns = np.zeros(N)
                for k in range(N):
                    if not (k >> ctrl) & 1:
                        ns[k] = v[k]
                    elif not (k >> tgt) & 1:
                        ns[k] += v[k] / sq2
                        ns[k | (1 << tgt)] += v[k] / sq2
                    else:
                        ns[k ^ (1 << tgt)] += v[k] / sq2
                        ns[k] -= v[k] / sq2
                v = ns
            elif name == 'barrier':
                pass
        return v

    for j in range(N):
        b = np.zeros(N); b[j] = 1.0
        M[:, j] = _apply(b)
    return M


W_matrix   = simulate_circuit_as_matrix(W_circ)
PCM_matrix = simulate_circuit_as_matrix(PCM_circ)

assert np.allclose(W_matrix @ W_matrix.T, np.eye(N), atol=1e-9),  "W not unitary!"
assert np.allclose(PCM_matrix @ PCM_matrix.T, np.eye(N), atol=1e-9), "PCM not unitary!"
assert np.allclose(W_matrix[:, 0], e1, atol=1e-9), "W[:,0] ≠ e1!"
assert np.allclose(PCM_matrix[0, :], e1, atol=1e-9), "PCM[0,:] ≠ e1!"
assert np.allclose(PCM_matrix, W_matrix.T, atol=1e-9), "PCM ≠ W^T!"

col_sums = PCM_matrix.T @ np.ones(N)
print(f"\n    Verification:")
print(f"      W unitary:            {np.allclose(W_matrix @ W_matrix.T, np.eye(N))}")
print(f"      PCM = W^T:            {np.allclose(PCM_matrix, W_matrix.T)}")
print(f"      PCM[0,:] = e1:        {np.allclose(PCM_matrix[0,:], e1)}")
print(f"      col_sums . e1 = 1:    {np.dot(col_sums, e1):.6f}")
print(f"      col_sums . e_s = ???: {np.dot(col_sums, e_s):.6f}  (target: large |value|)")

print(f"\n    Gate count summary:")
print(f"      Single-qubit:  H(q0), RY(q4), Z(q4), Z(q0)  →  4 gates")
print(f"      Two-qubit:     CX(q4,q1), CX(q4,q2), CH(q4,q3)  →  3 gates")
print(f"      Total:         7 gates,  depth 4,  O(n) scaling")
print(f"\n    Compare to:")
print(f"      Original 7-gate (CZ version):  gap = 0.23pp")
print(f"      This circuit (Z version):      gap = 20.23pp")
print(f"      Optimal PCM_opt (full matrix): gap = 38.04pp  [O(4^n) circuit]")


# =============================================================================
# SECTION 3 — CLASSICAL VERIFICATION
# =============================================================================
print(f"\n[3] Classical Ideal Probabilities")

SHOTS = 1024
ideal_probs = {}
for lbl, vec in [('LetP',LetP),('LetA',LetA),('Input1',Input1),('Input2',Input2)]:
    out = InvAvg @ PCM_matrix @ vec
    assert abs(norm(out) - 1.0) < 1e-8
    ideal_probs[lbl] = out ** 2

print(f"\n    {'Circuit':<35}  {'P(|00000>)':>11}  {'Expected /1024':>15}")
print(f"    {'-'*65}")
for lbl in ('LetP','LetA','Input1','Input2'):
    p = ideal_probs[lbl][0]
    print(f"    InvAvg.PCM_poly.|{lbl:<7}  {p*100:>10.4f}%  {p*SHOTS:>14.1f}")

gap = abs(ideal_probs['LetP'][0] - ideal_probs['LetA'][0])
print(f"\n    |P_LetP - P_LetA| = {gap*100:.4f} pp  (vs optimal 38.04 pp = {gap/0.3804*100:.1f}% of max)")
print(f"\n    Top-5 outcomes per circuit:")
for lbl in ('LetP','LetA','Input1','Input2'):
    top5 = np.argsort(ideal_probs[lbl])[::-1][:5]
    p = ideal_probs[lbl]
    print(f"\n      InvAvg.PCM_poly.|{lbl}>:")
    for i in top5:
        print(f"        |{i:05b}>  {p[i]*100:.4f}%  (~{p[i]*SHOTS:.1f})")


# =============================================================================
# SECTION 4 — FULL QUANTUM CIRCUITS
# =============================================================================
print(f"\n[4] Building Full Quantum Circuits")


def build_full_circuit(state_vec:    np.ndarray,
                       pcm_subckt:   QuantumCircuit,
                       label:        str) -> QuantumCircuit:
    """INIT → PCM_poly → InvAvg → MEASURE (5 qubits)."""
    n_qubits = 5
    q = QuantumRegister(n_qubits, 'q')
    c = ClassicalRegister(n_qubits, 'c')
    qc = QuantumCircuit(q, c, name=label)
    qc.initialize(state_vec.tolist(), q)
    qc.barrier(label='init')
    qc.append(pcm_subckt.to_gate(label='PCM_poly'), q)
    qc.barrier(label='PCM')
    # InvAvg: H^5, X^5, H.MCX.H on q4, X^5, H^5
    qc.h(q)
    qc.barrier(label='inv_H1')
    qc.x(q)
    qc.barrier(label='inv_X1')
    qc.h(q[4])
    qc.mcx([q[0], q[1], q[2], q[3]], q[4])
    qc.h(q[4])
    qc.barrier(label='inv_MCZ')
    qc.x(q)
    qc.barrier(label='inv_X2')
    qc.h(q)
    qc.barrier(label='inv_H2')
    qc.measure(q, c)
    return qc


circuits = {lbl: build_full_circuit(vec, PCM_circ, f'InvAvg.PCM_poly.|{lbl}>')
            for lbl, vec in [('LetP',LetP),('LetA',LetA),('Input1',Input1),('Input2',Input2)]}

print(f"    LetP circuit (high-level view):")
print(circuits['LetP'].draw('text', fold=90))


# =============================================================================
# SECTION 5 — QUANTUM SIMULATION  (AerSimulator, 1024 shots)
# =============================================================================
print(f"\n[5] Quantum Simulation  (AerSimulator, {SHOTS} shots each)")

simulator = AerSimulator()
counts = {}
for lbl, qc in circuits.items():
    qc_t = transpile(qc, simulator, optimization_level=1)
    counts[lbl] = simulator.run(qc_t, shots=SHOTS).result().get_counts()
    print(f"    {lbl}: {len(counts[lbl])} distinct outcomes")


# =============================================================================
# SECTION 6 — MEASUREMENT STATISTICS
# =============================================================================

def report(label: str, cnts: dict, ideal: np.ndarray, shots: int = 1024) -> None:
    c0  = cnts.get('00000', 0)
    c_o = shots - c0
    p0  = c0  / shots * 100
    po  = c_o / shots * 100

    print(f"\n  +--------------------------------------------------------------+")
    print(f"  |  {label:<60}|")
    print(f"  |  {shots} shots{' '*(59-len(str(shots))-6)}|")
    print(f"  +--------------------------------------------------------------+")
    print(f"  |  Outcome   Observed     Ideal (expected)                      |")
    print(f"  +--------------------------------------------------------------+")
    print(f"  |  '00000'   {c0:5d}        ~{ideal[0]*shots:5.1f}  ({ideal[0]*100:.3f}%)"
          f"  {'#'*int(p0/2):<26}|")
    print(f"  |  other     {c_o:5d}        ~{(1-ideal[0])*shots:5.1f}  ({(1-ideal[0])*100:.3f}%)"
          f"  {'#'*int(po/2):<26}|")
    print(f"  +--------------------------------------------------------------+")
    print(f"  |  '00000' = {p0:5.1f}%   |   other = {po:5.1f}%"
          + " " * 25 + "|")
    print(f"  +--------------------------------------------------------------+")
    print(f"  |  Full counts (sorted):                                        |")
    for bs, cnt in sorted(cnts.items(), key=lambda x: -x[1]):
        idx = int(bs, 2)
        bar = '|' * int(cnt / shots * 25)
        print(f"  |    |{bs}>  obs={cnt:4d} ({cnt/shots*100:5.1f}%)"
              f"  ideal~{ideal[idx]*100:5.3f}%  {bar:<15}|")
    print(f"  +--------------------------------------------------------------+")


print("\n\n" + "=" * 70)
print("  QUANTUM MEASUREMENT RESULTS  —  Polynomial PCM  (7-gate, O(n))")
print("  Circuit: InvAvg . PCM_poly . |input>   |   1024 shots each")
print("=" * 70)

for lbl in ('LetP','LetA','Input1','Input2'):
    report(f"InvAvg . PCM_poly . |{lbl}>", counts[lbl], ideal_probs[lbl], SHOTS)


# =============================================================================
# SECTION 7 — COMPACT SUMMARY
# =============================================================================
print("\n" + "-" * 70)
print("  COMPACT SUMMARY  —  '00000'  vs.  everything else")
print("-" * 70)
for lbl in ('LetP','LetA','Input1','Input2'):
    c0    = counts[lbl].get('00000', 0)
    c_oth = SHOTS - c0
    ideal = ideal_probs[lbl][0]
    print(f"\n  InvAvg.PCM_poly.|{lbl:<7}  (ideal {ideal*100:.3f}%):")
    print(f"      '00000'  :  {c0:4d}  ({c0/SHOTS*100:.1f}%)")
    print(f"      other    :  {c_oth:4d}  ({c_oth/SHOTS*100:.1f}%)")

gap_sim = abs(counts['LetP'].get('00000',0) - counts['LetA'].get('00000',0)) / SHOTS
gap_ideal = abs(ideal_probs['LetP'][0] - ideal_probs['LetA'][0])
print(f"\n  Gap |P_LetP − P_LetA| (ideal) :  {gap_ideal*100:.4f} pp")
print(f"  Gap |P_LetP − P_LetA| (simul.) :  {gap_sim*100:.4f} pp")
print(f"  Fraction of global optimum    :  {gap_ideal/0.3804*100:.1f}%")
print(f"  vs original 7-gate (CZ) gap   :  0.23 pp  (87x improvement)")
print("-" * 70)
print("  Done.\n")
