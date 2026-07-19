"""Qiskit simulation of the binary classifier under depolarizing noise (Section 4.2).

The input state is set exactly (noise-free), consistent with the access model;
depolarizing noise (p1 = p2/10 on single-qubit gates, p2 on CNOT) acts on the
classifier circuit only (PCM + Grover diffusion, generic QSD synthesis).
Produces noise_simulation.png (Figure in Section 4.2).
Tested with qiskit 2.5.0, qiskit-aer 0.17.2.
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import qiskit_aer.library  # enables qc.set_statevector
import matplotlib.pyplot as plt

def state(kets):
    v = np.zeros(32)
    for k in kets: v[int(k, 2)] = 0.25
    return v

LetP = state(['00000','00010','00100','00111','01000','01011','01101','01110',
              '10000','10010','10100','10111','11000','11011','11101','11111'])
LetA = state(['00001','00010','00100','00111','01000','01011','01101','01110',
              '10000','10010','10100','10110','11000','11011','11101','11110'])
S1 = state(['00001','00010','00100','00111','01000','01011','01101','01110',   # faulty P
            '10000','10010','10100','10111','11000','11011','11101','11111'])
S2 = state(['00001','00010','00100','00111','01000','01011','01101','01110',   # faulty A
            '10000','10010','10100','10110','11000','11011','11101','11111'])

# PCM from NumPy SVD of the centred data matrix (as in Section 4)
mu = 0.5*(LetP+LetA)
U, s, Vt = np.linalg.svd(np.column_stack([LetP-mu, LetA-mu]), full_matrices=True)
PCM = U.T.copy()
D = 2/32*np.ones((32, 32)) - np.eye(32)
if (D @ (PCM @ LetA))[0]**2 < (D @ (PCM @ LetP))[0]**2:
    PCM[0] = -PCM[0]  # sign convention: |00000> labels A

clf = QuantumCircuit(5, 5)
clf.unitary(PCM, range(5), label='PCM')
clf.h(range(5)); clf.x(range(5)); clf.h(4); clf.mcx([0,1,2,3], 4); clf.h(4)
clf.x(range(5)); clf.h(range(5)); clf.measure(range(5), range(5))
tclf = transpile(clf, basis_gates=['cx','rz','sx','x','id'],
                 optimization_level=3, seed_transpiler=11)
print("classifier circuit: depth =", tclf.depth(), " CX =", tclf.count_ops().get('cx', 0))

def noise_model(p2):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p2/10, 1), ['sx','x','rz','id'])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ['cx'])
    return nm

def run(v, p2, shots=8192, seed=7):
    qc = QuantumCircuit(5, 5)
    qc.set_statevector(v)
    qc.compose(tclf, inplace=True)
    backend = AerSimulator(noise_model=noise_model(p2)) if p2 > 0 else AerSimulator()
    counts = backend.run(qc, shots=shots, seed_simulator=seed).result().get_counts()
    return counts.get('00000', 0)/shots

inputs = [("LetP", LetP), ("LetA", LetA), ("faulty P", S1), ("faulty A", S2)]
p2s = [0.0, 0.001, 0.005, 0.01]
data = np.array([[run(v, p2) for p2 in p2s] for _, v in inputs])
for (name, _), row in zip(inputs, data):
    print(name, " ".join(f"p2={p2}: {100*f:6.2f}%" for p2, f in zip(p2s, row)))

labels = ["ideal\n(statevector)", "$p_2=10^{-3}$", "$p_2=5\\cdot10^{-3}$", "$p_2=10^{-2}$"]
x = np.arange(len(labels)); w = 0.2
colors = ['#1f77b4', '#d62728', '#9ecae1', '#ff9896']
plt.figure(figsize=(7.6, 4.4))
for i, (name, _) in enumerate(inputs):
    plt.bar(x+(i-1.5)*w, 100*data[i], w, label=name, color=colors[i])
plt.axhline(100/32, color='gray', ls='--', lw=1)
plt.text(2.62, 100/32+0.5, 'fully mixed  (1/32)', color='gray', fontsize=8)
plt.ylabel('$P(|00000\\rangle)$  [%]')
plt.title('Classifier under depolarizing noise')
plt.xticks(x, labels); plt.legend(fontsize=9); plt.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig('noise_simulation.png', dpi=200)
