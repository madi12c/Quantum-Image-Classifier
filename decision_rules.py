"""Decision-rule comparison for the quantum binary classifier (Section 3.2).

Compares the OR rule, majority vote, and the maximum-likelihood rule over
k independent repetitions of Algorithm 2, using the single-shot probabilities
of measuring |00000> reported in Section 3 (Mathematica PCM):
  p_P = 0.195%  (input = letter P representative)
  p_A = 28.088% (input = letter A representative)
Produces decision_rules.png (Figure in Section 3.2).
"""
import numpy as np
from math import comb, log
import matplotlib.pyplot as plt

pP, pA = 0.00195, 0.28088
s = 13/16  # overlap <P|A>

def binom_cdf(m, k, p):
    return sum(comb(k, i) * p**i * (1-p)**(k-i) for i in range(m+1))

ks = range(1, 16)
avg_or, avg_maj, hel = [], [], []
for k in ks:
    or_p, or_a = 1-(1-pP)**k, (1-pA)**k              # OR: label A iff m >= 1
    t = k//2 + 1                                      # majority: label A iff m > k/2
    maj_p, maj_a = max(0.0, 1-binom_cdf(t-1, k, pP)), binom_cdf(t-1, k, pA)
    avg_or.append(50*(or_p+or_a)); avg_maj.append(50*(maj_p+maj_a))
    hel.append(50*(1-np.sqrt(1-s**(2*k))))            # collective Helstrom bound

# ML threshold: label A iff m > k*ln[(1-pP)/(1-pA)] / ln[pA(1-pP)/(pP(1-pA))]
per = log((1-pP)/(1-pA)); den = log(pA/pP) + per
print(f"ML rule == OR rule for all k <= {int(den/per)}  (threshold slope {per/den:.4f})")

plt.figure(figsize=(7.2, 4.4))
plt.semilogy(list(ks), avg_or, 'o-', color='#1f77b4',
             label='OR rule  (= maximum-likelihood rule for $k\\leq 16$)')
plt.semilogy(list(ks), avg_maj, 's-', color='#d62728', label='Majority vote')
plt.semilogy(list(ks), hel, 'k--', label='Collective Helstrom bound (joint measurement)')
plt.xlabel('Number of repetitions  $k$'); plt.ylabel('Average misclassification rate [%]')
plt.title('Decision rules over $k$ independent repetitions of Algorithm 2')
plt.grid(True, which='both', alpha=0.3); plt.legend(fontsize=9); plt.xticks(list(ks))
plt.tight_layout(); plt.savefig('decision_rules.png', dpi=200)
