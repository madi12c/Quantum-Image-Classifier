"""Generalization and robustness study (Section 6).

All single-copy schemes are rank-one projections, P(distinguished outcome | v) = (w . v)^2,
so every probability below is exact (no sampling of measurement outcomes); the only
sampling is that of test images, for which 95% confidence intervals are reported.

Part A  exhaustive pixel perturbations of the P / A templates (all images within Hamming
        distance 1 and 2 of a template; ground truth = nearest template, ties excluded)
Part B  all 66 pairs of twelve 4 x 4 glyphs: templates and their one-pixel perturbations
Part C  scaling with image size: fixed overlap (pixel replication) and fixed number of
        differing pixels (glyph embedded in a larger white canvas)
Part D  independent test samples: scikit-learn 8 x 8 handwritten digits, binarised;
        templates = class medoids of the training half (and randomly drawn training
        images), evaluation on the held-out half, all 45 digit pairs

Decision rules over k copies (m = number of distinguished outcomes):
  ML   maximum-likelihood count threshold calibrated on the two templates (the OR rule
       of the paper whenever the threshold is 1)
  MID  template-calibrated midpoint rule  m/k > (p1 + p2)/2
  CAL  (Part D only) one scalar threshold on m/k fitted on the training images

Produces generalization_perturbations.png, generalization_pairs.png, scaling.png,
generalization_digits.png and prints all numbers quoted in the paper.
"""
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qic_common import (GLYPHS, encode, SCHEMES, LABELS, ml_threshold, midpoint_threshold,
                        prob_label_hi, prob_label_hi_mid, template_error, copies_needed,
                        collective_helstrom, zero_leakage_exists)

KS = (1, 5, 12)
ORDER = ['kl_grover_zl', 'kl_grover_opt', 'template', 'dark', 'helstrom']
COL = {'kl_grover_zl': '#08306b', 'kl_grover_opt': '#2171b5', 'template': '#d62728',
       'dark': '#ff9896', 'helstrom': '#2ca02c'}

def p_correct(w, psi1, psi2, V, labels, k, rule):
    """Exact probability of a correct label for every test vector (rows of V); labels in {1, 2}."""
    p1, p2 = (w @ psi1) ** 2, (w @ psi2) ** 2
    p = (V @ w) ** 2
    if rule == 'ML':
        t, hi = ml_threshold(p1, p2, k)
        ph = prob_label_hi(p, k, t)
    else:
        t, hi = midpoint_threshold(p1, p2, k)
        ph = prob_label_hi_mid(p, k, t)
    return np.where(labels == hi, ph, 1 - ph)

def neighbours(bm1, bm2, radius):
    """All images within Hamming distance <= radius of either template, labelled by the nearest one."""
    out = {}
    flat = [bm1.ravel(), bm2.ravel()]
    npx = bm1.size
    for src in (0, 1):
        for r in range(1, radius + 1):
            for idx in itertools.combinations(range(npx), r):
                img = flat[src].copy(); img[list(idx)] ^= 1
                d1, d2 = int(np.sum(img != flat[0])), int(np.sum(img != flat[1]))
                if d1 != d2 and min(d1, d2) >= 1:
                    out[img.tobytes()] = (img.reshape(bm1.shape), 1 if d1 < d2 else 2, min(d1, d2))
    imgs = [v[0] for v in out.values()]
    return (np.array([encode(i) for i in imgs]), np.array([v[1] for v in out.values()]),
            np.array([v[2] for v in out.values()]))

# ============================================================================= Part A
print("=" * 100, "\nPart A: exhaustive perturbations of the P / A templates\n")
bP, bA = GLYPHS['P'], GLYPHS['A']
P, A = encode(bP), encode(bA)
V, lab, dist = neighbours(bP, bA, 2)
print(f"test set: {np.sum(dist == 1)} images at distance 1 and {np.sum(dist == 2)} at distance 2 "
      f"from the nearest template (ties excluded)")
resA = {}
for rule in ('ML', 'MID'):
    print(f"\n rule = {rule}:  mean / worst-case probability of the correct label [%]")
    print(f" {'scheme':48s}" + "".join(f"   d=1,k={k:<2d}   " for k in KS) + "".join(f"   d=2,k={k:<2d}   " for k in KS))
    for s in ORDER:
        w = SCHEMES[s](P, A)
        cells = []
        for d in (1, 2):
            for k in KS:
                pc = p_correct(w, P, A, V[dist == d], lab[dist == d], k, rule)
                cells.append((100 * pc.mean(), 100 * pc.min()))
        resA[(rule, s)] = cells
        print(f" {LABELS[s]:48s}" + "".join(f" {m:5.1f} /{mn:5.1f} " for m, mn in cells))

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
for j, rule in enumerate(('ML', 'MID')):
    xs = np.arange(6); wb = 0.16
    for i, s in enumerate(ORDER):
        ax[j].bar(xs + (i - 2) * wb, [c[0] for c in resA[(rule, s)]], wb, color=COL[s], label=LABELS[s])
    ax[j].set_xticks(xs); ax[j].set_xticklabels([f"$d$={d}\n$k$={k}" for d in (1, 2) for k in KS])
    ax[j].axhline(50, color='gray', ls='--', lw=1); ax[j].grid(axis='y', alpha=0.3)
    ax[j].set_title('ML (OR-type) rule calibrated on the templates' if rule == 'ML' else 'Midpoint rule')
ax[0].set_ylabel('Mean probability of the correct label [%]'); ax[1].legend(fontsize=6.5, loc='lower right')
fig.suptitle('All images within Hamming distance $d$ of the P / A templates (exact)')
fig.tight_layout(); fig.savefig('generalization_perturbations.png', dpi=200)

# ============================================================================= Part B
print("\n" + "=" * 100, "\nPart B: all pairs of twelve 4 x 4 glyphs\n")
names = list(GLYPHS)
rows = []
for a, b in itertools.combinations(names, 2):
    psi1, psi2 = encode(GLYPHS[a]), encode(GLYPHS[b])
    d = int(np.sum(GLYPHS[a] != GLYPHS[b]))
    Vn, labn, _ = neighbours(GLYPHS[a], GLYPHS[b], 1)
    row = {'pair': a + b, 'd': d, 'zl': zero_leakage_exists(psi1, psi2)}
    for s in ORDER:
        w = SCHEMES[s](psi1, psi2)
        p1, p2 = (w @ psi1) ** 2, (w @ psi2) ** 2
        row[s] = (template_error(p1, p2, 12),
                  min(p_correct(w, psi1, psi2, Vn, labn, 12, r).mean() for r in ('ML',)),
                  p_correct(w, psi1, psi2, Vn, labn, 12, 'MID').mean(), abs(p1 - p2))
    row['hel'] = collective_helstrom(float(psi1 @ psi2), 12)
    rows.append(row)
ds = np.array([r['d'] for r in rows])
print(f"{len(rows)} pairs; differing pixels d: min {ds.min()}, median {np.median(ds):.0f}, max {ds.max()}; "
      f"pairs with d <= 3 (zero-leakage completion exists): {int(np.sum(ds <= 3))}")
assert all(r['zl'] == (r['d'] <= 3) for r in rows), "zero-leakage criterion must coincide with d <= 3"
print("\naverage error on the two templates at k = 12 [%]  (median, [min, max] over pairs)")
for grp, mask in (("d <= 3", ds <= 3), ("d >= 4", ds >= 4), ("all", ds >= 0)):
    print(f"  pairs with {grp} ({int(mask.sum())}):")
    for s in ORDER:
        e = 100 * np.array([r[s][0] for r, m in zip(rows, mask) if m])
        print(f"    {LABELS[s]:48s} median {np.median(e):8.4f}   [{e.min():8.4f}, {e.max():8.4f}]")
    e = 100 * np.array([r['hel'] for r, m in zip(rows, mask) if m])
    print(f"    {'Collective Helstrom bound':48s} median {np.median(e):8.4f}   [{e.min():8.4f}, {e.max():8.4f}]")
print("\nmean probability of the correct label on all one-pixel perturbations, k = 12 [%] (median over pairs [min, max])")
for s in ORDER:
    ml = 100 * np.array([r[s][1] for r in rows]); mid = 100 * np.array([r[s][2] for r in rows])
    print(f"    {LABELS[s]:48s} ML rule {np.median(ml):5.1f} [{ml.min():5.1f}, {ml.max():5.1f}]   "
          f"midpoint rule {np.median(mid):5.1f} [{mid.min():5.1f}, {mid.max():5.1f}]")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
jit = np.linspace(-0.25, 0.25, len(ORDER))
for i, s in enumerate(ORDER):
    ax[0].semilogy(ds + jit[i], [max(100 * r[s][0], 1e-6) for r in rows], 'o', ms=3.5, color=COL[s], label=LABELS[s], alpha=0.8)
    ax[1].plot(ds + jit[i], [100 * r[s][2] for r in rows], 'o', ms=3.5, color=COL[s], alpha=0.8)
ax[0].axvline(3.5, color='gray', ls='--', lw=1); ax[1].axvline(3.5, color='gray', ls='--', lw=1)
ax[0].set_xlabel('Number of differing pixels $d$ between the two templates')
ax[1].set_xlabel('Number of differing pixels $d$ between the two templates')
ax[0].set_ylabel('Average error on the templates, $k=12$ [%]')
ax[1].set_ylabel('Correct label on 1-pixel perturbations,\n$k=12$, midpoint rule [%]')
ax[0].legend(fontsize=6.5); ax[0].grid(alpha=0.3, which='both'); ax[1].grid(alpha=0.3)
fig.suptitle('All 66 pairs of twelve $4\\times 4$ glyphs')
fig.tight_layout(); fig.savefig('generalization_pairs.png', dpi=200)

# ============================================================================= Part C
print("\n" + "=" * 100, "\nPart C: scaling with image size\n")
def scale_rows(make):
    out = []
    for f in (1, 2, 4, 8, 16):
        b1, b2 = make(f)
        psi1, psi2 = encode(b1), encode(b2)
        N = len(psi1)
        r = {'N': N, 'n': int(np.log2(N)), 'side': b1.shape[0], 'overlap': float(psi1 @ psi2)}
        for s in ORDER:
            w = SCHEMES[s](psi1, psi2)
            p1, p2 = (w @ psi1) ** 2, (w @ psi2) ** 2
            r[s] = (abs(p1 - p2), copies_needed(p1, p2, 0.01, kmax=2 ** 22))
        r['hel'] = next(k for k in range(1, 10 ** 6) if collective_helstrom(r['overlap'], k) <= 0.01)
        out.append(r)
    return out
rep = lambda f: (np.kron(bP, np.ones((f, f), int)), np.kron(bA, np.ones((f, f), int)))
def emb(f):
    c1 = np.zeros((4 * f, 4 * f), int); c2 = c1.copy(); c1[:4, :4] = bP; c2[:4, :4] = bA
    return c1, c2
scal = {'fixed overlap 13/16 (pixel replication)': scale_rows(rep),
        'fixed d = 3 differing pixels (embedding in a white canvas)': scale_rows(emb)}
for title, rws in scal.items():
    print(f" {title}\n  {'image':>8s} {'qubits':>6s} {'overlap':>8s} | gap [points] / copies needed for 1% average error on the templates")
    print("  " + " " * 25 + "".join(f"{s:>22s}" for s in ORDER) + f"{'coll. Helstrom':>16s}")
    for r in rws:
        print(f"  {r['side']:>3d}x{r['side']:<4d} {r['n']:>6d} {r['overlap']:8.5f} |" +
              "".join(f"{100*r[s][0]:10.3f} /{str(r[s][1]):>9s} " for s in ORDER) + f"{r['hel']:>14d}")
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
for j, (title, rws) in enumerate(scal.items()):
    for s in ORDER:
        ax[j].loglog([r['N'] for r in rws], [r[s][1] for r in rws], 'o-', color=COL[s], ms=4, label=LABELS[s])
    ax[j].loglog([r['N'] for r in rws], [r['hel'] for r in rws], 'k--', label='Collective Helstrom bound')
    ax[j].set_xlabel('Hilbert-space dimension $N = 2\\,\\times$ number of pixels'); ax[j].set_title(title, fontsize=9)
    ax[j].grid(alpha=0.3, which='both')
ax[0].set_ylabel('Copies needed for 1% average error'); ax[0].legend(fontsize=6.5)
fig.tight_layout(); fig.savefig('scaling.png', dpi=200)

# ============================================================================= Part D
print("\n" + "=" * 100, "\nPart D: independent test samples (scikit-learn 8 x 8 digits, binarised)\n")
from sklearn.datasets import load_digits
dg = load_digits()
X = (dg.images > 7).astype(int); y = dg.target
rng = np.random.default_rng(2026)
train = np.zeros(len(y), bool)
for c in range(10):
    idx = np.flatnonzero(y == c); rng.shuffle(idx); train[idx[:len(idx) // 2]] = True
test = ~train
enc = np.array([encode(im) for im in X])
flatX = X.reshape(len(X), -1)
def medoid(c):
    idx = np.flatnonzero(train & (y == c))
    dmat = (flatX[idx][:, None, :] != flatX[idx][None, :, :]).sum(-1)
    return idx[np.argmin(dmat.sum(1))]
med = {c: medoid(c) for c in range(10)}
print(f"{train.sum()} training images (used only to choose the templates), {test.sum()} held-out test images, "
      f"N = {enc.shape[1]} ({int(np.log2(enc.shape[1]))} qubits)")

def evaluate(i1, i2, c1, c2, ks=(12, 50, 200)):
    """Accuracy on the held-out images of classes c1, c2 with templates X[i1], X[i2]."""
    psi1, psi2 = enc[i1], enc[i2]
    sel = test & ((y == c1) | (y == c2))
    Vt, lt = enc[sel], np.where(y[sel] == c1, 1, 2)
    seltr = train & ((y == c1) | (y == c2))
    Vtr, ltr = enc[seltr], np.where(y[seltr] == c1, 1, 2)
    out = {}
    for s in ORDER:
        w = SCHEMES[s](psi1, psi2)
        for k in ks:
            out[(s, k)] = p_correct(w, psi1, psi2, Vt, lt, k, 'MID')
        p = (Vt @ w) ** 2; thr = ((w @ psi1) ** 2 + (w @ psi2) ** 2) / 2
        hi = 1 if (w @ psi1) ** 2 >= (w @ psi2) ** 2 else 2
        out[(s, 'inf')] = (np.where(p > thr, hi, 3 - hi) == lt).astype(float)
        # CAL: one scalar threshold on p, fitted on the TRAINING images of the two classes
        ptr = (Vtr @ w) ** 2
        cand = np.unique(np.concatenate([[0.0], (np.sort(ptr)[1:] + np.sort(ptr)[:-1]) / 2]))
        acc1 = np.array([np.mean(np.where(ptr > c, 1, 2) == ltr) for c in cand])
        best = np.argmax(np.maximum(acc1, 1 - acc1)); cthr = cand[best]; chi = 1 if acc1[best] >= 0.5 else 2
        for k in ks:
            ph = prob_label_hi_mid(p, k, k * cthr)
            out[(s, k, 'cal')] = np.where(lt == chi, ph, 1 - ph)
        out[(s, 'inf', 'cal')] = (np.where(p > cthr, chi, 3 - chi) == lt).astype(float)
    d1 = (flatX[sel] != flatX[i1]).sum(1); d2 = (flatX[sel] != flatX[i2]).sum(1)
    out['classical'] = np.where(d1 == d2, 0.5, (np.where(d1 < d2, 1, 2) == lt).astype(float))
    return out, int((flatX[i1] != flatX[i2]).sum())

def ci(v, B=2000, seed=0):
    r = np.random.default_rng(seed)
    bs = v[r.integers(0, len(v), size=(B, len(v)))].mean(1)
    return 100 * v.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)

pairs = list(itertools.combinations(range(10), 2))
allres, dists = {}, []
for c1, c2 in pairs:
    allres[(c1, c2)], d = evaluate(med[c1], med[c2], c1, c2); dists.append(d)
print(f"medoid templates: differing pixels between the two templates: min {min(dists)}, "
      f"median {np.median(dists):.0f}, max {max(dists)} (of 64)")
keys = ([(s, k) for s in ORDER for k in (12, 50, 200, 'inf')] +
        [(s, k, 'cal') for s in ORDER for k in (12, 50, 200, 'inf')] + ['classical'])
pooled = {key: np.concatenate([allres[p][key] for p in pairs]) for key in keys}
perpair = {key: np.array([allres[p][key].mean() for p in pairs]) for key in keys}
print("\nheld-out accuracy, medoid templates, midpoint rule; pooled over the 45 pairs "
      "[mean (95% bootstrap CI)] and worst pair")
print(f"  {'scheme':48s} {'k=12':>22s} {'k=50':>22s} {'k=200':>22s} {'k -> inf':>22s}")
for s in ORDER:
    cells = []
    for k in (12, 50, 200, 'inf'):
        m, lo, hi = ci(pooled[(s, k)]); cells.append(f"{m:5.1f} ({lo:4.1f}-{hi:4.1f}) w{100*perpair[(s, k)].min():5.1f}")
    print(f"  {LABELS[s]:48s} " + " ".join(f"{c:>22s}" for c in cells))
print("\nsame, with one scalar threshold per classifier fitted on the training half (CAL rule)")
for s in ORDER:
    cells = []
    for k in (12, 50, 200, 'inf'):
        m, lo, hi = ci(pooled[(s, k, 'cal')]); cells.append(f"{m:5.1f} ({lo:4.1f}-{hi:4.1f}) w{100*perpair[(s, k, 'cal')].min():5.1f}")
    print(f"  {LABELS[s]:48s} " + " ".join(f"{c:>22s}" for c in cells))
m, lo, hi = ci(pooled['classical'])
print(f"  {'Classical nearest template (Hamming, full pixel access)':48s} {m:5.1f} ({lo:4.1f}-{hi:4.1f}) "
      f"worst pair {100*perpair['classical'].min():5.1f}")

# sensitivity to the choice of templates: random training images as class representatives
R = 20
sens = {key: [] for key in keys}
for c1, c2 in pairs:
    for _ in range(R):
        i1 = rng.choice(np.flatnonzero(train & (y == c1))); i2 = rng.choice(np.flatnonzero(train & (y == c2)))
        res, _ = evaluate(i1, i2, c1, c2)
        for key in keys:
            sens[key].append(res[key].mean())
print(f"\nrandomly drawn training images as templates ({R} draws per pair, {R*len(pairs)} classifiers): "
      "held-out accuracy [%], mean (5th - 95th percentile over classifiers)")
for s in ORDER:
    cells = []
    for key in ((s, 12), (s, 200), (s, 200, 'cal')):
        v = 100 * np.array(sens[key]); cells.append(f"{v.mean():5.1f} ({np.percentile(v,5):4.1f}-{np.percentile(v,95):4.1f})")
    print(f"  {LABELS[s]:48s} MID k=12 {cells[0]}   MID k=200 {cells[1]}   CAL k=200 {cells[2]}")
v = 100 * np.array(sens['classical'])
print(f"  {'Classical nearest template':48s} {v.mean():5.1f} ({np.percentile(v,5):4.1f}-{np.percentile(v,95):4.1f})")

fig, ax = plt.subplots(figsize=(7.6, 4.4))
ks_plot = [12, 50, 200, 'inf']; xs = np.arange(len(ks_plot)); wb = 0.16
for i, s in enumerate(ORDER):
    vals = [ci(pooled[(s, k, 'cal')]) for k in ks_plot]
    ax.bar(xs + (i - 2) * wb, [v[0] for v in vals], wb, color=COL[s], label=LABELS[s],
           yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], capsize=2, error_kw={'lw': 0.8})
ax.axhline(ci(pooled['classical'])[0], color='k', ls='--', lw=1, label='Classical nearest template (full pixel access)')
ax.axhline(50, color='gray', ls=':', lw=1)
ax.set_xticks(xs); ax.set_xticklabels(['$k=12$', '$k=50$', '$k=200$', '$k\\to\\infty$'])
ax.set_ylabel('Held-out accuracy [%]'); ax.set_ylim(40, 100); ax.grid(axis='y', alpha=0.3); ax.legend(fontsize=6.5, loc='lower right', framealpha=0.95)
ax.set_title('Binarised $8\\times 8$ handwritten digits: 45 class pairs, medoid templates, held-out test images\n(decision threshold fitted on the training half; 95% bootstrap intervals)', fontsize=9)
fig.tight_layout(); fig.savefig('generalization_digits.png', dpi=200)
print("\nwrote generalization_perturbations.png, generalization_pairs.png, scaling.png, generalization_digits.png")
