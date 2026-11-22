"""Shared definitions for the revision scripts of the Light-Weight Quantum Image Classifier.

Every single-copy scheme considered in the paper is a two-outcome projective
measurement {|w><w|, I - |w><w|}: the "distinguished" outcome occurs with
probability p(v) = (w . v)^2 for an input state v.  This module provides

  * the image encoding of Eq. (QImgRep):  |x>|y>|c>,  c = 0 (black) / 1 (white)
  * the measurement vector w of each scheme
  * the decision rules over k independent copies

Schemes
  kl_grover_opt : proposed classifier (KL transform + Grover diffusion) with the
                  gap-optimal null-space completion, w = PCM^T D |0>
                  = -(1-2/N) e1 + (2/N) sqrt(N-1) e_s
  kl_grover_zl  : proposed classifier with the zero-leakage completion (w . psi1 = 0),
                  which exists iff delta^2 <= 16 (N-1)/N^2; otherwise = kl_grover_opt
  template      : projection onto the first template,  w = psi1            ("bright")
  dark          : projection onto the part of psi2 orthogonal to psi1      ("dark port")
  helstrom      : single-copy Helstrom (minimum-error) projector in span{psi1, psi2}
"""
import numpy as np
from scipy.stats import binom

# ----------------------------------------------------------------------------- encoding
def encode(bitmap):
    """Binary image (2-D array, 1 = ink/black) -> real unit vector of Eq. (QImgRep).

    Basis index = (x, y, c) read as a bit string, x = row, y = column,
    c = 0 for a black pixel and 1 for a white pixel.
    """
    bm = np.asarray(bitmap, dtype=int)
    r, c = bm.shape
    v = np.zeros(2 * r * c)
    idx = (np.arange(r)[:, None] * c + np.arange(c)[None, :]) * 2 + (1 - bm)
    v[idx.ravel()] = 1.0
    return v / np.sqrt(r * c)

def glyph(rows):
    return np.array([[1 if ch == '#' else 0 for ch in row] for row in rows])

GLYPHS = {name: glyph(rows) for name, rows in {
    'P': ["###.", "#..#", "###.", "#..."],      # class representative of the paper
    'A': [".##.", "#..#", "####", "#..#"],      # class representative of the paper
    'R': ["###.", "#..#", "###.", "#..#"],
    'F': ["####", "#...", "###.", "#..."],
    'E': ["####", "###.", "#...", "####"],
    'H': ["#..#", "####", "#..#", "#..#"],
    'L': ["#...", "#...", "#...", "####"],
    'T': ["####", ".##.", ".##.", ".##."],
    'O': ["####", "#..#", "#..#", "####"],
    'C': ["####", "#...", "#...", "####"],
    'U': ["#..#", "#..#", "#..#", "####"],
    'D': ["###.", "#..#", "#..#", "###."],
}.items()}

def paper_states():
    """|P> and |A> exactly as listed in Eqs. (eqLetP), (eqLetA) of the paper."""
    def st(kets):
        v = np.zeros(32)
        for k in kets:
            v[int(k, 2)] = 0.25
        return v
    P = st("00000 00010 00100 00111 01000 01011 01101 01110 "
           "10000 10010 10100 10111 11000 11011 11101 11111".split())
    A = st("00001 00010 00100 00111 01000 01011 01101 01110 "
           "10000 10010 10100 10110 11000 11011 11101 11110".split())
    assert np.allclose(P, encode(GLYPHS['P'])) and np.allclose(A, encode(GLYPHS['A']))
    return P, A

# ----------------------------------------------------------------------------- measurement vectors
def directions(psi1, psi2):
    e1 = (psi1 - psi2) / np.linalg.norm(psi1 - psi2)
    es = (psi1 + psi2) / np.linalg.norm(psi1 + psi2)
    return e1, es

def w_from_pcm(PCM):
    """Measurement vector of the proposed scheme for an explicit PCM matrix."""
    N = PCM.shape[0]
    D = 2.0 / N * np.ones((N, N)) - np.eye(N)
    return PCM.T @ D[:, 0]

def w_kl_grover_opt(psi1, psi2):
    N = len(psi1)
    e1, es = directions(psi1, psi2)
    return -(1 - 2.0 / N) * e1 + (2.0 / N) * np.sqrt(N - 1) * es

def zero_leakage_exists(psi1, psi2):
    """True iff a completion with P(|0> | psi1) = 0 exists:  delta^2 <= 16 (N-1) / N^2.

    For the binary-image encoding (delta^2 = 4 d / N, d = number of differing pixels)
    this is equivalent to d <= 3, for every image size.
    """
    N = len(psi1)
    d2 = float(np.sum((psi1 - psi2) ** 2))
    return d2 <= 16.0 * (N - 1) / N ** 2 + 1e-12

def free_direction(psi1, psi2):
    """Canonical unit vector g orthogonal to both templates (from the uniform vector)."""
    e1, es = directions(psi1, psi2)
    g = np.ones(len(psi1))
    g = g - (g @ e1) * e1 - (g @ es) * es
    return g / np.linalg.norm(g)

def w_kl_grover_zl(psi1, psi2):
    """Proposed scheme with the zero-leakage completion (falls back to gap-optimal).

    v_free = sqrt(N-1) (cos(phi) e_s + sin(phi) g) with phi chosen such that w . psi1 = 0;
    then P(|0> | psi1) = 0 exactly and P(|0> | psi2) = ((N-2)/N)^2 delta^2.
    """
    N = len(psi1)
    if not zero_leakage_exists(psi1, psi2):
        return w_kl_grover_opt(psi1, psi2)
    e1, es = directions(psi1, psi2)
    delta = np.linalg.norm(psi1 - psi2)
    cosphi = (N - 2) * delta / (4 * np.sqrt(N - 1) * np.sqrt(1 - delta ** 2 / 4))
    f = cosphi * es + np.sqrt(max(0.0, 1 - cosphi ** 2)) * free_direction(psi1, psi2)
    return -(1 - 2.0 / N) * e1 + (2.0 / N) * np.sqrt(N - 1) * f

def pcm_from_free_direction(e1, f):
    """Explicit orthogonal PCM with first row e1 and column sums e1 + sqrt(N-1) f  (f unit, f _|_ e1)."""
    N = len(e1)
    Q, _ = np.linalg.qr(np.column_stack([e1, f, np.random.default_rng(0).normal(size=(N, N - 2))]))
    G = Q[:, 2:]
    Us, _, _ = np.linalg.svd(np.eye(N - 1) - np.ones((N - 1, N - 1)) / (N - 1))
    S = Us[:, :N - 2]                                   # rows = vertices of a regular simplex
    M = np.vstack([e1] + [f / np.sqrt(N - 1) + G @ S[k] for k in range(N - 1)])
    assert np.allclose(M @ M.T, np.eye(N), atol=1e-9)
    return M

def w_dark(psi1, psi2):
    """'Dark-port' template projection: unit vector in span{psi1, psi2} orthogonal to psi1."""
    w = psi2 - (psi1 @ psi2) * psi1
    return w / np.linalg.norm(w)

def w_template(psi1, psi2):
    return psi1.copy()

def w_helstrom(psi1, psi2):
    e1, es = directions(psi1, psi2)
    return (es + e1) / np.sqrt(2)          # 45 degrees from the bisector, on the psi1 side

SCHEMES = {'kl_grover_zl': w_kl_grover_zl, 'kl_grover_opt': w_kl_grover_opt,
           'template': w_template, 'dark': w_dark, 'helstrom': w_helstrom}
LABELS = {'kl_grover_zl': 'KL + Grover diffusion, zero-leakage completion',
          'kl_grover_opt': 'KL + Grover diffusion, gap-optimal completion',
          'template': 'Template projection (bright)',
          'dark': 'Template projection (dark port)',
          'helstrom': 'Single-copy Helstrom projector'}

# ----------------------------------------------------------------------------- decision rules over k copies
def ml_threshold(p1, p2, k):
    """Maximum-likelihood count rule calibrated on the two templates (equal priors).

    Returns (t, hi): decide the class with the larger p ('hi' = 1 or 2) iff the number
    m of distinguished outcomes satisfies m >= t.
    """
    hi = 1 if p1 >= p2 else 2
    ph, pl = max(p1, p2), min(p1, p2)
    m = np.arange(k + 1)
    favours_hi = binom.pmf(m, k, ph) > binom.pmf(m, k, pl)
    t = int(np.argmax(favours_hi)) if favours_hi.any() else k + 1
    return t, hi

def midpoint_threshold(p1, p2, k):
    """Template-calibrated 'nearest mean' rule: hi iff m/k > (p1 + p2)/2 (ties split)."""
    hi = 1 if p1 >= p2 else 2
    return k * (p1 + p2) / 2.0, hi

def prob_label_hi(p, k, t):
    """P(decide 'hi') for a sample with single-shot probability p; real t => ties split."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    if float(t).is_integer():
        t = int(t)
        return binom.sf(t - 1, k, p)                       # P(m >= t)
    return binom.sf(np.floor(t), k, p)                     # P(m > t), no tie possible

def prob_label_hi_mid(p, k, t):
    """Midpoint rule with ties counted as a fair coin."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    out = binom.sf(np.floor(t), k, p)
    if float(t).is_integer():
        out = out + 0.5 * binom.pmf(int(t), k, p)
    return out

def template_error(p1, p2, k):
    """Average error on the two templates under the ML count rule (equal priors)."""
    t, hi = ml_threshold(p1, p2, k)
    ph, pl = max(p1, p2), min(p1, p2)
    return 0.5 * ((1 - prob_label_hi(ph, k, t)) + prob_label_hi(pl, k, t))

def collective_helstrom(overlap, k):
    return 0.5 * (1 - np.sqrt(1 - overlap ** (2 * k)))

def copies_needed(p1, p2, target, kmax=20000):
    """Smallest k with template_error <= target (None if not reached by kmax)."""
    k = 1
    while k <= kmax:
        if template_error(p1, p2, k) <= target:
            lo = max(1, k // 2)
            for kk in range(lo, k + 1):
                if template_error(p1, p2, kk) <= target:
                    return kk
        k *= 2
    return None
