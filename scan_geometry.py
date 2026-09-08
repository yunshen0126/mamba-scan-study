"""
scan_geometry.py  --  design scalars and path generators for the scan-path study.

Two scalars describe any set of scan paths on an n x n token grid:

    lambda  = mean over spatially 4-adjacent cell pairs (u, v) of log(1 + |pi[u] - pi[v]|)
              small = each path keeps spatial neighbours close in the sequence
    alpha   = mean pairwise separation of the per-path step-orientation angles, / 90 deg
              large = the paths in the set sweep different spatial axes

and one auxiliary scalar guards against the obvious objection that a set of
near-monotone sweeps is "really the same path":

    succ_overlap = fraction of ordered consecutive-visit pairs two paths share

Representation convention, matching the existing pipeline:
    order[t] = row-major index of the cell visited at step t   (fed to index_select)
    pi[u]    = step at which cell u is visited
    pi[order] == arange(N)
Every generator returns `order` and asserts the inverse relation.

numpy only. Seeds are PCG64; substitute torch.Generator if bank-level parity
with the existing frozen artefacts is required.
"""

import json
import numpy as np

# ---------------------------------------------------------------------------
# representation
# ---------------------------------------------------------------------------

def to_pi(order):
    order = np.asarray(order, dtype=np.int64)
    N = order.size
    assert np.array_equal(np.sort(order), np.arange(N)), "order is not a permutation"
    pi = np.empty(N, dtype=np.int64)
    pi[order] = np.arange(N, dtype=np.int64)
    assert np.array_equal(pi[order], np.arange(N)), "order/pi inverse violated"
    return pi


def to_order(pi):
    return to_pi(pi)          # the map is an involution on permutations


def rc(idx, n):
    idx = np.asarray(idx)
    return idx // n, idx % n


# ---------------------------------------------------------------------------
# lambda : locality
# ---------------------------------------------------------------------------

def adjacent_pairs(n):
    idx = np.arange(n * n).reshape(n, n)
    h = np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1)
    v = np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1)
    return np.concatenate([h, v], axis=0)


def locality(order, n, pairs=None):
    """legacy d_seq statistics plus the lambda scalar"""
    pi = to_pi(order)
    if pairs is None:
        pairs = adjacent_pairs(n)
    d = np.abs(pi[pairs[:, 0]] - pi[pairs[:, 1]]).astype(np.float64)
    return {"d_mean": d.mean(), "d_p50": np.percentile(d, 50),
            "d_p90": np.percentile(d, 90), "lambda": np.log1p(d).mean()}


# ---------------------------------------------------------------------------
# alpha : axis spread of a path SET
# ---------------------------------------------------------------------------

def step_orientation(order, n):
    """theta in [0, 90]: 0 = every step horizontal, 90 = every step vertical.
    Built from mean |dc| and |dr|, hence invariant under path reversal."""
    order = np.asarray(order, dtype=np.int64)
    r, c = rc(order, n)
    dr = np.abs(np.diff(r)).mean()
    dc = np.abs(np.diff(c)).mean()
    return {"mean_abs_dc": dc, "mean_abs_dr": dr,
            "theta_deg": np.degrees(np.arctan2(dr, dc)),
            "axis_bias": (dc - dr) / (dc + dr)}


def alpha(orders, n):
    th = np.array([step_orientation(o, n)["theta_deg"] for o in orders])
    k = len(th)
    acc = [abs(th[i] - th[j]) for i in range(k) for j in range(i + 1, k)]
    return float(np.mean(acc) / 90.0), th


# ---------------------------------------------------------------------------
# path-set diversity
# ---------------------------------------------------------------------------

def successor_overlap(a, b):
    sa = set(zip(np.asarray(a)[:-1].tolist(), np.asarray(a)[1:].tolist()))
    sb = set(zip(np.asarray(b)[:-1].tolist(), np.asarray(b)[1:].tolist()))
    return len(sa & sb) / max(len(sa), 1)


def set_diversity(orders):
    k = len(orders)
    pis = [to_pi(o) for o in orders]
    ov = [successor_overlap(orders[i], orders[j]) for i in range(k) for j in range(i + 1, k)]
    cr = [abs(np.corrcoef(pis[i], pis[j])[0, 1]) for i in range(k) for j in range(i + 1, k)]
    return float(np.mean(ov)), float(np.mean(cr))


def describe(name, orders, n, pairs=None):
    if pairs is None:
        pairs = adjacent_pairs(n)
    st = [locality(o, n, pairs) for o in orders]
    a, th = alpha(orders, n)
    ov, cr = set_diversity(orders)
    return {"name": name,
            "lambda": float(np.mean([s["lambda"] for s in st])),
            "d_mean": float(np.mean([s["d_mean"] for s in st])),
            "d_p50": float(np.mean([s["d_p50"] for s in st])),
            "d_p90": float(np.mean([s["d_p90"] for s in st])),
            "alpha": a, "theta": np.round(th, 1).tolist(),
            "succ_overlap": ov, "pi_corr": cr, "k": len(orders)}


HDR = (f"{'condition':<20}{'lambda':>8}{'d_mean':>9}{'alpha':>7}"
       f"{'succ_ov':>9}{'|pi_corr|':>11}  theta(deg)")


def fmt(d):
    return (f"{d['name']:<20}{d['lambda']:>8.3f}{d['d_mean']:>9.1f}{d['alpha']:>7.3f}"
            f"{d['succ_overlap']:>9.4f}{d['pi_corr']:>11.3f}  {d['theta']}")


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------

def gen_G(n):
    """canonical rasters G1..G4, from the frozen analytic definition"""
    N = n * n
    t = np.arange(N)
    g1 = t.copy()
    g2 = g1[::-1].copy()
    g3 = (t % n) * n + (t // n)
    g4 = g3[::-1].copy()
    out = [g1, g2, g3, g4]
    for o in out:
        to_pi(o)
    return out


def gen_R(n, seed, k=4):
    rng = np.random.default_rng(seed)
    out = [rng.permutation(n * n) for _ in range(k)]
    for o in out:
        to_pi(o)
    return out


def gen_A(n, seed, w=8, k=4):
    """
    Condition A -- locality preserved, axis redundant.

    Every path is a row-major sweep. Inside each row the column order is a
    bounded random permutation, key = c + w * U(0,1); w = 0 reproduces G1.
    Half the paths are reversed, which leaves lambda and theta untouched but
    removes all shared ordered successors between a path and its twin.

    All paths share the coarse row sweep, so their step orientations coincide
    and alpha = 0 by construction.
    """
    assert k % 2 == 0
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(k // 2):
        rows = [r * n + np.argsort(np.arange(n) + w * rng.random(n), kind="stable")
                for r in range(n)]
        base = np.concatenate(rows)
        out.extend([base, base[::-1].copy()])
    for o in out:
        to_pi(o)
    return out


def gen_B(n, seed, k=4):
    """
    Condition B -- locality broken, axis complementary.

    Half the paths visit rows, half visit columns. The lines themselves are
    visited in random order and the cells inside each line are in random order.
    Every step but the line-to-line jump stays inside one line, so the step
    orientation remains axis-pure however far the sequence distance is pushed.
    """
    assert k % 2 == 0
    rng = np.random.default_rng(seed)

    def one(horizontal):
        pieces = []
        for line_idx in rng.permutation(n):
            cells = (line_idx * n + np.arange(n)) if horizontal else (np.arange(n) * n + line_idx)
            pieces.append(rng.permutation(cells))
        return np.concatenate(pieces)

    out = [one(True) for _ in range(k // 2)] + [one(False) for _ in range(k // 2)]
    for o in out:
        to_pi(o)
    return out


# ---------------------------------------------------------------------------
# loading a frozen bank of unknown schema
# ---------------------------------------------------------------------------

def load_bank(path, n):
    """
    Walk an arbitrary JSON structure and collect every list of integers that is
    a permutation of arange(n*n). Returns [(dotted_key, array), ...] so the
    caller can see which fields were picked up.
    """
    N = n * n
    found = []

    def walk(node, key):
        if isinstance(node, dict):
            for kk, vv in node.items():
                walk(vv, f"{key}.{kk}" if key else str(kk))
        elif isinstance(node, list):
            if len(node) == N and all(isinstance(x, int) for x in node):
                a = np.asarray(node, dtype=np.int64)
                if np.array_equal(np.sort(a), np.arange(N)):
                    found.append((key, a))
                    return
            for i, vv in enumerate(node):
                walk(vv, f"{key}[{i}]")

    with open(path) as f:
        walk(json.load(f), "")
    return found
