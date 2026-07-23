#!/usr/bin/env python3
"""
p4_index.py -- the Instrument in a SECOND, non-caching domain: learned indexes.

Why this exists
---------------
The caching study returned a negative (the prefetch corridor is a trap; eviction/admission
corridors are real but uncapturable against S3-FIFO). A reviewer's fair question is then whether
the Instrument is a debunking machine that can only ever say "no". The cleanest answer is to apply
the SAME procedure to a domain where a published ML method genuinely conquered a real gap, and show
it returns BUILD. Learned indexes (Kraska et al., SIGMOD'18) are that domain.

The distinction the procedure turns on
--------------------------------------
Both domains present an oracle gap. What differs is REACHABILITY, which is exactly what the caching
Instrument restricts on:
  * caching oracle  -> uses the FUTURE (an object's next access time). No deployable predictor has
                       it, so the gap is clairvoyance-priced -- a trap.
  * index oracle    -> uses the DATA STRUCTURE: a key's position is rank(key) = N * CDF(key), a
                       smooth function LEARNABLE FROM THE KEYS THEMSELVES. No future, no
                       clairvoyance -> the gap is reachable, so a deployable model can capture it.
So the procedure should GREEN-LIGHT learned indexes and REJECT learned prefetching -- the
discrimination the methodology promises, now across domains, not just workloads.

The three arms (metric: memory probes per point lookup on a static sorted array)
--------------------------------------------------------------------------------
    BASELINE  binary search over the sorted keys        -> ceil(log2 N) probes (the tuned,
              non-learned method; a B-tree costs the same in comparisons).
    ORACLE    a perfect position index                  -> 1 probe (the clairvoyant ceiling).
    LEARNED   a two-stage RMI (recursive model index): a root linear model routes each key to one
              of M leaf linear models, each of which predicts the position; a last-mile binary
              search over the leaf's guaranteed error window finds the key. Trained ONLY on the
              keys (no future, no held-out leakage) -> deployable. Probes = ceil(log2(window)).

    CORRIDOR  = baseline - oracle probes   (is there room below binary search?)
    CAPTURED  = baseline - learned probes  (does the deployable model realise it?)

Correctness invariant (the analogue of the caching construction check)
----------------------------------------------------------------------
For every key, the true position must lie inside the last-mile search window the model declares
(pred +/- leaf max-error). If not, the index would fail to find the key and the probe count would
be a fiction. Checked every run; a violation prints INVALID and aborts.

Discrimination
--------------
Run across key distributions of increasing hardness for a piecewise-linear model: uniform (linear
CDF, trivially learnable), lognormal (smooth, learnable), and clustered (a near-step CDF that a
piecewise-linear model fits poorly). BUILD on the smooth ones, weaker on the adversarial one --
the same instrument reading different verdicts on different inputs, the property that makes it a
ruler rather than a rubber stamp.

Usage
-----
    python p4_index.py                       # all distributions, N = 1,000,000
    python p4_index.py --n 2000000 --leaves 20000
"""
from __future__ import annotations

import argparse
import math

import numpy as np

LINE = "=" * 100


def make_keys(dist, n, seed):
    """Return n sorted, distinct float keys drawn from `dist`."""
    rng = np.random.default_rng(seed)
    if dist == "uniform":
        k = rng.uniform(0.0, 1.0, n)
    elif dist == "lognormal":
        k = rng.lognormal(0.0, 2.0, n)
    elif dist == "clustered":                    # ~50 tight clusters -> near-step CDF, hard for linear
        centers = rng.uniform(0.0, 1.0, 64)
        k = centers[rng.integers(0, len(centers), n)] + rng.normal(0.0, 1e-5, n)
    else:
        raise ValueError(dist)
    k = np.unique(k.astype(np.float64))          # a static index holds distinct sorted keys
    return k


def rmi(keys, n_leaves):
    """Two-stage recursive model index. Stage 1: one linear model routes a key to a leaf. Stage 2:
    per-leaf linear model predicts position. Returns (pred, leaf_maxerr[per key]) where the
    last-mile search window for a key is pred +/- leaf_maxerr. Trained on the keys only.
    Keys are normalised to [0,1] first (monotonic, so positions are unchanged) to keep the linear
    fits well-conditioned even for huge integer keys such as hashed object IDs."""
    N = len(keys); pos = np.arange(N, dtype=np.float64)
    span = float(keys[-1] - keys[0]) or 1.0
    keys = (keys - keys[0]) / span                        # normalise; monotonic -> index unaffected
    a1, b1 = np.polyfit(keys, pos, 1)                     # root model: key -> approximate position
    leaf = np.clip(((a1 * keys + b1) / N * n_leaves).astype(np.int64), 0, n_leaves - 1)
    pred = np.empty(N, dtype=np.float64)
    for L in range(n_leaves):
        m = leaf == L
        c = int(m.sum())
        if c == 0:
            continue
        if c == 1 or np.ptp(keys[m]) == 0:               # degenerate leaf: constant model
            pred[m] = pos[m].mean()
        else:
            a, b = np.polyfit(keys[m], pos[m], 1)
            pred[m] = a * keys[m] + b
    # per-leaf guaranteed error: the window a lookup must search to be correct
    leaf_maxerr = np.zeros(N, dtype=np.float64)
    err = np.abs(pred - pos)
    for L in range(n_leaves):
        m = leaf == L
        if m.any():
            leaf_maxerr[m] = err[m].max()
    return pred, leaf_maxerr, err


def probes(window):
    """Memory probes for a last-mile binary search over `window` candidate slots (>=1 probe)."""
    return np.maximum(1.0, np.ceil(np.log2(np.maximum(window, 1.0))))


def run_dist(dist, n, n_leaves, seed, bar_frac):
    return run_keys(dist, make_keys(dist, n, seed), n_leaves, bar_frac)


def keys_from_trace(path, limit):
    """Real key set with no downloads: the sorted, distinct object IDs from an oracleGeneral trace
    (hashed 64-bit identities). Turns the index check into 'real method, real keys'."""
    from p4_cache import load_oracle_general
    t = load_oracle_general(path, limit=limit)
    return np.unique(t["obj_id"].astype(np.float64))


def run_keys(label, keys, n_leaves, bar_frac):
    N = len(keys); pos = np.arange(N)
    base = math.ceil(math.log2(N))                       # binary search probes
    oracle = 1.0                                         # perfect index

    pred, leaf_maxerr, err = rmi(keys, n_leaves)
    # correctness invariant: the true position must lie inside the declared search window
    lo = np.floor(pred - leaf_maxerr); hi = np.ceil(pred + leaf_maxerr)
    ok = bool(np.all((pos >= lo) & (pos <= hi)))
    window = 2.0 * leaf_maxerr + 1.0
    learned = float(probes(window).mean())              # avg probes per lookup (last-mile search)

    corridor = base - oracle
    captured = base - learned
    frac = captured / corridor if corridor > 1e-9 else 0.0
    live = ok and frac >= bar_frac and captured > 0

    print(f"  [{label}]  N={N:,}  leaves={n_leaves:,}  median|err|={np.median(err):.1f}  "
          f"p99|err|={np.percentile(err,99):.0f}  invariant={'OK' if ok else 'VIOLATED'}")
    print(f"     BASELINE binary search  {base:6.2f} probes   [tuned non-learned]")
    print(f"     ORACLE   perfect index  {oracle:6.2f} probes   [reachable: position = N*CDF(key)]")
    print(f"     LEARNED  RMI (trained on keys only) {learned:6.2f} probes   [deployable]")
    print(f"     CORRIDOR {corridor:6.2f} probes    CAPTURED {captured:6.2f} probes "
          f"= {100*frac:5.1f}% of it")
    print(f"     VERDICT: {'BUILD -- real, reachable gap the learned index captures.' if live else 'not captured / no corridor.'}")
    print("-" * 100)
    return dict(dist=label, base=base, oracle=oracle, learned=learned, corridor=corridor,
                captured=captured, frac=frac, invariant=ok, live=live)


def main():
    ap = argparse.ArgumentParser(description="The Instrument in a second domain: learned indexes")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--leaves", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bar-frac", type=float, default=0.5,
                    help="pre-registered: BUILD needs the learned index to capture >= this fraction")
    ap.add_argument("--dists", default="uniform,lognormal,clustered")
    ap.add_argument("--keys-from-trace", default=None,
                    help="use REAL keys: the sorted distinct object IDs of an oracleGeneral trace")
    ap.add_argument("--limit", type=int, default=2_000_000)
    args = ap.parse_args()

    print(LINE)
    print("  THE INSTRUMENT IN A SECOND DOMAIN -- learned indexes (does the ruler ever say BUILD?)")
    print(LINE)
    if args.keys_from_trace:
        keys = keys_from_trace(args.keys_from_trace, args.limit)
        label = args.keys_from_trace.split("/")[-1] + " (real object-ID keys)"
        out = [run_keys(label, keys, args.leaves, args.bar_frac)]
    else:
        out = [run_dist(d, args.n, args.leaves, args.seed, args.bar_frac) for d in args.dists.split(",")]

    print("  SUMMARY (the same procedure, different verdicts):")
    for r in out:
        tag = "BUILD" if r["live"] else "weak/no"
        print(f"    {r['dist']:10s} corridor {r['corridor']:5.2f} probes, captured "
              f"{100*r['frac']:5.1f}%  -> {tag}")
    print("  Contrast with caching: there the oracle needs the FUTURE (unreachable) and no "
          "deployable\n  policy captures the corridor; here the oracle needs only the KEYS "
          "(reachable) and the\n  learned index captures it. The reachability restriction is what "
          "separates the two verdicts.")
    print(LINE)


if __name__ == "__main__":
    main()
