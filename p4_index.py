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
import warnings

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
    fits well-conditioned even for huge integer keys such as hashed object IDs.

    Vectorised: the per-leaf least-squares fits are computed with segmented sums (np.bincount) in
    O(N) rather than by looping over leaves in Python (which was O(n_leaves * N) and made a leaf
    sweep infeasible). A sweep is required, not optional: tuning the non-learned baseline family
    while leaving the learned model at one fixed capacity is the same under-tuning error this paper
    is about, pointed the other way. Both sides must be swept."""
    N = len(keys); pos = np.arange(N, dtype=np.float64)
    span = float(keys[-1] - keys[0]) or 1.0
    keys = (keys - keys[0]) / span                        # normalise; monotonic -> index unaffected
    warnings.simplefilter("ignore")                       # ill-conditioned leaves -> constant-ish fit, harmless
    a1, b1 = np.polyfit(keys, pos, 1)                     # root model: key -> approximate position
    leaf = np.clip(((a1 * keys + b1) / N * n_leaves).astype(np.int64), 0, n_leaves - 1)

    # segmented least squares: per-leaf slope/intercept from grouped sums
    M = n_leaves
    cnt = np.bincount(leaf, minlength=M).astype(np.float64)
    sx = np.bincount(leaf, weights=keys, minlength=M)
    sy = np.bincount(leaf, weights=pos, minlength=M)
    sxx = np.bincount(leaf, weights=keys * keys, minlength=M)
    sxy = np.bincount(leaf, weights=keys * pos, minlength=M)
    den = cnt * sxx - sx * sx
    safe = den > 1e-12                                    # else degenerate leaf -> constant model
    a = np.zeros(M); b = np.zeros(M)
    a[safe] = (cnt[safe] * sxy[safe] - sx[safe] * sy[safe]) / den[safe]
    b[safe] = (sy[safe] - a[safe] * sx[safe]) / cnt[safe]
    nz = cnt > 0
    b[~safe & nz] = sy[~safe & nz] / cnt[~safe & nz]       # constant fit = mean position
    pred = a[leaf] * keys + b[leaf]

    # per-leaf guaranteed error: the window a lookup must search to be correct
    err = np.abs(pred - pos)
    lmax = np.zeros(M)
    np.maximum.at(lmax, leaf, err)
    leaf_maxerr = lmax[leaf]
    return pred, leaf_maxerr, err


def probes(window):
    """Memory probes for a last-mile binary search over `window` candidate slots (>=1 probe)."""
    return np.maximum(1.0, np.ceil(np.log2(np.maximum(window, 1.0))))


def interpolation_probes(keys, q_idx, max_probes=96):
    """MEASURED probe count for interpolation search, simulated key by key (vectorised over a
    query sample). Counts an actual probe every time the algorithm reads keys[p].

    Why this baseline is mandatory, not optional. Interpolation search exploits EXACTLY the signal
    this study credits to the learned index -- that position is approximately N*CDF(key) -- and it
    does so with no model, no training, and no space. On a smooth key distribution it runs in
    O(log log N) probes against binary search's O(log N). Omitting it and calling binary search
    "the tuned non-learned baseline" is the same under-tuned-baseline error the caching half of this
    paper exists to catch, so the index arm has to face it or the BUILD verdict is not earned."""
    N = len(keys)
    q = keys[q_idx].astype(np.float64)
    lo = np.zeros(len(q), dtype=np.int64)
    hi = np.full(len(q), N - 1, dtype=np.int64)
    cnt = np.zeros(len(q), dtype=np.int64)
    done = np.zeros(len(q), dtype=bool)
    for _ in range(max_probes):
        act = ~done
        if not act.any():
            break
        klo, khi = keys[lo], keys[hi]
        den = khi - klo
        frac = np.where(den > 0, (q - klo) / np.where(den > 0, den, 1.0), 0.0)
        frac = np.clip(frac, 0.0, 1.0)
        p = lo + (frac * (hi - lo)).astype(np.int64)
        p = np.clip(p, lo, hi)
        cnt += act
        kp = keys[p]
        hit = act & (kp == q)
        done |= hit
        right = act & ~hit & (kp < q)
        left = act & ~hit & (kp > q)
        lo = np.where(right, p + 1, lo)
        hi = np.where(left, p - 1, hi)
        done |= act & ~hit & (lo > hi)          # range exhausted (cannot happen for present keys)
    return cnt


def interpolation_lines(keys, q_idx, cl, max_probes=96):
    """Distinct CACHE LINES touched by interpolation search (`cl` keys per line).

    Probe counts and cache-line counts are different units and they do not rank these structures
    the same way, which matters because the RMI's last-mile search over a small window may touch a
    single line while costing log2(window) comparisons. Charging comparisons to one arm and line
    fetches to another is the unit error that decides this comparison, so both are measured."""
    N = len(keys)
    q = keys[q_idx].astype(np.float64)
    lo = np.zeros(len(q), dtype=np.int64)
    hi = np.full(len(q), N - 1, dtype=np.int64)
    lines = np.zeros(len(q), dtype=np.int64)
    last = np.full(len(q), -1, dtype=np.int64)
    done = np.zeros(len(q), dtype=bool)
    for _ in range(max_probes):
        act = ~done
        if not act.any():
            break
        klo, khi = keys[lo], keys[hi]
        den = khi - klo
        frac = np.clip(np.where(den > 0, (q - klo) / np.where(den > 0, den, 1.0), 0.0), 0.0, 1.0)
        p = np.clip(lo + (frac * (hi - lo)).astype(np.int64), lo, hi)
        blk = p // cl
        lines += act & (blk != last)          # a repeat visit to the same line is free
        last = np.where(act, blk, last)
        kp = keys[p]
        hit = act & (kp == q)
        done |= hit
        right = act & ~hit & (kp < q)
        left = act & ~hit & (kp > q)
        lo = np.where(right, p + 1, lo)
        hi = np.where(left, p - 1, hi)
        done |= act & ~hit & (lo > hi)
    return lines


def window_dist(window, cl):
    """Distribution of last-mile search windows, in CACHE LINES, plus the mean lines actually
    charged. This is the diagnostic that explains a probes-vs-lines disagreement.

    The window is set by each leaf's MAX error, not its median, so a single badly-fit key drags
    every key in its leaf. Two consequences this measures directly:
      * SATURATION: any window <= cl fits in ONE line, so tightening it further is invisible to the
        cache-line metric while probes (which count log2(window) comparisons) keep improving. A
        model whose bulk is already sub-line gains nothing here from more capacity.
      * TAIL DOMINANCE: the mean is carried by the few keys in badly-fit leaves. If the >64-line
        share is flat as leaves increase, added capacity is fixing the bulk (invisible) and not the
        tail (what the mean is made of) -- which is exactly a probes-BUILD / lines-miss signature.
    """
    wl = np.maximum(window / cl, 1.0)
    charged = np.minimum(np.ceil(wl), np.maximum(1.0, np.ceil(np.log2(np.maximum(wl, 2.0)))))
    return dict(one=float((wl <= 1).mean()), few=float(((wl > 1) & (wl <= 8)).mean()),
                many=float(((wl > 8) & (wl <= 64)).mean()), tail=float((wl > 64).mean()),
                charged=float(charged.mean()))


def btree_probes(n, fanout):
    """Node accesses for a cache-resident B-tree of the given fanout: ceil(log_fanout N).

    One node occupies one cache line, so one node access is one memory probe -- the same unit the
    RMI's last-mile search is counted in. With 8-byte keys and 64-byte lines a fanout of 8-16 is
    the standard cache-optimised setting, which is why 'a B-tree costs the same as binary search'
    holds for COMPARISONS but not for PROBES, the metric used here."""
    return math.ceil(math.log(n, fanout))


def run_dist(dist, n, n_leaves, seed, bar_frac, fanout=16, cap_div=16):
    return run_keys(dist, make_keys(dist, n, seed), n_leaves, bar_frac, fanout, cap_div=cap_div)


def keys_from_trace(path, limit):
    """Real key set with no downloads: the sorted, distinct object IDs from an oracleGeneral trace
    (hashed 64-bit identities). Turns the index check into 'real method, real keys'. NOTE: hashed
    IDs are near-uniform by construction (the easy case for a learned index); use SOSD natural-key
    datasets for the realistic, harder distributions."""
    from p4_cache import load_oracle_general
    t = load_oracle_general(path, limit=limit)
    return np.unique(t["obj_id"].astype(np.float64))


def keys_from_sosd(path, n_target):
    """Real NATURAL keys: a SOSD benchmark dataset (books / osm_cellids / fb / wiki_ts), the
    standard proving ground for learned indexes. SOSD binary format = uint64 count header, then
    that many keys (uint32 or uint64, inferred from the filename). Strided-subsampled to n_target
    (preserves the sorted CDF shape). No credentials needed to download these."""
    with open(path, "rb") as f:
        count = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
        dt = np.uint32 if "uint32" in path else np.uint64
        arr = np.fromfile(f, dtype=dt, count=count)
    arr = np.unique(arr)                                  # sorted + distinct
    if len(arr) > n_target:
        arr = arr[np.linspace(0, len(arr) - 1, n_target).astype(np.int64)]
    return arr.astype(np.float64)


def run_keys(label, keys, n_leaves, bar_frac, fanout=16, sample=50_000, seed=0, cap_div=16):
    N = len(keys); pos = np.arange(N)
    oracle = 1.0                                         # perfect index

    # ---- the non-learned FAMILY, swept; the baseline is the best of it (Instrument step 1) ----
    binary = float(math.ceil(math.log2(N)))
    btree = float(btree_probes(N, fanout))
    rng = np.random.default_rng(seed)
    q_idx = rng.choice(N, size=min(sample, N), replace=False)
    interp = float(interpolation_probes(keys, q_idx).mean())
    base = min(binary, btree, interp)
    winner = {binary: "binary", btree: f"btree(B={fanout})", interp: "interpolation"}[base]

    # ---- the LEARNED model, ALSO swept (Instrument step 1, applied symmetrically) ----
    # Sweeping the non-learned family while pinning the RMI at one capacity would be the same
    # under-tuned-baseline error this paper indicts, aimed the other way. The cap is N//cap_div
    # leaves so the model stays an INDEX rather than becoming a position lookup table: at fanout 16
    # a B-tree already carries ~N/16 internal entries, so an RMI of ~N/16 leaves is the
    # space-comparable opponent, not a free win bought with memory.
    grid = sorted({g for g in ([n_leaves] if n_leaves else []) + [1_000, 10_000, 100_000,
                               max(1, N // cap_div)] if 1 <= g <= max(1, N // cap_div)})
    sweep = []
    for g in grid:
        p_g, lme_g, e_g = rmi(keys, g)
        lo_g = np.floor(p_g - lme_g); hi_g = np.ceil(p_g + lme_g)
        ok_g = bool(np.all((pos >= lo_g) & (pos <= hi_g)))
        w_g = 2.0 * lme_g + 1.0
        sweep.append((float(probes(w_g).mean()), g, ok_g, w_g, e_g))
    learned, best_leaves, ok, window, err = min(sweep, key=lambda t: t[0])

    corridor = base - oracle
    captured = base - learned
    frac = captured / corridor if corridor > 1e-9 else 0.0
    live = ok and frac >= bar_frac and captured > 0

    naive_frac = (binary - learned) / (binary - oracle) if binary > oracle else 0.0

    # ---- SECOND UNIT: distinct cache lines touched (cl keys per 64-byte line) ----
    cl = max(1, 64 // 8)
    binary_L = float(max(1.0, math.ceil(math.log2(max(N / cl, 2)))))   # last log2(cl) levels are free
    btree_L = float(math.ceil(math.log(N, cl)))                        # one node = one line
    interp_L = float(interpolation_lines(keys, q_idx, cl).mean())
    # RMI: root params + leaf params + the last-mile cost, where the implementation is credited
    # with whichever last-mile strategy is cheaper -- a linear scan of the window's lines, or a
    # binary search within it. Charging only the scan would penalise the wide-window cases unfairly.
    wl = np.maximum(window / cl, 1.0)
    learned_L = float(2.0 + np.minimum(np.ceil(wl),
                                       np.maximum(1.0, np.ceil(np.log2(np.maximum(wl, 2.0))))).mean())
    base_L = min(binary_L, btree_L, interp_L)
    winner_L = {binary_L: "binary", btree_L: f"btree(B={cl})", interp_L: "interpolation"}[base_L]
    frac_L = (base_L - learned_L) / (base_L - 1.0) if base_L > 1.0 else 0.0
    live_L = ok and frac_L >= bar_frac and (base_L - learned_L) > 0

    print(f"  [{label}]  N={N:,}  median|err|={np.median(err):.1f}  "
          f"p99|err|={np.percentile(err,99):.0f}  invariant={'OK' if ok else 'VIOLATED'}")
    print(f"     non-learned family:  binary {binary:5.2f} | btree(B={fanout}) {btree:5.2f} | "
          f"interpolation {interp:5.2f}  (measured, n={len(q_idx):,})")
    print(f"     BASELINE = best of family: {base:6.2f} probes   [{winner}]")
    print(f"     RMI leaf sweep (capped at N/{cap_div} = {max(1, N//cap_div):,} leaves):  "
          + " | ".join(f"{g:,}:{p:.2f}p/{window_dist(w, cl)['tail']:.0%}tail"
                       for p, g, _, w, _ in sweep)
          + "   [probes / share of keys whose window spans >64 cache lines]")
    wd = window_dist(window, cl)
    print(f"     WINDOW DIST (tuned model, {cl} keys/line):  <=1 line {wd['one']:5.1%} | "
          f"2-8 {wd['few']:5.1%} | 9-64 {wd['many']:5.1%} | >64 {wd['tail']:5.1%}   "
          f"-> {wd['charged']:.2f} lines charged per lookup")
    print(f"     ORACLE   perfect index    {oracle:6.2f} probes   [reachable: position = N*CDF(key)]")
    print(f"     LEARNED  RMI (keys only)  {learned:6.2f} probes   [deployable, TUNED: "
          f"{best_leaves:,} leaves]")
    print(f"     CORRIDOR {corridor:6.2f} probes    CAPTURED {captured:6.2f} probes "
          f"= {100*frac:5.1f}% of it")
    print(f"     (against binary search alone the capture would read {100*naive_frac:5.1f}% -- "
          f"the under-tuned-baseline number)")
    print(f"     CACHE-LINE unit: binary {binary_L:5.2f} | btree(B={cl}) {btree_L:5.2f} | "
          f"interp {interp_L:5.2f} | RMI {learned_L:5.2f}  -> base {base_L:5.2f} [{winner_L}], "
          f"captured {100*frac_L:6.1f}%")
    v = "BUILD" if live else "NOT captured"
    vL = "BUILD" if live_L else "NOT captured"
    print(f"     VERDICT  probes: {v}   |   cache lines: {vL}"
          + ("   *** UNIT-DEPENDENT ***" if live != live_L else ""))
    print("-" * 100)
    return dict(dist=label, base=base, binary=binary, btree=btree, interp=interp, winner=winner,
                oracle=oracle, learned=learned, corridor=corridor, captured=captured, frac=frac,
                naive_frac=naive_frac, invariant=ok, live=live, best_leaves=best_leaves,
                base_L=base_L, learned_L=learned_L, frac_L=frac_L, live_L=live_L, winner_L=winner_L)


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
    ap.add_argument("--keys-from-sosd", default=None,
                    help="use REAL NATURAL keys: a SOSD dataset file (books/osm/fb/wiki_ts)")
    ap.add_argument("--limit", type=int, default=2_000_000)
    ap.add_argument("--fanout", type=int, default=16,
                    help="B-tree fanout (keys per cache line); 16 = 4-byte keys on a 64-byte line")
    ap.add_argument("--cap-div", type=int, default=16,
                    help="RMI leaf sweep cap = N/cap_div. Keeps the learned model space-comparable "
                         "to the B-tree it races (which carries ~N/fanout internal entries) instead "
                         "of degenerating into a position lookup table.")
    args = ap.parse_args()

    print(LINE)
    print("  THE INSTRUMENT IN A SECOND DOMAIN -- learned indexes (does the ruler ever say BUILD?)")
    print(LINE)
    if args.keys_from_sosd:
        keys = keys_from_sosd(args.keys_from_sosd, args.limit)
        label = args.keys_from_sosd.split("/")[-1] + " (SOSD natural keys)"
        out = [run_keys(label, keys, args.leaves, args.bar_frac, args.fanout, cap_div=args.cap_div)]
    elif args.keys_from_trace:
        keys = keys_from_trace(args.keys_from_trace, args.limit)
        label = args.keys_from_trace.split("/")[-1] + " (real object-ID keys)"
        out = [run_keys(label, keys, args.leaves, args.bar_frac, args.fanout, cap_div=args.cap_div)]
    else:
        out = [run_dist(d, args.n, args.leaves, args.seed, args.bar_frac, args.fanout,
                        cap_div=args.cap_div)
               for d in args.dists.split(",")]

    print("  SUMMARY (the same procedure, different verdicts):")
    for r in out:
        tag = "BUILD" if r["live"] else "weak/no"
        print(f"    {r['dist']:34s} base {r['base']:5.2f} ({r['winner']:14s}) corridor "
              f"{r['corridor']:5.2f}, captured {100*r['frac']:6.1f}%  -> {tag}"
              f"   [vs binary alone: {100*r['naive_frac']:5.1f}%]")
    # The closing read is DERIVED, never asserted. An earlier version of this script printed
    # "the learned index captures it" unconditionally -- a conclusion hardcoded independently of the
    # measurement, which is precisely the failure mode this paper is about. It now reports what the
    # arms actually returned.
    nb = sum(1 for r in out if r["live"])
    print(f"  BUILD verdicts: {nb} of {len(out)} (probes unit).")
    if nb == len(out) and out:
        print("  Reachability reading HOLDS: the caching oracle needs the FUTURE (unreachable) and no\n"
              "  deployable policy captures that corridor; here the oracle needs only the KEYS\n"
              "  (reachable) and the tuned learned index does capture it.")
    elif nb == 0:
        print("  Reachability reading NOT SUPPORTED on these inputs: even with a reachable oracle, the\n"
              "  tuned learned index does not beat the tuned non-learned family. Reachability is then\n"
              "  necessary but NOT sufficient -- a strong non-learned method can already sit on the\n"
              "  reachable ceiling, which is the same bar-saturation mechanism the caching half reports.")
    else:
        print("  MIXED: reachability permits capture on some key distributions and not others; report\n"
              "  per-distribution, and do not state a single cross-domain verdict.")
    print(LINE)


if __name__ == "__main__":
    main()
