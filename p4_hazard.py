#!/usr/bin/env python3
"""
p4_hazard.py -- Necessity Ladder rung (ii), part A: the CAUSAL hazard model + eviction-survival
curve that the HJS-L scheduler (p4_hjsl.py) consumes, plus the F2 learnability gate.

Unlike F1/F4 (clairvoyant oracles that read oracle_next), everything here is causal: the hazard
model is trained on the TRAINING prefix only and predicts a candidate's lag-to-next-use
distribution p(tau | features) from features known at emission time (predictor confidence, object
size). It never sees the future at inference. This is what makes HJS-L a deployable policy rather
than a bound.

Two artefacts, saved to an .npz the scheduler loads:
  1. HAZARD MODEL  p(lag_bin | conf_bin, size_bin), a binned empirical estimator (the interpretable
     GBT-analog; an NTPP is the upgrade, out of scope for the first working scheduler). Lag binned
     log-uniformly into 16 bins over [1, 1e6] plus a "never" bin.
  2. SURVIVAL CURVE  S(delta) = P(a prefetched-not-yet-used object is still cached after delta
     requests), a Kaplan-Meier estimate (used prefetches are right-censored) from an instrumented
     bar replay -- a measured property of S3-FIFO under prefetch load, not an exponential assumption.

F2 GATE (pre-registered, on the held-out EVAL half):
  - Spearman(predicted median lag, true lag) >= 0.20  AND
  - never-bin AUC (P(never) separating used from never-used) >= 0.60
  Below either on a trace -> the *when* is unlearnable there; HJS-L cannot win regardless of the
  scheduler, and that trace drops out of the capture claim. Report it.

    python p4_hazard.py --trace data/wiki_2019t.oracleGeneral --pred markov2 --tau 0.05 --k 1 \
        --out haz/wiki.npz
    python p4_hazard.py --selftest        # runs on a synthetic trace, no data needed
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from p4_cache import NEVER
from p4_coldsplit import PREDS
from p4_f4 import capture_ctx, conf_lookup                # reuse the exact confidence lookup
from p4_prefetch import PFCache, oracle_next
from p4_sweep import prep

# ---- fixed binning (shared with p4_hjsl.py via the saved npz) --------------------------------
LAG_EDGES = np.logspace(0.0, 6.0, 17)      # 17 edges -> 16 finite log bins over [1, 1e6]
NEVER_BIN = 16
N_LAG = 17
CONF_EDGES = np.array([0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.01])   # 7 conf bins
SIZE_EDGES = np.logspace(1.0, 8.0, 8)      # 7 size bins over [10 B, 1e8 B]
N_CONF, N_SIZE = 7, 7


def lag_bin(lag: int) -> int:
    if lag < 0:
        return NEVER_BIN
    b = int(np.searchsorted(LAG_EDGES, lag, side="right")) - 1
    return min(max(b, 0), 15)


def conf_bin(c: float) -> int:
    return min(max(int(np.searchsorted(CONF_EDGES, c, side="right")) - 1, 0), N_CONF - 1)


def size_bin(s: int) -> int:
    return min(max(int(np.searchsorted(SIZE_EDGES, s, side="right")) - 1, 0), N_SIZE - 1)


def harvest(pred, trace, lo, hi, szs, pos):
    """(conf, size, lag) for every predictor emission with i in [lo, hi). lag = -1 for 'never'.

    Replays the predictor over the WHOLE trace with an empty cache (so state advances correctly and
    the candidate set is the predictor's own top-k), recording only emissions in the window. Same
    convention as p4_f1.emit_vocab, so the hazard model sees exactly the candidates HJS-L will."""
    ids = trace["obj_id"]
    empty = frozenset()
    pred.reset()
    conf_a, size_a, lag_a = [], [], []
    for i in range(trace["n"]):
        o = int(ids[i])
        ctx = capture_ctx(pred)                         # BEFORE suggest mutates predictor state
        objs = pred.suggest(o, empty, i)
        if not objs or not (lo <= i < hi):
            continue
        cmap = conf_lookup(pred, ctx, o)
        tau = getattr(pred, "tau", 0.05)
        for x in objs:
            x = int(x)
            s = szs.get(x)
            if s is None:
                continue
            nx = oracle_next(pos, x, i)
            conf_a.append(cmap.get(x, tau))
            size_a.append(s)
            lag_a.append(int(nx) - i if nx < int(NEVER) else -1)
    return np.array(conf_a), np.array(size_a, dtype=np.int64), np.array(lag_a, dtype=np.int64)


def build_hazard(conf, size, lag):
    """p(lag_bin | conf_bin, size_bin) as a (N_CONF, N_SIZE, N_LAG) normalised histogram.
    Empty cells fall back to the global lag distribution so the scheduler never divides by zero."""
    hist = np.zeros((N_CONF, N_SIZE, N_LAG), dtype=np.float64)
    for c, s, lg in zip(conf, size, lag):
        hist[conf_bin(c), size_bin(s), lag_bin(int(lg))] += 1.0
    glob = hist.sum(axis=(0, 1))
    glob = glob / max(glob.sum(), 1.0)
    for a in range(N_CONF):
        for b in range(N_SIZE):
            tot = hist[a, b].sum()
            hist[a, b] = hist[a, b] / tot if tot > 0 else glob
    return hist


def survival_km(deltas, censored, grid):
    """Kaplan-Meier S(delta) on prefetch lifetimes. censored=1 (object was USED) -> right-censored;
    censored=0 (object was EVICTED unused) -> a death event. Returns S evaluated on `grid`."""
    if len(deltas) == 0:
        return np.ones_like(grid, dtype=np.float64)
    order = np.argsort(deltas, kind="stable")
    d, cen = deltas[order], censored[order]
    uniq = np.unique(d)
    at_risk = len(d)
    S = 1.0
    times, vals = [], []
    for t in uniq:
        m = d == t
        total = int(m.sum())
        deaths = int((cen[m] == 0).sum())
        if at_risk > 0 and deaths > 0:
            S *= 1.0 - deaths / at_risk
        times.append(t)
        vals.append(S)
        at_risk -= total
    times, vals = np.asarray(times), np.asarray(vals)
    idx = np.searchsorted(times, grid, side="right") - 1
    return np.where(idx >= 0, vals[np.clip(idx, 0, len(vals) - 1)], 1.0)


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def auc(scores, labels):
    """Mann-Whitney AUC of `scores` predicting binary `labels` (1 = positive)."""
    labels = np.asarray(labels)
    npos, nneg = int(labels.sum()), int((labels == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(scores)) + 1
    return float((ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def median_lag_pred(hist_row):
    """Median lag (bin midpoint) of a predicted lag distribution, ignoring the 'never' mass."""
    finite = hist_row[:16].copy()
    tot = finite.sum()
    if tot <= 0:
        return LAG_EDGES[-2]
    cdf = np.cumsum(finite / tot)
    b = int(np.searchsorted(cdf, 0.5))
    b = min(b, 15)
    return float(np.sqrt(LAG_EDGES[b] * LAG_EDGES[b + 1]))       # geometric bin midpoint


def run(trace, cap, pos, szs, args):
    tf = args.train_frac
    n = trace["n"]
    cut = int(n * tf)

    def mk():
        return PREDS[args.pred](trace, train_frac=tf, window=args.window, k=args.k, tau=args.tau)

    # ---- hazard model: TRAIN on the prefix only, then F2-evaluate on the held-out half ----
    c_tr, s_tr, l_tr = harvest(mk(), trace, 0, cut, szs, pos)
    hist = build_hazard(c_tr, s_tr, l_tr)

    c_ev, s_ev, l_ev = harvest(mk(), trace, cut, n, szs, pos)
    used = l_ev >= 0
    pred_med = np.array([median_lag_pred(hist[conf_bin(c), size_bin(s)])
                         for c, s in zip(c_ev, s_ev)])
    rho = spearman(pred_med[used], l_ev[used].astype(np.float64))
    p_never = np.array([hist[conf_bin(c), size_bin(s), NEVER_BIN] for c, s in zip(c_ev, s_ev)])
    never_auc = auc(p_never, (~used).astype(int))

    # ---- survival curve S(delta): instrumented bar replay over the full trace ----
    barrun = PFCache(cap, "s3fifo", mk(), positions=pos, sizes=szs).run(
        trace, cold_train_frac=tf, record_survival=True)
    surv = barrun["pf_survival"]
    if surv:
        d = np.array([x[0] for x in surv], dtype=np.int64)
        cen = np.array([x[1] for x in surv], dtype=np.int64)     # 1 = used (censored), 0 = evicted
    else:
        d = np.array([], dtype=np.int64); cen = np.array([], dtype=np.int64)
    surv_grid = np.unique(np.round(np.logspace(0, 6, 60)).astype(np.int64))
    surv_vals = survival_km(d, cen, surv_grid)

    f2_pass = (not np.isnan(rho) and rho >= 0.20) and (not np.isnan(never_auc) and never_auc >= 0.60)

    print(f"\n  {args.trace}   [hazard: {args.pred} tau={args.tau} k={args.k}]")
    print(f"  train emissions {len(l_tr):,}   eval emissions {len(l_ev):,}   "
          f"({used.sum():,} used, {(~used).sum():,} never)")
    print(f"  survival events {len(d):,}   median lifetime "
          f"{int(np.median(d)) if len(d) else 0:,} req   S(100)={survival_km(d,cen,np.array([100]))[0]:.3f}")
    print(f"\n  F2  Spearman(pred median lag, true lag)  {rho:+.3f}   [gate >= 0.20]")
    print(f"  F2  never-bin AUC (P(never) vs used)     {never_auc:.3f}   [gate >= 0.60]")
    print(f"  F2 GATE  {'PASS -- when is learnable; build the scheduler' if f2_pass else 'FAIL -- when is unlearnable here; HJS-L cannot win, drop this trace'}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.savez_compressed(
            args.out, hist=hist, lag_edges=LAG_EDGES, conf_edges=CONF_EDGES, size_edges=SIZE_EDGES,
            surv_grid=surv_grid, surv_vals=surv_vals,
            rho=rho, never_auc=never_auc, f2_pass=f2_pass,
            pred=args.pred, tau=args.tau, k=args.k, train_frac=tf)
        print(f"  saved -> {args.out}")
    print()
    return f2_pass


def selftest():
    """Runs the whole pipeline on a synthetic trace so crashes surface without box data."""
    class A:
        trace = "SYNTH"; limit = 60000; cache_frac = 0.01
        pred = "markov1"; tau = 0.05; k = 1; window = 16; train_frac = 0.5; out = None
    trace, cap, pos, szs = prep("SYNTH", A.limit, A.cache_frac)
    run(trace, cap, pos, szs, A())
    print("  [p4_hazard selftest] completed without error\n")


def main():
    ap = argparse.ArgumentParser(description="HJS-L hazard model + survival curve + F2 gate")
    ap.add_argument("--trace")
    ap.add_argument("--limit", type=int, default=2_000_000)
    ap.add_argument("--cache-frac", type=float, default=0.01)
    ap.add_argument("--pred", choices=sorted(PREDS))
    ap.add_argument("--tau", type=float)
    ap.add_argument("--k", type=int)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="save the hazard+survival model to this .npz")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if not (args.trace and args.pred and args.tau is not None and args.k is not None):
        ap.error("need --trace --pred --tau --k (or --selftest)")
    trace, cap, pos, szs = prep(args.trace, args.limit, args.cache_frac)
    run(trace, cap, pos, szs, args)


if __name__ == "__main__":
    main()
