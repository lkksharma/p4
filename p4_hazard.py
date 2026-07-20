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
# RECENCY -- time since the candidate object was last requested -- is the primary temporal
# feature: temporal locality means a recently-seen object tends to be seen again soon. This is
# THE predictor of use-lag; conf+size alone cannot separate lags and make F2 a false-negative.
REC_EDGES = np.logspace(0.0, 6.0, 17)      # 16 finite recency bins over [1, 1e6]
REC_COLD = 16                              # object not yet seen in the replay -> "cold" bin
# FREQUENCY -- how many times the candidate has been requested so far -- is the second temporal
# feature the proposal named (§2.1). Added as diligence: a decisive F2 fail WITH recency+frequency
# both present is a genuine "the when is unlearnable" finding, not an under-featured artifact.
FREQ_EDGES = np.array([1, 2, 4, 8, 16, 64, 256])       # bin 0 = unseen; 7 freq bins
N_CONF, N_SIZE, N_REC, N_FREQ = 7, 7, 17, 7


def lag_bin(lag: int) -> int:
    if lag < 0:
        return NEVER_BIN
    b = int(np.searchsorted(LAG_EDGES, lag, side="right")) - 1
    return min(max(b, 0), 15)


def conf_bin(c: float) -> int:
    return min(max(int(np.searchsorted(CONF_EDGES, c, side="right")) - 1, 0), N_CONF - 1)


def size_bin(s: int) -> int:
    return min(max(int(np.searchsorted(SIZE_EDGES, s, side="right")) - 1, 0), N_SIZE - 1)


def recency_bin(r: int) -> int:
    if r < 0:
        return REC_COLD                    # never seen yet in the replay
    b = int(np.searchsorted(REC_EDGES, r, side="right")) - 1
    return min(max(b, 0), 15)


def freq_bin(f: int) -> int:
    return min(int(np.searchsorted(FREQ_EDGES, f, side="right")), N_FREQ - 1)


def harvest(pred, trace, lo, hi, szs, pos, horizon):
    """(recency, freq, conf, size, lag) for every predictor emission with i in [lo, hi). lag = -1
    for 'never', recency = -1 for 'not yet seen'. Recency = i - (last request index of the
    candidate), tracked causally over the whole replay. Replays with an empty cache (state advances
    correctly; candidate set = the predictor's own top-k), recording only emissions in the window.

    FIXED OBSERVATION HORIZON. 'never' means "not used within `horizon` requests", NOT "never again
    in the rest of the trace". The latter is not a well-defined label: an emission at i=1.99M is
    called 'never' merely because the trace ends, while one at i=0.5M has 1.5M requests in which to
    be vindicated -- so the label's meaning drifts with position, and the train and eval windows end
    up measuring different things. A fixed window makes every emission's label identical in meaning,
    and emissions with fewer than `horizon` requests remaining are DROPPED as unobservable rather
    than silently mislabelled. It also matches the decision the scheduler actually faces: a use one
    million requests away is unreachable by prefetching regardless."""
    ids = trace["obj_id"]
    n = trace["n"]
    empty = frozenset()
    pred.reset()
    last_seen, freq = {}, {}
    rec_a, frq_a, conf_a, size_a, lag_a = [], [], [], [], []
    for i in range(n):
        o = int(ids[i])
        ctx = capture_ctx(pred)                         # BEFORE suggest mutates predictor state
        objs = pred.suggest(o, empty, i)
        if objs and lo <= i < hi and i + horizon <= n:  # else: not enough future to observe
            cmap = conf_lookup(pred, ctx, o)
            tau = getattr(pred, "tau", 0.05)
            for x in objs:
                x = int(x)
                s = szs.get(x)
                if s is None:
                    continue
                nx = oracle_next(pos, x, i)
                lag = int(nx) - i if nx < int(NEVER) else -1
                if lag > horizon:
                    lag = -1                            # beyond the window == never, by definition
                rec_a.append(i - last_seen[x] if x in last_seen else -1)
                frq_a.append(freq.get(x, 0))
                conf_a.append(cmap.get(x, tau))
                size_a.append(s)
                lag_a.append(lag)
        last_seen[o] = i                                # causal: o is now seen at i
        freq[o] = freq.get(o, 0) + 1
    return (np.array(rec_a, dtype=np.int64), np.array(frq_a, dtype=np.int64), np.array(conf_a),
            np.array(size_a, dtype=np.int64), np.array(lag_a, dtype=np.int64))


def build_hazard(rec, frq, conf, lag):
    """p(lag_bin | recency_bin, freq_bin, conf_bin) as (N_REC, N_FREQ, N_CONF, N_LAG), normalised.
    Empty cells fall back to the global lag distribution so the scheduler never divides by zero."""
    hist = np.zeros((N_REC, N_FREQ, N_CONF, N_LAG), dtype=np.float64)
    for r, f, c, lg in zip(rec, frq, conf, lag):
        hist[recency_bin(int(r)), freq_bin(int(f)), conf_bin(c), lag_bin(int(lg))] += 1.0
    glob = hist.sum(axis=(0, 1, 2))
    glob = glob / max(glob.sum(), 1.0)
    for a in range(N_REC):
        for b in range(N_FREQ):
            for d in range(N_CONF):
                tot = hist[a, b, d].sum()
                hist[a, b, d] = hist[a, b, d] / tot if tot > 0 else glob
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
    cut = int(n * tf)                                  # predictor's own training boundary
    hcut = int(n * args.haz_frac)                      # hazard-train / F2-eval boundary
    H = args.horizon

    def mk():
        return PREDS[args.pred](trace, train_frac=tf, window=args.window, k=args.k, tau=args.tau,
                                top_m=getattr(args, "top_m", 16))

    # ---- THREE-WAY SPLIT (this is load-bearing, not hygiene) --------------------------------
    # [0, cut)     the PREDICTOR trains here
    # [cut, hcut)  the HAZARD MODEL trains here -- on the predictor's OUT-OF-SAMPLE behaviour
    # [hcut, n)    F2 evaluates here
    #
    # Harvesting hazard-training data from [0, cut) -- the predictor's own training window -- is a
    # design error that inverts the model. In-sample, a high-confidence table row means "memorised,
    # and reliably right". Out-of-sample, a high-confidence row is often one built from very few
    # observations, i.e. overfit and MORE likely wrong. So the confidence->outcome relationship
    # flips sign between the windows, and a model fit on the first is anti-predictive on the second
    # (never-AUC lands BELOW 0.5, the signature of exactly this bug rather than of a weak model).
    # The hazard model must see the predictor as it will actually behave at deployment: unseen data.
    import sys
    print(f"  [hazard] harvesting train window [{cut:,},{hcut:,}) -- full-trace replay, silent, "
          f"~10-25 min at wide k...", file=sys.stderr, flush=True)
    r_tr, f_tr, c_tr, s_tr, l_tr = harvest(mk(), trace, cut, hcut, szs, pos, H)
    hist = build_hazard(r_tr, f_tr, c_tr, l_tr)

    print(f"  [hazard] harvesting eval window [{hcut:,},{n:,})...", file=sys.stderr, flush=True)
    r_ev, f_ev, c_ev, s_ev, l_ev = harvest(mk(), trace, hcut, n, szs, pos, H)
    used = l_ev >= 0
    pred_med = np.array([median_lag_pred(hist[recency_bin(int(r)), freq_bin(int(f)), conf_bin(c)])
                         for r, f, c in zip(r_ev, f_ev, c_ev)])
    rho = spearman(pred_med[used], l_ev[used].astype(np.float64))
    p_never = np.array([hist[recency_bin(int(r)), freq_bin(int(f)), conf_bin(c), NEVER_BIN]
                        for r, f, c in zip(r_ev, f_ev, c_ev)])
    never_auc = auc(p_never, (~used).astype(int))

    # ---- survival curve S(delta): instrumented bar replay over the full trace ----
    print("  [hazard] survival replay...", file=sys.stderr, flush=True)
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

    tr_never = float((l_tr < 0).mean()) if len(l_tr) else float("nan")
    ev_never = float((~used).mean()) if len(l_ev) else float("nan")
    print(f"\n  {args.trace}   [hazard: {args.pred} tau={args.tau} k={args.k}]")
    print(f"  split  predictor [0,{cut:,})  hazard-train [{cut:,},{hcut:,})  "
          f"F2-eval [{hcut:,},{n:,})   horizon {H:,}")
    print(f"  hazard-train emissions {len(l_tr):,} (never {tr_never:.1%})   "
          f"F2-eval emissions {len(l_ev):,} (never {ev_never:.1%})")
    # The two never-rates should now be COMPARABLE. A large gap means the windows still see
    # different regimes and the model is being asked to extrapolate rather than predict.
    if not (np.isnan(tr_never) or np.isnan(ev_never)) and abs(tr_never - ev_never) > 0.15:
        print(f"  !! never-rate gap {abs(tr_never-ev_never):.1%} between windows -- the hazard "
              f"model is extrapolating; treat F2 as unreliable")
    print(f"  survival events {len(d):,}   median lifetime "
          f"{int(np.median(d)) if len(d) else 0:,} req   S(100)={survival_km(d,cen,np.array([100]))[0]:.3f}")
    print(f"\n  F2  Spearman(pred median lag, true lag)  {rho:+.3f}   [gate >= 0.20]")
    print(f"  F2  never-bin AUC (P(never) vs used)     {never_auc:.3f}   [gate >= 0.60]")
    print(f"  F2 GATE  {'PASS -- when is learnable; build the scheduler' if f2_pass else 'FAIL -- when is unlearnable here; HJS-L cannot win, drop this trace'}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.savez_compressed(
            args.out, hist=hist, lag_edges=LAG_EDGES, conf_edges=CONF_EDGES, size_edges=SIZE_EDGES,
            rec_edges=REC_EDGES, freq_edges=FREQ_EDGES, surv_grid=surv_grid, surv_vals=surv_vals,
            rho=rho, never_auc=never_auc, f2_pass=f2_pass,
            pred=args.pred, tau=args.tau, k=args.k, train_frac=tf,
            haz_frac=args.haz_frac, horizon=H)
        print(f"  saved -> {args.out}")
    print()
    return f2_pass


def selftest():
    """Runs the whole pipeline on a synthetic trace so crashes surface without box data."""
    class A:
        trace = "SYNTH"; limit = 60000; cache_frac = 0.01
        pred = "markov1"; tau = 0.05; k = 1; window = 16; train_frac = 0.5; out = None
        haz_frac = 0.75; horizon = 5000
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
    ap.add_argument("--top-m", type=int, default=16,
                    help="successors stored per context in the predictor. Default 16 = the narrow "
                         "bar config. Set the WIDE config (e.g. --k 32 --tau 0.0 --top-m 32) to "
                         "train the hazard model on the wide emission stream -- the pre-registered "
                         "F7 fair-shot fallback. The survival/F2 outputs then also describe the "
                         "wide stream, which is the point.")
    ap.add_argument("--train-frac", type=float, default=0.5,
                    help="predictor training fraction -- must match the bar config")
    ap.add_argument("--haz-frac", type=float, default=0.75,
                    help="hazard model trains on [train_frac, haz_frac); F2 evaluates on "
                         "[haz_frac, 1.0). MUST exceed --train-frac or the hazard model is fit on "
                         "the predictor's own training data and inverts (never-AUC < 0.5).")
    ap.add_argument("--horizon", type=int, default=100_000,
                    help="observation window: 'never' means not used within this many requests. "
                         "Emissions with less remaining trace are dropped as unobservable.")
    ap.add_argument("--out", default=None, help="save the hazard+survival model to this .npz")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if not (args.trace and args.pred and args.tau is not None and args.k is not None):
        ap.error("need --trace --pred --tau --k (or --selftest)")
    if args.haz_frac <= args.train_frac:
        ap.error(f"--haz-frac ({args.haz_frac}) must exceed --train-frac ({args.train_frac}): the "
                 "hazard model must be fit on the predictor's OUT-OF-SAMPLE behaviour")
    trace, cap, pos, szs = prep(args.trace, args.limit, args.cache_frac)
    run(trace, cap, pos, szs, args)


if __name__ == "__main__":
    main()
