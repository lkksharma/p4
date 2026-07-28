#!/usr/bin/env python3
"""
p4_survf7.py -- measure S(100) on the EXACT arm the mechanism section describes.

Why this exists
---------------
The paper's endogenous-slack claim is:

    "the probability that a prefetched object is still cached one hundred requests after
     admission falls from 1.00 under the baseline to 0.055 on Wikipedia and 0.135 on cluster50"

and it is attached to the r6/F7 *model* arm ("a wide causal policy that mistimes ... its precision
falls toward 0.05"). But `p4_r8.py --survcheck` does NOT measure that arm. It instantiates r8's
VolPolicy at pf_byte_rate=None, whose precision is 0.002-0.039 -- a different policy under a
different budget regime. Its verdict is therefore evidence about r8, not about the sentence in the
paper, and cannot settle the claim in either direction.

This script closes that gap: it rebuilds the F7 model arm with exactly the constructor arguments
p4_f7.run() uses, replays it with record_survival=True, and reports S(100) under three estimators
alongside the bar. Whatever it prints is dispositive for the mechanism sentence, because it is the
same policy, same budget, same evictor, same trace prefix.

    python p4_survf7.py --trace data/wiki_2019t.oracleGeneral --pred markov2 --tau 0.05 --k 1 \
        --haz haz/wiki.npz --wide-k 32 --gamma 0.1
    python p4_survf7.py --trace data/cluster50.sample10.oracleGeneral --pred markov3 --tau 0.06 \
        --k 1 --haz haz/cluster50.npz --wide-k 16 --gamma 0.1
"""
from __future__ import annotations

import argparse

import numpy as np

from p4_coldsplit import PREDS
from p4_f7 import CausalTimedCover
from p4_prefetch import PFCache
from p4_sweep import prep

LINE = "=" * 100


def s_at(surv, d=100):
    """Three estimators of S(d) on a prefetch-lifetime log of (delta, was_used) pairs.

      km            Kaplan-Meier, uses censored (used before eviction) as censoring
      naive         fraction of ALL admissions whose observed lifetime exceeded d
      evicted_only  fraction of EVICTIONS that happened after d (ignores used objects entirely)

    Three views because they fail differently: km is the principled one, naive under-counts when
    most objects are used early, evicted_only is the pessimistic bound. If the window really
    collapses, all three fall together.
    """
    if not surv:
        return float("nan"), float("nan"), float("nan"), 0, 0.0, 0
    a = np.asarray(surv, dtype=np.int64)
    delta, used = a[:, 0], a[:, 1]
    n = len(delta)
    naive = float((delta > d).mean())
    ev = delta[used == 0]
    evo = float((ev > d).mean()) if len(ev) else float("nan")
    order = np.argsort(delta, kind="stable")
    dd, cc = delta[order], used[order]
    at_risk, S, km_at_d = n, 1.0, 1.0
    for t in np.unique(dd):
        if t > d:
            break                                  # past the horizon; S(d) is already fixed
        m = dd == t
        tot = int(m.sum()); deaths = int((cc[m] == 0).sum())
        if at_risk > 0 and deaths > 0:
            S *= 1.0 - deaths / at_risk            # at-risk count is taken JUST BEFORE t
        km_at_d = S
        at_risk -= tot
    return float(km_at_d), naive, evo, n, float(used.mean()), int(np.median(delta))


def main():
    ap = argparse.ArgumentParser(description="S(100) on the F7 model arm (the mechanism's own arm)")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--limit", type=int, default=2_000_000)
    ap.add_argument("--cache-frac", type=float, default=0.01)
    ap.add_argument("--pred", choices=sorted(PREDS), required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--haz", required=True)
    ap.add_argument("--wide-k", type=int, default=32)
    ap.add_argument("--wide-tau", type=float, default=0.0)
    ap.add_argument("--wide-top-m", type=int, default=32)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--lead", type=int, default=1)
    ap.add_argument("--max-wait", type=int, default=200_000)
    ap.add_argument("--max-pool", type=int, default=4096)
    ap.add_argument("--at", type=int, default=100)
    args = ap.parse_args()

    trace, cap, pos, szs = prep(args.trace, args.limit, args.cache_frac)
    haz = np.load(args.haz)
    tf, n = args.train_frac, trace["n"]

    def mk():
        return PREDS[args.pred](trace, train_frac=tf, window=args.window, k=args.k, tau=args.tau)

    def mk_wide():
        p = PREDS[args.pred](trace, train_frac=tf, window=args.window, k=args.k, tau=args.tau,
                             top_m=args.wide_top_m)
        p.k, p.tau = args.wide_k, args.wide_tau
        return p

    print(f"\n{LINE}\n  S({args.at}) ON THE F7 MODEL ARM -- {args.trace}\n{LINE}")

    bar = PFCache(cap, "s3fifo", mk(), positions=pos, sizes=szs).run(
        trace, cold_train_frac=tf, record_survival=True)
    rate = bar["prefetch_bytes"] / max(n, 1)
    budget = bar["prefetch_bytes"]

    # identical construction to p4_f7.run()'s model arm
    sch = CausalTimedCover(mk_wide(), pos, szs, haz, rate, budget, timing="model",
                           gamma=args.gamma, lead=args.lead, max_wait=args.max_wait,
                           maxpool=args.max_pool, verbose=False, n=n)
    f7 = PFCache(cap, "s3fifo", sch, positions=pos, sizes=szs, pf_byte_rate=None).run(
        trace, cold_train_frac=tf, record_survival=True)

    hdr = f"  {'arm':<22}{'KM':>8}{'naive':>8}{'evicted':>9}{'n':>12}{'used%':>8}{'med age':>9}{'prec':>7}"
    print(hdr)
    for name, r in (("BAR (tuned)", bar), ("F7 model arm", f7)):
        km, nv, ev, cnt, um, med = s_at(r["pf_survival"], args.at)
        print(f"  {name:<22}{km:>8.3f}{nv:>8.3f}{ev:>9.3f}{cnt:>12,}{100*um:>7.1f}%"
              f"{med:>9,}{r['pf_precision']:>7.3f}")

    kb = s_at(bar["pf_survival"], args.at)[0]
    kf = s_at(f7["pf_survival"], args.at)[0]
    print(f"  {'-'*94}")
    print(f"  paper claims  S({args.at}): 1.00 -> 0.055 (wiki) / 0.135 (cluster50)")
    print(f"  measured here S({args.at}): {kb:.3f} -> {kf:.3f}   (ratio {kf/max(kb,1e-9):.2f}x)")
    print(f"  fetch volume: bar {bar['pf_issued']:,} -> F7 model {f7['pf_issued']:,} "
          f"({f7['pf_issued']/max(bar['pf_issued'],1):.2f}x)")
    if kf <= 0.20:
        v = ("MECHANISM SUPPORTED -- the window does collapse on this arm; the paper's sentence "
             "stands and the earlier r8 survcheck was simply measuring a different policy.")
    elif kf >= 0.80:
        v = ("MECHANISM REFUTED ON ITS OWN ARM -- the residence window stays open, so the failure "
             "cannot be attributed to a policy evicting its own future targets. Restate as "
             "precision collapse under a fixed budget.")
    else:
        v = ("PARTIAL -- the window narrows but not to the claimed value; quote the measured "
             "number, not 0.055/0.135.")
    print(f"  VERDICT  {v}\n{LINE}\n")


if __name__ == "__main__":
    main()
