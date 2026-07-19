#!/usr/bin/env python3
"""p4_f5.py -- Necessity Ladder rung (i.5): the F5 COVERAGE-CEILING gate.

WHY THIS EXISTS. F1 came in two flavours and they disagree, and the disagreement is the whole
finding of the capture track so far:
    F1-vocab  (any emit-vocab object, clairvoyant timing)         wiki +12.72   c50 +17.56
    F1-stream (the k=1 emission stream, clairvoyant timing)       wiki  -7.01   c50 -10.82  (<= bar!)
So perfect *timing* of what the predictor emits at k=1 cannot beat the bar -- the corridor is NOT
a fetch-timing gap. It is a COVERAGE / object-availability gap: at k=1 the predictor names each
object too rarely to have a live candidate before most of its uses. F5 asks the one question that
decides whether a positive scheduling policy can exist at all:

    How much of the corridor becomes CAUSALLY REACHABLE if the predictor emits WIDE (fanout k)
    instead of k=1, given perfect selection + timing under the bar's exact byte budget?

HONEST CEILING (an upper bound on ANY causal wide-emit + selection policy, by construction):
  * a USE of object X at request t is CAUSALLY COVERABLE iff the wide predictor emitted X as a
    candidate at some request in (prev_use(X), t) -- i.e. a causal wide-emit policy had a FRESH
    signal that X was coming since it last saw X, so it COULD have prefetched X in time.
  * F5 = a clairvoyant JIT prefetcher (perfect timing, perfect selection, soonest-first) RESTRICTED
    to cover only causally-coverable uses, at the bar's byte rate (token bucket -> iso-bandwidth).
  * By construction  F1-stream <= F5 <= F1-vocab , and F5's cold hits are 0 (coverability requires
    a prefix-trained emission, so only in-training-vocab objects are ever fundable).
  * The sweep over fanout k traces the curve from timing-only (k=1, ~F1-stream) toward full
    coverage (k=max, -> F1-vocab). The widest point is a genuine UPPER BOUND: if even it stalls at
    the bar, no causal wide-emit policy can beat the bar, and Policy 1 is dead before it is built.

PRE-REGISTERED FORK (checklist point 1, fixed BEFORE this run):
  F5 >= bar + 8 (CI lo > 0) on >= 2 live traces -> the coverage lever EXISTS -> build Policy 1
                                                   (the wide-emission lambda-price bandwidth market).
  F5 ~= bar                                      -> width does not help with this predictor ->
                                                   Policy 1 is blocked; the bottleneck is the
                                                   FORECASTER (go to Policy 2), or -- if an oracle
                                                   forecaster also stalls -- the corridor is not
                                                   causally reachable and the honest paper is the
                                                   negative.

    python p4_f5.py --trace data/wiki_2019t.oracleGeneral        --pred markov2 --tau 0.05 --k 1
    python p4_f5.py --trace data/cluster50.sample10.oracleGeneral --pred markov3 --tau 0.06 --k 1
    python p4_f5.py --selftest
"""
from __future__ import annotations

import argparse

import numpy as np

from p4_coldsplit import PREDS
from p4_prefetch import PFCache, Prescient
from p4_sweep import KS, prep


def block_ci(diff, blocks, resamples, seed):
    """Paired moving-block bootstrap -- same convention as p4_f1/p4_hjsl."""
    R = len(diff)
    L = R // blocks
    bm = diff[:blocks * L].reshape(blocks, L).mean(axis=1)
    rng = np.random.default_rng(seed)
    boot = 100.0 * bm[rng.integers(0, blocks, size=(resamples, blocks))].mean(axis=1)
    return np.percentile(boot, [2.5, 97.5])


def build_coverable(pred, trace, k_wide, wide_tau):
    """One causal pass: for every request t, is the object at t CAUSALLY COVERABLE?

    coverable[t] = True iff the wide predictor emitted objid[t] as a candidate at some request in
    (prev_use(objid[t]), t). Emissions are harvested against an ALWAYS-EMPTY cache (like
    emit_vocab), so caching never removes a candidate -- the maximal, still-honest causal candidate
    set. Ordering matters: for the use at t we read last_emit / prev_use as they stand BEFORE t's
    own emission and BEFORE recording t as a use, so the signal is strictly in t's past."""
    ids = trace["obj_id"]
    n = trace["n"]
    pred.k = k_wide                              # widen the fanout (attributes, not set_params: works
    pred.tau = wide_tau                          # for every _MarkovBase predictor uniformly)
    r = getattr(pred, "reset", None)
    if r:
        r()
    EMPTY: frozenset = frozenset()
    coverable = np.zeros(n, dtype=bool)
    last_emit: dict = {}                         # obj -> last request index it was emitted at
    prev_use: dict = {}                          # obj -> last request index it was requested at
    n_cov = 0
    for i in range(n):
        x = int(ids[i])
        if last_emit.get(x, -1) > prev_use.get(x, -1):   # emitted since its last use, in the past
            coverable[i] = True
            n_cov += 1
        prev_use[x] = i
        for c in pred.suggest(x, EMPTY, i):     # harvest this request's wide emissions
            last_emit[int(c)] = i
    return coverable, n_cov


class CoverGatedPrescient:
    """F5 arm: the clairvoyant JIT prefetcher, but allowed to cover ONLY causally-coverable uses.

    Identical to Prescient (soonest-to-be-used, not in cache, JIT insertion) except a candidate X
    is admitted only if X's imminent use is coverable[j]. A prefetch issued now would mechanically
    hit X's SOONEST future use, so the gate keys on that soonest use: if it is not coverable, X is
    not prefetched now (it may still be prefetched at a later request whose soonest use of X IS
    coverable). This makes F5 an upper bound on any causal wide-emit policy, never below it."""
    name = "f5_cover"

    def __init__(self, trace, coverable, k, lookahead=2000):
        self.ids = trace["obj_id"]
        self.cov = coverable
        self.k = k
        self.la = lookahead
        self.n = trace["n"]

    def reset(self):
        pass

    def suggest(self, o, cached, i):
        out, seen = [], set()
        end = min(i + 1 + self.la, self.n)
        for j in range(i + 1, end):
            x = int(self.ids[j])
            if x in cached or x in seen:
                continue
            seen.add(x)                          # decide X once, on its SOONEST future use
            if not self.cov[j]:
                continue                         # imminent use not causally coverable -> do not fund
            out.append(x)
            if len(out) >= self.k:
                break
        return out


def run(trace, cap, pos, szs, args):
    tf = args.train_frac
    n = trace["n"]

    def mk():
        return PREDS[args.pred](trace, train_frac=tf, window=args.window, k=args.k, tau=args.tau)

    base = PFCache(cap, "s3fifo", None, positions=pos, sizes=szs).run(trace)
    bar = PFCache(cap, "s3fifo", mk(), positions=pos, sizes=szs).run(
        trace, cold_train_frac=tf, return_hits=True)
    rate = bar["prefetch_bytes"] / max(n, 1)
    bar_tx = bar["origin_bytes"] / max(base["origin_bytes"], 1)

    # WARM CEILING -- the paper's LEARNABLE corridor and the true clairvoyant upper bound: Prescient
    # over the full training-prefix vocab, at the bar's byte rate. F5 <= warm BY CONSTRUCTION (F5
    # covers a causally-gated SUBSET of the uses warm can cover, at the same rate). Its corridor is
    # the honest denominator; (F1-vocab over the k=1 emit set is NOT -- F5's wide net can name
    # objects outside it, so F5 can exceed it without exceeding the true ceiling).
    cut = int(n * tf)
    tvocab = set(int(x) for x in trace["obj_id"][:cut])
    warm = PFCache(cap, "s3fifo", Prescient(trace, k=max(KS), vocab=tvocab), positions=pos,
                   sizes=szs, pf_byte_rate=rate).run(trace, cold_train_frac=tf)
    warm_corr = 100.0 * (warm["ohr"] - bar["ohr"])

    print(f"\n  {args.trace}   [F5 coverage ceiling | wide_tau={args.wide_tau} "
          f"budget={args.budget_mode}]")
    print(f"  BAR        {args.pred} tau={args.tau} k={args.k}  OHR {bar['ohr']:.4f} @{bar_tx:.2f}x   "
          f"pf {bar['pf_issued']:,} useful {bar['pf_useful']:,} (prec {bar['pf_precision']:.3f})")
    print(f"  WARM CEIL  OHR {warm['ohr']:.4f}   corridor {warm_corr:+.2f} pts   "
          f"(learnable corridor = denominator)")
    print(f"  {'':11s}(reference: F1-stream was measured <= bar -- timing on the k=1 stream cannot win)")
    print(f"  {'-'*74}")

    best = None
    for kw in args.wide_k:
        coverable, n_cov = build_coverable(mk(), trace, kw, args.wide_tau)
        f5pf = CoverGatedPrescient(trace, coverable, k=max(KS), lookahead=args.lookahead)
        f5 = PFCache(cap, "s3fifo", f5pf, positions=pos, sizes=szs,
                     pf_byte_rate=(None if args.budget_mode == "total" else rate)).run(
            trace, cold_train_frac=tf, return_hits=True)
        f5_tx = f5["origin_bytes"] / max(base["origin_bytes"], 1)
        f5_corr = 100.0 * (f5["ohr"] - bar["ohr"])
        frac = f5_corr / warm_corr if warm_corr > 0 else float("nan")
        pfb_ratio = f5["prefetch_bytes"] / max(bar["prefetch_bytes"], 1)
        d = f5["hits_series"].astype(np.float64) - bar["hits_series"].astype(np.float64)
        lo, hi = block_ci(d, args.blocks, args.resamples, args.seed)

        # invariants -- a violation means the ARM is wrong, not the finding
        inv = []
        if f5["pf_cold_hits"] != 0:
            inv.append(f"F5 cold hits {f5['pf_cold_hits']} != 0 -- coverability leaking OOV objects")
        if f5["ohr"] > warm["ohr"] + 1e-9:
            inv.append("F5 OHR > WARM ceiling -- gate looser than the clairvoyant ceiling; bug")
        if args.budget_mode == "rate" and pfb_ratio > 1.02:
            inv.append(f"F5 prefetch bytes {pfb_ratio:.2f}x of bar -- NOT iso-bandwidth")
        ok = "all pass" if not inv else "FAIL"

        gate = ("CLEARS 8" if lo >= 8 else
                "CI straddles 8" if hi >= 8 else "below 8")
        print(f"  k={kw:<3d} cover {n_cov/n:5.1%}   F5 OHR {f5['ohr']:.4f} @{f5_tx:.2f}x "
              f"(pf {pfb_ratio:.2f}x)   corridor {f5_corr:+.2f} [{lo:+.2f},{hi:+.2f}]   "
              f"{frac:5.1%} of corridor   [{gate}] inv:{ok}")
        for m in inv:
            print(f"      !! {m}")
        if best is None or f5_corr > best[1]:
            best = (kw, f5_corr, lo, hi, frac)

    kw, corr, lo, hi, frac = best
    print(f"  {'-'*74}")
    if lo >= 8:
        verdict = (f"CLEARS 8 at k={kw} (corridor {corr:+.2f}, CI lo {lo:+.2f}) -- COVERAGE LEVER "
                   f"EXISTS. Build Policy 1 (wide lambda-market).")
    elif hi >= 8:
        verdict = (f"INCONCLUSIVE -- best k={kw} corridor {corr:+.2f}, CI [{lo:+.2f},{hi:+.2f}] "
                   f"straddles 8. Widen the net (larger --wide-k / lower --wide-tau) or add traces.")
    else:
        verdict = (f"BELOW 8 -- even the widest net tops out at {corr:+.2f} (CI hi {hi:+.2f}). Width "
                   f"does NOT open the corridor: Policy 1 is BLOCKED. Bottleneck is the FORECASTER "
                   f"(Policy 2), or -- if an oracle forecaster also stalls -- the corridor is not "
                   f"causally reachable (write the negative).")
    print(f"  FORK (>=8)  {verdict}\n")
    return best


def selftest():
    class A:
        trace = "SYNTH"; limit = 60000; cache_frac = 0.01; pred = "markov1"; tau = 0.05; k = 1
        window = 16; train_frac = 0.5; wide_tau = 0.0; wide_k = [1, 4, 8, 16]
        lookahead = 2000; budget_mode = "rate"; blocks = 100; resamples = 500; seed = 0
    a = A()
    trace, cap, pos, szs = prep("SYNTH", a.limit, a.cache_frac)
    run(trace, cap, pos, szs, a)
    print("  [p4_f5 selftest] completed without error\n")


def main():
    ap = argparse.ArgumentParser(description="F5 coverage-ceiling gate (Necessity Ladder rung i.5)")
    ap.add_argument("--trace")
    ap.add_argument("--limit", type=int, default=2_000_000)
    ap.add_argument("--cache-frac", type=float, default=0.01)
    ap.add_argument("--pred", choices=sorted(PREDS))
    ap.add_argument("--tau", type=float, help="the BAR's tau (for building the tuned bar)")
    ap.add_argument("--k", type=int, help="the BAR's k (for building the tuned bar)")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--wide-k", default="1,4,8,16",
                    help="comma list of emission fanouts to sweep; the widest is the UPPER bound. "
                         "Capped by the table's top_m (16) per context.")
    ap.add_argument("--wide-tau", type=float, default=0.0,
                    help="confidence floor for WIDE emission. 0.0 = emit the full top-k regardless "
                         "of confidence (the loosest, highest ceiling / true upper bound). Raise to "
                         "test a deployable threshold.")
    ap.add_argument("--lookahead", type=int, default=2000, help="clairvoyant scan window (matches F1)")
    ap.add_argument("--budget-mode", choices=("rate", "total"), default="rate",
                    help="rate = token bucket at the bar's byte rate (iso-BW, the honest ceiling). "
                         "total = cumulative cap only (looser; front-loads -- diagnostic).")
    ap.add_argument("--blocks", type=int, default=1000)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not (args.trace and args.pred and args.tau is not None and args.k is not None):
        ap.error("need --trace --pred --tau --k")
    args.wide_k = [int(x) for x in str(args.wide_k).split(",") if x.strip()]
    trace, cap, pos, szs = prep(args.trace, args.limit, args.cache_frac)
    run(trace, cap, pos, szs, args)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
