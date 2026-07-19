#!/usr/bin/env python3
"""p4_policy1.py -- Necessity Ladder rung (ii), the POSITIVE policy: the wide-emission bandwidth
market ("The Price of a Byte"), plus the diagnostic that decides whether Policy 2 is needed.

WHY THIS, AND WHY NOW. The capture track spent five gates proving what does NOT work:
  * F1-stream  : perfect TIMING on the k=1 emission stream <= bar  -> the corridor is not timing.
  * oracle arm : perfect forecaster, fetch-at-emission, k=1, only TIES the bar.
  * F4'        : a joint/RL arbitration adds < 1 pt                 -> RL is unnecessary.
  * DEFER      : S(100)=1.0, no eviction race                       -> deferral is dead.
  * F5         : with a WIDE emission net, a clairvoyant selector recovers most of the learnable
                 corridor (cluster50 +11.42 CLEARS 8; wiki +8.18 and rising) -> the corridor is a
                 COVERAGE gap, and width + price-selection is the lever that opens it.

Policy 1 is the CAUSAL realization of the F5 ceiling. One request at a time, no future:
  1. EMIT WIDE     -- the predictor at fanout k_wide, low tau (many emission events per object).
  2. VALUE         -- v(X) from a causal forecaster (see below). No clairvoyance.
  3. PRICE & FUND  -- hold candidates in a pool; fund highest v(X)/size(X) first while a token
                      bucket at the bar's byte rate (the bandwidth shadow price) allows, under a
                      hard total-byte cap = the bar's prefetch bytes. Iso-bandwidth by construction.
  No DEFER (dead). No RL (F4'). The ONLY new lever over the bar is width + price-selection.

THE FORECASTER IS PLUGGABLE -- this is exactly where Policy 2 attaches:
  markov : v(X) = the predictor's own emission confidence.        [Policy 1: the cheap causal default]
  blend  : v(X) = conf(X) reweighted by causal recency/frequency. [Policy 2 v0: recall-first, cheap]
  oracle : v(X) = 1 if X is used within the horizon else 0.       [DIAGNOSTIC ONLY -- reads the
           future. The gap market(oracle) - market(markov) is Policy 2's PRIZE, measured before a
           single neural parameter is trained. NEVER report `oracle` as a capture result.]

  Ladder logic: if market(oracle) >> market(markov) but market(blend) also stalls, a cheap reweight
  cannot recover the prize and the heavier learned forecaster (EA-MTPP's what-predictor) is JUSTIFIED
  -- that, and only that, licenses Policy 2's neural rung. If market(markov) already ~ market(oracle),
  the forecaster is not the bottleneck and Policy 2 is unnecessary.

REPORTED against the F5 coverage ceiling (the honest denominator for a wide-emit policy) AND the bar,
two-axis (OHR + prefetch bytes). PRE-REGISTERED PASS BAR: capture >= 25% of the F5 ceiling corridor,
bootstrap CI excluding zero, on every trace whose F5 cleared.

    python p4_policy1.py --trace data/cluster50.sample10.oracleGeneral --pred markov3 --tau 0.06 \
        --k 1 --wide-k 16 --forecasters markov,blend,oracle
    python p4_policy1.py --selftest
"""
from __future__ import annotations

import argparse

import numpy as np

from p4_cache import NEVER
from p4_coldsplit import PREDS
from p4_f4 import capture_ctx, conf_lookup
from p4_f5 import CoverGatedPrescient, build_coverable
from p4_prefetch import PFCache, oracle_next
from p4_sweep import KS, prep


def block_ci(diff, blocks, resamples, seed):
    R = len(diff); L = R // blocks
    bm = diff[:blocks * L].reshape(blocks, L).mean(axis=1)
    rng = np.random.default_rng(seed)
    boot = 100.0 * bm[rng.integers(0, blocks, size=(resamples, blocks))].mean(axis=1)
    return np.percentile(boot, [2.5, 97.5])


class WideMarket:
    """Causal wide-emission bandwidth market. Owns its budget outright (PFCache runs with
    pf_byte_rate=None), so the token bucket + hard cap here are the ONLY iso-bandwidth control and
    the v/size price ranking is never overridden by PFCache's head-of-line break."""
    name = "policy1"

    def __init__(self, base, pos, szs, cap, rate, budget, forecaster="markov",
                 horizon=100_000, floor=0.0, max_wait=200_000, budget_mode="rate"):
        self.base, self.pos, self.szs, self.cap = base, pos, szs, cap
        self.rate, self.budget = rate, budget
        self.forecaster, self.horizon = forecaster, horizon
        self.floor, self.max_wait, self.budget_mode = floor, max_wait, budget_mode
        self.tokens = self.spent = 0.0
        self.last_seen: dict = {}                 # obj -> last request index (causal recency)
        self.freq: dict = {}                      # obj -> count so far (causal frequency)
        self.pool: dict = {}                      # obj -> (size, value, first_seen_i)
        self.issued = 0

    def reset(self):
        self.tokens = self.spent = 0.0
        self.last_seen, self.freq, self.pool = {}, {}, {}
        self.issued = 0
        r = getattr(self.base, "reset", None)
        if r:
            r()

    def _value(self, x, conf, i):
        if self.forecaster == "oracle":                       # DIAGNOSTIC: reads the future
            nx = oracle_next(self.pos, x, i)
            if nx >= i + self.horizon:
                return 0.0
            return 1.0 / (1.0 + (nx - i))                     # prioritise SOONEST uses (survive, low
                                                              # occupancy) -- mirrors F5's selection
        if self.forecaster == "blend":                        # recall-first: boost popular repeaters
            return float(conf * (1.0 + np.log1p(self.freq.get(x, 0))))
        return float(conf)                                    # markov: raw emission confidence

    def suggest(self, o, cached, i):
        # bandwidth shadow price = an accruing token bucket at the bar's byte rate
        self.tokens = float("inf") if self.budget_mode == "total" else self.tokens + self.rate

        # (1) harvest WIDE candidates, value each causally, add to the pool
        ctx = capture_ctx(self.base)                          # snapshot BEFORE suggest() mutates it
        objs = self.base.suggest(o, cached, i)
        if objs:
            cmap = conf_lookup(self.base, ctx, o)
            tau = getattr(self.base, "tau", 0.05)
            for x in objs:
                x = int(x)
                if x in cached or x in self.pool:
                    continue
                s = self.szs.get(x)
                if s is None or s > self.cap:                 # unadmittable -> never fund (keeps the
                    continue                                  # budget mirror in sync with PFCache)
                v = self._value(x, cmap.get(x, tau), i)
                if v <= self.floor:
                    continue
                self.pool[x] = (s, v, i)
        self.last_seen[o] = i
        self.freq[o] = self.freq.get(o, 0) + 1

        # (2) expire demand-filled / stale candidates
        drop = [x for x, (s, v, fi) in self.pool.items()
                if x in cached or i - fi > self.max_wait]
        for x in drop:
            self.pool.pop(x, None)
        if not self.pool:
            return []

        # (3) fund highest value-per-byte first, within tokens AND the hard total cap
        ranked = sorted(self.pool.items(), key=lambda kv: kv[1][1] / kv[1][0], reverse=True)
        out = []
        for x, (s, v, fi) in ranked:
            if self.spent + s > self.budget:
                continue
            if self.budget_mode != "total" and self.tokens < s:
                continue
            self.tokens -= s
            self.spent += s
            self.pool.pop(x, None)
            self.issued += 1
            out.append(x)
        out.sort(key=lambda x: self.szs.get(x, 0))            # smallest-first: head-of-line safety
        return out


def run(trace, cap, pos, szs, args):
    tf = args.train_frac
    n = trace["n"]
    kw = args.wide_k

    def mk():                                                 # the tuned bar (default top_m)
        return PREDS[args.pred](trace, train_frac=tf, window=args.window, k=args.k, tau=args.tau)

    def mk_wide():                                            # the WIDE emitter (k_wide, low tau)
        p = PREDS[args.pred](trace, train_frac=tf, window=args.window, k=args.k, tau=args.tau,
                             top_m=args.wide_top_m)
        p.k, p.tau = kw, args.wide_tau
        return p

    base = PFCache(cap, "s3fifo", None, positions=pos, sizes=szs).run(trace)
    bar = PFCache(cap, "s3fifo", mk(), positions=pos, sizes=szs).run(
        trace, cold_train_frac=tf, return_hits=True)
    rate = bar["prefetch_bytes"] / max(n, 1)
    budget = bar["prefetch_bytes"]
    bar_tx = bar["origin_bytes"] / max(base["origin_bytes"], 1)

    # F5 COVERAGE CEILING at the same wide config -- the honest denominator for a wide-emit policy.
    coverable, n_cov = build_coverable(mk_wide(), trace, kw, args.wide_tau)
    f5 = PFCache(cap, "s3fifo", CoverGatedPrescient(trace, coverable, k=max(KS), lookahead=2000),
                 positions=pos, sizes=szs, pf_byte_rate=rate).run(
        trace, cold_train_frac=tf, return_hits=True)
    f5_corr = 100.0 * (f5["ohr"] - bar["ohr"])

    print(f"\n  {args.trace}   [Policy 1: wide market | k_wide={kw} tau={args.wide_tau} "
          f"budget={args.budget_mode}]")
    print(f"  BAR       OHR {bar['ohr']:.4f} @{bar_tx:.2f}x   pf {bar['pf_issued']:,} "
          f"useful {bar['pf_useful']:,} (prec {bar['pf_precision']:.3f})")
    print(f"  F5 CEIL   OHR {f5['ohr']:.4f}   corridor {f5_corr:+.2f} pts   (coverage {n_cov/n:.1%}) "
          f"<- the denominator")
    print(f"  {'-'*76}")

    results = {}
    for fc in args.forecasters:
        sch = WideMarket(mk_wide(), pos, szs, cap, rate, budget, forecaster=fc,
                         horizon=args.horizon, floor=args.floor, max_wait=args.max_wait,
                         budget_mode=args.budget_mode)
        p1 = PFCache(cap, "s3fifo", sch, positions=pos, sizes=szs, pf_byte_rate=None).run(
            trace, cold_train_frac=tf, return_hits=True)
        p1_tx = p1["origin_bytes"] / max(base["origin_bytes"], 1)
        p1_corr = 100.0 * (p1["ohr"] - bar["ohr"])
        cap_f5 = p1_corr / f5_corr if f5_corr > 0 else float("nan")
        pfb = p1["prefetch_bytes"] / max(bar["prefetch_bytes"], 1)
        d = p1["hits_series"].astype(np.float64) - bar["hits_series"].astype(np.float64)
        lo, hi = block_ci(d, args.blocks, args.resamples, args.seed)

        inv = []
        if p1["pf_cold_hits"] != 0:
            inv.append(f"cold hits {p1['pf_cold_hits']} != 0 -- funding OOV objects")
        if pfb > 1.02:
            inv.append(f"prefetch bytes {pfb:.2f}x of bar -- NOT iso-bandwidth")
        okinv = "all pass" if not inv else "FAIL"

        tag = "  (DIAGNOSTIC -- reads future, not reportable)" if fc == "oracle" else ""
        gate = ("PASS" if (lo > 0 and cap_f5 >= 0.25) else
                f"below 25% ({cap_f5:.0%})" if lo > 0 else "CI includes 0")
        print(f"  [{fc:6s}] OHR {p1['ohr']:.4f} @{p1_tx:.2f}x   pf {p1['pf_issued']:,} "
              f"useful {p1['pf_useful']:,} (prec {p1['pf_precision']:.3f})   spent "
              f"{sch.spent/max(budget,1):.2f}x{tag}")
        print(f"           corridor {p1_corr:+.2f} [{lo:+.2f},{hi:+.2f}]   capture {cap_f5:.1%} of F5 "
              f"| pf bytes {pfb:.2f}x   [{gate}] inv:{okinv}")
        for m in inv:
            print(f"           !! {m}")
        results[fc] = (p1_corr, lo, hi, cap_f5)

    # Policy-2 verdict: is the forecaster the bottleneck?
    if "markov" in results and "oracle" in results:
        gap = results["oracle"][0] - results["markov"][0]
        line = f"  POLICY-2 PRIZE   market(oracle) - market(markov) = {gap:+.2f} pts"
        if "blend" in results:
            closed = results["blend"][0] - results["markov"][0]
            line += f"   | blend closes {closed:+.2f}"
        print(f"  {'-'*76}\n{line}")
        if gap < 2:
            print("  -> forecaster is NOT the bottleneck; Policy 2 unnecessary (market ~ its own ceiling).")
        elif "blend" in results and results["blend"][0] - results["markov"][0] >= 0.5 * gap:
            print("  -> a CHEAP reweight (blend) recovers most of the prize; ship blend, skip the neural rung.")
        else:
            print("  -> prize is real and cheap reweight stalls -> the learned recall forecaster "
                  "(Policy 2 neural rung) is JUSTIFIED.")
    print()
    return results


def selftest():
    class A:
        trace = "SYNTH"; limit = 60000; cache_frac = 0.01; pred = "markov1"; tau = 0.05; k = 1
        window = 16; train_frac = 0.5; wide_k = 16; wide_tau = 0.0; wide_top_m = 16
        forecasters = ["markov", "blend", "oracle"]; horizon = 5000; floor = 0.0
        max_wait = 20000; budget_mode = "rate"; blocks = 100; resamples = 500; seed = 0
    a = A()
    trace, cap, pos, szs = prep("SYNTH", a.limit, a.cache_frac)
    run(trace, cap, pos, szs, a)
    print("  [p4_policy1 selftest] completed without error\n")


def main():
    ap = argparse.ArgumentParser(description="Policy 1: wide-emission bandwidth market (ladder rung ii)")
    ap.add_argument("--trace")
    ap.add_argument("--limit", type=int, default=2_000_000)
    ap.add_argument("--cache-frac", type=float, default=0.01)
    ap.add_argument("--pred", choices=sorted(PREDS))
    ap.add_argument("--tau", type=float, help="the BAR's tau")
    ap.add_argument("--k", type=int, help="the BAR's k")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--wide-k", type=int, default=16, help="emission fanout (the F5-cleared width)")
    ap.add_argument("--wide-tau", type=float, default=0.0)
    ap.add_argument("--wide-top-m", type=int, default=16, help="successors/context in the wide emitter")
    ap.add_argument("--forecasters", default="markov,blend,oracle",
                    help="comma list of causal value functions to run: markov (default), blend "
                         "(Policy 2 v0, recall-first), oracle (DIAGNOSTIC ceiling, not reportable)")
    ap.add_argument("--horizon", type=int, default=100_000, help="[oracle] use-within-horizon label")
    ap.add_argument("--floor", type=float, default=0.0, help="drop candidates with value <= floor")
    ap.add_argument("--max-wait", type=int, default=200_000)
    ap.add_argument("--budget-mode", choices=("rate", "total"), default="rate")
    ap.add_argument("--blocks", type=int, default=1000)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not (args.trace and args.pred and args.tau is not None and args.k is not None):
        ap.error("need --trace --pred --tau --k")
    args.forecasters = [x for x in str(args.forecasters).split(",") if x.strip()]
    trace, cap, pos, szs = prep(args.trace, args.limit, args.cache_frac)
    run(trace, cap, pos, szs, args)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
