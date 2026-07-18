#!/usr/bin/env python3
"""
p4_f1.py -- RUNG (i) of the Necessity Ladder: the F1 timing-only oracle.

F1 asks the one question the v3 corridor cannot answer: how much of the certified corridor is
reachable by SCHEDULING ALONE? It keeps the fixed predictor's candidate stream unchanged (same
*what*) and replaces only the issue TIMING with an oracle: each candidate is released just before
its true next use, and candidates never used again are dropped instead of funded. Same evictor,
same token bucket, same byte rate as the bar.

    F1 corridor = OHR(F1) - OHR(tuned bar)          <- the honest capture DENOMINATOR
    v3 corridor = OHR(warm-Prescient) - OHR(bar)    <- the certified corridor (also picks objects)
    F1 / v3     = the fraction of the corridor that is TIMING rather than object choice

Necessarily F1 <= warm-Prescient: the warm ceiling also gets to choose better objects, which is a
predictor's job, not a scheduler's. If F1/v3 is close to 1, the corridor is almost pure timing --
a finding in its own right.

PRE-REGISTERED GATE: F1 corridor >= 8 pts on all three live traces to proceed up the ladder.
Below 8 on a trace -> no timing-only scheduler clears the project's own bar there; drop that trace
from the capture claim. Below 8 everywhere -> the capture program is dead before a model is built.

Example (wiki):
    python p4_f1.py --trace data/wiki_2019t.oracleGeneral --pred markov2 --tau 0.05 --k 1
"""
from __future__ import annotations

import argparse
import heapq

import numpy as np

from p4_cache import NEVER
from p4_coldsplit import PREDS
from p4_prefetch import PFCache, Prescient, oracle_next
from p4_sweep import KS, prep


class JITSchedule:
    """F1 arm: the base predictor's candidates, ORACLE issue timing.

    Implemented as a stateful wrapper -- PFCache.run() calls suggest() after EVERY request, so a
    scheduler can hold candidates internally and return [] until their wake time arrives. No
    modification to the replayer is required.

    Wake time = (true next use) - lead. Fetch is instantaneous in this simulator (no latency
    model), so lead=1 is the true minimum and F1 is therefore an OPTIMISTIC bound relative to any
    real system, where lead time is a latency distribution. Do not claim F1 is deployable.
    """
    name = "jit_oracle"

    def __init__(self, base, positions, sizes, lead=1, order="urgent"):
        self.base, self.pos, self.sizes = base, positions, sizes
        self.lead, self.order = lead, order
        self.heap: list = []      # (wake, obj), lazy-deleted
        self.wake: dict = {}      # obj -> wake time (source of truth)
        self.dropped_never = 0    # candidates the oracle refused to fund (never used again)
        self.harvested = 0

    def reset(self):
        """MANDATORY: PFCache.run() calls reset() on the prefetcher. Without clearing `pending`
        here, a reused instance leaks candidates across runs and corrupts every later arm."""
        self.heap, self.wake = [], {}
        self.dropped_never = self.harvested = 0
        r = getattr(self.base, "reset", None)
        if r:
            r()

    def suggest(self, o, cached, i):
        # (1) harvest this request's candidates from the fixed predictor
        for x in self.base.suggest(o, cached, i):
            if x in self.wake or x in cached:
                continue
            self.harvested += 1
            nx = oracle_next(self.pos, x, i)
            if nx >= NEVER:
                self.dropped_never += 1
                continue                       # oracle drop: never used again -> never fund it
            w = max(int(nx) - self.lead, i)    # clamp: an imminent use releases now
            self.wake[x] = w
            heapq.heappush(self.heap, (w, x))

        # (2) release everything whose wake time has arrived
        out = []
        while self.heap and self.heap[0][0] <= i:
            w, x = heapq.heappop(self.heap)
            if self.wake.get(x) != w:
                continue                       # stale heap entry (lazy deletion)
            del self.wake[x]
            if x in cached:
                continue                       # demand-filled while pending -> cancel
            if oracle_next(self.pos, x, i) >= NEVER:
                continue                       # use already passed -> stale, drop
            out.append(x)

        if len(out) > 1:
            # HOL BLOCKING: PFCache breaks (not continues) when tokens < size, so one oversized
            # candidate at the front starves every smaller one behind it for this request.
            # --order small is the sanity ablation that exposes it.
            if self.order == "small":
                out.sort(key=lambda x: self.sizes.get(x, 0))
            else:
                out.sort(key=lambda x: oracle_next(self.pos, x, i))
        return out


def block_ci(diff, blocks, resamples, seed):
    """Paired moving-block bootstrap, identical convention to p4_bootstrap.py."""
    R = len(diff)
    L = R // blocks
    bm = diff[:blocks * L].reshape(blocks, L).mean(axis=1)
    rng = np.random.default_rng(seed)
    boot = 100.0 * bm[rng.integers(0, blocks, size=(resamples, blocks))].mean(axis=1)
    return np.percentile(boot, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description="F1 timing-only oracle (Necessity Ladder rung i)")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--limit", type=int, default=2_000_000)
    ap.add_argument("--cache-frac", type=float, default=0.01)
    ap.add_argument("--pred", choices=sorted(PREDS), required=True,
                    help="winning bar predictor from the strong-bar log")
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--lead", type=int, default=1, help="release this many requests before use")
    ap.add_argument("--order", choices=("urgent", "small"), default="urgent",
                    help="release order; 'small' is the head-of-line-blocking ablation")
    ap.add_argument("--blocks", type=int, default=1000)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    trace, cap, pos, szs = prep(args.trace, args.limit, args.cache_frac)
    tf = args.train_frac
    base = PFCache(cap, "s3fifo", None, positions=pos, sizes=szs).run(trace)

    def mk():
        return PREDS[args.pred](trace, train_frac=tf, window=args.window, k=args.k, tau=args.tau)

    # ---- BAR: fixed predictor, immediate issue (fixes the iso-BW rate) ----
    bar = PFCache(cap, "s3fifo", mk(), positions=pos, sizes=szs).run(
        trace, cold_train_frac=tf, return_hits=True)
    rate = bar["prefetch_bytes"] / max(trace["n"], 1)
    bar_tx = bar["origin_bytes"] / max(base["origin_bytes"], 1)

    # ---- F1: same candidates, ORACLE timing, same byte rate ----
    jit = JITSchedule(mk(), pos, szs, lead=args.lead, order=args.order)
    f1 = PFCache(cap, "s3fifo", jit, positions=pos, sizes=szs, pf_byte_rate=rate).run(
        trace, cold_train_frac=tf, return_hits=True)
    f1_tx = f1["origin_bytes"] / max(base["origin_bytes"], 1)

    # ---- v3 warm ceiling (the certified corridor), for the F1/v3 ratio ----
    cut = int(trace["n"] * tf)
    tvocab = set(int(x) for x in trace["obj_id"][:cut])
    warm = PFCache(cap, "s3fifo", Prescient(trace, k=max(KS), vocab=tvocab), positions=pos,
                   sizes=szs, pf_byte_rate=rate).run(trace, cold_train_frac=tf)
    warm_tx = warm["origin_bytes"] / max(base["origin_bytes"], 1)

    f1_corr = 100.0 * (f1["ohr"] - bar["ohr"])
    v3_corr = 100.0 * (warm["ohr"] - bar["ohr"])
    ratio = f1_corr / v3_corr if v3_corr > 0 else float("nan")

    d = f1["hits_series"].astype(np.float64) - bar["hits_series"].astype(np.float64)
    lo, hi = block_ci(d, args.blocks, args.resamples, args.seed)

    print(f"\n  {args.trace}   [lead={args.lead} order={args.order}]")
    print(f"  BAR            {args.pred} tau={args.tau} k={args.k}  OHR {bar['ohr']:.4f} "
          f"@{bar_tx:.2f}x   pf {bar['pf_issued']:,} useful {bar['pf_useful']:,} "
          f"(prec {bar['pf_precision']:.3f})")
    print(f"  F1 (JIT ORACLE)                     OHR {f1['ohr']:.4f} @{f1_tx:.2f}x   "
          f"pf {f1['pf_issued']:,} useful {f1['pf_useful']:,} (prec {f1['pf_precision']:.3f})")
    print(f"  WARM CEILING (v3)                   OHR {warm['ohr']:.4f} @{warm_tx:.2f}x")
    print(f"\n  F1 CORRIDOR   {f1_corr:+.2f} pts   95% CI [{lo:+.2f}, {hi:+.2f}]")
    print(f"  v3 CORRIDOR   {v3_corr:+.2f} pts")
    print(f"  F1 / v3       {ratio:.1%}   <- fraction of the corridor reachable by TIMING alone")
    print(f"  oracle dropped {jit.dropped_never:,} of {jit.harvested:,} harvested candidates "
          f"(never used again)")

    # ---- invariants: a violation means the ARM is wrong, not the finding ----
    inv = []
    if bar["pf_cold_hits"] != 0:
        inv.append(f"bar cold hits = {bar['pf_cold_hits']}, expect 0")
    if f1["pf_cold_hits"] != 0:
        inv.append(f"F1 cold hits = {f1['pf_cold_hits']}, expect 0")
    if f1["ohr"] < bar["ohr"]:
        inv.append("OHR(F1) < OHR(bar) -- suspect head-of-line blocking; rerun --order small")
    if f1["ohr"] > warm["ohr"] + 1e-9:
        inv.append("OHR(F1) > OHR(warm ceiling) -- F1 is funding out-of-vocab objects; bug")
    if f1_tx > bar_tx + 0.01:
        inv.append(f"F1 traffic {f1_tx:.2f}x > bar {bar_tx:.2f}x -- NOT iso-bandwidth, Pareto fails")
    print("  INVARIANTS    " + ("all pass" if not inv else "FAIL"))
    for m in inv:
        print(f"    !! {m}")

    gate = ("CLEARS 8 -- proceed up the ladder" if lo >= 8 else
            "CI straddles 8 -- inconclusive" if hi >= 8 else "BELOW 8 -- timing-only cannot win here")
    print(f"  GATE (F1 >= 8) {gate}\n")


if __name__ == "__main__":
    main()
