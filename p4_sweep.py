#!/usr/bin/env python3
"""
p4_sweep.py -- the two gates that decide P4 after the joint thesis died on MSR.

The joint ("co-optimize eviction + prefetch") thesis is dead: on msr_proj_0 the interaction is
NEGATIVE and monotone (eviction gain +7.13 -> +6.22 -> +5.78 as prefetch improves), and Belady --
the joint-optimal evictor for prefetched objects, since it evicts by true next-access -- converts
31k FEWER prefetches into hits than S3-FIFO, wastes 73% more of them, and still wins by 6.2 pts.
The oracle declines the "retained prefetch" prize. There is nothing there to recover.

What survives is the TIMING wedge: S3-FIFO+Prescient@BW = 0.7784 @1.05x traffic, precision 0.903 --
a REALISTIC evictor with a perfectly-timed prefetcher, +10.75 pts over our bar at LESS traffic.
Prescient beats Belady by +21.3 pts for 3% more traffic. A correctly-timed prefetch is nearly free.

Two gates stand between that observation and a paper.

GATE A -- TUNED BAR.  Our 0.6709 bar used tau=0.05, k=2: defaults I picked, not a sweep. If the
    corridor is really just "our baseline is undertuned", a reviewer tunes it for us and the paper
    dies in review instead of here. The bar must be the best DECOUPLED system available:
        best OHR at <= --max-traffic over {Markov-1, Markov-2} x tau x k.
    Markov-2 is mandatory: sweeping only tau/k on Markov-1 leaves the stronger predictor untried.
    PRE-REGISTERED: corridor = S3FIFO+Prescient@BW - tuned_bar.  >= 8 pts -> timing pivot live.
                                                                 <  8 pts -> P4 is dead. Stop.

GATE B -- n > 1.  "Eviction and prefetching are substitutes, not complements" died on ONE MSR
    block trace. That is the same small-n error that produced the +64% headroom read and the
    +2.96 autopsy artifact. Replay the identical 2x2 on every trace given. Negative on all three
    families (block / CDN / KV) and it is structural -- and a publishable measurement result that
    kills a plausible-sounding direction for everyone, not a consolation prize.

Usage:
    python p4_sweep.py --trace data/msr_proj_0.oracleGeneral --gate a
    python p4_sweep.py --trace data/msr_proj_0.oracleGeneral \
                       --trace data/wiki2019.oracleGeneral \
                       --trace data/twitter_c52.oracleGeneral --gate b
    python p4_sweep.py --trace ... --gate both        # A on trace 1, B on all
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from p4_cache import footprint_bytes, load_oracle_general, synth_trace
from p4_prefetch import (Markov1, Markov2, NoPrefetch, PFCache, Prescient, build_obj_positions,
                         build_obj_sizes, selftest)

LINE = "=" * 100
TAUS = (0.05, 0.10, 0.20, 0.35, 0.50)
KS = (1, 2, 4)


def prep(path, limit, cache_frac):
    trace = load_oracle_general(path, limit=limit) if path != "SYNTH" else synth_trace(
        n_req=min(limit, 200_000), seq_frac=0.3)
    fp = footprint_bytes(trace)
    cap = max(int(fp * cache_frac), 1)
    t0 = time.time()
    pos, szs = build_obj_positions(trace), build_obj_sizes(trace)
    print(f"[trace] {path}: {trace['n']:,} requests, {len(np.unique(trace['obj_id'])):,} objects, "
          f"footprint {fp:,} B, cache {cache_frac*100:.1f}% = {cap:,} B  ({time.time()-t0:.1f}s)")
    return trace, cap, pos, szs


def bw_rate(trace, cap, pos, szs, mk):
    """Prefetch bandwidth budget for the oracle arms, matched to LRU+Markov-1 -- the same
    convention p4_t1.py uses, as a RATE (bytes/request), not a total budget."""
    r = PFCache(cap, "lru", mk, positions=pos, sizes=szs).run(trace)
    return r["prefetch_bytes"] / max(trace["n"], 1)


# ------------------------------------------------------------------ GATE A: the tuned A1 bar
def gate_a(trace, cap, pos, szs, args, preds=None):
    """preds: optional {name: predictor} to sweep INSTEAD of the default Markov-1/2 pair.
    Callers (p4_strongbar.py) pass supersets -- the bar is the max over everything offered.
    Default path (preds=None) is unchanged and bit-identical to the pre-refactor runs."""
    print(LINE)
    print("  GATE A  TUNED BAR -- the A1 bar must be the best DECOUPLED system, not our default")
    print(LINE)
    base = PFCache(cap, "s3fifo", NoPrefetch(), positions=pos, sizes=szs).run(trace)
    base_tx = base["origin_bytes"]
    print(f"  S3-FIFO (no prefetch)  OHR {base['ohr']:.4f}   [traffic denominator]")

    if preds is None:
        t0 = time.time()
        preds = {}
        for nm, cls in (("markov1", Markov1), ("markov2", Markov2)):
            preds[nm] = cls(trace, train_frac=args.train_frac, window=args.window)
            print(f"  [build] {nm} table: {len(preds[nm].table):,} contexts  ({time.time()-t0:.1f}s)")

    print(f"  {'predictor':10s} {'tau':>5s} {'k':>3s} {'OHR':>8s} {'traffic x':>10s} "
          f"{'pf prec':>8s} {'admissible':>11s}")
    best = None
    for nm, pf in preds.items():
        for tau in (args.taus or TAUS):
            for k in KS:
                r = PFCache(cap, "s3fifo", pf.set_params(k=k, tau=tau),
                            positions=pos, sizes=szs).run(trace)
                tx = r["origin_bytes"] / max(base_tx, 1)
                ok = tx <= args.max_traffic
                if ok and (best is None or r["ohr"] > best[1]["ohr"]):
                    best = ((nm, tau, k), r, tx)
                print(f"  {nm:10s} {tau:5.2f} {k:3d} {r['ohr']:8.4f} {tx:10.2f} "
                      f"{r['pf_precision']:8.3f} {'yes' if ok else 'NO -- over BW':>11s}",
                      flush=True)

    print("-" * 100)
    if best is None:
        print(f"  !! no config fits the {args.max_traffic:.2f}x traffic budget. Widen or stop.")
        return None
    (nm, tau, k), rbar, tx = best

    # ---- ISO-BANDWIDTH ceiling ----------------------------------------------------------------
    # The ceiling MUST get the tuned bar's prefetch byte rate, not the default config's. Matching
    # the ceiling to a config the bar is not allowed to use makes the "corridor" partly a
    # bandwidth difference: on synth the bar is capped at 1.10x while Markov-1's default wants
    # 1.27x, which manufactured a +42-point corridor out of nothing.
    #
    # Note this HANDICAPS the oracle, and deliberately so. A correctly-timed prefetch is nearly
    # traffic-free (it moves a fetch earlier rather than adding one), so Prescient can issue MORE
    # prefetch bytes than Markov-1 while costing LESS origin traffic. Capping its prefetch bytes
    # at the bar's is therefore conservative: the true timing prize is >= what we print here.
    # ceiling k = max(KS), not the bar's k: the token bucket is what constrains the ceiling, and
    # every k in KS was on offer to the bar too (and lost). Binding the bound by BOTH the bar's
    # bytes and the bar's k would understate the prize twice over.
    rate = rbar["prefetch_bytes"] / max(trace["n"], 1)
    ceil = PFCache(cap, "s3fifo", Prescient(trace, k=max(KS)), positions=pos, sizes=szs,
                   pf_byte_rate=rate).run(trace)
    ctx = ceil["origin_bytes"] / max(base_tx, 1)
    corridor = 100 * (ceil["ohr"] - rbar["ohr"])
    dominates = ceil["ohr"] > rbar["ohr"] and ctx <= tx

    print(f"  TUNED A1 BAR        {nm} tau={tau} k={k}  OHR {rbar['ohr']:.4f} @{tx:.2f}x  "
          f"(precision {rbar['pf_precision']:.3f})")
    print(f"  CEILING (iso-BW)    S3-FIFO + Prescient  OHR {ceil['ohr']:.4f} @{ctx:.2f}x  "
          f"(precision {ceil['pf_precision']:.3f})")
    print(f"  TIMING CORRIDOR     {corridor:+.2f} pts at iso prefetch-bandwidth   "
          f"[pre-registered bar: >= 8.00]")
    print(f"  PARETO              ceiling {'DOMINATES' if dominates else 'does NOT dominate'} "
          f"the bar (better OHR at no more traffic)")
    if not dominates:
        print("    !! the ceiling buys OHR with traffic -- the corridor is NOT a pure timing")
        print("       prize. Do not claim it until the arms are matched on origin traffic.")
    live = corridor >= 8.0 and dominates
    print(f"  VERDICT: {'LIVE -- timing pivot survives a tuned baseline.' if live else 'DEAD -- corridor collapses under a tuned baseline. Stop P4.'}")
    print(LINE)
    return dict(bar=rbar["ohr"], pred=nm, tau=tau, k=k, ceiling=ceil["ohr"],
                corridor=corridor, dominates=dominates, live=live)


# ------------------------------------------------- GATE B: the interaction 2x2, replayed per trace
def gate_b_one(trace, cap, pos, szs, args):
    """The identical 2x2 from p4_t1.py: {S3-FIFO, Belady} x {none, Markov-1, Prescient@BW}."""
    mk = Markov1(trace, train_frac=args.train_frac, window=args.window, k=args.k, tau=args.tau)
    rate = bw_rate(trace, cap, pos, szs, mk)
    o = {}
    for pol in ("s3fifo", "belady"):
        o[(pol, "none")] = PFCache(cap, pol, NoPrefetch(), positions=pos, sizes=szs).run(trace)
        o[(pol, "markov1")] = PFCache(cap, pol, mk, positions=pos, sizes=szs).run(trace)
        o[(pol, "prescient")] = PFCache(cap, pol, Prescient(trace, k=args.k), positions=pos,
                                        sizes=szs, pf_byte_rate=rate).run(trace)
    g = lambda p, f: o[(p, f)]["ohr"]
    e_none = 100 * (g("belady", "none") - g("s3fifo", "none"))
    e_mk = 100 * (g("belady", "markov1") - g("s3fifo", "markov1"))
    e_pr = 100 * (g("belady", "prescient") - g("s3fifo", "prescient"))
    print(f"  {'2x2 OHR':22s} {'no pf':>9s} {'Markov-1':>9s} {'Prescient@BW':>13s}")
    print(f"  {'S3-FIFO (realistic)':22s} {g('s3fifo','none'):9.4f} {g('s3fifo','markov1'):9.4f} "
          f"{g('s3fifo','prescient'):13.4f}")
    print(f"  {'Belady  (oracle)':22s} {g('belady','none'):9.4f} {g('belady','markov1'):9.4f} "
          f"{g('belady','prescient'):13.4f}")
    print(f"    eviction gain   no pf: {e_none:+6.2f}   Markov-1: {e_mk:+6.2f}   "
          f"Prescient: {e_pr:+6.2f} pts")
    print(f"    INTERACTION (oracle pf - no pf) = {e_pr - e_none:+6.2f} pts   "
          f"{'SUBSTITUTES' if e_pr - e_none < 0 else 'COMPLEMENTS'}")
    # the oracle-refuses-the-prize check, which needs no autopsy label to be true
    for pol in ("s3fifo", "belady"):
        r = o[(pol, "markov1")]
        print(f"    {pol:7s}+Markov-1  prefetches kept&used {r['pf_useful']:8,}  "
              f"wasted {r['pf_wasted']:8,} ({100*r['pf_wasted']/max(r['pf_issued'],1):4.1f}%)  "
              f"OHR {r['ohr']:.4f}")
    return e_pr - e_none


def main():
    p = argparse.ArgumentParser(description="P4 gates A (tuned bar) and B (interaction, n>1)")
    p.add_argument("--trace", action="append", default=[],
                   help="repeatable; use SYNTH for the smoke test")
    p.add_argument("--gate", choices=("a", "b", "both"), default="both")
    p.add_argument("--limit", type=int, default=2_000_000)
    p.add_argument("--cache-frac", type=float, default=0.01)
    # 1.15, not 1.10: our T1 default config (tau=0.05, k=2) lands at 1.10x EXACTLY, and a cap of
    # 1.10 could reject it on float noise -- handing the "tuned bar" to a weaker config and
    # inflating our own corridor. 1.15 errs against us: it lets the baseline tune UP, never down.
    p.add_argument("--max-traffic", type=float, default=1.15,
                   help="traffic ceiling the tuned bar must respect (vs same-policy no-pf arm)")
    p.add_argument("--taus", type=lambda s: tuple(float(x) for x in s.split(",")), default=None,
                   help="override the Gate A tau grid (comma-separated), e.g. 0.05,0.06,0.07,0.08,0.09")
    p.add_argument("--k", type=int, default=2, help="k for the fixed-config 2x2 arms")
    p.add_argument("--tau", type=float, default=0.05, help="tau for the fixed-config 2x2 arms")
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--train-frac", type=float, default=0.5)
    a = p.parse_args()
    if not a.trace:
        p.error("give at least one --trace PATH (or --trace SYNTH)")

    selftest()
    inter = {}
    for i, path in enumerate(a.trace):
        trace, cap, pos, szs = prep(path, a.limit, a.cache_frac)
        if a.gate in ("a", "both") and i == 0:       # tuned bar on the primary trace only
            gate_a(trace, cap, pos, szs, a)
        if a.gate in ("b", "both"):
            print(LINE)
            print(f"  GATE B  INTERACTION 2x2 -- {path}")
            print(LINE)
            inter[path] = gate_b_one(trace, cap, pos, szs, a)
        del trace, pos, szs                          # traces are large; drop before the next

    if len(inter) > 1:
        print(LINE)
        print("  GATE B SUMMARY -- is 'substitutes, not complements' structural or n=1?")
        for k, v in inter.items():
            print(f"    {v:+6.2f} pts   {k}")
        if all(v < 0 for v in inter.values()):
            print(f"  ALL {len(inter)} NEGATIVE -> substitution is structural across trace families.")
            print("    This is a RESULT, not a consolation: it kills the joint-cache-RL direction")
            print("    for everyone, and it is a section of whatever paper follows.")
        else:
            print("  MIXED -> substitution is workload-dependent. 'Structural' is unsupported;")
            print("    the joint thesis may be alive on the positive-interaction family. Investigate.")
        print(LINE)


if __name__ == "__main__":
    main()
