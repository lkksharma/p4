#!/usr/bin/env python3
"""
p4_t1.py -- the T1 skeleton (guide W2) + the mechanism test, with ZERO RL.

THE MONEY QUESTION, answerable before any learning:
    Do hits exist OUTSIDE Belady's decision space, and does a realistic prefetcher convert them?

    A1_structural = Belady+Markov1  -  Belady      (a BOUND: oracle eviction makes bad prefetches
                                                    free, so this FLATTERS prefetch. It measures
                                                    the structural headroom, not an achievement.)
    A1_realistic  = LRU+Markov1     -  LRU          (does prefetch pay when waste actually hurts?)
    The RL's job (W4) is to beat LRU+Markov1 -- the tuned separate-combined system.

Usage:
    python p4_t1.py --trace data/msr_proj_0.oracleGeneral --limit 2000000
    python p4_t1.py --synth
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from p4_cache import footprint_bytes, load_oracle_general, synth_trace
from p4_prefetch import (Markov1, NoPrefetch, PFCache, Prescient, build_obj_positions,
                         build_obj_sizes, selftest)

LINE = "=" * 100


def main():
    p = argparse.ArgumentParser(description="P4 T1 skeleton + prefetch mechanism test")
    p.add_argument("--trace", default=None)
    p.add_argument("--synth", action="store_true")
    p.add_argument("--limit", type=int, default=2_000_000)
    p.add_argument("--cache-frac", type=float, default=0.01)
    p.add_argument("--k", type=int, default=2, help="objects prefetched per request")
    p.add_argument("--tau", type=float, default=0.05, help="Markov-1 confidence threshold")
    p.add_argument("--window", type=int, default=16, help="Markov-1 successor window")
    p.add_argument("--train-frac", type=float, default=0.5, help="prefix used to build Markov-1")
    a = p.parse_args()
    if not a.trace and not a.synth:
        p.error("give --trace PATH or --synth")

    selftest()                                  # the accounting contract must hold first

    if a.trace:
        trace = load_oracle_general(a.trace, limit=a.limit)
        tag = a.trace
    else:
        trace = synth_trace(n_req=min(a.limit, 200_000), seq_frac=0.3)
        tag = "SYNTHETIC"
    fp = footprint_bytes(trace)
    cap = max(int(fp * a.cache_frac), 1)
    print(f"\n[trace] {tag}: {trace['n']:,} requests, {len(np.unique(trace['obj_id'])):,} objects, "
          f"footprint {fp:,} B, cache {a.cache_frac*100:.1f}% = {cap:,} B")

    t0 = time.time()
    positions = build_obj_positions(trace)
    sizes = build_obj_sizes(trace)
    mk = Markov1(trace, train_frac=a.train_frac, window=a.window, top_m=16, k=a.k, tau=a.tau)
    print(f"[build] positions+sizes+markov1 in {time.time()-t0:.1f}s "
          f"(markov table: {len(mk.table):,} objects)")

    # pass 1: the no-prefetch arms establish the traffic BASELINE per policy
    res = {}
    for name, pol in (("LRU", "lru"), ("S3-FIFO", "s3fifo"), ("Belady", "belady")):
        res[name] = PFCache(cap, pol, NoPrefetch(), positions=positions, sizes=sizes).run(trace)
    base_traffic = {"lru": res["LRU"]["origin_bytes"],
                    "s3fifo": res["S3-FIFO"]["origin_bytes"],
                    "belady": res["Belady"]["origin_bytes"]}
    # S3-FIFO + Markov-1 = the HONEST separate-combined bar (guide A1): heuristic-SOTA evictor
    # + a tuned prefetcher, each tuned independently. This is what the joint agent must beat.
    res["S3-FIFO + Markov-1"] = PFCache(cap, "s3fifo", mk, positions=positions,
                                        sizes=sizes).run(trace)

    # pass 2: prefetch arms. The prescient bound is BANDWIDTH-MATCHED to LRU+Markov-1's prefetch
    # traffic -- unlimited-bandwidth prescient trivially reaches OHR 1.0 and bounds nothing.
    r_lrupf = PFCache(cap, "lru", mk, positions=positions, sizes=sizes).run(trace)
    res["LRU + Markov-1"] = r_lrupf
    # bandwidth matching as a RATE (bytes of prefetch per request), spread over the whole trace --
    # not a total budget, which would be burned greedily in the first few % of requests.
    rate = r_lrupf["prefetch_bytes"] / max(trace["n"], 1)
    res["Belady + Markov-1"] = PFCache(cap, "belady", mk, positions=positions,
                                       sizes=sizes).run(trace)
    res["Prescient @matched-BW"] = PFCache(cap, "belady", Prescient(trace, k=a.k),
                                           positions=positions, sizes=sizes,
                                           pf_byte_rate=rate).run(trace)
    res["Prescient @unlimited"] = PFCache(cap, "belady", Prescient(trace, k=a.k),
                                          positions=positions, sizes=sizes).run(trace)
    # THE DECISIVE CELL: oracle prefetch under REALISTIC eviction. Completes the 2x2 and splits the
    # corridor into "better prefetcher" vs "jointness". Without it the corridor's composition is
    # unknown and a joint RL is a bet, not an experiment.
    res["S3-FIFO + Prescient@BW"] = PFCache(cap, "s3fifo", Prescient(trace, k=a.k),
                                            positions=positions, sizes=sizes,
                                            pf_byte_rate=rate).run(trace)

    order = ["LRU", "LRU + Markov-1", "S3-FIFO", "S3-FIFO + Markov-1", "S3-FIFO + Prescient@BW",
             "Belady", "Belady + Markov-1", "Prescient @matched-BW", "Prescient @unlimited"]
    print(LINE)
    print(f"  {'arm':24s} {'OHR':>8s} {'BHR':>8s} {'traffic x':>10s} {'pf issued':>11s} "
          f"{'pf prec':>8s} {'wasted':>10s}")
    print(LINE)
    for name in order:
        r = res[name]
        # traffic vs the SAME-POLICY no-prefetch arm (correct denominator)
        tx = r["origin_bytes"] / max(base_traffic[r["policy"]], 1)
        print(f"  {name:24s} {r['ohr']:8.4f} {r['bhr']:8.4f} {tx:10.2f} "
              f"{r['pf_issued']:11,} {r['pf_precision']:8.3f} {r['pf_wasted']:10,}", flush=True)
    print(LINE)

    lru, lrupf = res["LRU"]["ohr"], res["LRU + Markov-1"]["ohr"]
    bel, belpf = res["Belady"]["ohr"], res["Belady + Markov-1"]["ohr"]
    pres = res["Prescient @matched-BW"]["ohr"]
    struct = 100 * (belpf - bel)
    real = 100 * (lrupf - lru)
    tx = res["LRU + Markov-1"]["origin_bytes"] / max(base_traffic["lru"], 1)
    print(f"  STRUCTURAL headroom  Belady+Markov1 - Belady = {struct:+.2f} pts   "
          f"(a BOUND: oracle eviction discards bad prefetches for free)")
    print(f"  REALISTIC gain       LRU+Markov1    - LRU    = {real:+.2f} pts   at traffic x{tx:.2f}")
    print(f"  Prescient @matched-BW {pres:.4f} vs Belady {bel:.4f} = {100*(pres-bel):+.2f} pts "
          f"(oracle evict+prefetch at LRU+Markov-1's bandwidth; a BOUND, not OPT --")
    print(f"                        joint OPT is intractable, which is the thesis)")
    print("-" * 100)
    # ---- the A1 corridor: what the joint agent must beat, and its ceiling ----
    sc = res["S3-FIFO + Markov-1"]["ohr"]
    print(f"  A1 BAR (honest)     S3-FIFO+Markov-1 = {sc:.4f}   <- the joint agent must beat THIS")
    print(f"  A1 BAR (upper)      Belady+Markov-1  = {belpf:.4f}   <- upper bound on ANY decoupled")
    print(f"                      system (no evictor beats Belady). Beat it => beat them all.")
    print(f"  CEILING             Prescient@BW     = {pres:.4f}")
    print(f"  CORRIDOR vs honest bar : {100*(pres-sc):+.2f} pts     "
          f"CORRIDOR vs upper bar : {100*(pres-belpf):+.2f} pts")
    if pres - sc < 0.01:
        print("  !! No corridor: a tuned decoupled system already reaches the bound. STOP.")
    print(LINE)

    # ================= W3 GATE: is the corridor JOINTNESS, or just a better prefetcher? =========
    # A corridor is necessary but NOT sufficient. If eviction quality and prefetch quality are
    # ADDITIVE, then tuning them separately is already near-optimal and a joint agent can only
    # reprint "S3-FIFO + a better prefetcher". The paper needs the INTERACTION to be positive.
    s3, s3pf = res["S3-FIFO"]["ohr"], res["S3-FIFO + Markov-1"]["ohr"]
    s3pres = res["S3-FIFO + Prescient@BW"]["ohr"]
    print("  W3  INTERACTION -- does eviction matter MORE when prefetching is on?")
    print(f"  {'2x2 OHR':22s} {'no pf':>9s} {'Markov-1':>9s} {'Prescient@BW':>13s}")
    print(f"  {'S3-FIFO (realistic)':22s} {s3:9.4f} {s3pf:9.4f} {s3pres:13.4f}")
    print(f"  {'Belady  (oracle)':22s} {bel:9.4f} {belpf:9.4f} {pres:13.4f}")
    e_none = 100 * (bel - s3)
    e_mk = 100 * (belpf - s3pf)
    e_or = 100 * (pres - s3pres)
    print(f"    eviction gain (Belady - S3-FIFO)   no pf: {e_none:+6.2f}   "
          f"Markov-1: {e_mk:+6.2f}   Prescient: {e_or:+6.2f} pts")
    print(f"    INTERACTION  = eviction gain WITH oracle pf - WITHOUT pf = {e_or - e_none:+6.2f} pts")
    print("      > 0 : eviction and prefetch are COUPLED -> a joint policy has a mechanism.")
    print("      ~ 0 : SEPARABLE -> decoupled tuning is already near-optimal; the corridor is")
    print("            prefetcher quality and a joint RL will reprint the decoupled system.")

    # --- the autopsy: name the prize mechanically, on the honest bar ---
    r = res["S3-FIFO + Markov-1"]
    w, wc, wm = r["pf_wasted"], r["pf_w_correct"], r["pf_w_mispred"]
    nreq = max(r["requests"], 1)
    print(f"  WASTED-PREFETCH AUTOPSY (S3-FIFO + Markov-1, {w:,} wasted of {r['pf_issued']:,}):")
    print(f"    correct, evicted before use : {wc:8,}  ({100*wc/max(w,1):5.1f}%)  "
          f"median dist to use {r['pf_w_correct_med_dist']:,} reqs")
    print(f"      -> JOINTNESS prize: <= {100*wc/nreq:+.2f} OHR pts. An UPPER bound -- retaining")
    print(f"         these costs cache space, which costs other hits. That tradeoff IS the")
    print(f"         joint policy's job. If this is ~0, the joint thesis has no prize here.")
    print(f"    mispredicted (never used)   : {wm:8,}  ({100*wm/max(w,1):5.1f}%)")
    print(f"      -> PREFETCHER-QUALITY loss: a better predictor fixes these. No jointness needed.")
    print(LINE)
    if struct <= 0:
        print("  VERDICT: NO hits outside Belady's decision space on this trace/config.")
        print("    Even with ORACLE eviction a realistic prefetcher adds nothing -> the joint")
        print("    thesis has no mechanism here. Change trace/k/tau BEFORE building the RL.")
    elif real <= 0:
        print("  VERDICT: MIXED. Structural headroom exists (prefetch DOES add hits Belady cannot),")
        print("    but with realistic eviction the prefetches cost more than they pay. That is")
        print("    precisely the tension the joint policy must resolve -- and it is the paper's")
        print("    reason to exist. Necessary condition met; the RL now has a real job.")
    else:
        print("  VERDICT: BOTH headrooms positive. Prefetch adds hits Belady cannot express AND")
        print("    pays under realistic eviction. W4's bar is S3-FIFO+Markov-1 (the honest")
        print("    separate-combined), NOT LRU+Markov-1 and NOT Belady. Report OHR *with* traffic:")
        print("    a strong evictor alone can Pareto-dominate a weak evictor + prefetcher.")
    print(LINE)


if __name__ == "__main__":
    main()
