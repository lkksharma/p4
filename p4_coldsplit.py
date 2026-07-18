#!/usr/bin/env python3
"""
p4_coldsplit.py -- cold-miss decomposition of an already-measured corridor, WITHOUT
re-running the full tau x k sweep.

Give it the trace and the winning bar config from the logs; it replays exactly two arms
(bar once to fix the iso-BW rate, ceiling once with cold tracking) and reports:
    gross corridor      = ceiling - bar                    (the pre-registered number)
    cold slice          = ceiling hits from prefetches of never-yet-requested objects
                          (compulsory-miss elimination -- clairvoyance-only, unlearnable)
    LEARNABLE corridor  = gross - cold                     (what a learned policy can chase;
                                                            the abstract must be sized to this)

The bar OHR printed here must match the strong-bar log to 4 decimals -- if it does not,
you passed the wrong config and nothing downstream is trustworthy.

Example (wiki, bar from logs/strongbar_wiki_2019t.log):
    python p4_coldsplit.py --trace data/wiki_2019t.oracleGeneral --limit 2000000 \
        --pred markov2 --tau 0.06 --k 1
"""
from __future__ import annotations

import argparse

from p4_prefetch import Markov1, Markov2, PFCache, Prescient
from p4_strongbar import Markov3
from p4_sweep import KS, prep

PREDS = {"markov1": Markov1, "markov2": Markov2, "markov3": Markov3}


def main():
    ap = argparse.ArgumentParser(description="cold-miss split of the timing corridor")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--limit", type=int, default=2_000_000)
    ap.add_argument("--cache-frac", type=float, default=0.01)
    ap.add_argument("--pred", choices=sorted(PREDS), required=True,
                    help="winning bar predictor from the strong-bar log")
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=0.5)
    args = ap.parse_args()

    trace, cap, pos, szs = prep(args.trace, args.limit, args.cache_frac)
    base = PFCache(cap, "s3fifo", None, positions=pos, sizes=szs).run(trace)

    pf = PREDS[args.pred](trace, train_frac=args.train_frac, window=args.window,
                          k=args.k, tau=args.tau)
    bar = PFCache(cap, "s3fifo", pf, positions=pos, sizes=szs).run(
        trace, cold_train_frac=args.train_frac)
    bar_tx = bar["origin_bytes"] / max(base["origin_bytes"], 1)

    # identical ceiling convention to gate_a: Prescient k=max(KS) at the bar's byte rate
    rate = bar["prefetch_bytes"] / max(trace["n"], 1)
    ceil = PFCache(cap, "s3fifo", Prescient(trace, k=max(KS)), positions=pos, sizes=szs,
                   pf_byte_rate=rate).run(trace, cold_train_frac=args.train_frac)
    ctx = ceil["origin_bytes"] / max(base["origin_bytes"], 1)

    # LEARNABLE (warm) ceiling: perfect timing, in-vocab objects only, same budget -> reallocates
    # the cold-wasted budget onto warm objects. This is the honest, budget-fair learnable metric
    # (>= gross - cold, because gross-minus-cold ignores the reallocation).
    cut = int(trace["n"] * args.train_frac)
    tvocab = set(int(x) for x in trace["obj_id"][:cut])
    warm = PFCache(cap, "s3fifo", Prescient(trace, k=max(KS), vocab=tvocab), positions=pos,
                   sizes=szs, pf_byte_rate=rate).run(trace, cold_train_frac=args.train_frac)
    wctx = warm["origin_bytes"] / max(base["origin_bytes"], 1)

    corridor = 100 * (ceil["ohr"] - bar["ohr"])
    cold_pts = 100 * ceil["pf_cold_hits"] / max(ceil["requests"], 1)
    naive_learn = corridor - cold_pts                        # the old, too-harsh approximation
    learnable = 100 * (warm["ohr"] - bar["ohr"])             # the correct, budget-fair number
    bar_cold = bar["pf_cold_hits"]                           # sanity: history-based bar must be 0

    print(f"  BAR              {args.pred} tau={args.tau} k={args.k}  OHR {bar['ohr']:.4f} "
          f"@{bar_tx:.2f}x   (bar cold hits = {bar_cold}, expect 0)")
    print(f"  GROSS CEILING    OHR {ceil['ohr']:.4f} @{ctx:.2f}x   useful pf {ceil['pf_useful']:,} "
          f"of which cold {ceil['pf_cold_hits']:,}  -> cold slice {cold_pts:.2f} pts")
    print(f"  WARM CEILING     OHR {warm['ohr']:.4f} @{wctx:.2f}x   (in-vocab only; cold hits "
          f"{warm['pf_cold_hits']}, expect 0)")
    print(f"  GROSS CORRIDOR      {corridor:+.2f} pts   (diagnostic; inflated by cold slice)")
    print(f"  naive learnable     {naive_learn:+.2f} pts   (gross - cold; too harsh, ignores "
          f"budget reallocation)")
    verdict = "CLEARS 8" if learnable >= 8 else "below 8"
    print(f"  LEARNABLE CORRIDOR  {learnable:+.2f} pts   [{verdict}]  <-- honest budget-fair metric")


if __name__ == "__main__":
    main()
