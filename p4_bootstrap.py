#!/usr/bin/env python3
"""
p4_bootstrap.py -- paired block-bootstrap 95% CI for the LEARNABLE corridor
(warm-restricted-Prescient OHR - tuned-bar OHR), the honest metric.

Why block, not iid: cache hit/miss labels are temporally autocorrelated (a hit depends on the
cache state built by prior requests), so an iid request bootstrap understates variance. We split
the POST-WARMUP trace into contiguous blocks and resample BLOCKS with replacement (moving-block
bootstrap, the guide's "10k resamples over trace windows"). The corridor is a paired quantity:
diff_i = warm_hit_i - bar_hit_i in {-1,0,1} on the SAME requests, so both arms are resampled
together and the CI reflects the paired difference, not two independent OHRs.

Reports point estimate, 95% CI, SE, and whether the CI lower bound clears the pre-registered
8-pt bar. A live trace is only "reliably live" if the LOWER bound >= 8.

Example (wiki, bar config from the strong-bar log):
    python p4_bootstrap.py --trace data/wiki_2019t.oracleGeneral --pred markov2 --tau 0.06 --k 1
"""
from __future__ import annotations

import argparse

import numpy as np

from p4_coldsplit import PREDS
from p4_prefetch import PFCache, Prescient
from p4_sweep import KS, prep


def main():
    ap = argparse.ArgumentParser(description="block-bootstrap CI for the learnable corridor")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--limit", type=int, default=2_000_000)
    ap.add_argument("--cache-frac", type=float, default=0.01)
    ap.add_argument("--pred", choices=sorted(PREDS), required=True,
                    help="winning bar predictor from the strong-bar log")
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--blocks", type=int, default=1000, help="contiguous blocks to resample")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    trace, cap, pos, szs = prep(args.trace, args.limit, args.cache_frac)

    pf = PREDS[args.pred](trace, train_frac=args.train_frac, window=args.window,
                          k=args.k, tau=args.tau)
    bar = PFCache(cap, "s3fifo", pf, positions=pos, sizes=szs).run(
        trace, cold_train_frac=args.train_frac, return_hits=True)
    rate = bar["prefetch_bytes"] / max(trace["n"], 1)

    cut = int(trace["n"] * args.train_frac)
    tvocab = set(int(x) for x in trace["obj_id"][:cut])
    warm = PFCache(cap, "s3fifo", Prescient(trace, k=max(KS), vocab=tvocab), positions=pos,
                   sizes=szs, pf_byte_rate=rate).run(
        trace, cold_train_frac=args.train_frac, return_hits=True)

    hb, hw = bar["hits_series"], warm["hits_series"]
    assert hb is not None and hw is not None and len(hb) == len(hw), "hit series misaligned"
    diff = hw.astype(np.float64) - hb.astype(np.float64)     # per-request corridor, in {-1,0,1}
    point = 100.0 * diff.mean()

    B, R = args.blocks, len(diff)
    L = R // B
    block_means = diff[:B * L].reshape(B, L).mean(axis=1)    # per-block corridor (fraction)
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, B, size=(args.resamples, B))
    boot = 100.0 * block_means[idx].mean(axis=1)             # resampled corridors (pts)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    se = boot.std(ddof=1)

    if lo >= 8:
        verdict = "CI LOWER BOUND >= 8 -- reliably LIVE"
    elif hi >= 8:
        verdict = "CI STRADDLES 8 -- not reliably distinguishable from the bar"
    else:
        verdict = "CI ENTIRELY BELOW 8 -- reliably DEAD"

    print(f"  {args.trace}")
    print(f"  BAR OHR {bar['ohr']:.4f}   WARM CEILING OHR {warm['ohr']:.4f}   "
          f"reqs {R:,}   blocks {B} x {L}   resamples {args.resamples:,}")
    print(f"  LEARNABLE CORRIDOR  {point:+.2f} pts   95% CI [{lo:+.2f}, {hi:+.2f}]   SE {se:.2f} pts")
    print(f"  vs 8-pt bar: {verdict}")


if __name__ == "__main__":
    main()
