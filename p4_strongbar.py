#!/usr/bin/env python3
"""
p4_strongbar.py -- the STRONGER decoupled bars the corridor must survive before we call it real.

Gate A's bar was Markov-1/2. The standing kill-risk: a better *decoupled* predictor
(higher-order Markov tonight, an LSTM tomorrow) eats the corridor, in which case the
"+26.5" was measuring Markov's weakness, not timing headroom. This module offers those
predictors to the SAME gate_a machinery (same iso-BW ceiling convention, same Pareto
check, same >=8 verdict line), with Markov-1/2 always included -- the bar is the max
over every predictor on offer, so this run can only RAISE the bar, never flatter us.

Tau grid note: LSTM confidences are softmax probabilities over a huge vocabulary, so
useful taus sit LOWER than Markov count-ratios. The default grid here extends down to
0.01 and is applied to every predictor equally (more configs for the bar to win with).

Day 0 (no GPU needed):
    python p4_strongbar.py --trace data/wiki_2019t.oracleGeneral --limit 2000000
Day 1 (after p4_lstm_train.py dumps predictions):
    python p4_strongbar.py --trace data/wiki_2019t.oracleGeneral --limit 2000000 \
        --lstm-preds preds/wiki_2019t.lstm.npz
"""
from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict

import numpy as np

from p4_prefetch import Markov1, Markov2, _MarkovBase, _rank
from p4_sweep import gate_a, prep, selftest

# 0.06-0.09 included permanently: on cluster26 the winning bar configs sat exactly there
# (markov2 tau=0.06 @1.14x, markov3 tau=0.05-0.06), flipping a provisional LIVE to DEAD.
# Every live-trace verdict must clear this grid, not the coarse one.
STRONG_TAUS = (0.01, 0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.20, 0.35, 0.50)


class Markov3(_MarkovBase):
    """Third-order association with 3->2->1 backoff (PPM-style). Context = the last three
    requests; sharper than Markov-2 where fixed multi-step sequences exist, sparser
    everywhere else -- hence the backoff chain. Built on the trace PREFIX only.
    Stateful (tracks prev two requests), so PFCache.run()'s reset() hook is required."""
    name = "markov3"

    def __init__(self, trace, train_frac=0.5, window=16, top_m=16, k=2, tau=0.05):
        self.k, self.tau = k, tau
        ids = trace["obj_id"]
        cut = int(len(ids) * train_frac)
        s3, s2, s1 = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
        for i in range(2, cut):
            c3 = (int(ids[i - 2]), int(ids[i - 1]), int(ids[i]))
            for j in range(i + 1, min(i + 1 + window, cut)):
                x = int(ids[j])
                s3[c3][x] += 1
                s2[c3[1:]][x] += 1
                s1[c3[2]][x] += 1
        self.t3, self.t2, self.t1 = _rank(s3, top_m), _rank(s2, top_m), _rank(s1, top_m)
        self.table = self.t3                     # so callers can report table size
        self.p2 = self.p1 = None

    def reset(self):
        self.p2 = self.p1 = None

    def suggest(self, o, cached, i):
        o = int(o)
        cand = self.t3.get((self.p2, self.p1, o)) if self.p2 is not None else None
        if cand is None and self.p1 is not None:
            cand = self.t2.get((self.p1, o))
        if cand is None:
            cand = self.t1.get(o, ())
        self.p2, self.p1 = self.p1, o
        return self._pick(cand, cached)


class LSTMTopK(_MarkovBase):
    """Decoupled LSTM next-item bar, replayed from PRECOMPUTED per-position predictions.

    The predictor is causal and cache-independent (its input is the request stream only),
    so p4_lstm_train.py runs the model over the trace once on GPU and dumps, for every
    position i, the top-M candidates for request i+1 with their softmax probabilities.
    Replay then just reads row i -- which makes the full tau x k sweep as cheap as Markov's.
    cand rows are probability-descending; -1 entries (UNK / padding) are skipped."""
    name = "lstm"

    def __init__(self, npz_path, n_expected, k=2, tau=0.05):
        d = np.load(npz_path)
        self.cand, self.prob = d["cand"], d["prob"]
        assert self.cand.shape[0] == n_expected, (
            f"{npz_path} has {self.cand.shape[0]} rows but trace has {n_expected} -- "
            f"regenerate with the same --limit")
        self.k, self.tau = k, tau
        self.table = {}                          # size reporting: not table-based

    def suggest(self, o, cached, i):
        out = []
        for x, conf in zip(self.cand[i], self.prob[i]):
            if conf < self.tau:
                break                            # rows are confidence-descending
            x = int(x)
            if x >= 0 and x not in cached:
                out.append(x)
                if len(out) >= self.k:
                    break
        return out


def main():
    ap = argparse.ArgumentParser(description="Strong decoupled bars through the Gate A machinery")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--limit", type=int, default=2_000_000)
    ap.add_argument("--cache-frac", type=float, default=0.01)
    ap.add_argument("--max-traffic", type=float, default=1.15)
    ap.add_argument("--taus", type=lambda s: tuple(float(x) for x in s.split(",")),
                    default=STRONG_TAUS)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--no-markov3", action="store_true",
                    help="skip the Markov-3 arm (e.g. Day 1 when only adding the LSTM)")
    ap.add_argument("--lstm-preds", default=None,
                    help="npz from p4_lstm_train.py; adds the LSTM arm")
    args = ap.parse_args()

    selftest()
    trace, cap, pos, szs = prep(args.trace, args.limit, args.cache_frac)

    t0 = time.time()
    preds = {}
    for nm, cls in (("markov1", Markov1), ("markov2", Markov2)):
        preds[nm] = cls(trace, train_frac=args.train_frac, window=args.window)
        print(f"  [build] {nm} table: {len(preds[nm].table):,} contexts  ({time.time()-t0:.1f}s)")
    if not args.no_markov3:
        preds["markov3"] = Markov3(trace, train_frac=args.train_frac, window=args.window)
        print(f"  [build] markov3 table: {len(preds['markov3'].table):,} contexts  "
              f"({time.time()-t0:.1f}s)")
    if args.lstm_preds:
        preds["lstm"] = LSTMTopK(args.lstm_preds, n_expected=trace["n"])
        print(f"  [load]  lstm predictions: {args.lstm_preds}")

    print(f"  [note] bar = max over {{{', '.join(preds)}}} x taus{args.taus} x k(1,2,4) "
          f"under {args.max_traffic:.2f}x traffic")
    gate_a(trace, cap, pos, szs, args, preds=preds)


if __name__ == "__main__":
    main()
