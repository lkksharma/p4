#!/usr/bin/env python3
"""
p4_hjsl.py -- Necessity Ladder rung (ii), part B: the HJS-L closed-form prefetch SCHEDULER, and
its capture-fraction evaluation against the F1 timing ceiling.

HJS-L is causal: for each candidate the fixed predictor emits, it reads the hazard model's
predicted use-lag distribution (p4_hazard.py) and the measured eviction-survival curve S(delta),
and schedules the fetch to win the race between the use clock and the eviction clock, under a
shared byte budget. Three levers, all from rung (ii)-A, none clairvoyant:
  * DEFER   : hold a candidate until wake = emission + quantile_gamma(lag) - lead, so it is fetched
              near its predicted use rather than at emission (avoids the eviction the bar suffers).
  * PRICE   : an accumulating token bucket at the bar's byte rate IS the shadow price -- release
              highest expected-utility-per-byte first, only while tokens (and a hard total-byte
              cap) allow. Iso-bandwidth by construction; a Pareto violation is impossible.
  * SURVIVE : utility U = sum_tau p(tau|not-yet-used) * S(tau - elapsed) -- discount a fetch by its
              chance of being evicted before use. Drop candidates whose future-use prob < floor.

CAPTURE (the result): corridor(HJS-L) / corridor(F1), where corridor = OHR - bar OHR and F1 is the
timing-only ceiling on the predictor's own candidates (p4_f1.py). Reporting against F1, not the v3
warm ceiling, is the honest denominator: HJS-L is a scheduler, so it is judged against the best a
scheduler could do on the same candidate stream, not against an oracle that also picks objects.

PRE-REGISTERED PASS BAR: capture >= 25% of F1, bootstrap CI excluding zero, on every live trace
whose F2 gate passed and whose F1 >= 8.

    python p4_hjsl.py --trace data/wiki_2019t.oracleGeneral --pred markov2 --tau 0.05 --k 1 \
        --haz haz/wiki.npz
    python p4_hjsl.py --selftest
"""
from __future__ import annotations

import argparse
import heapq

import numpy as np

from p4_coldsplit import PREDS
from p4_f1 import emit_vocab
from p4_f4 import capture_ctx, conf_lookup
from p4_hazard import CONF_EDGES, LAG_EDGES, NEVER_BIN, SIZE_EDGES, conf_bin, run as haz_run, size_bin
from p4_prefetch import PFCache, Prescient
from p4_sweep import KS, prep


class HJSL:
    """The closed-form hazard-priced JIT scheduler, as a stateful suggest() wrapper."""
    name = "hjsl"

    def __init__(self, base, pos, szs, rate, haz, budget,
                 gamma=0.75, lead_safety=1, floor=0.05, max_wait=200000,
                 defer=True, survive=True):
        self.base, self.pos, self.szs, self.rate = base, pos, szs, rate
        self.budget, self.gamma = budget, gamma
        self.lead_safety, self.floor, self.max_wait = lead_safety, floor, max_wait
        self.defer, self.survive = defer, survive

        self.hist = haz["hist"]                                    # (7,7,17) p(lag|conf,size)
        self.mids = np.sqrt(LAG_EDGES[:-1] * LAG_EDGES[1:])        # (16,) geometric bin midpoints
        self.sg, self.sv = haz["surv_grid"], haz["surv_vals"]      # S(delta) step function
        # wake offset per (conf_bin,size_bin): the gamma-quantile lag (fixed gamma -> precompute)
        self.woff = np.zeros((7, 7))
        for a in range(7):
            for b in range(7):
                self.woff[a, b] = self._quantile(self.hist[a, b], gamma)

        self.pending = {}          # obj -> (size, conf_bin, size_bin, t0, wake)
        self.heap = []             # (wake, obj) lazy-deleted
        self.ready = {}            # obj -> (size, conf_bin, size_bin, t0)
        self.tokens = 0.0
        self.spent = 0.0

    def reset(self):
        self.pending, self.heap, self.ready = {}, [], {}
        self.tokens = self.spent = 0.0
        r = getattr(self.base, "reset", None)
        if r:
            r()

    def _quantile(self, row, g):
        f = row[:16]
        tot = f.sum()
        if tot <= 0:
            return self.mids[-1]
        cdf = np.cumsum(f / tot)
        return float(self.mids[min(int(np.searchsorted(cdf, g)), 15)])

    def _S(self, deltas):
        deltas = np.maximum(deltas, 0)
        idx = np.searchsorted(self.sg, deltas, side="right") - 1
        return np.where(idx >= 0, self.sv[np.clip(idx, 0, len(self.sv) - 1)], 1.0)

    def _utility(self, cb, sb, elapsed):
        """(U, p_future): expected hit credit of fetching NOW, and P(use still to come)."""
        row = self.hist[cb, sb]
        mask = self.mids > elapsed
        resid = row[:16][mask]
        rs = resid.sum()
        never = row[NEVER_BIN]
        p_future = rs / (rs + never) if (rs + never) > 0 else 0.0
        if rs <= 0:
            return 0.0, p_future
        Sd = self._S(self.mids[mask] - elapsed) if self.survive else 1.0
        return float((resid * Sd).sum()), p_future

    def suggest(self, o, cached, i):
        self.tokens += self.rate

        # (1) harvest candidates with confidence, schedule a wake time
        ctx = capture_ctx(self.base)
        objs = self.base.suggest(o, cached, i)
        if objs:
            cmap = conf_lookup(self.base, ctx, o)
            tau = getattr(self.base, "tau", 0.05)
            for x in objs:
                x = int(x)
                if x in self.pending or x in self.ready or x in cached:
                    continue
                s = self.szs.get(x)
                if s is None or s > self.budget:
                    continue
                cb, sb = conf_bin(cmap.get(x, tau)), size_bin(s)
                if self.defer:
                    lead = s / max(self.rate, 1.0) + self.lead_safety
                    wake = max(int(i + self.woff[cb, sb] - lead), i)
                else:
                    wake = i                                       # ablation: fetch at emission
                self.pending[x] = (s, cb, sb, i, wake)
                heapq.heappush(self.heap, (wake, x))

        # (2) move due candidates into the ready set
        while self.heap and self.heap[0][0] <= i:
            _, x = heapq.heappop(self.heap)
            info = self.pending.pop(x, None)
            if info is None:
                continue                                           # stale (already handled)
            s, cb, sb, t0, _ = info
            self.ready[x] = (s, cb, sb, t0)

        if not self.ready or self.tokens < min(v[0] for v in self.ready.values()):
            return []

        # (3) score ready candidates; expire the hopeless ones
        scored, drop = [], []
        for x, (s, cb, sb, t0) in self.ready.items():
            if x in cached:                                        # demand-filled -> cancel
                drop.append(x); continue
            elapsed = i - t0
            if elapsed > self.max_wait:
                drop.append(x); continue
            U, p_future = self._utility(cb, sb, elapsed)
            if p_future < self.floor or U <= 0:
                drop.append(x); continue
            scored.append((U / s, x, s))
        for x in drop:
            self.ready.pop(x, None)
        if not scored:
            return []

        # (4) release highest utility-per-byte first, within tokens AND the hard byte cap
        scored.sort(reverse=True)
        out = []
        for _, x, s in scored:
            if self.tokens < s or self.spent + s > self.budget:
                continue
            self.tokens -= s; self.spent += s
            self.ready.pop(x, None)
            out.append(x)
        out.sort(key=lambda x: self.szs.get(x, 0))                 # smallest-first (HOL safety)
        return out


def block_ci(diff, blocks, resamples, seed):
    R = len(diff); L = R // blocks
    bm = diff[:blocks * L].reshape(blocks, L).mean(axis=1)
    rng = np.random.default_rng(seed)
    boot = 100.0 * bm[rng.integers(0, blocks, size=(resamples, blocks))].mean(axis=1)
    return np.percentile(boot, [2.5, 97.5])


def run(trace, cap, pos, szs, haz, args):
    tf = args.train_frac

    def mk():
        return PREDS[args.pred](trace, train_frac=tf, window=args.window, k=args.k, tau=args.tau)

    base = PFCache(cap, "s3fifo", None, positions=pos, sizes=szs).run(trace)
    bar = PFCache(cap, "s3fifo", mk(), positions=pos, sizes=szs).run(
        trace, cold_train_frac=tf, return_hits=True)
    rate = bar["prefetch_bytes"] / max(trace["n"], 1)
    budget = bar["prefetch_bytes"]

    # F1 timing-only ceiling (the honest denominator) -- vocab oracle, same as p4_f1.py
    emit = emit_vocab(mk(), trace, trace["n"])
    f1 = PFCache(cap, "s3fifo", Prescient(trace, k=max(KS), vocab=emit), positions=pos, sizes=szs,
                 pf_byte_rate=rate).run(trace, cold_train_frac=tf)

    sch = HJSL(mk(), pos, szs, rate, haz, budget, gamma=args.gamma, floor=args.floor,
               max_wait=args.max_wait, defer=(args.defer != "none"),
               survive=(args.survival != "none"))
    hj = PFCache(cap, "s3fifo", sch, positions=pos, sizes=szs, pf_byte_rate=rate).run(
        trace, cold_train_frac=tf, return_hits=True)

    bar_tx = bar["origin_bytes"] / max(base["origin_bytes"], 1)
    hj_tx = hj["origin_bytes"] / max(base["origin_bytes"], 1)
    f1_corr = 100.0 * (f1["ohr"] - bar["ohr"])
    hj_corr = 100.0 * (hj["ohr"] - bar["ohr"])
    capture = hj_corr / f1_corr if f1_corr > 0 else float("nan")

    d = hj["hits_series"].astype(np.float64) - bar["hits_series"].astype(np.float64)
    lo, hi = block_ci(d, args.blocks, args.resamples, args.seed)

    print(f"\n  {args.trace}   [gamma={args.gamma} defer={args.defer} survival={args.survival}]")
    print(f"  BAR       OHR {bar['ohr']:.4f} @{bar_tx:.2f}x")
    print(f"  HJS-L     OHR {hj['ohr']:.4f} @{hj_tx:.2f}x   pf {hj['pf_issued']:,} "
          f"useful {hj['pf_useful']:,} (prec {hj['pf_precision']:.3f})   spent {sch.spent/max(budget,1):.2f}x budget")
    print(f"  F1 CEIL   OHR {f1['ohr']:.4f}")
    print(f"\n  HJS-L CORRIDOR   {hj_corr:+.2f} pts   95% CI [{lo:+.2f}, {hi:+.2f}]")
    print(f"  F1 CORRIDOR      {f1_corr:+.2f} pts")
    print(f"  CAPTURE          {capture:.1%} of F1   (corridor CI [{lo:+.2f},{hi:+.2f}])")

    inv = []
    if hj["pf_cold_hits"] != 0:
        inv.append(f"HJS-L cold hits {hj['pf_cold_hits']} != 0 -- funding OOV objects")
    if hj_tx > bar_tx + 0.02:
        inv.append(f"HJS-L traffic {hj_tx:.2f}x > bar {bar_tx:.2f}x -- NOT iso-bandwidth")
    print("  INVARIANTS       " + ("all pass" if not inv else "FAIL"))
    for m in inv:
        print(f"    !! {m}")

    if f1_corr < 8:
        verdict = "F1 < 8 -- timing-only cannot win here; HJS-L not applicable to this trace"
    elif lo > 0 and capture >= 0.25:
        verdict = "PASS -- capture >= 25% of F1, CI excludes 0"
    elif lo > 0:
        verdict = f"positive but below 25% bar (captured {capture:.1%})"
    else:
        verdict = "FAIL -- corridor CI includes 0"
    print(f"  GATE (>=25% F1)  {verdict}\n")
    return capture


def selftest():
    class A:
        trace = "SYNTH"; limit = 60000; cache_frac = 0.01; pred = "markov1"; tau = 0.05; k = 1
        window = 16; train_frac = 0.5; out = "/tmp/_hjsl_haz.npz"; gamma = 0.75; floor = 0.05
        max_wait = 20000; defer = "hazard"; survival = "km"; blocks = 100; resamples = 500; seed = 0
    a = A()
    trace, cap, pos, szs = prep("SYNTH", a.limit, a.cache_frac)
    haz_run(trace, cap, pos, szs, a)                    # writes /tmp/_hjsl_haz.npz
    haz = np.load(a.out)
    run(trace, cap, pos, szs, haz, a)
    print("  [p4_hjsl selftest] completed without error\n")


def main():
    ap = argparse.ArgumentParser(description="HJS-L scheduler + capture vs F1 (Necessity Ladder rung ii)")
    ap.add_argument("--trace")
    ap.add_argument("--limit", type=int, default=2_000_000)
    ap.add_argument("--cache-frac", type=float, default=0.01)
    ap.add_argument("--pred", choices=sorted(PREDS))
    ap.add_argument("--tau", type=float)
    ap.add_argument("--k", type=int)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--haz", help="hazard+survival .npz from p4_hazard.py")
    ap.add_argument("--gamma", type=float, default=0.75, help="use-lag quantile for the wake time")
    ap.add_argument("--floor", type=float, default=0.05, help="drop candidates with P(future use) below this")
    ap.add_argument("--max-wait", type=int, default=200000)
    ap.add_argument("--defer", choices=("hazard", "none"), default="hazard",
                    help="none = fetch at emission (isolates the deferral lever; should ~= bar)")
    ap.add_argument("--survival", choices=("km", "none"), default="km",
                    help="none = assume S(delta)=1 (isolates the eviction-survival lever)")
    ap.add_argument("--blocks", type=int, default=1000)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not (args.trace and args.pred and args.tau is not None and args.k is not None and args.haz):
        ap.error("need --trace --pred --tau --k --haz")
    trace, cap, pos, szs = prep(args.trace, args.limit, args.cache_frac)
    haz = np.load(args.haz)
    run(trace, cap, pos, szs, haz, args)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
