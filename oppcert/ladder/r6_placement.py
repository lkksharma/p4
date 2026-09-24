#!/usr/bin/env python3
"""oppcert/ladder/r6_placement.py -- the JIT-REALIZABILITY gate: the final rung of the capture ladder.

PAPER: Stage 2 rung r6, causal placement (internal name: F7). Oracle arm reproduces r4 (+9.69 / +11.69,
construction check); causal model arm lands -40.06 / -26.01 at precision 0.062 / 0.054.

Where the ladder stands:
  * F5        : wide emission + clairvoyant JIT + clairvoyant selection = +10 to +11 (the corridor
                IS reachable in principle).
  * Policy 1  : wide emission + AT-EMISSION captures <=6% of it, even with a perfect forecaster using
                F5's OWN selection -> JIT insertion is the missing ingredient (clean, audited).
  * Open      : is causal USE-LAG prediction accurate enough to PLACE the JIT insertion, or does it
                overshoot (LATE) the way HJS-L's k=1 defer did (55% LATE on real traces)?

F7 isolates exactly that and nothing else. It keeps F5's clairvoyant coverage AND selection -- it
funds ONLY objects that will actually be used again (the "which" is guaranteed right) -- and varies
only WHEN the fetch is inserted:

    timing=oracle : wake = true_next_use - lead          (= F5 itself; the JIT ceiling / construction check)
    timing=model  : wake = emission + hazard_lag(gamma) - lead   (the causal use-lag forecast -- THE test)
    timing=point  : wake = emission + hazard_median - lead       (median only; does the shape matter?)

So oracle-vs-model is a PURE timing-accuracy comparison: same objects, same budget, only the wake
clock differs. LATE = the wake landed AFTER the use (a total loss); on-time = before it.

PRE-REGISTERED GATE: timing=model captures >= 50% of F5's corridor, bootstrap CI excluding zero, on
a live trace. Below 50% the on-time rate is too low for a deployable scheduler.
  PASS -> causal JIT is viable -> build the full DEFER-for-wide policy (the positive capture method).
  FAIL -> the corridor is reachable only by CLAIRVOYANT timing; no causal policy captures it (the
          clean negative -- the measurement half of the paper carries it).

    python -m oppcert.ladder.r6_placement --trace data/wiki_2019t.oracleGeneral --pred markov2 --tau 0.05 --k 1 \
        --wide-k 32 --wide-top-m 32 --haz haz/wiki.npz
    python -m oppcert.ladder.r6_placement --selftest
"""
from __future__ import annotations

import argparse
import heapq
import time

import numpy as np

from oppcert.sim.trace import NEVER
from oppcert.instrument.cold_split import PREDS
from oppcert.ladder.r2_arbitration import capture_ctx, conf_lookup
from oppcert.ladder.r4_coverage import CoverGatedPrescient, build_coverable
from oppcert.ladder.r6_hazard_model import (LAG_EDGES, N_CONF, N_FREQ, N_REC, conf_bin, freq_bin, recency_bin,
                       run as haz_run)
from oppcert.sim.prefetch import PFCache, oracle_next
from oppcert.instrument.gates import KS, prep


def block_ci(diff, blocks, resamples, seed):
    R = len(diff); L = R // blocks
    bm = diff[:blocks * L].reshape(blocks, L).mean(axis=1)
    rng = np.random.default_rng(seed)
    boot = 100.0 * bm[rng.integers(0, blocks, size=(resamples, blocks))].mean(axis=1)
    return np.percentile(boot, [2.5, 97.5])


def _quantile(row, g):
    """Lag at the g-quantile of the conditional (used) lag distribution, at the bin's LOWER edge
    (same convention as HJS-L: log bins are wide and uses cluster near the low end, so the midpoint
    systematically overshoots -> LATE)."""
    f = row[:16]
    tot = f.sum()
    if tot <= 0:
        return float(LAG_EDGES[0])
    cdf = np.cumsum(f / tot)
    return float(LAG_EDGES[min(int(np.searchsorted(cdf, g)), 15)])


class CausalTimedCover:
    """F5's clairvoyant coverage + selection, with causal-vs-oracle TIMING. Funds only objects that
    WILL be used again (clairvoyant 'which'), inserts at a wake time set by the timing mode. The
    single free variable across arms is the wake clock -> oracle-vs-model isolates timing accuracy."""
    name = "f7"

    def __init__(self, base, pos, szs, haz, rate, budget, timing="model", gamma=0.5, lead=1,
                 max_wait=200_000, maxpool=4096, verbose=False, n=0, vstep=200_000):
        self.base, self.pos, self.szs = base, pos, szs
        self.rate, self.budget, self.timing = rate, budget, timing
        self.gamma, self.lead, self.max_wait = gamma, lead, max_wait
        self.maxpool = maxpool
        self.verbose, self.n, self.vstep, self.t_start = verbose, n, vstep, None
        hist = haz["hist"]
        self.woff = np.zeros((N_REC, N_FREQ, N_CONF))     # gamma-quantile lag per (rec,freq,conf) bin
        self.wmed = np.zeros((N_REC, N_FREQ, N_CONF))     # median lag, for the 'point' arm
        for a in range(N_REC):
            for b in range(N_FREQ):
                for d in range(N_CONF):
                    self.woff[a, b, d] = _quantile(hist[a, b, d], gamma)
                    self.wmed[a, b, d] = _quantile(hist[a, b, d], 0.5)
        self.last_seen, self.freq = {}, {}
        self.pending, self.heap = {}, []                  # obj -> (size, true_use, emit_time); heap (wake, obj)
        self.tokens = self.spent = 0.0
        self.on_time = self.late = 0
        self.wake_over = self.wake_early = 0               # forecast-overshoot instrumentation

    def reset(self):
        self.last_seen, self.freq = {}, {}
        self.pending, self.heap = {}, []
        self.tokens = self.spent = 0.0
        self.on_time = self.late = 0
        self.wake_over = self.wake_early = 0
        self.t_start = time.time()
        r = getattr(self.base, "reset", None)
        if r:
            r()

    def suggest(self, o, cached, i):
        self.tokens += self.rate

        if self.verbose and i and i % self.vstep == 0:
            import sys
            el = time.time() - (self.t_start or time.time())
            rt = i / el if el > 0 else 0.0
            eta = (self.n - i) / rt / 60.0 if rt > 0 else 0.0
            tot = max(self.on_time + self.late, 1)
            print(f"    [{self.timing:6s}] {i:>9,}/{self.n:,} ({i/max(self.n,1):4.0%}) | "
                  f"LATE {self.late/tot:4.0%} | pool {len(self.heap):>6,} | "
                  f"spent {self.spent/max(self.budget,1):.2f}x | "
                  f"{rt:>5.0f} req/s | ETA {eta:4.0f}m", file=sys.stderr, flush=True)

        # (1) harvest wide emissions; CLAIRVOYANT selection (fund iff used again) + timed wake
        ctx = capture_ctx(self.base)
        objs = self.base.suggest(o, cached, i)
        if objs:
            cmap = conf_lookup(self.base, ctx, o)
            tau = getattr(self.base, "tau", 0.05)
            for x in objs:
                x = int(x)
                if x in self.pending or x in cached:
                    continue
                s = self.szs.get(x)
                if s is None or s > self.budget:
                    continue
                nx = oracle_next(self.pos, x, i)
                if nx >= int(NEVER):                       # clairvoyant selection: never fund never-used
                    continue
                if self.timing == "oracle":
                    wake = int(nx) - self.lead             # = F5: fetch just before the true use
                else:
                    rb = recency_bin(i - self.last_seen[x] if x in self.last_seen else -1)
                    fb = freq_bin(self.freq.get(x, 0))
                    cb = conf_bin(cmap.get(x, tau))
                    off = self.wmed[rb, fb, cb] if self.timing == "point" else self.woff[rb, fb, cb]
                    wake = int(i + off - self.lead)        # causal: fetch at predicted use time
                wake = max(wake, i)
                if wake > i + self.max_wait:               # discard if wake is too far in the future
                    continue
                # de-confounding instrumentation (ZERO funding impact): did the FORECAST wake itself
                # land after the true use? At low gamma nearly every wake is early, so a high LATE that
                # is NOT matched by a high overshoot rate is bandwidth-queuing under the bar-sized token
                # bucket, NOT causal-timing error -- the paper must not conflate the two.
                if wake > int(nx):
                    self.wake_over += 1
                else:
                    self.wake_early += 1
                self.pending[x] = (s, int(nx), i)          # track emit_time for staleness checks
                heapq.heappush(self.heap, (wake, x))
        self.last_seen[o] = i
        self.freq[o] = self.freq.get(o, 0) + 1

        # (2) collect due candidates (wake reached), fund smallest-first within the rate token bucket
        due = []
        while self.heap and self.heap[0][0] <= i:
            _, x = heapq.heappop(self.heap)
            info = self.pending.get(x)
            if info is None:
                continue                                   # already handled
            s, nx, emit_t = info
            if x in cached:                                # demand-filled while pending -> cancel
                self.pending.pop(x, None); continue
            if i - emit_t > self.max_wait:                 # stale: waited too long -> discard
                self.pending.pop(x, None); continue
            due.append((s, x, nx, emit_t))
        due.sort()                                         # smallest-first (head-of-line safety)
        out = []
        for idx, (s, x, nx, emit_t) in enumerate(due):
            if self.spent + s > self.budget:
                self.pending.pop(x, None)                  # will never afford this -> discard
                continue
            if self.tokens < s:
                # Can't afford this tick. Re-push with a bounded wait.
                wait = int((s - self.tokens) / self.rate) + 1 if self.rate > 0 else 1_000_000
                if i + wait - emit_t > self.max_wait:      # would exceed max_wait -> discard
                    self.pending.pop(x, None)
                else:
                    heapq.heappush(self.heap, (i + wait, x))
                for s2, x2, nx2, et2 in due[idx+1:]:
                    if self.spent + s2 > self.budget:
                        self.pending.pop(x2, None)
                    else:
                        w2 = int((s2 - self.tokens) / self.rate) + 1 if self.rate > 0 else 1_000_000
                        if i + w2 - et2 > self.max_wait:
                            self.pending.pop(x2, None)
                        else:
                            heapq.heappush(self.heap, (i + w2, x2))
                break
            
            self.tokens -= s; self.spent += s
            self.pending.pop(x, None)
            if i <= nx:                                    # inserted BEFORE the use -> on-time
                self.on_time += 1
            else:                                          # inserted AFTER the use -> LATE (a miss)
                self.late += 1
            out.append(x)

        # (3) BOUND THE POOL -- causal arms only; run() passes maxpool=0 for the oracle arm.
        # The oracle arm is the construction check / the gate's DENOMINATOR and must be exact:
        # pruning it drops future-waking candidates it WOULD later fund (tokens are plentiful in
        # the out-of-sample half), freezing spend. Measured on wiki: a size-keyed prune froze
        # oracle spend at 0.52x and read capture -60% of F5 vs +93% unpruned. It never hung
        # unpruned (wakes spread at true use times keep its due-set tiny), so it needs no cap.
        # For the causal arms keep the maxpool SOONEST-WAKE entries (the raw heap key): funding is
        # wake-gated, so this preserves the funding frontier; re-pushed entries carry
        # affordability-ordered wakes, and the far-wake tail is displaced by fresher sooner
        # candidates before it could ever be funded. NOT keyed by size (drops the soonest-due
        # large candidates -- the measured oracle-corruption bug above) and NOT by true-use nx
        # (clairvoyance leak into the causal arm). Hysteresis so the O(pool) trim amortizes
        # instead of firing every request once the pool saturates. NOTE the pool explodes at the
        # train/eval boundary (i ~ n*train_frac): out-of-sample emissions have farther next-uses.
        if self.maxpool > 0 and len(self.heap) > self.maxpool + max(self.maxpool // 4, 32):
            kept = heapq.nsmallest(self.maxpool, self.heap)
            self.heap = [e for e in kept if e[1] in self.pending]
            heapq.heapify(self.heap)
            live = {x for _, x in self.heap}
            self.pending = {x: v for x, v in self.pending.items() if x in live}
        return out


def run(trace, cap, pos, szs, haz, args):
    tf = args.train_frac
    n = trace["n"]
    kw = args.wide_k

    def mk():
        return PREDS[args.pred](trace, train_frac=tf, window=args.window, k=args.k, tau=args.tau)

    def mk_wide():
        p = PREDS[args.pred](trace, train_frac=tf, window=args.window, k=args.k, tau=args.tau,
                             top_m=args.wide_top_m)
        p.k, p.tau = kw, args.wide_tau
        return p

    if args.verbose:
        import sys
        print("  [prelude] base + bar replays (silent, a few minutes)...",
              file=sys.stderr, flush=True)
    base = PFCache(cap, "s3fifo", None, positions=pos, sizes=szs).run(trace)
    bar = PFCache(cap, "s3fifo", mk(), positions=pos, sizes=szs).run(
        trace, cold_train_frac=tf, return_hits=True)
    rate = bar["prefetch_bytes"] / max(n, 1)
    budget = bar["prefetch_bytes"]

    # F5 ceiling (wide + clairvoyant JIT) -- the denominator.
    if args.verbose:
        import sys
        print(f"  [prelude] coverable set + F5 ceiling replay (wide k={kw}) -- the SLOW, SILENT "
              f"part, ~10-25 min on 2M reqs; not a hang...", file=sys.stderr, flush=True)
    coverable, n_cov = build_coverable(mk_wide(), trace, kw, args.wide_tau)
    f5 = PFCache(cap, "s3fifo", CoverGatedPrescient(trace, coverable, k=max(KS), lookahead=2000),
                 positions=pos, sizes=szs, pf_byte_rate=rate).run(
        trace, cold_train_frac=tf, progress=("F5-ceiling" if args.tqdm else None))
    f5_corr = 100.0 * (f5["ohr"] - bar["ohr"])

    print(f"\n  {args.trace}   [F7 JIT-realizability | k_wide={kw} gamma={args.gamma}]")
    print(f"  BAR       OHR {bar['ohr']:.4f}   pf {bar['pf_issued']:,} (prec {bar['pf_precision']:.3f})")
    print(f"  F5 CEIL   OHR {f5['ohr']:.4f}   corridor {f5_corr:+.2f} pts   (coverage {n_cov/n:.1%}) "
          f"<- the denominator (wide + PERFECT JIT)")
    print(f"  {'-'*78}")

    results = {}
    for timing in args.timings:
        if args.verbose:
            import sys
            print(f"  >> timing arm '{timing}' starting ({n:,} reqs)...", file=sys.stderr, flush=True)
        sch = CausalTimedCover(mk_wide(), pos, szs, haz, rate, budget, timing=timing,
                               gamma=args.gamma, lead=args.lead, max_wait=args.max_wait,
                               # oracle = the construction check: must be exact, runs unbounded (it
                               # never hung -- only the causal arms' due-storm needs the cap)
                               maxpool=(0 if timing == "oracle" else args.max_pool),
                               verbose=args.verbose, n=n)
        f7 = PFCache(cap, "s3fifo", sch, positions=pos, sizes=szs, pf_byte_rate=None).run(
            trace, cold_train_frac=tf, return_hits=True,
            progress=(f"F7:{timing}" if args.tqdm else None))
        f7_corr = 100.0 * (f7["ohr"] - bar["ohr"])
        capf5 = f7_corr / f5_corr if f5_corr > 0 else float("nan")
        pfb = f7["prefetch_bytes"] / max(bar["prefetch_bytes"], 1)
        tot = max(sch.on_time + sch.late, 1)
        d = f7["hits_series"].astype(np.float64) - bar["hits_series"].astype(np.float64)
        lo, hi = block_ci(d, args.blocks, args.resamples, args.seed)

        inv = []
        # A frozen predictor cannot prefetch an object it never saw in training, so a cold hit is
        # a harness bug and must fail. An ONLINE predictor legitimately can (prereg/online_predictor.md
        # 7a): reaching cold objects is the capability, not an accounting error. --allow-cold
        # therefore reports the slice instead of failing, and is opt-in so every pre-registered
        # frozen arm keeps the original check untouched.
        if f7["pf_cold_hits"] != 0:
            if args.allow_cold:
                cold_pts = 100.0 * f7["pf_cold_hits"] / max(f7["n_scored"], 1) \
                    if "n_scored" in f7 else float("nan")
                print(f"  [cold] {f7['pf_cold_hits']:,} cold prefetch hits permitted "
                      f"(--allow-cold); this arm is scored GROSS, not warm-restricted"
                      + (f", cold slice {cold_pts:.2f} pts" if cold_pts == cold_pts else ""))
            else:
                inv.append(f"cold hits {f7['pf_cold_hits']} != 0")
        if pfb > 1.02:
            inv.append(f"prefetch bytes {pfb:.2f}x of bar -- NOT iso-bandwidth")
        if timing == "oracle" and not (np.isfinite(capf5) and capf5 >= 0.75):
            inv.append(f"CONSTRUCTION CHECK: oracle-timing = {capf5:.0%} of F5 (expect ~100%) -- "
                       f"harness broken; NOTHING in this run is reportable")
        okinv = "all pass" if not inv else "FAIL"

        tagd = "  (= F5 ceiling / construction check)" if timing == "oracle" else \
               "  (DIAGNOSTIC)" if timing == "point" else ""
        print(f"  [{timing:6s}] OHR {f7['ohr']:.4f}   pf {f7['pf_issued']:,} "
              f"useful {f7['pf_useful']:,} (prec {f7['pf_precision']:.3f})   spent "
              f"{sch.spent/max(budget,1):.2f}x{tagd}")
        wtot = max(sch.wake_over + sch.wake_early, 1)
        overshoot = sch.wake_over / wtot
        print(f"           on-time {sch.on_time:,} ({sch.on_time/tot:.1%})   LATE {sch.late:,} "
              f"({sch.late/tot:.1%})   forecast-overshoot {overshoot:.1%} (wake > true use at emission)")
        print(f"           corridor {f7_corr:+.2f} [{lo:+.2f},{hi:+.2f}]   capture {capf5:.1%} of F5 "
              f"| pf bytes {pfb:.2f}x   inv:{okinv}")
        if timing == "model" and sch.late / tot > 0.25:
            if sch.late / tot - overshoot > 0.25:
                print("           -> LATE >> forecast-overshoot: the LATE is BANDWIDTH-QUEUING (the "
                      "bar-sized bucket cannot drain the wide net's bunched demand), NOT timing error. "
                      "Do NOT read this as 'causal timing is the wall' -- report the confound.")
            else:
                print("           -> LATE tracks forecast-overshoot: the causal wake genuinely "
                      "overshoots the use. Lower --gamma (fetch earlier) and re-check.")
        for m in inv:
            print(f"           !! {m}")
        results[timing] = (f7_corr, lo, hi, capf5, sch.on_time / tot, overshoot)

    # verdict -- the clean comparison is model-vs-ORACLE (same F7 mechanics, only the wake clock
    # differs), which isolates timing accuracy. F5 is reported for continuity but F7-oracle is the
    # correct same-mechanics JIT ceiling (it need not equal F5 -- different object selection).
    if "model" in results and "oracle" in results:
        mc, mlo, mhi, mcap_f5, montime, movershoot = results["model"]
        oc_corr, _, _, oc_capf5, _, _ = results["oracle"]
        mcap = mc / oc_corr if oc_corr > 0 else float("nan")   # model capture of the JIT ceiling
        print(f"  {'-'*78}")
        print(f"  CONSTRUCTION    oracle-timing = {oc_capf5:.0%} of F5 (the same-mechanics JIT ceiling)")
        print(f"  CAUSAL TIMING   model = {mcap:.0%} of the JIT ceiling ({mcap_f5:.0%} of F5) | "
              f"on-time {montime:.0%} | forecast-overshoot {movershoot:.0%}")
        mlate = 1.0 - montime
        if mlate > movershoot:
            print(f"  CONFOUND        LATE {mlate:.0%} vs forecast-overshoot {movershoot:.0%} -- the "
                  f"gap is bandwidth-queuing under the bar-sized bucket, NOT causal-timing error")
        else:
            print(f"  CONFOUND        LATE {mlate:.0%} <= forecast-overshoot {movershoot:.0%} -- "
                  f"overshot candidates were censored (cancelled/pruned/never afforded) before "
                  f"funding; denominators differ (emissions vs funded)")
        if not (np.isfinite(oc_capf5) and oc_capf5 >= 0.75):
            pass                                   # verdict handled below: INVALID, no gate read
        elif montime >= 0.75 and mcap < 0.50:
            print("  NOTE  on-time high but capture low -> fetches land BEFORE the use but the object is")
            print("        EVICTED before it (fetched too early). On real traces S(100)=1.0 this should not")
            print("        occur; if it does it is a survival problem, not timing overshoot -> raise --gamma.")
        if not (np.isfinite(oc_capf5) and oc_capf5 >= 0.75):
            v = ("INVALID -- the oracle arm did not reproduce F5 (construction check failed). "
                 "Harness bug: fix and re-run before reading ANY verdict from this output.")
        elif mlo > 0 and mcap >= 0.50:
            v = ("PASS -- causal JIT viable (>=50% of the JIT ceiling, CI>0). BUILD the full "
                 "DEFER-for-wide policy: the positive capture method.")
        elif mlo > 0:
            v = (f"POSITIVE but below the 50% bar ({mcap:.0%}). Sweep --gamma; if it tops out here, "
                 "causal timing is partial -- a weak positive, not the headline.")
        else:
            v = ("FAIL -- causal use-lag timing cannot place the JIT insertion (CI includes 0). The "
                 "corridor needs CLAIRVOYANT timing; no causal policy captures it -- the measurement "
                 "half carries the paper, plus the Necessity-Ladder triple-kill.")
        print(f"  GATE (>=50% ceiling)  {v}")
    print()
    return results


def selftest():
    class A:
        trace = "SYNTH"; limit = 60000; cache_frac = 0.01; pred = "markov1"; tau = 0.05; k = 1
        window = 16; train_frac = 0.5; haz_frac = 0.75; horizon = 5000; out = "/tmp/_f7_haz.npz"
        wide_k = 16; wide_tau = 0.0; wide_top_m = 16; timings = ["oracle", "model", "point"]
        gamma = 0.25; lead = 1; max_wait = 20000; max_pool = 4096; verbose = False; tqdm = False
        blocks = 100; resamples = 500; seed = 0
    a = A()
    trace, cap, pos, szs = prep("SYNTH", a.limit, a.cache_frac)
    haz_run(trace, cap, pos, szs, a)                       # writes /tmp/_f7_haz.npz
    haz = np.load(a.out)
    run(trace, cap, pos, szs, haz, a)
    print("  [oppcert.ladder.r6_placement selftest] completed without error\n")


def main():
    ap = argparse.ArgumentParser(description="F7 JIT-realizability gate (final capture-ladder rung)")
    ap.add_argument("--trace")
    ap.add_argument("--limit", type=int, default=2_000_000)
    ap.add_argument("--cache-frac", type=float, default=0.01)
    ap.add_argument("--pred", choices=sorted(PREDS))
    ap.add_argument("--tau", type=float)
    ap.add_argument("--k", type=int)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--haz", help="hazard+survival .npz from oppcert/ladder/r6_hazard_model.py (the use-lag model)")
    ap.add_argument("--wide-k", type=int, default=16, help="emission fanout (the F5-cleared width)")
    ap.add_argument("--wide-tau", type=float, default=0.0)
    ap.add_argument("--wide-top-m", type=int, default=16)
    ap.add_argument("--timings", default="oracle,model,point",
                    help="oracle (=F5 ceiling/construction check), model (causal use-lag -- the test), "
                         "point (median only)")
    ap.add_argument("--gamma", type=float, default=0.25,
                    help="use-lag quantile the causal wake targets. LOW = fetch EARLY (safe against "
                         "LATE, costs occupancy); HIGH = fetch late and miss. THE knob to sweep.")
    ap.add_argument("--lead", type=int, default=1)
    ap.add_argument("--max-wait", type=int, default=200_000)
    ap.add_argument("--max-pool", type=int, default=4096,
                    help="cap on pending candidates in the CAUSAL arms only (model/point) -- the "
                         "oracle arm always runs unbounded: it is the construction check and must "
                         "be exact (a capped oracle froze spend at 0.52x and read -60%% of F5). "
                         "Keyed by SOONEST WAKE (the funding frontier), never by size. Sweep it "
                         "(e.g. 4096 vs 16384) and quote a verdict only where capture and "
                         "on-time/LATE are cap-stable.")
    ap.add_argument("--verbose", action="store_true",
                    help="clean newline progress every 200k reqs (%% done, LATE rate, req/s, ETA). "
                         "Flush-safe and flicker-free -- use this for parallel runs.")
    ap.add_argument("--tqdm", action="store_true",
                    help="live tqdm bar per replay. Avoid with parallel runs (bars flicker).")
    ap.add_argument("--allow-cold", action="store_true",
                    help="permit and report cold prefetch hits instead of failing the "
                         "invariant. Required for online-updating predictors, which can name "
                         "objects first seen after the training cut (prereg/online_predictor.md 7a). "
                         "Never use with a frozen predictor: there a cold hit is a real bug.")
    ap.add_argument("--blocks", type=int, default=1000)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not (args.trace and args.pred and args.tau is not None and args.k is not None and args.haz):
        ap.error("need --trace --pred --tau --k --haz")
    args.timings = [t for t in str(args.timings).split(",") if t.strip()]
    trace, cap, pos, szs = prep(args.trace, args.limit, args.cache_frac)
    haz = np.load(args.haz)
    run(trace, cap, pos, szs, haz, args)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
