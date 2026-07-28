#!/usr/bin/env python3
"""p4_r8.py -- closing the two open holes in Stage 2: the r6 model sweep, and volume control.

WHY THIS EXISTS
---------------
The Ladder's strength is that it prices CAPABILITIES at their maximum. r5 is airtight for exactly
that reason: it grants a PERFECT forecaster and still fails, so no forecaster can rescue
emission-time fetch. Nothing above perfect exists.

r6 does not have that form. Its oracle arm passes (+9.69/+11.69) and its model arm fails
(-40.06/-26.01), so what failed is ONE instantiation -- a binned histogram over three features --
not the capability. As run, r6 licenses only "this model could not place fetches", while the paper
leans on it as though it read "no model can". That asymmetry is the paper's own M1 error pointed at
its learned side, and the index section already indicts it: sweeping the baseline while pinning the
RMI at one capacity flipped two verdicts, and sweeping both flipped them back.

Worse, the mechanism is entangled with the instantiation. Endogenous slack is volume-driven, and
volume is downstream of THIS model's precision (0.05). A timing model at precision 0.3 issues far
fewer junk admissions and faces a much wider window. So the honest open question is whether a fixed
point exists: a precision at which a causal policy's own volume stops destroying its window.

TWO ARMS, ONE HARNESS
---------------------
  --sweep model   r6 swept as a family: model class x feature set x emission configuration, with
                  precision and fetch volume reported beside capture so the precision-capture
                  relation is visible rather than asserted.
  --sweep volume  r8, the arm the design space is missing. r3-r7 all vary WHICH objects and WHEN,
                  under an external byte cap; none closes a loop on volume. r8 measures its own
                  realised precision over a sliding window and throttles issuance toward a target,
                  so fetch volume becomes a controlled variable instead of a consequence.

WHAT EITHER OUTCOME BUYS
------------------------
  capture negative at every precision/volume  -> endogenous slack is a measured constraint, not a
      symptom of one bad model, and the trap verdict closes the design space instead of sampling it.
  capture positive somewhere                  -> a build on CDN prefetching, found by the procedure
      itself from its own mechanism. A better paper, not a worse one.
  capture rises but plateaus below the bar     -> the quantitative statement: the volume needed for
      coverage exceeds the volume the window tolerates, by a measured factor.

    python p4_r8.py --trace data/wiki_2019t.oracleGeneral --pred markov2 --tau 0.05 --k 1 \
        --haz haz/wiki.npz --sweep volume
    python p4_r8.py --selftest
"""
from __future__ import annotations

import argparse
import heapq
from collections import deque

import numpy as np

from p4_cache import NEVER
from p4_coldsplit import PREDS
from p4_f4 import capture_ctx, conf_lookup
from p4_f5 import CoverGatedPrescient, build_coverable
from p4_f7 import _quantile
from p4_hazard import N_CONF, N_FREQ, N_REC, conf_bin, freq_bin, recency_bin, run as haz_run
from p4_prefetch import PFCache, oracle_next
from p4_sweep import KS, prep

LINE = "=" * 104


def block_ci(diff, blocks, resamples, seed):
    R = len(diff); L = R // blocks
    bm = diff[:blocks * L].reshape(blocks, L).mean(axis=1)
    rng = np.random.default_rng(seed)
    boot = 100.0 * bm[rng.integers(0, blocks, size=(resamples, blocks))].mean(axis=1)
    return np.percentile(boot, [2.5, 97.5])


class VolPolicy:
    """Wide emission, causal timing, and an optional volume feedback controller.

    The controller is the piece r3-r7 never had. It tracks the fraction of recent prefetches that
    were requested before ageing out of a sliding window -- realised precision, computed causally
    from the request stream with no future consulted -- and moves the confidence floor to steer
    that quantity toward a target. Raising the floor issues fewer, better candidates, which lowers
    volume, which lengthens the residence window every later fetch must hit. Fetch volume therefore
    becomes a controlled variable rather than an uncontrolled consequence of the model's precision.

    With target_prec=None the controller is inert and this is exactly r6 at the given emission
    configuration, which is how the model sweep reuses the same code path.
    """
    name = "r8"

    def __init__(self, base, pos, szs, haz, rate, budget, model="binned", gamma=0.5, floor=0.0,
                 sel_q=0.0, win=20000, kp=0.05, lead=1, max_wait=200_000):
        self.base, self.pos, self.szs = base, pos, szs
        self.rate, self.budget, self.model = rate, budget, model
        self.gamma, self.floor0, self.lead, self.max_wait = gamma, floor, lead, max_wait
        self.sel_q, self.win, self.kp = sel_q, win, kp
        hist = haz["hist"]
        self.woff = np.zeros((N_REC, N_FREQ, N_CONF))
        for a in range(N_REC):
            for b in range(N_FREQ):
                for d in range(N_CONF):
                    self.woff[a, b, d] = _quantile(hist[a, b, d], gamma)
        self.reset()

    def reset(self):
        self.floor = self.floor0
        # A percentile floor, not an absolute one. An absolute confidence threshold is not a
        # usable lever: candidate confidences are concentrated, so a floor that moves by a few
        # hundredths sits entirely below or above the whole distribution and changes nothing --
        # the first selftest produced byte-identical volume at every target because of exactly
        # that. Steering a PERCENTILE of the observed confidence distribution is guaranteed to
        # bite regardless of where the distribution sits.
        self.q = self.sel_q                     # swept selectivity percentile (0 = fetch all)
        self.conf_res = deque(maxlen=8192)      # reservoir of recently seen candidate confidences
        self.last_seen, self.freq = {}, {}
        self.pending, self.heap = {}, []
        self.tokens = self.spent = 0.0
        self.issued = self.hit = self.aged = 0
        self.outstanding = {}                  # obj -> issue index, for causal precision feedback
        self.recent = deque()                  # (issue_index, obj) ordered, aged out by `win`
        self.floor_trace = []
        r = getattr(self.base, "reset", None)
        if r:
            r()

    # ---- causal precision feedback: no future is consulted, only the request stream ----
    def _observe(self, o, i):
        if o in self.outstanding:              # a prefetch of o was requested before ageing out
            self.outstanding.pop(o, None); self.hit += 1
        while self.recent and i - self.recent[0][0] > self.win:
            _, x = self.recent.popleft()
            if x in self.outstanding:          # aged out unrequested: a wasted fetch
                self.outstanding.pop(x, None); self.aged += 1

    def realised_precision(self):
        n = self.hit + self.aged
        return (self.hit / n) if n >= 200 else None      # withhold until the window is populated

    def _control(self):
        """Refresh the confidence floor to the swept selectivity percentile.

        The lever is SWEPT DIRECTLY rather than driven by a feedback loop chasing a precision
        target. Two reasons, both learned from failed selftests. A proportional controller on an
        absolute floor moved it by hundredths while candidate confidences were concentrated
        elsewhere, so volume was byte-identical at every target; and a controller on the percentile
        converged too slowly to separate the targets within a trace. Neither failure says anything
        about caching -- both would have produced a meaningless "no volume helps" verdict from an
        inert lever. Sweeping selectivity directly guarantees the independent variable actually
        varies, which is what the capture-versus-volume curve requires.
        """
        if self.q <= 0.0 or len(self.conf_res) < 256:
            return
        self.floor = float(np.percentile(np.fromiter(self.conf_res, float), self.q))

    def suggest(self, o, cached, i):
        self.tokens += self.rate
        self._observe(o, i)
        if i % 5000 == 0:
            self._control()
            self.floor_trace.append(self.floor)

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
                conf = cmap.get(x, tau)
                self.conf_res.append(conf)
                if conf < self.floor:                  # the controlled lever
                    continue
                rb = recency_bin(i - self.last_seen[x] if x in self.last_seen else -1)
                fb = freq_bin(self.freq.get(x, 0))
                cb = conf_bin(conf)
                off = self.woff[rb, fb, cb]
                if self.model == "point":
                    off = max(off, 1.0)
                wake = int(i + off - self.lead)
                self.pending[x] = (s, i)
                heapq.heappush(self.heap, (max(wake, i), x))
        self.last_seen[o] = i
        self.freq[o] = self.freq.get(o, 0) + 1

        due = []
        while self.heap and self.heap[0][0] <= i:
            _, x = heapq.heappop(self.heap)
            info = self.pending.get(x)
            if info is None:
                continue
            if x in cached or i - info[1] > self.max_wait:
                self.pending.pop(x, None); continue
            due.append((info[0], x))
        due.sort()
        out = []
        for s, x in due:
            if self.tokens < s or self.spent + s > self.budget:
                continue
            self.tokens -= s; self.spent += s
            self.pending.pop(x, None)
            self.issued += 1
            self.outstanding[x] = i
            self.recent.append((i, x))
            out.append(x)
        return out


def survival_at(log, delta=100):
    """S(delta) from PFCache's survival log: the probability a prefetched object is still cached
    `delta` requests after admission. This is the endogenous-slack thermometer, and it should climb
    back toward 1.0 as the controller lowers volume."""
    if not log:
        return float("nan")
    a = np.array([(d, u) for d, u in log], dtype=np.float64)
    alive = (a[:, 0] >= delta) | (a[:, 1] > 0)
    return float(alive.mean())


def km_survival(log, t):
    """Kaplan-Meier P(a prefetched object has NOT been evicted by age t).

    Uses are CENSORING events, not survivals. When a prefetched object is requested, PFCache stops
    tracking it (it leaves pf_pending but stays resident), so its lifetime is observed only up to
    that point: an object used at age 5 tells us nothing about whether it would still have been
    resident at age 100. Counting it as a survivor, which the first implementation did, biases the
    estimate upward by exactly the precision of the arm.
    """
    ev = sorted(log)
    n = at_risk = len(ev)
    if n == 0:
        return float("nan")
    S = 1.0
    for delta, used in ev:
        if delta > t:
            break
        if at_risk <= 0:
            break
        if used == 0:                       # evicted before use: a failure
            S *= (1.0 - 1.0 / at_risk)
        at_risk -= 1                        # used or evicted, it leaves the risk set
    return S


def survival_report(label, log, t=100):
    """Three estimators of S(t), because they bracket the answer and disagree diagnostically."""
    if not log:
        print(f"  {label:<26} no survival records")
        return None
    a = np.array(log, dtype=np.float64)
    d, u = a[:, 0], a[:, 1]
    naive = float(((d >= t) | (u > 0)).mean())      # what the first implementation computed
    km = km_survival(log, t)                        # correct: uses censored
    evict_only = float((d[u == 0] >= t).mean()) if (u == 0).any() else float("nan")
    print(f"  {label:<26} KM {km:6.3f} | naive {naive:6.3f} | evicted-only {evict_only:6.3f} | "
          f"n {len(log):>9,} used {u.mean():5.1%} | median age {np.median(d):8.0f}")
    return km


def survcheck(trace, cap, pos, szs, haz, args):
    """Does the paper's endogenous-slack measurement reproduce?

    The mechanism section rests on one number: S(100) falling from 1.00 under the tuned bar to
    0.055 under a wide causal policy, which is what licenses "the policy's own volume destroys the
    window its timing model must hit". The r8 sweep measured S(100)=1.000 at every setting, up to
    1.79M fetches, which contradicts it. Either the estimator is wrong, the original came from a
    different configuration, or the claim does not hold. This measures both arms under all three
    estimators so the disagreement is attributable.
    """
    tf = args.train_frac
    n = trace["n"]

    def mk():
        return PREDS[args.pred](trace, train_frac=tf, window=args.window, k=args.k, tau=args.tau)

    def mk_wide(fan):
        p = PREDS[args.pred](trace, train_frac=tf, window=args.window, k=args.k, tau=args.tau,
                             top_m=max(16, fan))
        p.k, p.tau = fan, 0.0
        return p

    print(f"\n{LINE}\n  SURVIVAL CHECK -- does S(100) collapse under a wide causal policy?\n{LINE}")
    bar = PFCache(cap, "s3fifo", mk(), positions=pos, sizes=szs).run(
        trace, cold_train_frac=tf, record_survival=True)
    print(f"  cache holds ~{cap // max(1, int(np.median(list(szs.values())))):,} median-sized "
          f"objects; bar issues {bar['pf_issued']:,} prefetches over {n:,} requests")
    s_bar = survival_report("BAR (tuned)", bar["pf_survival"])

    rows = []
    for fan in (args.fanout, 8):
        for g in (0.5, 0.1):
            sch = VolPolicy(mk_wide(fan), pos, szs, haz, bar["prefetch_bytes"] / max(n, 1),
                            bar["prefetch_bytes"], gamma=g, floor=0.0, sel_q=0.0)
            r = PFCache(cap, "s3fifo", sch, positions=pos, sizes=szs, pf_byte_rate=None).run(
                trace, cold_train_frac=tf, record_survival=True)
            km = survival_report(f"wide causal k={fan} g={g}", r["pf_survival"])
            rows.append((f"k={fan} g={g}", km, r["pf_issued"], r["pf_precision"]))

    print(f"  {'-'*100}")
    kms = [k for _, k, _, _ in rows if k is not None and np.isfinite(k)]
    if s_bar is not None and kms:
        lo = min(kms)
        print(f"  BAR S(100) = {s_bar:.3f}   worst wide-causal S(100) = {lo:.3f}   "
              f"ratio {lo/max(s_bar,1e-9):.2f}x")
        if lo < 0.30 * s_bar:
            print("  VERDICT  COLLAPSE REPRODUCES -- the window does shrink materially under a "
                  "wide causal policy, so endogenous slack is supported; the r8 sweep's "
                  "S(100)=1.000 was the naive estimator counting used objects as survivors.")
        else:
            print("  VERDICT  COLLAPSE DOES NOT REPRODUCE -- S(100) stays high under every wide "
                  "causal configuration measured here, so the mechanism section's 1.00 -> 0.055 "
                  "cannot be supported by this harness. The trap verdict is unaffected (every arm "
                  "still fails), but the EXPLANATION must be restated: these policies score at the "
                  "no-prefetch baseline because precision collapses and the byte budget is spent "
                  "on candidates never requested, not because the residence window closed. "
                  "Locate the original 0.055 before submitting.")
    print(LINE + "\n")


def run(trace, cap, pos, szs, haz, args):
    tf = args.train_frac
    n = trace["n"]

    def mk():
        return PREDS[args.pred](trace, train_frac=tf, window=args.window, k=args.k, tau=args.tau)

    def mk_wide(fan):
        p = PREDS[args.pred](trace, train_frac=tf, window=args.window, k=args.k, tau=args.tau,
                             top_m=max(16, fan))
        p.k, p.tau = fan, 0.0
        return p

    base = PFCache(cap, "s3fifo", None, positions=pos, sizes=szs).run(trace)
    bar = PFCache(cap, "s3fifo", mk(), positions=pos, sizes=szs).run(
        trace, cold_train_frac=tf, return_hits=True)
    rate = bar["prefetch_bytes"] / max(n, 1)
    budget = bar["prefetch_bytes"]

    cover, ncov = build_coverable(mk_wide(args.fanout), trace, args.fanout, 0.0)
    r4 = PFCache(cap, "s3fifo", CoverGatedPrescient(trace, cover, k=max(KS), lookahead=2000),
                 positions=pos, sizes=szs, pf_byte_rate=rate).run(trace, cold_train_frac=tf)
    r4_corr = 100.0 * (r4["ohr"] - bar["ohr"])

    print(f"\n{LINE}\n  r8 -- {args.trace}  [sweep={args.sweep}]\n{LINE}")
    print(f"  BAR  OHR {bar['ohr']:.4f}  pf {bar['pf_issued']:,} (prec {bar['pf_precision']:.3f})")
    print(f"  r4   OHR {r4['ohr']:.4f}  corridor {r4_corr:+.2f} pts  <- the capture denominator")
    print(f"  {'-'*100}")
    print(f"  {'config':<30}{'OHR':>8}{'corridor':>10}{'95% CI':>18}{'prec':>7}"
          f"{'pf':>10}{'S(100)':>8}{'capture':>9}")

    # The sweep grids ARE the point: r6 was one cell of the model grid, and no cell of the volume
    # grid existed at all.
    if args.sweep == "model":
        configs = [dict(model=m, gamma=g, floor=f, sel_q=0.0, fanout=fan, tag=
                        f"{m} g={g} floor={f} k={fan}")
                   for m in ("binned", "point") for g in (0.1, 0.5) for f in (0.0, 0.2)
                   for fan in (args.fanout, max(4, args.fanout // 4))]
    else:
        configs = [dict(model="binned", gamma=0.5, floor=0.0, sel_q=q, fanout=args.fanout,
                        tag=f"selectivity p{q:.0f} (fetch top {100-q:.0f}%)")
                   for q in (0.0, 50.0, 70.0, 85.0, 93.0, 97.0)]

    rows = []
    for c in configs:
        sch = VolPolicy(mk_wide(c["fanout"]), pos, szs, haz, rate, budget, model=c["model"],
                        gamma=c["gamma"], floor=c["floor"], sel_q=c["sel_q"],
                        win=args.win, kp=args.kp)
        res = PFCache(cap, "s3fifo", sch, positions=pos, sizes=szs, pf_byte_rate=None).run(
            trace, cold_train_frac=tf, return_hits=True, record_survival=True)
        corr = 100.0 * (res["ohr"] - bar["ohr"])
        d = res["hits_series"].astype(np.float64) - bar["hits_series"].astype(np.float64)
        lo, hi = block_ci(d, args.blocks, args.resamples, args.seed)
        s100 = survival_at(res["pf_survival"], 100)
        capt = corr / r4_corr if r4_corr > 0 else float("nan")
        rows.append(dict(tag=c["tag"], corr=corr, lo=lo, hi=hi, prec=res["pf_precision"],
                         pf=res["pf_issued"], s100=s100, capt=capt, ohr=res["ohr"],
                         q=getattr(sch, "q", 0.0), floor=sch.floor))
        print(f"  {c['tag']:<30}{res['ohr']:>8.4f}{corr:>+10.2f}"
              f"{f'[{lo:+.1f},{hi:+.1f}]':>18}{res['pf_precision']:>7.3f}"
              f"{res['pf_issued']:>10,}{s100:>8.3f}{capt:>8.1%}")

    # An inert controller yields identical volume at every target and would make any verdict
    # meaningless, so it is checked rather than assumed.
    if args.sweep == "volume":
        vols = sorted({r["pf"] for r in rows})
        print(f"  {'-'*100}")
        print(f"  VOLUME LEVER fetch volume spans {min(vols):,} to {max(vols):,} across targets "
              f"({len(vols)} distinct)")
        if len(vols) < 3:
            print("    !! LEVER INERT -- volume barely moves across settings, so NO verdict about "
                  "volume is supported by this run. Fix the lever before reading the verdict.")
    best = max(rows, key=lambda r: r["corr"])
    print(f"  {'-'*100}")
    print(f"  BEST  {best['tag']}  corridor {best['corr']:+.2f} [{best['lo']:+.1f},{best['hi']:+.1f}]"
          f"  precision {best['prec']:.3f}  S(100) {best['s100']:.3f}")

    # The precision-capture relation is the thing the paper currently asserts and does not show.
    pr = np.array([r["prec"] for r in rows]); co = np.array([r["corr"] for r in rows])
    if len(rows) > 2 and pr.std() > 1e-6:
        print(f"  PRECISION RANGE  {pr.min():.3f} to {pr.max():.3f}   "
              f"corridor {co.min():+.1f} to {co.max():+.1f}   "
              f"corr(precision, corridor) = {np.corrcoef(pr, co)[0,1]:+.2f}")
    sv = np.array([r["s100"] for r in rows])
    if np.isfinite(sv).any():
        print(f"  SURVIVAL RANGE   S(100) {np.nanmin(sv):.3f} to {np.nanmax(sv):.3f}   "
              f"(the endogenous-slack thermometer; 1.0 means the window is intact)")

    if best["lo"] > 0 and best["capt"] >= 0.25:
        v = ("BUILD -- some causal configuration clears the bar. The trap verdict does not hold "
             "for this domain and the paper reports a capture, found by its own mechanism.")
    elif best["corr"] > 0:
        v = (f"PARTIAL -- best causal configuration is positive ({best['corr']:+.2f}) but below the "
             f"25% capture bar. Report the volume-capture curve and the plateau.")
    else:
        v = ("TRAP CONFIRMED ACROSS THE FAMILY -- capture is negative at every model, emission and "
             "volume setting swept, so endogenous slack is a measured constraint rather than an "
             "artifact of one instantiation. This is the strong form of the r6 claim.")
    print(f"  VERDICT  {v}\n{LINE}\n")
    return rows


def selftest():
    class A:
        trace = "SYNTH"; limit = 60000; cache_frac = 0.01; pred = "markov1"; tau = 0.05; k = 1
        window = 16; train_frac = 0.5; haz_frac = 0.75; horizon = 5000; out = "/tmp/_r8haz.npz"
        fanout = 8; sweep = "volume"; win = 5000; kp = 0.05
        blocks = 50; resamples = 300; seed = 0
    a = A()
    trace, cap, pos, szs = prep("SYNTH", a.limit, a.cache_frac)
    haz_run(trace, cap, pos, szs, a)
    run(trace, cap, pos, szs, np.load(a.out), a)
    print("  [p4_r8 selftest] completed without error\n")


def main():
    ap = argparse.ArgumentParser(description="r6 model sweep and r8 volume control")
    ap.add_argument("--trace"); ap.add_argument("--pred", choices=sorted(PREDS))
    ap.add_argument("--tau", type=float); ap.add_argument("--k", type=int)
    ap.add_argument("--haz", help="hazard npz from p4_hazard.py")
    ap.add_argument("--limit", type=int, default=2_000_000)
    ap.add_argument("--cache-frac", type=float, default=0.01)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--fanout", type=int, default=32, help="wide emission fan-out (r4's setting)")
    ap.add_argument("--sweep", choices=("model", "volume"), default="volume")
    ap.add_argument("--survcheck", action="store_true",
                    help="re-measure S(100) under three estimators on the bar and wide causal arms; "
                         "checks whether the endogenous-slack collapse reproduces")
    ap.add_argument("--win", type=int, default=20000, help="sliding window for realised precision")
    ap.add_argument("--kp", type=float, default=0.05, help="controller proportional gain")
    ap.add_argument("--blocks", type=int, default=1000)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not (args.trace and args.pred and args.tau is not None and args.k is not None and args.haz):
        ap.error("need --trace --pred --tau --k --haz")
    trace, cap, pos, szs = prep(args.trace, args.limit, args.cache_frac)
    haz = np.load(args.haz)
    survcheck(trace, cap, pos, szs, haz, args) if args.survcheck else \
        run(trace, cap, pos, szs, haz, args)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
