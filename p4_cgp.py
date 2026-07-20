#!/usr/bin/env python3
"""p4_cgp.py -- Corroboration-Gated Prefetch (CGP): the CONSTRUCTIVE rung.

The capture ladder proved the corridor is reachable in principle (F5: wide coverage + clairvoyant
JIT) but that every causal ingredient fails: RL arbitration is unnecessary (F4'), timing the offered
k=1 stream loses (F1-stream), fetching the wide net at emission thrashes (Policy 1), and causal
absolute-time hazard placement is LATE-dominated (F7). The mechanism is ENDOGENOUS SLACK: a wide
causal policy that mistimes issues far more fetches, whose volume collapses the cache-survival curve
(S(100): 1.0 -> 0.05), shrinking the residence window it must hit. Rank-learnable != placeable.

CGP is the one policy class that structurally escapes that loop: buy coverage WITHOUT buying volume.

    ARM  (zero bytes)  : the WIDE predictor (k up to 32, tau=0) names candidates -> arm[X]=t.
                         Pure metadata, no fetch, no cache occupancy. A byte-free watchlist.
    FIRE (event)       : the TIGHT predictor (the bar's own family, low tau_fire) proposes
                         candidates now; each scores s(X) = conf_fire(X) * (1 + beta * armed(X)),
                         where armed(X) = [X named by the wide arm within the last W_arm requests].
    ALLOCATE (iso-BW)  : fund descending s(X) under a token bucket at the BAR's byte rate + a hard
                         total cap = the bar's prefetch bytes. Volume <= bar BY CONSTRUCTION, so
                         corroborated fires displace the bar's low-confidence near-tau fires within
                         the SAME byte budget -- reallocation, not addition.

Why it escapes each named kill:
  * F4'       : still a separable per-candidate threshold on s(X); no cross-candidate knapsack.
  * F1-stream : the funded set differs from the bar's offered stream -- it adds armed sub-tau_bar
                objects the bar never emits (coverage) and prunes un-armed near-tau fires (precision).
  * Policy 1  : does NOT fetch at wide emission; the wide pass only ARMS (free). Fetch binds to the
                tight fire event, so no at-emission flood.
  * F7        : NO absolute-time hazard forecast. Insertion time = the observed tight-predecessor
                event, empirically ~1 hop before use in associative workloads. Event-triggered.
  * substitutes: evictor untouched (S3-FIFO frozen); no prefetch protection.

Defeats endogenous slack: byte volume is pinned at the bar's budget, so S(.) stays near the bar's
(no collapse); the arm-conjunction raises precision, so the fires that land ~1 hop early convert.

beta = 0 is the FIRE-ONLY ablation (no arm gate): the decisive internal control. If CGP (beta>0)
does not beat fire-only at matched volume, the arm signal is redundant and CGP collapses to Policy 1.

PRE-REGISTERED GATE (fixed before any run; see SPEC_cgp_prereg.md):
  capture >= +2.5 OHR pts over the tuned bar at iso-bandwidth, block-bootstrap CI lower bound > 0,
  on BOTH live traces. Hyperparameters {tau_fire, W_arm, beta} tuned by OHR on the replay (same
  regime as the bar's tau/k), grid printed for transparency. F5 corridor is the reference denominator.
  PASS  -> the first causal policy to beat the honest bar: the paper's constructive contribution.
  FAIL  -> ladder rung 6; the negative hardens (even coverage/volume decoupling fails).

    python p4_cgp.py --trace data/wiki_2019t.oracleGeneral        --pred markov2 --tau 0.05 --k 1 \
        --wide-k 32 --wide-top-m 32 --f5-ref 10.41 --verbose
    python p4_cgp.py --trace data/cluster50.sample10.oracleGeneral --pred markov3 --tau 0.06 --k 1 \
        --wide-k 16 --f5-ref 11.42 --verbose
    python p4_cgp.py --selftest
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from p4_coldsplit import PREDS
from p4_f4 import capture_ctx, conf_lookup
from p4_prefetch import PFCache
from p4_sweep import prep


def block_ci(diff, blocks, resamples, seed):
    R = len(diff); L = R // blocks
    bm = diff[:blocks * L].reshape(blocks, L).mean(axis=1)
    rng = np.random.default_rng(seed)
    boot = 100.0 * bm[rng.integers(0, blocks, size=(resamples, blocks))].mean(axis=1)
    return np.percentile(boot, [2.5, 97.5])


class CGP:
    """Corroboration-Gated Prefetch. See module docstring. beta=0 -> fire-only ablation.

    Owns its byte budget outright (PFCache runs with pf_byte_rate=None): the token bucket + hard cap
    here are the ONLY iso-bandwidth control, so the s(X) ranking is never overridden by PFCache."""
    name = "cgp"

    def __init__(self, wide, fire, pos, szs, rate, budget, w_arm, tau_fire, beta, fire_k, cap,
                 max_arm=4_000_000, verbose=False, n=0, vstep=200_000):
        self.wide, self.fire = wide, fire
        self.pos, self.szs, self.cap = pos, szs, cap
        self.rate, self.budget = rate, budget
        self.w_arm, self.tau_fire, self.beta, self.fire_k = w_arm, tau_fire, beta, fire_k
        self.max_arm = max_arm
        self.verbose, self.n, self.vstep, self.t_start = verbose, n, vstep, None
        self.arm = {}                      # obj -> last wide-emission request index (the watchlist)
        self.tokens = self.spent = 0.0
        self.issued = self.armed_fires = 0

    def reset(self):
        self.arm = {}
        self.tokens = self.spent = 0.0
        self.issued = self.armed_fires = 0
        self.t_start = time.time()
        for p in (self.wide, self.fire):
            r = getattr(p, "reset", None)
            if r:
                r()

    def suggest(self, o, cached, i):
        self.tokens += self.rate
        if self.verbose and i and i % self.vstep == 0:
            el = time.time() - (self.t_start or time.time()); rt = i / el if el > 0 else 0.0
            print(f"    [cgp b{self.beta:.0f} w{self.w_arm} t{self.tau_fire}] {i:>9,}/{self.n:,} "
                  f"({i/max(self.n,1):4.0%}) | arm {len(self.arm):>7,} | issued {self.issued:>8,} | "
                  f"armed {self.armed_fires/max(self.issued,1):3.0%} | spent {self.spent/max(self.budget,1):.2f}x"
                  f" | {rt:>5.0f} req/s", file=sys.stderr, flush=True)

        empty: frozenset = frozenset()
        # (1) FIRE -- uses arm state STRICTLY in the past (arming for THIS request happens in step 2)
        ctx = capture_ctx(self.fire)
        cands = self.fire.suggest(o, cached, i)
        out = []
        if cands:
            cmap = conf_lookup(self.fire, ctx, o)
            scored = []
            for x in cands:
                x = int(x)
                if x in cached:
                    continue
                s = self.szs.get(x)
                if s is None or s > self.cap or s > self.budget:
                    continue
                conf = cmap.get(x, self.tau_fire)
                armed = 1.0 if (i - self.arm.get(x, -(1 << 60))) <= self.w_arm else 0.0
                scored.append((conf * (1.0 + self.beta * armed), s, x, armed))
            scored.sort(reverse=True)                       # highest corroborated score first
            for score, s, x, armed in scored:
                if self.spent + s > self.budget:            # hard total cap -> iso-bandwidth
                    continue
                if self.tokens < s:                         # rate bucket -> no front-loading
                    continue
                self.tokens -= s; self.spent += s
                self.issued += 1; self.armed_fires += int(armed)
                out.append(x)

        # (2) ARM -- record this request's WIDE emissions (byte-free). After firing, so arm is past-only.
        for c in self.wide.suggest(o, empty, i):
            self.arm[int(c)] = i
        if len(self.arm) > self.max_arm:                    # bounded watchlist (deployable + memory)
            cutoff = i - self.w_arm
            self.arm = {k: v for k, v in self.arm.items() if v >= cutoff}

        out.sort(key=lambda x: self.szs.get(x, 0))          # smallest-first: head-of-line safety
        return out


def _cgp_run(cap, pos, szs, trace, tf, sch, hits=False):
    return PFCache(cap, "s3fifo", sch, positions=pos, sizes=szs, pf_byte_rate=None).run(
        trace, cold_train_frac=tf, return_hits=hits)


def run(trace, cap, pos, szs, args):
    tf = args.train_frac
    n = trace["n"]

    t0 = time.time()
    bar_pred = PREDS[args.pred](trace, train_frac=tf, window=args.window, k=args.k, tau=args.tau)
    fire_pred = PREDS[args.pred](trace, train_frac=tf, window=args.window, k=args.fire_k, tau=0.0)
    wide_pred = PREDS[args.pred](trace, train_frac=tf, window=args.window, k=args.k, tau=args.tau,
                                 top_m=args.wide_top_m)
    wide_pred.k, wide_pred.tau = args.wide_k, args.wide_tau
    if args.verbose:
        print(f"  [build] predictors ({time.time()-t0:.1f}s)", file=sys.stderr, flush=True)

    base = PFCache(cap, "s3fifo", None, positions=pos, sizes=szs).run(trace)
    bar = PFCache(cap, "s3fifo", bar_pred, positions=pos, sizes=szs).run(
        trace, cold_train_frac=tf, return_hits=True)
    rate = bar["prefetch_bytes"] / max(n, 1)
    budget = bar["prefetch_bytes"]
    bar_tx = bar["origin_bytes"] / max(base["origin_bytes"], 1)
    # BAR-CAP: the bar under the SAME token bucket CGP faces -> isolates scheduling from the
    # burst-vs-bucket protocol cost (HJS-L / Policy-1 precedent). Headline gate is vs BAR.
    barc = PFCache(cap, "s3fifo", bar_pred, positions=pos, sizes=szs, pf_byte_rate=rate).run(
        trace, cold_train_frac=tf, return_hits=True)
    protocol = 100.0 * (barc["ohr"] - bar["ohr"])

    print(f"\n  {args.trace}   [CGP | wide k={args.wide_k} top_m={args.wide_top_m} | fire k={args.fire_k}]")
    print(f"  BASE      OHR {base['ohr']:.4f}")
    print(f"  BAR       OHR {bar['ohr']:.4f} @{bar_tx:.2f}x   pf {bar['pf_issued']:,} "
          f"(prec {bar['pf_precision']:.3f})   <- pre-registered baseline")
    print(f"  BAR-CAP   OHR {barc['ohr']:.4f}   protocol cost {protocol:+.2f} pts (bar burst vs bucket)")
    if args.f5_ref is not None:
        print(f"  F5 REF    corridor {args.f5_ref:+.2f} pts   (clairvoyant coverage+JIT ceiling, from results.md 6)")
    print(f"  {'-'*80}")

    # ---- tune {tau_fire, W_arm, beta} by OHR on the replay (same regime as the bar's tau/k) ----
    betas = [float(b) for b in args.betas.split(",")]
    taus = [float(t) for t in args.tau_fires.split(",")]
    warms = [int(w) for w in args.w_arms.split(",")]
    print(f"  {'config':28s} {'OHR':>8s} {'vs BAR':>8s} {'pf issued':>10s} {'prec':>6s} {'armed%':>7s} {'bytes':>7s}")
    best = best_fo = None                                   # best CGP (beta>0) and best fire-only (beta=0)
    for be in betas:
        for tfire in taus:
            for wa in warms:
                if be == 0.0 and wa != warms[0]:
                    continue                                # fire-only is W_arm-independent
                fire_pred.set_params(tau=tfire)
                sch = CGP(wide_pred, fire_pred, pos, szs, rate, budget, wa, tfire, be, args.fire_k,
                          cap, verbose=(args.verbose and args.tune_verbose), n=n)
                r = _cgp_run(cap, pos, szs, trace, tf, sch)
                ohr = r["ohr"]; d = 100.0 * (ohr - bar["ohr"])
                pfb = r["prefetch_bytes"] / max(budget, 1)
                tag = "  (fire-only)" if be == 0.0 else ""
                print(f"  b={be:<3.0f} tf={tfire:<4.2f} wa={wa:<6d}{tag:>2s} {ohr:8.4f} {d:+8.2f} "
                      f"{r['pf_issued']:>10,} {r['pf_precision']:6.3f} "
                      f"{sch.armed_fires/max(sch.issued,1):6.0%} {pfb:6.2f}x", flush=True)
                if be > 0.0 and (best is None or ohr > best[1]):
                    best = ((be, tfire, wa), ohr)
                if be == 0.0 and (best_fo is None or ohr > best_fo[1]):
                    best_fo = ((0.0, tfire, warms[0]), ohr)  # fire-only tuned over its OWN tau
    print(f"  {'-'*80}")

    if best is None or best_fo is None:
        print("  !! grid must include beta=0 and at least one beta>0 -- check --betas"); return None
    (be, tfire, wa), _ = best
    (_, fo_tfire, _), _ = best_fo                            # fire-only at ITS best tau (fair ablation)

    # ---- finalize: best CGP + best fire-only, WITH hit series, bootstrap CIs vs BAR and BAR-CAP ----
    fire_pred.set_params(tau=tfire)
    cgp_sch = CGP(wide_pred, fire_pred, pos, szs, rate, budget, wa, tfire, be, args.fire_k, cap,
                  verbose=args.verbose, n=n)
    cgp = _cgp_run(cap, pos, szs, trace, tf, cgp_sch, hits=True)

    fire_pred.set_params(tau=fo_tfire)
    fo_sch = CGP(wide_pred, fire_pred, pos, szs, rate, budget, warms[0], fo_tfire, 0.0, args.fire_k,
                 cap, verbose=False, n=n)
    fo = _cgp_run(cap, pos, szs, trace, tf, fo_sch, hits=True)

    def corr_ci(arm_hits, ref_hits, ref_ohr, arm_ohr):
        d = arm_hits.astype(np.float64) - ref_hits.astype(np.float64)
        lo, hi = block_ci(d, args.blocks, args.resamples, args.seed)
        return 100.0 * (arm_ohr - ref_ohr), lo, hi

    cgp_corr, clo, chi = corr_ci(cgp["hits_series"], bar["hits_series"], bar["ohr"], cgp["ohr"])
    cgp_ccorr, cclo, cchi = corr_ci(cgp["hits_series"], barc["hits_series"], barc["ohr"], cgp["ohr"])
    fo_corr, folo, fohi = corr_ci(fo["hits_series"], bar["hits_series"], bar["ohr"], fo["ohr"])
    cgp_tx = cgp["origin_bytes"] / max(base["origin_bytes"], 1)
    pfb = cgp["prefetch_bytes"] / max(budget, 1)

    inv = []
    if cgp["pf_cold_hits"] != 0:
        inv.append(f"CGP cold hits {cgp['pf_cold_hits']} != 0 -- funding OOV objects")
    if pfb > 1.02:
        inv.append(f"CGP prefetch bytes {pfb:.2f}x of bar -- NOT iso-bandwidth")
    okinv = "all pass" if not inv else "FAIL"

    print(f"  BEST CGP  b={be:.0f} tf={tfire:.2f} wa={wa}   OHR {cgp['ohr']:.4f} @{cgp_tx:.2f}x   "
          f"pf {cgp['pf_issued']:,} (prec {cgp['pf_precision']:.3f}, armed {cgp_sch.armed_fires/max(cgp_sch.issued,1):.0%})")
    print(f"  CGP corridor vs BAR      {cgp_corr:+.2f}  [{clo:+.2f},{chi:+.2f}]   pf bytes {pfb:.2f}x   inv:{okinv}")
    print(f"  CGP corridor vs BAR-CAP  {cgp_ccorr:+.2f}  [{cclo:+.2f},{cchi:+.2f}]   (like-for-like protocol)")
    print(f"  FIRE-ONLY vs BAR         {fo_corr:+.2f}  [{folo:+.2f},{fohi:+.2f}]   (tf={fo_tfire:.2f}, "
          f"own-best; CGP must beat THIS to prove the arm signal is not redundant with fire conf)")
    if args.f5_ref:
        print(f"  capture of F5            {100.0*cgp_corr/args.f5_ref:.0f}% of the {args.f5_ref:+.2f} corridor")
    for m in inv:
        print(f"    !! {m}")

    arm_lift = cgp_corr - fo_corr
    if inv:
        v = "INVALID -- iso-bandwidth or cold invariant failed; fix before reading the gate."
    elif clo > 0 and cgp_corr >= args.gate:
        v = (f"PASS -- CGP beats the honest bar by {cgp_corr:+.2f} (CI lo {clo:+.2f} > 0, >= {args.gate} gate). "
             f"Arm lift over fire-only {arm_lift:+.2f}. The first causal policy to capture the corridor.")
    elif clo > 0:
        v = (f"POSITIVE but below the {args.gate}-pt gate ({cgp_corr:+.2f}, CI lo {clo:+.2f}). "
             f"A weak capture; report as partial, not the headline.")
    else:
        v = (f"FAIL -- CGP does not beat the bar (CI includes 0). Ladder rung 6; the negative holds. "
             f"Arm lift over fire-only {arm_lift:+.2f}.")
    print(f"  GATE (>= {args.gate} vs BAR, CI>0)  {v}")
    print()
    return dict(cgp=cgp_corr, ci=(clo, chi), fire_only=fo_corr, arm_lift=arm_lift,
                config=(be, tfire, wa), inv_ok=not inv)


def selftest():
    class A:
        trace = "SYNTH"; limit = 60000; cache_frac = 0.01; pred = "markov1"; tau = 0.05; k = 1
        window = 16; train_frac = 0.5; wide_k = 16; wide_top_m = 16; wide_tau = 0.0; fire_k = 8
        betas = "0,3"; tau_fires = "0.02,0.05"; w_arms = "2000,10000"; f5_ref = None
        gate = 2.5; blocks = 100; resamples = 500; seed = 0; verbose = False; tune_verbose = False
    a = A()
    trace, cap, pos, szs = prep("SYNTH", a.limit, a.cache_frac)
    r = run(trace, cap, pos, szs, a)
    assert r is not None and r["inv_ok"], "selftest: invariants failed"
    print("  [p4_cgp selftest] completed without error\n")


def main():
    ap = argparse.ArgumentParser(description="Corroboration-Gated Prefetch (constructive rung)")
    ap.add_argument("--trace")
    ap.add_argument("--limit", type=int, default=2_000_000)
    ap.add_argument("--cache-frac", type=float, default=0.01)
    ap.add_argument("--pred", choices=sorted(PREDS))
    ap.add_argument("--tau", type=float, help="the BAR's tau")
    ap.add_argument("--k", type=int, help="the BAR's k")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--wide-k", type=int, default=32, help="arm fanout (the F5-cleared width)")
    ap.add_argument("--wide-top-m", type=int, default=32)
    ap.add_argument("--wide-tau", type=float, default=0.0)
    ap.add_argument("--fire-k", type=int, default=8, help="candidates the tight predictor proposes per request")
    ap.add_argument("--betas", default="0,2,4", help="arm-boost grid; 0 = fire-only ablation")
    ap.add_argument("--tau-fires", default="0.02,0.05", help="fire-threshold grid")
    ap.add_argument("--w-arms", default="2000,10000", help="arm-window grid (requests)")
    ap.add_argument("--f5-ref", type=float, default=None,
                    help="F5 corridor for %%-capture context (wiki 10.41, cluster50 11.42 from results.md)")
    ap.add_argument("--gate", type=float, default=2.5, help="pre-registered OHR-pt gate vs BAR")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--tune-verbose", action="store_true", help="per-arm progress during the tuning grid too")
    ap.add_argument("--blocks", type=int, default=1000)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not (args.trace and args.pred and args.tau is not None and args.k is not None):
        ap.error("need --trace --pred --tau --k")
    trace, cap, pos, szs = prep(args.trace, args.limit, args.cache_frac)
    run(trace, cap, pos, szs, args)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
