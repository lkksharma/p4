#!/usr/bin/env python3
"""
oppcert/ladder/breakeven_precision.py -- THE BREAK-EVEN PRECISION FRONTIER: what precision must a causal scheduler reach?

PAPER: Stage 2 mechanism (internal name: r9). The break-even precision frontier p* = 0.582, against the
0.062 r6 achieves.

Why this rung exists
--------------------
r8 established the trap robustly (capture negative at every setting swept) but the EXPLANATION
offered for it -- endogenous slack, a policy's own volume evicting its future targets -- did not
survive its own check: --survcheck measures S(100) at 0.989 (wiki) and 0.911 (cluster50), not the
0.055/0.135 the mechanism claimed. The residence window never closes. So the finding stands and the
story does not, and this rung supplies the story that the same data actually supports.

The coverage-precision scissors
-------------------------------
Read r3 through r8 as one sequence and the mechanism is visible without any appeal to eviction:

    arm                                coverage    precision   corridor
    tuned bar (k=1)                    narrow      0.874       0 (by definition)
    r3 perfect timing, own candidates  narrow      high        -7.01 / -10.82
    r4 wide + clairvoyant JIT          wide        ~1.0        +10.41 / +11.42
    r6 wide + causal timing            wide        0.05        -40.06 / -26.01
    r8 wide, every setting swept       wide        0.002-0.039 -385%

r3 proves the corridor is NOT reachable by re-timing the candidates the baseline already names:
coverage must widen. r4 proves that widening reaches it. But precision falls from 0.874 to 0.002
exactly when coverage widens, and under a FIXED byte budget those two trade against each other:
the budget spent on never-requested candidates is budget not spent on real ones. Acquiring the
coverage the corridor requires destroys the precision needed to pay for it. That is a statement
about resource allocation, not about eviction, which is why it survives the survival check.

What this rung measures
-----------------------
The scissors implies a threshold. Somewhere between precision 1.0 (r4, +10.41) and precision 0.002
(r8, -40.0) there is a precision p* at which a wide-coverage policy exactly matches the tuned
baseline. p* is the SPECIFICATION any causal scheduler must meet. It is measured here by
construction, not asserted:

    * take the wide (fan-out K, no confidence floor) candidate stream -- r4's own coverage set
    * label each candidate by ORACLE: is it actually requested within `horizon` after emission?
    * admit every true positive, plus a deterministic fraction f of the false positives
    * f = 0   -> precision 1.0  (this reproduces r4)
      f = 1   -> precision = the wide stream's natural rate (this reproduces r8's arms)
    * sweep f, record (achieved precision, OHR), and interpolate where OHR crosses the bar

Everything else is held identical to every other arm: same evictor, same iso-bandwidth token
bucket at the bar's byte rate, same warmup and scoring, paired block-bootstrap CIs.

This is a CONSTRUCTION that traces a frontier, exactly like r4, and is never a deployable policy:
it reads the future to decide which candidates to keep. Its purpose is to price what precision is
worth, so the gap between p* and what causal models actually achieve (0.05 at r6, 0.002-0.039
across r8) becomes a measured quantity rather than a narrative.

    python -m oppcert.ladder.breakeven_precision --trace data/wiki_2019t.oracleGeneral --pred markov2 --tau 0.05 --k 1
"""
from __future__ import annotations

import argparse

import numpy as np

from oppcert.sim.trace import NEVER
from oppcert.instrument.cold_split import PREDS
from oppcert.sim.prefetch import PFCache, oracle_next
from oppcert.instrument.gates import prep

LINE = "=" * 104


def _keep_fp(x, f, salt=0x9E3779B9):
    """Deterministic per-object coin for false-positive admission, so an arm is reproducible and
    the same object is treated identically every time it is emitted (a per-emission random draw
    would let a candidate slip in on its second chance and quietly raise coverage with f)."""
    if f <= 0.0:
        return False
    if f >= 1.0:
        return True
    h = (int(x) * salt) & 0xFFFFFFFF
    return (h / 0xFFFFFFFF) < f


class PrecisionArm:
    """Wide emission, oracle-filtered to a target precision, just-in-time insertion.

    True positives (requested within `horizon`) are woken just before their true use, which is r4's
    insertion rule. False positives have no use to aim at, so they are issued at emission -- which
    is what a real policy that believed in them would do, and is what makes them cost budget.
    """
    name = "r9_precision"

    def __init__(self, base, pos, szs, f, horizon, lead=1):
        self.base, self.pos, self.szs = base, pos, szs
        self.f, self.horizon, self.lead = f, horizon, lead
        self.wake: dict = {}
        self.tp = self.fp = 0

    def reset(self):
        self.wake = {}
        self.tp = self.fp = 0
        r = getattr(self.base, "reset", None)
        if r:
            r()

    def suggest(self, o, cached, i):
        for x in self.base.suggest(o, cached, i):
            x = int(x)
            if x in self.wake or x in cached:
                continue
            nx = oracle_next(self.pos, x, i)
            used = nx < int(NEVER) and (int(nx) - i) <= self.horizon
            if used:
                self.wake[x] = max(int(nx) - self.lead, i)      # JIT, as in r4
                self.tp += 1
            else:
                if not _keep_fp(x, self.f):
                    continue                                    # filtered out by the target precision
                self.wake[x] = i                                # no use to aim at -> issue now
                self.fp += 1
        out = [x for x, w in self.wake.items() if w <= i]
        for x in out:
            del self.wake[x]
        out = [x for x in out if x not in cached]
        out.sort(key=lambda x: self.szs.get(x, 0))              # smallest-first (HOL safety)
        return out


def block_ci(diff, blocks, resamples, seed):
    R = len(diff); L = R // blocks
    bm = diff[:blocks * L].reshape(blocks, L).mean(axis=1)
    rng = np.random.default_rng(seed)
    boot = 100.0 * bm[rng.integers(0, blocks, size=(resamples, blocks))].mean(axis=1)
    return np.percentile(boot, [2.5, 97.5])


def run(trace, cap, pos, szs, args):
    tf = args.train_frac
    n = trace["n"]

    def mk():
        return PREDS[args.pred](trace, train_frac=tf, window=args.window, k=args.k, tau=args.tau)

    def mk_wide():
        p = PREDS[args.pred](trace, train_frac=tf, window=args.window, k=args.k, tau=args.tau,
                             top_m=max(16, args.fanout))
        p.k, p.tau = args.fanout, 0.0
        return p

    base = PFCache(cap, "s3fifo", None, positions=pos, sizes=szs).run(trace)
    bar = PFCache(cap, "s3fifo", mk(), positions=pos, sizes=szs).run(
        trace, cold_train_frac=tf, return_hits=True)
    rate = bar["prefetch_bytes"] / max(n, 1)
    bar_tx = bar["origin_bytes"] / max(base["origin_bytes"], 1)

    print(f"\n{LINE}\n  r9 -- BREAK-EVEN PRECISION -- {args.trace}\n{LINE}")
    print(f"  BASE (no prefetch)  OHR {base['ohr']:.4f}")
    print(f"  BAR  (tuned, k={args.k})  OHR {bar['ohr']:.4f} @{bar_tx:.2f}x   "
          f"pf {bar['pf_issued']:,}  precision {bar['pf_precision']:.3f}   <- the target to beat")
    print(f"  wide emission: fan-out {args.fanout}, no confidence floor; horizon {args.horizon:,}")
    print("-" * 104)
    print(f"  {'f (FP admit)':>13} {'precision':>10} {'OHR':>8} {'corridor':>9} "
          f"{'95% CI':>18} {'pf':>11} {'traffic':>8}")

    rows = []
    for f in args.fracs:
        sch = PrecisionArm(mk_wide(), pos, szs, f, args.horizon)
        r = PFCache(cap, "s3fifo", sch, positions=pos, sizes=szs, pf_byte_rate=rate).run(
            trace, cold_train_frac=tf, return_hits=True)
        corr = 100.0 * (r["ohr"] - bar["ohr"])
        d = r["hits_series"].astype(np.float64) - bar["hits_series"].astype(np.float64)
        lo, hi = block_ci(d, args.blocks, args.resamples, args.seed)
        tx = r["origin_bytes"] / max(base["origin_bytes"], 1)
        prec = r["pf_precision"]
        rows.append((f, prec, r["ohr"], corr, lo, hi, r["pf_issued"], tx))
        print(f"  {f:>13.3f} {prec:>10.3f} {r['ohr']:>8.4f} {corr:>+9.2f} "
              f"{f'[{lo:+.1f},{hi:+.1f}]':>18} {r['pf_issued']:>11,} {tx:>7.2f}x")

    print("-" * 104)
    # break-even: the precision at which the corridor crosses zero, linearly interpolated in
    # precision between the bracketing arms
    pos_rows = [r for r in rows if r[3] > 0]
    neg_rows = [r for r in rows if r[3] <= 0]
    if pos_rows and neg_rows:
        a = min(pos_rows, key=lambda r: r[1])          # lowest-precision arm that still wins
        b = max(neg_rows, key=lambda r: r[1])          # highest-precision arm that loses
        if a[1] != b[1]:
            t = (0.0 - b[3]) / (a[3] - b[3])
            pstar = b[1] + t * (a[1] - b[1])
        else:
            pstar = a[1]
        print(f"  BREAK-EVEN PRECISION  p* = {pstar:.3f}   "
              f"(bracketed by {b[1]:.3f} at {b[3]:+.2f} and {a[1]:.3f} at {a[3]:+.2f})")
        # These reference precisions were once literals in this file. That is the same defect
        # already removed from oppcert/domains/learned_index.py and oppcert/domains/substrate_control.py: a printed comparison that does
        # not move when the measurement moves. They were also trace-independent, so the r6 figure
        # for Wikipedia was being reported unchanged on cluster50. Supply them per trace with
        # --ref, taking each value from that trace's own arm log.
        best = None
        for spec in (args.ref or []):
            name, _, val = spec.partition("=")
            # A malformed --ref must never destroy a completed sweep: every row above cost a
            # full replay, and the break-even line below is the point of the run.
            try:
                got = float(val)
            except ValueError:
                print(f"     !! --ref {spec!r} is not a number; skipped (p* below is unaffected)")
                continue
            if got <= 0:
                continue
            print(f"     {name:<22} achieves {got:.3f}  ->  short by {pstar / got:.1f}x")
            best = got if best is None else max(best, got)
        print(f"  READS AS: a causal scheduler must convert {100*pstar:.0f}% of the bytes it spends "
              f"into hits\n            to match the tuned baseline at this coverage.", end="")
        if best is not None:
            print(f" The best causal arm supplied\n            converts {100*best:.0f}%.", end="")
        else:
            print("\n            No --ref arms supplied, so no shortfall is claimed here.", end="")
        print(" The corridor is real and the\n            specification to reach it is now quantified.")
    elif not neg_rows:
        print("  every arm beats the bar -- p* lies below the lowest precision swept; widen --fracs")
    else:
        print("  no arm beats the bar -- p* lies above 1.0 at this coverage, i.e. even a perfectly")
        print("  precise wide policy cannot match the baseline. That is a stronger negative than a")
        print("  precision gap and should be reported as such.")
    print(f"{LINE}\n")
    return rows


def main():
    ap = argparse.ArgumentParser(description="r9: break-even precision for a causal scheduler")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--limit", type=int, default=2_000_000)
    ap.add_argument("--cache-frac", type=float, default=0.01)
    ap.add_argument("--pred", choices=sorted(PREDS), required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--fanout", type=int, default=32, help="wide emission fan-out (r4's setting)")
    ap.add_argument("--horizon", type=int, default=100_000,
                    help="a candidate counts as a true positive if used within this many requests")
    ap.add_argument("--fracs", type=float, nargs="+",
                    default=[0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                    help="fractions of false positives admitted; 0 = precision 1.0 (r4)")
    ap.add_argument("--ref", nargs="*", metavar="NAME=PRECISION",
                    help="measured precisions of real causal arms on THIS trace, e.g. "
                         "--ref 'r6 causal placement=0.062' 'r8 best swept=0.039'. Read each "
                         "value from that trace's own log; nothing here supplies a default.")
    ap.add_argument("--blocks", type=int, default=1000)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    trace, cap, pos, szs = prep(args.trace, args.limit, args.cache_frac)
    run(trace, cap, pos, szs, args)


if __name__ == "__main__":
    main()
