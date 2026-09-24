#!/usr/bin/env python3
"""
oppcert/ladder/r2_arbitration.py -- the NECESSITY GAP: the gate on rung (iv) of the Necessity Ladder (residual RL).

PAPER: Stage 2 rung r2, arbitration (internal names: F4, F4'). Optimal joint chooser vs a per-candidate
threshold rule: +0.01 / -0.01 against the pre-registered 2-point bar.

The question RL must answer BEFORE it earns a training budget: how much can a NON-SEPARABLE
policy add over a SEPARABLE per-candidate rule, given identical value estimates and an identical
byte budget? That gap bounds what RL can contribute. Measure it with replay; do not assume it.

Two arms, identical in every respect except the funding rule:
    SEP   -- separable: each pooled candidate decided on its own (fund iff value/size >= lambda,
             in arrival order, while the token bucket allows). No cross-candidate comparison.
    JOINT -- non-separable: choose the value-maximising SUBSET of the pending pool under the same
             token budget (knapsack, greedy-by-density + bounded swap improvement). May skip a
             large candidate to fund several small ones.

    necessity gap = OHR(JOINT) - OHR(SEP)

Two value regimes:
    --mode clair  F4  (CONTROL). value = 1 if the object is ever used again, else 0. Clairvoyant.
                  EXPECTED TO READ ~0, and ~0 EXACTLY on uniform-size traces. Under clairvoyance
                  every funded candidate pays off, so value collapses to ~1 per item, the separable
                  rule becomes a pure size cutoff, and the joint arm becomes a 0/1 knapsack whose
                  gap over greedy is bounded by roughly one item. A ~0 here is a SOLVER CHECK, not
                  evidence about RL: making both arms clairvoyant erases uncertainty, which is the
                  only thing that makes arbitration hard.
    --mode est    F4' (THE ACTUAL GATE). value = the predictor's own confidence for that candidate,
                  read from the Markov table rows (which are [(obj, conf), ...] confidence-desc).
                  Heterogeneous values are what make arbitration non-trivial: with uniform values,
                  maximising count under a byte budget is exactly "take smallest first", which the
                  separable rule already does, and the gap is 0 by construction.

PRE-REGISTERED GATE (fix before running):
    F4' < 1 pt on all live traces  -> arbitration is separable; RL is measurably unnecessary here.
                                      Report the number as the finding. Do not train.
    F4' >= 2 pts on any live trace -> RL earns a training budget on that trace, with F4' as its
                                      pre-registered target (must capture >=50%, CI clear of 0).
    1-2 pts                        -> inconclusive; the closed-form rule stands as the method.

Example (wiki, the real gate):
    python -m oppcert.ladder.r2_arbitration --trace data/wiki_2019t.oracleGeneral --pred markov2 --tau 0.05 --k 1 --mode est
"""
from __future__ import annotations

import argparse

import numpy as np

from oppcert.sim.trace import NEVER
from oppcert.instrument.cold_split import PREDS
from oppcert.sim.prefetch import PFCache, oracle_next
from oppcert.instrument.gates import prep


def capture_ctx(pred):
    """Snapshot the predictor's context BEFORE suggest() mutates it.

    Markov3 keeps (p2, p1); Markov2 keeps prev; Markov1 is contextless. Must be read first or the
    confidence lookup below reads the row for the WRONG context.
    """
    if hasattr(pred, "p2"):
        return (pred.p2, pred.p1)
    return getattr(pred, "prev", None)


def conf_lookup(pred, ctx, o):
    """{obj: confidence} for the context that produced this request's candidates.

    Mirrors each predictor's own backoff chain exactly (t3 -> t2 -> t1 for Markov3, t2 -> t1 for
    Markov2), so the values match what the bar actually acted on. Objects absent from the row fall
    back to tau at the call site -- approximate by design, and identical across both arms, so the
    GAP stays valid regardless.
    """
    o = int(o)
    t3 = getattr(pred, "t3", None)
    if t3 is not None:                                   # Markov3: (p2, p1, o) -> (p1, o) -> o
        p2, p1 = ctx if isinstance(ctx, tuple) else (None, None)
        rows = t3.get((p2, p1, o)) if p2 is not None else None
        if rows is None and p1 is not None:
            rows = pred.t2.get((p1, o))
        if rows is None:
            rows = pred.t1.get(o, ())
        return dict(rows)
    t2 = getattr(pred, "t2", None)
    if t2 is not None:                                   # Markov2: (prev, o) -> o
        rows = t2.get((ctx, o)) if ctx is not None else None
        if rows is None:
            rows = pred.t1.get(o, ())
        return dict(rows)
    tbl = getattr(pred, "table", None)                   # Markov1
    return dict(tbl.get(o, ())) if tbl is not None else {}


class PoolArm:
    """Shared mechanics for both arms: a pending candidate pool + a mirrored token bucket.

    The pool is what makes arbitration exist at all. The live bars all use k=1, so at most one
    candidate arrives per request -- with immediate issue there is nothing to arbitrate between.
    Holding candidates for `ttl` requests creates a real "which of these do my accumulated tokens
    buy" decision, which is precisely the question RL would be asked to answer.

    The bucket is MIRRORED (not read from PFCache, which does not expose it): same accrual rate,
    deducted on every candidate we emit. Candidates larger than the cache are skipped because
    PFCache's _admit would reject them without deducting, which would desync the mirror.
    """
    name = "f4_pool"

    def __init__(self, base, positions, sizes, rate, mode, arm, ttl, lam, cap_bytes, budget):
        self.base, self.pos, self.sizes = base, positions, sizes
        self.rate, self.mode, self.arm = rate, mode, arm
        self.ttl, self.lam, self.cap = ttl, lam, cap_bytes
        self.budget = budget          # HARD total-byte cap (= bar prefetch bytes) -> iso-bandwidth
        self.pool: dict = {}          # obj -> (value, size, expiry); insertion order = arrival order
        self.tokens = 0.0
        self.spent = 0.0              # cumulative prefetch bytes emitted by this arm
        self.densities: list = []     # diagnostic: value/size seen, for calibrating lambda

    def reset(self):
        self.pool, self.tokens, self.spent, self.densities = {}, 0.0, 0.0, []
        r = getattr(self.base, "reset", None)
        if r:
            r()

    def _value(self, x, i, conf):
        if self.mode == "clair":
            return 1.0 if oracle_next(self.pos, x, i) < NEVER else 0.0
        return float(conf)

    def suggest(self, o, cached, i):
        self.tokens += self.rate

        ctx = capture_ctx(self.base)                     # capture BEFORE suggest mutates it
        objs = self.base.suggest(o, cached, i)
        if objs:
            cmap = conf_lookup(self.base, ctx, o)
            tau = getattr(self.base, "tau", 0.05)
            for x in objs:
                x = int(x)
                if x in self.pool or x in cached:
                    continue
                s = self.sizes.get(x)
                if s is None or s > self.cap:
                    continue                              # PFCache would reject; keeps mirror sane
                v = self._value(x, i, cmap.get(x, tau))
                if v <= 0.0:
                    continue
                self.pool[x] = (v, s, i + self.ttl)
                self.densities.append(v / s)

        if not self.pool:
            return []
        # expire stale candidates
        dead = [x for x, (_, _, e) in self.pool.items() if e <= i]
        for x in dead:
            del self.pool[x]
        if not self.pool:
            return []

        # early exit: nothing is affordable yet (the common case -- keeps this tractable at 2M reqs)
        if self.tokens < min(s for _, s, _ in self.pool.values()):
            return []

        chosen = self._select_sep() if self.arm == "sep" else self._select_joint()
        if not chosen:
            return []
        # Emit smallest-first (PFCache BREAKS when tokens < size, so a large item at the front
        # starves smaller ones behind it) AND enforce the HARD total-byte budget: neither arm may
        # exceed the bar's prefetch bytes, so the SEP vs JOINT comparison is iso-bandwidth by
        # construction -- the gap is pure arbitration, never a spend difference.
        kept = []
        for x in sorted(chosen, key=lambda x: self.sizes.get(x, 0)):
            s = self.pool[x][1]
            if self.spent + s > self.budget:
                continue
            kept.append(x)
            self.spent += s
            self.tokens -= s
            del self.pool[x]
        return kept

    def _select_sep(self):
        """Separable threshold rule: fund every candidate whose density v/s clears lambda, taken
        HIGHEST-DENSITY FIRST (the optimal separable ordering). Funding in arrival order instead
        is suboptimal and lets JOINT out-pack SEP even with no real arbitration to do -- which is
        exactly why the clairvoyant control read +14 instead of ~0. Under clairvoyance (v=1) this
        reduces to smallest-first, identical to JOINT's greedy, so F4(clair) -> ~0 as specified."""
        left, out = self.tokens, []
        for x, (v, s, _) in sorted(self.pool.items(), key=lambda kv: -kv[1][0] / kv[1][1]):
            if v / s < self.lam:
                continue
            if s <= left:
                out.append(x)
                left -= s
        return out

    def _select_joint(self):
        """Non-separable: value-maximising subset under the same budget (knapsack)."""
        items = sorted(self.pool.items(), key=lambda kv: -kv[1][0] / kv[1][1])
        budget, used, out = self.tokens, 0.0, []
        for x, (v, s, _) in items:
            if used + s <= budget:
                out.append(x)
                used += s
        # bounded swap improvement: try to admit a skipped item by dropping the lowest-density
        # chosen items, if that raises total value. Captures the integrality gap greedy leaves.
        chosen = set(out)
        for x, (v, s, _) in items:
            if x in chosen or s > budget:
                continue
            drop, freed, dval = [], budget - used, 0.0
            for y in sorted(chosen, key=lambda y: self.pool[y][0] / self.pool[y][1]):
                if freed >= s:
                    break
                drop.append(y)
                freed += self.pool[y][1]
                dval += self.pool[y][0]
            if freed >= s and v > dval:
                for y in drop:
                    chosen.discard(y)
                    used -= self.pool[y][1]
                chosen.add(x)
                used += s
        return list(chosen)


def block_ci(diff, blocks, resamples, seed):
    R = len(diff)
    L = R // blocks
    bm = diff[:blocks * L].reshape(blocks, L).mean(axis=1)
    rng = np.random.default_rng(seed)
    boot = 100.0 * bm[rng.integers(0, blocks, size=(resamples, blocks))].mean(axis=1)
    return np.percentile(boot, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description="F4 / F4' necessity gap (Necessity Ladder rung iv gate)")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--limit", type=int, default=2_000_000)
    ap.add_argument("--cache-frac", type=float, default=0.01)
    ap.add_argument("--pred", choices=sorted(PREDS), required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--mode", choices=("clair", "est"), default="est",
                    help="clair = F4 control (expect ~0); est = F4' (the actual RL gate)")
    ap.add_argument("--ttl", type=int, default=1000, help="requests a candidate stays poolable")
    ap.add_argument("--lam", type=float, default=0.0, help="separable-arm density threshold")
    ap.add_argument("--blocks", type=int, default=1000)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    trace, cap, pos, szs = prep(args.trace, args.limit, args.cache_frac)
    tf = args.train_frac

    def mk():
        return PREDS[args.pred](trace, train_frac=tf, window=args.window, k=args.k, tau=args.tau)

    # bar fixes the iso-BW byte rate, exactly as every other arm in this project does
    bar = PFCache(cap, "s3fifo", mk(), positions=pos, sizes=szs).run(
        trace, cold_train_frac=tf, return_hits=True)
    rate = bar["prefetch_bytes"] / max(trace["n"], 1)

    budget = bar["prefetch_bytes"]                       # the iso-bandwidth ceiling for both arms
    arms = {}
    for arm in ("sep", "joint"):
        sch = PoolArm(mk(), pos, szs, rate, args.mode, arm, args.ttl, args.lam, cap, budget)
        arms[arm] = (PFCache(cap, "s3fifo", sch, positions=pos, sizes=szs, pf_byte_rate=rate).run(
            trace, cold_train_frac=tf, return_hits=True), sch)

    sep, sep_s = arms["sep"]
    joint, joint_s = arms["joint"]
    gap = 100.0 * (joint["ohr"] - sep["ohr"])
    d = joint["hits_series"].astype(np.float64) - sep["hits_series"].astype(np.float64)
    lo, hi = block_ci(d, args.blocks, args.resamples, args.seed)

    label = "F4  (clairvoyant CONTROL)" if args.mode == "clair" else "F4' (causal -- THE GATE)"
    print(f"\n  {args.trace}   mode={args.mode} ttl={args.ttl} lam={args.lam}")
    print(f"  BAR      {args.pred} tau={args.tau} k={args.k}   OHR {bar['ohr']:.4f}")
    print(f"  SEP      OHR {sep['ohr']:.4f}   pf {sep['pf_issued']:,}  bytes {sep['prefetch_bytes']:,}")
    print(f"  JOINT    OHR {joint['ohr']:.4f}   pf {joint['pf_issued']:,}  "
          f"bytes {joint['prefetch_bytes']:,}")
    print(f"\n  {label}")
    print(f"  NECESSITY GAP  {gap:+.2f} pts   95% CI [{lo:+.2f}, {hi:+.2f}]")

    if sep_s.densities:
        q = np.percentile(sep_s.densities, [10, 50, 90])
        print(f"  density v/s quantiles (for calibrating --lam): "
              f"p10 {q[0]:.3e}  p50 {q[1]:.3e}  p90 {q[2]:.3e}")

    inv = []
    sb, jb = sep["prefetch_bytes"], joint["prefetch_bytes"]
    if max(sb, jb) > 1.02 * max(min(sb, jb), 1):
        inv.append(f"arms not iso-bandwidth: sep {sb:,} vs joint {jb:,} bytes -- gap is confounded")
    if args.mode == "clair" and abs(gap) > 1.0:
        inv.append(f"clairvoyant control reads {gap:+.2f} (expected ~0) -- suspect the joint solver")
    print("  INVARIANTS     " + ("all pass" if not inv else "FAIL"))
    for m in inv:
        print(f"    !! {m}")

    if args.mode == "est":
        v = ("RL EARNS A TRAINING BUDGET (gap >= 2)" if lo >= 2 else
             "RL MEASURABLY UNNECESSARY (gap < 1)" if hi < 1 else
             "INCONCLUSIVE (1-2 pts) -- closed-form rule stands as the method")
        print(f"  GATE           {v}\n")
    else:
        print("  (control arm -- read F4' with --mode est for the actual gate)\n")


if __name__ == "__main__":
    main()
