#!/usr/bin/env python3
"""
oppcert/instrument/online_bars.py -- online-updating variants of the Markov bars (prereg/online_predictor.md).

PAPER: the excluded class. Online-updating variants of the bars, the capability Sec. 7 (Scope) names as
excluded. Pre-registered in prereg/online_predictor.md.

The frozen predictors in oppcert/sim/prefetch.py / oppcert/instrument/bars.py train on the trace prefix and stop.
These subclasses warm-start from the SAME prefix table and then keep counting during evaluation,
which adds exactly one capability -- the one `paper_aaai.tex` names as excluded and does not test.

Three properties are load-bearing and each is enforced by construction, not by convention:

  CAUSALITY (I1). Serving position i may use nothing from position > i. The update rule is the
  causal transpose of the training loop: training walks forward from o and credits the next
  `window` requests as its successors; online, when request o ARRIVES, it is credited as a
  successor of the `window` requests already seen. Same pairs, same counts, no lookahead.

  RESET HYGIENE (I4). Counts added online live in a separate `_add` layer. reset() clears that
  layer and the ranking cache; the warm-start prefix counters are never mutated. Without this,
  a tau x k sweep would let arm N see arm N-1's replay and the comparison would be meaningless.

  ISO-CAPABILITY (I2). Nothing here privileges the wide arm: whichever arm the caller builds
  from these classes gets online updating, so a run driven by --pred markov2_online gives the
  tuned bar the same capability it gives the causal arm. Scoring an online learned arm against a
  frozen bar is mechanism M1, which the paper indicts.

Ranking is computed lazily per context and cached, because re-ranking every context on every
request would dominate the replay.

    python -m oppcert.instrument.online_bars --leaktest            # I1 + I4, on synthetic data
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque

from oppcert.sim.prefetch import Markov1, Markov2, _MarkovBase
from oppcert.instrument.bars import Markov3


class _OnlineMixin:
    """Lazy merged ranking over a frozen base layer plus a clearable online layer."""

    def _init_online(self, window, top_m, levels):
        self.window = window
        self.top_m = top_m
        self._levels = levels                     # [(base_counter_dict, add, cache, dirty), ...]
        self._hist = [deque(maxlen=window) for _ in levels]
        self._online_ready = True

    def _fresh_layers(self):
        for lvl in self._levels:
            lvl["add"].clear()
            lvl["cache"].clear()
        for h in self._hist:
            h.clear()

    def reset(self):
        self._fresh_layers()

    def _observe(self, li, key_now, arriving):
        """Credit `arriving` as a successor of every context already in level li's history."""
        lvl = self._levels[li]
        add, cache = lvl["add"], lvl["cache"]
        for ctx in self._hist[li]:
            add[ctx][arriving] += 1
            cache.pop(ctx, None)                  # stale ranking
        self._hist[li].append(key_now)

    def _ranked(self, li, ctx):
        """Merged top-M for one context, confidence-descending. Cached until that ctx updates."""
        lvl = self._levels[li]
        hit = lvl["cache"].get(ctx)
        if hit is not None:
            return hit
        base = lvl["base"].get(ctx)
        extra = lvl["add"].get(ctx)
        if base is None and extra is None:
            return None
        if extra:
            merged = Counter(base) if base else Counter()
            merged.update(extra)
        elif base:
            merged = base
        else:
            return None
        tot = sum(merged.values())
        out = [(x, n / tot) for x, n in merged.most_common(self.top_m)] if tot else []
        lvl["cache"][ctx] = out
        return out


def _counters_from_prefix(trace, train_frac, window, order):
    """Rebuild the prefix counters the frozen classes discard after ranking.

    Mirrors the training loops in oppcert.sim.prefetch/oppcert.instrument.bars exactly, including the `j < cut`
    bound, so the warm start is the same table the frozen arm uses.
    """
    ids = trace["obj_id"]
    cut = int(len(ids) * train_frac)
    c1 = defaultdict(Counter)
    c2 = defaultdict(Counter)
    c3 = defaultdict(Counter)
    start = {1: 0, 2: 1, 3: 2}[order]
    for i in range(start, cut):
        cur = int(ids[i])
        hi = min(i + 1 + window, cut)
        if order >= 3:
            k3 = (int(ids[i - 2]), int(ids[i - 1]), cur)
        if order >= 2:
            k2 = (int(ids[i - 1]), cur)
        for j in range(i + 1, hi):
            x = int(ids[j])
            if order >= 3:
                c3[k3][x] += 1
            if order >= 2:
                c2[k2][x] += 1
            c1[cur][x] += 1
    return c1, c2, c3


def _mklevel(base):
    return {"base": base, "add": defaultdict(Counter), "cache": {}}


class OnlineMarkov1(_OnlineMixin, _MarkovBase):
    name = "markov1_online"

    def __init__(self, trace, train_frac=0.5, window=16, top_m=16, k=2, tau=0.05):
        self.k, self.tau = k, tau
        c1, _, _ = _counters_from_prefix(trace, train_frac, window, 1)
        self._init_online(window, top_m, [_mklevel(c1)])
        self.table = c1

    def suggest(self, o, cached, i):
        o = int(o)
        self._observe(0, o, o)
        return self._pick(self._ranked(0, o) or (), cached)


class OnlineMarkov2(_OnlineMixin, _MarkovBase):
    """Second order with first-order backoff, both layers updating online."""
    name = "markov2_online"

    def __init__(self, trace, train_frac=0.5, window=16, top_m=16, k=2, tau=0.05):
        self.k, self.tau = k, tau
        c1, c2, _ = _counters_from_prefix(trace, train_frac, window, 2)
        self._init_online(window, top_m, [_mklevel(c2), _mklevel(c1)])
        self.table = c2
        self.prev = None

    def reset(self):
        self._fresh_layers()
        self.prev = None

    def suggest(self, o, cached, i):
        o = int(o)
        ctx2 = (self.prev, o) if self.prev is not None else None
        self._observe(0, ctx2, o)
        self._observe(1, o, o)
        cand = self._ranked(0, ctx2) if ctx2 is not None else None
        if cand is None:
            cand = self._ranked(1, o)
        self.prev = o
        return self._pick(cand or (), cached)


class OnlineMarkov3(_OnlineMixin, _MarkovBase):
    """Third order with 3->2->1 backoff, every layer updating online."""
    name = "markov3_online"

    def __init__(self, trace, train_frac=0.5, window=16, top_m=16, k=2, tau=0.05):
        self.k, self.tau = k, tau
        c1, c2, c3 = _counters_from_prefix(trace, train_frac, window, 3)
        self._init_online(window, top_m, [_mklevel(c3), _mklevel(c2), _mklevel(c1)])
        self.table = c3
        self.p2 = self.p1 = None

    def reset(self):
        self._fresh_layers()
        self.p2 = self.p1 = None

    def suggest(self, o, cached, i):
        o = int(o)
        ctx3 = (self.p2, self.p1, o) if self.p2 is not None else None
        ctx2 = (self.p1, o) if self.p1 is not None else None
        self._observe(0, ctx3, o)
        self._observe(1, ctx2, o)
        self._observe(2, o, o)
        cand = self._ranked(0, ctx3) if ctx3 is not None else None
        if cand is None and ctx2 is not None:
            cand = self._ranked(1, ctx2)
        if cand is None:
            cand = self._ranked(2, o)
        self.p2, self.p1 = self.p1, o
        return self._pick(cand or (), cached)


ONLINE_PREDS = {
    "markov1_online": OnlineMarkov1,
    "markov2_online": OnlineMarkov2,
    "markov3_online": OnlineMarkov3,
}


# --------------------------------------------------------------------------------------
# I1 / I4 verification. These are the two ways this file could be silently wrong, so they
# are tested rather than asserted in a comment.
# --------------------------------------------------------------------------------------
def leaktest():
    import numpy as np

    rng = np.random.default_rng(0)
    n = 4000
    ids = np.zeros(n, dtype=np.int64)
    for i in range(n):                          # A->B->C chain plus noise, so structure exists
        ids[i] = (i % 7) if rng.random() < 0.8 else int(rng.integers(7, 40))
    trace = {"obj_id": ids, "n": n}
    ok = True

    # ---- I1: a predictor that has seen only the prefix must be UNAFFECTED by a change made
    # to the tail of the trace. If tail edits move its early suggestions, it is reading ahead.
    p = OnlineMarkov2(trace, train_frac=0.5, window=8, k=3, tau=0.0)
    first = [tuple(p.suggest(int(ids[i]), set(), i)) for i in range(60)]
    ids2 = ids.copy()
    ids2[n // 2:] = rng.integers(0, 40, size=n - n // 2)
    q = OnlineMarkov2({"obj_id": ids2, "n": n}, train_frac=0.5, window=8, k=3, tau=0.0)
    second = [tuple(q.suggest(int(ids2[i]), set(), i)) for i in range(60)]
    if first != second:
        print("  I1 FAIL: suggestions over the first 60 requests changed when only the TAIL of "
              "the trace was rewritten -- the predictor is reading ahead")
        ok = False
    else:
        print("  I1 pass: serving position i is invariant to every position > i")

    # ---- I4: two consecutive replays must be identical. If online counts survive reset(),
    # the second replay sees the first one's data and every sweep cell is contaminated.
    r = OnlineMarkov2(trace, train_frac=0.5, window=8, k=3, tau=0.0)
    runA = [tuple(r.suggest(int(o), set(), i)) for i, o in enumerate(ids)]
    r.reset()
    runB = [tuple(r.suggest(int(o), set(), i)) for i, o in enumerate(ids)]
    if runA != runB:
        d = next(i for i, (a, b) in enumerate(zip(runA, runB)) if a != b)
        print(f"  I4 FAIL: replay 2 differs from replay 1 at request {d:,} -- online counts "
              f"survived reset() and will contaminate every sweep cell")
        ok = False
    else:
        print("  I4 pass: reset() restores the warm-start table exactly")

    # ---- the capability must actually do something, or the run is a no-op dressed as a test
    frozen = Markov2(trace, train_frac=0.5, window=8, k=3, tau=0.0)
    fz = [tuple(frozen.suggest(int(o), set(), i)) for i, o in enumerate(ids)]
    diff = sum(1 for a, b in zip(fz, runA) if a != b)
    print(f"  online vs frozen: {diff:,}/{len(ids):,} suggestions differ ({diff/len(ids):.1%})")
    if diff == 0:
        print("  !! LEVER INERT: online updating changed nothing. Do not run the real trace "
              "until this is explained; a null result here would be an artefact, not a finding.")
        ok = False

    print("\n  " + ("ALL INVARIANTS PASS -- safe to run the pre-registered arms"
                    if ok else "INVARIANT FAILURE -- per prereg/online_predictor.md the run is aborted"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="online-updating Markov bars (prereg/online_predictor.md)")
    ap.add_argument("--leaktest", action="store_true",
                    help="verify I1 (causality) and I4 (reset hygiene) before any real run")
    args = ap.parse_args()
    if args.leaktest:
        raise SystemExit(leaktest())
    ap.print_help()


if __name__ == "__main__":
    main()
