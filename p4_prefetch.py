#!/usr/bin/env python3
"""
p4_prefetch.py -- prefetch-aware simulation + the T1 skeleton (guide W2 deliverable).

Answers the paper's money question with ZERO RL: are there hits OUTSIDE Belady's decision space,
and does a realistic prefetcher convert them?

Reference lines produced (guide §4):
    LRU                  floor
    LRU + Markov-1       realistic "separate-combined": a tuned evictor + a tuned prefetcher,
                         run together. THIS is the baseline the joint thesis must beat (A1).
    Belady               eviction-only optimum. EXACT on uniform-size block traces.
    Belady + Markov-1    oracle eviction + REALISTIC prefetch. Reported as a BOUND, not a policy:
                         under oracle eviction a useless prefetch has next_access=NEVER and is
                         evicted immediately at zero cost, so this construction FLATTERS prefetch.
                         It measures "how many hits exist outside Belady's space if eviction were
                         perfect" -- the structural headroom, not an achievable result.
    Prescient-prefetch   oracle eviction + oracle prefetch (guide's upper bound; not OPT --
                         joint OPT is intractable, which is the thesis).

Prefetch accounting (guide pitfall #2, defined once, unit-tested in selftest()):
    * a prefetched object that is later REQUESTED while still cached counts as a HIT;
    * every prefetch counts its bytes as origin traffic, hit or not;
    * a prefetch evicted before use is WASTED: it cost bandwidth AND cache space.
No temporal leakage: the Markov-1 table is built on the trace PREFIX only (guide pitfall #1).
"""
from __future__ import annotations

import heapq
from bisect import bisect_right
from collections import Counter, OrderedDict, defaultdict

import numpy as np

from p4_cache import NEVER

# --------------------------------------------------------------- oracle index structures
def build_obj_positions(trace) -> dict:
    """obj_id -> sorted np.array of request indices. Lets us answer 'next access after t' exactly
    for ANY object -- including one we are about to prefetch (which has no current request)."""
    pos = defaultdict(list)
    for i, o in enumerate(trace["obj_id"]):
        pos[int(o)].append(i)
    return {o: np.asarray(v, dtype=np.int64) for o, v in pos.items()}


def oracle_next(positions, obj: int, t: int) -> int:
    """Exact next access strictly after t, else NEVER."""
    p = positions.get(obj)
    if p is None:
        return int(NEVER)
    j = bisect_right(p, t)
    return int(p[j]) if j < len(p) else int(NEVER)


def build_obj_sizes(trace) -> dict:
    """obj_id -> size. A real cache knows object size from metadata before fetching, so using
    this for a not-yet-seen prefetch candidate is realistic, not an oracle."""
    out = {}
    for o, s in zip(trace["obj_id"], trace["size"]):
        out[int(o)] = int(s)
    return out


# ------------------------------------------------------------------------- prefetchers
class NoPrefetch:
    name = "none"
    def suggest(self, o, cached, i):
        return ()


class Markov1:
    """First-order association prefetcher: succ[o] -> top-M objects seen within `window` after o.
    Built on the trace PREFIX only (no leakage). Suggests the top-k unseen-in-cache successors
    whose confidence (count / total) clears tau."""
    name = "markov1"

    def __init__(self, trace, train_frac=0.5, window=16, top_m=16, k=2, tau=0.05):
        self.k, self.tau = k, tau
        ids = trace["obj_id"]
        cut = int(len(ids) * train_frac)
        succ = defaultdict(Counter)
        for i in range(cut):
            o = int(ids[i])
            for j in range(i + 1, min(i + 1 + window, cut)):
                succ[o][int(ids[j])] += 1
        self.table = {}
        for o, c in succ.items():
            tot = sum(c.values())
            self.table[o] = [(x, n / tot) for x, n in c.most_common(top_m) if n / tot >= tau]

    def suggest(self, o, cached, i):
        out = []
        for x, _conf in self.table.get(int(o), ()):
            if x not in cached:
                out.append(x)
                if len(out) >= self.k:
                    break
        return out


class Prescient:
    """Oracle prefetcher (guide's bound): the <=k soonest-to-be-requested objects not in cache,
    found by looking ahead in the trace. A BOUND CONSTRUCTION, not a deployable policy."""
    name = "prescient"

    def __init__(self, trace, k=2, lookahead=2000):
        self.ids, self.k, self.la = trace["obj_id"], k, lookahead

    def suggest(self, o, cached, i):
        out, seen = [], set()
        for j in range(i + 1, min(i + 1 + self.la, len(self.ids))):
            x = int(self.ids[j])
            if x in cached or x in seen:
                continue
            seen.add(x); out.append(x)
            if len(out) >= self.k:
                break
        return out


# ----------------------------------------------------------------- prefetch-aware cache
class PFCache:
    """Byte-capacity cache with an optional prefetch hook. Deterministic."""

    def __init__(self, capacity_bytes: int, policy: str, prefetcher=None,
                 positions=None, sizes=None, pf_byte_rate=None):
        """pf_byte_rate: bandwidth matching as a TOKEN BUCKET (bytes of prefetch allowed per
        request), None = unlimited.

        Why a rate and not a total budget: a hard total budget is spent GREEDILY at the start of
        the trace and then prefetching stops dead, so the 'matched' arm only helps early and
        scores far below an arm that spreads the same bytes over the whole trace. That is not
        bandwidth matching -- it is front-loading, and it made an ORACLE prefetcher score below a
        heuristic one (the impossible result that exposed the bug)."""
        assert policy in ("lru", "belady")
        self.cap, self.policy = int(capacity_bytes), policy
        self.pf = prefetcher or NoPrefetch()
        self.positions, self.sizes = positions, sizes
        self.pf_rate = pf_byte_rate

    def run(self, trace, warmup_frac=0.05) -> dict:
        cap, pol = self.cap, self.policy
        ids, szs, nxt = trace["obj_id"], trace["size"], trace["next_vtime"]
        n = trace["n"]; warm = int(n * warmup_frac)
        oracle = pol == "belady"

        cached: OrderedDict = OrderedDict()      # obj -> size
        used = 0
        key = {}                                 # obj -> current belady key (next access)
        heap = []
        pf_pending = set()                       # prefetched, not yet used
        hits = reqs = 0
        hit_b = tot_b = 0
        miss_b = pf_b = 0                        # origin traffic components (post-warmup)
        tokens = 0.0                             # token bucket for bandwidth matching
        pf_issued = pf_useful = pf_wasted = 0

        def _evict_one(t):
            nonlocal used, pf_wasted
            while cached:
                if oracle:
                    while heap:
                        negx, vo = heapq.heappop(heap)
                        if vo in cached and key.get(vo) == -negx:
                            used -= cached.pop(vo); key.pop(vo, None)
                            if vo in pf_pending:
                                pf_pending.discard(vo); pf_wasted += 1
                            return
                    vo, vs = cached.popitem(last=False); used -= vs
                else:
                    vo, vs = cached.popitem(last=False); used -= vs
                if vo in pf_pending:
                    pf_pending.discard(vo); pf_wasted += 1
                return

        def _admit(o, s, t, is_pf):
            nonlocal used
            if s > cap:
                return False
            while used + s > cap and cached:
                _evict_one(t)
            if used + s > cap:
                return False
            cached[o] = s; used += s
            if oracle:
                k = oracle_next(self.positions, o, t)
                key[o] = k; heapq.heappush(heap, (-k, o))
            if is_pf:
                pf_pending.add(o)
            return True

        for i in range(n):
            o, s, nx = int(ids[i]), int(szs[i]), int(nxt[i])
            counted = i >= warm
            if counted:
                reqs += 1; tot_b += s

            if o in cached:                                   # ---- HIT ----
                if counted:
                    hits += 1; hit_b += s
                if o in pf_pending:                           # a prefetch paid off
                    pf_pending.discard(o); pf_useful += 1
                if not oracle:
                    cached.move_to_end(o)
                else:
                    key[o] = nx; heapq.heappush(heap, (-nx, o))
            else:                                             # ---- MISS ----
                if counted:
                    miss_b += s                               # fetched from origin
                _admit(o, s, i, is_pf=False)

            # ---- prefetch hook: after serving the request ----
            if self.pf_rate is not None:
                tokens += self.pf_rate                        # token bucket accrues every request
            for x in self.pf.suggest(o, cached, i):
                xs = self.sizes.get(x) if self.sizes else None
                if xs is None or x in cached:
                    continue
                if self.pf_rate is not None and tokens < xs:
                    break                                     # bandwidth-matched: no tokens now
                if _admit(x, xs, i, is_pf=True):
                    pf_issued += 1
                    if self.pf_rate is not None:
                        tokens -= xs
                    if counted:
                        pf_b += xs                            # prefetch bytes ARE origin traffic

        origin_b = miss_b + pf_b
        return dict(policy=pol, prefetcher=self.pf.name, capacity=cap,
                    requests=reqs, hits=hits, ohr=hits / max(reqs, 1),
                    bhr=hit_b / max(tot_b, 1),
                    # NOTE: traffic must be compared to the SAME-POLICY no-prefetch arm, computed
                    # by the caller. Dividing by this arm's own miss_bytes is wrong -- miss_bytes
                    # SHRINKS as prefetching succeeds, so the ratio explodes toward infinity.
                    origin_bytes=origin_b, miss_bytes=miss_b, prefetch_bytes=pf_b,
                    pf_issued=pf_issued, pf_useful=pf_useful, pf_wasted=pf_wasted,
                    pf_precision=pf_useful / max(pf_issued, 1))


# ------------------------------------------------------------------------- selftest
def selftest():
    """Unit-test the prefetch accounting contract (guide pitfall #2) on a hand-built trace.
    Trace: A B A  with a prefetcher that always suggests B after A, capacity = 2 objects.
      i=0 req A  -> miss, admit A; prefetch B (traffic +1)
      i=1 req B  -> B is cached BECAUSE of the prefetch  -> HIT, and pf_useful=1
      i=2 req A  -> HIT
    So: hits=2/3, pf_issued=1, pf_useful=1, precision=1.0, origin = A + prefetched B = 2 bytes."""
    ids = np.array([1, 2, 1], dtype=np.uint64)
    trace = dict(obj_id=ids, size=np.ones(3, dtype=np.int64),
                 next_vtime=np.array([2, NEVER, NEVER], dtype=np.int64), n=3)

    class AlwaysB:
        name = "test"
        def suggest(self, o, cached, i):
            return [2] if int(o) == 1 else []

    r = PFCache(2, "lru", AlwaysB(), sizes={1: 1, 2: 1}).run(trace, warmup_frac=0.0)
    assert r["hits"] == 2, f"prefetched-then-requested must count as a HIT: {r}"
    assert r["pf_issued"] == 1 and r["pf_useful"] == 1, f"prefetch bookkeeping: {r}"
    assert r["prefetch_bytes"] == 1, f"prefetch bytes must count as origin traffic: {r}"
    assert r["origin_bytes"] == 2, f"origin = miss(A)+prefetch(B): {r}"

    # a WASTED prefetch must be charged traffic and produce no hit
    ids2 = np.array([1, 1], dtype=np.uint64)
    t2 = dict(obj_id=ids2, size=np.ones(2, dtype=np.int64),
              next_vtime=np.array([1, NEVER], dtype=np.int64), n=2)
    r2 = PFCache(2, "lru", AlwaysB(), sizes={1: 1, 2: 1}).run(t2, warmup_frac=0.0)
    assert r2["pf_issued"] >= 1 and r2["pf_useful"] == 0, f"unused prefetch must not be useful: {r2}"
    assert r2["prefetch_bytes"] >= 1, f"unused prefetch still costs traffic: {r2}"
    print("[p4_prefetch selftest] PASS -- prefetch accounting contract holds "
          "(hit credit, traffic charge, waste tracking)")


if __name__ == "__main__":
    selftest()
