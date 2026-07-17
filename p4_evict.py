#!/usr/bin/env python3
"""
p4_evict.py -- pluggable eviction policies.

The cache (p4_prefetch.PFCache) owns bytes, hit accounting and the prefetch hook; an Evictor owns
ONLY the eviction order. This split is what lets a prefetcher be combined with ANY evictor -- the
whole point of the "separate-combined" baseline the joint thesis must beat.

Contract:
    hit(o)              object o was requested and is resident
    admit(o, size, t)   o just entered the cache (by request OR by prefetch)
    evict_one() -> o    choose + release ONE resident object; internal promotion/demotion
                        (S3-FIFO moves objects between queues) happens inside and does NOT return.
                        Returns None only if nothing is resident.
    forget(o)           o left the cache; drop metadata

S3-FIFO (Yang et al., SOSP'23 "FIFO queues are all you need for cache eviction"):
    small FIFO S (10% of bytes) + main FIFO M (90%) + ghost G (ids evicted from S).
    hit -> freq = min(freq+1, 3).
    miss+admit -> if id in ghost, insert to M, else insert to S.
    evict -> if S over its share: pop S head; freq>1 promotes to M, else evict + record in ghost.
             else: pop M head; freq>0 demotes (freq-1, reinsert at tail), else evict.
MUST be parity-checked against libCacheSim's S3FIFO before use (p4_gates --parity-s3fifo);
this is a reimplementation, and S3-FIFO has variants.
"""
from __future__ import annotations

import heapq
from collections import OrderedDict, deque

from p4_cache import NEVER


class LRUEvictor:
    name = "lru"

    def __init__(self, cap):
        self.order = OrderedDict()          # obj -> True, LRU at the front

    def hit(self, o):
        self.order.move_to_end(o)

    def admit(self, o, size, t):
        self.order[o] = True

    def evict_one(self):
        if not self.order:
            return None
        o, _ = self.order.popitem(last=False)
        return o

    def forget(self, o):
        self.order.pop(o, None)


class BeladyEvictor:
    """Evict the object whose next access is FARTHEST in the future. Exact optimum for
    uniform-size objects; a strong reference (not a proven optimum) for variable sizes."""
    name = "belady"

    def __init__(self, cap, positions=None):
        self.positions = positions
        self.key, self.heap = {}, []

    def _set(self, o, t):
        from p4_prefetch import oracle_next
        k = oracle_next(self.positions, o, t)
        self.key[o] = k
        heapq.heappush(self.heap, (-k, o))

    def hit(self, o, t=None):
        pass                                 # PFCache calls admit-style refresh via touch()

    def touch(self, o, t):
        self._set(o, t)

    def admit(self, o, size, t):
        self._set(o, t)

    def evict_one(self):
        while self.heap:
            negk, o = heapq.heappop(self.heap)
            if o in self.key and self.key[o] == -negk:      # not a stale heap entry
                del self.key[o]
                return o
        return None

    def forget(self, o):
        self.key.pop(o, None)


class S3FIFOEvictor:
    name = "s3fifo"

    def __init__(self, cap, small_frac=0.10, ghost_mult=1.0):
        self.cap = cap
        self.s_cap = cap * small_frac
        self.S, self.M = deque(), deque()
        self.G = OrderedDict()                              # ghost: obj -> True (FIFO)
        self.freq, self.sz, self.loc = {}, {}, {}
        self.s_bytes = self.m_bytes = 0
        self.ghost_cap = max(int(cap * ghost_mult), 1)      # ghost sized in BYTES of main
        self.g_bytes = 0

    def hit(self, o):
        self.freq[o] = min(self.freq.get(o, 0) + 1, 3)

    def admit(self, o, size, t):
        self.sz[o] = size
        self.freq[o] = 0
        if o in self.G:                                     # seen before -> straight to main
            self.g_bytes -= self.sz.get(o, size); del self.G[o]
            self.M.append(o); self.loc[o] = "M"; self.m_bytes += size
        else:
            self.S.append(o); self.loc[o] = "S"; self.s_bytes += size

    def _to_ghost(self, o):
        self.G[o] = True
        self.g_bytes += self.sz.get(o, 0)
        while self.g_bytes > self.ghost_cap and self.G:
            g, _ = self.G.popitem(last=False)
            self.g_bytes -= self.sz.get(g, 0)

    def evict_one(self):
        while True:
            if self.s_bytes >= self.s_cap and self.S:
                o = self.S.popleft(); self.s_bytes -= self.sz.get(o, 0)
                if self.freq.get(o, 0) > 1:                 # hot -> promote to main
                    self.M.append(o); self.loc[o] = "M"; self.m_bytes += self.sz.get(o, 0)
                    continue
                self._to_ghost(o); self.loc.pop(o, None)
                return o
            if self.M:
                o = self.M.popleft(); self.m_bytes -= self.sz.get(o, 0)
                if self.freq.get(o, 0) > 0:                 # demote, keep
                    self.freq[o] -= 1
                    self.M.append(o); self.m_bytes += self.sz.get(o, 0)
                    continue
                self.loc.pop(o, None)
                return o
            if self.S:                                      # main empty; S under its share
                o = self.S.popleft(); self.s_bytes -= self.sz.get(o, 0)
                self._to_ghost(o); self.loc.pop(o, None)
                return o
            return None

    def forget(self, o):
        self.freq.pop(o, None); self.loc.pop(o, None)
        # size kept: the ghost queue needs it to account bytes


def make_evictor(policy, cap, positions=None):
    if policy == "lru":
        return LRUEvictor(cap)
    if policy == "belady":
        return BeladyEvictor(cap, positions)
    if policy == "s3fifo":
        return S3FIFOEvictor(cap)
    raise ValueError(f"unknown policy {policy!r}")
