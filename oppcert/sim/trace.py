#!/usr/bin/env python3
"""
oppcert/sim/trace.py -- trace IO + deterministic cache simulator (the P4 env core).

PAPER: replay substrate. oracleGeneral trace IO + the deterministic cache simulator every arm runs on.

W1 scope: everything needed for the parity + gap kill-tests. The RL-facing MDP (eviction /
prefetch heads, features) is deliberately NOT here yet -- the guide is explicit that the env
must match libCacheSim's LRU hit counts EXACTLY before anything is built on top of it.

Trace formats
-------------
oracleGeneral (libCacheSim's binary format, 24 bytes/request, little-endian):
    uint32 real_time | uint64 obj_id | uint32 obj_size | int64 next_access_vtime
`next_access_vtime` is the *virtual time* (request index) of this object's next access, or a
sentinel (<=0 / huge) if never again. That column is what makes Belady free.

The layout is asserted, not assumed: verify_trace_format() re-derives next_access_vtime from the
obj_id sequence with an independent backward pass and requires an exact match. If the struct were
wrong, the derived column could not possibly agree.

Policies
--------
LRU / FIFO / LFU  : classical baselines.
Belady            : evict the object whose next access is FARTHEST in the future.
                    NOTE (matters for the paper's framing): MIN/Belady is *provably optimal* only
                    for UNIFORM object sizes. On variable-size CDN objects, evict-farthest is a
                    strong heuristic reference, NOT a proven optimum (variable-size eviction is
                    NP-hard). On block traces (uniform 4KB) the optimality claim is exact -- which
                    is a second reason the block-I/O traces are the right home for the headline.
"""
from __future__ import annotations

import heapq
import os
from collections import OrderedDict

import numpy as np

# 4 + 8 + 4 + 8 = 24 bytes, packed little-endian
ORACLE_GENERAL = np.dtype([("real_time", "<u4"), ("obj_id", "<u8"),
                           ("obj_size", "<u4"), ("next_vtime", "<i8")])
NEVER = np.int64(1) << 62          # internal sentinel for "no next access"


# --------------------------------------------------------------------------- trace IO
def compute_next_access(obj_ids: np.ndarray) -> np.ndarray:
    """Independent backward pass: next_access[i] = index of the next request to obj_ids[i],
    or NEVER. This is the ground truth we validate the file's own column against."""
    n = len(obj_ids)
    nxt = np.full(n, NEVER, dtype=np.int64)
    last = {}
    for i in range(n - 1, -1, -1):
        o = int(obj_ids[i])
        if o in last:
            nxt[i] = last[o]
        last[o] = i
    return nxt


def load_oracle_general(path: str, limit: int | None = None):
    """-> dict(obj_id, size, next_vtime) as numpy arrays. next_vtime normalised to NEVER."""
    raw = np.fromfile(path, dtype=ORACLE_GENERAL, count=-1 if limit is None else limit)
    if raw.size == 0:
        raise ValueError(f"{path}: empty or wrong struct size (expected {ORACLE_GENERAL.itemsize}B/req)")
    nxt = raw["next_vtime"].astype(np.int64)
    nxt[(nxt < 0) | (nxt >= NEVER)] = NEVER          # libCacheSim uses -1 / huge for 'never'
    return dict(obj_id=raw["obj_id"].astype(np.uint64),
                size=raw["obj_size"].astype(np.int64),
                next_vtime=nxt, n=len(raw))


def verify_trace_format(path: str, limit: int = 200_000) -> dict:
    """W1 loader self-check: re-derive next_access from obj_ids and require an EXACT match with
    the column parsed out of the file. Proves the 24-byte struct layout is right WITHOUT needing
    libCacheSim installed. (The file's next_vtime is absolute; ours is too, within the prefix --
    so only compare entries whose true next access falls inside the prefix.)"""
    t = load_oracle_general(path, limit=limit)
    derived = compute_next_access(t["obj_id"])
    inside = derived < len(t["obj_id"])              # next access lands inside the loaded prefix
    ok = bool(np.array_equal(derived[inside], t["next_vtime"][inside]))
    return dict(ok=ok, n_checked=int(inside.sum()), n=t["n"],
                mismatches=int((derived[inside] != t["next_vtime"][inside]).sum()),
                sizes_positive=bool((t["size"] > 0).all()))


def synth_trace(n_req=200_000, n_obj=20_000, zipf_a=1.1, seq_frac=0.3, run_len=8,
                size_lo=1, size_hi=1, seed=0):
    """Synthetic trace so the env/gates are testable with no download.
    zipf popularity + sequential runs (the runs create the temporal correlation a prefetcher can
    exploit; seq_frac=0 gives a pure-popularity trace with no prefetch signal).
    size_lo=size_hi=1 -> uniform 'block-like' objects (where Belady is exactly optimal)."""
    rng = np.random.default_rng(seed)
    ids = []
    while len(ids) < n_req:
        if rng.random() < seq_frac:                  # a sequential run: o, o+1, o+2, ...
            start = int(rng.zipf(zipf_a) % max(n_obj - run_len, 1))
            ids.extend(range(start, start + int(rng.integers(2, run_len + 1))))
        else:
            ids.append(int(rng.zipf(zipf_a) % n_obj))
    ids = np.asarray(ids[:n_req], dtype=np.uint64)
    sizes = (np.full(n_req, size_lo, dtype=np.int64) if size_lo == size_hi
             else rng.integers(size_lo, size_hi + 1, n_req).astype(np.int64))
    # one size per object (an object has a fixed size in a real trace)
    uniq, inv = np.unique(ids, return_inverse=True)
    per_obj = (np.full(len(uniq), size_lo, dtype=np.int64) if size_lo == size_hi
               else rng.integers(size_lo, size_hi + 1, len(uniq)).astype(np.int64))
    sizes = per_obj[inv]
    return dict(obj_id=ids, size=sizes, next_vtime=compute_next_access(ids), n=n_req)


def footprint_bytes(trace) -> int:
    """Total unique bytes -- cache sizes are expressed as a fraction of THIS (never absolute
    bytes, or cross-trace numbers are meaningless; guide pitfall #4)."""
    uniq, idx = np.unique(trace["obj_id"], return_index=True)
    return int(trace["size"][idx].sum())


# ----------------------------------------------------------------------- cache simulator
class Cache:
    """Deterministic byte-capacity cache. run() returns hit stats. No RL, no prefetch (W1)."""

    def __init__(self, capacity_bytes: int, policy: str = "lru"):
        assert policy in ("lru", "fifo", "lfu", "belady"), policy
        self.cap, self.policy = int(capacity_bytes), policy

    def run(self, trace, warmup_frac: float = 0.05) -> dict:
        cap, pol = self.cap, self.policy
        ids, sizes, nxt = trace["obj_id"], trace["size"], trace["next_vtime"]
        n = trace["n"]
        warm = int(n * warmup_frac)

        cached = OrderedDict()                     # obj_id -> size   (order = LRU/FIFO order)
        used = 0
        freq = {}                                  # lfu
        heap = []                                  # belady/lfu lazy heap
        key = {}                                   # obj_id -> current heap key (staleness check)
        hits = reqs = 0
        hit_b = tot_b = 0

        for i in range(n):
            o, s, nx = int(ids[i]), int(sizes[i]), int(nxt[i])
            counted = i >= warm
            if counted:
                reqs += 1; tot_b += s

            if o in cached:                        # ---- HIT ----
                if counted:
                    hits += 1; hit_b += s
                if pol in ("lru",):
                    cached.move_to_end(o)
                if pol == "lfu":
                    freq[o] = freq.get(o, 0) + 1
                    key[o] = freq[o]; heapq.heappush(heap, (freq[o], i, o))
                if pol == "belady":
                    key[o] = nx; heapq.heappush(heap, (-nx, o))
                continue

            # ---- MISS ----
            if s > cap:                            # object can never fit; serve without caching
                continue
            while used + s > cap and cached:       # evict until it fits
                if pol in ("lru", "fifo"):
                    vo, vs = cached.popitem(last=False)
                    used -= vs; key.pop(vo, None); freq.pop(vo, None)
                elif pol == "lfu":
                    while heap:
                        f, _t, vo = heapq.heappop(heap)
                        if vo in cached and key.get(vo) == f:
                            used -= cached.pop(vo); freq.pop(vo, None); key.pop(vo, None)
                            break
                    else:
                        vo, vs = cached.popitem(last=False); used -= vs
                elif pol == "belady":
                    while heap:
                        negx, vo = heapq.heappop(heap)
                        if vo in cached and key.get(vo) == -negx:
                            used -= cached.pop(vo); key.pop(vo, None)
                            break
                    else:
                        vo, vs = cached.popitem(last=False); used -= vs
            cached[o] = s; used += s
            if pol == "lfu":
                freq[o] = 1; key[o] = 1; heapq.heappush(heap, (1, i, o))
            if pol == "belady":
                key[o] = nx; heapq.heappush(heap, (-nx, o))

        return dict(policy=pol, capacity=cap, requests=reqs, hits=hits,
                    ohr=hits / max(reqs, 1), bhr=hit_b / max(tot_b, 1))


def run_policies(trace, capacity, policies=("lru", "belady")) -> dict:
    return {p: Cache(capacity, p).run(trace) for p in policies}
