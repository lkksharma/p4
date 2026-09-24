#!/usr/bin/env python3
"""
oppcert/sim/invariants.py -- the W1/W2 kill-tests. Nothing gets built on top of the env until these pass.

PAPER: the machine-checked invariants of Sec. 1. Loader self-check, LRU and S3-FIFO parity against
libCacheSim, and the headroom/successor-signal kill-tests.

  W1a  FORMAT   loader self-check: re-derive next_access from obj_ids; must match the file EXACTLY.
  W1b  PARITY   our LRU vs an INDEPENDENT reference LRU (always) and vs libCacheSim (if installed).
                Guide: "same trace + LRU must produce identical hit counts. Do not proceed until
                exact match." Hit counts must be EQUAL, not close.
  W2   GAP      LRU -> Belady OHR gap at 0.1/1/10% of footprint. Guide sanity: 10-25 points at 1%;
                < 5 points => no room to win, pick another trace/cache size.
  W2b  PREFETCH (added) Is there any predictable successor structure to prefetch? The guide gates
                the EVICTION headroom but never gates the PREFETCH signal -- yet the whole joint
                thesis ("hits Belady cannot express") dies without it. Measures a first-order
                association table's held-out top-M coverage vs a popularity baseline. This is the
                "no gap, no paper" check, applied to the half of the thesis the guide left ungated.

Usage:
  python -m oppcert.sim.invariants --synth                      # everything, on a synthetic trace (no download)
  python -m oppcert.sim.invariants --trace data/wiki.oracleGeneral --limit 5000000
"""
from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict

import numpy as np

from oppcert.sim.trace import (Cache, load_oracle_general, synth_trace, verify_trace_format,
                      footprint_bytes, compute_next_access)

LINE = "=" * 74


# ------------------------------------------------------------------ W1b reference LRU
def reference_lru(trace, cap, warmup_frac=0.05) -> dict:
    """Deliberately naive, obviously-correct O(n)-per-op LRU. Independent of Cache() -- no shared
    code path -- so an exact hit-count match is real evidence, not a tautology. Slow: prefix only."""
    order, held, used, hits, reqs = [], {}, 0, 0, 0
    warm = int(trace["n"] * warmup_frac)
    for i in range(trace["n"]):
        o, s = int(trace["obj_id"][i]), int(trace["size"][i])
        counted = i >= warm
        if counted:
            reqs += 1
        if o in held:
            if counted:
                hits += 1
            order.remove(o); order.append(o)
            continue
        if s > cap:
            continue
        while used + s > cap and order:
            v = order.pop(0); used -= held.pop(v)
        order.append(o); held[o] = s; used += s
    return dict(requests=reqs, hits=hits, ohr=hits / max(reqs, 1))


LCS_CLASS = {"lru": ["LRU"], "s3fifo": ["S3FIFO", "S3Fifo", "s3fifo", "S3_FIFO"]}


def libcachesim_policy(path, cap, policy="lru", limit=None):
    """Run libCacheSim's own implementation of `policy` on the same trace. Returns None (with a
    reason) if unavailable -- NEVER silently 'passes'."""
    try:
        import libcachesim as lcs
    except Exception as e:
        return None, f"libcachesim not importable ({e}); `pip install libcachesim`"
    ctor = next((getattr(lcs, n) for n in LCS_CLASS[policy] if hasattr(lcs, n)), None)
    if ctor is None:
        avail = [x for x in dir(lcs) if not x.startswith("_")]
        return None, f"no libcachesim class for {policy!r}; tried {LCS_CLASS[policy]}; available: {avail}"
    try:
        reader = lcs.TraceReader(path, lcs.TraceType.ORACLE_GENERAL_TRACE)
        cache = ctor(cache_size=int(cap))
        hits = reqs = 0
        for i, req in enumerate(reader):
            if limit and i >= limit:
                break
            reqs += 1
            hits += int(bool(cache.get(req)))
        return dict(requests=reqs, hits=hits, ohr=hits / max(reqs, 1)), "ok"
    except Exception as e:
        return None, f"libcachesim present but API call failed ({type(e).__name__}: {e})"


def libcachesim_lru(path, cap, limit=None):
    return libcachesim_policy(path, cap, "lru", limit)


def gate_parity_s3fifo(trace, cap, path=None, limit=None):
    """S3-FIFO is a REIMPLEMENTATION (oppcert.sim.evict.S3FIFOEvictor) and S3-FIFO has variants -- it must
    clear the same bar LRU did, or the 'honest A1 bar' it defines is fiction."""
    from oppcert.sim.prefetch import NoPrefetch, PFCache
    print(LINE); print("  W1c  PARITY(S3-FIFO) -- our reimplementation vs libCacheSim"); print(LINE)
    ours = PFCache(cap, "s3fifo", NoPrefetch()).run(trace, warmup_frac=0.0)
    if not path:
        print("  SKIPPED: needs a real trace file (libCacheSim reads from disk)."); return None
    got, why = libcachesim_policy(path, cap, "s3fifo", limit)
    if got is None:
        print(f"  libCacheSim S3-FIFO: SKIPPED -- {why}")
        print("    !! S3-FIFO+Markov-1 is the HONEST A1 BAR. Until this parity runs, that bar is")
        print("       unverified and any 'joint beats separate-combined' claim rests on it.")
        return None
    ok = ours["hits"] == got["hits"]
    print(f"  ours        hits={ours['hits']:,}  OHR={ours['ohr']:.6f}")
    print(f"  libCacheSim hits={got['hits']:,}  OHR={got['ohr']:.6f}")
    print(f"  W1c: {'PASS -- exact match' if ok else 'FAIL -- our S3-FIFO differs; the A1 bar is NOT trustworthy'}")
    return ok


# ------------------------------------------------------------------------ the gates
def gate_format(path, limit):
    print(LINE); print("  W1a  FORMAT -- loader self-check (re-derive next_access, require exact match)"); print(LINE)
    r = verify_trace_format(path, limit=limit)
    print(f"  checked {r['n_checked']:,} of {r['n']:,} requests   mismatches={r['mismatches']}   "
          f"sizes>0={r['sizes_positive']}")
    print(f"  W1a: {'PASS -- 24B oracleGeneral struct confirmed' if r['ok'] else 'FAIL -- struct layout wrong; do NOT trust any number below'}")
    return r["ok"]


def gate_parity(trace, cap, path=None, limit=None):
    print(LINE); print("  W1b  PARITY -- our LRU must EXACTLY match an independent implementation"); print(LINE)
    t0 = time.time(); ours = Cache(cap, "lru").run(trace); t_ours = time.time() - t0
    t0 = time.time(); ref = reference_lru(trace, cap); t_ref = time.time() - t0
    same = (ours["hits"] == ref["hits"]) and (ours["requests"] == ref["requests"])
    print(f"  ours       hits={ours['hits']:,}/{ours['requests']:,}  OHR={ours['ohr']:.6f}  ({t_ours:.1f}s)")
    print(f"  reference  hits={ref['hits']:,}/{ref['requests']:,}  OHR={ref['ohr']:.6f}  ({t_ref:.1f}s)")
    print(f"  exact match (independent impl): {same}")
    lcs_ok = None
    if path:
        got, why = libcachesim_lru(path, cap, limit)
        if got is None:
            print(f"  libCacheSim: SKIPPED -- {why}")
            print("    !! The guide's real unit test is parity vs libCacheSim. A pass here is")
            print("       PROVISIONAL until that runs -- it validates our two impls agree, not")
            print("       that both agree with the field's reference simulator.")
        else:
            # libCacheSim counts from request 0 (no warmup) -> compare on the same basis
            ours_nw = Cache(cap, "lru").run(trace, warmup_frac=0.0)
            lcs_ok = (ours_nw["hits"] == got["hits"])
            print(f"  libCacheSim hits={got['hits']:,}/{got['requests']:,}  OHR={got['ohr']:.6f}")
            print(f"  ours(no warmup) hits={ours_nw['hits']:,}  -> EXACT match: {lcs_ok}")
    ok = same and (lcs_ok is not False)
    print(f"  W1b: {'PASS' if ok else 'FAIL -- fix the simulator before anything else'}"
          f"{' (libCacheSim parity still pending)' if lcs_ok is None else ''}")
    return ok


def gate_gap(trace, fp):
    print(LINE); print("  W2   GAP -- LRU -> Belady headroom (guide: 10-25 OHR pts at 1%; <5 = no room)"); print(LINE)
    print(f"  footprint = {fp:,} bytes")
    print(f"  {'cache':>8s} {'LRU OHR':>10s} {'Belady OHR':>12s} {'gap (pts)':>11s}")
    gap1 = None
    for frac in (0.001, 0.01, 0.10):
        cap = max(int(fp * frac), 1)
        lru = Cache(cap, "lru").run(trace); bel = Cache(cap, "belady").run(trace)
        gap = 100 * (bel["ohr"] - lru["ohr"])
        if abs(frac - 0.01) < 1e-9:
            gap1 = gap
        print(f"  {frac*100:7.1f}% {lru['ohr']:10.4f} {bel['ohr']:12.4f} {gap:11.2f}")
    ok = gap1 is not None and gap1 >= 5.0
    good = gap1 is not None and 10.0 <= gap1 <= 25.0
    print(f"  W2: {'PASS' if ok else 'FAIL'} -- gap@1% = {gap1:.2f} pts"
          f"{'  (in the guide 10-25 band)' if good else '  (outside 10-25; usable but check the trace)' if ok else '  -- NO ROOM TO WIN: pick another trace/cache size'}")
    return ok


def gate_prefetch_signal(trace, window=64, top_m=16, split=0.6):
    """Is the next request predictable from the current object? Build succ[] on the FIRST `split`
    only (no temporal leakage -- guide pitfall #1), score on the held-out suffix."""
    print(LINE); print("  W2b  PREFETCH SIGNAL -- is there successor structure to prefetch at all?"); print(LINE)
    ids = trace["obj_id"]; n = len(ids); cut = int(n * split)
    succ = defaultdict(Counter)
    for i in range(cut):                                   # train: co-occurrence within `window`
        o = int(ids[i])
        for j in range(i + 1, min(i + 1 + window, cut)):
            succ[o][int(ids[j])] += 1
    table = {o: [x for x, _ in c.most_common(top_m)] for o, c in succ.items()}
    pop = [o for o, _ in Counter(int(x) for x in ids[:cut]).most_common(top_m)]
    pop_set = set(pop)
    hit_assoc = hit_pop = seen = 0
    for i in range(cut, n - 1):                            # test: is the NEXT request predicted?
        o, nxt = int(ids[i]), int(ids[i + 1])
        if o not in table:
            continue
        seen += 1
        hit_assoc += int(nxt in table[o])
        hit_pop += int(nxt in pop_set)
    a = hit_assoc / max(seen, 1); p = hit_pop / max(seen, 1)
    print(f"  held-out next-request coverage @top-{top_m} (n={seen:,}):")
    print(f"    association table : {a:.4f}")
    print(f"    popularity baseline: {p:.4f}   (what a prefetcher gets for free)")
    lift = a - p
    ok = (a >= 0.10) and (lift >= 0.05)
    print(f"    lift = {lift:+.4f}")
    print(f"  W2b: {'PASS -- real successor structure; prefetching can add hits Belady cannot' if ok else 'FAIL -- next request is NOT predictable beyond popularity'}")
    if not ok:
        print("    Without prefetch signal the JOINT thesis is dead on this trace regardless of the")
        print("    RL: there are no 'hits outside Belady's decision space' to win. Use a trace with")
        print("    spatial/temporal locality (block I/O: MSR, CloudPhysics) before building anything.")
    return ok


# ---------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="P4 W1/W2 kill-tests")
    p.add_argument("--trace", default=None, help="oracleGeneral binary trace")
    p.add_argument("--synth", action="store_true", help="use a synthetic trace (no download)")
    p.add_argument("--limit", type=int, default=2_000_000, help="max requests to load")
    p.add_argument("--parity-prefix", type=int, default=100_000,
                   help="requests used for the O(n) reference-LRU parity check")
    p.add_argument("--cache-frac", type=float, default=0.01, help="parity cache size (frac of footprint)")
    p.add_argument("--seq-frac", type=float, default=0.3, help="synth: fraction of sequential runs")
    a = p.parse_args()

    if not a.trace and not a.synth:
        p.error("give --trace PATH or --synth")

    ok_fmt = True
    if a.trace:
        ok_fmt = gate_format(a.trace, limit=min(a.limit, 200_000))
        trace = load_oracle_general(a.trace, limit=a.limit)
        print(f"\n[trace] {a.trace}: {trace['n']:,} requests, "
              f"{len(np.unique(trace['obj_id'])):,} unique objects")
    else:
        trace = synth_trace(n_req=a.limit if a.limit < 500_000 else 200_000, seq_frac=a.seq_frac)
        print(f"[trace] SYNTHETIC: {trace['n']:,} requests, "
              f"{len(np.unique(trace['obj_id'])):,} unique objects, seq_frac={a.seq_frac}")

    fp = footprint_bytes(trace)
    cap = max(int(fp * a.cache_frac), 1)

    # parity on a prefix (reference LRU is O(n) per op)
    pre = {k: (v[:a.parity_prefix] if isinstance(v, np.ndarray) else v) for k, v in trace.items()}
    pre["n"] = min(a.parity_prefix, trace["n"])
    pre_cap = max(int(footprint_bytes(pre) * a.cache_frac), 1)
    ok_par = gate_parity(pre, pre_cap, path=a.trace, limit=pre["n"])
    ok_s3 = gate_parity_s3fifo(pre, pre_cap, path=a.trace, limit=pre["n"])

    ok_gap = gate_gap(trace, fp)
    ok_pre = gate_prefetch_signal(trace)

    print(LINE)
    s3txt = "PASS" if ok_s3 else ("FAIL" if ok_s3 is False else "skipped")
    print(f"  W1a FORMAT {'PASS' if ok_fmt else 'FAIL'} | W1b PARITY {'PASS' if ok_par else 'FAIL'} "
          f"| W1c S3FIFO {s3txt} | W2 GAP {'PASS' if ok_gap else 'FAIL'} "
          f"| W2b PREFETCH {'PASS' if ok_pre else 'FAIL'}")
    allok = ok_fmt and ok_par and ok_gap and ok_pre and (ok_s3 is not False)
    print(f"  -> {'ALL GATES PASS: build the RL env on this trace' if allok else 'STOP: a gate failed. Do not build on this trace/config.'}")
    print(LINE)


if __name__ == "__main__":
    main()
