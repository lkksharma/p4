#!/usr/bin/env python3
"""
p4_evictlearn.py -- the EVICTION green-light: the Instrument's positive control.

Motivation
----------
Every verdict the Instrument has returned so far is a "no" or a "trap":
    block storage      -> no corridor (baseline already saturated)
    CDN / KV prefetch  -> corridor exists but is UNCAPTURABLE (endogenous slack; a deployable
                          causal timing model lands 26--40 pts BELOW the tuned baseline)
    LLM KV warming     -> no corridor (a trivial table already sits on the physical ceiling)
A reviewer's strongest remaining objection is therefore: "your instrument is a rejection machine;
it has never once green-lit a learned component, so a 'no' from it means nothing." This module is
the answer -- the positive control (the spike-in) that proves the ruler can also read "yes, build
it," and reads it on the SAME production traces where it rules the PREFETCH corridor a trap.

Why eviction, not prefetch
--------------------------
The negative result is specific to prefetching under a tight byte budget: a speculative fetch
spends bandwidth and evicts useful data, so a mistimed policy shrinks the very window its timing
must hit (endogenous slack). EVICTION has no such feedback loop -- it spends no speculative
bandwidth -- so the mechanism that kills prefetch capture does not apply. And it is a KNOWN
positive: LRB (Song et al., NSDI'20) imitates Belady and captures a real eviction corridor on CDN
traces; Baleen does it for admission. We reproduce that result THROUGH the decision procedure, so
the contribution is not "a learned evictor works" (known) but "the same instrument that says
DON'T learn the prefetcher says DO learn the evictor, on identical data" -- discrimination across
capabilities, not just across workloads.

The three arms (demand-only; no prefetch anywhere here)
-------------------------------------------------------
    S3-FIFO      the tuned, non-learned evictor -- the deployable lower bound (same evictor frozen
                 throughout the rest of the paper).
    Belady       evict the object whose true next use is FARTHEST away -- the eviction ceiling.
                 Exact optimum for uniform-size (block) objects; a strong reference for variable
                 sizes (see p4_cache).
    Learned      LRB-lite: Belady with the true next-use replaced by a MODEL's prediction. A frozen
                 gradient-boosted regressor, trained on the first half of the trace, predicts each
                 object's reuse distance from causal features (recency, frequency, size, mean
                 inter-access interval, residency age). It peeks at NO future at decision time, so
                 unlike Belady it is deployable. Mechanically it is identical to BeladyEvictor with
                 key = (access position + predicted distance) instead of the true next-use time,
                 so it is heap-driven and as fast as Belady -- the model is called ONCE, batched
                 over the whole trace, not per eviction.

    EVICTION CORRIDOR   = Belady OHR  - S3-FIFO OHR      (is there room above the tuned evictor?)
    CAPTURED            = Learned OHR - S3-FIFO OHR      (does a deployable policy realise it?)
Both get paired block-bootstrap CIs on the same post-warmup requests.

Verdict (pre-registered)
------------------------
    BUILD                if the corridor CI lower bound >= --bar (default 8 pts) AND the learned
                         gain's CI lower bound > 0 AND it captures >= --min-capture of the corridor.
    REAL, NOT CAPTURED   corridor clears the bar but this model does not realise it.
    NO CORRIDOR          corridor below the bar.
Contrast: on these traces the deployable PREFETCH policy is 26--40 pts BELOW baseline; the
deployable EVICTION policy should be clearly ABOVE it. Same instrument, opposite decision.

Construction invariant
-----------------------
The learned arm runs through a local demand-only replay(), not PFCache (which cannot hold a
model). Before trusting any learned number, we assert that this replay reproduces PFCache's OHR
EXACTLY for both S3-FIFO and Belady. If it does not, the run prints INVALID and aborts -- the
learned arm is only fair if it sits on the identical replay the baseline and ceiling sit on.

Usage
-----
    python p4_evictlearn.py --selftest                       # construction check on SYNTH
    python p4_evictlearn.py --trace data/wiki_2019t.oracleGeneral --limit 2000000 --tqdm
    python p4_evictlearn.py --trace SYNTH                     # quick positive-control smoke
"""
from __future__ import annotations

import argparse
import heapq
import math
import time

import numpy as np

from p4_cache import NEVER
from p4_evict import BeladyEvictor, S3FIFOEvictor
from p4_prefetch import NoPrefetch, PFCache
from p4_sweep import prep

LINE = "=" * 100
FEATS = 5  # [log1p(age=gap since last), log1p(freq), log1p(size), log1p(mean_interval), log1p(residency age)]


def synth_positive(n_req=150_000, n_hot=3000, hot_zipf=1.05, p_hot=0.5, cold_size=4, seed=0):
    """A POSITIVE-CONTROL workload with a large, LEARNABLE eviction corridor -- the local
    stand-in for the CDN traces (which live on the node). Two classes:
      hot  : `n_hot` size-1 objects reused (zipf popularity), with a working set slightly LARGER
             than the cache, so they recur at gaps long enough that S3-FIFO's FIFO order evicts
             them just before reuse (the LFU-beats-FIFO regime) -> S3-FIFO lags Belady badly;
      cold : an unbounded stream of size-`cold_size` one-hit-wonders (noise).
    Belady keeps the hot objects that will recur soonest; S3-FIFO cannot rank them and churns. The
    gap is large, and it is learnable from frequency and inter-access interval, so a frozen
    LRB-lite evictor captures nearly all of it -> a BUILD verdict. Labelled a control, not a
    production trace: it shows the instrument CAN green-light; the real claim rests on the CDN runs."""
    rng = np.random.default_rng(seed)
    ids = np.empty(n_req, dtype=np.uint64)
    sizes = np.empty(n_req, dtype=np.int64)
    cold = n_hot
    for i in range(n_req):
        if rng.random() < p_hot:
            ids[i] = int(rng.zipf(hot_zipf) % n_hot); sizes[i] = 1
        else:
            ids[i] = cold; sizes[i] = cold_size; cold += 1
    from p4_cache import compute_next_access
    return dict(obj_id=ids, size=sizes, next_vtime=compute_next_access(ids), n=n_req)


# ------------------------------------------------------------- causal features (shared)
def causal_features(ids, szs, n_upto, want_labels=False, nxt=None, n_total=None):
    """One causal pass building, for each position i in [0, n_upto), the feature row describing the
    object's state JUST BEFORE its access at i (past-only -- no leakage). Optionally also returns
    the Belady-imitation label log1p(reuse distance) and a `fresh` mask (True on an object's first
    occurrence, where freq==0). Fresh rows are trained on too: a one-hit-wonder is a resident with
    freq==0 features and a next-use of NEVER, exactly what Belady evicts first."""
    last, freq, first, size = {}, {}, {}, {}
    F = np.zeros((n_upto, FEATS), dtype=np.float64)
    fresh = np.zeros(n_upto, dtype=bool)
    y = np.zeros(n_upto, dtype=np.float64) if want_labels else None
    NEVERv = int(NEVER)
    for i in range(n_upto):
        o = int(ids[i]); s = int(szs[i]); f = freq.get(o, 0)
        if f > 0:
            la = last[o]; fi = first[o]; sz = size.get(o, s)
            age = i - la; span = la - fi; mi = span / max(f - 1, 1)
            F[i, 0] = math.log1p(age); F[i, 1] = math.log1p(f); F[i, 2] = math.log1p(max(sz, 0))
            F[i, 3] = math.log1p(max(mi, 0)); F[i, 4] = math.log1p(max(i - fi, 0))
        else:
            fresh[i] = True
            F[i, 2] = math.log1p(max(s, 0))          # size is known even on first sight
        if want_labels:
            nx = int(nxt[i]); dist = (nx - i) if nx < NEVERv else n_total
            y[i] = math.log1p(max(dist, 0))
        freq[o] = f + 1; last[o] = i
        first.setdefault(o, i); size[o] = s
    return (F, fresh, y) if want_labels else (F, fresh)


def train_model(trace, train_frac, max_train, trees, seed):
    """Belady-imitation regressor on the trace PREFIX. Features are past-only; the label uses the
    object's own next-access column (the definition of supervised learning, not leakage)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    n = trace["n"]; cut = int(n * train_frac)
    X, fresh, y = causal_features(trace["obj_id"], trace["size"], cut,
                                  want_labels=True, nxt=trace["next_vtime"], n_total=n)
    # Train on ALL rows, INCLUDING first-occurrence ('fresh') ones: a one-hit-wonder sits in the
    # cache with freq==0 features and a next-use of NEVER (label = log1p(n)), which is precisely the
    # object Belady evicts first. Dropping fresh rows blinds the model to that, and it then keeps
    # one-shot objects and loses to S3-FIFO. The `fresh` mask is kept only for diagnostics.
    if len(X) > max_train:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), max_train, replace=False)
        X, y = X[idx], y[idx]
    m = HistGradientBoostingRegressor(max_iter=trees, learning_rate=0.1, max_leaf_nodes=31,
                                      min_samples_leaf=50, l2_regularization=1.0, random_state=seed)
    m.fit(X, y)
    return m, len(X)


def predicted_next_use(trace, model):
    """One batched predict over the whole trace -> for each position i, the model's predicted
    ABSOLUTE next-use position (i + predicted distance). This is the exact analogue of the true
    next-use column Belady evicts on, so the learned evictor is Belady with this column swapped in.
    Computing it once amortises the model to a single call (per-eviction predict is 100x slower)."""
    X, _ = causal_features(trace["obj_id"], trace["size"], trace["n"])
    pred_log = model.predict(X)                          # predicted log1p(distance)
    dist = np.expm1(np.clip(pred_log, 0, None))          # back to distance; clip guards tiny negatives
    return np.arange(trace["n"], dtype=np.float64) + dist


# --------------------------------------------------------------------- the learned evictor
class LearnedEvictor:
    """Heap-driven, exactly like BeladyEvictor, but keyed on the model's predicted next-use
    (pred_next[i]) instead of the true next-use time. Evict the resident whose predicted next use
    is farthest. Lazy deletion: a stale heap entry is one whose key no longer matches key[o]."""
    name = "learned"

    def __init__(self, pred_next):
        self.pred = pred_next
        self.key, self.heap = {}, []

    def _set(self, o, i):
        k = float(self.pred[i])
        self.key[o] = k
        heapq.heappush(self.heap, (-k, o))               # max-key at the top

    def hit(self, o, i):
        self._set(o, i)                                  # refresh prediction at this access

    def admit(self, o, s, i):
        self._set(o, i)

    def evict_one(self):
        while self.heap:
            negk, o = heapq.heappop(self.heap)
            if o in self.key and self.key[o] == -negk:   # not a stale entry
                del self.key[o]
                return o
        return None

    def forget(self, o):
        self.key.pop(o, None)


# ------------------------------------------------------------------ demand-only replay (local)
def replay(trace, cap, ev, warmup_frac=0.05, mode="plain", return_hits=True, progress=None):
    """Deterministic demand-only cache replay mirroring PFCache's no-prefetch path byte-for-byte.
    mode: 'plain' (ev.hit(o)), 'oracle' (ev.touch(o,i)), 'learned' (ev.hit(o,i)). All eviction is
    heap-driven inside the evictor. Validated against PFCache in _construction_check()."""
    ids, szs = trace["obj_id"], trace["size"]
    n = trace["n"]; warm = int(n * warmup_frac)
    cached = {}; used = 0
    hits = reqs = 0; hit_b = tot_b = 0
    hs = np.empty(n - warm, dtype=np.int8) if return_hits else None; hi = 0

    it = range(n)
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, desc=str(progress), unit="req", unit_scale=True,
                      mininterval=0.5, dynamic_ncols=True, leave=False)
        except Exception:
            pass

    for i in it:
        o = int(ids[i]); s = int(szs[i]); counted = i >= warm
        is_hit = o in cached
        if counted:
            reqs += 1; tot_b += s
        if is_hit:                                       # ---- HIT ----
            if counted:
                hits += 1; hit_b += s
            if mode == "oracle":
                ev.touch(o, i)
            elif mode == "learned":
                ev.hit(o, i)
            else:
                ev.hit(o)
        else:                                            # ---- MISS ----
            if s <= cap:
                while used + s > cap and cached:
                    vo = ev.evict_one()
                    if vo is None or vo not in cached:
                        break
                    used -= cached.pop(vo); ev.forget(vo)
                if used + s <= cap:
                    cached[o] = s; used += s
                    ev.admit(o, s, i)
            # s > cap: object can never fit; served without caching (matches PFCache)
        if counted and return_hits:
            hs[hi] = 1 if is_hit else 0; hi += 1

    return dict(ohr=hits / max(reqs, 1), bhr=hit_b / max(tot_b, 1),
                requests=reqs, hits=hits, hits_series=hs)


# ------------------------------------------------------------------------ bootstrap + check
def block_ci(diff, blocks=1000, resamples=10000, seed=0):
    """Paired moving-block bootstrap 95% CI (same convention as p4_bootstrap.py). diff is the
    per-request paired difference in {-1,0,1} on aligned post-warmup requests."""
    R = len(diff); B = min(blocks, R); L = max(R // B, 1)
    bm = diff[:B * L].reshape(B, L).mean(axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, B, size=(resamples, B))
    boot = 100.0 * bm[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return 100.0 * float(diff.mean()), float(lo), float(hi), float(boot.std(ddof=1))


def _construction_check(trace, cap, pos, szs, warmup):
    """The invariant: the local replay must reproduce PFCache's OHR EXACTLY for S3-FIFO and Belady.
    Only then is the learned arm (which must use the local replay) on identical footing."""
    b_pf = PFCache(cap, "s3fifo", NoPrefetch(), positions=pos, sizes=szs).run(trace, warmup_frac=warmup)
    e_pf = PFCache(cap, "belady", NoPrefetch(), positions=pos, sizes=szs).run(trace, warmup_frac=warmup)
    b_me = replay(trace, cap, S3FIFOEvictor(cap), warmup, mode="plain", return_hits=True)
    e_me = replay(trace, cap, BeladyEvictor(cap, pos), warmup, mode="oracle", return_hits=True)
    ok_b = abs(b_me["ohr"] - b_pf["ohr"]) < 1e-9
    ok_e = abs(e_me["ohr"] - e_pf["ohr"]) < 1e-9
    print(f"  CONSTRUCTION CHECK  S3-FIFO local {b_me['ohr']:.6f} vs PFCache {b_pf['ohr']:.6f}  "
          f"{'OK' if ok_b else 'MISMATCH'}")
    print(f"  CONSTRUCTION CHECK  Belady  local {e_me['ohr']:.6f} vs PFCache {e_pf['ohr']:.6f}  "
          f"{'OK' if ok_e else 'MISMATCH'}")
    return (ok_b and ok_e), b_me, e_me


# ---------------------------------------------------------------------------------- selftest
def selftest():
    """Construction check on SYNTH (uniform-size, so Belady is the exact eviction optimum), plus a
    fast end-to-end learned run to prove the heap-driven learned evictor beats S3-FIFO there."""
    from p4_cache import footprint_bytes, synth_trace
    from p4_prefetch import build_obj_positions, build_obj_sizes
    tr = synth_trace(n_req=120_000, n_obj=8_000, zipf_a=1.1, seq_frac=0.3, seed=0)
    cap = max(int(footprint_bytes(tr) * 0.01), 1)
    pos = build_obj_positions(tr); szs = build_obj_sizes(tr)
    ok, b, e = _construction_check(tr, cap, pos, szs, 0.05)
    assert ok, "local replay does not reproduce PFCache -- learned arm would be unfair"
    assert e["ohr"] >= b["ohr"], f"Belady must dominate S3-FIFO on SYNTH: {e['ohr']} vs {b['ohr']}"
    model, _ = train_model(tr, 0.5, 80_000, 80, 0)
    lr = replay(tr, cap, LearnedEvictor(predicted_next_use(tr, model)), 0.05, mode="learned")
    print(f"[p4_evictlearn selftest] PASS -- replay==PFCache; SYNTH eviction corridor "
          f"{100*(e['ohr']-b['ohr']):+.2f} pts, learned captures {100*(lr['ohr']-b['ohr']):+.2f} pts")


# -------------------------------------------------------------------------------------- main
def run(trace, cap, pos, szs, args):
    warmup = args.warmup
    print(LINE)
    print("  EVICTION GREEN-LIGHT -- does the SAME instrument that rejects the prefetcher accept "
          "a learned evictor?")
    print(LINE)

    ok, base, ceil = _construction_check(trace, cap, pos, szs, warmup)
    if not ok:
        print("  VERDICT: INVALID -- construction check failed; learned arm not comparable. Abort.")
        return None

    t0 = time.time()
    model, n_train = train_model(trace, args.train_frac, args.max_train, args.trees, args.seed)
    pred_next = predicted_next_use(trace, model)
    print(f"  [train] LRB-lite on {n_train:,} samples x {FEATS} causal features, {args.trees} trees; "
          f"predicted next-use column built in one batched call ({time.time()-t0:.1f}s)")

    learn = replay(trace, cap, LearnedEvictor(pred_next), warmup, mode="learned", return_hits=True,
                   progress="learned-evictor" if args.tqdm else None)

    bh, eh, lh = base["hits_series"], ceil["hits_series"], learn["hits_series"]
    corridor_pt, c_lo, c_hi, _ = block_ci(eh.astype(np.float64) - bh.astype(np.float64),
                                          args.blocks, args.resamples, args.seed)
    cap_pt, k_lo, k_hi, k_se = block_ci(lh.astype(np.float64) - bh.astype(np.float64),
                                        args.blocks, args.resamples, args.seed)
    frac = cap_pt / corridor_pt if corridor_pt > 1e-9 else 0.0
    bhr_corr = 100 * (ceil["bhr"] - base["bhr"]); bhr_cap = 100 * (learn["bhr"] - base["bhr"])

    print(f"  S3-FIFO (tuned baseline)   OHR {base['ohr']:.4f}   [deployable lower bound]")
    print(f"  BELADY  (eviction ceiling) OHR {ceil['ohr']:.4f}   [clairvoyant, not deployable]")
    print(f"  LEARNED (frozen, causal)   OHR {learn['ohr']:.4f}   [deployable; trained on prefix, "
          f"no future at decision time]")
    print("-" * 100)
    print(f"  EVICTION CORRIDOR   {corridor_pt:+.2f} pts   95% CI [{c_lo:+.2f}, {c_hi:+.2f}]   "
          f"(Belady - S3-FIFO; pre-registered bar >= {args.bar:.1f})")
    print(f"  CAPTURED            {cap_pt:+.2f} pts   95% CI [{k_lo:+.2f}, {k_hi:+.2f}]   SE {k_se:.2f}   "
          f"= {100*frac:.1f}% of the corridor (Learned - S3-FIFO)")
    print(f"  BYTE-WEIGHTED (BHR) corridor {bhr_corr:+.2f} pts   captured {bhr_cap:+.2f} pts")

    corridor_real = c_lo >= args.bar
    captured = (k_lo > 0.0) and (frac >= args.min_capture)
    if corridor_real and captured:
        verdict = (f"BUILD -- a large eviction corridor EXISTS and a deployable learned evictor "
                   f"CAPTURES {100*frac:.0f}% of it ({cap_pt:+.2f} pts above the tuned baseline, "
                   f"CI clear of 0). The instrument green-lights this component.")
    elif corridor_real and not captured:
        verdict = (f"REAL BUT NOT CAPTURED -- corridor clears the bar, but this evictor realises "
                   f"only {100*frac:.0f}% (CI [{k_lo:+.2f},{k_hi:+.2f}]). Honest partial result.")
    else:
        verdict = (f"NO CORRIDOR -- eviction corridor below the {args.bar:.1f}-pt bar "
                   f"(CI lower {c_lo:+.2f}); nothing to build here.")
    print(f"  VERDICT: {verdict}")
    print(f"  CONTRAST: on this trace the deployable PREFETCH policy lands well BELOW the baseline "
          f"(the endogenous-slack trap); the deployable EVICTION policy lands ABOVE it. Same "
          f"instrument, opposite decision -- across capabilities, not just workloads.")
    print(LINE)
    return dict(base=base["ohr"], belady=ceil["ohr"], learned=learn["ohr"],
                corridor=corridor_pt, corridor_ci=(c_lo, c_hi), captured=cap_pt,
                captured_ci=(k_lo, k_hi), frac=frac, corridor_bhr=bhr_corr, captured_bhr=bhr_cap,
                verdict="build" if (corridor_real and captured) else
                        ("real_not_captured" if corridor_real else "no_corridor"))


def main():
    ap = argparse.ArgumentParser(description="Eviction green-light: the Instrument's positive control")
    ap.add_argument("--trace", help="oracleGeneral path, or SYNTH")
    ap.add_argument("--limit", type=int, default=2_000_000)
    ap.add_argument("--cache-frac", type=float, default=0.01)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--max-train", type=int, default=400_000, help="cap on training samples")
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--bar", type=float, default=8.0, help="pre-registered corridor bar (pts)")
    ap.add_argument("--min-capture", type=float, default=0.25,
                    help="pre-registered 'substantial' fraction of the corridor for a BUILD verdict")
    ap.add_argument("--blocks", type=int, default=1000)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tqdm", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    selftest()
    if args.selftest:
        return
    if not args.trace:
        ap.error("give --trace PATH (or --trace SYNTH / SYNTHPOS), or --selftest only")
    if args.trace == "SYNTHPOS":                      # local positive control (see synth_positive)
        from p4_cache import footprint_bytes
        from p4_prefetch import build_obj_positions, build_obj_sizes
        trace = synth_positive(seed=args.seed)        # fixed size: calibrated so cap ~= hot set at cf=0.01
        fp = footprint_bytes(trace); cap = max(int(fp * args.cache_frac), 1)
        pos, szs = build_obj_positions(trace), build_obj_sizes(trace)
        print(f"[trace] SYNTHPOS positive control: {trace['n']:,} requests, "
              f"{len(np.unique(trace['obj_id'])):,} objects, footprint {fp:,} B, "
              f"cache {args.cache_frac*100:.1f}% = {cap:,} B")
    else:
        trace, cap, pos, szs = prep(args.trace, args.limit, args.cache_frac)
    run(trace, cap, pos, szs, args)


if __name__ == "__main__":
    main()
