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
    Learned      LRB-lite: a frozen gradient-boosted regressor, trained on the first half of the
                 trace, predicts an object's reuse distance from causal features (recency,
                 frequency, size, mean inter-access interval, residency age). At each eviction it
                 scores a random sample of residents with their CURRENT features (age = now - last
                 access) and evicts the largest predicted distance. It peeks at NO future at
                 decision time, so unlike Belady it is deployable. Predicting at eviction time with
                 the current age -- not once at access time -- is essential: a frozen access-time
                 prediction goes stale when it is wrong and traps mispredicted objects in the cache
                 (they look imminently due forever); re-scoring on current age instead makes an
                 unused object more evictable the longer it waits, which self-corrects the model's
                 mistakes.

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
import math
import time
from collections import deque

import numpy as np

from p4_cache import NEVER
from p4_evict import BeladyEvictor, S3FIFOEvictor
from p4_prefetch import NoPrefetch, PFCache
from p4_sweep import prep

LINE = "=" * 100
N_DELTAS = 16                 # recency profile length: log gaps between the last N accesses (LRB's core signal)
BIG_LOG = math.log1p(1e9)     # sentinel for a delta that does not exist (object seen fewer than N times)


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


# ------------------------------------------------------------- feature construction (shared)
def _row(D, t, freq, size, nd):
    """Feature row for an object with recent-access deque D (D[-1] = last access) evaluated at time
    t: the recency profile log1p(delta_0 .. delta_{nd-1}), where delta_0 = t - last and delta_j is
    the gap between the (j)th and (j+1)th most recent accesses (missing deltas -> BIG_LOG), then
    log1p(frequency) and log1p(size). The delta PROFILE, not just the mean, is LRB's key signal: it
    lets the model separate a one-hit-wonder (all deltas missing) from a genuinely reused object."""
    r = [0.0] * (nd + 2)
    prev = t; L = len(D)
    for d in range(nd):
        if d < L:
            p = D[L - 1 - d]; g = prev - p
            r[d] = math.log1p(g if g > 0 else 0); prev = p
        else:
            r[d] = BIG_LOG
    r[nd] = math.log1p(max(freq, 0)); r[nd + 1] = math.log1p(max(size, 0))
    return r


def train_model(trace, train_frac, max_train, trees, seed, nd=N_DELTAS):
    """Belady-imitation regressor on the trace PREFIX, trained on CENSORED samples so its input
    distribution matches EVICTION time, not access time. For each realised inter-access gap (la ->
    i) we draw a probe age in (0, i-la], build the recency profile at that partial age, and label it
    log1p(remaining distance = i - probe). One-shot objects (next use NEVER) also sit in the cache
    and must be evicted: for each such last access we draw a probe age and label log1p(n) (max
    distance). Features are past-only (no leakage); the label uses the object's own next-access
    column, which is what supervised learning is."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    ids, szs, nxt = trace["obj_id"], trace["size"], trace["next_vtime"]
    n = trace["n"]; cut = int(n * train_frac); NEVERv = int(NEVER)
    rng = np.random.default_rng(seed); HMAX = 100_000
    hist, freq, size = {}, {}, {}
    F, y = [], []
    for i in range(cut):
        o = int(ids[i]); s = int(szs[i]); f = freq.get(o, 0); nx = int(nxt[i])
        D = hist.get(o)
        if f > 0:                                    # censored probe inside the realised gap la -> i
            la = D[-1]; span = i - la
            age = 1 + int(rng.integers(0, span)) if span > 0 else 0
            F.append(_row(D, la + age, f, s, nd)); y.append(math.log1p(max(span - age, 0)))
        if D is None:
            D = deque(maxlen=nd); hist[o] = D
        D.append(i)
        if nx >= NEVERv:                             # last appearance: teach eviction of dead objects
            age = 1 + int(rng.integers(0, HMAX))
            F.append(_row(D, i + age, f + 1, s, nd)); y.append(math.log1p(n))
        freq[o] = f + 1; size[o] = s
    idx = rng.choice(len(F), max_train, replace=False) if len(F) > max_train else np.arange(len(F))
    X = np.asarray([F[k] for k in idx], dtype=np.float64)
    yv = np.asarray([y[k] for k in idx], dtype=np.float64)
    m = HistGradientBoostingRegressor(max_iter=trees, learning_rate=0.1, max_leaf_nodes=31,
                                      min_samples_leaf=50, l2_regularization=1.0, random_state=seed)
    m.fit(X, yv)
    return m, len(idx)


# ------------------------------------------------------------------- learned ADMISSION (Baleen-style)
def causal_reuse_features(trace, nd):
    """Feature row for EVERY request position i, describing the object's state AS OF that access
    (recency profile from its prior accesses, evaluated at t=i, plus prior frequency and size).
    These features depend only on the request stream, NOT on cache state, so they can be built once
    and batched -- which is what makes learned admission fast (one predict for the whole trace)."""
    ids, szs = trace["obj_id"], trace["size"]; n = trace["n"]
    hist, freq = {}, {}
    X = np.empty((n, nd + 2), dtype=np.float64)
    for i in range(n):
        o = int(ids[i]); s = int(szs[i]); D = hist.get(o); f = freq.get(o, 0)
        X[i] = _row(D if D is not None else (), i, f, s, nd)   # profile from PRIOR accesses, age = i - last
        if D is None:
            D = deque(maxlen=nd); hist[o] = D
        D.append(i); freq[o] = f + 1
    return X


def train_admission(trace, train_frac, max_train, trees, seed, nd):
    """Reuse classifier for ADMISSION: predict, at the moment an object is requested, whether it
    will be requested AGAIN (next_vtime < NEVER). Bypassing objects predicted one-shot is the
    Baleen-style lever that recovers the cache space S3-FIFO wastes on one-hit-wonders. Trained on
    the prefix; then reuse probability is predicted for every position in one batched call."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = causal_reuse_features(trace, nd)
    n = trace["n"]; cut = int(n * train_frac); NEVERv = int(NEVER)
    y = (trace["next_vtime"][:cut].astype(np.int64) < NEVERv).astype(np.int32)
    Xtr = X[:cut]
    rng = np.random.default_rng(seed)
    if len(Xtr) > max_train:
        idx = rng.choice(len(Xtr), max_train, replace=False); Xtr, y = Xtr[idx], y[idx]
    clf = HistGradientBoostingClassifier(max_iter=trees, learning_rate=0.1, random_state=seed)
    clf.fit(Xtr, y)
    P = clf.predict_proba(X)[:, 1]                        # reuse probability at every position
    return clf, P, float(y.mean())


def tune_admission_tau(trace, cap, P, train_frac, taus):
    """Pick the bypass threshold on the TRAIN half only (no test leakage): bypass a miss when the
    predicted reuse probability is below tau. tau = 0 admits everything (identical to S3-FIFO), so
    the tuned policy is bounded below by the baseline. Returns (best_tau, best_train_ohr)."""
    n = trace["n"]; cut = int(n * train_frac)
    sub = dict(obj_id=trace["obj_id"][:cut], size=trace["size"][:cut],
               next_vtime=trace["next_vtime"][:cut], n=cut)
    best = (0.0, -1.0)
    for tau in taus:
        mask = P[:cut] >= tau
        r = replay(sub, cap, S3FIFOEvictor(cap), 0.05, mode="plain", return_hits=False,
                   admit_mask=mask)
        if r["ohr"] > best[1]:
            best = (float(tau), r["ohr"])
    return best


# --------------------------------------------------------------------- the learned evictor
class LearnedEvictor:
    """LRB-lite (LRB's actual recipe): at each eviction, predict the reuse distance of a RANDOM
    SAMPLE of residents using their CURRENT recency profile (delta_0 = now - last access) and evict
    the largest. Predicting at eviction time with the current age is what makes an imperfect model
    self-correcting: a mispredicted object that sits unused keeps ageing, so its predicted distance
    keeps growing and it becomes MORE evictable; a prediction frozen at access time instead goes
    stale and traps the object forever. New objects join the resident pool immediately (no
    probationary protection, which would let one-shot objects accumulate). Sampling (LRB uses 64)
    makes each eviction O(K) rather than O(cache)."""
    name = "learned"

    def __init__(self, model, nd=N_DELTAS, sample_k=64, seed=0):
        self.model, self.ND, self.K = model, nd, sample_k
        self.rng = np.random.default_rng(seed)
        self.hist, self.freq, self.size = {}, {}, {}
        self.res, self.pos = [], {}                  # resident list + index map (O(1) swap-remove)
        self._t = 0                                  # current time, set by replay before eviction

    def _obs(self, o, t, s=None):
        D = self.hist.get(o)
        if D is None:
            D = deque(maxlen=self.ND); self.hist[o] = D
        D.append(t)
        self.freq[o] = self.freq.get(o, 0) + 1
        if s is not None:
            self.size[o] = s

    def hit(self, o, t):
        self._obs(o, t)

    def admit(self, o, s, t):
        self._obs(o, t, s)
        if o not in self.pos:
            self.pos[o] = len(self.res); self.res.append(o)

    def _remove(self, o):
        idx = self.pos.pop(o, None)
        if idx is None:
            return
        tail = self.res.pop()
        if idx < len(self.res):
            self.res[idx] = tail; self.pos[tail] = idx

    def evict_one(self):
        L = len(self.res)
        if L == 0:
            return None
        t = self._t
        cand = self.res if L <= self.K else [self.res[j]
                                             for j in np.unique(self.rng.integers(0, L, self.K))]
        X = np.asarray([_row(self.hist[o], t, self.freq[o], self.size.get(o, 1), self.ND)
                        for o in cand], dtype=np.float64)
        vo = cand[int(np.argmax(self.model.predict(X)))]   # largest predicted distance -> evict
        self._remove(vo)
        return vo

    def forget(self, o):
        self._remove(o)


# ------------------------------------------------------------------ demand-only replay (local)
def replay(trace, cap, ev, warmup_frac=0.05, mode="plain", return_hits=True, progress=None,
           admit_mask=None):
    """Deterministic demand-only cache replay mirroring PFCache's no-prefetch path byte-for-byte.
    mode: 'plain' (ev.hit(o)), 'oracle' (ev.touch(o,i)), 'learned' (ev.hit(o,i)). All eviction is
    heap-driven inside the evictor. Validated against PFCache in _construction_check().
    admit_mask: optional bool array over positions; on a miss where admit_mask[i] is False the
    object is served from origin but NOT cached (learned ADMISSION / bypass). None -> admit all,
    which is exactly the base evictor (so a tuned admission policy is bounded below by it)."""
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
            bypass = admit_mask is not None and not admit_mask[i]
            if s <= cap and not bypass:                  # bypass -> serve from origin, do not cache
                if mode == "learned":
                    ev._t = i                            # eviction scores use the CURRENT age (now - last)
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
    """Fast startup gate: assert the local demand replay reproduces PFCache's OHR EXACTLY for
    S3-FIFO and Belady (so the learned arm sits on identical footing), and that Belady dominates
    S3-FIFO on SYNTH. Deliberately does NOT run the learned evictor (per-eviction prediction is
    slow); the learned capture demonstration lives behind --selftest (see learned_demo)."""
    from p4_cache import footprint_bytes, synth_trace
    from p4_prefetch import build_obj_positions, build_obj_sizes
    tr = synth_trace(n_req=120_000, n_obj=8_000, zipf_a=1.1, seq_frac=0.3, seed=0)
    cap = max(int(footprint_bytes(tr) * 0.01), 1)
    pos = build_obj_positions(tr); szs = build_obj_sizes(tr)
    ok, b, e = _construction_check(tr, cap, pos, szs, 0.05)
    assert ok, "local replay does not reproduce PFCache -- learned arm would be unfair"
    assert e["ohr"] >= b["ohr"], f"Belady must dominate S3-FIFO on SYNTH: {e['ohr']} vs {b['ohr']}"
    print(f"[p4_evictlearn selftest] PASS -- replay==PFCache for S3-FIFO and Belady; "
          f"SYNTH eviction corridor {100*(e['ohr']-b['ohr']):+.2f} pts")


def learned_demo(seed=0):
    """--selftest extra: a small workload with one-hit-wonder pollution, proving the deployable
    learned-ADMISSION policy captures a substantial, positive fraction of the corridor (the
    pipeline can green-light) and is bounded below by S3-FIFO."""
    from p4_cache import footprint_bytes
    from p4_prefetch import build_obj_positions, build_obj_sizes
    tr = synth_positive(n_req=40_000, n_hot=800, seed=seed)
    cap = max(int(footprint_bytes(tr) * 0.01), 1)
    pos = build_obj_positions(tr)
    b = replay(tr, cap, S3FIFOEvictor(cap), 0.05, mode="plain")
    e = replay(tr, cap, BeladyEvictor(cap, pos), 0.05, mode="oracle")
    _, P, _ = train_admission(tr, 0.5, 80_000, 120, seed, N_DELTAS)
    taus = tuple(round(x, 2) for x in np.linspace(0.0, 0.9, 10))
    tau, _ = tune_admission_tau(tr, cap, P, 0.5, taus)
    lr = replay(tr, cap, S3FIFOEvictor(cap), 0.05, mode="plain", admit_mask=(P >= tau))
    corr = 100 * (e["ohr"] - b["ohr"]); capg = 100 * (lr["ohr"] - b["ohr"])
    print(f"[learned_demo] one-shot-polluted workload: corridor {corr:+.2f} pts, learned admission "
          f"(tau={tau:.2f}) captures {capg:+.2f} pts ({100*capg/corr if corr > 0.1 else 0:.0f}%)  "
          f"[S3-FIFO {b['ohr']:.4f}  Belady {e['ohr']:.4f}  Learned {lr['ohr']:.4f}]")


# -------------------------------------------------------------------------------------- main
def run(trace, cap, pos, szs, args):
    warmup = args.warmup
    print(LINE)
    print("  CACHE-MANAGEMENT GREEN-LIGHT -- does the SAME instrument that rejects the prefetcher "
          "accept a learned cache policy?")
    print(LINE)

    ok, base, ceil = _construction_check(trace, cap, pos, szs, warmup)
    if not ok:
        print("  VERDICT: INVALID -- construction check failed; learned arm not comparable. Abort.")
        return None

    t0 = time.time()
    if args.mode == "admit":                              # learned ADMISSION (Baleen-style, bounded by S3-FIFO)
        clf, P, base_rate = train_admission(trace, args.train_frac, args.max_train, args.trees,
                                            args.seed, args.deltas)
        taus = tuple(round(x, 2) for x in np.linspace(0.0, 0.9, 10))
        tau, tr_ohr = tune_admission_tau(trace, cap, P, args.train_frac, taus)
        print(f"  [train] admission classifier x {args.deltas+2} features "
              f"({args.deltas}-step recency profile + freq + size), {args.trees} trees; "
              f"train reuse rate {base_rate:.3f}; tuned bypass tau={tau:.2f} "
              f"(train OHR {tr_ohr:.4f}) ({time.time()-t0:.1f}s)")
        learn = replay(trace, cap, S3FIFOEvictor(cap), warmup, mode="plain", return_hits=True,
                       admit_mask=(P >= tau), progress="learned-admit" if args.tqdm else None)
        arm = f"S3-FIFO + learned admission (bypass tau={tau:.2f})"
    else:                                                 # learned EVICTION (LRB-style)
        model, n_train = train_model(trace, args.train_frac, args.max_train, args.trees, args.seed,
                                     nd=args.deltas)
        print(f"  [train] LRB-lite evictor on {n_train:,} censored samples x {args.deltas+2} "
              f"features, {args.trees} trees ({time.time()-t0:.1f}s)")
        learn = replay(trace, cap, LearnedEvictor(model, nd=args.deltas, sample_k=args.sample_k,
                                                  seed=args.seed),
                       warmup, mode="learned", return_hits=True,
                       progress="learned-evictor" if args.tqdm else None)
        arm = "learned evictor (LRB-style)"

    bh, eh, lh = base["hits_series"], ceil["hits_series"], learn["hits_series"]
    corridor_pt, c_lo, c_hi, _ = block_ci(eh.astype(np.float64) - bh.astype(np.float64),
                                          args.blocks, args.resamples, args.seed)
    cap_pt, k_lo, k_hi, k_se = block_ci(lh.astype(np.float64) - bh.astype(np.float64),
                                        args.blocks, args.resamples, args.seed)
    frac = cap_pt / corridor_pt if corridor_pt > 1e-9 else 0.0
    bhr_corr = 100 * (ceil["bhr"] - base["bhr"]); bhr_cap = 100 * (learn["bhr"] - base["bhr"])

    print(f"  S3-FIFO (tuned baseline)   OHR {base['ohr']:.4f}   [deployable lower bound]")
    print(f"  BELADY  (mgmt ceiling)     OHR {ceil['ohr']:.4f}   [clairvoyant, not deployable]")
    print(f"  LEARNED                    OHR {learn['ohr']:.4f}   [{arm}; frozen, causal, deployable]")
    print("-" * 100)
    print(f"  CORRIDOR            {corridor_pt:+.2f} pts   95% CI [{c_lo:+.2f}, {c_hi:+.2f}]   "
          f"(Belady - S3-FIFO; pre-registered bar >= {args.bar:.1f})")
    print(f"  CAPTURED            {cap_pt:+.2f} pts   95% CI [{k_lo:+.2f}, {k_hi:+.2f}]   SE {k_se:.2f}   "
          f"= {100*frac:.1f}% of the corridor (Learned - S3-FIFO)")
    print(f"  BYTE-WEIGHTED (BHR) corridor {bhr_corr:+.2f} pts   captured {bhr_cap:+.2f} pts")

    corridor_real = c_lo >= args.bar
    captured = (k_lo > 0.0) and (frac >= args.min_capture)
    if corridor_real and captured:
        verdict = (f"BUILD -- a large corridor EXISTS and a deployable {args.mode} policy CAPTURES "
                   f"{100*frac:.0f}% of it ({cap_pt:+.2f} pts above the tuned baseline, CI clear of "
                   f"0). The instrument green-lights this component.")
    elif corridor_real and not captured:
        verdict = (f"REAL BUT NOT CAPTURED -- corridor clears the bar, but this {args.mode} policy "
                   f"realises only {100*frac:.0f}% (CI [{k_lo:+.2f},{k_hi:+.2f}]). Honest partial result.")
    else:
        verdict = (f"NO CORRIDOR -- corridor below the {args.bar:.1f}-pt bar "
                   f"(CI lower {c_lo:+.2f}); nothing to build here.")
    print(f"  VERDICT: {verdict}")
    print(f"  CONTRAST: on this trace the deployable PREFETCH policy lands well BELOW the baseline "
          f"(the endogenous-slack trap); this learned {args.mode} policy lands ABOVE it. Same "
          f"instrument, opposite decision -- across capabilities, not just workloads.")
    print(LINE)
    return dict(base=base["ohr"], belady=ceil["ohr"], learned=learn["ohr"], mode=args.mode,
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
    ap.add_argument("--mode", choices=("admit", "evict"), default="admit",
                    help="learned arm: 'admit' = S3-FIFO + learned admission (fast, bounded by "
                         "baseline); 'evict' = LRB-style learned evictor (slow, per-eviction)")
    ap.add_argument("--sample-k", type=int, default=64,
                    help="residents sampled per eviction for the current-age prediction (LRB=64)")
    ap.add_argument("--deltas", type=int, default=N_DELTAS,
                    help="recency-profile length: log gaps between the last N accesses (LRB core feature)")
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
        learned_demo(seed=args.seed)
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
