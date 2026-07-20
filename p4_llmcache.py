#!/usr/bin/env python3
"""p4_llmcache.py -- PHANTOM HEADROOM, ported to LLM KV-CACHE SERVING (Mooncake traces).

WHY THIS EXISTS. The CDN/KV paper proves two things: (1) most claimed prefetch headroom is
phantom -- it evaporates under an adversarially-tuned bar and a vocab-restricted, iso-bandwidth
clairvoyant ceiling; (2) the headroom that survives is real but CLAIRVOYANCE-PRICED -- a
pre-registered oracle decomposition (F1-stream / F5 / Policy-1 / F7) shows it requires
clairvoyant insertion timing, and the mechanism (endogenous slack: a causal policy's own fetch
volume destroys the residence window its timing model relies on) explains why no causal
scheduler captures it. The instrument therefore DISCRIMINATES: it can tell, before a single
model is trained, whether a domain's headroom is (a) absent, (b) conquerable, or (c) phantom.

This script is the instrument's first cross-domain application: KV-cache warming in LLM
serving, on Mooncake's production traces (Kimi/Moonshot, FAST'25). Nobody has priced KV-cache
warming headroom honestly: the literature reports system hit rates, but never separates
"headroom a learnable warmer could capture" from "compulsory/clairvoyant-only headroom".

DOMAIN MAPPING (each choice is the honest one, argued):
  * object    = a 512-token KV block (`hash_ids` entry; identical id == reusable KVCache).
                Uniform size (=1 here; any constant scales out) -- so Belady is EXACT and the
                gap gate is an optimum, not a heuristic.
  * request   = one LLM request re-reads its whole prefix chain (its hash_ids, in order).
  * prefetch  = WARMING a block into the HBM cache from tiered store before its next use.
                OHR == fraction of block reads served from cache == prefill compute/transfer
                saved (the TTFT lever).
  * TURN GATE (the domain-honest decision epoch -- the methodological novelty of this port):
                a request's blocks are read SIMULTANEOUSLY at prefill, so per-access prefetch
                inside the in-flight request would fake-convert whole chains at zero latency
                and destroy the measurement. EVERY arm -- tuned bar and clairvoyant ceiling
                alike -- may only issue warmings at request boundaries: warming helps FUTURE
                turns only. This makes the corridor measure exactly the deployable question
                ("cross-turn warming"), unlike a naive transplant.
  * vocab restriction: in CDN this was an analytic device; here it is PHYSICAL -- a KV block
                that was never computed and stored cannot be warmed by any system. The
                learnable ceiling is therefore the TRUE ceiling in this domain.
  * iso-bandwidth: warming traffic (blocks moved) capped at the tuned bar's rate via the same
                token bucket -- headroom must not be purchasable with free bandwidth.

PRE-REGISTERED (fixed before the first real-trace run; bars inherited unchanged from the paper):
  GATE      learnable corridor >= 8 OHR pts, 95% block-bootstrap CI lo > 0, Pareto (ceiling
            traffic <= bar traffic).
  F2-lite   use-lag learnability on the bar's own emission stream: Spearman >= 0.20 AND
            never-AUC >= 0.60 (p4_hazard, three-way split, unchanged).
  FORK 1    corridor < 8            -> NO corridor: tuned causal warming already captures the
            reuse. The instrument says do NOT build a learned warmer here. (A result: the
            discipline that killed our own five ideas, applied prospectively.)
  FORK 2    corridor >= 8, F2 PASS  -> headroom exists AND the "when" (human think-time /
            agent-step cadence) is learnable -> candidate CONQUEST domain; the capture ladder
            (F5/F7 analogs) is justified as the next step -- and only then.
  FORK 3    corridor >= 8, F2 FAIL  -> the phantom mechanism GENERALIZES beyond CDN/KV.

ONE COMMAND -> ONE TABLE (bar, ceiling, corridor, CI, gate, W2b signal, F2, S(100)) + fork.

    curl -L -o data/mooncake_conv.jsonl \
      https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl
    python -u p4_llmcache.py --trace data/mooncake_conv.jsonl --verbose
    python -u p4_llmcache.py --selftest

Known limitations (stated, not hidden): lags are in ACCESS-INDEX units (consistent with the
paper), not wall-clock -- a wall-clock hazard is future work; one hour of one provider's
traffic is evidence, not a universal claim; results here are a domain screen, not a method.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from p4_cache import compute_next_access, footprint_bytes
from p4_gates import gate_gap, gate_prefetch_signal
from p4_prefetch import (Markov1, Markov2, PFCache, Prescient, build_obj_positions,
                         build_obj_sizes)
from p4_strongbar import Markov3, STRONG_TAUS
from p4_sweep import KS
import p4_hazard

LINE = "=" * 100
PREDS = {"markov1": Markov1, "markov2": Markov2, "markov3": Markov3}


def block_ci(diff, blocks, resamples, seed):
    """Paired moving-block bootstrap -- identical convention to every other rung."""
    R = len(diff)
    L = max(R // blocks, 1)
    b = R // L
    bm = diff[:b * L].reshape(b, L).mean(axis=1)
    rng = np.random.default_rng(seed)
    boot = 100.0 * bm[rng.integers(0, b, size=(resamples, b))].mean(axis=1)
    return np.percentile(boot, [2.5, 97.5])


class TurnGate:
    """Release prefetch suggestions ONLY at request boundaries.

    KV warming cannot help the in-flight request: prefill reads the whole prefix chain at
    once, so a fetch triggered by block i of request R lands 'during' R -- physically too
    late to serve R, but our access-index simulator would credit it at zero latency. This
    wrapper holds every suggestion harvested during R and releases the batch after R's last
    block access, so warming only ever targets FUTURE turns. Applied to EVERY arm (bar and
    clairvoyant ceiling alike) -- the constraint is identical, so the corridor stays a fair
    like-for-like difference."""

    QCAP = 256                                     # retained-queue bound

    def __init__(self, base, turn_end):
        self.base, self.turn_end = base, turn_end
        self.name = f"turngate({getattr(base, 'name', '?')})"
        self.fresh, self.retained, self.pset = [], [], set()

    def set_params(self, k=None, tau=None):
        self.base.set_params(k=k, tau=tau)
        return self

    def reset(self):
        r = getattr(self.base, "reset", None)
        if r:
            r()
        self.fresh, self.retained, self.pset = [], [], set()

    def suggest(self, o, cached, i):
        for x in self.base.suggest(o, cached, i):
            if x not in self.pset:
                self.fresh.append(x)
                self.pset.add(x)
        if not self.turn_end[i]:
            return ()
        # Release FRESH suggestions first (this request's calls, in their emitted order --
        # for Prescient that is soonest-next-use first, the JIT-correct head), then retained
        # older ones. RETAIN what the token bucket cannot yet afford (PFCache admits a prefix
        # of this list until tokens run out): dropping unadmitted candidates starved the
        # rate-limited ceiling arm of its own budget while the burst-capable bar lost nothing
        # -- a measured -1.5pt inversion the selftest invariants caught. Cached candidates
        # are pruned (demand-filled or already warmed); the queue is capped so stale
        # never-affordable candidates age out instead of pinning memory.
        out = [x for x in self.fresh + self.retained if x not in cached]
        if len(out) > self.QCAP:
            out = out[:self.QCAP]
        self.fresh, self.retained, self.pset = [], list(out), set(out)
        return out


# --------------------------------------------------------------------------- trace loading
def _to_trace(events):
    """events: list of (timestamp, [hash_ids]) sorted by timestamp -> replayable trace dict.
    Block size = 1 uniformly (any constant scales out of OHR and of the iso-BW ratio), so
    'bytes' == 'blocks' everywhere downstream and Belady is an exact optimum."""
    ids, turn_end = [], []
    for _, chain in events:
        for j, h in enumerate(chain):
            ids.append(int(h))
            turn_end.append(j == len(chain) - 1)
    ids = np.asarray(ids, dtype=np.uint64)
    return dict(obj_id=ids, size=np.ones(len(ids), dtype=np.int64),
                next_vtime=compute_next_access(ids), n=len(ids),
                turn_end=np.asarray(turn_end, dtype=bool), n_req=len(events))


def load_mooncake(path, limit=0):
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            h = r.get("hash_ids") or []
            if not h:
                continue
            events.append((float(r.get("timestamp", 0)), h))
    events.sort(key=lambda e: e[0])
    if limit:
        events = events[:limit]
    return _to_trace(events)


def synth_sessions(n_sessions=300, seed=0):
    """Selftest generator: multi-turn sessions with growing prefix chains, shared
    system-prompt blocks across sessions of the same 'app', and exponential think times.
    Validates the converter, the TurnGate, and every arm end-to-end without a download."""
    rng = np.random.default_rng(seed)
    nxt = 100
    events = []
    for _ in range(n_sessions):
        app = int(rng.integers(0, 3))
        chain = [app * 2, app * 2 + 1]              # shared system-prefix blocks
        t = float(rng.uniform(0, 3_000_000))
        for _ in range(int(rng.integers(2, 8))):
            new = int(rng.integers(2, 6))
            chain = chain + list(range(nxt, nxt + new))
            nxt += new
            events.append((t, list(chain)))
            t += float(rng.exponential(30_000))     # think time
    events.sort(key=lambda e: e[0])
    return _to_trace(events)


# --------------------------------------------------------------------------------- the run
def run(trace, args, name):
    tf = args.train_frac
    n = trace["n"]
    fp = footprint_bytes(trace)
    cap = max(int(fp * args.cache_frac), 1)
    pos, szs = build_obj_positions(trace), build_obj_sizes(trace)
    tend = trace["turn_end"]
    uniq = len(np.unique(trace["obj_id"]))
    print(f"[trace] {name}: {trace['n_req']:,} requests -> {n:,} block accesses, "
          f"{uniq:,} unique blocks (reuse {1 - uniq / n:.1%}), "
          f"cache {args.cache_frac:.1%} = {cap:,} blocks")

    # context gates (cheap; a failure here means the trace/config cannot host a corridor)
    gate_gap(trace, fp)
    gate_prefetch_signal(trace)

    # ---- no-prefetch base: the traffic denominator ----
    base = PFCache(cap, "s3fifo", None, positions=pos, sizes=szs).run(trace)
    base_tx = base["origin_bytes"]
    print(f"\n  S3-FIFO base (no warming)  OHR {base['ohr']:.4f}")

    # ---- TUNED BAR: max over {markov1,2,3} x tau x k, turn-gated, <= 1.15x traffic ----
    # Identical convention to gate_a/strongbar: the bar is the best DECOUPLED causal warmer
    # on offer, so the corridor cannot be an under-tuned-baseline artifact.
    taus = (0.05, 0.10, 0.20, 0.35, 0.50) if args.fast else STRONG_TAUS
    pnames = ("markov1", "markov2") if args.fast else ("markov1", "markov2", "markov3")
    t0 = time.time()
    gated = {}
    for nm in pnames:
        gated[nm] = TurnGate(PREDS[nm](trace, train_frac=tf, window=args.window), tend)
        print(f"  [build] {nm} ({time.time() - t0:.1f}s)", flush=True)
    print(f"  {'predictor':9s} {'tau':>5s} {'k':>3s} {'OHR':>8s} {'traffic':>8s} "
          f"{'prec':>6s} {'admissible':>11s}")
    best = None
    for nm, wg in gated.items():
        for tau in taus:
            for k in KS:
                r = PFCache(cap, "s3fifo", wg.set_params(k=k, tau=tau),
                            positions=pos, sizes=szs).run(trace)
                tx = r["origin_bytes"] / max(base_tx, 1)
                ok = tx <= args.max_traffic
                if ok and (best is None or r["ohr"] > best[1]["ohr"]):
                    best = ((nm, tau, k), r, tx)
                if args.verbose:
                    print(f"  {nm:9s} {tau:5.2f} {k:3d} {r['ohr']:8.4f} {tx:8.2f} "
                          f"{r['pf_precision']:6.3f} {'yes' if ok else 'NO -- over BW':>11s}",
                          flush=True)
    if best is None:
        print("  !! no config fits the traffic budget -- cannot measure a corridor. STOP.")
        return None
    (wname, wtau, wk), _, wtx = best

    # winner re-run with hit series (for the paired bootstrap) + cold accounting
    bar = PFCache(cap, "s3fifo", gated[wname].set_params(k=wk, tau=wtau), positions=pos,
                  sizes=szs).run(trace, cold_train_frac=tf, return_hits=True)
    rate = bar["prefetch_bytes"] / max(n, 1)

    # BAR-CAP -- the bar under the SAME token bucket the ceiling faces (HJS-L / Policy-1
    # precedent). The bar bursts; the ceiling is rate-limited at the bar's average. On CDN
    # the corridor dwarfed this protocol cost so the burst-vs-bucket asymmetry was safely
    # conservative; here the corridor can be small enough that protocol cost flips the sign
    # of a naive comparison (measured: it did). Corridor vs BAR stays the pre-registered
    # headline; corridor vs BAR-CAP is the like-for-like scheduling comparison; the
    # ceiling-validity invariant is checked against BAR-CAP (same protocol).
    barcap = PFCache(cap, "s3fifo", gated[wname].set_params(k=wk, tau=wtau), positions=pos,
                     sizes=szs, pf_byte_rate=rate).run(trace, cold_train_frac=tf,
                                                       return_hits=True)

    # ---- LEARNABLE (= physical) CEILING: warm-Prescient, turn-gated, iso-BW ----
    # vocab = blocks seen in the training prefix: a block never yet computed+stored cannot be
    # warmed by ANY system, so this ceiling is not just budget-fair, it is physically maximal.
    cut = int(n * tf)
    tvocab = set(int(x) for x in trace["obj_id"][:cut])
    warm = PFCache(cap, "s3fifo",
                   TurnGate(Prescient(trace, k=max(KS), lookahead=args.lookahead,
                                      vocab=tvocab), tend),
                   positions=pos, sizes=szs, pf_byte_rate=rate).run(
        trace, cold_train_frac=tf, return_hits=True)

    bar_tx = bar["origin_bytes"] / max(base_tx, 1)
    warm_tx = warm["origin_bytes"] / max(base_tx, 1)
    corridor = 100.0 * (warm["ohr"] - bar["ohr"])
    d = warm["hits_series"].astype(np.float64) - bar["hits_series"].astype(np.float64)
    lo, hi = block_ci(d, args.blocks, args.resamples, args.seed)
    pareto = warm["ohr"] > bar["ohr"] and warm_tx <= bar_tx + 0.01
    protocol = 100.0 * (bar["ohr"] - barcap["ohr"])
    corr_cap = 100.0 * (warm["ohr"] - barcap["ohr"])
    dc = warm["hits_series"].astype(np.float64) - barcap["hits_series"].astype(np.float64)
    clo, chi = block_ci(dc, args.blocks, args.resamples, args.seed)

    # ---- F2-lite: is the "when" learnable on the bar's own emission stream? ----
    class _H:
        pass
    h = _H()
    h.trace, h.pred, h.tau, h.k = name, wname, wtau, wk
    h.window, h.train_frac, h.haz_frac = args.window, tf, 0.75
    h.horizon = args.horizon if args.horizon > 0 else max(1000, n // 20)
    h.out, h.top_m = None, 16
    fm = {}
    p4_hazard.run(trace, cap, pos, szs, h, metrics=fm)

    # ---- invariants: a violation means the ARM is wrong, not the finding ----
    inv = []
    if warm["pf_cold_hits"] != 0:
        inv.append(f"ceiling cold hits {warm['pf_cold_hits']} != 0 -- vocab leak")
    if bar["pf_cold_hits"] != 0:
        inv.append(f"bar cold hits {bar['pf_cold_hits']} != 0 -- history-based bar funding OOV")
    if warm["prefetch_bytes"] > 1.02 * max(bar["prefetch_bytes"], 1):
        inv.append("ceiling warming bytes exceed bar's -- NOT iso-bandwidth")
    if warm["ohr"] < barcap["ohr"] - 1e-9:
        inv.append("ceiling below BAR-CAP (same token bucket) -- clairvoyance losing under "
                   "identical protocol: TurnGate or vocab wiring broken")

    # ------------------------------- THE TABLE -------------------------------
    gate = corridor >= 8.0 and lo > 0 and pareto
    print(f"\n{LINE}")
    print(f"  PHANTOM HEADROOM x LLM KV-CACHE SERVING -- {name}")
    print(f"  (turn-gated warming; ceiling vocab = physically warmable blocks; iso-BW)")
    print(LINE)
    print(f"  {'arm':28s} {'OHR':>8s} {'traffic':>8s} {'warmings':>10s} {'prec':>6s}")
    print(f"  {'S3-FIFO base (no warming)':28s} {base['ohr']:8.4f} {'1.00':>8s} "
          f"{'-':>10s} {'-':>6s}")
    print(f"  {'TUNED BAR ' + f'({wname} t{wtau} k{wk})':28s} {bar['ohr']:8.4f} "
          f"{bar_tx:8.2f} {bar['pf_issued']:>10,} {bar['pf_precision']:6.3f}")
    print(f"  {'BAR-CAP (same bucket)':28s} {barcap['ohr']:8.4f} "
          f"{barcap['origin_bytes'] / max(base_tx, 1):8.2f} {barcap['pf_issued']:>10,} "
          f"{barcap['pf_precision']:6.3f}")
    print(f"  {'LEARNABLE CEILING (warm)':28s} {warm['ohr']:8.4f} {warm_tx:8.2f} "
          f"{warm['pf_issued']:>10,} {warm['pf_precision']:6.3f}")
    print(f"  {'-' * 96}")
    print(f"  CORRIDOR vs BAR      {corridor:+.2f} pts   95% CI [{lo:+.2f},{hi:+.2f}]   "
          f"Pareto {'yes' if pareto else 'NO'}   [pre-registered gate >= 8, CI>0]   "
          f"-> {'EXISTS' if gate else 'below gate'}")
    print(f"  CORRIDOR vs BAR-CAP  {corr_cap:+.2f} pts   95% CI [{clo:+.2f},{chi:+.2f}]   "
          f"(like-for-like protocol)   PROTOCOL COST {protocol:+.2f} pts (bar burst vs bucket)")
    print(f"  F2-lite  Spearman {fm.get('rho', float('nan')):+.3f} [>=0.20]   "
          f"never-AUC {fm.get('never_auc', float('nan')):.3f} [>=0.60]   "
          f"S(100)={fm.get('s100', float('nan')):.3f}   "
          f"-> {'when LEARNABLE' if fm.get('f2_pass') else 'when NOT learnable'}")
    print(f"  INVARIANTS  " + ("all pass" if not inv else "FAIL"))
    for m in inv:
        print(f"    !! {m}")

    # ---- the pre-registered fork ----
    if not gate:
        v = ("FORK 1 -- NO corridor: tuned causal warming already captures the reuse. The "
             "instrument says do NOT build a learned warmer here; that prospective 'do not "
             "build' is the phantom-headroom discipline paying out in a second domain.")
    elif fm.get("f2_pass"):
        v = ("FORK 2 -- corridor EXISTS and the 'when' is LEARNABLE: candidate CONQUEST "
             "domain. Next (and only now justified): the capture ladder -- F5/F7 analogs, "
             "then a think-time-hazard warming scheduler as the positive method.")
    else:
        v = ("FORK 3 -- corridor exists but the 'when' is NOT learnable: the phantom "
             "mechanism (clairvoyance-priced timing) GENERALIZES beyond CDN/KV. The negative "
             "result is cross-domain, which strengthens the paper's central claim.")
    print(f"  VERDICT  {v}")
    print(LINE + "\n")
    return dict(bar=bar["ohr"], warm=warm["ohr"], corridor=corridor, ci=(lo, hi),
                gate=gate, f2=fm, inv_ok=not inv)


def main():
    ap = argparse.ArgumentParser(description="Phantom Headroom x LLM KV-cache serving "
                                             "(one command -> one table + fork verdict)")
    ap.add_argument("--trace", help="Mooncake-format JSONL (timestamp, hash_ids)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="max requests (0 = all)")
    ap.add_argument("--cache-frac", type=float, default=0.01,
                    help="HBM block-cache size as a fraction of unique blocks (paper primary)")
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--max-traffic", type=float, default=1.15)
    ap.add_argument("--lookahead", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=0, help="F2 horizon (0 = auto n/20)")
    ap.add_argument("--fast", action="store_true", help="coarse tau grid, markov1/2 only")
    ap.add_argument("--verbose", action="store_true", help="print every sweep config")
    ap.add_argument("--blocks", type=int, default=1000)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.selftest:
        trace = synth_sessions()
        r = run(trace, args, "SYNTH-SESSIONS")
        assert r is not None and r["inv_ok"], "selftest: invariants failed"
        print("  [p4_llmcache selftest] completed without error\n")
        return
    if not args.trace:
        ap.error("need --trace PATH (or --selftest)")
    trace = load_mooncake(args.trace, args.limit)
    run(trace, args, args.trace)


if __name__ == "__main__":
    main()
