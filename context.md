# P4 — context & what is running

**One-line status (2026-07-17):** the *joint* "co-optimize eviction + prefetch" thesis is **dead**
on MSR (endorsed by peer). What may survive is a **timing** wedge — *predict WHEN to prefetch, not
WHAT*. Three pre-registered gates decide whether that wedge is a paper. Two run on the GPU box; one
(lit check) is running in the background here.

---

## What is running RIGHT NOW

| what | where | id / how | decides |
|---|---|---|---|
| **Lit check** — is "predict WHEN, not what" prior art for object caches? | background deep-research workflow (this machine) | task `wvuuxxzh8`, run `wf_c171d85b-aab` | GATE C below |
| **Gate A** — tuned-baseline corridor | your GPU box (manual) | `python p4_sweep.py --trace data/msr_proj_0.oracleGeneral --limit 2000000 --gate a` | is the timing corridor real |
| **Gate B** — interaction 2×2 across 3 trace families | your GPU box (manual, needs 2 more traces) | `python p4_sweep.py --gate b --trace msr --trace wiki --trace twitter --limit 2000000` | is "substitutes not complements" structural |

**Lit-check history:** first launch (`wf_4a0b1417-a0c`) stopped with no completion record when the
previous session's process exited. A resume-from-runId dropped the `args` and errored with 0 agents
(args are not stored with a runId — always relaunch by `name`+`args`, not resume, when the prior run
never produced cached agents). Relaunched fresh as `wvuuxxzh8`.

---

## The story so far (why the joint thesis died)

The paper's original claim: a single RL policy that *jointly* decides eviction + prefetching beats
any *decoupled* (separately-tuned evictor + prefetcher) system, because joint optimization is
intractable and a learned policy can approximate it.

**T1 on `msr_proj_0` (2M reqs, 260,712 objects, footprint 10.97 GB, cache 1% = 109.7 MB):**

```
arm                        OHR      BHR    traffic×   pf prec   wasted
LRU                      0.4863   0.2202     1.00        —          0
LRU + Markov-1           0.6178   0.3646     1.10      0.733   108,888
S3-FIFO                  0.5516   0.3143     1.00        —          0
S3-FIFO + Markov-1       0.6709   0.4449     1.10      0.734    97,535   ← honest A1 bar
S3-FIFO + Prescient@BW   0.7784   0.4989     1.05      0.903    51,817   ← timing ceiling (realistic evict)
Belady                   0.6229   0.4266     1.00        —          0
Belady + Markov-1        0.7332   0.5472     1.19      0.586   168,944   ← upper bound on ANY decoupled
Prescient @matched-BW    0.8363   0.6282     1.03      0.997     1,329
Prescient @unlimited     1.0000   1.0000     1.94      0.749   313,289
```

**Two independent kill-arguments (both verified arithmetically by peer):**

1. **Interaction is negative & monotone.** Eviction gain (Belady − S3-FIFO) *shrinks* as prefetch
   improves: `no-pf +7.13 → Markov-1 +6.22 → Prescient +5.78`. Eviction and prefetching fix the
   **same misses** → they are **substitutes, not complements**. A joint policy has no coupling to
   exploit.
2. **The oracle refuses the "retained prefetch" prize.** Belady is the joint-optimal evictor for
   prefetched objects (it evicts by true next-access). It converts **31k FEWER** prefetches into
   hits than S3-FIFO, wastes **73% more** of them, and still **wins by 6.2 pts**. Retaining
   "correct-but-evicted" prefetches is a *mistake the oracle declines to make*, not an unrealized
   gain.

**A labeling artifact I shipped and then retracted:** the wasted-prefetch autopsy first counted a
prefetch as "correct, evicted too early" if the object was *ever* requested again → 57.7%, implying
a +2.96 pt jointness prize. But median distance-to-use was **116,612 requests** into a ~2,600-object
cache: those objects were never going to be used from cache. The "prize" was measuring object
*existence*, not prediction correctness. Same small-n / wrong-label error class as the earlier +64%
headroom read. **Prize is ≈ 0. Do not build the joint RL (W4).**

---

## What survives: the TIMING wedge

A **realistic** evictor (S3-FIFO) + a **perfectly-timed** prefetcher (Prescient@BW) = **0.7784 @
1.05× traffic**, i.e. **+10.75 pts over the honest bar at LESS traffic**. Prescient beats Belady by
+21.3 pts for 3% more traffic. Mechanism: **a correctly-timed prefetch is nearly free** — it moves a
fetch earlier (miss→hit) rather than adding one. Markov-1's failure is not picking wrong objects; it
is picking right objects at wrong *times* (it over-issues: 368k prefetches @1.10× vs the oracle's
531k @1.05×). Regression target = **time-to-next-access** (a computable train-time oracle, regressed,
acted on greedily — same machinery as the v3 RAO lesson, new decision variable).

---

## The three pre-registered gates (decision rules fixed BEFORE results)

**GATE A — TUNED BAR.** Our 0.6709 bar used `tau=0.05, k=2` (defaults, not a sweep). The bar must be
the *best decoupled system available*: `best OHR at ≤1.15× traffic over {Markov-1, Markov-2} × tau ×
k`. Markov-2 is mandatory — sweeping only Markov-1 lets a reviewer tune the better predictor for us.
Ceiling is **iso-bandwidth** (Prescient given the tuned bar's prefetch byte-rate) and must **Pareto-
dominate** (better OHR at no more traffic) or the corridor is a bandwidth difference, not a timing
prize.
> **Rule: corridor ≥ 8 pts AND ceiling dominates → timing pivot LIVE. < 8 → P4 DEAD, stop.**
> (Synthetic smoke: Markov-2 beat the default bar by +4.2 pts at less traffic, collapsing a fake
> +42 pt corridor to +5.87. Real number is genuinely uncertain — that's a working gate.)

**GATE B — n > 1.** "Substitutes, not complements" died on **one** MSR block trace. Replay the
identical 2×2 on a Wikipedia CDN trace + a Twitter KV trace. **Negative on all three families →
structural**, and that is itself a publishable measurement result (kills the joint-cache-RL direction
for everyone), not a consolation prize. Mixed → workload-dependent, "structural" is unsupported.
> Caveat: on variable-size CDN/KV objects Belady is a strong heuristic, **not** a proven optimum
> (variable-size eviction is NP-hard). Only MSR's uniform 4KB blocks earn the word "optimal".

**GATE C — LIT CHECK (running).** Is "predict WHEN, not what" already claimed for object caches?
Adjudicating DEAP, Pythia (MICRO'21 RL HW prefetch), Voyager/Hashemi (ICML'18), TTL/proactive
caching, and learned-caching work (LRB, GL-Cache, Baleen, C2DN, …). Verdict target: (a) novel /
(b) done in hardware prefetching but not object caches / (c) already done for object caches.
> If (c) → wedge dies. If prior art exists on *when-prefetching*, tighten the ≥8 pt bar.

### Combined outcomes (all pre-committed)
| corridor (A) | lit (C) | outcome |
|---|---|---|
| ≥ 8 pts | clean | **timing paper LIVE**; substitutes finding (B) becomes a section |
| ≥ 8 pts | prior art | reframe against it, or stop |
| < 8 pts | any | **P4 dead**; honest output = the substitutes *measurement study* + Domain-B mechanism note |

---

## Files

| file | role |
|---|---|
| `p4_cache.py` | trace IO (24B oracleGeneral), footprint, base cache sim |
| `p4_evict.py` | pluggable evictors: LRU, Belady, S3-FIFO |
| `p4_prefetch.py` | prefetchers (Markov-1, **Markov-2**, Prescient bound) + PFCache (byte accounting, token-bucket bandwidth match, wasted-prefetch autopsy) + `selftest()` |
| `p4_gates.py` | W1 kill-tests: format self-check, LRU parity vs libCacheSim, LRU→Belady gap, prefetch-signal, S3-FIFO parity |
| `p4_t1.py` | the T1 reference table + interaction 2×2 + autopsy (the run that killed the joint thesis) |
| `p4_sweep.py` | **Gate A** (tuned bar) + **Gate B** (interaction across traces) |
| `README.md` | W1 scope + data fetch instructions |

### Known open items
- **W1c S3-FIFO parity FAILS by 82 hits / 100k (0.08 pt)** = variant drift, not a broken sim. Does
  NOT block the build (LRU parity — the env unit test — passes EXACT; the go/no-go uses Belady, not
  S3-FIFO). But our S3-FIFO number **cannot be printed as "S3-FIFO" in a paper table** until it
  matches libCacheSim exactly. Fix = sweep variant knobs (small_frac, ghost size, freq-reset-on-
  promote) against libCacheSim. Keep the gate RED until fixed.
- **Markov-2 table is memory-hungry** (~1M pair-contexts × 16). If a CDN trace OOMs, drop
  `--train-frac` to 0.25.
- Git: remote `github.com/lkksharma/p4.git`, branch `main`. Latest commit `d9f14ca` (Gates A+B).
  **Push from Mac before pulling on the box.**

## Environment note
Prepare/self-test locally in the `badminton` conda env; **run heavy jobs on the GPU box** (user runs
them). Local python with numpy/torch: `/usr/local/bin/python3`.
