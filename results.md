# P4 Capture-Method Paper — Results Ledger (the Necessity Ladder)

Companion to [results_p4.md](results_p4.md), which is the **measurement paper** ("Phantom Headroom")
-- finished, bootstrap-confirmed, submission-ready, and NOT touched by anything below. This file
tracks the **second, separate paper**: given the proven corridor, can any causal policy capture it?
Targets AAAI-28 / IAAI-27, not the measurement paper's deadline.

**Provenance key** -- every entry is tagged so staleness is never ambiguous:
  - `[LOG]`  read directly from a `.log` file this session (cite the filename) -- highest confidence.
  - `[PASTE]` numbers the user pasted directly from a live terminal this session -- as good as [LOG].
  - `[CARRIED]` reported in a prior session's own recap, pasted into this one as prose, NOT
    independently re-verified against a log file in this repo. Treat as provisional until re-run.
  - `[SELFTEST]` synthetic-trace validation only (`--selftest`, `SYNTH`, a 149-object pathological
    cache) -- proves the CODE runs and the invariants hold, proves NOTHING about real-trace efficacy.

House rules this ledger follows (see `context.md` §5): every gate pre-registered before the run that
could satisfy it; a gate that never kills anything is not trusted; no n=1 generalization; diagnostics
(`hazard=oracle`, `forecaster=oracle`) reported as diagnostics, never as a capture result.

---

## Rung-by-rung status

| rung | question | status |
|---|---|---|
| F1 | timing-only ceiling on the predictor's own (k=1) candidates | `[CARRIED]` trustworthy per prior session, not re-run here |
| F4 / F4' | does joint/RL arbitration add anything beyond a threshold rule? | `[CARRIED]` RL unnecessary |
| F2 | is use-lag (the "when") learnable at all? | `[CARRIED]` PASS post-fix, not re-verified here |
| HJS-L | closed-form hazard-priced scheduler, capture vs F1 | `[LOG]`+`[PASTE]` **DEAD** -- DEFER lever killed |
| F1-stream | perfect timing on the ACTUAL k=1 emission stream | `[PASTE]` **BELOW bar on both traces** -- decisive |
| F5 | coverage ceiling: wide emission + oracle-select | `[PASTE]` built + run: c50 CLEARS, wiki inconclusive |
| Policy 1 | wide-emission bandwidth market (causal) | `[SELFTEST]` only -- real-trace run PENDING |
| Policy 2 | recall-first forecaster | not built as its own rung; `blend` v0 wired inside Policy 1, unevaluated |

---

## 1. F1 -- timing-only oracle `[CARRIED]`

Bug found and fixed (prior session): `emit_vocab` scanned the whole tau-cleared table row instead of
the real top-k `suggest()` output, inflating wiki's emit set to ~519,465 objects (near-full training
vocab), collapsing F1 onto the warm ceiling by tautology. Fixed by replaying with an always-empty
cache to harvest genuine k/tau-correct emissions.

| trace | F1 corridor | F1/v3 | gate (>=8) |
|---|---|---|---|
| wiki | +12.19 | 95.8% | clears |
| cluster50 | +16.01 | 91.2% | clears |
| cluster53 | +1.44 | 7.3% | below -- dropped from capture claim |

*(Note: `results_p4.md`'s own F1-vocab numbers, read directly `[LOG]` this session, differ slightly
-- wiki +12.72, cluster50 +17.56 -- consistent with a later `train_frac`/config tweak between
sessions. Treat the `[LOG]` numbers in results_p4.md as current; this row is the historical F1 run.)*

## 2. F4 / F4' -- RL necessity gap `[CARRIED]`

Bug found and fixed (prior session): SEP arm was arrival-order with no hard iso-bandwidth byte cap;
fixed to density-order with a hard byte cap.

| trace | F4' | verdict |
|---|---|---|
| wiki | ~+0.01 | RL unnecessary |
| cluster50 | ~-0.01 | RL unnecessary |
| cluster53 | ~0.00 | moot (F1 already below gate) |

**Standing conclusion: RL arbitration adds <1 pt everywhere. Closed the RL rung of the ladder.**

## 3. F2 -- hazard / when-learnability gate `[CARRIED]`

**First run** (hazard model trained on the SAME window as the predictor -- later found to be a bug):

| trace | Spearman | never-AUC | F2 |
|---|---|---|---|
| wiki | +0.068 | 0.433 | FAIL |
| cluster50 | +0.023 | 0.390 | FAIL |
| cluster53 | +0.650 | 0.468 | FAIL |

Bug: train/eval window overlap flips the confidence->outcome relationship (cluster50's AUC < 0.5 is
the signature). Fixed with a three-way split (predictor / hazard-model / eval) + fixed observation
horizon.

**Post-fix run:**

| trace | Spearman | never-AUC | F2 |
|---|---|---|---|
| wiki | +0.604 | 0.920 | PASS |
| cluster50 | +0.429 | 0.812 | PASS |
| cluster53 | +0.611 | 0.952 | PASS |

**Caveat: this post-fix run was reported to me as already-done in a prior-session recap. It has not
been independently re-verified against a log file in THIS conversation.** Given how decisively HJS-L
and F1-stream later killed the *timing* lever regardless, re-verifying F2 is low-priority now -- the
gate it feeds (HJS-L) is already dead for a different, independently-confirmed reason (see §4).

## 4. HJS-L -- closed-form hazard-priced scheduler `[LOG]` + `[PASTE]` -- **DEAD**

### 4a. Early runs (prior-session recap, `[CARRIED]`)

Run A (`gamma=0.75, floor=0.05, defer=hazard`): wiki corridor -40.10 (CI [-42.69,-37.48]), cluster50
-29.65 (CI [-31.58,-27.68]). Both FAIL, invariant FAIL (traffic > bar). Issue autopsy: LATE dominates
(43.2% wiki) -- deferral overshoot.

Bug found: `_quantile` used the log-bin MIDPOINT as the wake offset; observed median use-lag (~16) sits
below the midpoint (20.5), so every wake overshot. Fixed to use the bin's LOWER edge. Defaults changed
`gamma` 0.75->0.10, `floor` 0.05->0.30.

Run B (post-fix defaults, still `defer=hazard`): still catastrophic -- wiki 0.1525 OHR / -40.07
corridor, cluster50 0.4148 / -29.32. **The quantile fix did not rescue the arm.**

### 4b. Confirmed `[LOG]` -- floor sweep at `defer=hazard`, still broken

Read directly from `hjsl_wiki_floor0.30.log`, `hjsl_wiki_floor0.40.log`, `hjsl_wiki_floor0.50.log`
(all `gamma=0.1, defer=hazard, survival=km`):

| floor | HJS-L OHR@tx | LATE % | corridor | CI | capture |
|---|---|---|---|---|---|
| 0.30 | 0.1525@1.05x | 55.4% | -40.07 | [-42.65,-37.46] | -328.8% |
| 0.40 | 0.1521@1.32x | 55.9% | -40.06 | [-42.65,-37.45] | -328.8% |
| 0.50 | 0.1527@1.04x | 59.3% | -40.04 | [-42.62,-37.43] | -328.6% |

**Floor is not the lever. LATE stays pinned at ~55-59% regardless of floor -> the failure is
deferral overshoot, not over-funding.**

### 4c. Fairness/protocol control -- BAR-CAP introduced `[CARRIED]`, confirmed `[LOG]`

`defer=none, hazard={model,oracle}`, real traces:

| trace | hazard | BAR-CAP | HJS-L | on-time | corridor | HJS-L vs BAR-CAP |
|---|---|---|---|---|---|---|
| wiki | model | 0.4673@1.09x | 0.4039@1.03x | 99.2% | -14.93 | -6.35 |
| wiki | oracle | 0.4673@1.09x | 0.4685@1.02x | 100.0% | -8.47 | **+0.11** |
| cluster50 | model | 0.6440@1.03x | 0.5804@1.00x | 98.1% | -12.76 | -6.37 |
| cluster50 | oracle | 0.6440@1.03x | 0.6459@1.00x | 100.0% | -6.21 | **+0.19** |

C1-C6 hypothesis sweep (prior session) eliminated HOL blocking, ranking bias, and forecaster
weakness as causes; landed on floor over-filtering (C4) and rate-cap unfairness (C5) as plausible,
plus the wrong-denominator critique (C6, later independently confirmed by F1-stream, see §5).

### 4d. `[LOG]` -- `budget_mode=total` (removes the rate-cap confound cleanly)

Read directly from `hjsl_wiki_total.log`, `hjsl_wiki_total_oracle.log`, `hjsl_c50_total.log`,
`hjsl_c50_total_oracle.log` (`defer=none, gamma=0.1, floor=0.30`):

| trace | hazard | OHR@tx | pf issued/useful (prec) | spent | corridor | CI | capture | two-axis |
|---|---|---|---|---|---|---|---|---|
| wiki | model | 0.4089@1.03x | 592,876/569,830 (0.961) | 0.61x | -14.43 | [-15.38,-13.46] | -118.4% | traffic saved (0.53x) at a cost |
| wiki | **oracle** | 0.5543@1.02x | 869,596/858,957 (0.988) | 0.98x | **+0.11** | [+0.10,+0.12] | 0.9% | **PARETO WIN, 0.88x bytes** |
| cluster50 | model | 0.5915@1.00x | 364,150/335,896 (0.922) | 0.62x | -11.65 | [-12.42,-10.87] | -72.8% | traffic saved (0.54x) at a cost |
| cluster50 | **oracle** | 0.6774@0.99x | 516,559/516,559 (1.000) | 1.00x | **-3.06** | [-3.85,-2.30] | -19.1% | traffic saved (0.87x) at a cost |

`PROTOCOL COST +0.00` in every `total`-mode run: the rate-cap confound is fully eliminated, not just
reduced. **What's left after removing it is the true finding.**

### Standing conclusion (§4)

`S(delta)=1.000` on all three real traces out to delta=100 (prefetched objects essentially never
evicted before use within that window) -> **there is no eviction race to win by deferring.** DEFER is
strictly counter-productive here. Even with a **perfect forecaster** and DEFER removed, HJS-L only
**ties** the bar (wiki +0.11, cluster50 -3.06) -- it does not beat it. **HJS-L is dead as a capture
mechanism.** The oracle-vs-model gap (wiki +14.5 pts of corridor recovered by a perfect forecaster,
c50 +8.6 pts) is real and important, but it is a *forecaster* gap, not something HJS-L's scheduling
design can fix -- confirmed independently by §5.

## 5. F1-stream -- the decisive re-test (C6) `[PASTE]`

Read directly from `f1_wiki_stream (1).log`, `f1_c50_stream (1).log`. This is PERFECT timing (JIT
oracle, drops never-used candidates, prec 1.000) applied ONLY to the actual k=1 emission stream --
not the vocab-restricted clairvoyant ceiling.

| trace | BAR | F1-stream | corridor | CI | F1-vocab corridor (reference) |
|---|---|---|---|---|---|
| wiki | 0.5531 | 0.4830 | **-7.01** | [-7.68,-6.33] | +12.72 |
| cluster50 | 0.7080 | 0.5998 | **-10.82** | [-11.70,-9.93] | +17.56 |

Invariant FAIL on both (`OHR(F1) < OHR(bar)`) -- flagged by the code itself as the expected signature
of stream-mode under-reading via second-order eviction/HOL, not a bug in the arm.

**Standing conclusion: perfect timing on the actual causal emission stream is BELOW the bar on both
traces.** The +12.72 / +17.56 corridor is not reachable by scheduling what the predictor already
names -- it requires naming MORE objects. This is what motivated F5.

## 6. F5 -- coverage-ceiling gate `[PASTE]` -- **built and run this session**

New file: `p4_f5.py`. Ceiling construction: a use of object X at t is "causally coverable" iff the
wide predictor emitted X since its previous use; F5 = clairvoyant JIT prefetch restricted to
covering only coverable uses, at the bar's byte rate. By construction F1-stream <= F5 <= warm ceiling.

Two real bugs caught and fixed during `[SELFTEST]` validation before the real-trace run: (1) the
invariant compared F5 against the k=1 F1-vocab set instead of the true warm ceiling (F5's wider net
can legitimately exceed a k=1 vocab without exceeding the true ceiling); (2) the iso-BW invariant
checked origin bytes instead of prefetch bytes (origin bytes can rise from extra misses, which is a
performance signal, not a bandwidth violation). Both fixed; selftest invariants all pass post-fix.

**Real-trace results, `wide_tau=0.0`, `--wide-k 16`, `top_m=16`:**

| trace | coverage | F5 OHR@tx (pf ratio) | corridor | CI | % of warm corridor | fork |
|---|---|---|---|---|---|---|
| wiki | 71.6% | 0.6349@1.04x (0.95x) | **+8.18** | [+6.65,+9.72] | 64.3% | straddles 8 at top_m=16 -> resolved below |
| cluster50 | 71.6% | 0.8221@1.05x (0.95x) | **+11.42** | [+9.95,+12.91] | 65.0% | **CLEARS 8** -- decisive |

**wiki wider-net confirmation** `[PASTE]` (`f5_wiki_wide32.log`, `--wide-top-m 32`) -- the
pre-registered "widen the net" escape, resolving wiki's top_m=16 straddle:

| k | coverage | F5 corridor | CI | % of warm corridor | fork |
|---|---|---|---|---|---|
| 24 | 73.1% | +9.60 | [+7.99,+11.23] | 75.5% | straddles 8 |
| 32 | 74.1% | **+10.41** | [+8.74,+12.08] | 81.8% | **CLEARS 8** |

**Fanout sweep** (`--wide-k 1,4,8,16`):

| k | wiki corridor | wiki coverage | cluster50 corridor | cluster50 coverage |
|---|---|---|---|---|
| 1 | -6.24 | 53.6% | -12.97 | 23.9% |
| 4 | +1.19 | 63.3% | +1.90 | 60.0% |
| 8 | +4.92 | 67.9% | +6.41 | 66.0% |
| 16 | +8.18 | 71.6% | +11.42 | 71.6% |

Coverage was STILL RISING at k=16 on both traces (not saturated) -- capped only by `top_m=16` per
context, an implementation limit, not a property of the hypothesis. `p4_f5.py` gained `--wide-top-m`
this session specifically to allow a wider-net re-run.

### Standing conclusion (§6) -- RESOLVED, both live traces clear

**BOTH live traces clear the pre-registered F5 gate (corridor >= 8, CI lo > 0):**
- **cluster50**: +11.42, CI lo +9.95 (at k=16, top_m=16).
- **wiki**: +10.41, CI lo +8.74 (at k=32, top_m=32).

wiki required a wider net (top_m 16 -> 32) to clear -- but the +8 bar was fixed before any F5 run,
and "widen the net" was the fork's own pre-registered escape, so this is a clean pass, not a moved
goalpost. Coverage was still rising with fanout on both traces (wiki 53.6 -> 74.1% across k=1..32),
confirming the k=16 straddle was an implementation cap (top_m), not a property of the hypothesis.

**The coverage lever is confirmed on both live traces. Policy 1 is fully justified.** Note the
config asymmetry it implies (see §7 open item): cluster50's lever clears at k=16, wiki's needs
k=32/top_m=32 -- so a fair Policy 1 evaluation must use the trace's OWN F5-clearing width as its
denominator (c50 at k=16, wiki at k=32), not a shared k=16.

## 7. Policy 1 -- wide-emission bandwidth market -- `[SELFTEST]` + first real-trace `[PASTE]` (markov/rate BELOW bar) -- full run PENDING

New file: `p4_policy1.py`. Causal realization of the F5 ceiling: emit wide, value each candidate with
a pluggable forecaster (`markov` = Policy 1 proper, `blend` = Policy 2 v0 recall-reweight, `oracle` =
diagnostic ceiling, never reportable), fund highest value-per-byte first under a token bucket + hard
total-byte cap. No DEFER (dead per §4), no RL (dead per §2).

One bug caught in selftest and fixed: the oracle forecaster's value function was a flat 0/1
"used-within-horizon" label, which is non-discriminative (nearly everything gets funded, ranked only
by size) and made oracle the WORST arm on synthetic data -- backwards for a diagnostic ceiling. Fixed
to weight by inverse time-to-use (prioritizes soonest, lowest-occupancy uses, mirroring F5's own
selection rule). Post-fix, selftest ordering is sane (oracle > markov > blend on SYNTH).

**`[SELFTEST]` only (SYNTH, 149-object pathological cache -- validates code correctness and
invariants, NOT real-trace efficacy):** all arms land below the SYNTH bar, expected -- F5 wins on
SYNTH via clairvoyant just-in-time insertion in a tiny cache; the causal market fetches at emission
and holds until use, which thrashes a 149-object cache. Real traces have `S(100)=1.000` (objects
survive far longer than the median use-lag of ~16), so this pathology should not transfer -- **only a
real-trace run can confirm this.**

**Pre-registered pass bar: capture >= 25% of the F5 corridor, bootstrap CI excluding zero.**

### 7a. First real-trace signal `[PASTE]` -- cluster50, markov arm, RATE mode, floor sweep -- BELOW bar

Read directly from `policy1_c50_floor0.0.log`, `policy1_c50_floor0.05.log` (floor=0.10 issued, not
yet returned). **markov arm only** (no oracle/blend arm here to isolate design vs forecaster), and
**`budget_mode=rate`** -- the token-bucket fairness confound HJS-L showed costs -6 to -8.6 pts is
present and uncontrolled in these runs.

| floor | OHR@tx | pf issued/useful (prec) | spent | corridor | CI | capture | failure mode |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.6372@1.31x | 841,224/420,338 (**0.500**) | 1.00x | **-7.08** | [-7.51,-6.65] | -62.0% | THRASH |
| 0.05 | 0.5053@1.00x | 164,205/138,778 (0.845) | **0.07x** | **-20.26** | [-21.61,-18.90] | -177.5% | STARVE |

**Both below the bar (0.7080), and there is no floor sweet spot between them:**
- **floor=0.0 (thrash):** funds the full wide net; precision collapses **0.863 -> 0.500**. The
  value/size ranking floods the cache with small, low-confidence candidates (issues 841k vs the bar's
  665k at the *same* prefetch bytes -> smaller objects), displacing useful content. Origin traffic
  rises to **1.31x** while prefetch bytes stay at 0.95x -- a thrash signature, not a BW violation.
- **floor=0.05 (starve):** the floor operates on markov confidence, and the wide net (emitted at
  tau=0.0) is mostly conf < 0.05, so floor=0.05 drops nearly the whole net -> spent **0.07x** -> the
  market barely prefetches -> OHR falls toward base. The useful *middle* of the wide net is exactly
  where raw markov confidence fails to separate signal from noise.

**Honest read -- do NOT conclude "Policy 1 fails" from this.** Three things are missing before any
verdict, in decreasing order of decisiveness:
1. **The oracle-forecaster arm** (isolates market DESIGN from forecaster quality). If
   `market(oracle) ~ F5 (+11.42)`, the fetch-at-emission market design is sound and the **markov
   forecaster is the bottleneck -> Policy 2 justified**. If `market(oracle) << F5`, the market design
   itself is limited (fetch-at-emission thrashes even with a perfect value signal). This single arm
   decides the branch and it is NOT in these logs.
2. **`--budget-mode total`** -- removes the rate-cap confound, as it did for HJS-L (protocol cost went
   to +0.00). Must be run before charging any deficit to the scheduler.
3. **wiki** (running at k=32) -- n=1 is not a verdict.

**Emerging lean (provisional, unconfirmed):** markov confidence is too imprecise on the wide net
(prec 0.500) to exploit a coverage lever that oracle-select F5 proves is worth +11.42. This points
toward **Policy 2 (a better per-candidate value function) being REQUIRED, not optional** -- but the
oracle arm must confirm `market(oracle) ~ F5` first, or the deficit could be a market-design problem
that Policy 2 cannot fix.

**Prediction-vs-reality note (on the record, house rule #2):** the prior-session "educated
prediction" put markov/rate/cluster50 at ~-2 to +3; reality is -7.08 (floor 0). The miss
under-weighted value/size small-object flooding on a wide, noisy net. Logged so the calibration error
stays visible rather than quietly forgotten.

### 7c. Full markov+blend+oracle run `[PASTE]` -- DISTORTED by `--max-pool 512`, NOT a verdict

Read from `policy1_c50.log`, `policy1_wiki_k32.log` (full 3-arm runs, `--max-pool 512`, RATE mode).
**These numbers are not trustworthy and are recorded as diagnostic-only**, for three reasons; the
first is a self-caught error in the tooling I introduced:

| trace | markov | blend | oracle | markov pf issued / prec | POLICY-2 auto-verdict |
|---|---|---|---|---|---|
| cluster50 | -0.77 | -0.95 | **-2.09** | 4,488,039 / **0.103** | "Policy 2 unnecessary" |
| wiki k=32 | -8.27 | -7.98 | **-4.88** | 1,932,512 / 0.352 | "Policy 2 justified" |

1. **`--max-pool 512` is NOT a neutral speed knob (my error -- earlier claimed it was).** Capping the
   pool by value/size systematically evicts large objects (they rank last per byte), so the pool fills
   with tiny speculative candidates -> 4.5M prefetches at prec 0.103 -> cache flooding. The corridor
   numbers are an artifact of this bias, not the market's real behaviour.
2. **The two auto-verdicts contradict each other** (c50 "unnecessary" vs wiki "justified") -> the
   readout, which assumes oracle >= markov, is unreliable here. On c50 oracle (-2.09) is BELOW markov
   (-0.77): a perfect forecaster performing worse than a dumb one is a broken-arm signature.
3. **RATE mode confound uncontrolled** (no BAR-CAP; the bar bursts, the market is rate-limited).

**The one signal that survives all three confounds:** the oracle arm is **below the bar on both
traces** (-2.09, -4.88), and on c50 it funds MORE useful prefetches (471k vs 461k) yet gets LOWER OHR
(0.687 vs 0.700) -- the eviction-coupling / substitution effect. A causal fetch-at-emission market,
even with a perfect forecaster, churns the cache on the wide stream and loses to the bar. This is the
SAME pattern HJS-L's oracle arm showed (§4c/§4d).

**Diagnosis (hypothesis, to be tested):** `S(100)=1.000` -- the "no eviction race, so DEFER is dead"
premise (§4) -- was measured under the BAR's modest prefetch load. The wide market issues millions of
prefetches; that churn very plausibly breaks the survival premise, re-introducing the eviction race.
F5 wins (+11.42) because it inserts JUST-IN-TIME (minimal occupancy); the at-emission market loses
because it holds. **The wide market may need DEFER re-introduced -- the opposite of the k=1 finding.**

**Fixes applied this session (before any re-run):**
- oracle value function: horizon 100k -> 2000 (a perfect forecaster does not fund a use so distant the
  object is evicted first; 100k over-funds and understates the ceiling). Makes the oracle arm a true
  ceiling -> the decisive test of the DEFER-for-wide fork.
- pool cap: now keeps highest RAW VALUE, not value/size, so it no longer evicts large useful objects
  (de-biases the flooding the -512 run exposed).

**PENDING (clean re-run, before any verdict):** rate AND total mode, moderate `--max-pool`, tightened
oracle. If the clean oracle arm still loses to the bar -> DEFER-for-wide fork is on. If it reaches ~F5
-> market design is sound and markov is the bottleneck -> Policy 2.

### 7d. Clean 4-config runs `[PASTE]` + EVAL AUDIT -- last verdict RETRACTED

Clean runs (de-biased pool, tightened oracle horizon=2000), markov+blend+oracle, rate & total:

| trace / mode | markov (reportable) | oracle | oracle prec / spent |
|---|---|---|---|
| cluster50 rate | -0.70 | -3.35 | 0.824 / 0.69x |
| cluster50 total | -3.08 | -3.07 | 1.000 / 1.00x |
| wiki rate | -8.39 | -6.22 | 0.956 / 0.60x |
| wiki total | **+0.05** | **+0.62** | 0.985 / 1.00x |

**Solid finding:** the reportable **markov market ties/loses the bar in every config** (best wiki-total
+0.05, a tie; F5 ceiling is +10 to +11). A wide-emission value/size market with markov confidence does
NOT capture the coverage corridor.

**EVAL AUDIT (self-caught, this session) -- last turn's "Policy 2 can't help, revive DEFER" is
RETRACTED as not-established:**
1. **[HIGH] The oracle arm was confounded.** It ranked by `value/size` -- the same dispatch rule as
   every arm -- so "oracle can't beat the bar" could not separate (a) fetch-at-emission timing, (b)
   the value/size dispatch rule, or (c) the forecaster. And the F5-vs-oracle gap I blamed on "needs
   JIT" conflated timing (JIT vs emission) AND selection (F5 funds soonest-first, oracle funded by
   value/size). The attribution to at-emission timing was NOT supported.
2. **[MEDIUM] No BAR-CAP.** Rate-mode corridors conflated scheduling with the token-bucket protocol
   cost (bar bursts, market is rate-limited) -- the same confound HJS-L controlled for.
3. **[LOW, conservative] Warmup budget accounting** -- `rate = bar_pf_bytes/n` (total n) vs post-warmup
   bytes; capture arms spend ~0.95x post-warmup. Consistent across all experiments, handicaps capture
   arms (never inflates a positive). Left as-is for cross-experiment comparability.

**FIXES applied:** (1) oracle now ranks by imminence (soonest-first) = F5's OWN selection, so
oracle-vs-F5 isolates JIT-vs-at-emission and oracle-vs-markov isolates forecaster quality under a fixed
rule; (2) BAR-CAP added to the report (protocol cost + market-vs-BAR-CAP); (3) warmup documented.

**PENDING (decisive):** re-run the fixed oracle. If oracle (F5's selection, at emission) captures
<25% of F5 -> the AT-EMISSION DESIGN is the ceiling -> DEFER/JIT fork. If oracle nears F5 but markov
lags -> Policy 2 (forecaster). The markov ties/loses result above stands regardless.

### 7e. Clean audited verdict `[PASTE]` -- at-emission is the ceiling, JIT is the missing ingredient

Fixed-oracle runs (oracle uses F5's OWN soonest-first selection, so oracle-vs-F5 isolates JIT vs
at-emission; BAR-CAP added). Files `p1v2_{c50,wiki}_{rate,total}.log`:

| trace / mode | oracle-market (at-emission) | F5 (JIT) | JIT gap | oracle capture of F5 |
|---|---|---|---|---|
| cluster50 total | 0.6773 | 0.8221 | **14.5 pts** | -27% |
| cluster50 rate | 0.6745 | 0.8221 | **14.8 pts** | -29% |
| wiki total | 0.5593 | 0.6572 | **9.8 pts** | +6% |
| wiki rate | 0.4910 | 0.6572 | **16.6 pts** | -60% |

**CLEAN, AUDITED FINDING:** with the SAME clairvoyant selection (F5's), a perfect forecaster fetching
AT EMISSION loses 10-15 pts to JIT insertion, in every config. At-emission captures <=6% of the
corridor. **JIT insertion is the missing ingredient, isolated cleanly** (no dispatch-rule confound).
BAR-CAP reproduced HJS-L's protocol costs exactly (c50 -6.40, wiki -8.58 -- consistency check).

### 7f. F7 JIT-realizability gate -- BUILT, decisive, PENDING real run

New file `p4_f7.py`. F5's clairvoyant coverage+selection (funds only will-be-used objects), varying
only WHEN it inserts: `oracle` (true next-use = JIT ceiling / construction check), `model` (causal
hazard use-lag), `point` (median). model-vs-oracle isolates timing ACCURACY. Autopsy: on-time vs LATE.

Selftest (SYNTH): construction validated (oracle reproduces a strong JIT ceiling, no crash, autopsy
works, embedded F2 PASS). Two SYNTH-only caveats handled in reporting: (1) F7-oracle need not equal
F5 (different object selection) -> the model arm is scored vs F7-ORACLE (same mechanics), F5 shown
for continuity; (2) SYNTH `model` was 99% on-time yet prec 0.157 -- an EARLY-EVICTION artifact of
SYNTH's `S(100)=0.020` (median lifetime 1 req), which will NOT transfer to real traces where
`S(100)=1.000`. So on real traces the only failure mode is LATE (overshoot), the HJS-L k=1 signature.

**PRE-REGISTERED GATE: model captures >= 50% of the JIT ceiling, bootstrap CI excluding zero, sweep
gamma.** PASS -> build the full DEFER-for-wide policy (positive capture method). FAIL -> corridor
needs clairvoyant timing; no causal policy captures it (clean negative; measurement half carries the
paper). Fair-shot fallback if model fails: retrain the hazard model on the WIDE emission stream (it
was trained on k=1) before declaring causal timing dead.

**PENDING:** `p4_f7.py` on wiki + c50, sweeping `--gamma {0.1, 0.25, 0.5}`.

### 7g. First real F7 runs `[LOG]` -- INVALID as numbers (self-caught harness bug), decisive as diagnostics

Logs: `f7_wiki_g0.1.log`, `f7_wiki_g0.1_mp4096.log`, `f7_wiki_g0.1_mp1024.log`, `f7_wiki_g0.25.log`
(partial), `f7_c50_g0.1.log`.

**Run 1 (unbounded pool):** model arm quadratic-stalled at ~50% (3h silent). Root cause: no size cap
on `pending`/`heap`; the pool explodes at **i ~ 1M -- the train/eval boundary** (out-of-sample
emissions have farther next-uses; in-sample the pool sat stable at ~0.5-2.7k), and the full backlog
is re-drained/re-sorted/re-pushed every request.

**Fix 1 (`--max-pool`, size-keyed, commit 48ab828) un-hung it but BROKE the oracle arm** -- the
construction check read capture **-60.1% of F5** on wiki (mp4096), **-75.7%** (mp1024), **+2.0%** on
c50, vs **+93.1%** in the valid pre-fix partial (`[oracle] OHR 0.6500, spent 0.98x, 100% on-time`).
Pruning by SIZE drops the soonest-WAKE large candidates the oracle funds; spend froze at 0.52x. The
mp1024-vs-mp4096 sweep caught it (oracle differed -> cap-sensitive -> unreportable). **Fix 2 (commit
77050b4):** oracle arm exempt from the prune (it never hung), causal arms pruned by SOONEST WAKE with
hysteresis, and the construction check is now a hard invariant (oracle must read ~100% of F5 or the
GATE line prints INVALID).

**Diagnostics that ARE valid (first half of each run, cap not binding -- identical dynamics to the
unbounded run):**
- wiki gamma=0.1 (the most favorable setting): model LATE **74% -> 92%** while spending at the token
  rate; final prec **0.047-0.055**; forecast-overshoot **17% of emissions** even at the earliest
  wake quantile. gamma=0.25 partial: LATE **95%** at the 10% mark -- monotone worse, as predicted.
- c50 gamma=0.1: LATE ~93% through the first half; prec 0.034.
- **Mechanism (new, publishable):** on-time 75-90% yet useful only 3-7% -> early fetches die in
  cache before their use. `S(100)=1.000` was measured under BAR load; under the model arm's own
  ~1.2M-fetch load survival collapses -- **the timing slack is ENDOGENOUS** (the more you mistime,
  the more you fetch, the tighter the residence window gets). Clairvoyant timing needs zero slack;
  log-binned hazard timing has multiplicative error on heavy-tailed lags -> overshoot AND
  early-death simultaneously, at every gamma. This is why F2-PASS (rank-learnable, rho 0.60)
  coexists with F7-FAIL: **rank-learnable != placeable.**

The verdict is all but sealed (model at roughly -380% of F5 vs a >=+50%-of-ceiling gate -- no
harness correction crosses that gap), but per house rules it is quotable only from a run whose own
construction check passes. **PENDING (final):** repaired-harness re-run at gamma=0.1 on both traces
with the k=1 hazard (the pre-registered primary), then the fair-shot fallback (hazard retrained on
the WIDE stream -- `p4_hazard.py --top-m` wired in 77050b4 for exactly this).

## 8b. Cross-domain screen: LLM KV-cache serving (Mooncake, FAST'25) `[LOG]` -- FORK 1, clean

New file `p4_llmcache.py` (commits 29bbb69, 3d3dea3): the instrument ported to KV-cache WARMING on
Mooncake's production traces. Domain-honest port: TURN-GATED warming (a request's blocks are read
simultaneously at prefill, so every arm -- bar AND ceiling -- may only warm at request boundaries;
the corridor measures cross-turn warming = the TTFT lever); the vocab restriction is PHYSICAL here
(a block never computed+stored cannot be warmed by any system). Pre-registered fork fixed in the
docstring BEFORE the first real run; bars inherited unchanged (corridor >= 8, CI>0, Pareto; F2
0.20/0.60). Two self-caught arm bugs, both flagged by the script's own invariants before any
verdict was read: (1) the release queue dropped candidates the token bucket couldn't yet afford,
starving the rate-limited ceiling (-1.5pt inversion on selftest); (2) the burst-vs-bucket protocol
confound (bar bursts, ceiling rate-limited) flipped the corridor sign -- controlled with a BAR-CAP
arm per HJS-L/Policy-1 precedent. Gate unchanged throughout.

Logs `llmcache_conv_v2.log`, `llmcache_toolagent_v2.log` (1% cache, full markov1/2/3 x STRONG_TAUS
x k grid, invariants ALL PASS):

| trace | base | tuned bar | BAR-CAP | warm ceiling | corridor vs BAR | CI | vs BAR-CAP | protocol | F2 rho/AUC |
|---|---|---|---|---|---|---|---|---|---|
| conversation (12,031 req, 289k accesses) | 0.0786 | **0.1509** (m2 t.05 k4, prec .862) | 0.1155 | 0.1453 | **-0.56** | [-1.14,+0.04] | +2.98 | +3.55 | +0.268 / 0.805 PASS |
| toolagent (23,608 req, 410k accesses) | 0.3664 | **0.4435** (m3 t.05 k4, prec .922) | 0.4034 | 0.4248 | **-1.87** | [-2.38,-1.36] | +2.14 | +4.01 | +0.003 / 0.993 FAIL |

**Standing conclusion: FORK 1 on both traces -- NO corridor.** The tuned Markov bar IS the ceiling
in this domain: prefix-chain reuse is so predictable (W2b association 0.728 vs popularity 0.008 on
conv) that a tuned table already captures everything a clairvoyant, physically-constrained,
iso-bandwidth warmer could. Even like-for-like (vs BAR-CAP) the clairvoyant margin is +2-3 pts,
far below the 8-pt gate. Fork 2 (conquest domain) did NOT materialize: conv's think-time is
learnable (F2 PASS) but there is no corridor for it to capture; toolagent has neither.

**What this buys the paper:** the instrument now discriminates across THREE workload families with
three distinct outcomes -- CDN/KV: corridor EXISTS but is clairvoyance-priced (the triple-kill);
MSR/block: corridor absent below the gate; LLM KV serving: corridor ABSENT because a tuned causal
warmer already sits on the ceiling. In every case the pre-registered verdict is "do not build the
learned scheduler," reached BEFORE any model was trained -- the discipline as a portable decision
procedure, not a one-domain autopsy. Limitations (stated): 1-hour traces, one provider, 1% cache
primary, access-index lags.

## 8. Policy 2 -- recall-first forecaster -- not yet its own rung

Decision logic wired inside Policy 1's report (`POLICY-2 PRIZE = market(oracle) - market(markov)`):
if the prize is small, Policy 2 is unnecessary; if `blend` closes most of the prize, ship the cheap
reweight and skip a neural model; if the prize is real and `blend` stalls, the neural recall
forecaster (EA-MTPP's *what*-predictor only, its timing/diffusion machinery discarded per §5) is
justified. **Not evaluated on real data yet -- gated on the Policy 1 real-trace run (§7).**

---

## Open items, in priority order

1. **Read back `policy1_c50.log` / `policy1_wiki.log`** (running) + **`policy1_wiki_k32.log`** (queued
   -- wiki's market must be evaluated at k=32/top_m=32, where its coverage lever actually clears F5,
   not the conservative k=16). The real-trace Policy 1 result -- decisive for whether this paper has a
   positive capture number at all.
2. ~~Read back `f5_wiki_wide32.log`~~ **DONE** `[PASTE]` -- wiki F5 clears at k=32 (+10.41, CI lo
   +8.74). Both live traces now pass the F5 gate. See §6.
3. Route to Policy 2 (or not) per the `POLICY-2 PRIZE` line in the Policy 1 output.
4. Re-verify F2 `[CARRIED]` numbers against current code if HJS-L is ever revisited (low priority --
   HJS-L is already independently dead per §4/§5 regardless of F2's status).
5. Lock every number here into a paper draft only after its provenance tag is `[LOG]` or `[PASTE]`,
   never while still `[CARRIED]` or `[SELFTEST]`.
