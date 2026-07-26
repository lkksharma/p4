# SPEC — F1 and F4 Oracles: the two sub-ceilings that fork the capture paper

**Purpose.** The v3 learnable corridor is proven (+12.7 to +19.7 pts, bootstrap CIs clear of 8 on
wiki / cluster50 / cluster53). Two sub-ceilings *inside* it are unmeasured, and they decide which
paper gets written:

- **F1 — timing-only ceiling.** What can a scheduler reach given the *fixed predictor's own candidate
  stream*, with perfect timing? This is the honest denominator for "capture fraction" — tighter and
  more defensible than the v3 corridor, which credits a scheduler for the oracle's superior *choice*
  of objects (a predictor's job, not a scheduler's).
- **F4 — necessity gap.** How much can any method add *beyond a closed-form per-candidate rule*? This
  bounds what RL could contribute, and is measurable **before** training anything.

Both are replay-only. No learning, no new dependencies. Pre-register thresholds before running.

---

## 0. Preconditions

- Live traces only: `wiki_2019t`, `cluster50.sample10`, `cluster53.sample10`.
- Same protocol as every other arm: 2M requests, 1% cache, 5% warmup excluded, S3-FIFO evictor.
- Bar configs (winning strong-bar, from `results.md`): wiki `markov2 τ=.05 k=1`;
  cluster50 `markov3 τ=.06 k=1`; cluster53 `markov2 τ=.20 k=1`.
- Budget: identical `pf_byte_rate` as the bar (computed exactly as `p4_bootstrap.py` does —
  `bar["prefetch_bytes"] / trace["n"]`).
- **Report `prefetch_bytes` for every arm.** Iso-bandwidth is a claim to verify, never assume.

---

## 1. F1 — the timing-only oracle

### 1.1 Definition
Take the fixed predictor's emitted candidates (unchanged — same *what*). Replace the issue *timing*
with oracle timing: each candidate is fetched as late as possible while still preceding its true next
use, and candidates never used again are dropped instead of funded. Same token bucket, same evictor.

`F1_corridor = OHR(F1) − OHR(tuned bar)`

This is strictly a *scheduling* ceiling: it cannot fetch anything the predictor didn't name.
Necessarily `F1_corridor ≤ v3_learnable_corridor`, because the v3 warm-Prescient ceiling also gets to
choose better objects. **If they turn out close, that itself is a finding** — it would mean the v3
corridor is almost entirely timing, which strengthens the measurement paper's framing.

### 1.2 Implementation — a stateful wrapper, no changes to `PFCache`
`PFCache.run` calls `self.pf.suggest(o, cached, i)` after *every* request (`p4_prefetch.py:329`). A
scheduler can therefore hold candidates internally and release them later, returning `[]` on
most requests. **No core replayer modification is required.**

```python
NEVER = <as in p4_prefetch>          # sentinel used by oracle_next

class JITSchedule:
    """F1 arm: the base predictor's candidates, ORACLE issue timing."""
    name = "jit_oracle"

    def __init__(self, base, positions, sizes, lead=1):
        self.base, self.pos, self.sizes, self.lead = base, positions, sizes, lead
        self.pending = {}                       # obj -> wake time (issue at/after this index)

    def reset(self):                            # MANDATORY: run() calls reset() (p4_prefetch.py:220)
        self.pending = {}
        r = getattr(self.base, "reset", None)
        if r: r()

    def suggest(self, o, cached, i):
        # (1) harvest this request's candidates from the fixed predictor
        for x in self.base.suggest(o, cached, i):
            if x in self.pending or x in cached:
                continue
            nx = oracle_next(self.pos, x, i)    # TRUE next use after i, else NEVER
            if nx >= NEVER:
                continue                        # oracle drop: never used again -> never fund it
            self.pending[x] = nx - self.lead    # wake just before the use

        # (2) release everything whose wake time has arrived, most-urgent first
        due = sorted((w, x) for x, w in self.pending.items() if w <= i)
        out = []
        for w, x in due:
            del self.pending[x]
            if x in cached:                     # demand-filled while pending -> cancel
                continue
            if oracle_next(self.pos, x, i) >= NEVER:   # use already passed -> stale, drop
                continue
            out.append(x)
        return out
```

Wire it as `PFCache(cap, "s3fifo", JITSchedule(bar_predictor, pos, szs), positions=pos, sizes=szs,
pf_byte_rate=rate)`.

### 1.3 Five implementation gotchas (read before running — each is a real trap in this replayer)

1. **`reset()` is mandatory.** `run()` calls `self.pf.reset()` if present. A stateful scheduler that
   doesn't clear `pending` will leak candidates across runs and silently corrupt every sweep after
   the first.
2. **Head-of-line blocking is real.** The issue loop does `break`, not `continue`, when
   `tokens < xs` (`p4_prefetch.py:333-334`). One oversized candidate at the front of the returned
   list blocks every smaller candidate behind it *for that request*. Since F1 returns
   most-urgent-first, a large urgent object can starve small ones. Report a variant sorted
   by size-ascending as a sanity ablation; if the two differ materially, HOL blocking is a
   confound in the corridor and must be disclosed.
3. **The token bucket is unbounded.** `tokens += pf_rate` accrues with no cap. Deferral therefore
   *accumulates* budget — a genuine advantage of JIT that the greedy bar cannot get. This is a real
   effect, not a bug, but it means **F1's traffic must be checked against the bar's**, not assumed
   equal. If F1 spends materially more, it is partly a bandwidth purchase and the Pareto clause
   applies exactly as it did to cluster26.
4. **Prefetch is instantaneous.** There is no fetch-latency model; admission completes within the
   request. So `lead=1` is the true minimum. This makes **F1 an optimistic bound relative to any
   real system** (where lead time is a latency distribution). State this as a limitation; do not
   claim F1 is achievable in deployment.
5. **`oracle_next` must be recomputed at release**, not trusted from emission time — the object may
   have been demand-filled and re-evicted in between.

### 1.4 Pre-registered threshold
> **F1_corridor ≥ 8 pts on all three live traces → proceed.**
> **< 8 on any live trace → no timing-only scheduler can clear the project's own bar on that trace;
> report it and drop that trace from the capture claim.**
> **< 8 on all three → the capture program is dead before any model is built. Publish that.**

Also report `F1_corridor / v3_learnable_corridor` — the fraction of the certified corridor that is
reachable by timing alone. This ratio is a headline number in its own right.

---

## 2. F4 — the necessity gap (and a problem with it)

### 2.1 Intended definition
`F4 = OHR(joint-optimizing clairvoyant) − OHR(threshold-form clairvoyant)`, both on the same
candidate stream, both clairvoyant, both iso-bandwidth. It is meant to bound what RL can add beyond a
separable per-candidate rule — i.e., cross-candidate packing and coupling effects.

### 2.2 ⚠ The problem: as defined, F4 is close to a knapsack integrality gap, and will likely read ≈ 0
Under clairvoyant JIT timing, every funded candidate is used and survives, so each candidate's value
is ≈ **1 hit**, with weight = its size. The threshold rule "value/size ≥ λ" then collapses to a pure
**size cutoff** (`size ≤ 1/λ`), and the joint optimum is a **0/1 knapsack** over the same items.
The gap between greedy-by-density and optimal knapsack is bounded by roughly one item's value — so
on a 2M-request replay, **F4 as specified is structurally near zero**, and near-exactly zero on
uniform-size traces (MSR 4KB blocks) by construction.

That is not a discovery about RL. It is an artifact of making both arms clairvoyant, which erases the
only thing that actually makes this problem hard: **uncertainty**. A clairvoyant scheduler has no
arbitration problem to solve — it already knows which candidates pay off.

**Consequence: do not gate the RL decision on F4 alone.** It would near-certainly return "RL
unnecessary" for a reason that has nothing to do with whether RL is useful under real uncertainty.

### 2.3 F4′ — the causal necessity gap (use this as the actual gate)
Both arms causal (no future knowledge), differing only in arbitration:

- **Arm L (separable):** the closed-form per-candidate rule — hazard/confidence-derived value,
  divided by size, thresholded against a dual-ascent price λ(t). Decisions independent across
  candidates given λ.
- **Arm J (joint):** the same value estimates, but each release step solves a small windowed
  knapsack over *currently pending* candidates against currently available tokens — allowed to skip
  a large high-value item to fund several smaller ones, and to defer across windows.

`F4′ = OHR(Arm J) − OHR(Arm L)`

This is the honest ceiling on what any non-separable method — RL included — can add, because it
isolates arbitration-under-uncertainty rather than packing-under-certainty. Report F4 as well; its
value is as a **control** showing the clairvoyant gap is ~0 while the causal gap is not (or is).

**Interim value estimate for both arms** (so F4′ can run before the hazard model exists): use the
predictor's own confidence × an empirical survival factor from the existing wasted-fetch autopsy
(`pf_w_correct` / `pf_issued` by size class). Crude, but identical across both arms, so the
*difference* is still meaningful.

### 2.4 Pre-registered thresholds
> **F4′ < 1 pt on all live traces → arbitration is separable; RL is measurably unnecessary here.
> Report it as the finding. Do not train.**
> **F4′ ≥ 2 pts on any live trace → RL earns a training budget on that trace, with F4′ as its
> pre-registered target (RL must capture ≥50% of F4′, CI excluding zero).**
> **1–2 pts → inconclusive; report the number and let the closed-form policy stand as the method.**

Both thresholds fixed **before** the arms are run.

---

## 3. Sanity invariants (fail any → the arm is wrong, not the finding)

- F1 cold hits = 0 when run with `cold_train_frac` set (F1 only funds predictor-emitted candidates,
  and the bar's cold hits are 0 by construction — this must carry through).
- `OHR(F1) ≤ OHR(warm-Prescient)` on every trace. A violation means the JIT scheduler is fetching
  something the warm oracle wouldn't — i.e. a bug in candidate harvesting.
- `OHR(F1) ≥ OHR(bar)` on every trace. Oracle timing on the same candidates cannot be worse than
  immediate issue, **unless** HOL blocking (gotcha 2) is biting — which is exactly how you'd detect it.
- F4 (clairvoyant, §2.2) should be ≈ 0 on uniform-size traces. If it isn't, the joint solver is wrong.

---

## 4. Reporting format

| trace | bar OHR | F1 OHR | F1 corridor | v3 corridor | F1/v3 | F1 traffic× | F4 (clair.) | F4′ (causal) |
|---|---|---|---|---|---|---|---|---|

With paired block-bootstrap CIs on F1_corridor and F4′ via the existing `p4_bootstrap.py`
machinery (it already takes two arms' `hits_series` and bootstraps the paired difference — point it
at F1-vs-bar and Arm-J-vs-Arm-L respectively; no new statistics code needed).

---

## 5. Build order

1. `JITSchedule` + F1 on all three live traces (~half a day). Check the four invariants in §3.
2. Report F1 corridor and F1/v3 ratio. **Gate: ≥8 pts.**
3. F4 clairvoyant (§2.2) — cheap, and expected ≈ 0; it is the control that justifies moving to F4′.
4. Arm L / Arm J with interim value estimates → F4′. **Gate: ≥2 pts to train RL.**
5. Only then: the ladder's rungs iii–iv (distilled student, residual RL).

---

## 6. What each outcome means (the fork, pre-committed)

| F1 | F4′ | the paper you are writing |
|---|---|---|
| < 8 | — | Capture is impossible on the fixed predictor's stream. Measurement paper stands alone; report this as its strongest limitation section. |
| ≥ 8 | < 1 | **Closed-form policy is the method.** RL measurably unnecessary — that negative, with the number and the instrument that produced it in a day, is the novel contribution. |
| ≥ 8 | ≥ 2 | **The Necessity Ladder as designed.** RL trains against a pre-registered, measured target, anchored to the closed-form rung. |
| ≥ 8 | 1–2 | Inconclusive. Closed-form policy is the method; RL reported as an ablation that did not clear its own gate. |

Every row is publishable. That property is the reason this structure was chosen over an RL-primary
bet, and it should survive into the paper's framing.
