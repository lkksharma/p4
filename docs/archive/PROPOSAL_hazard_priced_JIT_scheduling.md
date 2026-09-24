# Proposal: Learning to Schedule Prefetches — Neural Hazard Forecasting + Oracle-Schedule Distillation (HJS-L)
### Answer to the R&D prompt "Propose a Method to Capture a Measured Cache-Prefetch Timing Gap"

**AI-first framing (v2 of this proposal).** The method's two core components are learned; the
pricing controller survives only as the budget-safety layer. (1) The when-forecaster is a **neural
temporal point process** (NTPP) — an active ML research area (conditional intensity modeling of
event streams), not a fitted curve. (2) The fetch-timing policy is **distilled from a computable
oracle teacher**: in deterministic replay the clairvoyant just-in-time schedule is exactly
computable per candidate, so the policy is learned by regressing a causal student onto a
train-time-computable optimal teacher — the same *relaxation-anchoring* mechanism validated in the
companion Domain-B study (regress the computable oracle's values, act greedily; conditional-mean
regression preserves the improvement guarantee through the information constraint). This makes the
paper an ML-methods contribution — *decision-aware event forecasting + oracle distillation for
budgeted scheduling* — with caching as the flagship application, rather than a systems paper with
an ML garnish.

---

## 1. The exploration, before convergence

Six candidate framings were considered against the evaluation contract (§5 of the prompt) and the
prior art (§4). Summary of why five were set aside:

| Framing | Verdict | Reason |
|---|---|---|
| Offline / online RL over fetch actions | **Baseline, not method** | The problem decomposes per-candidate once bandwidth is priced (see below); RL re-learns that structure from samples, pays variance for it, and lands nearest to DeePref/DEAP in §4 — weakest novelty position. Keep as an ablation arm. |
| Restless-bandit / Whittle index | Absorbed | The Whittle index for this problem *is* a per-candidate marginal-value-per-byte threshold against a budget multiplier — i.e., it reduces to the pricing method below, minus closed-form tractability (indexability proofs, per-arm value iteration). Take the structure, skip the machinery. |
| Rolling-horizon MPC / knapsack per step | Ablation (upper bound) | Needs the same time-of-use estimates as the chosen method but pays an optimization solve per request; use "MPC with oracle hazards" as a diagnostic ceiling, not the deployed mechanism. |
| Earliest-deadline-first & classical scheduling heuristics | Ingredient | EDF has the right skeleton (deadlines order urgency) but no notion of confidence, size, retention risk, or a soft budget. Its skeleton survives inside the method as the wake-up queue. |
| Queueing/fluid control of the token bucket alone (Pythia-style global dial) | Rejected | A single aggressiveness dial cannot express *per-candidate timing*, which is where the corridor was measured to live. |
| **Survival-analysis timing + dual-priced budget** | **CHOSEN** | Matches the measured physics of the gap exactly; every component is either closed-form or plain supervised learning on data that already exists (§3 logged sweeps); iso-bandwidth compliance holds *by construction*. |

**The physics the measurement certified:** a fetch's value is a race between two clocks — the
object's time-until-request (uncertain, learnable) and its time-until-eviction once idle in the
cache (a property of the fixed eviction policy, directly measurable). Fetch too early and the
eviction clock wins; too late and the fetch misses its deadline behind the token bucket. The
corridor is the value of resolving this race well, per candidate, under a shared budget. That is a
*hazard-scheduling* problem, and it has a natural near-separable structure: candidates interact
**only** through the budget. Price the budget, and each candidate's decision becomes independent
and (almost) closed-form.

---

## 2. The method

Three components. Two are measured/learned offline; one is a 10-line online controller.

### 2.1 Per-candidate time-of-use hazard model (learned, supervised)
When the fixed predictor emits candidate *i* (object, confidence *c_i*, size *s_i*) at request-time
*t₀*, estimate a **discrete distribution over its lag-to-actual-use**:

    p_i(τ) = P(object i requested at lag τ | features),  τ binned log-uniformly (e.g. 16 bins, 1 → 10⁶ requests, plus "never")

- **Model (primary): a neural temporal point process.** A small transformer/GRU encodes the
  object's own access-gap history plus the global request context; the head outputs either a
  conditional intensity over continuous lag or a softmax over log-spaced lag bins (start with the
  binned head; the intensity head is the ML-novelty upgrade). This treats each object's accesses
  as a marked event stream — the standard NTPP setting — and the prefetch candidate's emission as
  a query about the next event time. Features additionally include predictor confidence and order
  (Markov-1/2/3/LSTM), recency/frequency stats, size.
- Calibrate with isotonic regression per bin. GBT over hand features is retained as the
  interpretable ablation (a reviewer will ask whether the NTPP earns its parameters — answer with
  this arm).
- **Training data already exists**: every logged sweep replay (§3) contains (emission, actual next
  use or absence) pairs. No new simulation is needed to train.
- Note: this does not touch the *what* (the predictor is fixed); it estimates *when* the
  predictor's own candidates get used. The "never" bin absorbs the predictor's false positives —
  the model learns to discount low-precision emissions without overriding the predictor.

### 2.2 Eviction-survival curve (measured, not modeled)
    S(δ) = P(a prefetched, not-yet-used object survives δ requests in cache)

Measured directly from the existing wasted-fetch autopsy instrumentation, per size-class and per
eviction policy (S3-FIFO's probationary queue gives prefetched objects a distinctive, short
first-life — this is exactly why "fetch at emission" wastes fetches, and why the curve must be
empirical, not assumed exponential). Recomputed over a sliding window online; no learning.

### 2.3 The scheduler: price the budget, wake at the peak
Each pending candidate has an **expected marginal utility of fetching now**:

    U_i(t) = Σ_{τ > t−t₀} p̃_i(τ | not yet used) · S(τ − (t−t₀))        [expected hit credit if fetched at t]

U_i(t) *rises* as predicted use approaches (less survival decay to pay) and collapses to a deadline:
the fetch must clear the token bucket before use, requiring lead time L(t) = s_i / r_avail(t) + safety.
So each candidate has a **peak window**: [q_i(γ) − L(t), q_i(γ)], where q_i(γ) is the γ-quantile of
its use-time distribution (γ ≈ 0.75 to start; swept).

Bandwidth is enforced by a **shadow price λ(t)** — dual ascent on the token bucket:

    every K requests:   λ ← max(0, λ + η · (bytes_spent_rate − budget_rate))

**Decision rule (the whole online policy):**

    on emission of i:  push i into a wake-up heap with wake time  w_i = q_i(γ) − L(t)
    on wake of i:      if U_i(t)/s_i ≥ λ(t):  fetch
                       elif P(use still to come) < floor:  drop
                       else:  re-arm at next quantile (γ ← γ + Δ)
    (if the object enters the cache by demand before its wake: cancel)

O(log n) per event. Iso-bandwidth compliance is structural: λ rises until spend equals budget, so
the method *cannot* buy hits with extra traffic — it satisfies the Pareto clause of the evaluation
contract by construction rather than by tuning.

### 2.4 Oracle-schedule distillation (the second learned component — the RAO mechanism, reapplied)
In deterministic replay, the **optimal fetch time of every logged candidate is computable**: given
the true use time, the token-bucket state, and the fixed evictor, the clairvoyant just-in-time
schedule (fetch as late as completion allows; skip if never used or unaffordable at any time) is
an exact, per-candidate teacher — the same epistemic object as Belady values or Domain B's Q_R.
Train a causal **policy head** π(features_t) → {fetch-now, wait, drop} by regressing onto this
teacher over the logged sweeps: at each wake-up of each candidate, the label is the teacher's
action at that instant. Per the Domain-B v3 lesson: **regress conditional means / action-values,
never softmax-CE on clairvoyant argmaxes, and never exp-weight raw oracle advantages** — outcome
luck must average out under L2, or the guarantee and the stability are both lost. At deployment
the student reads only causal features (hazard summary, bucket occupancy, λ, cache pressure);
the λ-price layer remains as a hard budget guardrail under the learned policy — the student
proposes, the price disposes. The §2.3 closed-form rule is retained as (a) the student's
initialization target and (b) the no-learning ablation that isolates what the learned policy adds
— mirroring the Domain-B finding, where distillation captured the headroom and further RL did not;
whether the richer student earns its parameters here is an explicit reported result, not an
assumption.

---

## 3. Nearest prior art and the delta (contract §5.4)

**Nearest: Baleen (FAST 2024).** Baleen gates *whether* an episode is worth admitting/prefetching
under a write-budget, via a learned probability-of-benefit against an offline OPT approximation.
HJS schedules *when* each already-emitted candidate is fetched under a shared instantaneous byte
budget, via a learned time-of-use distribution raced against a measured eviction-survival curve and
a dual price. Baleen's decision is binary and episode-level; HJS's is temporal and per-candidate.
If Baleen is "ML-When" in name, it is admission-when; HJS is literally fetch-when. The overlap is
the budget discipline; the mechanism, decision granularity, and learning target are disjoint.
Deltas from the rest of §4 in one line each: Demand-MIN changes the *eviction* objective under
prefetching (fixed here, and measured to be a substitute); DEAP/2510.10862 co-learn eviction+
prefetch (measured substitutes — co-design contradicts the evidence); DeePref picks one video per
request with no shared-budget candidate stream; Pythia tunes one global aggressiveness dial;
Cold-RL is eviction-only; LightCacheRL is hardware-line-granularity bandits without timing.

---

## 4. Falsifiers, pre-registered (contract §5.3), cheapest first

- **F1 — timing-only corridor (no ML, ~1 day):** keep the fixed predictor's emissions, give each
  candidate its TRUE use time (oracle timing), schedule just-in-time under the same token bucket.
  This is the ceiling *conditional on the fixed predictor's candidates and timing-only control*.
  **If F1 < 8 points over the tuned bar on the live traces, no timing scheduler can win and HJS is
  dead before any model is trained.** (F1 also replaces the corridor as the honest denominator for
  capture-fraction reporting.)
- **F2 — hazard learnability (~2 days):** train the §2.1 model on the train split; if the Spearman
  correlation between predicted and true lag on validation is < 0.2, or the calibrated "never" bin
  fails to separate wasted from used fetches (AUC < 0.6), the *when* is unlearnable here and HJS is
  dead regardless of the scheduler.
- **F2b — teacher-student gap (~1 day, after F2):** evaluate the distilled policy head (§2.4)
  against its own teacher on held-out replay. If the student recovers < 50% of the teacher's
  advantage over fetch-at-emission, the information constraint (causal features) is where the
  corridor dies — report which features close the gap in ablation before declaring failure.
- **F3 — capture (the main result):** HJS must capture **≥ 25% of F1's timing corridor on all three
  live traces at iso-bandwidth**, with block-bootstrap CIs excluding zero. Below that on any live
  trace → falsified; report as negative.

## 5. What is realistically reachable (contract §5.5)
The learnable ceiling (12–20 pts) assumes perfect foresight of *when*. Realistic capture is bounded
by F1 (timing-only, likely 60–85% of the learnable corridor, since the corridor was constructed on
the same candidate stream) multiplied by hazard-model fidelity (Twitter KV inter-arrivals are
strongly periodic → likely good; Wikipedia heavier-tailed → worse). Honest expectation: **30–60% of
the learnable corridor**, i.e., roughly +4 to +12 OHR points depending on trace. 100% is not
reachable by any causal method and will not be claimed.

## 6. Strongest reason it might fail
The corridor might be *timing-concentrated in unpredictable mass*: traces where the clairvoyant's
edge comes precisely from the reuses whose lags are irregular (bursty, human-driven), so p_i(τ) is
flat exactly where the value is. F2 is designed to detect this in two days. Secondary risk: the
dual price λ oscillates under bursty emission load (standard fix: exponential smoothing on the
spend-rate estimate; worst case, replace dual ascent with the bucket-deficit heuristic already
present in the token-bucket implementation).

## 7. The RL question, answered by measurement (pre-registered)

The problem is formally a constrained MDP; its Lagrangian decomposes into per-candidate threshold
policies — which is HJS. RL-as-algorithm can therefore add value **only** through what the fluid
decomposition misses: (a) integrality/knapsack effects from lumpy fetch sizes, and (b) residual
coupling between concurrent in-flight fetches. Both are bounded by a quantity computable in replay
with no training:

- **F4 — RL-necessity gap:** clairvoyant *full-MPC* oracle (joint fetch schedule, exact
  optimization over a sliding window) minus clairvoyant *threshold-form* oracle (best per-candidate
  price policy). This gap is the mathematical ceiling on ANY improvement RL could bring beyond
  oracle distillation. Pre-registered rule: gap < 1 pt on all live traces → RL is provably
  unnecessary here and the paper states so with the measurement; gap ≥ 2 pts → an offline-RL
  fine-tune of the distilled policy is added as a method arm with that gap as its target.

Slots where RL was considered and rejected regardless of F4: the λ controller (that is Pythia — a
learned global dial; it forfeits the structural Pareto guarantee and collapses the prior-art
delta) and multi-agent candidate-bidding (the equilibrium of agents under a shared resource
constraint is the shadow price; the framing re-derives λ with extra machinery). End-to-end RL over
fetch actions remains as the strawman baseline arm only (nearest neighbors: DeePref, DEAP,
LightCacheRL, arXiv 2510.10862 — all compared against, none of which schedule a candidate stream
under an instantaneous shared byte budget).

## 8. Prototype order (one week against existing infra)
1. F1 oracle-timing replay (gate).  2. Extract (emission → use-lag) pairs from logged sweeps; train
+ calibrate hazard model; F2 (gate).  3. Wire §2.3 controller (wake heap + λ) into the replayer.
4. HJS vs tuned bar on the three live traces, block-bootstrap CIs, capture fraction vs F1.
5. Ablations: λ off (greedy timing), hazard → point-estimate (no distribution), S(δ) → exponential
assumption, MPC-with-oracle-hazards (upper diagnostic), offline-RL-from-logged-sweeps (the RL
strawman, for the reviewers who will ask).
