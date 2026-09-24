# Proposal: RAO-RL v2 — Necessity-Gated Residual RL for Budgeted Prefetch Scheduling
### An RL-primary answer to the R&D prompt "Propose a Method to Capture a Measured Cache-Prefetch Timing Gap"

**Relationship to `PROPOSAL_hazard_priced_JIT_scheduling.md` (HJS-L).** That proposal already explored
the full framing space (§1 of that document) and picked a closed-form, hazard-priced policy over
end-to-end RL, arguing RL "re-learns per-candidate decomposition from samples, pays variance for it,
and lands nearest to DeePref/DEAP — weakest novelty position." That verdict stands as a real,
reasoned finding, not a strawman. This proposal takes seriously the alternative choice — make RL the
paper's headline method — while trying to survive HJS-L's own critique: RL here does **not** learn
from scratch. It is anchored to HJS-L's own closed-form policy (not naive immediate-fetch) and its
scope is gated by a pre-registered, measured necessity number **before** any network is trained. If
that number comes back near zero, this proposal's own falsifier says so, and HJS-L's negative-RL
measurement becomes the honest paper instead.

---

## 1. Where RL sits relative to the existing machinery

Reuse, don't reinvent:
- The **hazard model** (HJS-L §2.1: per-candidate predicted use-time distribution, learned/supervised,
  trained on logged sweep data) — becomes an **input feature** to the RL state, not a rival method.
- The **eviction-survival curve** S(δ) (HJS-L §2.2, measured directly from the wasted-fetch autopsy) —
  becomes another **state feature**.
- The **dual-priced closed-form scheduler** (HJS-L §2.3: wake-heap + shadow price λ(t) dual ascent) —
  becomes the **RAO anchor**: the baseline RL starts exactly equal to, and is only asked to improve on.

RL's job is narrowed to exactly the place HJS-L's own analysis (§7) says RL could plausibly matter:
integrality/knapsack effects from lumpy fetch sizes, and residual coupling between concurrent
in-flight fetches — structure a per-candidate closed-form threshold rule cannot see by construction.

---

## 2. Formalization

**Decision points.** Whenever the fixed predictor emits a candidate, and at each scheduled wake-check
for previously-emitted, not-yet-decided candidates (reuse HJS-L's wake-heap for efficiency — an
implementation detail, not a modeling choice).

**State** `s` for candidate *i* at wake time *t*:
- candidate features: predictor confidence, size, predicted use-time distribution `p_i(τ)` (from the
  reused hazard model), time-since-emission
- budget state: token-bucket occupancy, accrual rate, current shadow price `λ(t)`
- cache-pressure state: eviction-survival estimate `S(δ)` for the current cache tail
- (optional, the genuinely new information RL could exploit) a summary of *other currently-pending*
  candidates competing for the same budget window — size, urgency, count — since this is exactly the
  cross-candidate structure a per-candidate threshold rule discards

**Action** `a ∈ [-1, 1]`: a continuous residual added to the anchor's own priority score for that
candidate (not a discrete issue/defer/drop — too blunt for multi-candidate arbitrage; both HJS-L's
own reasoning against pure discrete actions and the earlier residual-RL sketch in this project
converged on a continuous adjustment independently, which is a point in its favor).

**Reward**: +1 per prefetch that becomes a hit; −λ_penalty · wasted_bytes per prefetch evicted before
use; 0 for a correct defer/drop. Budget is enforced structurally by the token-bucket dispatcher — a
hard constraint on what can physically be issued, never a soft reward term.

**Anchor.** `Q_θ(s,a) = V_anchor(s).detach() + f_θ(s,a)`, where `V_anchor(s)` is the *closed-form*
hazard-priced policy's value (HJS-L §2.3), computed once by exact replay and cached (no gradient
through it), and `f_θ` is a small residual network with its final layer zero-initialized. At
initialization, `f_θ ≡ 0`, so the dispatch order and every fetch/defer/drop decision the combined
system makes is **identical** to the closed-form policy's own decisions — this is a strictly stronger
anchor than "naive immediate fetch" (the anchor used in the original turn-1 RAO-RL sketch in this
project), because it means day-one performance already matches an already-designed, credible
baseline (HJS-L's own honest estimate: 30–60% of the corridor), not the weak all-fetch-immediately
bar. **Verify this exactly** — bit-for-bit dispatch match, or OHR-for-OHR match on a held-out replay —
before training a single gradient step. This is the same wiring-proof discipline (KT1) used
throughout this project; if the anchor doesn't reproduce, nothing downstream is trustworthy.

---

## 3. Training — offline, and an honest caveat the earlier sketch in this project glossed over

Every logged sweep configuration (hundreds of predictor × τ × k combinations, already recorded during
the Gate A/B measurement phase) constitutes a distinct behavior policy replayed over the identical
deterministic trace. That is the free dataset this project has been claiming since the first pass at
RAO-RL. Train `f_θ` by regressing the **residual advantage** — observed return-to-go minus the
anchor's cached value — never by softmax cross-entropy on an oracle argmax and never by exponentially
weighting raw oracle advantages (both distort under outcome noise; HJS-L's §2.4 already learned this
the hard way and it generalizes here).

**The caveat that needs stating plainly, not assumed away:** exact counterfactual replay from a state
under a *different* policy than the one that generated the trajectory prefix is only valid if the
chosen state is genuinely Markov-sufficient for the decision — i.e., if nothing about the candidate's
eventual outcome depends on history not captured in `s`. The original framing of this project's
free-dataset claim ("the env is deterministic replay, so the return from any state is exactly
computable") stated this too casually. If cross-candidate effects leak through channels the state
doesn't capture (e.g., two pending candidates' fates are coupled through a shared eviction event not
summarized in either one's own state), the "zero off-policy error" property degrades back toward
ordinary offline-RL distributional-shift risk. **This must be checked empirically** (e.g., by
comparing recomputed counterfactual returns against actually-replayed returns for a held-out set of
perturbed policies) before the "certified, exact" framing is used in the paper.

---

## 4. The necessity gate — build this first, before any network

Measure, directly by replay, with no learning involved:

> **F4 — RL-necessity gap** = (clairvoyant oracle optimizing *jointly* over a sliding lookahead
> window of concurrently pending candidates) − (clairvoyant oracle restricted to the closed-form
> policy's own per-candidate threshold structure, i.e., HJS-L's own diagnostic ceiling).

This number is the exact mathematical ceiling on how much *any* method — RL included — could add
beyond the closed-form anchor. It costs a day of replay work, no training.

- **Gap < 1–2 pts on all three live traces (wiki, cluster50, cluster53) → RL is provably close to
  unnecessary here.** Stop. The honest, publishable result is the measurement itself, alongside
  HJS-L's closed-form policy — not a trained model that has nothing real to add.
- **Gap ≥ 2 pts on any live trace → proceed to full training**, with that specific gap as the
  pre-registered target for what "success" looks like on that trace.

---

## 5. Evaluation

Capture fraction of the certified learnable corridor at iso-bandwidth, on the three live traces, with
95% bootstrap CIs via the same paired moving-block procedure already built (`oppcert/sim/bootstrap.py`) —
directly reusable, no new tooling required.

**Pre-registered pass bar:** RL must capture **≥50% of the measured F4 necessity gap** on all three
live traces, CI excluding zero. This is deliberately *not* a fraction of the whole 12–20pt corridor —
tying the bar to what RL could possibly add (rather than an arbitrary fraction of the total) is what
makes the claim reviewer-proof: it can't be dismissed as a cherry-picked threshold, because the
ceiling itself is measured, not assumed.

---

## 6. Prior-art delta

None of the systems surfaced in the Gate C lit check anchor an RL residual to an already-strong,
independently-designed closed-form teacher, or gate the decision to invest in RL training by a
measured, pre-registered necessity bound computed before training:

- **Baleen** gates admission by a learned probability of benefit against an offline OPT
  approximation — binary, episode-level, no anchoring, no necessity measurement.
- **DeePref** is online per-request RL with no shared-budget candidate stream and no anchor.
- **DEAP** and the joint-HW paper (arXiv 2510.10862) co-learn eviction and prefetch end-to-end —
  contradicted by this project's own Gate B finding that eviction and prefetch are structural
  substitutes, not complements.
- **Pythia** tunes a single global aggressiveness dial online, not per-candidate.
- **LightCacheRL** is a myopic bandit at hardware-line granularity, no long-horizon budget state.
- **Cold-RL** is eviction-only.

The delta is the composition: anchored (can't start worse than an already-strong baseline),
necessity-gated (scope measured before investment, not assumed), and trained fully offline from a
dataset that already exists at zero marginal collection cost.

---

## 7. Honest reachability

Because this RL targets a *residual* on top of a policy HJS-L already estimates captures 30–60% of
the learnable corridor, RL's own incremental contribution is bounded above by F4 — plausibly single
digits of additional OHR points on most traces, not the full 12–20pt corridor. The abstract must
claim exactly this: *"RL closes X% of the measured residual gap a closed-form policy provably
leaves,"* not *"RL captures the timing corridor."* The latter overclaims against this project's own
measurements and would not survive review from anyone who reads the methods section carefully.

---

## 8. Strongest reason this fails

**F4 comes back near zero.** This is not a remote possibility — it is HJS-L's own default prediction
(§1, row 1: "the problem decomposes per-candidate once bandwidth is priced"). If the closed-form
policy is already close to whatever a jointly-optimizing oracle could do within a lookahead window,
there is no real cross-candidate structure left for RL to exploit, and this entire proposal reduces
to a well-anchored, well-instrumented way of re-deriving that HJS-L was right the first time. That
outcome is still a publishable, honest result — a measured demonstration that RL is unnecessary here,
with the exact number to back it — but it is not the paper this proposal is written to produce.

Secondary risk: the Markov-sufficiency caveat in §3 fails in practice, degrading the "zero off-policy
error" claim back to standard offline-RL estimation risk, which would require falling back to
conventional conservative offline-RL techniques (pessimistic value regularization, behavior-policy
constraints) and would weaken the "certified, exact" framing that is this proposal's main
methodological distinction from ordinary offline-RL-for-systems work.

---

## 9. Build order (one to two weeks against existing infra)

1. **F4 necessity gap** (replay only, no training) — the gate. If it fails, stop here and report it.
2. Anchor wiring proof: verify `f_θ ≡ 0` reproduces the closed-form policy's decisions exactly
   (KT1-style check) on all three live traces.
3. Assemble the offline dataset from logged sweep configurations; verify the Markov-sufficiency
   caveat (§3) on a held-out sample of perturbed policies before trusting exact counterfactual
   returns.
4. Train `f_θ` via residual-advantage regression on the offline dataset.
5. Evaluate capture fraction of F4 against the ≥50% pre-registered bar, bootstrap CIs, all three live
   traces.
6. Ablations: anchor off (naive-immediate-fetch anchor, matching the original turn-1 RAO-RL sketch,
   as a comparison point), cross-candidate state features off (isolates whether RL is using the
   coupling information F4 says should exist), necessity-gate-predicted ceiling vs. achieved capture.
