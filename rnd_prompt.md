# R&D PROMPT — Revolutionizing Caching / Paging: the Tempo Layer

> Hand this to a strong research model or collaborator. It is self-contained: it carries
> everything we have already established, then asks for the breakthrough directions that lie
> *along our path*. The house rules at the bottom are non-negotiable — proposals that violate
> them (mechanism-before-measurement, n=1 generalization, no falsifier) are worthless to us.

---

## 0. The ask, in one sentence

We have **measured** a control problem that sixty years of caching research never named, and
certified it is worth 20–33 hit-rate points at fixed bandwidth on real traces. **Tell us how to
revolutionize caching/paging by owning that layer** — the mechanism, the theory, the highest-value
domains, and the sharpest way each claim can die.

---

## 1. The reframe our own results forced (read this first — it is the whole thesis)

There are three jobs in a cache. Two are closed problems; the third is open and unowned.

1. **WHAT TO KEEP** — eviction. *Closed.* Belady in theory; ARC/TinyLFU/S3-FIFO in practice. Our
   gates confirm S3-FIFO with no prefetch is brutal to beat.
2. **WHAT'S COMING** — prediction. *Effectively closed.* Our sweeps show a plain counting table
   (Markov-2/3) hits **0.87–0.91 precision**. Prediction is commoditized; the field keeps polishing
   it because it's the hammer everyone owns.
3. **WHEN THE BYTES MOVE** — scheduling. **Open. Unowned. Measured by us.**

The sentence the whole program stands on:

> A cache with a near-perfect evictor **and** a near-perfect predictor still leaves **20–33
> hit-rate points on the table — at identical bandwidth** — purely because nobody schedules *when*
> a fixed byte-budget gets spent.

That is not eviction's job (an evictor cannot un-waste a prefetch fired too early) and not
prediction's job (a predictor is blind to budget and cache pressure). It is a **scheduling MDP**:

> Given a fixed predictor's candidate stream, a hard bandwidth budget, and current cache pressure —
> **which candidates get tokens, and when** (issue-now / defer / drop)?

We call this the **tempo layer** of the memory hierarchy. It is predictor-agnostic and
evictor-agnostic, which is why it is portable to every tiered-memory system that exists.

---

## 2. What we have already done (the ground truth you must build on)

### 2a. Killed the wrong thesis, honestly
- Original claim was a **joint** RL policy co-optimizing eviction + prefetching. **Dead**, killed by
  our own arithmetic and endorsed by a peer:
  - **Interaction is negative & monotone.** Eviction gain shrinks as prefetch improves
    (`+7.13 → +6.22 → +5.78`). Eviction and prefetching fix the **same misses** → **substitutes, not
    complements** → a joint policy has no coupling to exploit.
  - **The oracle refuses the "retained prefetch" prize.** Belady converts *fewer* prefetches to hits
    than S3-FIFO, wastes more, and still wins — retaining "correct-but-evicted" prefetches is a
    mistake the optimum declines to make, not unrealized headroom.
  - Retracted our own +2.96 pt "jointness prize" as a small-n / wrong-label artifact (median
    distance-to-use was 116k requests into a ~2,600-object cache). **Prize ≈ 0.**
- This is the fifth project we have killed on purpose (CAPO, CLCA, Certifend, RAO-RL joint-cache
  flagship, P4-MSR). **Killing our own results is our credibility.**

### 2b. Built the instrument (Ring 1 — "the Gauge")
A pre-registered, **iso-bandwidth**, adversarially-tuned measurement protocol that tells any
operator, on their own trace, whether timing headroom exists and how much is learnable in principle.
- **Corridor** = (iso-BW Prescient ceiling) − (best *decoupled* tuned bar), where the bar is
  `best OHR at ≤1.15× traffic over {Markov-1,2,3} × τ × k`, plus a fine-τ sweep, plus an **LSTM**
  next-item bar. Adversarially tuned *for the baseline* so we can't fool ourselves.
- **Decision rule, fixed before results:** corridor **≥ 8 OHR pts AND ceiling Pareto-dominates**
  (better OHR at no more traffic) → timing is live. Otherwise the project is dead and we say so.

### 2c. Ran it at breadth and found structure, not a single win
**12 traces, 3+ workload families → 6 LIVE / 6 DEAD.** Liveness is a **trace property**
(Markov-saturation vs. prediction-limited reuse), *not* a family label. Selected verdicts:
- **LIVE:** wiki +26.5, meta_rprn +32.6, meta_reag +24.6, cluster53 +24.5, cluster50 +20.4,
  msr_hm_0 +10.1, cluster26 +12.5 (cap-sensitive).
- **DEAD:** msr_proj_0 +4.2 (robust across BW caps), msr_web_2 −3.8, w105 −0.5,
  cluster10 −1.2 (degenerate), w87 +32.7 but Pareto-fails by 0.01× (boundary case).
- **Day-1 LSTM kill-test: PASSED 3/3** — wiki/cluster50/meta_rprn stay LIVE with the bar =
  `max{Markov-1,2,3, LSTM} × fine-τ`. The LSTM never took the bar.

### 2d. Discovered three distinct kill mechanisms (the instrument's teeth)
The Gauge doesn't just say live/dead — it says **why** it died:
1. **Markov-saturation** — the counting table already captures the reuse (msr_proj_0).
2. **Traffic-bought corridor** — the apparent gap is a bandwidth difference, not a timing prize;
   caught by the Pareto test (cluster26 flipped LIVE→marginal when a stronger bar spent 1.16× vs
   1.09×).
3. **Cold-dominated** — the ceiling's advantage comes from prefetching **never-seen** objects that
   **no history-based predictor can reach**. We instrumented a **cold-split** (`pf_cold_hits`) so the
   reportable, learnable corridor = **gross − cold slice**. (Numbers pending; this gates final claim
   wording.)

### 2e. Designed the mechanism, not yet built (Ring 2 — "the Conductor")
**Budgeted Prefetch Scheduling via Relaxation-Anchored Offline RL (RAO-RL).**
- **State:** per-candidate (confidence, predicted time-to-access, size, cache-residency), budget
  (token level + accrual rate), cache pressure (victim quality, bytes pinned by pending prefetches).
- **Action per candidate:** {issue-now, defer, drop} (+ optional issue+protect/pin).
- **Reward:** +1 per prefetch-hit, −λ per wasted byte; **budget is a hard env constraint, not reward
  shaping.**
- **The anchor (the methodological crown jewel):** replay is deterministic, so the tuned bar's return
  is *exactly computable* from any state. Learn `Q_θ(s,a) = V_bar(s).detach() + f_θ(s,a)` with a
  zero-init residual → **at initialization the agent IS the tuned bar.** "Will RL even beat the
  baseline?" becomes true *by construction*, not by gamble.
- **Free offline dataset:** every sweep config we logged is a distinct behavior policy over the same
  deterministic replay. **Our evaluation protocol produces our training data as a byproduct.**
- **Evaluation standard no learned cache system has faced:** report **capture fraction of the
  certified learnable corridor** (gross − cold) at iso-bandwidth against the strong bar — not delta
  over LRU, not "Belady-gap closure."
- **Pre-registered kill-tests:** KT0 (a dumb JIT-defer heuristic — if it captures ≥⅔ of the corridor,
  RL is unmotivated, stop before building), KT1 (anchor reproduces the bar's OHR at init — wiring
  proof), KT2 (RAO-IQL beats the strong bar on wiki at iso-BW, capturing a pre-registered ≥⅓ of the
  learnable corridor — else the RL paper is dead and the measurement paper stands alone).

### 2f. Where the prior art stands (as of our lit sweep)
Nearest neighbors and the gap we take: Joint HW Caching+Prefetching (NeurIPS'25 wksp — representation
level, HW, uniform lines); Pythia (online RL, **one global aggressiveness knob**, HW); LightCacheRL
(bandit, **myopic**, no long-horizon timing value); Cold-RL (offline RL **eviction only**); DEAP
(supervised joint, 2020, never beat its oracle); Residual RL / Policy Decorator (**online robotics**;
residual on *actions*, not anchored to a computable relaxation *return*); ReCache (diffusion-inference
scheduling — **keyword collision only**). **No one has formulated per-candidate, token-bucket,
variable-size prefetch scheduling for object caches, offline, anchored to a certified ceiling.**

---

## 3. The research questions (this is what we want back)

Answer along **our path** — the tempo/scheduling layer — not a fresh detour into eviction or
prediction. For every direction, we care about *measured headroom before mechanism* and *how it
dies.*

**Q1 — Where else does the corridor live, and how big?**
Rank tiered-memory domains by (bandwidth price × plausibility that a timing corridor exists):
OS/page-cache readahead, mobile app prewarm, edge/CDN egress, and the 2026 jackpot — **LLM serving**
(KV-cache + weight paging across GPU/CPU/NVMe is *literally paging*, near-deterministic access,
the most expensive bandwidth on earth). For each: what is the analog of a "trace," could we run the
Gauge on it, and what would kill the corridor there? We deploy into a domain **only if the Gauge
finds a corridor** — propose the cheapest experiment that answers yes/no per domain.

**Q2 — What is the strongest form of the mechanism?**
Push RAO-RL past the current sketch. Is per-candidate {issue/defer/drop} the right action space, or
is there a better formalization (e.g., a scheduling policy over a *priority queue* of live candidates,
continuous issue-time regression, a restless-bandit or queueing-theoretic framing)? Where does the
anchor idea generalize — can we anchor to a *family* of relaxations (multiple τ,k bars) rather than
one? What breaks the "agent-starts-as-baseline" guarantee under stochastic (non-replay) bandwidth?

**Q3 — Is there a theorem here?**
cluster26 already handed us a theorem-shaped fact: **even clairvoyant greed is suboptimal under a
hard byte-budget** (it bought hits with traffic). Formalize budgeted, variable-size prefetch
scheduling. Is it NP-hard? Is there a competitive-ratio result for an online tempo scheduler vs. the
budgeted-clairvoyant optimum? What is the right offline-optimal oracle to measure capture against
when object sizes vary and Belady is only a heuristic?

**Q4 — Where does the whole program die?**
Be adversarial. The sharpest failure modes we can name: (a) the **cold slice eats the corridor**
(the reachable headroom is mostly never-seen objects) — how likely across domains, and what's the
strongest history-plus-content predictor that narrows it without becoming "a better predictor" (Job
2, not our layer)? (b) a **dumb JIT-defer heuristic captures most of it** (KT0) — what cross-candidate
arbitrage genuinely requires learning vs. a rule? (c) real bandwidth is **stochastic/bursty**, not a
clean token bucket — does the corridor survive? Name the experiment that would falsify each.

**Q5 — The 18-month map.**
Given Ring 1 (Gauge, shipping) + Ring 2 (Conductor, conditional on KT2) + Ring 3 (the regime
taxonomy as a decision procedure for the hierarchy) — what is the tightest publishable sequence, and
what is the single highest-leverage experiment we are **not** currently planning?

---

## 4. House rules (a proposal that breaks any of these is discarded)

1. **Measurement before mechanism.** Every "revolution" must start from headroom that is measured or
   cheaply measurable, not asserted. We already made the mistake of building mechanisms and praying
   for headroom — five times.
2. **A falsifier bolted to every claim.** State the pre-registered threshold and the experiment that
   kills the idea *before* running it. Ideas without kill-tests are not proposals.
3. **No n=1 generalization.** One trace is never "structural." Say what n is and what the variance is.
4. **Iso-bandwidth or it doesn't count.** Any hit-rate gain must be shown at no extra traffic
   (Pareto), or it's a bandwidth purchase wearing a timing costume.
5. **Cite the neighbor, name the gap.** Especially Pythia, LightCacheRL, Cold-RL, Residual-RL,
   Voyager/Glider. If someone already did it, say so and reframe or stop.
6. **Distinguish reachable from ideal.** Prescient ceilings and cold slices are upper bounds. Learnable
   ≠ ideal. Report capture of the *certified learnable* corridor, never the gross gap.

---

## 5. Output format we want back

For each direction (Q1–Q5): **the claim**, **the smallest experiment that supports or kills it**,
**the pre-registered pass/fail number**, **the nearest prior art + the gap**, and **the single most
likely reason it dies.** Rank the directions by (leverage ÷ cost-to-first-signal). End with the one
experiment you would run next week if you had our box and our traces.
