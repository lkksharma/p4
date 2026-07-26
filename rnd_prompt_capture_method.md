# R&D PROMPT — Propose a Method to Capture a Measured Cache-Prefetch Timing Gap

> Hand this to a researcher or research model with no other context from this project. It is
> self-contained. It states only measured facts, fixed constraints, and existing prior art — it
> deliberately does **not** suggest a mechanism, a mathematical framework, or a solution family.
> Explore the full space yourself before converging on an answer.

---

## 0. The ask

A measurement study has certified that a specific, bounded resource-scheduling gap exists and is
real (not a measurement artifact) in several production-scale cache workloads. No method that
closes this gap has been proposed yet. **Propose one.** Explore broadly first — do not assume the
answer is reinforcement learning, or any other named technique — then converge on the single
method you judge most promising, and explain it precisely enough that someone could implement and
test it.

---

## 1. The problem, stated precisely and with nothing presupposed about the solution

A cache system has three decisions to make, and two of them are **already fixed by other
components you may not redesign**:

1. **What to evict when the cache is full** — handled by an existing eviction policy. Fixed.
2. **What object is likely to be requested soon** — handled by an existing predictor that outputs a
   stream of candidate objects, each with a confidence score, based on request history. Fixed.
3. **Given that stream of predicted candidates, a hard limit on how many bytes may be fetched
   per unit of request-time (the "bandwidth budget"), and the current state of the cache — which
   candidates actually get fetched, and when?** This decision is **open**. Nothing currently makes
   it in a principled way.

That third decision is the one you are being asked to solve. To be concrete about why it is
non-trivial: fetching a candidate the instant it is predicted is not obviously correct, because the
object may sit in the cache for a long time before it is actually needed and get evicted by
something else before its request ever arrives — the fetch is wasted, and the budget spent on it
could have gone to a candidate needed sooner. Conversely, waiting too long to fetch a candidate
means the fetch doesn't complete in time and the object is still a miss when requested. Multiple
candidates compete for the same limited budget concurrently, and the right choice for one candidate
depends on what else is currently in flight and how full the budget currently is.

You are free to define the shape of this decision however you want — continuous, discrete, myopic,
long-horizon, learned, closed-form, or something with no name yet. Nothing about the "state,"
"action," or "policy" framing above is prescribed; that language is only used to describe the
decision, not to imply it must be solved as a Markov decision process, a bandit, an auction, a
control system, or any other named formalism. Consider all of them, and anything else, on their
merits.

---

## 2. What has been measured (facts, not proposals)

A byte-accounted, deterministic cache replay simulator was used to measure, on each of 12 real
request traces (spanning enterprise block storage, a Wikipedia CDN trace, and Twitter key-value
traces, 2,000,000 requests each, cache sized to 1% of each trace's working-set footprint):

- **The tuned bar**: the best achievable hit rate using the *existing, fixed* predictor family
  (three orders of Markov table, tuned over its threshold and fanout hyperparameters, plus a
  trained LSTM next-item model) *combined with immediate, un-scheduled fetching* — i.e., every
  candidate the predictor emits is fetched the instant it is emitted, no scheduling at all — subject
  to a cap of at most 1.15× the traffic a no-prefetch baseline would use.
- **The gross ceiling**: the hit rate obtained by a clairvoyant oracle (exact future knowledge) that
  is only permitted to fetch at the *same total byte rate* the tuned bar spent (an iso-bandwidth
  comparison, so the ceiling cannot simply win by spending more).
- **The learnable ceiling** (the primary, more conservative measurement): the same clairvoyant
  oracle, but additionally restricted to only fetch objects that appeared at least once in the
  *training portion* of the trace — i.e., objects any history-based predictor could in principle
  have learned to anticipate. Objects that appear for the first time ever in the test portion
  (compulsory/cold misses) are excluded, because no predictor trained on history could have
  predicted them regardless of how the fetch is scheduled.
- **Learnable corridor** = learnable ceiling − tuned bar, in hit-rate percentage points. A trace is
  called "live" only if this corridor is ≥ 8 points **and** the ceiling does not achieve it by
  spending more bandwidth than the bar (Pareto dominance at equal or lower traffic). Both the
  threshold and the dominance requirement were fixed before any real-trace result was seen.

Result, with 95%-confidence intervals from a paired block-bootstrap (10,000 resamples, blocks of
contiguous requests to respect temporal autocorrelation):

| trace | tuned bar OHR | learnable ceiling OHR | learnable corridor (pts) | 95% CI |
|---|---|---|---|---|
| Twitter KV trace A | 0.5973 | 0.7940 | **+19.67** | [+19.41, +19.93] |
| Twitter KV trace B | 0.7080 | 0.8836 | **+17.56** | [+15.78, +19.39] |
| Wikipedia CDN trace | 0.5531 | 0.6803 | **+12.72** | [+10.93, +14.52] |

Three additional traces (one enterprise-block, two CDN) were measured and found to have a
learnable corridor below the 8-point threshold (range +3.6 to +4.9 pts) — on those traces the tuned
bar already captures nearly all of what a perfectly scheduled fetch could achieve, so there is
little left to win by better scheduling. Six further traces were also measured and killed for one
of three distinct reasons: (a) the predictor already saturates near-perfect precision, leaving no
headroom regardless of scheduling; (b) the apparent gap only existed because the ceiling spent
materially more bandwidth than the bar (a bandwidth purchase, not a scheduling win, caught by the
Pareto-dominance requirement); (c) one trace had a degenerate, near-fully-deterministic access
pattern where the simple predictor was already essentially optimal.

A separate measurement (interaction between the eviction policy and the prefetch policy, tested
across the same 6 live/near-live traces) found that improving eviction and improving prefetching
are **substitutes, not complements** — the marginal benefit of a better evictor shrinks as the
prefetcher improves, consistently and significantly, across all 6 traces tested. This means any
proposed method should treat the eviction policy as genuinely fixed and orthogonal — there is
measured evidence that jointly co-designing eviction and fetch-timing does not unlock additional
value beyond optimizing fetch-timing alone.

---

## 3. What infrastructure already exists (for prototyping, not as a hint about the method)

A working, tested simulation environment exists: deterministic trace replay; a byte-capacity cache
core supporting pluggable eviction policies (LRU, Belady, S3-FIFO) and pluggable predictors
(Markov-1/2/3, a trained LSTM); token-bucket-based bandwidth-rate accounting (i.e., the budget is
currently operationalized as bytes-per-request-of-replay, refilling continuously — this is simply
how the measurement above was taken, not a requirement that any proposed method must use the same
operationalization); an instrumented autopsy of wasted fetches (fetched-but-evicted-before-use vs.
fetched-and-never-requested); the cold/warm object-vocabulary split described above; and a
block-bootstrap confidence-interval tool. Every sweep run logged during measurement (hundreds of
configurations of predictor × threshold × fanout, each producing a full request-by-request replay
trace) is available as recorded data, if any candidate method would want to learn from logged
interaction data rather than only from live simulation.

---

## 4. What already exists in the published literature (facts only — draw your own conclusions)

The following systems and papers are the closest published work found by a lit search on this
topic. They are listed with only a factual description of what each one does. Determine for
yourself what, if anything, remains unaddressed.

- **Baleen** (FAST 2024) — a machine-learned flash cache admission and prefetching system. It uses
  a probability-of-benefit gate ("ML-When") to decide whether a group of related accesses
  ("episode") is worth admitting/prefetching at all, evaluated against an offline admission
  approximation of optimal ("OPT"), under a fixed flash-write-rate budget (drive-writes-per-day).
- **Demand-MIN** (ISCA 2018) — a variant of Belady's optimal-eviction algorithm, adapted for
  hardware CPU caches with prefetching, that minimizes *demand* misses (as opposed to total misses
  including prefetch traffic) rather than the standard Belady objective. Operates on uniform-size
  hardware cache lines.
- **DEAP Cache** (2020) — an end-to-end deep-learning pipeline that jointly learns eviction,
  admission, and prefetching for a single cache, using sequence models for prefetch prediction and
  online RL for the eviction-strategy mix.
- **DeePref** (2023) — an online deep-RL agent for CDN edge video prefetching. At each incoming user
  request, it chooses to either prefetch one specific video or take no action; it is not scheduling
  a stream of pre-generated candidates under a shared bandwidth budget.
- **Cold-RL** (2025) — an offline-RL-trained eviction policy deployed as an inference sidecar next
  to NGINX, selecting which of the coldest cached objects to evict. Falls back deterministically to
  LRU if inference fails or times out. Does not address prefetching or fetch timing.
- **LightCacheRL** (2025) — a lightweight online multi-armed-bandit formulation unifying prefetch
  and replacement decisions for hardware caches, optimized for low runtime/storage overhead.
- **A joint hardware caching+prefetching paper** (arXiv 2510.10862, 2025) — learns shared neural
  representations between a cache-replacement predictor and a prefetch predictor for CPU caches,
  trained via multi-agent RL, evaluated on a cycle-level hardware simulator.
- **Pythia** (MICRO 2021) — an online RL framework for tuning hardware prefetcher *aggressiveness*
  (a single global dial) using program context features and system-level bandwidth feedback as
  reward signal.
- **"A Sober Look at Progress in Language Model Reasoning"** (arXiv 2504.07086, 2025) — not a
  caching paper. Demonstrates, in the language-model-reasoning field, that pre-registered,
  seed-averaged evaluation protocols reveal that much of a field's reported progress does not
  survive rigorous re-evaluation. Included because it is methodologically the closest precedent
  found for the *measurement discipline* used in section 2 above (pre-registration, corridor
  deflation) — it is not caching-related and says nothing about scheduling mechanisms.

---

## 5. The evaluation contract any proposed method will be judged against

Whatever method is proposed, if it is tested, it will be tested against this standard — stated here
so the proposal can be designed with it in mind, not because it constrains what the method itself
should look like:

1. **The metric is capture fraction of the certified learnable corridor** — i.e., of the 12–20-point
   gap between the tuned bar and the learnable ceiling (section 2), what percentage does the
   proposed method actually close, on held-out replay of a live trace, at bandwidth no greater than
   the tuned bar used? A method that merely raises the hit rate by spending more bandwidth does not
   count — it must clear the same iso-bandwidth / Pareto-dominance bar the ceiling itself was held
   to.
2. **No result on a single trace generalizes.** Any claimed capture must be shown on at least the
   three live traces above (ideally more), and the method should state what its performance would
   likely depend on (workload properties, prediction confidence calibration, budget tightness,
   etc.) rather than reporting one number.
3. **State a falsifier.** Before testing, state the smallest experiment that would show the method
   does *not* work, and the specific pre-registered threshold that would constitute failure — the
   same discipline used to establish the +8-point / Pareto-dominance rule for the measurement
   itself.
4. **Name the nearest prior art from section 4 and state the delta.** If the proposed method turns
   out to be substantially similar to something already in section 4, say so explicitly rather than
   presenting it as more novel than it is.
5. **Distinguish what the method achieves from what is achievable.** The learnable ceiling in
   section 2 is itself an upper bound assuming perfect foresight; no realistic method restricted to
   causal information (only the request history observed so far) should be expected to reach 100%
   capture. State plainly what fraction of the corridor you believe is realistically reachable and
   why, separate from what fraction is mathematically possible in the limit.

---

## 6. Explicit instruction on how to approach this

Do not assume the solution is reinforcement learning, a Markov-decision-process formulation, an
auction/pricing mechanism, a discrete or continuous action space over candidates, a queueing model,
a statistical/survival model, or any other named technique — none of these has been ruled in or out.
Explore the full space of plausible approaches on their own merits, including approaches outside
machine learning entirely if they fit. Consider multiple candidate directions before you commit,
evaluate each one against section 5's evaluation contract and section 4's prior art, and then
converge on the single method you judge to be the most promising combination of **novel** (not
already covered by section 4), **implementable** against the infrastructure in section 3, and
**capable of a real test with a real falsifier** in section 5's terms. Present that one method in
enough mechanistic detail that someone could build a first prototype of it, along with your honest
assessment of the strongest reason it might fail.
