# SPEC — JOB / cardinality estimation: Opportunity Certification in a fifth domain

**Pre-registered 2026-07-26, before any query is run against IMDB.** Every threshold, arm, and fork
below is fixed now. No PostgreSQL instance and no IMDB data exist on the authoring machine, so
nothing in this document was written with knowledge of its own outcome. Numbers quoted from prior
work are cited as such.

---

## 0. Why this domain, and what makes it different from the other four

The paper's four existing settings all end in the same *kind* of verdict: a gap that is absent,
clairvoyance-priced, or captured. A fifth setting is worth its pages only if it exhibits a **failure
mode the caching half cannot produce**, and query optimization does:

| | caching prefetch | learned indexes | **query optimization** |
|---|---|---|---|
| oracle needs | the FUTURE | the DATA | the DATA |
| reachable? | no → trap | yes → build | **yes, and the field did capture it** |
| what failed | placement (endogenous slack) | nothing (build) | **the captured quantity did not convert to the objective** |

Learned cardinality estimators **succeeded at their stated task**: \citet{wang2021ready} report that
learned methods are more accurate than traditional ones. The gap was real, it was reachable, and it
was captured. What did not follow was the end-to-end win, because accuracy is a *proxy* and latency
is the objective. That is a fourth certificate the paper does not currently have a name for, and it
is the single most common failure in ML-for-systems: **optimising a reachable proxy that does not
map to the objective.**

The retrospective value is unusually clean. \citet{leis2015good} measured the oracle gap in 2015 by
injecting true cardinalities into PostgreSQL and reported that "true cardinalities do not
significantly affect runtimes for most queries" while "a substantial fraction benefit dramatically."
The number that bounds the opportunity was therefore published *in the paper that motivated the
chase*, six years before \citet{wang2021ready} reported the chase had not paid off, and a decade
before \citet{leis2025still} asked the same question again. Stage 2 is precisely the step that reads
such a number before committing.

---

## 1. The mapping

| Instrument component | caching | this domain |
|---|---|---|
| objective metric | request-weighted hit ratio | **end-to-end query latency**, including estimation time |
| BASE | S3-FIFO, no prefetch | PostgreSQL at defaults, no extended statistics |
| TUNED BASELINE | best of {Markov-1/2/3, LSTM} × τ × k under 1.15× traffic | best of the swept non-learned configuration family (§2) |
| GROSS CEILING | clairvoyant over all objects | **true cardinalities injected everywhere** (Leis's construction) |
| REACHABLE CEILING | clairvoyant restricted to training vocabulary | **true base-table cardinalities only**, DBMS join estimation on top (§3) |
| iso-resource (M2) | prefetch byte rate | **planning + estimation time charged to every arm** (§4) |
| Pareto requirement | ceiling must not use more traffic | ceiling must not use more planning time |

---

## 2. The tuned baseline: the M1 discipline, which is the whole point

Most learned-cardinality work compares against PostgreSQL at or near defaults. That is the same
error the index section found: **binary search is not the tuned baseline, and neither is default
PostgreSQL.** The baseline is the best of this swept family, chosen per query set:

1. `default_statistics_target` ∈ {100 (default), 1000, 10000} — histogram granularity.
2. **Extended statistics** (`CREATE STATISTICS ... (ndistinct, dependencies, mcv)`) on the correlated
   column groups appearing in JOB predicates, present in PostgreSQL since v10. **This is the direct
   analogue of interpolation search**: it captures the multi-column correlation signal that learned
   estimators are built to model, with no model, no training, and no inference cost.
3. Exhaustive enumeration forced (`geqo = off`, `join_collapse_limit` and `from_collapse_limit`
   raised past the largest query's relation count), since \citet{leis2015good} found exhaustive
   enumeration helps even under bad estimates.
4. A modern server version, **not** the 9.4 used in 2015. Any part of the 2015 gap that a decade of
   non-learned estimator work has already closed is M1 operating across the literature, and
   measuring it is a result in itself.

**Pre-registered prediction (falsifiable, recorded before running):** the tuned baseline will beat
default PostgreSQL by a non-trivial margin, and the gap to the true-cardinality oracle measured
against the *tuned* baseline will be materially smaller than the gap measured against defaults. If
the two are within noise, this prediction is wrong and the M1 mechanism does not operate here.

---

## 3. The reachability split, and why base-table versus join is the right cut

A cardinality is a deterministic function of the data, so it is reachable in the abstract. But a
bounded statistical summary cannot represent arbitrary multi-way join-key correlation, and error
compounds multiplicatively along a join path. The split therefore is:

- **REACHABLE CEILING** — inject true cardinalities for **single-table selections only**; the DBMS
  estimates joins from those corrected inputs by its usual rules. A per-table learned model *can*
  in principle attain this, so it is the honest upper bound for that model class.
- **GROSS CEILING** — inject true cardinalities for **every subquery, joins included**. This credits
  multi-way correlation knowledge that no per-table summary holds, and is reported as a diagnostic
  only, exactly as the gross clairvoyant ceiling is in the caching half.

The difference between them is this domain's **phantom slice**, and it is the quantity the paper's
central mechanism (M4, cold conflation) predicts will be large. A corridor that lives mostly in the
gross-minus-reachable gap is one no per-table estimator can reach, however accurate it becomes.

**Invariant (machine-checked, aborts the run on violation):** injecting true cardinalities must never
*increase* the optimizer's estimate error for the injected nodes, and the gross ceiling must weakly
dominate the reachable ceiling on total latency. A violation means the injection is wired wrong.

---

## 4. Iso-resource: charge estimation time

Caching charges prefetch bytes; here the scarce resource is **planning time**. Learned estimators
pay per-subquery neural inference, which \citet{wang2021ready} identify as a principal obstacle, and
which is routinely excluded from reported plan latencies. Every arm is therefore measured on
**total time = planning + estimation + execution**, and the oracle arms are additionally reported
with their cardinality lookup charged at the cost a deployed model would pay.

Because true cardinalities are precomputed offline, the oracle is *free* at query time and so is
strictly optimistic. That asymmetry is stated, not corrected: it means the measured corridor is an
**upper bound** on what any estimator could deliver, which is the direction that makes a negative
verdict conservative.

---

## 5. The pre-registered bar

Two conditions, both fixed now, mirroring the caching gate's conjunctive form (corridor ≥ 8 points
**and** Pareto dominance).

**(a) Magnitude.** The reachable corridor must reduce total workload latency by **≥ 20%** relative
to the tuned baseline, with a bootstrap confidence interval whose upper bound also clears 20%
(1,000 resamples over queries, since queries are the independent unit here, unlike autocorrelated
cache requests).

**(b) It must exceed configuration noise.** The reachable corridor must be **strictly larger than
the span of total workload latency across the swept non-learned configurations of §2**. If moving
`default_statistics_target` or adding extended statistics shifts latency more than perfect
base-table cardinalities do, the corridor sits inside the space a competent DBA already reaches and
no learned component is warranted. This is the domain's Pareto-dominance analogue and the condition
the caching half would have applied to itself.

**(c) Reported, not gated: concentration.** Per-query improvement distribution, and the fraction of
queries carrying the aggregate gain. Pre-registered as a reporting requirement because
\citet{leis2015good} indicate the gain is concentrated, and a concentrated gain implies a different
engineering response (re-optimisation or a fallback for the affected queries) than a broad one
(a better estimator for all queries). A mean alone would hide this.

---

## 6. The fork, fixed before running

- **Reachable corridor clears both (a) and (b)** → *build*, for a per-table cardinality model. The
  paper reports a second build alongside learned indexes, and the retrospective claim weakens to
  "the field chased a real and reachable corridor, and the obstacle was deployment cost, not
  headroom." Honest and still publishable, but it is not the finding this section is for.
- **Corridor real but lives mostly in gross-minus-reachable** → the corridor requires multi-way join
  correlation no per-table summary holds. Certificate: *trap*, by the same reachability logic as
  caching but for a different reason (hypothesis-class capacity rather than the future).
- **Corridor fails (a) or (b) against the tuned baseline** → **the primary hypothesis**: the
  opportunity is inside the span a tuned non-learned configuration already covers. Certificate:
  *no corridor*, mechanism M1, and the retrospective claim is at its strongest.
- **Corridor clears but the concentration is extreme** (a small query minority carries it) → a fourth
  certificate the paper should name: *real, reachable, and captured, but not convertible in
  aggregate*. This is the outcome most consistent with \citet{leis2015good} and \citet{wang2021ready}
  read together, and the one that would justify the section's existence most strongly.

**No outcome is a failure of the section.** Each maps to a distinct certificate, which is the
property that makes this a procedure rather than a prior.

---

## 7. Retrospective claim: the constraints it must respect

The counterfactual ("this procedure would have redirected the effort") is the persuasive part and
the easiest to overstate. Three rules:

1. **Only pre-2016 information.** The counterfactual may use only what \citet{leis2015good} published
   and what PostgreSQL 9.4 offered. Any use of a later result is hindsight, not prediction.
2. **Claim redirection, never refutation.** The learned estimators worked at their stated task. The
   claim is that Stage 2 would have identified the *conversion* question earlier, not that the work
   was wrong.
3. **State what the procedure would have missed.** Applied in 2015, the procedure would have measured
   the corridor and its concentration, and it would *not* have predicted the inference-cost obstacle
   \citet{wang2021ready} found, because Stage 1 as instantiated in the caching half does not price
   model serving cost. Naming that limit is what keeps the retrospective honest, and §4 exists to
   partially repair it.

---

## 8. Build plan and cost

| step | work | est. |
|---|---|---|
| 1 | IMDB load, modern PostgreSQL, ANALYZE; verify all 113 JOB queries run | 0.5 d |
| 2 | Baseline sweep of §2; record total workload latency per configuration | 1 d |
| 3 | True cardinality computation for the subquery lattice (restrict to connected join subgraphs; cap relation count if the lattice explodes, and **report the cap** rather than silently truncating) | 1–2 d |
| 4 | Injection via `pg_hint_plan` `Rows()` hints; gross and reachable ceilings; invariants of §3 | 1 d |
| 5 | Bootstrap, concentration analysis, fork evaluation, write-up | 1 d |

**Total ≈ 1 week wall-clock.** Dependencies: PostgreSQL ≥ 16, `pg_hint_plan`, the public IMDB
snapshot and JOB query set from the \citet{leis2015good} artifact. None exist on the authoring
machine; all steps run on the DGX node.

**Scope guards.** Query latency is machine- and cache-state dependent, so: fixed hardware, warm
buffer cache via a pre-run pass, three repetitions per query with the median taken, and a documented
timeout with timed-out queries reported separately rather than dropped. Any query that fails to
complete under a configuration is a result about that configuration, not missing data.

---

## 9. Citations to add

`leis2015good` (VLDB 2015, *How Good Are Query Optimizers, Really?*), `wang2021ready` (VLDB 2021,
*Are We Ready For Learned Cardinality Estimation?*, Best EA&B Paper), `leis2025still` (VLDB 2025,
*Still Asking: How Good Are Query Optimizers, Really?*). Author lists and venues to be verified
against the published records before camera-ready, per the project's standing citation rule.

---

## 10. Amendment, 2026-07-26: the reachability split, changed before any measurement

**What changed.** Section 3 pre-registered the reachability split as **base-table versus join**:
inject true cardinalities for single-table selections (reachable), against true cardinalities
everywhere (gross). It is now **pairwise versus n-way**: inject true cardinalities for 2-way joins
(reachable), against joins up to 4 relations (gross).

**Why, and when.** `pg_hint_plan`'s `Rows()` hint corrects "row number of a result of the joins on
the tables specified" and is silently ignored on a single relation. The base-table arm is therefore
not expressible with this instrument at all. This was discovered by the hint-efficacy invariant
(§11) failing on the target machine **before a single latency measurement was taken**, and no
corridor number existed at the time of the change. The amendment is forced by tooling, not chosen
after seeing a result.

**Why the substitute tests the same question.** The original split asked whether a *bounded*
statistical summary can reach the corridor, or whether it needs correlation structure no such
summary holds. Pairwise versus n-way asks exactly that, and arguably asks it more directly:
\citet{leis2015good} identify multiplicative error compounding along join paths as the mechanism,
and pairwise-versus-higher-order is precisely where that compounding begins. Base-table estimates
are, in that same work, reported as comparatively accurate, so the base/join cut was the weaker of
the two candidate boundaries in any case. Both arms now share identical base-table estimates, which
makes the base-table contribution a controlled constant rather than a confound.

**What is unchanged.** The bar (§5), the fork (§6), the retrospective constraints (§7) and the
iso-resource discipline (§4) are untouched. The corridor is still reachable-ceiling versus tuned
baseline on total time, and 20% plus the configuration-span condition still decide it.

## 11. The hint-efficacy invariant, and why it exists

`pg_hint_plan` does not error when it is inactive: it ignores hint comments. An unloaded library
would leave both ceiling arms executing the baseline plan, both corridors would read approximately
zero, and the harness would print **NO CORRIDOR** with every appearance of a measurement. That is
the outcome most favourable to this section's retrospective claim, which makes it exactly the
failure that would never have been questioned.

`enable_hints()` therefore proves efficacy before anything is measured: it issues a deliberately
absurd `Rows(x y #4242)` hint against a self-join and reads the optimizer's estimate back. If the
estimate does not move to 4242, the run **aborts**. The probe must be a join, for the same reason
the base-table arm had to be abandoned.

On first execution against the loaded IMDB database this invariant reported `INERT` and stopped the
run, which is how the `Rows()` limitation was found. It has already paid for itself.
