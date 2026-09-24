# P4 Results Ledger

All numbers produced by the pre-registered Gate A protocol (`oppcert/instrument/gates.py` / `oppcert/instrument/bars.py`):
**bar** = best decoupled predictor (Markov-1/2[/3] × τ × k) under a 1.15× origin-traffic cap;
**ceiling** = S3-FIFO + Prescient (clairvoyant greedy) at the bar's prefetch byte rate (iso-BW);
**corridor** = ceiling − bar, in OHR points; **LIVE** requires corridor ≥ 8.00 AND Pareto dominance
(pre-registered before any real-trace run). 2M-request prefixes, 1% cache, warmup 5% excluded.

## Gate A breadth sweep — 2026-07-18 (`logs/breadth_summary.txt`)

| trace | family | base OHR | tuned bar (config) | bar OHR @traffic | ceiling @traffic | corridor | verdict |
|---|---|---|---|---|---|---|---|
| msr_proj_0 | block (MSR) | 0.5516 | markov2 τ=.05 k=2 | 0.7568 @1.13× | 0.7984 @1.09× | **+4.16** | DEAD |
| msr_hm_0 | block (MSR) | 0.6570 | markov2 τ=.05 k=1 | 0.8380 @1.13× | 0.9388 @1.05× | **+10.08** | LIVE |
| msr_web_2 | block (MSR) | 0.0095 | markov2 τ=.05 k=4 | 0.8284 @1.00× | 0.7903 @1.00× | **−3.81** | DEAD |
| w87 | block (CloudPhysics) | 0.0949 | markov2 τ=.05 k=1 | 0.5383 @1.12× | 0.8650 @1.13× | +32.67 | DEAD (Pareto)* |
| w105 | block (CloudPhysics) | 0.2694 | markov2 τ=.05 k=4 | 0.8494 @1.05× | 0.8449 @1.31× | **−0.45** | DEAD |
| wiki_2019t | CDN (Wikipedia) | 0.1522 | markov2 τ=.05 k=1 | 0.5531 @1.09× | 0.8181 @1.01× | **+26.50** | LIVE |
| meta_reag | CDN (Meta) | 0.5383 | markov2 τ=.05 k=1 | 0.7543 @1.15× | 0.9998 @0.71× | **+24.55** | LIVE |
| meta_rprn | CDN (Meta) | 0.3898 | markov2 τ=.05 k=1 | 0.6736 @1.06× | 0.9996 @0.66× | **+32.59** | LIVE |
| cluster10 | KV (Twitter) | 0.5000 | markov1 τ=.05 k=1 | 0.7368 @1.00× | 0.7250 @1.00× | **−1.18** | DEAD† |
| cluster26 | KV (Twitter) | 0.7864 | markov3 τ=.05 k=2 | 0.8907 @1.09× | 0.9935 @1.16× | +10.27 | **DEAD (strong bar)**‡ |
| cluster50 | KV (Twitter) | 0.4164 | markov2 τ=.05 k=1 | 0.7078 @1.02× | 0.9120 @0.99× | **+20.42** | LIVE |
| cluster53 | KV (Twitter) | 0.5951 | markov2 τ=.20 k=1 | 0.5973 @1.01× | 0.8419 @1.00× | **+24.46** | LIVE |

**Tally: 6 LIVE / 6 DEAD across 12 traces after strong-bar check #2 (cluster26 flipped),
with live and dead points in every family.**
LIVE: wiki_2019t, meta_reag, meta_rprn (CDN) · cluster50, cluster53 (KV) · msr_hm_0 (block).

\* w87: corridor +32.67 but ceiling traffic 1.13× vs bar 1.12× — Pareto dominance fails by 0.01×
(hairline, likely display-resolution). Pre-registered rule says DEAD and we keep it, but this is a
boundary case: byte-weighted view pending. Do not cite w87 in either direction without the follow-up.

† cluster10 is degenerate: 1,000,026 objects / 2M requests, ~22-byte objects, base OHR exactly
0.5000, Markov precision 1.000 at 1.00× — a near-deterministic each-object-twice pattern where
Markov is already perfect (bar *exceeds* the greedy clairvoyant ceiling). Valid dead point;
mechanism obvious.

‡ cluster26: the provisional LIVE (+12.48 on the coarse Markov-1/2 grid) did NOT survive the
pre-registered fine-τ + Markov-3 check (row shows the strong-bar result). Two things flipped it:
(i) the fine grid found the cap-edge configs the coarse grid missed (markov2 τ=.06 k=1 =
0.8905 @1.14×), and (ii) Markov-3's triple-contexts genuinely helped here (unlike wiki),
raising the bar 5 pts to 0.8907. Corridor +10.27 still ≥8, but the CEILING needed 1.16× vs the
bar's 1.09× — Pareto dominance fails decisively (not a hairline like w87): even clairvoyance
buys the remaining hits with extra traffic on this trace. Third distinct kill mechanism
demonstrated by the gate (after bar-saturation and degenerate predictability):
**traffic-bought corridor**.

## Strong-bar check #1 — Markov-3, wiki (2026-07-18, `logs/strongbar_m3_wiki.log`)

Bar = max over {markov1, markov2, markov3} × τ(0.01…0.5) × k(1,2,4). Result: **Markov-3 adds
nothing** (backoff makes it ≈ Markov-2 row-for-row; best config identical at 0.5531 @1.09×).
Extended low-τ grid also changes nothing — every low-τ config blows the traffic cap first.

> TUNED A1 BAR markov3 τ=0.05 k=1 OHR 0.5531 @1.09× · CEILING 0.8180 @1.01× ·
> **CORRIDOR +26.49 · LIVE.** Higher-order Markov does not eat the wiki corridor.

## Strong-bar check #2 — Markov-3 + fine τ, cluster26 (2026-07-18, `logs/strongbar_m3_cluster26.log`)

**Provisional LIVE overturned.** Bar rose 0.8409 → 0.8907 (markov3 τ=.05 k=2 @1.09×; Markov-3's
triple-contexts genuinely disambiguate on this trace, and the fine grid found the cap-edge
markov2 τ=.06 config the coarse grid missed). Corridor +10.27 still clears 8, but Pareto fails
decisively: ceiling 0.9935 needs **1.16×** vs bar's 1.09× — even clairvoyant timing buys the
remaining hits with traffic here. Pre-registered conjunctive rule (corridor ≥8 AND dominance):

> **VERDICT: DEAD.** cluster26 joins the dead column; KV family stays live via cluster50 (+20.42)
> and cluster53 (+24.46). Consequences applied: STRONG_TAUS now permanently includes 0.06–0.09,
> and every remaining LIVE verdict is provisional until it passes the same Markov-3 + fine-τ +
> LSTM bar (Day 1 runs all three on wiki, cluster50, meta_rprn).

## Strong-bar check #3 — LSTM + Markov-3 + fine τ, three live traces (2026-07-19, `logs/day1_nohup.log`)

**The pre-registered aliveness test. All three PASSED — the corridor is not a weak-predictor
artifact.** Bar = max over {markov1/2/3, LSTM} × τ(0.01–0.5, incl. 0.06–0.09) × k(1,2,4):

| trace | winning bar | bar OHR | ceiling | corridor | verdict |
|---|---|---|---|---|---|
| wiki_2019t | markov2 τ=.06 k=1 | 0.5531 @1.09× | 0.8181 @1.01× | **+26.49** | **LIVE** |
| cluster50 | markov3 τ=.06 k=1 | 0.7080 @1.02× | 0.9116 @0.99× | **+20.37** | **LIVE** |
| meta_rprn | markov3 τ=.06 k=1 | 0.6738 @1.03× | 0.9995 @0.66× | **+32.57** | **LIVE** |

The LSTM never took the bar anywhere: wiki peak 0.172 (precision ≤0.42, aggressive configs over
BW), cluster50 peak 0.459 vs Markov-3's 0.708, meta_rprn peak 0.400 vs 0.674.

**Scope caveat on the LSTM arm (word the paper accordingly):** next-token objective + sample10
traces = handicap (sampling destroys exact adjacency: cluster50 eval top-1 = 0.0001 despite 99%
vocab coverage; unsampled meta_rprn got 0.143 yet still gained ~1 pt as a prefetcher — next-token
accuracy is not the prefetch objective). Claim as *"survives every predictor we fielded,
including a neural sequence model"*, NOT *"no learnable predictor can close it"*. The
provably-unlearnable part is the cold slice (below).

## COLD SPLIT — the learnable corridor ⚠ v1 RETRACTED (bad definition), v2 re-running

**v1 (2026-07-19, RETRACTED):** first cold-split defined cold = "object not yet requested in
replay at issue time." WRONG — a frozen predictor legitimately prefetches an object ahead of its
first LOCAL occurrence (it learned the association in training), so the sanity invariant failed:
**bar cold hits = 442,188 on wiki, not 0.** That over-counts the cold slice and makes learnable
corridors far too pessimistic. Retracted numbers (do NOT cite): wiki −10.02, meta_rprn −17.61,
meta_reag −11.90, msr_hm_0 +0.17, cluster53 +20.29, cluster50 +15.50.

**v2 fix (committed):** cold = object OUT of the predictor's TRAINING-prefix vocabulary (true
OOV → table/LSTM has no entry → only clairvoyance can prefetch it). This makes **bar cold hits
exactly 0 by construction** (invariant restored) and counts only the genuinely unlearnable
compulsory slice on the ceiling. Since train-cold ⊆ replay-cold, every v2 learnable corridor is
**≥ its retracted v1 value** — the true picture is strictly better than the scary v1 negatives.

**Mechanism the split still exposes (a real 4th inflation mode):** on high-churn traces the iso-BW
clairvoyant ceiling spends budget prefetching compulsory OOV misses a history-based scheduler
can never reach. How much of each gross corridor is truly OOV vs. mere prefetch-ahead is exactly
what v2 measures — unknown until the re-run.

**Scope note (unchanged):** OOV cold is unreachable *by a history-based scheduler* (our layer). A
content-aware predictor could recover some — but that is Job 2 (prediction), a different layer.

## v3 LEARNABLE CORRIDOR — warm-restricted Prescient (2026-07-19) ← PRIMARY METRIC

Learnable ceiling = Prescient restricted to in-training-vocab objects, at the bar's byte budget
(reallocates the budget the gross oracle wastes on unreachable cold objects onto warm ones).
This is the honest, budget-fair upper bound for a history-based scheduler. Verdict is sized
against IT. bar cold hits = 0 and warm cold hits = 0 (invariants, verified in every run).

| trace | family | churn | bar | warm ceiling | **v3 learnable** | verdict |
|---|---|---|---|---|---|---|
| cluster53 | KV | 0.070 | 0.5973 | 0.7940 | **+19.67** | LIVE |
| cluster50 | KV | 0.069 | 0.7080 | 0.8836 | **+17.56** | LIVE |
| wiki_2019t | CDN | 0.439 | 0.5531 | 0.6803 | **+12.72** | **LIVE (revived from +7.27)** |
| msr_hm_0 | block | 0.137 | 0.8380 | 0.8869 | +4.89 | dead |
| meta_rprn | CDN | 0.507 | 0.6738 | 0.7169 | +4.32 | dead |
| meta_reag | CDN | 0.368 | 0.7543 | 0.7906 | +3.63 | dead |

**LIVE SET (final) = {wiki, cluster50, cluster53}** — 3 traces, CDN + KV.

**Mechanism (sharpened — NOT churn):** dead traces are where the tuned bar already saturates the
warm ceiling (meta_reag 0.754 vs 0.791; msr_hm_0 0.838 vs 0.887; meta_rprn 0.674 vs 0.717 — bar
captures the warm timing). LIVE traces are where the bar FAILS to schedule exploitable warm reuse
(wiki 0.553 vs 0.680). Learnable headroom = exploitable warm reuse × predictor's failure to time
it = exactly the gap an RL scheduler targets. This is the paper's thesis, mechanistically stated.

**wiki revived: +7.27 → +12.72** — its gross oracle wasted 27% of budget on cold objects and wiki
has enough exploitable warm reuse to absorb the reallocation (+5.4 pts). Predicted even-odds
BEFORE running; cleared. Bar not moved.

**Correction to an earlier claim:** v3 is NOT strictly ≥ v2. warm > naive only where warm reuse is
exploitable under budget (wiki); on high-churn traces it ≈ naive or dips a hair (meta_rprn
+4.67→+4.32) via eviction-coupling. The metric helps exactly where real warm timing headroom
exists — which is the honest behavior.

**Live set now spans CDN + KV** (wiki + Twitter clusters), not "low-churn KV only" — a broader,
better-defended claim. Headroom lives where warm reuse is exploitable under budget.

## BOOTSTRAP CI — v3 learnable corridor, all 3 live traces (2026-07-19, `oppcert/sim/bootstrap.py`)

Paired **moving-block bootstrap** 95% CI on the honest metric (warm-restricted-Prescient OHR −
tuned-bar OHR). Blocks resampled together on the *same* post-warmup requests → the CI is on the
paired per-request difference (diff ∈ {−1,0,1}), so it reflects timing-corridor variance, not two
independent OHRs. Block bootstrap (not iid) because hit/miss labels are temporally autocorrelated.
Config: 1,900,000 post-warmup reqs, 1000 blocks × 1900, 10,000 resamples, seed 0. **Pre-registered
reliability rule: a trace is *reliably* live only if the CI lower bound ≥ 8.**

| trace | bar OHR | warm ceiling | learnable (pt) | 95% CI | SE | verdict |
|---|---|---|---|---|---|---|
| cluster53 | 0.5973 | 0.7940 | **+19.67** | [+19.41, +19.93] | 0.13 | reliably LIVE |
| cluster50 | 0.7080 | 0.8836 | **+17.56** | [+15.78, +19.39] | 0.93 | reliably LIVE |
| wiki_2019t | 0.5531 | 0.6803 | **+12.72** | [+10.93, +14.52] | 0.92 | reliably LIVE |

**All three lower bounds clear 8 decisively** — wiki, the revived and lowest-margin trace, still
lands at **+10.93** (≈3 pts of slack below the bound). The revival is not a point-estimate artifact.

Note: cluster53's CI is ~7× tighter (SE 0.13 vs ~0.9) — its per-block corridor is unusually
stationary (smallest footprint, 747 MB; corridor near-constant across the trace). Faithfully
recorded; if anything it strengthens the point, but worth an eyeball before it goes in a table.

## GATE B — interaction 2×2, n=6 (2026-07-19, `logs/gate_b_live.log`)

ALL SIX interactions negative → **substitutes, not complements, is STRUCTURAL** (msr_proj_0 −1.34,
msr_hm_0 −2.31, wiki −3.88, cluster53 −6.66, cluster50 −7.83, meta_rprn −7.94). Kills JOINT
eviction+prefetch co-training (original P4 thesis). Does NOT touch the tempo/scheduling pivot
(fixed evictor + prefetch timing) — if anything supports decoupling. Say this explicitly in paper.

## GATE C — LIT CHECK (2026-07-19, inline web sweep; supersedes pending `wia7wmc22`)

**Question:** (A) is the *instrument* — pre-registered iso-BW timing corridor + cold/warm learnable
deflation for object caches — prior art? (B) is "predict WHEN not WHAT" already claimed for object
caches? **Verdict: wedge SURVIVES; one mandatory reframe.**

- **Claim A (the AAAI-27 measurement paper): NOVEL.** No prior work builds a pre-registered,
  iso-bandwidth instrument that measures the prefetch *timing* corridor on object caches and deflates
  it via an in-training-vocab (warm) vs OOV (compulsory-cold) split. Nearest neighbors are on other
  axes: **Demand-MIN** (ISCA'18) = Belady variant minimizing *demand* misses under prefetch, but HW /
  uniform lines / no bandwidth corridor / no learnable split; **Baleen** (FAST'24) quantifies a
  gap-to-OPT but OPT is an *admission* bound, no cold/warm split. The evaluation-protocol genre
  (pre-registration + seed-averaging to deflate non-reproducible gains) exists only in a *different
  field* — **"A Sober Look at Progress in LM Reasoning"** (arXiv 2504.07086). Cite as lineage; nobody
  has done it for caching.
- **Claim B (the RL paper): SLOGAN pre-empted, FORMULATION novel.** "Predict WHEN not WHAT" cannot be
  the headline — prefetch *timeliness* is foundational and already gated on: **Baleen's "ML-When"**
  gates episodes by benefit probability, **Pythia** (MICRO'21) classifies timely-vs-late with
  bandwidth feedback. But the specific MDP — budgeted, per-candidate {issue-now/defer/drop} scheduling
  of a fixed token-bucket across competing candidates, offline, anchored to a computable relaxation
  return — is unclaimed. DeePref (arXiv'23) = online per-request *what*+aggressiveness; LightCacheRL
  (2025) = myopic bandit; joint-HW (2510.10862) = representation-sharing; Cold-RL (2025) = eviction
  only (deployment template). DEAP (2020) = the joint thesis already killed.

**Consequences (pre-registered "tighten if prior art"):** (1) drop the "when not what" slogan from the
abstract — lead with the instrument + phantom-headroom deflation (Claim A). (2) Related Work must
foreground Baleen + Demand-MIN with the one-sentence gap each, and cite A-Sober-Look for the genre.
The numeric ≥8 bar does NOT tighten (already survived Markov-3 + LSTM); only the framing does.

**Neighbor map (Related Work spine):** Baleen FAST'24 (nearest; joint admit+prefetch, DWPD budget, no
timing corridor / cold-warm split / pre-reg) · Demand-MIN ISCA'18 (HW prefetch-aware optimal, no BW
corridor) · DEAP'20 (joint, killed thesis) · DeePref'23 (online RL CDN, what+aggressiveness) · Cold-RL
'25 (offline RL eviction only; template) · LightCacheRL'25 (myopic bandit, HW) · Joint-HW 2510.10862
(reps+MARL, HW) · Pythia MICRO'21 (RL aggressiveness, one knob, HW) · A Sober Look 2504.07086 (eval-
protocol genre, different field). Also noted: LLM prefix/KV caching heating up (openreview Vj48eXaQDM)
— supports the Ring-3 LLM-serving direction, not a threat to Ring 1.

---
## (superseded) v2 learnable = gross − cold — too harsh, kept for provenance

**v2 RESULTS (2026-07-19, all `bar cold hits = 0`) — learnable column now superseded by v3:**

| trace | family | objs | churn=objs/2M | gross | cold | **learnable** | verdict |
|---|---|---|---|---|---|---|---|
| cluster53 | KV | 140k | 0.070 | +24.46 | 4.62 | **+19.84** | **LIVE** |
| cluster50 | KV | 138k | 0.069 | +20.37 | 2.69 | **+17.67** | **LIVE** |
| wiki_2019t | CDN | 877k | 0.439 | +26.49 | 19.22 | +7.27 | below 8 |
| meta_rprn | CDN | 1.01M | 0.507 | +32.57 | 27.90 | +4.67 | below 8 |
| msr_hm_0 | block | 274k | 0.137 | +10.08 | 5.82 | +4.26 | below 8 |
| meta_reag | CDN | 736k | 0.368 | +24.55 | 20.71 | +3.84 | below 8 |

**LEARNABLE-LIVE SET = {cluster50, cluster53}** — both low-churn Twitter KV. Everything else falls
below the pre-registered 8-pt bar on the honest metric. wiki held at +7.27 (NOT moved).

**THE CLEAN FINDING (the paper's spine now):** the cold slice — unlearnable compulsory-miss
elimination — is governed by **object churn** and is nearly monotone in it:
`churn 0.069→cold 2.69 | 0.070→4.62 | 0.137→5.82 | 0.368→20.71 | 0.439→19.22 | 0.507→27.90`.
Genuine learnable timing headroom (≥8) survives ONLY where churn is low (≲0.1). On high-churn
CDN/storage, the celebrated gross corridors are 60–90% clairvoyant compulsory-miss elimination that
no history-based scheduler — and no RL policy on one — can ever reach. **The gross corridor is a
mirage on exactly the workloads the field most often cites (CDN).**

**Consequences:** (1) headline = low-churn KV, not wiki/CDN; (2) n=2 live is THIN — expand via the
other ~52 Twitter clusters (cold-split each, ~4 min) to build a proper low-churn-KV benchmark and
give the RL method a real eval set; (3) RL scheduler targets cluster50/cluster53 + expanded set.

**THE INVERSION (the paper's real finding):** cold slice tracks **object churn**. Learnable timing
headroom concentrates in LOW-churn workloads; high-churn CDN corridors are dominated by unlearnable
compulsory-miss elimination. wiki, the former headline, falls to +7.27 (below bar). **Headline
shifts from wiki/CDN to low-churn KV (cluster50 +17.67, cluster53 pending).** The cold-split is the
instrument that distinguishes the regimes — that IS the contribution now, not the raw corridor.
Bar held at ≥8: wiki does not clear, full stop.

**Consequence for the RL method:** retarget from wiki to cluster50/cluster53 — build & demo the
scheduler where learnable headroom actually exists.

## Mechanism read (revised — the family story was wrong, the property story is better)

Pre-run prediction was "block dead, CDN/KV live." Reality: **liveness is a trace property, not a
family label** (msr_hm_0 is a LIVE block trace; cluster10 a DEAD KV trace). What actually predicts
the verdict across all 12:

- **DEAD ⟺ the bar saturates the ceiling**: traces whose sequences are so deterministic that
  Markov hits ~1.0 precision at ~1.00× traffic (msr_web_2 prec 0.996, w105 0.946, cluster10
  1.000) leave nothing for timing/prediction to win. Corridor collapses or goes negative.
- **LIVE ⟺ prediction-limited reuse**: large object populations with reuse that Markov's
  high-confidence predictions cannot cover (wiki: 877k objects, bar stuck at 0.553 even
  uncapped) while the future still contains the hits (ceiling 0.82–1.00).

## Caveats (status as of 2026-07-19 — most now RESOLVED)

1. ✅ **Cold-miss share — RESOLVED.** Handled by the v2 cold-split then the v3 warm-restricted
   ceiling (primary metric above). The learnable corridor now excludes all compulsory OOV hits by
   construction (bar/warm cold hits = 0). Live set finalized at {wiki, cluster50, cluster53}.
2. ⚠ **Meta ceilings at 0.66–0.71× traffic** (gross diagnostic only): perfect timing reduces origin
   bytes by killing evict-refetch cycles. Real effect; report byte-hit-rate columns in the
   supplement. Note the Metas are DEAD on the learnable metric regardless, so this no longer
   affects any live verdict — supplement-only.
3. ✅ **LSTM aliveness bar — PASSED** (strong-bar check #3). Corridor survives a learned decoupled
   predictor on all live traces.
4. ⬜ **Gate C citations UNVERIFIED** — Baleen FAST'24, Demand-MIN ISCA'18, A-Sober-Look 2504.07086
   were not independently confirmed; verify each against its actual paper before writing §2.
5. ⬜ **Bootstrap CI is per-trace sampling variance only** — does not cover trace-choice / split /
   tuning uncertainty. Report as "on trace X, corridor = P [lo,hi]", never a general ±.

## Verdict state (2026-07-19)

- Gate A breadth ×12: **DONE** — 6 live on gross corridor.
- Strong-bar (Markov-3 + fine-τ + LSTM) on all live traces: **DONE** — cluster26 killed; wiki,
  cluster50, cluster53, meta_reag, meta_rprn, msr_hm_0 survive on gross.
- Cold-split v2 → warm-ceiling v3 (learnable corridor, PRIMARY METRIC): **DONE** — LIVE SET FINAL
  **{wiki +12.72, cluster50 +17.56, cluster53 +19.67}** (CDN + KV); Metas + msr_hm_0 dead.
- Bootstrap 95% CI on the 3 live traces: **DONE** — all lower bounds ≥ 8 (wiki lo = +10.93).
- Gate B (interaction 2×2, n=6): **DONE** — substitutes structural; supports decoupling.
- Gate C (lit check): **DONE** — wedge survives; drop "when not what" slogan, lead with instrument;
  citations pending verification.

## Run log

| date | run | log |
|---|---|---|
| 2026-07-17 | Gate A msr_proj_0 (first real-trace kill) | (session) |
| 2026-07-17 | Gate A wiki_2019t, cluster26 | (session) |
| 2026-07-18 | Breadth Gate A ×12 | `logs/breadth_summary.txt`, `logs/breadth_nohup.log` |
| 2026-07-18 | Strong bar Markov-3 wiki | `logs/strongbar_m3_wiki.log` |
| 2026-07-18 | Markov-3 + fine-τ cluster26 → **flips LIVE→DEAD** | `logs/strongbar_m3_cluster26.log` |
| 2026-07-19 | Day 1 LSTM + Markov-3 + fine-τ bar → **3/3 LIVE** | `logs/day1_nohup.log`, `logs/strongbar_summary.txt` |
| 2026-07-19 | Bootstrap CI ×3 live traces → **all lower bounds ≥ 8** | `oppcert/sim/bootstrap.py` (wiki/cluster50/cluster53) |
| 2026-07-19 | **Gate C lit check (inline web sweep)** → wedge LIVE, reframe | (session; citations UNVERIFIED) |
| 2026-07-19 | Cold split v2 (OOV) → v3 (warm ceiling) ×6 | `logs/warm_*.log` |
| 2026-07-19 | Strong-bar battery: meta_reag, cluster53, msr_hm_0 → all dead on learnable | `logs/day1b_nohup.log` |
| 2026-07-19 | **Live set FINAL {wiki, cluster50, cluster53}; all bootstrap lower bounds ≥ 8** | (session) |
