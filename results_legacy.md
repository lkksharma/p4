<!--
RESTORED FROM GIT HISTORY (was `git rm`'d as `results.md`, commit prior to HEAD). This is an
EARLIER SNAPSHOT of the measurement paper's ledger -- every section here (Gate A breadth sweep,
strong-bar checks #1-#3, the cold-split v1(retracted)->v2(superseded)->v3 saga, Gate B) is fully
subsumed by results_p4.md, which is the later, complete version of the SAME ledger (same headers,
same numbers, plus the bootstrap-CI section this snapshot predates). Kept here verbatim, unedited,
purely for provenance / history -- results_p4.md remains the authoritative measurement-paper ledger.
Not to be confused with results.md, which is this repo's NEW, separate ledger for the second paper
(the capture-method / Necessity Ladder track: F1, F4, F2, HJS-L, F5, Policy 1, Policy 2).
-->

# P4 Results Ledger

All numbers produced by the pre-registered Gate A protocol (`p4_sweep.py` / `p4_strongbar.py`):
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

## GATE B — interaction 2×2, n=6 (2026-07-19, `logs/gate_b_live.log`)

ALL SIX interactions negative → **substitutes, not complements, is STRUCTURAL** (msr_proj_0 −1.34,
msr_hm_0 −2.31, wiki −3.88, cluster53 −6.66, cluster50 −7.83, meta_rprn −7.94). Kills JOINT
eviction+prefetch co-training (original P4 thesis). Does NOT touch the tempo/scheduling pivot
(fixed evictor + prefetch timing) — if anything supports decoupling. Say this explicitly in paper.

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

## ⚠ Open caveats (must resolve before the abstract overclaims)

1. **Cold-miss share of the corridor — the big one.** Prescient prefetches objects it has *never
   seen* (clairvoyant first-fetch eliminates compulsory misses); NO history-based predictor,
   learned or not, can do that. On high-cold-miss traces (wiki base 0.15, meta_rprn base 0.39,
   ceilings ~1.00) an unknown fraction of the corridor is unlearnable-in-principle. Needed:
   decompose ceiling prefetch-hits into first-seen vs previously-seen objects and report the
   **learnable corridor** (previously-seen only) alongside. Until then the +26.5 is an upper
   bound in TWO senses (clairvoyance AND cold-start).
2. **Meta ceilings at 0.66–0.71× traffic**: perfect timing *reduces* total origin bytes
   (eliminates evict-refetch cycles). Real effect, worth a sentence in the paper — but also
   inflates Pareto dominance on those traces; report byte-hit-rate columns in the supplement.
3. **LSTM bar (Day 1) still pending — the pre-registered aliveness test.** Nothing is final
   until the corridor survives a learned decoupled predictor on wiki + one Meta + cluster26.

## Verdict state

- Pre-registered Gate A: **PASSED at breadth** (6/12 live after strong-bar #2, replicated
  within every family, corpses reported).
- Strong bar #1 (Markov-3 + extended τ): **PASSED on wiki; KILLED cluster26** — the remaining
  five LIVE verdicts are provisional until they clear the same grid (bundled into Day 1).
- Strong bar #3 (LSTM + Markov-3 + fine τ): **PASSED on wiki, cluster50, meta_rprn** — the
  aliveness gate is cleared on one trace per live family.
- Still provisional (coarse grid only): meta_reag, cluster53, msr_hm_0 — queue the same battery.
- Cold-miss corridor decomposition: **instrumented** (`pf_cold_hits` in PFCache; gate_a now
  prints COLD SLICE + LEARNABLE CORRIDOR; `p4_coldsplit.py` retrofits the split onto already-
  measured corridors in 2 replays). Numbers pending — blocks final abstract wording.
- Gate B (evict×prefetch interaction, n>1): pending.

## Run log

| date | run | log |
|---|---|---|
| 2026-07-17 | Gate A msr_proj_0 (first real-trace kill) | (session) |
| 2026-07-17 | Gate A wiki_2019t, cluster26 | (session) |
| 2026-07-18 | Breadth Gate A ×12 | `logs/breadth_summary.txt`, `logs/breadth_nohup.log` |
| 2026-07-18 | Strong bar Markov-3 wiki | `logs/strongbar_m3_wiki.log` |
| 2026-07-18 | Markov-3 + fine-τ cluster26 → **flips LIVE→DEAD** | `logs/strongbar_m3_cluster26.log` |
| 2026-07-19 | Day 1 LSTM + Markov-3 + fine-τ bar → **3/3 LIVE** | `logs/day1_nohup.log`, `logs/strongbar_summary.txt` |
| — | Cold split ×6 live traces (`p4_coldsplit.py`) | pending |
| — | Strong-bar battery: meta_reag, cluster53, msr_hm_0 | pending |
