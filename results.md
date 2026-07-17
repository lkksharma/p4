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
