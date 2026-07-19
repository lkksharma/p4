# P4 — Context Prompt: the idea, the proven gap, and what's left

*Written 2026-07-19/20. Supersedes the earlier version of this file (2026-07-17), which predates
the measurement paper, F1/F4, and HJS-L entirely. If you are a fresh reader or a fresh session,
read this top to bottom — it is self-contained.*

---

## 1. The idea, in one paragraph

A cache has three decisions: **what to evict** (solved — S3-FIFO), **what to prefetch** (solved
well enough — Markov tables and an LSTM both saturate quickly), and **given a stream of predicted
candidates and a hard shared byte budget, which ones get fetched and when** (open — nobody
schedules this in a principled way). We built an instrument that measures, honestly, how much hit
rate is on the table from getting that third decision right. Then, rather than jumping straight to
building an RL agent to capture it, we adopted a discipline — the **Necessity Ladder**: climb one
rung of model complexity at a time, and only pay for the next rung if a cheap, pre-registered,
replay-only measurement proves it's needed. Every rung's outcome is a publishable result, including
"this doesn't work" — that's the point. This project has now killed itself honestly five separate
times before this idea (CAPO, CLCA, Certifend, the RAO-RL joint-cache flagship, and the original
"joint eviction+prefetch RL" thesis this same repo started with) — the discipline is not incidental,
it is the entire reason anything here is trustworthy.

---

## 2. The proven gap — DONE, this is a separate, submittable paper

This is finished, measured, bootstrap-confirmed, and compiled into a PDF. It does not depend on
anything below it.

**Instrument.** Pre-registered, iso-bandwidth, adversarially-tuned: the "tuned bar" is the best of
{Markov-1, Markov-2, Markov-3, an LSTM} × threshold × fanout, capped at 1.15× no-prefetch traffic.
The **learnable corridor** = (a clairvoyant oracle restricted to objects the predictor could ever
have learned, at the bar's exact byte rate) − (the tuned bar). Restricting the ceiling to
in-training-vocabulary objects (not "anything the future contains") is the whole trick — it prices
out compulsory-miss elimination, which no history-based scheduler can ever reach.

**Result, 12 production traces (CDN, KV, block), pre-registered bar = corridor ≥ 8 pts + Pareto
dominance:**

| trace | family | tuned bar OHR | learnable ceiling OHR | corridor | 95% CI (block bootstrap) |
|---|---|---|---|---|---|
| **cluster53** (Twitter KV) | KV | 0.5973 | 0.7940 | **+19.67** | [+19.41, +19.93] |
| **cluster50** (Twitter KV) | KV | 0.7080 | 0.8836 | **+17.56** | [+15.78, +19.39] |
| **wiki_2019t** (Wikipedia CDN) | CDN | 0.5531 | 0.6803 | **+12.72** | [+10.93, +14.52] |
| msr_hm_0 | block | 0.8380 | 0.8869 | +4.89 | dead |
| meta_rprn | CDN | 0.6738 | 0.7169 | +4.32 | dead |
| meta_reag | CDN | 0.7543 | 0.7906 | +3.63 | dead |
| msr_proj_0, msr_web_2, w105, w87, cluster10, cluster26 | — | — | — | — | dead by 3 other mechanisms (bar-saturation, traffic-bought corridor, degenerate predictability) |

**Live set (final): {wiki_2019t, cluster50, cluster53}** — 3 traces, spanning CDN and KV. All three
lower bootstrap bounds clear 8 (wiki's is the closest, at +10.93). Gross corridors before the
warm/cold split were far bigger (up to +32.6) — 14–87% of each was phantom (unreachable compulsory
misses); the deflation itself, by workload class, is a headline finding.

**Also measured:** eviction and prefetch quality are **substitutes, not complements** (n=6, all
interactions negative) — kills joint co-optimization, supports treating the evictor as fixed.
**Lit check:** the instrument is novel; nearest neighbors are Baleen (FAST'24) and Demand-MIN
(ISCA'18), neither of which does the timing-corridor + warm/cold split. *(4 arXiv-only citations
still need author-list verification before camera-ready — see `results_p4.md` caveat #4.)*

**Deliverables that exist:** `paper_draft.md` (markdown draft), `paper.tex` + `paper.pdf`
(compiled, 10 pages, 3 TikZ figures, 3 tables). `results_p4.md` is the full ledger with every
retraction on the record (v1 cold-split bug, the naive-subtraction-vs-warm-ceiling correction,
etc.) — that audit trail is itself part of the paper's credibility argument.

**Status: essentially submission-ready**, pending citation verification and a red-team pass. This
targeted AAAI-27 (abstract Jul 21 / paper Jul 28) — confirm that deadline is still live before
treating it as final; if it has passed, retarget the next appropriate venue without re-deriving
anything above.

---

## 3. The capture-method track (Necessity Ladder) — IN PROGRESS, do not trust the last real-data numbers yet

This is the **separate, later** paper that asks: given the proven corridor above, can any *causal*
policy actually capture it? It targets AAAI-28 / IAAI-27, not the deadline above.

### 3.1 The ladder, and where each rung stands

| rung | question | status |
|---|---|---|
| **F1** | Timing-only ceiling: how much of the corridor is reachable by perfect *scheduling* alone, restricted to what the predictor could ever name? | ✅ **Fixed, verified, trustworthy.** wiki 95.8% of corridor, cluster50 91.2%, cluster53 7.3% (cluster53's corridor is mostly *object-choice*, not timing — a real, informative boundary result). |
| **F4** | Necessity gap: how much could a *non-separable* (joint/packing) policy add over a per-candidate threshold rule? Bounds what RL could contribute. | ✅ **Fixed, verified, trustworthy.** <1 pt on all 3 live traces → **RL is measurably unnecessary here.** This closes the RL rung of the ladder as a positive, pre-registered, no-training-required finding. |
| **F2** | Is the *when* (use-lag) learnable at all from causal features? Gates whether HJS-L (the closed-form scheduler) can work. | ⚠️ **Built, but the last real-data run used SUPERSEDED code. Numbers below are not trustworthy — re-run required before drawing any conclusion.** |
| **HJS-L** | The closed-form hazard-priced scheduler itself: capture fraction of F1, ablations, Pareto/two-axis reporting. | 🔧 **Built (significant recent additions), never run on real trace data.** |

### 3.2 ⚠️ Critical: why the F2 numbers you have are stale

The F2 gate was run once on real data (wiki/cluster50/cluster53) with a hazard model keyed on
`(recency, frequency, confidence)`. Result at that time: **wiki FAIL** (Spearman 0.065, never-AUC
0.71), **cluster50 FAIL** (Spearman 0.037, never-AUC **0.389** — *below 0.5*), **cluster53 PASS**
(Spearman 0.659, never-AUC 0.862).

**Since that run, `p4_hazard.py` was substantially rewritten** (commit `90b3413`, "fixing hazard
overfit") to fix a real methodological bug: the hazard model had been trained on the *same window*
the predictor itself trained on. In-sample, high table-confidence means "memorized and reliable";
out-of-sample it often means "built from too few observations and likely wrong" — the
confidence→outcome relationship **flips sign** between the two regimes. This is very plausibly
*exactly* what produced cluster50's AUC of 0.389 (the code now explicitly documents "never-AUC
lands BELOW 0.5" as the signature of this bug). The fix introduces a proper **three-way split**:

```
[0, cut)      the PREDICTOR trains here            (unchanged, cut = train_frac * n)
[cut, hcut)   the HAZARD MODEL trains here          (NEW — on the predictor's out-of-sample behavior)
[hcut, n)     F2 evaluates here                     (unchanged in spirit, now genuinely held out)
```

plus a **fixed observation horizon** for the "never" label (a use one million requests away is
unreachable regardless, and without a fixed horizon the label's meaning silently drifts with
position in the trace — see the docstring in `harvest()` for the full argument). Both are real
fixes to real bugs, not tuning.

**Consequence: the wiki/cluster50 FAIL verdicts might reverse, might not — genuinely unknown until
re-run.** Do not write "F2 fails on wiki/cluster50" into any paper or conclusion until it has been
re-verified against current code. cluster53's PASS is the least likely to flip (it was decisive,
not marginal) but should be re-confirmed for consistency regardless.

### 3.3 What HJS-L (`p4_hjsl.py`) has grown into

Beyond the original scheduler design, several rigor additions landed (commits `a87d3d1`,
`85edd44`, `44c6e11`) that materially change what a capture number will mean:

- **`hazard=` diagnostic modes**: `model` (the real causal policy), `oracle` (true lag, a point
  mass — reads the future, *never reportable as a result*, exists only to separate "the scheduler
  design is bad" from "the forecaster is bad"), `point` (learned median only, no distribution
  shape — tests whether the shape earns its keep).
- **`budget_mode=`**: `rate` (token bucket, the project's standing iso-BW convention) vs `total`
  (cumulative cap only, no instantaneous rate limit — the same freedom the unlimited bar already
  has). This exists because of a **fairness asymmetry**: the bar runs with unlimited instantaneous
  rate (can burst), while every scheduler arm was capped to the bar's *average* rate — a capped
  arm can lose a candidate forever if it arrives while the bucket is low, a penalty the bar never
  pays. `BAR-CAP` (the bar re-run under the same token bucket the schedulers face) isolates this:
  any gap between BAR and BAR-CAP is charged to the *protocol*, not the *scheduler*. **Report
  HJS-L against BAR-CAP, not against the unlimited bar**, or a real scheduling win can be masked by
  a fairness artifact.
- **Issue autopsy** (`on-time` / `LATE` / `never-used`): separates two distinct failure modes that
  demand opposite fixes — late issues mean deferral overshoot (lower `--gamma`); never-used issues
  mean filtering failure (raise `--floor`). Defaults are already tuned once from this signal
  (`gamma` 0.75→0.10, `floor` 0.05→0.30) using the *quantile bin-edge vs bin-midpoint* insight: log
  bins are wide, uses cluster near the low end of a bin, and a wake time computed off the midpoint
  systematically overshoots (see the `_quantile` docstring).
- **Two-axis (Pareto) reporting**: the pass/fail gate scores OHR alone, but an arm that matches the
  bar's hit rate on half the prefetch bytes is currently recorded as a failure — which is wrong on
  a byte-billed system. Both axes are now printed; read the Pareto verdict line, not just capture%.

None of this has touched real trace data yet.

---

## 4. What to do with the checklist — concrete next actions, in order

The Phase 1 / Phase 2 / Phase 3 / Phase 4 checklist from earlier stands structurally, but Phase 1
must be **re-run**, not read from old logs, before Phase 2 means anything.

### Phase 1 (re-run — mandatory before anything else)

```bash
cd /workspace/p4 && git pull && mkdir -p haz logs

python p4_hazard.py --trace data/wiki_2019t.oracleGeneral        --pred markov2 --tau 0.05 --k 1 --out haz/wiki.npz \
  2>&1 | tee logs/hazard_wiki_v2.log
python p4_hazard.py --trace data/cluster50.sample10.oracleGeneral --pred markov3 --tau 0.06 --k 1 --out haz/cluster50.npz \
  2>&1 | tee logs/hazard_cluster50_v2.log
python p4_hazard.py --trace data/cluster53.sample10.oracleGeneral --pred markov2 --tau 0.20 --k 1 --out haz/cluster53.npz \
  2>&1 | tee logs/hazard_cluster53_v2.log
```

Read the `!! never-rate gap ... extrapolating` warning if it prints — that's the code's own
self-check that the 3-way split is doing its job. Compare the new Spearman/AUC numbers to §3.2
above; do not assume they'll match.

**Decision point:** for each trace, F2 PASS or FAIL. Only traces that PASS proceed to Phase 2.
- All 3 FAIL → HJS-L has no viable target on any live trace; write that up as the finding (still
  publishable — see the pre-registered fork table in `SPEC_f1_f4_oracles.md` §6, and note F4
  already separately confirmed RL isn't the answer either, so this would mean *no* causal policy —
  closed-form or learned — can capture this corridor on these traces, which is itself a strong,
  citable negative result).
- ≥1 PASS → proceed to Phase 2 on the passing trace(s) only.

### Phase 2 (only on F2-passing traces)

```bash
python p4_hjsl.py --trace data/<TRACE> --pred <PRED> --tau <TAU> --k <K> --haz haz/<trace>.npz \
  2>&1 | tee logs/hjsl_<trace>.log
```

Read, in order: (1) invariants pass, (2) issue autopsy — tune `--gamma`/`--floor` if late-rate or
never-rate dominates, (3) **HJS-L vs BAR-CAP** (the like-for-like number, not vs. the unlimited
bar), (4) capture % of F1, (5) the two-axis Pareto verdict. Ablations worth running once a base
config looks sane: `--defer none`, `--survival none`, `--hazard oracle` (diagnostic ceiling only,
never a reported result), `--hazard point`, `--budget-mode total`.

**Pre-registered pass bar (unchanged): capture ≥ 25% of F1, CI excluding zero.**

### Phase 3 / Phase 4 (writing, unchanged from before)

Only start once Phase 2 has a stable, tuned, reported number on at least one trace. Reuse Gate C's
related-work map; add HJS-L's method section; report F1/F4/F2 as the paper's own necessity-gating
methodology (this triad — measure timing-headroom, measure RL-necessity, measure when-learnability,
*before* building anything — is arguably the more novel contribution than the scheduler itself).

---

## 5. House rules this project has earned the hard way (do not relitigate these)

1. **Every gate is pre-registered before the run that could satisfy it.** Thresholds don't move
   after seeing a number, in either direction — not to save a discouraging trace, not to make a
   flattering one "final."
2. **A gate that never kills anything is not trusted.** This project's credibility rests on a
   visible trail of self-caught bugs (the synthetic +42 that became −0.02; the cluster26 flip; the
   retracted cold-split v1; the F1 vocab-cap bug; the F4 SEP-ordering bug; now the hazard
   train/eval overlap bug) — each one caught before it reached a paper, not after.
2b. **When a gate fails, ask "is the gate broken?" before "is the hypothesis dead?"** — but only
   once per gate, with a specific, falsifiable fix, not indefinitely. Recency rescued F2 on
   cluster53; the 3-way split may or may not rescue wiki/cluster50 — find out by running it, don't
   assume either way.
3. **No n=1 generalization.** "Structural" requires replication across families; a single trace is
   a data point, not a claim.
4. **Report diagnostics as diagnostics.** `hazard=oracle` reads the future and exists only to
   separate forecaster quality from scheduler quality — it must never appear as a capture number in
   a table.
5. **Fairness controls are not optional decoration.** BAR-CAP exists because an unfair comparison
   (unlimited bar vs. rate-capped scheduler) would make a real win invisible or a fake loss look
   real. Always report against the like-for-like control.
