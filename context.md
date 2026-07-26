# P4 — Context Prompt: the idea, the proven gap, and where capture stands

*Written 2026-07-20, evening. Supersedes the 2026-07-19/20 version of this file, which predates F5,
Policy 1, F7, the at-emission-ceiling finding, and the decision to submit ONE combined paper. If you
are a fresh reader or a fresh session, read this top to bottom — it is self-contained.*

---

## 0. The single most important fact for a fresh session

**This is now ONE paper, not two.** The user decided to submit the proven measurement result
(§1–2 below) and the capture-method investigation (§3 below) as a **single combined AAAI
submission**, not as separate papers on separate timelines. Do not plan around "paper 1 / paper 2"
as independent deliverables — the capture track's outcome (still resolving, see §3.6) determines
the second half of *the same* paper, not a different one. (Saved to memory: `p4-paper-plan.md`.)

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
it is the entire reason anything here is trustworthy. The ladder has since killed three more rungs
inside this idea (RL arbitration, at-emission dispatch, and — pending §3.6 — possibly causal JIT
timing), and each kill made the surviving finding sharper, not weaker.

---

## 2. The proven gap — DONE, forms the first half of the combined paper

Finished, measured, bootstrap-confirmed. Does not depend on anything in §3.

**Instrument.** Pre-registered, iso-bandwidth, adversarially-tuned: the "tuned bar" is the best of
{Markov-1, Markov-2, Markov-3, an LSTM} × threshold × fanout, capped at 1.15× no-prefetch traffic.
The **learnable corridor** = (a clairvoyant oracle restricted to objects the predictor could ever
have learned, at the bar's exact byte rate) − (the tuned bar). Restricting the ceiling to
in-training-vocabulary objects is the whole trick — it prices out compulsory-miss elimination.

**Result, 12 production traces (CDN, KV, block), pre-registered bar = corridor ≥ 8 pts + Pareto
dominance. Live set (final): {wiki_2019t, cluster50, cluster53}**:

| trace | family | tuned bar OHR | learnable ceiling OHR | corridor | 95% CI |
|---|---|---|---|---|---|
| cluster53 (Twitter KV) | KV | 0.5973 | 0.7940 | **+19.67** | [+19.41, +19.93] |
| cluster50 (Twitter KV) | KV | 0.7080 | 0.8836 | **+17.56** | [+15.78, +19.39] |
| wiki_2019t (Wikipedia CDN) | CDN | 0.5531 | 0.6803 | **+12.72** | [+10.93, +14.52] |
| msr_hm_0, meta_rprn, meta_reag | — | — | — | +4.89 / +4.32 / +3.63 | dead (below 8) |
| msr_proj_0, msr_web_2, w105, w87, cluster10, cluster26 | — | — | — | — | dead by 3 other mechanisms |

**Also measured:** eviction and prefetch quality are **substitutes, not complements** (n=6, all
interactions negative) — kills joint co-optimization, supports treating the evictor as fixed. This
result is now directly relevant to §3: every capture experiment below keeps S3-FIFO frozen.

**Lit check (done this session, web-verified):** the instrument's novelty holds against Baleen
(FAST'24 — nearest neighbor; its OPT target is *not* vocab-restricted, so it inherits exactly the
phantom component we price out), Demand-MIN (ISCA'18 — explicitly not iso-bandwidth), the
Pythia/CRL-Pythia/Micro-MAMA RL-prefetcher line, Whittle-index caching (different control problem —
eviction/freshness, not prefetch-candidate selection), and a NeurIPS'25-workshop joint
eviction+prefetch paper (hardware CPU lines, preliminary — our substitution finding is the opposite
conclusion, on a different domain; don't claim universality against it). Full neighbor table and
honest tensions are in the conversation record; not yet written to a `related_work.md` file.

**Deliverables that exist:** `paper_draft.md`, `paper.tex` + `paper.pdf` (compiled, 10 pages, 3
TikZ figures, 3 tables). `results_p4.md` is the full ledger with every retraction on record.
`results_legacy.md` is an earlier, fully-superseded snapshot of the same ledger, restored from git
history for provenance only — never cite numbers from it.

**Status: essentially submission-ready on its own terms**, pending citation verification. Originally
targeted AAAI-27 (abstract Jul 21 / paper Jul 28) — **confirm that deadline is still live** before
treating it as final. Per §0, this half now ships together with whatever §3 resolves to, not alone.

---

## 3. The capture-method track — MUCH FURTHER ALONG than the last context.md said

Asks: given the proven corridor above, can any *causal* policy actually capture it? This session
resolved F2/HJS-L's stale-numbers problem definitively (HJS-L is dead, cleanly, for a documented
reason), built and ran two new rungs (F5, Policy 1), and built a third (F7) that is the decisive,
still-resolving gate as of this writing.

### 3.1 Ladder status table (current)

| rung | question | status |
|---|---|---|
| **F1** | Timing-only ceiling on the predictor's k=1 candidates | ✅ trustworthy. wiki 95.8% of v3 corridor, cluster50 91.2%, cluster53 7.3% |
| **F4** | Does joint/RL arbitration beat a threshold rule? | ✅ trustworthy. **<1 pt everywhere → RL unnecessary.** Closed. |
| **F2** | Is the causal use-lag ("when") learnable at all? | ✅ PASS on all 3 traces post the 3-way-split fix (Spearman 0.60–0.65, AUC 0.81–0.95). Superseded in relevance by §3.6 below — kept alive only as the F7 hazard model's training source. |
| **HJS-L** | Closed-form hazard-priced scheduler, capture vs F1 | ❌ **DEAD.** DEFER killed cleanly: `S(100)=1.000` on all 3 real traces (no eviction race exists), so deferring to time a fetch "just right" only ever risks arriving LATE. A perfect forecaster with DEFER only *ties* the bar (wiki +0.11, cluster50 −3.06). Do not revisit DEFER at k=1. |
| **F1-stream** | Perfect timing on the ACTUAL k=1 emission stream | ❌ **BELOW the bar on both traces** (wiki −7.01, cluster50 −10.82). Decisive: the corridor is NOT a timing problem on the narrow stream — it's a *coverage* problem. This redirected the whole track. |
| **F5** | Coverage ceiling: wide emission (k=16–32) + clairvoyant JIT + clairvoyant selection | ✅ **CONFIRMED on both live traces.** wiki +10.41 (CI lo +8.74, k=32/top_m=32), cluster50 +11.42 (CI lo +9.95, k=16). The corridor IS reachable in principle by width, not by timing. |
| **Policy 1** | Causal wide-emission market, AT EMISSION (no defer) | ❌ **DEAD, cleanly audited.** Even a perfect forecaster using F5's own selection rule captures ≤6% of F5 at emission (wiki +0.62 at best; cluster50 −3 to −27%). At-emission is the ceiling, not the forecaster — Policy 2 (a better predictor) cannot rescue this. |
| **F7** | JIT-realizability: is CAUSAL use-lag timing accurate enough to insert just-in-time? | 🔧 **Built, running now. Decisive rung — the paper's final positive/negative fork.** Partial wiki result (gamma=0.1) showing LATE climbing to 92% by the halfway point — looking like a likely FAIL, but not yet confirmed across the gamma sweep or on cluster50. |

### 3.2 Why HJS-L is dead (do not rebuild it)

`S(100)=1.000` on wiki, cluster50, cluster53 — objects essentially never get evicted within 100
requests of being prefetched. There is no eviction race to win by deferring a fetch to land "just
before use." Deferring only ever adds risk of the wake landing *after* the use (LATE — a total
loss). HJS-L's own real-trace runs bore this out repeatedly (LATE rate 43–59% across several
configs, several bug-fix rounds — quantile-midpoint bug, floor sweep, budget-mode total to remove
the rate-cap confound — none of it rescued DEFER). Even the diagnostic oracle-hazard arm (perfect
forecaster, DEFER on) only tied the bar. **This is closed.** The `_quantile` bin-edge convention it
established (log bins are wide, use lags cluster near the low end, so the LOWER edge — not the
midpoint — is the correct convention) is reused correctly in F7 below.

### 3.3 Why F1-stream redirected everything

F1-stream is the JIT oracle (perfect timing, drops never-used candidates, prec 1.000) applied to
the literal single-candidate-per-request stream the tuned bar already uses. It came in BELOW the
bar on both traces. Perfect timing on what's already offered cannot win. That forced the question:
is the corridor an *object-coverage* problem instead — does the predictor simply not name enough
candidates, often enough, for anything to schedule? F5 tested this directly and confirmed yes.

### 3.4 F5 — confirmed, both traces (`p4_f5.py`, new this session)

Ceiling construction: a use of object X at t is "causally coverable" iff the wide predictor emitted
X since its previous use; F5 = clairvoyant JIT prefetch restricted to covering only coverable uses,
at the bar's byte rate. wiki needed a wider net (top_m 16→32, k up to 32) to clear the pre-registered
≥8-pt gate cleanly — this was the fork's own pre-registered escape ("widen the net"), not a moved
goalpost; coverage was still climbing at k=16 (53.6%→74.1% across the sweep), confirming the k=16
straddle was an implementation cap, not a property of the hypothesis. **Both traces now clear.**

### 3.5 Policy 1 — dead, but the audit trail matters (`p4_policy1.py`, new this session)

First build: a wide-emission bandwidth market (emit wide, value by raw Markov confidence, fund
highest value-per-byte first under a shared token bucket, no DEFER since §3.2 killed it). Real-trace
results were confusing and required **two rounds of self-caught bugs** before the verdict could be
trusted — this is exactly the "is the gate broken?" discipline working as intended:

1. **Speed bug**: unbounded pending-candidate pool re-sorted every request → crawled for 4+ hours.
   Fixed: bounded pool, kept by highest raw VALUE (not value/size — capping by value/size
   systematically evicted large useful objects, causing a 4.5M-prefetch flood at prec 0.10 in one
   run before the fix).
2. **Confound bug (the important one)**: the diagnostic `oracle` forecaster arm ranked candidates by
   `value/size` — the SAME dispatch rule as every causal arm — so "even a perfect forecaster loses"
   could not distinguish *at-emission timing* from *the dispatch rule* from *forecaster quality*.
   Fixed: the oracle arm now ranks by imminence (soonest-first) — F5's OWN selection rule — so
   oracle-vs-F5 isolates JIT-vs-at-emission cleanly, and oracle-vs-markov isolates forecaster
   quality under a fixed rule. Also added BAR-CAP (bar under the same rate token bucket the market
   faces) — reproduced HJS-L's exact protocol-cost numbers (wiki −8.58, cluster50 −6.40) as a
   cross-check that the fairness control itself is correct.

**Clean, audited result:** with F5's identical selection rule, a perfect forecaster fetching AT
EMISSION loses ~10–15 points to the same perfect forecaster inserting JUST-IN-TIME, in every
config (rate/total × both traces). At-emission captures ≤6% of F5's corridor. **This isolates JIT
insertion as the one missing ingredient — cleanly, not by inference.** Policy 2 (a better
forecaster) is provably not the answer here: even a perfect forecaster fails at emission.

### 3.6 F7 — the decisive gate, running now (`p4_f7.py`, new this session)

Isolates *only* timing accuracy: keeps F5's clairvoyant coverage AND selection (funds only objects
that will actually be used again — the "which" is guaranteed right), varies only the wake clock:
`oracle` (true next-use = the JIT ceiling / construction check), `model` (causal hazard use-lag
model from F2), `point` (median-only diagnostic). Autopsies on-time vs LATE.

**Pre-registered gate: `model` captures ≥50% of the JIT ceiling (oracle arm), CI excluding zero,
swept over `--gamma {0.1, 0.25, 0.5}`.**
- **PASS** → causal JIT is viable → build the full DEFER-for-wide policy → **the paper gets a
  positive capture method.**
- **FAIL** → the corridor needs clairvoyant timing; no causal policy captures it → **the paper's
  second half becomes the honest triple-kill** (RL unnecessary / at-emission insufficient / causal
  JIT unreachable), carried by the measurement half.

**Selftest status:** construction validated on SYNTH (oracle arm reproduces a strong JIT ceiling,
autopsy works, no crash). Two SYNTH-only caveats already handled in the reporting code: (1) F7-oracle
need not equal F5 exactly (different object-selection detail) — the `model` arm is scored against
F7's OWN oracle arm, not F5, for a clean same-mechanics comparison; (2) SYNTH's tiny cache has
`S(100)=0.020` (objects die almost immediately), which manufactures an early-eviction failure mode
that will NOT occur on the real traces (`S(100)=1.000`) — so on real data the only failure mode to
worry about is LATE (overshoot), the true test of causal timing accuracy.

**Live signal, in progress (not yet final):** on wiki at `gamma=0.1` (the most conservative,
earliest-wake setting), the `oracle` arm nailed the construction check (99.8% capture of F5,
100% on-time) — the harness is solid. The `model` arm's LATE rate was climbing fast (74%→92% by the
50% mark) even at this most-forgiving gamma, which is a bad sign: higher gammas in the sweep target
*later* wake quantiles and should only get worse, not better. **Not yet confirmed as a final
verdict** — wait for the full sweep and cluster50.

**The fair-shot fallback, queued, not yet run, if `model` fails the gate:** the hazard model
(`haz/wiki.npz`, `haz/cluster50.npz`) was trained on the **narrow k=1 stream**. F7 evaluates it on
the **wide, tau=0.0 stream** — dominated by long-tail, low-confidence candidates the model's
`(recency, freq, conf)` bins were never well-conditioned on. Before declaring causal timing dead,
**retrain the hazard model on the wide-emission stream** (same k/top_m/tau as F5/Policy1/F7 use) and
re-run F7's `model` arm against that. This is the single "is the gate broken?" check the house
rules require before trusting a kill — do it once, then accept the answer either way.

```bash
# regenerate hazard models if stale
python p4_hazard.py --trace data/wiki_2019t.oracleGeneral        --pred markov2 --tau 0.05 --k 1 --out haz/wiki.npz
python p4_hazard.py --trace data/cluster50.sample10.oracleGeneral --pred markov3 --tau 0.06 --k 1 --out haz/cluster50.npz

# the gate (in flight / to finish)
for g in 0.1 0.25 0.5; do
  python -u p4_f7.py --trace data/wiki_2019t.oracleGeneral --pred markov2 --tau 0.05 --k 1 \
      --wide-k 32 --wide-top-m 32 --haz haz/wiki.npz --gamma $g --verbose 2>&1 | tee logs/f7_wiki_g$g.log
  python -u p4_f7.py --trace data/cluster50.sample10.oracleGeneral --pred markov3 --tau 0.06 --k 1 \
      --wide-k 16 --haz haz/cluster50.npz --gamma $g --verbose 2>&1 | tee logs/f7_c50_g$g.log
done
```

Use `--verbose` (clean, flush-safe, parallel-friendly progress with ETA and live LATE rate), not
`--tqdm` (flickers badly across parallel runs — learned the hard way this session).

---

## 4. What to do next, in order

1. **Let the F7 gamma sweep finish** on both wiki and cluster50 (6 runs total). Read, per run: the
   `CONSTRUCTION` line (oracle should be ~100% of F5 — sanity), then `CAUSAL TIMING model = X% of
   the JIT ceiling | on-time Y%`, then the `GATE` verdict line.
2. **If `model` fails everywhere**, run the fair-shot fallback (retrain hazard on the wide stream,
   §3.6) once before accepting the negative.
3. **Branch on the final verdict:**
   - PASS anywhere → design and build the full DEFER-for-wide policy (reuse `p4_f7.py`'s
     `CausalTimedCover` machinery, generalize the clairvoyant selection to a causal one) — this
     becomes the paper's positive method section.
   - FAIL even after the fallback → write up the negative: three independent, pre-registered kills
     (RL arbitration, at-emission dispatch, causal JIT timing) converging on "the corridor is real
     but only clairvoyantly reachable" — pair this with §2's measurement result as the combined
     paper's full arc.
4. **Update `results.md`** (the capture-track ledger; distinct from `results_p4.md`, the finished
   measurement ledger) with the final F7 numbers and whichever branch is taken.
5. Only then start writing the combined paper draft — do not draft prose around an unresolved gate.

---

## 5. House rules this project has earned the hard way (do not relitigate these)

1. **Every gate is pre-registered before the run that could satisfy it.** Thresholds don't move
   after seeing a number, in either direction.
2. **A gate that never kills anything is not trusted.** The growing, visible trail of self-caught
   bugs is the credibility argument, not an embarrassment: the synthetic +42 that became −0.02; the
   cluster26 flip; the retracted cold-split v1; the F1 vocab-cap bug; the F4 SEP-ordering bug; the
   hazard train/eval overlap bug; HJS-L's quantile-midpoint bug; Policy 1's pool-cap value/size bias
   (4.5M-prefetch flood); Policy 1's confounded oracle arm (fixed to isolate JIT-vs-dispatch-rule
   cleanly). Every one caught before it reached a paper.
3. **When a gate fails, ask "is the gate broken?" before "is the hypothesis dead?"** — but only
   once per gate, with a specific, falsifiable fix. (F7's queued fallback — retrain hazard on the
   wide stream — is this check, already scoped in advance so it can't be moved after seeing the
   number.)
4. **No n=1 generalization.** Both live traces, not one, before any capture claim.
5. **Report diagnostics as diagnostics.** `oracle` forecaster/timing arms read the future and exist
   only to bound what's possible — never a reportable capture number.
6. **Fairness controls are not optional decoration.** BAR-CAP (bar under the same token bucket the
   scheduler faces) must accompany any rate-mode comparison, or protocol cost masquerades as a
   scheduling failure (or a scheduling win masquerades as protocol-fairness).
7. **One combined paper, per §0.** Don't re-fork this into two submissions without the user
   re-deciding that explicitly.
