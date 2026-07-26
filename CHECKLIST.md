# P4 — Consolidated Checklist (what does what · what's running · what's left)

*Built 2026-07-20 by reading every script on this laptop. Companion to `context.md` (the narrative)
and `results.md` (the capture ledger) / `results_p4.md` (the measurement ledger). All heavy outputs
(`.log`, `haz/*.npz`, `preds/*.npz`, `data/*.oracleGeneral`) live on the **H100 DGX node**, NOT here
— this laptop holds only the scripts, so every "read the log" step below happens on the node.*

---

## PART A — What each script does

### A.0 Foundation (env + accounting — the trust floor)

| file | role | status |
|---|---|---|
| `p4_cache.py` | Trace IO (libCacheSim 24B `oracleGeneral`), deterministic byte-cache (LRU/FIFO/LFU/Belady), `synth_trace`, `footprint_bytes`, `verify_trace_format`. | ✅ stable |
| `p4_evict.py` | Pluggable evictors: `LRUEvictor`, `BeladyEvictor`, `S3FIFOEvictor` (SOSP'23). Evictor owns ORDER; `PFCache` owns bytes. | ✅ stable |
| `p4_prefetch.py` | `PFCache` (prefetch-aware sim + token-bucket bandwidth matching + survival log + hit-series for bootstrap); prefetchers `Markov1`, `Markov2`, `Prescient` (oracle, vocab-restrictable); `oracle_next`, `build_obj_positions/sizes`. The engine every rung runs on. | ✅ stable |
| `p4_sweep.py` | `prep()` (load+cache-size), **Gate A** (tuned bar = best decoupled predictor under 1.15× traffic; iso-BW learnable ceiling), **Gate B** (eviction×prefetch interaction 2×2). Defines `KS=(1,2,4)`, `TAUS`. | ✅ stable |
| `p4_gates.py` | W1/W2 kill-tests: FORMAT, PARITY (our LRU/S3-FIFO vs independent + libCacheSim), GAP (LRU→Belady), PREFETCH SIGNAL. "No gap, no paper." | ✅ stable |
| `p4_bootstrap.py` | Paired moving-block bootstrap CIs for the measurement corridor. *(not re-read this pass — CI convention mirrored inline in every rung)* | ✅ stable |
| `p4_t1.py` | Early T1 2×2 skeleton (LRU / +Markov / Belady / Prescient). Superseded by `p4_sweep` Gate B. | 🗄️ legacy |

### A.1 Measurement paper — "Phantom Headroom" (FIRST half, DONE)

| file | role | status |
|---|---|---|
| `p4_strongbar.py` | Runs the STRONGER decoupled bars (`Markov3` PPM + optional `LSTMTopK`) through Gate A. Bar = max over {M1,M2,M3,LSTM}×tau×k — can only RAISE the bar. This is the adversarial baseline the corridor must survive. | ✅ done |
| `p4_coldsplit.py` | Cold-miss decomposition: gross corridor → cold slice (compulsory, clairvoyant-only) → **learnable corridor** (warm/in-vocab ceiling). `PREDS = {markov1,markov2,markov3}`. | ✅ done |
| `p4_lstm_train.py` | GPU LSTM next-item trainer → dumps `preds/*.npz` for `strongbar --lstm-preds`. *(GPU-only; not re-read this pass)* | ✅ done |

**Result (bootstrap-confirmed, submission-ready):** learnable corridor ≥ 8 pts + Pareto dominance on
`{wiki_2019t +12.72, cluster50 +17.56, cluster53 +19.67}`; eviction & prefetch are **substitutes**
(kills joint co-optimization). Deliverables: `paper_draft.md`, `paper.tex/.pdf` (10pp, 3 figs, 3
tables), `results_p4.md`.

### A.2 Capture ladder — "can any causal policy capture the corridor?" (SECOND half)

| rung | file | what it isolates | verdict |
|---|---|---|---|
| **F1** | `p4_f1.py` | timing-only oracle on the predictor's own candidates (`vocab` = valid ceiling; `stream` = JIT deferral diagnostic). | ✅ trustworthy — wiki 95.8%, c50 91.2% of corridor; c53 dropped |
| **F4/F4'** | `p4_f4.py` | separable threshold rule vs joint knapsack arbitration (`clair` control ~0; `est` = the RL gate). | ✅ **RL unnecessary** (<1 pt everywhere). Closed. |
| **F2** | `p4_hazard.py` | is causal use-lag learnable? Builds hazard model `p(lag \| recency,freq,conf)` + KM survival curve → saves `haz/*.npz`. Three-way split (predictor/hazard/eval). | ✅ PASS post-fix (Spearman .60–.65, AUC .81–.95) — but `[CARRIED]`, feeds F7 |
| **HJS-L** | `p4_hjsl.py` | closed-form hazard-priced DEFER scheduler. *(not re-read this pass)* | ❌ **DEAD** — `S(100)=1.0` ⇒ no eviction race; DEFER only ties/loses |
| **F1-stream** | `p4_f1.py --mode stream` | perfect timing on the actual k=1 stream. | ❌ below bar (wiki −7.01, c50 −10.82) → corridor is COVERAGE, not timing |
| **F5** | `p4_f5.py` | coverage ceiling: wide emission + clairvoyant JIT + clairvoyant select. `build_coverable`, `CoverGatedPrescient`. | ✅ **CLEARS 8 both traces** (wiki +10.41 @k32, c50 +11.42 @k16) |
| **Policy 1** | `p4_policy1.py` | causal wide-emission bandwidth market, AT EMISSION (no defer). `WideMarket`; forecasters markov/blend/oracle. | ❌ **DEAD, audited** — perfect forecaster at emission captures ≤6% of F5 → JIT insertion is the missing ingredient |
| **F7** | `p4_f7.py` | **the decisive gate.** F5's clairvoyant coverage+selection, vary ONLY the wake clock: `oracle`(=JIT ceiling), `model`(causal hazard lag), `point`(median). Isolates causal-timing ACCURACY. `CausalTimedCover`. | 🔧 **built, selftest passes, running on H100 — verdict UNRESOLVED** |
| **Policy 2** | *(not built)* | recall-first forecaster (`blend` v0 wired in Policy 1). | ⏸️ gated on F7 branch |

---

## PART B — Status board

### ✅ DONE
- Env + accounting + all kill-tests (`p4_cache/evict/prefetch/sweep/gates`).
- Measurement paper fully proven & drafted (see A.1). Pending only citation verification.
- Ladder kills, all pre-registered & audited: **RL unnecessary** (F4'), **DEFER dead** (HJS-L),
  **timing-not-the-gap** (F1-stream), **at-emission insufficient** (Policy 1).
- **F5 coverage lever confirmed on both live traces** — the corridor IS reachable in principle.

### 🔧 IN PROGRESS (on the H100 — needs log readback)
- **F7 JIT-realizability gate.** Code built, `CausalTimedCover` heap-growth bug fixed (HEAD commit
  `64197d4`), selftest passes on SYNTH. Pre-registered gate: **`model` captures ≥50% of the JIT
  ceiling (oracle arm), CI>0, swept over gamma {0.1,0.25,0.5}, both traces.**
  - Live signal so far (per `context.md`): wiki gamma=0.1 oracle arm nailed construction (99.8%,
    100% on-time) — harness solid; but `model` LATE rate climbed 74%→92% by the halfway mark →
    **looks like a likely FAIL**, not yet confirmed across gamma or on cluster50.

### ⬜ LEFT TO DO (in order)
1. **Regenerate hazard models on the node** if stale (`haz/wiki.npz`, `haz/cluster50.npz`).
2. **Finish the F7 gamma sweep** — 6 runs (wiki×3, c50×3). Per run read: `CONSTRUCTION` line
   (oracle ~100% of F5 = sanity), `CAUSAL TIMING model = X% of ceiling | on-time Y%`, `GATE` verdict.
3. **If `model` fails everywhere → run the one pre-registered fair-shot fallback ONCE:** retrain the
   hazard model on the **wide (tau=0.0, k=32/16) emission stream** (it was trained on k=1), re-run
   F7's `model` arm. This is the "is the gate broken?" check — do it once, then accept the answer.
4. **Branch on the final F7 verdict:**
   - **PASS** → build the full DEFER-for-wide policy (reuse `CausalTimedCover`, make selection
     causal) → the paper gets a **positive capture method**.
   - **FAIL** (even after fallback) → write the **honest triple-kill negative** (RL / at-emission /
     causal-JIT), carried by the measurement half.
5. **Update `results.md`** with final F7 numbers + branch taken (only `[LOG]`/`[PASTE]` numbers).
6. **Verify F2 `[CARRIED]` numbers** against current code (low priority — only matters if the
   positive branch is taken, since F7 depends on the hazard model).
7. **Write the UNIFIED paper** (measurement + capture) — only after the gate resolves. Do NOT draft
   prose around an unresolved gate. Current drafts (`paper.tex`) cover the measurement half only.

---

## PART C — The exact commands left (run on the H100)

```bash
# 1) (re)build hazard models if stale — trained on the k=1 stream
python p4_hazard.py --trace data/wiki_2019t.oracleGeneral        --pred markov2 --tau 0.05 --k 1 --out haz/wiki.npz
python p4_hazard.py --trace data/cluster50.sample10.oracleGeneral --pred markov3 --tau 0.06 --k 1 --out haz/cluster50.npz

# 2) the decisive gate — full gamma sweep, both traces (use --verbose, NOT --tqdm, for parallel runs)
for g in 0.1 0.25 0.5; do
  python -u p4_f7.py --trace data/wiki_2019t.oracleGeneral --pred markov2 --tau 0.05 --k 1 \
      --wide-k 32 --wide-top-m 32 --haz haz/wiki.npz --gamma $g --verbose 2>&1 | tee logs/f7_wiki_g$g.log
  python -u p4_f7.py --trace data/cluster50.sample10.oracleGeneral --pred markov3 --tau 0.06 --k 1 \
      --wide-k 16 --haz haz/cluster50.npz --gamma $g --verbose 2>&1 | tee logs/f7_c50_g$g.log
done

# 3) ONLY if step 2's `model` arm fails everywhere — the one fair-shot fallback:
#    retrain hazard on the WIDE emission stream, then re-run F7's model arm against it.
```

**Config asymmetry to respect (house rule):** wiki's coverage lever clears F5 at **k=32/top_m=32**,
cluster50's at **k=16** — F7 must use each trace's own F5-clearing width (baked into the commands above).
