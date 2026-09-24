# Opportunity Certification

**Deciding Whether a Learned Component Is Warranted, Before Training One**

Replay code for every arm the paper reports. The paper's claim is a procedure, not a system: given
an apparent gap between a deployed heuristic and an oracle, decide by replay — before training a
model — whether the gap is real (Stage 1, the *Instrument*) and whether it is reachable
(Stage 2, the *Necessity Ladder*). The output is a certificate: **no corridor**, a **trap**, or a
**build**.

Nothing here trains a model to produce a verdict. The two trained policies that do appear
(`oppcert/domains/substrate_control.py`, and the `r6` model arm) exist to bound the procedure's
false-negative rate, which is the error nobody otherwise observes.

## Check it runs, without downloading anything

Every arm carries a self-test on synthetic data. All ten pass on a laptop; the slowest is 90 s.

```bash
pip install -r requirements.txt
python -m oppcert.sim.invariants --synth --limit 200000 --seq-frac 0.3   # all gates PASS
python -m oppcert.sim.invariants --synth --limit 200000 --seq-frac 0.0   # W2b FAILS (negative control)
python -m oppcert.domains.learned_index --n 200000 --dists uniform       # the positive domain, synthetic keys
python -m oppcert.ladder.r6_placement --selftest                         # the ladder's decisive rung
```

The negative control matters: the same gate that passes on a trace with successor structure has to
fail on one without it, or it is not measuring anything.

## Layout

```
oppcert/
  sim/            replay substrate: trace IO, cache sim, evictors, byte-budget meter,
                  block-bootstrap CIs, and the machine-checked invariants
  instrument/     Stage 1: the tuned bar family, Gate A, Gate B, the cold-miss restriction
  ladder/         Stage 2: rungs r1-r7, the break-even precision frontier, survival
  domains/        the procedure transferred unchanged: KV-cache serving, learned indexes,
                  and the substrate-competence control
  unreported/     arms that were run but are NOT in the paper -- see its README
prereg/           pre-registrations, written before the runs they govern
paper/            LaTeX source (AAAI template), bibliography, figure sources
scripts/          trace fetch and the multi-trace drivers
logs/             raw output of selected runs
docs/             the results ledger, the acceptance checklist, and archived drafts
```

Modules run as modules, from the repository root:

```bash
python -m oppcert.instrument.gates --trace data/wiki_2019t.oracleGeneral --gate a --limit 2000000
```

Every module's docstring opens with a `PAPER:` line naming the claim it produces, and `--help`
documents its flags.

## Data

Public, no credentials. `scripts/fetch_traces.sh` streams **2M-request prefixes** (~48 MB each)
straight out of S3 rather than downloading whole traces, so a 27 GB trace costs the same as an
80 MB one — the prefix is exactly what `--limit 2000000` would have used anyway.

```bash
pip install awscli zstandard
scripts/fetch_traces.sh              # the twelve Gate A traces into data/
```

The twelve: `wiki_2019t`, `meta_rprn`, `meta_reag` (CDN); `cluster50`, `cluster53`, `cluster26`,
`cluster10` (key-value); `msr_proj_0`, `msr_hm_0`, `msr_web_2`, `w105`, `w87` (block). The other two
domains need their own inputs:

```bash
curl -L -o data/mooncake_conv.jsonl \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl
# learned indexes: books / osm_cellids / fb key sets from the SOSD benchmark, into data/
```

## Reproducing the paper

Two operating points recur: **wiki** is `--pred markov2 --tau 0.05 --k 1`, **cluster50** is
`--pred markov3 --tau 0.06 --k 1`. Both run at a 1% cache over a 2M-request prefix.

### Stage 1 — the Instrument

| paper | command | what it returns |
|---|---|---|
| Table 1, Gate A on one trace | `python -m oppcert.instrument.gates --trace data/wiki_2019t.oracleGeneral --gate a --limit 2000000` | tuned baseline, gross corridor, verdict |
| Table 1, all twelve traces | `scripts/run_gate_a.sh` | one log per trace + the breadth summary |
| Table 2, the 14–87% deflation | `python -m oppcert.instrument.cold_split --trace data/wiki_2019t.oracleGeneral --limit 2000000 --pred markov2 --tau 0.06 --k 1` | gross → cold → learnable corridor |
| §4, the LSTM never recovers a corridor | `scripts/run_lstm_bar.sh` | trains the LSTM bar, re-runs Gate A against it (needs torch) |
| §4, eviction and prefetching are substitutes | `scripts/run_gate_b.sh` | the 2×2 on six traces, all negative |
| every bracketed interval | `python -m oppcert.sim.bootstrap --trace ... --pred markov2 --tau 0.06 --k 1` | paired block-bootstrap 95% CI |
| the machine-checked invariants | `python -m oppcert.sim.invariants --trace data/wiki_2019t.oracleGeneral` | loader self-check, LRU/S3-FIFO parity, headroom gates |
| §7, the excluded class | `python -m oppcert.instrument.online_bars --leaktest` | leakage invariants for the online-updating bars |

### Stage 2 — the Necessity Ladder

Each rung grants one perfect capability and prices it. Figures below are wiki / cluster50.

| rung | command (wiki config) | paper |
|---|---|---|
| r1 timing ceiling | `python -m oppcert.ladder.r1_r3_timing --trace data/wiki_2019t.oracleGeneral --pred markov2 --tau 0.05 --k 1 --mode vocab` | recovers 95.8% / 91.2% of the corridor |
| r2 arbitration | `python -m oppcert.ladder.r2_arbitration --trace ... --mode est` | +0.01 / −0.01, against a 2-point bar |
| r3 offered stream | `python -m oppcert.ladder.r1_r3_timing --trace ... --mode stream` | −7.01 / −10.82 |
| r4 coverage | `python -m oppcert.ladder.r4_coverage --trace ... --pred markov2 --tau 0.05 --k 1` | +10.41 / +11.42 — reachable in principle |
| r5 emission-time fetch | `python -m oppcert.ladder.r5_emission --trace ...` | +0.62 / −3.07, ≤6% of the corridor |
| r6 causal placement | `python -m oppcert.ladder.r6_hazard_model --trace ... --out haz/wiki.npz` then `python -m oppcert.ladder.r6_placement --trace ... --wide-k 32 --wide-top-m 32 --haz haz/wiki.npz` | oracle arm +9.69 / +11.69 (construction check); model arm −40.06 / −26.01 |
| r6 robustness | `python -m oppcert.ladder.r6_sweep --trace ... --haz haz/wiki.npz --sweep volume` | retrain to ρ 0.68 captures nothing further |
| r7 CGP | `python -m oppcert.ladder.r7_cgp --trace ... --wide-k 32 --wide-top-m 32 --f5-ref 10.41 --verbose` | −7.60 / −5.39; +0.98 / +1.01 capped-baseline |
| break-even precision | `python -m oppcert.ladder.breakeven_precision --trace ... --pred markov2 --tau 0.05 --k 1` | p\* = 0.582 against the 0.062 achieved |
| survival (suppl. §C) | `python -m oppcert.ladder.survival --trace ... --haz haz/wiki.npz --wide-k 32 --gamma 0.1` | S(100) = 0.984 against the baseline's 1.000 |

### The procedure across domains

| verdict | command | paper |
|---|---|---|
| **no corridor** — LLM KV-cache serving | `python -u -m oppcert.domains.kvcache --trace data/mooncake_conv.jsonl --verbose` | −0.56 / −1.87; +2.98 / +2.14 capped, under an 8-point gate |
| **build** — learned indexes | `python -m oppcert.domains.learned_index --keys-from-sosd data/books --n 2000000` | Table 4: build on 1 of 5 real key sets |
| substrate-competence control | `python -m oppcert.domains.substrate_control --trace data/wiki_2019t.oracleGeneral --limit 2000000` | eviction corridors +11.55 / +12.50 / +12.75, learners capture essentially none |

## Paper names and internal names

The code was written before the paper's rung numbering settled, and the older names survive inside
some docstrings and in `docs/results_ledger.md`. The mapping:

| paper | internal | module |
|---|---|---|
| r1 / r3 | F1-vocab / F1-stream | `oppcert/ladder/r1_r3_timing.py` |
| r2 | F4, F4′ | `oppcert/ladder/r2_arbitration.py` |
| r4 | F5 | `oppcert/ladder/r4_coverage.py` |
| r5 | Policy 1 | `oppcert/ladder/r5_emission.py` |
| r6 | F7 | `oppcert/ladder/r6_placement.py` |
| r6 robustness | r8 | `oppcert/ladder/r6_sweep.py` |
| r7 | CGP | `oppcert/ladder/r7_cgp.py` |
| break-even precision | r9 | `oppcert/ladder/breakeven_precision.py` |
| Gate A / Gate B | — | `oppcert/instrument/gates.py` |

## Pre-registration

The paper's thresholds were fixed before the runs they govern, and `prereg/` is the record:
`r1_r2_oracles.md`, `r7_cgp.md`, `online_predictor.md`. Each states its bars, its invariants, and
what outcome would have falsified it. `docs/results_ledger.md` is the run-by-run ledger, including
the arms that were retracted and why.

## Status

- **Traces are not committed** and never will be; `data/` is ignored. Everything is fetchable with
  the commands above.
- **libCacheSim parity is provisional unless you install it.** Without `libcachesim`, the parity
  gate prints `SKIPPED`, the pass is labelled provisional, and the independent reference LRU is the
  only cross-check. It never silently passes.
- **`logs/` is partial** — selected runs, not the full record. The ledger in `docs/` is the
  complete account.
- **Figures are `.drawio` sources.** The built PDFs are not committed, so a local LaTeX build shows
  a placeholder box where each figure goes.
- **No licence file yet.** Nothing here may be reused until one is added.
- `oppcert/unreported/` holds arms the paper does not report, labelled as such.

## Data and tools this work depends on

| | |
|---|---|
| Cache traces (all twelve) | [cacheMon/cache_dataset](https://github.com/cacheMon/cache_dataset) — already in `oracleGeneral` format, so Belady is free |
| Simulator parity oracle | [libCacheSim](https://github.com/1a1a11a/libCacheSim) — independent LRU and S3-FIFO hit counts |
| Learned-index key sets | [SOSD](https://github.com/learnedsystems/SOSD) — `books`, `osm_cellids`, `fb` |
| KV-cache traces | [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) — FAST'25 conversation and tool/agent traces |
| Reference policies reimplemented, not run | [LRB](https://github.com/sunnyszy/lrb), [Baleen](https://github.com/wonglkd/BCacheSim) |
