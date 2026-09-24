# Unreported arms

Nothing in this directory backs a claim in the paper. It ships because the paper's own recipe
asks for "reporting every arm that the escalation kills", and an artifact that shows only the
arms that made the cut is the same selective reporting the paper argues against.

Read these as provenance, not as results.

| module | what it is | why it is not in the paper |
|---|---|---|
| `hjsl_scheduler.py` | HJS-L, a closed-form hazard-priced prefetch scheduler that reads the `r6_hazard_model` use-lag distribution and the measured eviction-survival curve | Dead arm. The DEFER lever it turns does not move the corridor; `r6` supersedes it as the causal-placement rung. Ledger: [`docs/results_ledger.md`](../../docs/results_ledger.md) §4 |
| `query_optimization.py` | Opportunity Certification applied to the Join Order Benchmark over IMDB, under PostgreSQL | A fifth domain, outside the four settings the paper certifies. Pre-registered in [`prereg/unreported_query_optimization.md`](../../prereg/unreported_query_optimization.md); database setup in [`scripts/unreported/`](../../scripts/unreported) |
| `t1_mechanism.py` | Early probe asking whether hits exist outside Belady's decision space and whether a realistic prefetcher converts them | Predates the Instrument. Its question is answered properly by Gate A and Gate B in `oppcert/instrument/gates.py` |

Each runs its own self-test without a trace or a database:

```bash
python -m oppcert.unreported.hjsl_scheduler --selftest
python -m oppcert.unreported.query_optimization --selftest
python -m oppcert.unreported.t1_mechanism --synth
```
