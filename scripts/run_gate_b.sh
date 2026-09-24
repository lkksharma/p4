#!/usr/bin/env bash
# Gate B -- eviction x prefetch INTERACTION 2x2, across families, in ONE process so the
# "substitutes, not complements -- structural or n=1?" aggregate summary is emitted.
# Runs --gate b only (no gate_a). Safe to run in parallel with the cold splits / cap sweep;
# it's CPU and doesn't touch their logs.
#
# Trace set = a spread across all three families so the interaction claim is n>1 by design:
#   block: msr_proj_0, msr_hm_0   CDN: wiki_2019t, meta_rprn   KV: cluster50, cluster53
# (Belady is the oracle-eviction column; on variable-size CDN/KV objects it is a STRONG
#  heuristic, not proven OPT -- the writeup must say so. Only MSR's uniform blocks earn "optimal".)
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

python -m oppcert.instrument.gates --gate b --limit 2000000 \
  --trace data/msr_proj_0.oracleGeneral \
  --trace data/msr_hm_0.oracleGeneral \
  --trace data/wiki_2019t.oracleGeneral \
  --trace data/meta_rprn.oracleGeneral \
  --trace data/cluster50.sample10.oracleGeneral \
  --trace data/cluster53.sample10.oracleGeneral \
  2>&1 | tee logs/gate_b_live.log

echo; echo "==================== GATE B INTERACTION SUMMARY ===================="
grep -E "INTERACTION|SUBSTITUTES|COMPLEMENTS|structural|MIXED|pts " logs/gate_b_live.log | tail -20
