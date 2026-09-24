#!/usr/bin/env bash
# Day 1 -- THE kill-test: does the corridor survive an LSTM decoupled bar?
# Runs train+dump+sweep per trace, sequentially. Default targets: the live headline
# (wiki), the marginal KV trace (cluster26), and the first metaCDN trace present.
# Usage: scripts/run_lstm_bar.sh [trace ...]   (paths under data/, no args = default set)
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p preds logs

# Default set = the strongest LIVE trace per family after the cluster26 flip:
# wiki (CDN headline), cluster50 (KV), meta_rprn (CDN/Meta). cluster26 is DEAD
# (strong bar, 2026-07-18) -- do not spend LSTM compute on it.
TRACES=("$@")
if [[ ${#TRACES[@]} -eq 0 ]]; then
  TRACES=(data/wiki_2019t.oracleGeneral)
  [[ -s data/cluster50.sample10.oracleGeneral ]] && TRACES+=(data/cluster50.sample10.oracleGeneral)
  [[ -s data/meta_rprn.oracleGeneral ]] && TRACES+=(data/meta_rprn.oracleGeneral)
fi

for f in "${TRACES[@]}"; do
  n=$(basename "$f" .oracleGeneral); n=${n%.sample10}
  echo "=================================================================== LSTM BAR: $n"
  python -m oppcert.instrument.lstm_bar --trace "$f" --limit 2000000 \
      --out "preds/${n}.lstm.npz" 2>&1 | tee "logs/lstm_train_${n}.log"
  python -m oppcert.instrument.bars --trace "$f" --limit 2000000 \
      --lstm-preds "preds/${n}.lstm.npz" 2>&1 | tee "logs/strongbar_${n}.log"
done

echo; echo "==================== STRONG-BAR SUMMARY (aliveness verdicts) ===================="
for f in logs/strongbar_*.log; do
  n=$(basename "$f" .log); n=${n#strongbar_}
  printf "%-20s %s\n" "$n" "$(grep 'TUNED A1 BAR' "$f" | tail -1 | sed 's/  */ /g')"
  printf "%-20s %s\n" ""   "$(grep 'TIMING CORRIDOR' "$f" | tail -1 | sed 's/  */ /g')"
  printf "%-20s %s\n\n" "" "$(grep 'VERDICT' "$f" | tail -1 | sed 's/  */ /g')"
done | tee logs/strongbar_summary.txt
