#!/usr/bin/env bash
# Day 1 -- THE kill-test: does the corridor survive an LSTM decoupled bar?
# Runs train+dump+sweep per trace, sequentially. Default targets: the live headline
# (wiki), the marginal KV trace (cluster26), and the first metaCDN trace present.
# Usage: scripts/day1_lstm_bar.sh [trace ...]   (paths under data/, no args = default set)
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p preds logs

TRACES=("$@")
if [[ ${#TRACES[@]} -eq 0 ]]; then
  TRACES=(data/wiki_2019t.oracleGeneral data/cluster26.oracleGeneral.sample10)
  meta=$(ls data/ | grep -i -m1 "meta" || true)
  [[ -n "$meta" ]] && TRACES+=("data/$meta")
fi

for f in "${TRACES[@]}"; do
  n=$(basename "$f" .oracleGeneral); n=${n%.sample10}
  echo "=================================================================== LSTM BAR: $n"
  python p4_lstm_train.py --trace "$f" --limit 2000000 \
      --out "preds/${n}.lstm.npz" 2>&1 | tee "logs/lstm_train_${n}.log"
  python p4_strongbar.py --trace "$f" --limit 2000000 \
      --lstm-preds "preds/${n}.lstm.npz" 2>&1 | tee "logs/strongbar_${n}.log"
done

echo; echo "==================== STRONG-BAR SUMMARY (aliveness verdicts) ===================="
for f in logs/strongbar_*.log; do
  n=$(basename "$f" .log); n=${n#strongbar_}
  printf "%-20s %s\n" "$n" "$(grep 'TUNED A1 BAR' "$f" | tail -1 | sed 's/  */ /g')"
  printf "%-20s %s\n" ""   "$(grep 'TIMING CORRIDOR' "$f" | tail -1 | sed 's/  */ /g')"
  printf "%-20s %s\n\n" "" "$(grep 'VERDICT' "$f" | tail -1 | sed 's/  */ /g')"
done | tee logs/strongbar_summary.txt
