#!/usr/bin/env bash
# Day 0 -- Gate A (tuned Markov bar + iso-BW ceiling) on EVERY trace in data/.
# Resumable: a trace with a completed log (contains VERDICT) is skipped, so you can
# re-run this script after adding traces or after an interruption.
# Output: logs/gatea_<trace>.log per trace + logs/breadth_summary.txt at the end.
set -uo pipefail
shopt -s nullglob
cd "$(dirname "$0")/.."
mkdir -p logs

for f in data/*.oracleGeneral data/*.oracleGeneral.sample10; do
  n=$(basename "$f" .oracleGeneral)
  log="logs/gatea_${n}.log"
  if [[ -s "$log" ]] && grep -q "VERDICT" "$log"; then
    echo "[skip] $n already done"; continue
  fi
  echo "=================================================================== GATE A: $n"
  python p4_sweep.py --trace "$f" --gate a --limit 2000000 2>&1 | tee "$log"
done

echo
echo "==================== BREADTH SUMMARY (the main-result table) ===================="
{
  printf "%-24s | %s\n" "trace" "bar / corridor / verdict"
  echo "-------------------------------------------------------------------------------"
  for f in logs/gatea_*.log; do
    n=$(basename "$f" .log); n=${n#gatea_}
    bar=$(grep "TUNED A1 BAR"    "$f" | tail -1 | sed 's/  */ /g')
    cor=$(grep "TIMING CORRIDOR" "$f" | tail -1 | sed 's/  */ /g' | sed 's/\[pre-registered.*//')
    ver=$(grep "VERDICT"         "$f" | tail -1 | sed 's/  */ /g')
    printf "%-24s |%s\n%-24s |%s\n%-24s |%s\n" "$n" "$bar" "" "$cor" "" "$ver"
    echo "-------------------------------------------------------------------------------"
  done
} | tee logs/breadth_summary.txt
