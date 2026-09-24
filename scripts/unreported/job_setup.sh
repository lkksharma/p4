#!/usr/bin/env bash
# job_setup.sh -- one-shot preparation for the JOB / IMDB instantiation (prereg/unreported_query_optimization.md).
# Idempotent: safe to re-run. Nothing here touches the pre-registered thresholds.
set -euo pipefail

DB=${DB:-imdb}
WORK=${WORK:-$PWD/jobdata}
JOB=${JOB:-$PWD/job}

echo "== 1/5 preconditions =="
if ! command -v psql >/dev/null; then
  echo "  psql not found. Run first:  bash install_postgres.sh"
  exit 1
fi
psql -V
pg_isready -q || { echo "  server not accepting connections; run: bash install_postgres.sh"; exit 1; }
python -c "import psycopg2" 2>/dev/null || pip install -q psycopg2-binary

echo "== 2/5 pg_hint_plan (the ceiling arms inject cardinalities through it) =="
# Distinguish "cannot connect" from "extension absent": conflating them sent the last run chasing
# a missing package when the real cause was a missing role.
if ! psql -d postgres -tAc "SELECT 1" >/dev/null 2>&1; then
  echo "  !! cannot connect as $(whoami). The role or database is missing, not the extension."
  echo "     Re-run:  bash install_postgres.sh   (its step 4 creates the role and database)"
  exit 1
fi
# Availability in the catalogue is NOT sufficient: an unloaded library ignores hints silently,
# which would make both ceilings equal the baseline and produce a fake NO CORRIDOR verdict.
# oppcert/unreported/query_optimization.py --setup proves efficacy with a live hint; this is only the coarse presence check.
if ! psql -d postgres -tAc \
    "SELECT 1 FROM pg_available_extensions WHERE name='pg_hint_plan'" | grep -q 1; then
  echo "  !! pg_hint_plan not installed. Run:  bash install_postgres.sh"
  exit 1
fi

echo "== 3a/5 JOB queries + schema =="
# The 113 queries, schema.sql and fkindexes.sql live in the benchmark repository, not in the
# data tarball.
if [ ! -f "$JOB/schema.sql" ]; then
  if command -v git >/dev/null; then
    rm -rf "$JOB"
    git clone --depth 1 https://github.com/gregrahn/join-order-benchmark.git "$JOB"
  else
    mkdir -p "$JOB" && cd "$JOB"
    curl -fL -o job.tar.gz \
      https://github.com/gregrahn/join-order-benchmark/archive/refs/heads/master.tar.gz
    tar xzf job.tar.gz --strip-components=1 && rm job.tar.gz
    cd - >/dev/null
  fi
fi
echo "  $(ls "$JOB"/*.sql 2>/dev/null | grep -cvE 'schema|fkindexes') queries, schema present"

echo "== 3b/5 IMDB snapshot =="
mkdir -p "$WORK" && cd "$WORK"
if ! psql -d "$DB" -tAc "SELECT to_regclass('title')" 2>/dev/null | grep -q title; then
  if [ ! -f imdb.tgz ]; then
    # The original CWI path 404s; try the maintained mirrors in turn and keep the first that works.
    ok=0
    for U in "https://event.cwi.nl/da/job/imdb.tgz" \
             "https://bonsai.cedardb.com/job/imdb.tgz" \
             "https://homepages.cwi.nl/~boncz/job/imdb.tgz"; do
      echo "  trying $U"
      if curl -fL --retry 2 -o imdb.tgz "$U"; then ok=1; break; fi
      rm -f imdb.tgz
    done
    [ "$ok" = 1 ] || { echo "  !! every mirror failed; download imdb.tgz manually into $WORK"; exit 1; }
  fi
  echo "  extracting (~3.7 GB)"
  tar xzf imdb.tgz
fi

echo "== 4/5 load =="
if ! psql -d "$DB" -tAc "SELECT to_regclass('title')" 2>/dev/null | grep -q title; then
  createdb "$DB" 2>/dev/null || true
  # schema + FK indexes ship with the JOB artifact
  psql -d "$DB" -f "$JOB/schema.sql"
  # The tarball has shipped the CSVs both at the top level and inside a directory across
  # revisions, so locate them rather than assuming a layout.
  CSVDIR=$(dirname "$(find "$WORK" -maxdepth 2 -name 'title.csv' | head -1)")
  [ -n "$CSVDIR" ] && [ -d "$CSVDIR" ] || { echo "  !! no title.csv found under $WORK"; exit 1; }
  echo "  loading from $CSVDIR"
  for f in "$CSVDIR"/*.csv; do
    t=$(basename "$f" .csv)
    echo "  COPY $t"
    psql -d "$DB" -c "\\copy $t FROM '$f' CSV ESCAPE '\\'"
  done
  psql -d "$DB" -f "$JOB/fkindexes.sql"
else
  echo "  already loaded, skipping"
fi
psql -d "$DB" -c "VACUUM ANALYZE"

echo "== 5/5 verify =="
cd - >/dev/null
python -m oppcert.unreported.query_optimization --setup --db "dbname=$DB" --queries "$JOB"

cat <<EOF

Ready. The single pre-registered run is:

  python -m oppcert.unreported.query_optimization --run --db "dbname=$DB" --queries "$JOB" --out logs/job.json 2>&1 | tee logs/job.log

It sweeps six non-learned configurations, measures the reachable and gross ceilings, and prints
the certificate. Expect several hours: 113 queries x 6 configurations x 3 repetitions, plus the
oracle arms. Nothing in the bar can be changed after this point.
EOF
