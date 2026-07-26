#!/usr/bin/env bash
# job_setup.sh -- one-shot preparation for the JOB / IMDB instantiation (SPEC_job_prereg.md).
# Idempotent: safe to re-run. Nothing here touches the pre-registered thresholds.
set -euo pipefail

DB=${DB:-imdb}
WORK=${WORK:-$PWD/jobdata}
JOB=${JOB:-$PWD/job}

echo "== 1/5 preconditions =="
command -v psql >/dev/null || { echo "psql not found; install PostgreSQL >= 16"; exit 1; }
psql -V
python -c "import psycopg2" 2>/dev/null || pip install -q psycopg2-binary

echo "== 2/5 pg_hint_plan (required: the ceiling arms inject cardinalities through it) =="
if ! psql -d postgres -tAc \
    "SELECT 1 FROM pg_available_extensions WHERE name='pg_hint_plan'" | grep -q 1; then
  cat <<'EOF'
  !! pg_hint_plan is NOT available. The oracle arms cannot run without it.
     Debian/Ubuntu:  apt-get install postgresql-16-pg-hint-plan
     from source:    git clone https://github.com/ossc-db/pg_hint_plan
                     cd pg_hint_plan && make && make install
     Then re-run this script.
EOF
  exit 1
fi

echo "== 3/5 IMDB snapshot =="
mkdir -p "$WORK" && cd "$WORK"
if [ ! -f imdb.tgz ] && ! psql -d "$DB" -tAc "SELECT to_regclass('title')" 2>/dev/null | grep -q title; then
  echo "  downloading the JOB IMDB snapshot (~1.2 GB compressed)"
  curl -fL -o imdb.tgz http://homepages.cwi.nl/~boncz/job/imdb.tgz
  tar xzf imdb.tgz
fi

echo "== 4/5 load =="
if ! psql -d "$DB" -tAc "SELECT to_regclass('title')" 2>/dev/null | grep -q title; then
  createdb "$DB" 2>/dev/null || true
  # schema + FK indexes ship with the JOB artifact
  psql -d "$DB" -f "$JOB/schema.sql"
  for f in *.csv; do
    t="${f%.csv}"
    echo "  COPY $t"
    psql -d "$DB" -c "\\copy $t FROM '$WORK/$f' CSV ESCAPE '\\'"
  done
  psql -d "$DB" -f "$JOB/fkindexes.sql"
else
  echo "  already loaded, skipping"
fi
psql -d "$DB" -c "VACUUM ANALYZE"

echo "== 5/5 verify =="
cd - >/dev/null
python p4_job.py --setup --db "dbname=$DB" --queries "$JOB"

cat <<EOF

Ready. The single pre-registered run is:

  python p4_job.py --run --db "dbname=$DB" --queries "$JOB" --out logs/job.json 2>&1 | tee logs/job.log

It sweeps six non-learned configurations, measures the reachable and gross ceilings, and prints
the certificate. Expect several hours: 113 queries x 6 configurations x 3 repetitions, plus the
oracle arms. Nothing in the bar can be changed after this point.
EOF
