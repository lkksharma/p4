#!/usr/bin/env bash
# install_postgres.sh -- PostgreSQL 16 + pg_hint_plan for the JOB instantiation.
# Written for a root shell on an Ubuntu/Debian node (DGX images qualify). Idempotent.
set -euo pipefail

PGVER=${PGVER:-16}

echo "== 0/4 environment =="
. /etc/os-release 2>/dev/null || { echo "cannot read /etc/os-release; not a Debian/Ubuntu host"; exit 1; }
echo "  ${PRETTY_NAME:-unknown}  codename=${VERSION_CODENAME:-unknown}"
[ "$(id -u)" -eq 0 ] || { echo "run as root (needed for apt and for starting the cluster)"; exit 1; }

if command -v psql >/dev/null && psql -V | grep -q " $PGVER\."; then
  echo "  PostgreSQL $PGVER already present, skipping install"
else
  echo "== 1/4 PGDG repository =="
  apt-get update -qq
  apt-get install -y -qq curl ca-certificates gnupg lsb-release >/dev/null
  install -d /usr/share/postgresql-common/pgdg
  curl -fsSL -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
    https://www.postgresql.org/media/keys/ACCC4CF8.asc
  echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] \
https://apt.postgresql.org/pub/repos/apt ${VERSION_CODENAME}-pgdg main" \
    > /etc/apt/sources.list.d/pgdg.list
  apt-get update -qq

  echo "== 2/4 server + pg_hint_plan =="
  apt-get install -y "postgresql-${PGVER}" "postgresql-${PGVER}-pg-hint-plan"
fi

echo "== 3/4 start the cluster =="
# Containers usually lack systemd, so prefer pg_ctlcluster and fall back to service/systemctl.
if ! pg_isready -q 2>/dev/null; then
  pg_ctlcluster "$PGVER" main start 2>/dev/null \
    || service postgresql start 2>/dev/null \
    || systemctl start postgresql 2>/dev/null \
    || true
fi
pg_isready || { echo "  !! cluster did not start; check /var/log/postgresql/"; exit 1; }
echo "  cluster up"

echo "== 4/4 load pg_hint_plan for every session =="
# LOAD works per session, but preloading removes any chance of an arm silently running unhinted,
# which would turn a broken tool into a fake NO CORRIDOR verdict.
CONF=$(sudo -u postgres psql -tAc "SHOW config_file")
if ! grep -q "pg_hint_plan" "$CONF"; then
  echo "shared_preload_libraries = 'pg_hint_plan'" >> "$CONF"
  pg_ctlcluster "$PGVER" main restart 2>/dev/null || service postgresql restart
  pg_isready || { echo "  !! restart failed"; exit 1; }
fi
echo "  shared_preload_libraries configured in $CONF"

# The experiment connects as the invoking (root) user; give it a superuser role and a database.
sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='root'" | grep -q 1 \
  || sudo -u postgres createuser -s root
sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='imdb'" | grep -q 1 \
  || sudo -u postgres createdb -O root imdb

echo
echo "PostgreSQL $PGVER ready, hint library preloaded, database 'imdb' created."
echo "Next:  bash job_setup.sh"
