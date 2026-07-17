#!/usr/bin/env bash
# Day 0 -- fetch 2M-request PREFIXES of the breadth traces.
#
# Trick: we never download a full trace. oracleGeneral records are 24 bytes
# (u4 ts, u8 obj_id, u4 size, i8 next_vtime), and every run uses --limit 2000000,
# so we stream from S3, decompress on the fly, and cut at 2M records (~48 MB).
# This makes a 27 GB trace cost the same as an 80 MB one. The truncated prefix is
# EXACTLY equivalent to downloading the full file and loading with --limit: next_vtime
# entries pointing past the prefix are the same in both cases, so all runs stay
# comparable with the msr/wiki/twitter numbers already produced.
#
# Breadth set (>=3 per family):
#   block : msr_proj_0 (have) + 2 smallest other MSR volumes + 2 smallest CloudPhysics
#   CDN   : wiki_2019t (have) + 2 smallest Meta CDN
#   KV    : cluster26 (have) + twitter clusters 10, 50, 53 (smallest three)
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data logs
B=s3://cache-datasets/cache_dataset_oracleGeneral
BYTES=$((2000000 * 24 + 1024))          # 2M records + slack

fetch_prefix () {                        # $1 = s3 key relative to $B, $2 = local basename
  local out="data/$2.oracleGeneral"
  if [[ -s "$out" ]]; then echo "[skip] already have $out"; return 0; fi
  echo "[fetch] $1 -> $out"
  # head closes the pipe after $BYTES; aws/zstd exit via SIGPIPE -- that is expected.
  aws s3 cp --no-sign-request "$B/$1" - 2>/dev/null | zstd -dc 2>/dev/null \
    | head -c "$BYTES" > "$out"
  local sz; sz=$(stat -c%s "$out" 2>/dev/null || stat -f%z "$out")
  if (( sz < 24000000 )); then
    echo "[WARN] $out is only $sz bytes (<1M requests) -- trace shorter than 2M reqs; keeping"
  fi
  ls -lh "$out"
}

smallest () {                            # $1 = family dir, $2 = count -> prints keys
  aws s3 ls --no-sign-request "$B/$1/" | awk '$4 ~ /\.zst$/ {print $3, $4}' \
    | sort -n | head -n "$2" | awk '{print $2}'
}

echo "== block: MSR (2 smallest volumes) =="
for key in $(smallest 2007_msr 3); do
  name="${key%%.oracleGeneral*}"
  [[ "$name" == "msr_proj_0" ]] && continue
  fetch_prefix "2007_msr/$key" "$name"
done

echo "== block: CloudPhysics (2 smallest) =="
for key in $(smallest 2015_cloudphysics 2); do
  fetch_prefix "2015_cloudphysics/$key" "${key%%.oracleGeneral*}"
done

echo "== CDN: Meta CDN (2 smallest) =="
for key in $(smallest 2022_metaCDN 2); do
  fetch_prefix "2022_metaCDN/$key" "${key%%.oracleGeneral*}"
done

echo "== KV: Twitter clusters 10, 50, 53 =="
for c in 10 50 53; do
  fetch_prefix "2020_twitter/cluster${c}.oracleGeneral.sample10.zst" "cluster${c}.sample10"
done

echo; echo "== data/ now contains =="
ls -lh data/
