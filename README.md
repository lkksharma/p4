# P4 — Joint Cache Eviction + Prefetching (W1)

W1 scope only: the deterministic cache env + the kill-tests that must pass **before** any RL is
built. Per the guide: *"same trace + LRU in your env and libCacheSim must produce identical hit
counts. This is your env's unit test; do not proceed until exact match."*

| file | role |
|---|---|
| `p4_cache.py` | trace IO (oracleGeneral binary + synthetic), footprint, cache sim (LRU/FIFO/LFU/Belady) |
| `p4_gates.py` | the W1/W2 kill-tests + CLI |

## The four gates

| gate | what it proves | bar |
|---|---|---|
| **W1a FORMAT** | the 24-byte `oracleGeneral` struct is parsed correctly — re-derives `next_access_vtime` from the `obj_id` sequence and requires an **exact** match with the file's own column | exact |
| **W1b PARITY** | our LRU == an **independent** O(n) reference LRU (no shared code path), and == **libCacheSim** | exact hit counts |
| **W2 GAP** | there is room for better eviction: LRU→Belady OHR gap | ≥5 pts @1% (guide band: 10–25) |
| **W2b PREFETCH** *(added)* | there is successor structure to prefetch at all | top-16 coverage ≥0.10 **and** lift over popularity ≥0.05 |

**Why W2b exists (it is not in the guide).** The guide gates the *eviction* headroom but never gates
the *prefetch* signal — yet the whole thesis ("hits Belady cannot express") is dead without it. This
is the same "no gap, no paper" check that would have saved weeks on the previous projects. Verified
to discriminate: a synthetic trace with sequential runs scores lift **+0.28 (PASS)**; the same trace
with the runs removed scores **−0.02 (FAIL)**.

## Run

```bash
pip install -r requirements.txt          # numpy; libcachesim is optional but required for real W1b

# 0. no data needed — proves the env + gates work end to end
python p4_gates.py --synth --limit 200000 --seq-frac 0.3      # all gates PASS
python p4_gates.py --synth --limit 200000 --seq-frac 0.0      # W2b FAILS (negative control)

# 1. on a real trace
python p4_gates.py --trace data/wiki2019.oracleGeneral --limit 5000000
```

## Data (public, no credentials)

```bash
pip install awscli zstandard
aws s3 ls --no-sign-request s3://cache-datasets/                     # browse
aws s3 cp --no-sign-request s3://cache-datasets/<path>/trace.zst data/
zstd -d data/trace.zst                                              # -> oracleGeneral binary
```
Hub: [cacheMon/cache_dataset](https://github.com/cacheMon/cache_dataset) (Wikipedia CDN 2019,
Twitter cluster, MSR, Tencent/CloudPhysics) — already in `oracleGeneral`, so **Belady is free**.

**Trace choice matters twice over.** Block-I/O traces (MSR, CloudPhysics) have spatial locality, so
(a) they are where W2b actually passes, and (b) their objects are **uniform-size**, which is the only
regime where Belady is *provably* optimal — making the headline claim ("exceeds the provable
eviction optimum") exact rather than heuristic. On variable-size CDN objects, evict-farthest is a
strong reference, **not** a proven optimum (variable-size eviction is NP-hard). Do not overclaim
"provably optimal" on Wikipedia/Twitter.

## Honest status

- **W1b vs libCacheSim is PENDING.** libCacheSim isn't installed here, so only the
  independent-reference parity has run. The gate prints `SKIPPED` loudly and calls the pass
  *provisional* — it never silently passes. **The libCacheSim API call in `libcachesim_lru()` is
  written from documentation, not verified against the installed package — expect to adjust it.**
- The RL MDP (eviction/prefetch heads, features, offline dataset) is deliberately **not built yet**.
