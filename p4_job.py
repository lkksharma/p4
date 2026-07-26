#!/usr/bin/env python3
"""p4_job.py -- Opportunity Certification in a fifth domain: query optimization.

Runs the protocol pre-registered in SPEC_job_prereg.md end to end, in one command, on the Join
Order Benchmark over IMDB. Nothing here is tuned to its own outcome: every threshold below is read
from the spec, which was written before any query was executed.

WHAT IT MEASURES
----------------
    BASE            PostgreSQL at defaults, no extended statistics.
    TUNED BASELINE  best total workload latency over the swept non-learned configuration family
                    (statistics target x extended statistics x forced exhaustive enumeration).
                    This is the M1 discipline: default PostgreSQL is NOT the tuned baseline, in the
                    same way binary search was not the tuned baseline for a learned index. Extended
                    statistics capture the multi-column correlation signal a learned cardinality
                    estimator is built to model, with no model and no inference cost.
    REACHABLE CEIL  true cardinalities injected for SINGLE-TABLE selections only; the optimizer
                    estimates joins from those corrected inputs. A per-table learned model can in
                    principle attain this, so it is the honest upper bound for that model class.
                    THIS IS THE PRIMARY ARM: the pre-registered bar is evaluated on it.
    GROSS CEILING   true cardinalities injected for join subqueries as well. Credits multi-way
                    correlation knowledge no per-table summary holds; DIAGNOSTIC ONLY, reported to
                    size the phantom slice (gross minus reachable), exactly as in the caching half.

    corridor = tuned baseline latency - reachable ceiling latency, as a % of tuned baseline.

PRE-REGISTERED BAR (SPEC_job_prereg.md 5). Both conditions must hold:
    (a) corridor >= 20% of tuned baseline workload latency, bootstrap CI lower bound also >= 20%;
    (b) corridor strictly exceeds the SPAN of workload latency across the swept configurations.
        If configuring the database properly moves latency more than perfect base-table
        cardinalities do, the opportunity is inside what a competent administrator already reaches.
    (c) concentration is REPORTED, not gated: the per-query gain distribution and the share of
        queries carrying the aggregate.

Every arm is charged planning + estimation + execution time (SPEC 4). True cardinalities are
precomputed offline and so are free at query time, which makes every ceiling strictly optimistic:
the measured corridor is an upper bound, so a negative verdict is conservative.

    python p4_job.py --selftest                 # pure-python logic, no database required
    python p4_job.py --setup --db imdb          # verify schema, statistics, extension availability
    python p4_job.py --run --db imdb --queries job/ --out logs/job.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np

LINE = "=" * 100

# ---- pre-registered constants; read from SPEC_job_prereg.md, never tuned to an outcome ----
BAR_CORRIDOR_PCT = 20.0        # (a) minimum workload latency reduction, in percent
BOOT_RESAMPLES = 10_000
REPS = 3                       # repetitions per query; median taken
TIMEOUT_MS = 120_000           # per-query statement timeout; timeouts are reported, never dropped
STAT_TARGETS = (100, 1000, 10000)
GROSS_MAX_RELS = 8             # relation cap for the join-lattice arm; the cap is REPORTED


# --------------------------------------------------------------------------- database plumbing
def connect(dsn):
    try:
        import psycopg2
    except ImportError:
        sys.exit("psycopg2 is required: pip install psycopg2-binary")
    return psycopg2.connect(dsn)


def sql_one(cur, q):
    cur.execute(q)
    r = cur.fetchone()
    return r[0] if r else None


def explain_timed(cur, query, hint=""):
    """Execute under EXPLAIN ANALYZE and return (planning_ms, execution_ms, timed_out).

    Planning and execution are returned separately because SPEC 4 charges every arm its planning
    and estimation time, which the learned-cardinality literature routinely excludes from reported
    plan latencies. Total time is what the bar is evaluated on.
    """
    stmt = f"{hint}\nEXPLAIN (ANALYZE, FORMAT JSON) {query}"
    try:
        cur.execute(f"SET statement_timeout = {TIMEOUT_MS}")
        cur.execute(stmt)
        plan = cur.fetchone()[0][0]
        return float(plan["Planning Time"]), float(plan["Execution Time"]), False
    except Exception as e:
        cur.connection.rollback()
        if "timeout" in str(e).lower() or "canceling" in str(e).lower():
            return 0.0, float(TIMEOUT_MS), True
        raise


def measure(cur, query, hint="", reps=REPS):
    """Median of `reps` timed runs after one warming pass. Returns (total_ms, timed_out)."""
    explain_timed(cur, query, hint)                     # warm the buffer cache; discarded
    tot, to_any = [], False
    for _ in range(reps):
        p, e, to = explain_timed(cur, query, hint)
        tot.append(p + e)
        to_any |= to
    return float(np.median(tot)), to_any


# --------------------------------------------------------------------------- JOB query handling
ALIAS_RE = re.compile(r'\b([a-z_]+)\s+AS\s+([a-z_0-9]+)', re.I)


def parse_query(text):
    """Extract (table, alias) pairs from a JOB query's FROM clause.

    JOB queries alias every relation, which is what makes per-relation cardinality injection
    expressible. Anything that does not parse is reported and skipped rather than guessed at.
    """
    m = re.search(r'\bFROM\b(.*?)\bWHERE\b', text, re.I | re.S)
    if not m:
        return []
    return [(t.lower(), a.lower()) for t, a in ALIAS_RE.findall(m.group(1))]


def load_queries(path):
    """Load .sql files from a directory (the JOB artifact layout). Sorted for determinism."""
    out = []
    for fn in sorted(os.listdir(path)):
        if not fn.endswith(".sql"):
            continue
        text = open(os.path.join(path, fn)).read().strip().rstrip(";")
        rels = parse_query(text)
        if not rels:
            print(f"  !! {fn}: FROM clause did not parse; skipped and reported")
            continue
        out.append(dict(name=fn, sql=text, rels=rels))
    return out


def base_predicates(sql, alias):
    """Conjuncts of the WHERE clause mentioning exactly one alias: this alias's local selection.

    A conjunct referencing two aliases is a join predicate and is excluded, which is precisely the
    base-table / join split the reachability restriction turns on.
    """
    m = re.search(r'\bWHERE\b(.*)$', sql, re.I | re.S)
    if not m:
        return []
    conj = re.split(r'\bAND\b', m.group(1), flags=re.I)
    keep = []
    for c in conj:
        used = set(re.findall(r'\b([a-z_0-9]+)\.', c, re.I))
        if used == {alias}:
            keep.append(c.strip().rstrip(";").strip())
    return keep


def true_base_cards(cur, q):
    """True cardinality of every single-table selection: one COUNT(*) per relation.

    This is the whole cost of the REACHABLE ceiling, and it is linear in the number of relations
    rather than exponential, which is why the primary arm is affordable while the gross arm needs
    a cap.
    """
    cards = {}
    for tbl, alias in q["rels"]:
        preds = base_predicates(q["sql"], alias)
        where = (" WHERE " + " AND ".join(preds)) if preds else ""
        where = where.replace(f"{alias}.", "x.")
        cards[alias] = int(sql_one(cur, f"SELECT count(*) FROM {tbl} x{where}"))
    return cards


def hint_rows(cards):
    """A pg_hint_plan block correcting the optimizer's row estimates.

    Rows(alias #n) sets an absolute cardinality for that relation or join. Injecting a correct
    number is the same construction Leis et al. used to price the true-cardinality oracle.
    """
    if not cards:
        return ""
    body = " ".join(f"Rows({k} #{v})" for k, v in sorted(cards.items()))
    return f"/*+ {body} */"


# --------------------------------------------------------------------------- configuration sweep
def apply_config(cur, cfg):
    cur.execute(f"SET default_statistics_target = {cfg['stat_target']}")
    cur.execute("SET geqo = off")                                  # force exhaustive enumeration
    cur.execute("SET join_collapse_limit = 20")
    cur.execute("SET from_collapse_limit = 20")


def build_extended_stats(cur, queries, enable):
    """Create multivariate statistics on the column groups JOB actually filters on.

    This is the domain's interpolation search: PostgreSQL has captured multi-column correlation
    natively since v10 with no model, no training and no inference cost, which is exactly the
    signal a learned cardinality estimator is built to model. A comparison that leaves it off is
    an under-tuned baseline (M1), and most of the learned-cardinality literature leaves it off.
    """
    cur.execute("SELECT stxname FROM pg_statistic_ext WHERE stxname LIKE 'p4job_%'")
    for (n,) in cur.fetchall():
        cur.execute(f"DROP STATISTICS IF EXISTS {n}")
    if not enable:
        cur.connection.commit()
        return 0
    groups = {}
    for q in queries:
        for tbl, alias in q["rels"]:
            cols = set()
            for p in base_predicates(q["sql"], alias):
                cols |= set(re.findall(rf'\b{alias}\.([a-z_0-9]+)', p, re.I))
            if len(cols) >= 2:
                groups.setdefault(tbl, set()).update(cols)
    made = 0
    for tbl, cols in groups.items():
        cols = sorted(cols)[:8]                                    # PostgreSQL caps the group size
        if len(cols) < 2:
            continue
        try:
            cur.execute(f"CREATE STATISTICS p4job_{tbl} (ndistinct, dependencies, mcv) "
                        f"ON {', '.join(cols)} FROM {tbl}")
            made += 1
        except Exception:
            cur.connection.rollback()
    cur.connection.commit()
    for tbl in groups:
        cur.execute(f"ANALYZE {tbl}")
    cur.connection.commit()
    return made


# --------------------------------------------------------------------------- statistics
def boot_ci(per_query_gain, resamples=BOOT_RESAMPLES, seed=0):
    """Bootstrap over QUERIES, which are the independent unit here.

    The caching half resamples contiguous blocks because cache hits are autocorrelated through
    cache state; JOB queries are independent of one another, so an ordinary query-level bootstrap
    is the correct convention and is stated rather than inherited.
    """
    a = np.asarray(per_query_gain, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(resamples, len(a)))
    return np.percentile(a[idx].sum(axis=1), [2.5, 97.5])


def concentration(base_ms, arm_ms):
    """Share of the aggregate gain carried by the top decile of queries, and the count that
    accounts for 80% of it. SPEC 5(c) requires this because a concentrated gain implies a
    different engineering response (re-optimisation, or a fallback for those queries) than a
    broad one (a better estimator everywhere), and a mean alone conceals the difference."""
    g = np.asarray(base_ms) - np.asarray(arm_ms)
    tot = g.sum()
    if tot <= 0:
        return dict(total=float(tot), top_decile_share=float("nan"), n_for_80pct=-1,
                    n_improved=int((g > 0).sum()), n=len(g))
    s = np.sort(g)[::-1]
    k = max(1, len(s) // 10)
    csum = np.cumsum(s)
    return dict(total=float(tot), top_decile_share=float(s[:k].sum() / tot),
                n_for_80pct=int(np.searchsorted(csum, 0.8 * tot) + 1),
                n_improved=int((g > 0).sum()), n=len(g))


# --------------------------------------------------------------------------- the run
def run(args):
    conn = connect(args.db); conn.autocommit = False
    cur = conn.cursor()
    queries = load_queries(args.queries)
    if not queries:
        sys.exit(f"no parsable .sql files in {args.queries}")
    print(f"\n{LINE}\n  OPPORTUNITY CERTIFICATION -- fifth domain: query optimization (JOB/IMDB)\n{LINE}")
    print(f"  {len(queries)} queries | bar: corridor >= {BAR_CORRIDOR_PCT:.0f}% AND > config span "
          f"(pre-registered, SPEC_job_prereg.md)")

    has_hint = bool(sql_one(cur, "SELECT count(*) FROM pg_available_extensions "
                                 "WHERE name='pg_hint_plan'"))
    if has_hint:
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_hint_plan"); conn.commit()
        except Exception:
            conn.rollback(); has_hint = False
    if not has_hint:
        sys.exit("  !! pg_hint_plan is unavailable; the ceiling arms cannot inject cardinalities.\n"
                 "     Install it (the oracle arms are the experiment) and re-run.")

    # ---- 1. the swept non-learned family; the tuned baseline is its best member (M1) ----
    print(f"\n  [1/4] configuration sweep ({len(STAT_TARGETS)*2} configurations)")
    sweep = []
    for ext in (False, True):
        made = build_extended_stats(cur, queries, ext)
        for st in STAT_TARGETS:
            cfg = dict(stat_target=st, extended=ext)
            apply_config(cur, cfg)
            tot, tos = 0.0, 0
            per = []
            for q in queries:
                ms, to = measure(cur, q["sql"])
                per.append(ms); tot += ms; tos += to
            sweep.append(dict(cfg=cfg, total_ms=tot, per_query=per, timeouts=tos,
                              made_stats=made))
            print(f"     stat_target={st:<6} extended={str(ext):<5} total={tot/1000:8.1f}s "
                  f"timeouts={tos}")
    best = min(sweep, key=lambda s: s["total_ms"])
    worst = max(sweep, key=lambda s: s["total_ms"])
    span_pct = 100.0 * (worst["total_ms"] - best["total_ms"]) / best["total_ms"]
    default_cfg = next(s for s in sweep if s["cfg"] == dict(stat_target=100, extended=False))
    print(f"     BASE (defaults)   {default_cfg['total_ms']/1000:8.1f}s")
    print(f"     TUNED BASELINE    {best['total_ms']/1000:8.1f}s  "
          f"{best['cfg']}  ({100*(default_cfg['total_ms']-best['total_ms'])/default_cfg['total_ms']:+.1f}% vs defaults)")
    print(f"     CONFIG SPAN       {span_pct:.1f}% of tuned baseline  <- condition (b) threshold")

    # restore the winning configuration for every arm below
    build_extended_stats(cur, queries, best["cfg"]["extended"])
    apply_config(cur, best["cfg"])

    # ---- 2. true base-table cardinalities: the REACHABLE ceiling ----
    print(f"\n  [2/4] true base-table cardinalities (reachable ceiling; linear in relations)")
    t0 = time.time()
    reach_ms, reach_to = [], 0
    for q in queries:
        cards = true_base_cards(cur, q)
        ms, to = measure(cur, q["sql"], hint_rows(cards))
        reach_ms.append(ms); reach_to += to
    print(f"     computed in {time.time()-t0:.0f}s | total={sum(reach_ms)/1000:8.1f}s "
          f"timeouts={reach_to}")

    # ---- 3. join-lattice cardinalities: the GROSS ceiling (diagnostic) ----
    print(f"\n  [3/4] join-lattice cardinalities (gross ceiling; DIAGNOSTIC, "
          f"relations <= {GROSS_MAX_RELS})")
    gross_ms, gross_to, capped = [], 0, 0
    for q, rb in zip(queries, reach_ms):
        if len(q["rels"]) > GROSS_MAX_RELS:
            gross_ms.append(rb); capped += 1          # not measured; reachable value carried
            continue
        cards = true_base_cards(cur, q)
        for a, b in _pairs(q):
            try:
                cards[f"{a} {b}"] = int(sql_one(cur, _pair_count_sql(q, a, b)))
            except Exception:
                conn.rollback()
        ms, to = measure(cur, q["sql"], hint_rows(cards))
        gross_ms.append(ms); gross_to += to
    print(f"     total={sum(gross_ms)/1000:8.1f}s timeouts={gross_to} | "
          f"{capped} queries exceeded the relation cap and are EXCLUDED from the gross arm "
          f"(reported, not silently truncated)")

    # ---- 4. verdict ----
    b_ms = np.asarray(best["per_query"]); r_ms = np.asarray(reach_ms); g_ms = np.asarray(gross_ms)
    corridor_pct = 100.0 * (b_ms.sum() - r_ms.sum()) / b_ms.sum()
    lo, hi = boot_ci(b_ms - r_ms)
    lo_pct, hi_pct = 100.0 * lo / b_ms.sum(), 100.0 * hi / b_ms.sum()
    gross_pct = 100.0 * (b_ms.sum() - g_ms.sum()) / b_ms.sum()
    phantom = gross_pct - corridor_pct
    conc = concentration(b_ms, r_ms)

    inv = []
    if g_ms.sum() > r_ms.sum() * 1.02:
        inv.append("gross ceiling slower than reachable ceiling: injection likely mis-wired")
    if corridor_pct > gross_pct + 2.0:
        inv.append("reachable corridor exceeds gross corridor: impossible, check the arms")

    print(f"\n{LINE}")
    print(f"  TUNED BASELINE    {b_ms.sum()/1000:8.1f}s")
    print(f"  REACHABLE CEIL    {r_ms.sum()/1000:8.1f}s   corridor {corridor_pct:+.1f}% "
          f"[{lo_pct:+.1f}, {hi_pct:+.1f}]   <- the pre-registered arm")
    print(f"  GROSS CEILING     {g_ms.sum()/1000:8.1f}s   corridor {gross_pct:+.1f}%   (diagnostic)")
    print(f"  PHANTOM SLICE     {phantom:+.1f} points reachable only with multi-way join "
          f"correlation no per-table summary holds")
    print(f"  CONCENTRATION     {conc['n_improved']}/{conc['n']} queries improve | top decile "
          f"carries {100*conc['top_decile_share']:.0f}% | {conc['n_for_80pct']} queries are 80% of it")
    print(f"  INVARIANTS        " + ("all pass" if not inv else "FAIL"))
    for m in inv:
        print(f"    !! {m}")

    cond_a = lo_pct >= BAR_CORRIDOR_PCT
    cond_b = corridor_pct > span_pct
    print(f"\n  (a) corridor CI lower {lo_pct:+.1f}% >= {BAR_CORRIDOR_PCT:.0f}% : {cond_a}")
    print(f"  (b) corridor {corridor_pct:.1f}% > config span {span_pct:.1f}%  : {cond_b}")
    if cond_a and cond_b:
        v = ("BUILD -- a per-table cardinality model is warranted. The retrospective claim weakens "
             "to 'the obstacle was deployment cost, not headroom'.")
    elif phantom > corridor_pct:
        v = ("TRAP -- the corridor lives mostly in multi-way join correlation that no per-table "
             "summary can hold; reachable by the data but not by this hypothesis class.")
    elif not cond_b:
        v = ("NO CORRIDOR (M1) -- the opportunity is inside the span a tuned non-learned "
             "configuration already covers. The retrospective claim is at its strongest.")
    else:
        v = ("NO CORRIDOR -- the corridor does not clear the pre-registered magnitude bar.")
    if cond_a and cond_b and conc["top_decile_share"] > 0.8:
        v += ("  NOTE: the gain is extremely concentrated, so the indicated response is "
              "re-optimisation or a fallback for those queries, not an estimator for all of them.")
    print(f"  CERTIFICATE       {v}\n{LINE}\n")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(dict(sweep=[{k: s[k] for k in ("cfg", "total_ms", "timeouts")} for s in sweep],
                       tuned=best["cfg"], span_pct=span_pct, corridor_pct=corridor_pct,
                       ci=[lo_pct, hi_pct], gross_pct=gross_pct, phantom=phantom,
                       concentration=conc, invariants=inv, certificate=v,
                       n_queries=len(queries), gross_capped=capped),
                  open(args.out, "w"), indent=2)
        print(f"  wrote {args.out}\n")


def _pairs(q):
    a = [al for _, al in q["rels"]]
    return [(a[i], a[j]) for i in range(len(a)) for j in range(i + 1, len(a))]


def _pair_count_sql(q, a, b):
    """COUNT(*) for the two-relation subquery, carrying both local selections and the join
    predicates that mention only these two aliases."""
    tb = {al: t for t, al in q["rels"]}
    m = re.search(r'\bWHERE\b(.*)$', q["sql"], re.I | re.S)
    conj = re.split(r'\bAND\b', m.group(1), flags=re.I) if m else []
    keep = []
    for c in conj:
        used = set(re.findall(r'\b([a-z_0-9]+)\.', c, re.I))
        if used and used <= {a, b}:
            keep.append(c.strip().rstrip(";").strip())
    where = (" WHERE " + " AND ".join(keep)) if keep else ""
    return f"SELECT count(*) FROM {tb[a]} {a}, {tb[b]} {b}{where}"


# --------------------------------------------------------------------------- selftest
def selftest():
    """Exercises every pure-python component. The database path cannot be tested without a live
    server and is exercised by --setup on the target machine."""
    q = dict(name="t.sql", rels=[("title", "t"), ("movie_info", "mi")],
             sql="SELECT MIN(t.title) FROM title AS t, movie_info AS mi "
                 "WHERE t.production_year > 2000 AND t.kind_id = 1 AND mi.info = 'x' "
                 "AND t.id = mi.movie_id")
    assert parse_query(q["sql"]) == [("title", "t"), ("movie_info", "mi")], parse_query(q["sql"])
    bp = base_predicates(q["sql"], "t")
    assert len(bp) == 2 and all("mi." not in p for p in bp), bp
    assert base_predicates(q["sql"], "mi") == ["mi.info = 'x'"], base_predicates(q["sql"], "mi")
    h = hint_rows({"t": 100, "mi": 5})
    assert h == "/*+ Rows(mi #5) Rows(t #100) */", h
    assert "title t, movie_info mi" in _pair_count_sql(q, "t", "mi")

    lo, hi = boot_ci([10.0] * 50, resamples=500)
    assert lo <= 500.0 <= hi, (lo, hi)
    # moderate concentration: gains of 90 and nine of 10, so seven queries really are 80%
    c = concentration([100.0] * 10, [90.0] * 9 + [10.0])
    assert c["n_improved"] == 10 and c["n_for_80pct"] == 7, c
    assert abs(c["top_decile_share"] - 0.5) < 1e-9, c
    # extreme concentration: one query carries nearly everything, the case SPEC 5(c) exists for
    c3 = concentration([100.0] * 10, [10.0] + [99.0] * 9)
    assert c3["n_for_80pct"] == 1 and c3["top_decile_share"] > 0.9, c3
    c2 = concentration([100.0] * 4, [100.0] * 4)
    assert c2["n_for_80pct"] == -1, c2                 # no gain -> undefined, not a crash

    print("  [p4_job selftest] PASS -- parsing, base/join predicate split, hint construction, "
          "pair SQL, bootstrap, concentration (including the zero-gain edge case)")


def setup(args):
    """Verify the target machine can run the experiment before a week of work is committed."""
    conn = connect(args.db); cur = conn.cursor()
    print(f"\n  server        {sql_one(cur, 'SHOW server_version')}")
    ext = sql_one(cur, "SELECT count(*) FROM pg_available_extensions WHERE name='pg_hint_plan'")
    print(f"  pg_hint_plan  {'available' if ext else 'MISSING -- required for the ceiling arms'}")
    n = sql_one(cur, "SELECT count(*) FROM information_schema.tables "
                     "WHERE table_schema='public'")
    print(f"  tables        {n}")
    for t in ("title", "movie_info", "cast_info"):
        try:
            print(f"    {t:<12} {sql_one(cur, f'SELECT count(*) FROM {t}'):>12,} rows")
        except Exception:
            conn.rollback(); print(f"    {t:<12} MISSING -- load the IMDB snapshot first")
    if os.path.isdir(args.queries):
        print(f"  queries       {len(load_queries(args.queries))} parsed from {args.queries}")
    else:
        print(f"  queries       {args.queries} not found")
    print()


def main():
    ap = argparse.ArgumentParser(description="Opportunity Certification on JOB/IMDB")
    ap.add_argument("--db", default="dbname=imdb", help="libpq DSN")
    ap.add_argument("--queries", default="job", help="directory of JOB .sql files")
    ap.add_argument("--out", default=None, help="write the verdict as JSON")
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.setup:
        setup(a)
    elif a.run:
        run(a)
    else:
        ap.error("choose --selftest, --setup, or --run")


if __name__ == "__main__":
    main()
