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
    REACHABLE CEIL  true cardinalities injected for PAIRWISE (2-way) joins only; the optimizer
                    estimates higher-order joins from those corrected inputs. A model holding only
                    pairwise join correlation can in principle attain this, so it is the honest
                    upper bound for that hypothesis class.
                    THIS IS THE PRIMARY ARM: the pre-registered bar is evaluated on it.
    GROSS CEILING   true cardinalities injected for join subqueries up to 4 relations. Credits
                    n-way correlation knowledge no bounded pairwise summary holds; DIAGNOSTIC ONLY,
                    reported to size the phantom slice (gross minus reachable), exactly as in the
                    caching half.

    The split was amended from base-table/join to pairwise/n-way on 2026-07-26, BEFORE any
    measurement, because pg_hint_plan's Rows() corrects only the result of joins and cannot set a
    base-table scan estimate. The amended split tests the same hypothesis-class question and the
    reason is recorded in SPEC_job_prereg.md 10.

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
# Reachability split, amended 2026-07-26 BEFORE any measurement (SPEC_job_prereg.md 10):
# pg_hint_plan's Rows() corrects only the result of JOINS, so the pre-registered base-table /
# join split is not expressible with this instrument. The split becomes pairwise / higher-order,
# which tests the same hypothesis-class question: can a model holding only PAIRWISE join
# correlation reach the corridor, or does it require n-way correlation no bounded summary holds?
REACH_MAX_SIZE = 2             # reachable ceiling: 2-way joins corrected
GROSS_MAX_SIZE = 4             # gross ceiling: up to 4-way; DIAGNOSTIC
CARD_BUDGET_S = 120            # per-query cardinality-computation budget; overruns are REPORTED


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


def enable_hints(cur):
    """Activate pg_hint_plan AND verify that hints actually take effect.

    This is the experiment's most dangerous silent failure. pg_hint_plan is activated by LOAD (or
    shared_preload_libraries), not necessarily by CREATE EXTENSION, and an unloaded library does
    not error: it simply ignores every hint comment. Both ceiling arms would then execute the
    baseline plan, both corridors would read ~0, and the harness would report NO CORRIDOR -- a
    false negative manufactured by a broken tool rather than measured from the data.

    So availability is not assumed from a catalogue lookup. A deliberately absurd Rows() hint is
    issued against a real relation and the optimizer's row estimate is read back: if the estimate
    does not move, hints are inert and the run aborts.
    """
    for stmt in ("LOAD 'pg_hint_plan'", "CREATE EXTENSION IF NOT EXISTS pg_hint_plan",
                 # Default message_level is 'log', which sends hint PARSE ERRORS to the server log
                 # and not to the client. A malformed Rows() hint would then be ignored in silence
                 # and that arm would be partially uncorrected with no visible sign. Raising it to
                 # notice puts those errors on the client connection where run() can count them.
                 "SET pg_hint_plan.message_level = 'notice'"):
        try:
            cur.execute(stmt); cur.connection.commit()
        except Exception:
            cur.connection.rollback()
    # The probe MUST be a join. Rows() corrects "the result of the joins on the tables specified"
    # and is silently ignored on a single relation, so a one-table probe reports INERT even when
    # the library is working perfectly. A self-join on any table's first column is always valid.
    try:
        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname='public' "
                    "ORDER BY tablename LIMIT 1")
        row = cur.fetchone()
        if not row:
            return False
        t = row[0]
        cur.execute("SELECT column_name FROM information_schema.columns "
                    f"WHERE table_name='{t}' ORDER BY ordinal_position LIMIT 1")
        col = cur.fetchone()[0]
        probe = f"SELECT 1 FROM {t} x, {t} y WHERE x.{col} = y.{col}"
        cur.execute(f"EXPLAIN (FORMAT JSON) {probe}")
        plain = float(cur.fetchone()[0][0]["Plan"]["Plan Rows"])
        cur.execute(f"/*+ Rows(x y #4242) */\nEXPLAIN (FORMAT JSON) {probe}")
        hinted = float(cur.fetchone()[0][0]["Plan"]["Plan Rows"])
        cur.connection.commit()
    except Exception:
        cur.connection.rollback()
        return False
    ok = abs(hinted - 4242.0) < 1.0 and abs(hinted - plain) > 0.5
    print(f"  hint efficacy  {'CONFIRMED' if ok else 'INERT'} "
          f"(estimate {plain:.0f} -> {hinted:.0f} under a 4242-row hint)")
    return ok


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
def boot_ci_pct(base_ms, arm_ms, resamples=BOOT_RESAMPLES, seed=0):
    """Bootstrap the workload percentage reduction, resampling QUERIES.

    The ratio must be recomputed inside each resample. Bootstrapping the SUM of per-query gains
    and dividing by the fixed observed total is wrong on this workload: JOB latency is heavy
    tailed, so a resample that draws the dominant query several times yields a "gain" exceeding
    the total it is divided by, and the interval runs past 100% reduction, which is impossible.
    Recomputing numerator and denominator together bounds the statistic by construction.
    """
    b, a = np.asarray(base_ms, float), np.asarray(arm_ms, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(b), size=(resamples, len(b)))
    bs, as_ = b[idx].sum(axis=1), a[idx].sum(axis=1)
    return np.percentile(100.0 * (bs - as_) / bs, [2.5, 97.5])


def boot_ci_diff_pct(base_ms, arm1_ms, arm2_ms, resamples=BOOT_RESAMPLES, seed=0):
    """Paired bootstrap of the DIFFERENCE between two arms' percentage reductions.

    Monotonicity in correction depth is a claim about arm1 versus arm2, so it must be tested on
    the paired difference and not by eyeballing whether two point estimates happen to be ordered.
    An interval containing zero means the depths are indistinguishable, whatever the ordering.
    """
    b = np.asarray(base_ms, float)
    a1, a2 = np.asarray(arm1_ms, float), np.asarray(arm2_ms, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(b), size=(resamples, len(b)))
    bs = b[idx].sum(axis=1)
    d = (100.0 * (bs - a1[idx].sum(axis=1)) / bs) - (100.0 * (bs - a2[idx].sum(axis=1)) / bs)
    return np.percentile(d, [2.5, 97.5])


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

    if not enable_hints(cur):
        sys.exit("  !! pg_hint_plan is unavailable or inactive; the ceiling arms cannot inject\n"
                 "     cardinalities. Run install_postgres.sh, then job_setup.sh, then re-run.")

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

    # ---- 2. PAIRWISE join cardinalities: the REACHABLE ceiling ----
    print(f"\n  [2/4] pairwise join cardinalities (reachable ceiling)")
    t0 = time.time()
    reach_ms, reach_to, reach_trunc = [], 0, 0
    for q in queries:
        cards, trunc = true_join_cards(cur, q, REACH_MAX_SIZE, CARD_BUDGET_S)
        ms, to = measure(cur, q["sql"], hint_rows(cards))
        reach_ms.append(ms); reach_to += to; reach_trunc += trunc
    print(f"     computed in {time.time()-t0:.0f}s | total={sum(reach_ms)/1000:8.1f}s "
          f"timeouts={reach_to} | {reach_trunc} queries hit the cardinality time budget")

    # ---- 3. HIGHER-ORDER join cardinalities: the GROSS ceiling (diagnostic) ----
    print(f"\n  [3/4] join subqueries up to {GROSS_MAX_SIZE} relations "
          f"(gross ceiling; DIAGNOSTIC)")
    gross_ms, gross_to, gross_trunc = [], 0, 0
    for q in queries:
        cards, trunc = true_join_cards(cur, q, GROSS_MAX_SIZE, CARD_BUDGET_S * 3)
        ms, to = measure(cur, q["sql"], hint_rows(cards))
        gross_ms.append(ms); gross_to += to; gross_trunc += trunc
    print(f"     total={sum(gross_ms)/1000:8.1f}s timeouts={gross_to} | "
          f"{gross_trunc} queries hit the cardinality time budget and are only PARTIALLY "
          f"corrected (reported, not silently presented as complete)")
    capped = gross_trunc

    # ---- 4. verdict ----
    b_ms = np.asarray(best["per_query"]); r_ms = np.asarray(reach_ms); g_ms = np.asarray(gross_ms)
    corridor_pct = 100.0 * (b_ms.sum() - r_ms.sum()) / b_ms.sum()
    lo_pct, hi_pct = boot_ci_pct(b_ms, r_ms)          # ratio bootstrap; bounded by 100%
    gross_pct = 100.0 * (b_ms.sum() - g_ms.sum()) / b_ms.sum()
    glo_pct, ghi_pct = boot_ci_pct(b_ms, g_ms)        # the gross arm was reported without a CI
    phantom = gross_pct - corridor_pct
    conc = concentration(b_ms, r_ms)

    inv = []
    if g_ms.sum() > r_ms.sum() * 1.02:
        inv.append("gross ceiling slower than reachable ceiling: injection likely mis-wired")
    if corridor_pct > gross_pct + 2.0:
        inv.append("reachable corridor exceeds gross corridor: impossible, check the arms")
    # Hint parse errors would leave an arm silently under-corrected; message_level was raised to
    # notice in enable_hints() precisely so they land here rather than in the server log.
    bad = [n for n in getattr(conn, "notices", []) if "hint" in str(n).lower()
           and ("error" in str(n).lower() or "syntax" in str(n).lower())]
    if bad:
        inv.append(f"{len(bad)} hint parse errors: an arm ran partially uncorrected "
                   f"(first: {str(bad[0])[:90].strip()})")

    print(f"\n{LINE}")
    print(f"  TUNED BASELINE    {b_ms.sum()/1000:8.1f}s")
    print(f"  REACHABLE CEIL    {r_ms.sum()/1000:8.1f}s   corridor {corridor_pct:+.1f}% "
          f"[{lo_pct:+.1f}, {hi_pct:+.1f}]   <- the pre-registered arm")
    print(f"  GROSS CEILING     {g_ms.sum()/1000:8.1f}s   corridor {gross_pct:+.1f}% "
          f"[{glo_pct:+.1f}, {ghi_pct:+.1f}]   (diagnostic)")
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
                       ci=[lo_pct, hi_pct], gross_pct=gross_pct, gross_ci=[glo_pct, ghi_pct],
                       per_query=dict(baseline=b_ms.tolist(), reachable=r_ms.tolist(),
                                      gross=g_ms.tolist(),
                                      names=[q["name"] for q in queries]),
                       phantom=phantom,
                       concentration=conc, invariants=inv, certificate=v,
                       n_queries=len(queries), gross_capped=capped),
                  open(args.out, "w"), indent=2)
        print(f"  wrote {args.out}\n")


def join_graph(q):
    """alias -> aliases sharing a join predicate with it. Conjuncts naming exactly two aliases
    are join edges; conjuncts naming one are local selections."""
    g = {al: set() for _, al in q["rels"]}
    m = re.search(r'\bWHERE\b(.*)$', q["sql"], re.I | re.S)
    if not m:
        return g
    for c in re.split(r'\bAND\b', m.group(1), flags=re.I):
        used = set(re.findall(r'\b([a-z_0-9]+)\.', c, re.I)) & set(g)
        if len(used) == 2:
            x, y = sorted(used)
            g[x].add(y); g[y].add(x)
    return g


def connected_subsets(g, size):
    """Connected vertex subsets of exactly `size` relations.

    Only connected subsets are enumerated: a disconnected subset is a cross product the optimizer
    never considers, so correcting it would be wasted work. This keeps the count far below the
    combinatorial bound on JOB's sparse star and snowflake graphs.
    """
    from itertools import combinations
    out = []
    for combo in combinations(sorted(g), size):
        s, seen, stack = set(combo), {combo[0]}, [combo[0]]
        while stack:
            for nb in g[stack.pop()] & s:
                if nb not in seen:
                    seen.add(nb); stack.append(nb)
        if seen == s:
            out.append(combo)
    return out


def subset_count_sql(q, aliases):
    """COUNT(*) for a join subquery, carrying every conjunct that mentions only these aliases
    (both their local selections and the joins among them)."""
    tb = {al: t for t, al in q["rels"]}
    s = set(aliases)
    m = re.search(r'\bWHERE\b(.*)$', q["sql"], re.I | re.S)
    keep = []
    for c in (re.split(r'\bAND\b', m.group(1), flags=re.I) if m else []):
        used = set(re.findall(r'\b([a-z_0-9]+)\.', c, re.I))
        if used and used <= s:
            keep.append(c.strip().rstrip(";").strip())
    frm = ", ".join(f"{tb[a]} {a}" for a in aliases)
    where = (" WHERE " + " AND ".join(keep)) if keep else ""
    return f"SELECT count(*) FROM {frm}{where}"


def true_join_cards(cur, q, max_size, budget_s):
    """True cardinalities for connected join subqueries of size 2..max_size.

    Returns (cards, truncated). `truncated` is True when the per-query time budget ran out, and
    the caller REPORTS it rather than presenting a partially corrected arm as fully corrected.
    """
    g = join_graph(q)
    cards, t0 = {}, time.time()
    for size in range(2, max_size + 1):
        for combo in connected_subsets(g, size):
            if time.time() - t0 > budget_s:
                return cards, True
            try:
                n = sql_one(cur, subset_count_sql(q, combo))
                cards[" ".join(combo)] = int(n)
            except Exception:
                cur.connection.rollback()
    return cards, False


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
    g = join_graph(q)
    assert g["t"] == {"mi"} and g["mi"] == {"t"}, g          # t.id = mi.movie_id is one edge
    assert connected_subsets(g, 2) == [("mi", "t")], connected_subsets(g, 2)
    sq = subset_count_sql(q, ("mi", "t"))
    assert "movie_info mi, title t" in sq and "t.id = mi.movie_id" in sq, sq
    # a disconnected pair must never be enumerated: it is a cross product no optimizer considers
    g2 = {"a": set(), "b": set()}
    assert connected_subsets(g2, 2) == [], connected_subsets(g2, 2)
    assert hint_rows({"mi t": 7}) == "/*+ Rows(mi t #7) */"   # join hint, the only valid form

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

    print("  [p4_job selftest] PASS -- parsing, predicate split, join graph, connected-subset "
          "enumeration, subset SQL, JOIN-form hints, bootstrap, concentration (zero-gain case)")


def quicktest(args):
    """Cheap diagnostics on a query subset, before committing to another full run.

    The full run left three things unresolved, and each is answerable on ~20 queries:
      1. Is the gross corridor significant? It was reported with no confidence interval.
      2. Is the accuracy-to-latency map NON-MONOTONIC? Correcting pairwise joins came in at
         -12.6% while correcting up to 4-way came in at +26.3%. If latency improves monotonically
         with correction depth, the pairwise regression was noise; if it dips and then recovers,
         partial correction genuinely harms and that is the section's mechanism.
      3. Is the pairwise regression a PLAN effect or hint-parsing overhead? Planning and execution
         are separated here so the two cannot be confused.
    """
    conn = connect(args.db); cur = conn.cursor()
    if not enable_hints(cur):
        sys.exit("  !! hints inert; fix before diagnosing")
    allq = load_queries(args.queries)
    # Stride, not head: the first N files alphabetically are the lighter JOB families (10a-15d),
    # and sampling them alone hid the heavy queries that drove the full run's regression.
    step = max(1, len(allq) // args.nq)
    qs = allq[::step][:args.nq]
    print(f"\n{LINE}\n  QUICKTEST -- {len(qs)} of {len(allq)} queries (stride {step}), "
          f"correction-depth sweep\n{LINE}")

    apply_config(cur, dict(stat_target=10000))          # the tuned configuration from the full run
    build_extended_stats(cur, qs, False)                # extended stats HURT in the full run

    def arm(depth):
        """depth 0 = uncorrected baseline; else correct joins up to `depth` relations."""
        plan, exe = [], []
        for q in qs:
            hint = ""
            if depth:
                cards, _ = true_join_cards(cur, q, depth, CARD_BUDGET_S)
                hint = hint_rows(cards)
            explain_timed(cur, q["sql"], hint)          # warm
            ps, es = [], []
            for _ in range(REPS):
                p, e, _ = explain_timed(cur, q["sql"], hint)
                ps.append(p); es.append(e)
            plan.append(float(np.median(ps))); exe.append(float(np.median(es)))
        return np.array(plan), np.array(exe)

    base_p, base_e = arm(0)
    base_t = base_p + base_e
    print(f"  depth 0 (baseline)   plan {base_p.sum():7.1f}ms  exec {base_e.sum()/1000:7.1f}s")

    rows = []
    for d in (2, 3, 4):
        p, e = arm(d)
        t = p + e
        gain = 100.0 * (base_t.sum() - t.sum()) / base_t.sum()
        lo, hi = boot_ci_pct(base_t, t)
        rows.append((d, gain, lo, hi, p.sum(), t))
        print(f"  depth {d}              plan {p.sum():7.1f}ms  exec {e.sum()/1000:7.1f}s   "
              f"corridor {gain:+6.1f}% [{lo:+.1f}, {hi:+.1f}]")

    print(f"\n  DEPTH CURVE          " + " -> ".join(f"{g:+.1f}%" for _, g, _, _, _, _ in rows))
    # Ordering of point estimates proves nothing on this variance. Test each adjacent pair on the
    # PAIRED difference and only claim a real dip when its interval excludes zero.
    real_dip = False
    for (d1, g1, _, _, _, t1), (d2, g2, _, _, _, t2) in zip(rows, rows[1:]):
        dlo, dhi = boot_ci_diff_pct(base_t, t1, t2)
        sig = "significant" if (dlo > 0 or dhi < 0) else "within noise"
        real_dip |= (g2 < g1) and dhi < 0
        print(f"    depth {d1}->{d2}: {g2-g1:+.1f} points  [{dlo:+.1f}, {dhi:+.1f}]  {sig}")
    print("    " + ("NON-MONOTONE and significant: partial correction genuinely harms, which is "
                    "the candidate mechanism."
                    if real_dip else
                    "No significant dip: the depth curve is flat within noise, so NO "
                    "non-monotonicity claim is supported by this sample."))

    print(f"  PLANNING OVERHEAD    baseline {base_p.sum():.0f}ms vs deepest {rows[-1][4]:.0f}ms "
          f"({100*(rows[-1][4]-base_p.sum())/max(base_p.sum(),1e-9):+.0f}%) -- if the regression "
          f"were hint parsing it would appear here, not in execution")

    worst = np.argsort(base_t - rows[0][5])[:5]
    print(f"\n  WORST REGRESSIONS at depth 2 (the arm that went backwards):")
    for i in worst:
        d = base_t[i] - rows[0][5][i]
        print(f"    {qs[i]['name']:<12} {base_t[i]/1000:7.2f}s -> {rows[0][5][i]/1000:7.2f}s  "
              f"({d/1000:+.2f}s, {len(qs[i]['rels'])} relations)")
    print(f"\n{LINE}\n")


def distribution(args):
    r"""Per-query distribution of the correction effect, not the workload sum.

    SPEC 5(c) pre-registered concentration as a REPORTING requirement precisely because
    \citet{leis2015good} indicate the effect is concentrated. The two quicktests made the reason
    concrete: a head sample showed +65% and a spread sample -59%, and in each case a single query
    moved more time than the entire rest of the sample. A sum over a heavy-tailed workload is
    therefore not a measurement of the typical query, and reporting only the sum would hide
    whichever direction the outliers happen to point.

    This reports the SAME arms under several aggregations. The pre-registered bar (workload sum)
    is untouched and still decides the certificate; these are the additional statistics the spec
    already required.
    """
    conn = connect(args.db); cur = conn.cursor()
    if not enable_hints(cur):
        sys.exit("  !! hints inert")
    allq = load_queries(args.queries)
    step = max(1, len(allq) // args.nq)
    qs = allq[::step][:args.nq]
    apply_config(cur, dict(stat_target=10000))
    build_extended_stats(cur, qs, False)
    print(f"\n{LINE}\n  DISTRIBUTION -- {len(qs)} of {len(allq)} queries (stride {step}), "
          f"depth 0 vs depth {GROSS_MAX_SIZE}\n{LINE}")

    base, corr, names, truncs = [], [], [], []
    for q in qs:
        b, _ = measure(cur, q["sql"])
        cards, trunc = true_join_cards(cur, q, args.depth, args.card_budget)
        c, _ = measure(cur, q["sql"], hint_rows(cards))
        base.append(b); corr.append(c); names.append(q["name"]); truncs.append(bool(trunc))
    b = np.array(base); c = np.array(corr); tr = np.array(truncs)
    ratio = b / np.maximum(c, 1e-9)            # >1 means the corrected arm is faster

    # Truncation is a HARNESS limit, not a property of the domain: a deployed estimator predicts
    # every subquery cheaply and never runs out of clock. A truncated query receives inconsistent,
    # half-corrected cardinalities, which is a different treatment from the one being measured, so
    # the split is reported and the untruncated subset is reported separately.
    if tr.any():
        print(f"  !! {tr.sum()}/{len(tr)} queries TRUNCATED (partial correction, harness artifact): "
              + ", ".join(n for n, t in zip(names, tr) if t))
        bb, cc = b[~tr], c[~tr]
        if len(bb):
            print(f"     untruncated only ({len(bb)} queries): workload sum "
                  f"{100*(bb.sum()-cc.sum())/bb.sum():+.1f}%  median x{np.median(bb/np.maximum(cc,1e-9)):.2f}")
    print(f"  AGGREGATIONS (all {len(b)} queries, the same measurements, summarised four ways)")
    print(f"    workload sum      {100*(b.sum()-c.sum())/b.sum():+7.1f}%   <- the pre-registered bar")
    print(f"    median query      {100*(1-1/np.median(ratio)):+7.1f}%   (speedup x{np.median(ratio):.2f})")
    print(f"    geometric mean    {100*(1-1/np.exp(np.mean(np.log(ratio)))):+7.1f}%   "
          f"(speedup x{np.exp(np.mean(np.log(ratio))):.2f})")
    print(f"    mean of per-query {100*np.mean((b-c)/b):+7.1f}%")

    imp = ratio > 1.05; reg = ratio < 0.95
    print(f"\n  DIRECTION         {imp.sum()} improved >5% | {(~imp & ~reg).sum()} unchanged | "
          f"{reg.sum()} regressed >5%   (n={len(b)})")

    gain = b - c
    pos, neg = gain[gain > 0], -gain[gain < 0]
    def share(x):
        if len(x) == 0: return "none"
        sx = np.sort(x)[::-1]
        k = int(np.searchsorted(np.cumsum(sx), 0.8 * sx.sum()) + 1)
        return f"{k} queries carry 80% ({sx.sum()/1000:.1f}s total)"
    print(f"  CONCENTRATION     gains: {share(pos)} | losses: {share(neg)}")

    o = np.argsort(gain)
    print(f"\n  WORST 3 (correction hurts)")
    for i in o[:3]:
        print(f"    {names[i]:<10} {b[i]/1000:8.2f}s -> {c[i]/1000:8.2f}s  x{ratio[i]:.2f}")
    print(f"  BEST 3 (correction helps)")
    for i in o[::-1][:3]:
        print(f"    {names[i]:<10} {b[i]/1000:8.2f}s -> {c[i]/1000:8.2f}s  x{ratio[i]:.2f}")

    lo, hi = boot_ci_pct(b, c)
    print(f"\n  workload-sum CI   [{lo:+.1f}, {hi:+.1f}]")
    # A median-ratio bootstrap is stable under heavy tails where the sum is not.
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(b), size=(BOOT_RESAMPLES, len(b)))
    mlo, mhi = np.percentile(np.median(ratio[idx], axis=1), [2.5, 97.5])
    print(f"  median-ratio CI   [x{mlo:.2f}, x{mhi:.2f}]   <- stable under heavy tails")
    print(f"{LINE}\n")


def explain_one(args):
    """Plan comparison for a single named query, corrected versus not.

    The distribution run showed all three worst regressions inside one query family, converging to
    the same wall time. Either the optimizer genuinely switches to a worse plan when handed correct
    numbers (a cost-model effect, and a real finding), or the injected cardinalities are wrong for
    this shape (a harness bug). Only the plans distinguish those, so print them.
    """
    conn = connect(args.db); cur = conn.cursor()
    if not enable_hints(cur):
        sys.exit("  !! hints inert")
    qs = {q["name"]: q for q in load_queries(args.queries)}
    q = qs.get(args.query) or qs.get(args.query + ".sql")
    if not q:
        sys.exit(f"  !! {args.query} not found")
    apply_config(cur, dict(stat_target=10000))
    cards, trunc = true_join_cards(cur, q, args.depth, args.card_budget)
    print(f"\n{LINE}\n  PLAN DIFF -- {q['name']} ({len(q['rels'])} relations, "
          f"{len(cards)} injected cardinalities, truncated={trunc})\n{LINE}")
    # A zero or one injected into a large intermediate would produce exactly the degenerate
    # nested-loop plan this family exhibits, so surface the smallest injected values explicitly.
    small = sorted(cards.items(), key=lambda kv: kv[1])[:5]
    print("  smallest injected cardinalities: " + ", ".join(f"{k}={v:,}" for k, v in small))
    for label, hint in (("BASELINE", ""), ("CORRECTED", hint_rows(cards))):
        cur.execute(f"{hint}\nEXPLAIN (ANALYZE, FORMAT JSON) {q['sql']}")
        p = cur.fetchone()[0][0]
        node = p["Plan"]
        print(f"\n  {label}: {p['Execution Time']/1000:.2f}s exec, top node "
              f"{node['Node Type']}, est {node['Plan Rows']:,} vs actual {node['Actual Rows']:,}")
        def walk(n, d=0):
            if d <= 2:
                print(f"    {'  '*d}{n['Node Type']:<22} est {n.get('Plan Rows',0):>10,} "
                      f"act {n.get('Actual Rows',0):>10,} {n.get('Actual Total Time',0)/1000:7.2f}s")
            for c in n.get("Plans", []):
                walk(c, d + 1)
        walk(node)
    print(f"\n{LINE}\n")


def setup(args):
    """Verify the target machine can run the experiment before a week of work is committed."""
    conn = connect(args.db); cur = conn.cursor()
    print(f"\n  server        {sql_one(cur, 'SHOW server_version')}")
    ok = enable_hints(cur)
    if not ok:
        print("  pg_hint_plan  MISSING or INERT -- the ceiling arms cannot run. Fix before the run:\n"
              "                apt-get install postgresql-16-pg-hint-plan, then restart the server.")
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
    ap.add_argument("--quicktest", action="store_true",
                    help="correction-depth sweep on a query subset: significance, monotonicity, "
                         "and whether the pairwise regression is a plan effect or hint overhead")
    ap.add_argument("--nq", type=int, default=20, help="queries for --quicktest/--dist")
    ap.add_argument("--card-budget", type=int, default=CARD_BUDGET_S,
                    help="seconds per query for cardinality computation. Truncation gives PARTIAL "
                         "correction, which no deployed estimator would suffer; raise this until "
                         "no query truncates.")
    ap.add_argument("--depth", type=int, default=GROSS_MAX_SIZE,
                    help="max join subquery size to correct")
    ap.add_argument("--query", default=None,
                    help="explain one query corrected vs not (e.g. --query 17d)")
    ap.add_argument("--dist", action="store_true",
                    help="per-query distribution of the correction effect under several "
                         "aggregations; the sum alone is outlier-dominated on this workload")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.quicktest:
        quicktest(a)
    elif a.dist:
        distribution(a)
    elif a.query:
        explain_one(a)
    elif a.setup:
        setup(a)
    elif a.run:
        run(a)
    else:
        ap.error("choose --selftest, --setup, --quicktest, or --run")


if __name__ == "__main__":
    main()
