# CGP pre-registration (fixed BEFORE the real-trace run)

Corroboration-Gated Prefetch (`oppcert/ladder/r7_cgp.py`), the constructive rung of the Necessity Ladder.
Committed before any wiki/cluster50 run so the gate cannot be chosen to be clearable. House rule #1.

## Hypothesis
The learnable corridor (F5: +10.41 wiki, +11.42 cluster50) is capturable by a causal policy that
buys coverage WITHOUT buying fetch volume: a byte-free wide ARM (watchlist) plus an event-triggered
tight FIRE, funded under the bar's exact byte budget. This escapes endogenous slack (F7) because
byte volume is pinned at the bar's, so the survival curve does not collapse.

## Method (frozen)
- Predictors frozen on the first 50%. Wide arm = trace's F5-clearing width (wiki k=32/top_m=32,
  cluster50 k=16). Fire = the bar's family at a low `tau_fire`, `fire_k=8`.
- Score `s(X) = conf_fire(X) * (1 + beta * armed(X))`, `armed = [wide named X within last W_arm]`.
- Fund descending `s` under a token bucket at the bar's byte rate + hard cap = bar prefetch bytes.
- Arm is recorded strictly in the past (fire decision uses arm state before this request's arming).
- No future information anywhere; evictor S3-FIFO frozen; iso-bandwidth by the bucket.

## Tuning (fixed regime)
`{beta, tau_fire, W_arm}` selected by OHR on the replay, the SAME regime the paper's bar uses to
pick its `tau, k`. Grid printed in full for transparency. Default grid:
`beta in {0,2,4}` (0 = fire-only ablation), `tau_fire in {0.02,0.05}`, `W_arm in {2000,10000}`.

## Arms (all in one run)
1. BASE (no prefetch), 2. BAR (pre-registered baseline, the paper's config), 3. BAR-CAP (bar under
CGP's token bucket; isolates burst-vs-bucket protocol cost), 4. CGP (best beta>0), 5. FIRE-ONLY
(best beta=0; the decisive ablation), F5 corridor as printed reference.

## Pre-registered gate (fixed)
CGP captures **>= +2.5 OHR pts over the BAR**, block-bootstrap 95% CI lower bound **> 0**, on
**BOTH** wiki and cluster50, at iso-bandwidth (prefetch bytes <= 1.02x bar, cold hits = 0).
- **PASS** -> the first causal policy to beat the honest bar; the paper's constructive contribution.
- **POSITIVE but < 2.5** -> partial capture, reported as such, not the headline.
- **FAIL (CI includes 0)** -> ladder rung 6; the negative hardens.

## Decisive controls (interpretation fixed in advance)
- **Arm lift = CGP - FIRE-ONLY**, each at its own tuned best. If arm lift is ~0, the arm signal is
  redundant with fire confidence and CGP collapses to Policy 1 (a fire-only threshold under a
  bucket). A PASS is only attributable to the corroboration idea if arm lift is clearly positive.
- **Re-tuned-bar adjudication (post-hoc, honest):** if CGP passes, add "recency-of-wide-emission" as
  a feature to the bar and re-tune; if that erases the gain, CGP reframes as a fifth corridor-
  inflation mechanism (under-featured bar), still a paper-positive, not a captured-corridor win.
- SYNTH selftest observed arm lift = +0.00 (structure too simple); real traces decide.

## Kill (single experiment)
At matched fetch volume, is armed-and-fire precision > fire-only precision? If not, dead.
