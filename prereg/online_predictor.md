# SPEC: online-updating prediction, the excluded class

**Registered 2026-07-28, before any online-updating arm was run on any trace.**
Nothing below may be changed after a result is seen, in either direction.

---

## 0. Why this run exists

`paper_aaai.tex` (Scope and Threats) states that "learnable" is defined against **frozen,
history-based** prediction, and that an online-updating or content-aware predictor "could reach cold
objects and would raise the surviving corridors". The conclusion goes further and names the target:
a predictor reaching roughly **0.58 precision at the oracle's coverage**.

The paper therefore names an escape hatch and does not test it. This run tests it. The cheapest
member of the excluded class is the one already implied by the existing predictors: they train on
the first half of the trace and **freeze**. Letting them keep updating during evaluation adds
exactly one capability and nothing else.

This is a **capability extension, not a correction**. The pre-registered verdicts stand as measured;
this run cannot alter them (see §4).

---

## 1. The arms

All three use the same predictor class, so the added capability is shared:

| arm | predictor | emission | timing |
|---|---|---|---|
| O-BAR | online-updating, tuned $(\tau, k)$ | narrow (bar config) | causal |
| O-R4 | online-updating | fan-out 32, no floor | clairvoyant JIT (ceiling) |
| O-R6 | online-updating | fan-out 32, no floor | causal hazard model |

The predictor is **warm-started from the same first-half prefix** the frozen arms use, then keeps
updating. This isolates online updating from "saw more data".

---

## 2. Pre-registered thresholds

Frozen reference values, from the logs, on `wiki_2019t`: r6 realised precision **0.062**, r6 corridor
**−40.06**, break-even precision **p\* = 0.582**.

- **T1 (movement).** O-R6 realised precision at fan-out 32 $\ge$ **0.150**, i.e. more than double the
  frozen arm, with a block-bootstrap CI lower bound above 0.062.
- **T2 (capture).** O-R6 corridor against **O-BAR** $\ge$ **+2.5** points with a block-bootstrap CI
  lower bound above zero. (2.5 is the gate already pre-registered for r7; it is reused unchanged
  rather than invented for this run.)
- **T3 (build).** T2 holds **and** O-R6 realised precision $\ge$ the break-even precision recomputed
  under online updating (`oppcert/ladder/breakeven_precision.py --pred *_online`). p\* must be recomputed, not assumed to stay at
  0.582, because the ceiling may move.

## 3. Pre-registered fork

- **T3 met** → the excluded class closes the gap. The flagship domain has a **build**, and the
  paper's negative is correctly scoped to frozen history-based prediction, which is what it already
  claims. This is the outcome that would most change the paper.
- **T2 but not T3** → partial capture; the corridor is reachable but not fully, and p\* remains the
  standing specification.
- **T1 only** → the direction the paper names is real but insufficient. p\* is validated as a live
  target rather than an unreachable abstraction.
- **None met** → the negative hardens materially, because the one class flagged as the escape route
  was tested and did not move. Report as such; do not re-sweep looking for a better cell.

## 4. Invariants (a violated invariant aborts the run, it does not get explained away)

- **I1 causality.** Serving position $i$ may use no observation at position $> i$. Online updates
  credit the arriving request as a successor of requests already seen, never the reverse.
  Asserted by `--leaktest`.
- **I2 iso-capability.** The tuned baseline uses the **same** online predictor. Scoring an
  online-updating learned arm against a frozen baseline is mechanism **M1**, which this paper
  indicts in others; committing it here would invalidate the result.
- **I3 iso-bandwidth.** Unchanged. The causal arm is metered to O-BAR's measured prefetch byte rate.
- **I4 reset hygiene.** Online counts accumulated during one replay must not carry into the next.
  Counts added online live in a separate layer that `reset()` clears; the warm-start prefix table is
  never mutated.

## 5. Reporting rule

Results appear as an explicitly labelled extension ("the excluded class"). They **do not** modify
Table 1, Table 2, Table 3, or any pre-registered verdict, because they change the definition of
"learnable" that the Instrument is built on. If T3 is met, the correct edit is to state that the
trap verdict holds for frozen history-based prediction and that a named, tested alternative escapes
it, not to retroactively relabel the original measurement.

## 6. What this run cannot show

Online updating is one member of the excluded class. Content-aware prediction is the other and is
not tested here. A negative on this run bounds the frozen-prediction negative more tightly; it does
not close the content-aware direction, and the paper must continue to say so.

---

## 7. Amendment (2026-07-28) — first run aborted, two design errors in this spec

Written after the first online arm was run on `wiki_2019t` and **before any further measurement**.
The first run's result was **unfavourable** (precision 0.029 against 0.062 frozen, T1 not met). That
is stated here deliberately: the corrections below are forced by design defects that would have
applied identically had the result been favourable, and the run is being discarded rather than
banked as a negative.

### 7a. The cold-hit invariant is structurally incompatible with the capability

`oppcert/ladder/r6_placement.py` asserts `pf_cold_hits == 0`: an arm may never prefetch an object not previously
requested. A **frozen** predictor satisfies this by construction, because its vocabulary is the
training prefix. An **online** predictor cannot, because updating during evaluation lets it name
objects first seen after the cut. The run failed with **663 cold hits**.

This is not a bug and it is not a violation to be waived through. It is precisely the behaviour the
paper predicts: *"an online-updating or content-aware predictor could reach cold objects and would
raise the surviving corridors."* The invariant was carried into §4 unexamined from the frozen
setting and is simply wrong for this class.

**Resolution, fixed before re-running.** Online arms are scored on the **gross** corridor, with cold
hits permitted and counted, and the cold slice reported separately, because for this class reaching
cold objects is part of the capability rather than an accounting error. The warm-restricted figure
is reported alongside so the two regimes stay distinguishable. Neither may replace a pre-registered
warm figure in Tables 1--3; §5 is unchanged.

### 7b. The hazard model was out of distribution

`haz/wiki.npz` is fit on the **frozen** predictor's emissions. Applying it to online emissions
under-tunes the learned arm, which is the mirror image of **M1** — the error this paper indicts,
aimed at the learned side instead of the baseline. The paper swept a 16-cell family in r8 precisely
so that no negative would rest on one badly fitted model; scoring the online arm with a timing model
fitted to a different emission distribution abandons that discipline.

Evidence it bit: the aborted run reports **on-time 100.0% / LATE 0.0%**, a timing profile unlike any
frozen arm, together with precision 0.029. A hazard model that never fires late while converting 3%
of its fetches is not describing the stream it was asked to schedule.

**Resolution.** A hazard model must be refit on the online predictor's own emissions, at the wide
setting actually used ($\tau=0$, fan-out 32), before the causal arm is scored.

### 7c. Status

The first run is **aborted**. Its numbers are not evidence about T1, T2 or T3: a run with a failed
invariant and an out-of-distribution timing model is not a measurement of the hypothesis. Thresholds
T1--T3 and the §3 fork are unchanged and remain binding on the re-run.
