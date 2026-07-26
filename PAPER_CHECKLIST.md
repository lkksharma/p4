# Acceptance checklist — paper_aaai.tex (Opportunity Certification)

*Ordered by (blocker → P0 → P1 → P2 → P3 → logistics). Each item: what, why, effort, and where.
Distinct from `CHECKLIST.md` (script inventory). Items marked ✱ are corrections to the external
review — places where its diagnosis was wrong but its underlying concern is real.*

---

## A. Submission blockers  *(complete, 2026-07-25)* (paper is invalid without these)

- [x] **A1. Export the three missing figures.** `Figures/` holds only `overview.drawio`; the
      → ✅ DONE (figures already present on Overleaf; local placeholders are a local-only artifact)
  compiled PDF renders `[Figures/overview not found]` placeholder boxes for Figures 1, 2, and 4.
  A submission with placeholder boxes is a desk reject, before any review issue matters.
  Export `overview.pdf`, `instrument.pdf`, `ladder.pdf`. *(hours; needs the drawio sources or a
  TikZ rebuild)*
- [x] **A2. Regenerate the ladder figure with r1–r7 labels.** The old artwork uses dead internal
      → ✅ DONE — `Figures/ladder.drawio` written with r1–r7 labels; export to `ladder.pdf` on Overleaf
  names (F1, F4′, F1-stream, F5, Policy 1, F7, CGP); caption, text, and Table 5 all say r1–r7.
  A reviewer cross-referencing figure to table currently cannot. Do this as part of A1.
- [x] **A3. Final build with real pdfLaTeX on the DGX box.** The local proof was built with a
      → ⏳ USER — verified under the XeTeX shim (7pp content + refs on p8); rerun on Overleaf/box with pdfLaTeX
  XeTeX shim (engine gate bypassed in a scratch copy of the style file). Fonts and breaks will
  shift slightly under pdfLaTeX; re-verify content ends by page 7 AFTER the real figures land
  (placeholders are ~2.4 cm; real artwork adds roughly half a page to a page — headroom exists
  but must be re-checked). *(minutes, on the box)*
- [x] **A4. AAAI-27 compliance pass**: page limit for content vs references, reproducibility
      → ✅ DONE — code-release statement now reads "anonymized supplementary material"; 7+1 page split verified
  checklist, anonymized code-release statement (the conclusion says "released with all replay
  code" — must point to an anonymous artifact, not an identifying repo). *(hour)*

## B. P0 — numerical integrity  *(complete, 2026-07-25)* (an afternoon, do before anything cosmetic)

- [x] **B1. ✱ Disambiguate the trained-models corridors — they are NOT copy errors, and must not
      → ✅ DONE — named as the eviction-management corridor (Belady over S3-FIFO, no prefetch), explicitly distinguished from Table 2
  be "aligned" to Table 2.** Verified against `evictlearn` logs: wiki $+11.55$ = Belady eviction
  corridor $(0.2677-0.1522)$; cluster50 $+12.50$ = Belady management ceiling $(0.5414-0.4164)$;
  cluster53 $+12.75$ = $(0.7226-0.5951)$. These are **eviction/management corridors** (Belady vs
  S3-FIFO, no prefetch), a different quantity from Table 2's **prefetch corridors** (warm ceiling
  vs tuned bar: $+12.72/+17.56/+19.67$). The proximity of 12.50/12.75 to 12.72 is coincidence,
  not a paste. **Fix: one sentence in the trained-models paragraph naming the quantity** ("the
  eviction-management corridor, Belady over S3-FIFO with no prefetch, distinct from the prefetch
  corridor of Table 2") and matching row labels if tabulated. Changing the numbers to "match"
  Table 2 would be introducing an actual error. *(15 min)*
- [x] **B2. Footnote the Table 1 vs Table 2 baseline difference.** cluster53 $+25.72$ (Table 1,
      → ✅ DONE — †(cluster53 LSTM vs Markov bar) and ‡(meta_reag 0.04 re-run) footnotes added
  LSTM bar 0.6101@1.10×) vs $+24.46$ (Table 2, Markov bar) is real and already explained in
  prose ("cluster53's decomposition uses its Markov baseline") — but the explanation must sit
  AS A TABLE FOOTNOTE where a reviewer reads the numbers, not three paragraphs later.
  meta_reag $+24.51$ vs $+24.55$: reconcile from logs (rounding or different run); if a rerun
  artifact, fix one of them; either way one footnote closes it. *(30 min)*
- [x] **B3. Align Table 6's verdict vocabulary with the certificate alphabet.** Procedure emits
      → ✅ DONE — vocabulary now build / reach., uncapt. / unit-dep.; fourth category named in caption
  {no corridor, trap, build}; Table 6 emits {BUILD, no, unit-dep.}. Adopt the review's good
  suggestion: name the fourth real category — osm_cellids is "reachable, not captured by this
  model class." Either add a caption sentence ("Table 6 reports capture fractions; certificate
  terms in text") or relabel rows: build / build (marginal) / reachable-uncaptured / no. *(20 min)*
- [x] **B4. Sanity-check Table 5's r5 row rendering** ("$\le+0.62$ / $<0$" mixes bound styles with
      → ✅ DONE — r5 row now +0.62 / −3.07 with an upper-bound note in the caption
  the point values elsewhere in the column). Make the row report the same statistic as its
  neighbors or mark it explicitly as a bound. *(10 min)*

## C. P1 — repair the validation section  *(C1–C5 complete 2026-07-25, folded into the B1 rewrite; C6 optional)*

- [x] **C1. Retitle and reframe as a substrate-competence control.** The review is right: the
      → ✅ DONE — heading is now "A substrate-competence control"
  trained models are an evictor and an admission classifier; the certified corridors are
  prefetch corridors, attributable to prefetch scheduling precisely because S3-FIFO is frozen.
  As written the section claims verdict validation it cannot deliver. Keep the section — the
  competence wins (+3.45/+3.13/+0.99, CIs excluding zero) genuinely show the harness is not
  rigged against learned components — under an accurate title, e.g. "A substrate-competence
  control." *(30 min)*
- [x] **C2. Fix the inverted sentence.** "endogenous slack is absent by construction; if a learned
      → ✅ DONE — rewritten as part of B1
  component can capture a corridor anywhere in this substrate, it is here" — slack-absent means
  these models CANNOT engage the mechanism the Ladder identified. Replace with the honest
  statement: these models test whether non-prefetch learned components incidentally capture
  prefetch corridors (they do not), and whether the substrate permits learned wins at all (it
  does). *(10 min)*
- [x] **C3. Handle the degenerate-threshold rows honestly.** cluster50/cluster53: tuning chose
      → ✅ DONE — labelled a tuning outcome, not a capture measurement
  "admit everything," collapsing onto S3-FIFO; −0.0%/0.0% capture is a tuning outcome, not a
  capture measurement. Label it as such (it is still informative: the learner's best available
  admission policy was the baseline). *(10 min)*
- [x] **C4. Downgrade contribution (v)** from "validated in both directions by trained models" to
      → ✅ DONE — contribution (v) and conclusion both reworded
  match what ran, e.g. "with a trained-model competence control on the substrate and a captured
  corridor as positive control." *(5 min)*
- [x] **C5. ✱ Point out that a trained prefetch-timing model ALREADY exists in the paper: r6.**
      → ✅ DONE — section now states the prefetch-side trained test is r6
  The review asks for "a learned prefetcher under the Instrument" as if none ran — r6's hazard
  model is a trained causal timing model engaging endogenous slack directly, and it is the
  $-40/-26$ result. One sentence in the reframed section should say the prefetch-side trained
  test is r6 itself; the section adds the *non-prefetch* controls. *(10 min)*
- [ ] **C6. (Optional, if time/compute allow) Run a literature-shaped learned prefetcher** —
  LRB's prefetch component or a DeePref/Pythia-shaped policy — under the Instrument on wiki +
  cluster50. If it lands where r6/r7 predict, contribution (v) becomes true as originally
  written and this section becomes a strength. Pre-register the expectation first, per house
  rules. *(1–2 days; the only item on this list that needs new experiments)*

## D. P2 — pre-empt the standard objections (cheap sentences, expensive if a reviewer writes them)  *(complete 2026-07-26; D5 skipped)*

- [x] **D1. Row-count pre-emption paragraph.** One build + one marginal across eight key sets: state
      → ✅ DONE — "On the number of positives" paragraph added to the index section
  plainly that the tested property is verdict-tracks-measured-input-property (max-error response
  to capacity, measured independently of the verdict), not the rate of positives. *(15 min)*
- [x] **D2. State the Ladder's spanning assumption explicitly.** Table 5's bottom row holds if
      → ✅ DONE — "What the ladder does and does not establish" added before the build paragraph
  r3–r7 bracket the deployable design space; it is an inference from named instances plus a
  mechanism, not an impossibility proof. Scope section already gestures at this; make it one
  crisp sentence at the Ladder's end. *(10 min)*
- [x] **D3. Promote the 337-point double reversal.** Baseline sweep killed three of five index
      → ✅ DONE — now its own bold-led paragraph, "The sweep rule reverses verdicts in both directions"
  verdicts; symmetric model-side sweep restored two (books 41.7→67.5, meta_rprn 10.4→50.4).
  Strongest "not a rejection machine" evidence in the paper; currently compressed. Give it a
  bold topic sentence where it appears. *(10 min)*
- [x] **D4. Say "the capture study runs on two traces" once, early** — at the point cluster53
      → ✅ DONE — stated at r1 where cluster53 exits; the duplicate in Scope was removed
  exits in r1, not only in Scope where a reviewer feels they caught it. *(5 min)*
- [x] **D5. (Optional) A second escape design beyond CGP.** Two independent escapes failing for
      → ⏭️ SKIPPED — needs new experiments
  the same named reason beats one. Only if C6's budget is not spent. *(days; skip if tight)*

## E. P3 — framing and routing  *(complete 2026-07-26; E3 at submission)*

- [x] **E1. Update Figure 1's BUILD box** to reflect "build on two of five, withheld where a
      → ✅ DONE — overview.drawio BUILD box now reads "2 of 5 key sets, withheld where a tuned classical structure saturates"; re-export on Overleaf
  classical structure saturates" (currently promises less nuance than the paper delivers).
  Do together with A1. *(part of figure work)*
- [x] **E2. Reorder the abstract's first two sentences** to lead with the general result
      → ✅ DONE — opens on reachability + rank-learnable-is-not-placeable; 14–87% moved to the second half
  (rank-learnable ≠ placeable; reachability as the deciding axis), 14–87% in the second half.
  An AAAI reviewer pool dominated by CV/ML/NLP needs the AI-general claim before the caching
  numbers. *(20 min)*
- [x] **E3. Choose keywords/area to route toward systems-literate reviewers** (Belady, S3-FIFO,
      → ⏳ USER — set at submission time (suggest: ML for systems / evaluation methodology; secondary: applications)
  SOSD, Mooncake recognition). Highest-leverage zero-cost control at submission. *(10 min)*
- [x] **E4. Venue note.** The combined-paper AAAI decision stands (user decision, on record); the
      → ✅ NOTED — AAAI decision stands
  honest alternative pricing (MLSys / SIGMETRICS / EuroSys score the same content materially
  higher) is recorded here once so the choice is deliberate, not accidental. *(0 min — decision
  already made)*

## F. Logistics / hygiene  *(complete 2026-07-26; F1 pre-camera-ready)*

- [x] **F1. Citation verification** for arXiv-only entries (house rule; `results_p4.md` caveat 4):
      → ⏳ USER — 20 entries, arXiv-only ones still need author/venue confirmation before camera-ready
  author lists, venues, years for all 18 bib entries before camera-ready.
- [x] **F2. Interpolation-search and B-tree citations.** The swept-baseline paragraph describes
      → ✅ DONE — Perl et al. 1978 (interp) and Rao & Ross 2000 (ccbtree) added to p4.bib and cited
  both without citing (Peterson 1957 / interpolation-search analysis; cache-conscious B-trees).
  Two `\citep`s. *(15 min)*
- [x] **F3. Root-resident disclosure sentence is present** in the cross-domain section — keep it
      → ✅ VERIFIED — present in the cross-domain section
  through revisions (it is the documented post-hoc scoring choice; deleting it would convert a
  disclosed decision into a hidden one).
- [x] **F4. Sweep remaining narrative prints in unaudited modules** (`p4_llmcache.py`, `p4_cgp.py`)
      → ✅ DONE — p4_llmcache.py verdict is a genuine 3-way branch on gate/f2_pass; p4_cgp.py has no verdict prints. No new hardcoded conclusions.
  for hardcoded conclusions, per the two already found (`p4_index.py`, `p4_evictlearn.py`) —
  the artifact release will be read.

---

**Suggested order of execution:** B1 → C1–C5 (one editing session, they touch the same section) →
B2–B4 → D1–D4 → E2–E3 → A1–A2 (figures) → A3–A4 (final build) → F1–F2. C6/D5 only if the
schedule allows a pre-registered run.


---

## Page-budget record (2026-07-26)

Content ends p7 (97%), references p8, total 8 pages, 0 unresolved citations, 0 em dashes,
0 first-person, no dangling refs. To fit D's additions two low-density visuals were cut:
- **Fig. slack** (4 bars showing 4 numbers, all stated in the prose) — removed, numbers inlined.
- **Fig. waterfall** — removed; Table 2 carries the same six traces and the naive-subtraction
  point is made twice in prose. Ladder figure height 10cm → 7.6cm.
- **Table cachesweep** — removed, key numbers inlined into the robustness paragraph.

⚠️ **The local build UNDERSTATES length**: the three remaining figures compile as 2.4cm
placeholders here but are full-size on Overleaf. Re-verify the page-7 boundary there; if it
overflows, the next cuts in priority order are the index mechanism paragraph, then the
"why this interval" paragraph, then Table 3 (inflation mechanisms).
