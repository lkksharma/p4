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

---

## Amendment: the endogenous-slack mechanism is withdrawn (2026-07-28)

**Found before any further measurement; recorded here before the claim was changed.**

**What was wrong.** The paper's named mechanism cited `S(100)` falling from `1.00` to `0.055`
(wiki) / `0.135` (cluster50). Those two numbers are real and correctly computed — the estimator
in `p4_hazard.py:150` is the same Kaplan-Meier used by `p4_survf7.py`, censoring on use — but they
were measured on the **wrong arm**. `p4_hazard.py:249` replays with `tau=0.0, k=32` and **no
`pf_byte_rate`**: an unmetered candidate flood logging 13,567,877 survival events on a 2M-request
trace, ~13.8x the bar's 982,954 fetches. The r6 arm the ladder actually scores is metered to the
bar (`spent 0.94x`). On that arm `p4_survf7.py` measures `S(100) = 0.984` (wiki) / `0.982` (c50).

The residence window does not close. Worse, the paper's own iso-bandwidth discipline is precisely
what *prevents* volume from being the operative mechanism, so the mechanism contradicted the
methodology — and §r6 already said so correctly ("volume is held fixed and precision alone accounts
for the collapse") while the mechanism paragraph overrode it with the flood-sourced story.

**What replaces it.** `p4_r9.py` prices the shortfall instead of naming it. Its arm holds coverage
at r4's fan-out 32, grants every genuinely-reused candidate r4's own JIT insertion, and degrades
**only** precision. Break-even precision `p* = 0.582`; r6's causal arm converts `0.062`; short by
9.4x. Timing is given away for free and the gap remains, so the failure is selection at width, not
placement. Supporting: k-sweep 2/4/8 flat at -40.8/-40.5/-40.3 while the r4 ceiling climbs
-2.66 → +4.92 (no interior optimum); demand-side hit rate unchanged to within 0.4 pts (no pollution).

**Slogan changed**: "rank-learnable is not placeable" → "rank-learnable is not budget-feasible".
Sites edited: abstract, intro, contribution (iv), Related Work (Baleen axis 2), r6, mechanism, r7,
substrate control, conclusion x2. `Figures/ladder.drawio` was already correct
("mechanism: coverage costs precision").

**Bug fixed en route.** `p4_r9.py:191` hardcoded the reference precisions (`0.05`, `0.039`) as
literals — third instance of the defect already removed from `p4_index.py` and `p4_evictlearn.py`.
They were also stale (r6 wiki is `0.062`, so the printed "11.6x" should be `9.4x`) and
trace-independent, so cluster50 would have printed Wikipedia's figure. Now supplied via `--ref`
with no default.

**⚠️ Open — page budget.** Content grew from 7 pages (97% of p7) to **7.63 pages** (63% of p8);
net +3,354 chars vs HEAD. AAAI allows 7 content pages + references. ~3,400 chars must come out,
and the local build still understates length. Decision pending on which section pays.

### Page-budget resolution (2026-07-28)

AAAI-27 is **7 pages content, 9 max, everything past p7 references only**; a technical appendix is
a **separate Supplementary Document**, so it cannot buy main-text space but costs nothing to use.
Created `supplementary.tex` (2pp, compiles clean) holding: (A) the full r9 precision sweep with both
conservatism notes, (B) the fan-out sweep incl. r3 at fan-out 1, (C) the survival table and the
explicit withdrawal of the endogenous-slack claim, (D) the inflation-mechanism table, (E) per-key-set
index error distributions. Main text now points to it in three places.

Cuts made, following this file's own recorded priority order:
- index mechanism paragraph → per-key-set numbers to supplement (priority 1)
- "why this interval" → compressed, argument intact (priority 2)
- **Table 3 (inflation mechanisms) removed** (priority 3), folded into prose as M1–M4. Side benefit:
  Related Work cited "(M1)" and "(M4)" but nothing in the paper had ever *defined* those labels.
- mechanism block merged 2 paragraphs → 1; r6/r7/substrate-control prose tightened.

**Status: content ends p8 at 40% locally (was 63% at worst, 97% of p7 before the rewrite).**
Still ~0.4 page over. ⚠️ Local build is NOT authoritative on length: `\figorbox` renders all three
figures as 2.4cm placeholders here but full size on Overleaf, so the true overage differs in an
unknown direction. **Do the final fit on Overleaf.** Remaining cut candidates, in order:
"What the ladder does and does not establish"; the LLM KV-cache paragraph detail; Table 5 (tab:decomp)
merged into prose.

### Real-build findings from AuthorKit27 (2).pdf (2026-07-28)

The Overleaf build (real figures) exposed two things the local build could not:

1. **Content ran to 66% of page 8** — over the AAAI-27 limit (7 pages content; everything past p7
   references only). Local said 40%, so the local build understated by 26 points, as warned.
2. **Two undefined citations on p6**, rendering `(?)` for the cache-conscious B-tree and
   interpolation search. Both entries (`ccbtree`, `interp`) ARE in the local `p4.bib` and resolve
   correctly here. ⚠️ **Overleaf has a stale `p4.bib` — re-upload it and recompile.** This is a
   submission blocker; nothing in the .tex needs changing.

**Cuts made for the overage** (all moved to `supplementary.tex`, nothing deleted):
- **Figure 2 (the Instrument flowchart) removed** — the four arms are fully specified in prose.
  Biggest single win, and worth ~0.2 page MORE on Overleaf than locally, because it rendered as a
  2.4cm placeholder here and full size there. Now supplement §F.
- **"On the number of positives"** → supplement §G.
- **"The recipe" fbox** → inlined as a single prose sentence; the five steps are preserved verbatim.
- "What the ladder does and does not establish" tightened.

**Local now: 8pp, content ends p8 at 3%.** Recalibrated estimate for Overleaf ≈ **7.1 pages** — the
local-vs-real gap shrank from 0.26 to ~0.055 page once the placeholder figure was removed. This is
marginal: **recompile on Overleaf to confirm.** If still over, next candidates in order: compress
"Scope and Threats to Validity"; merge Table 3 (tab:decomp) into prose; move the LLM KV-cache
paragraph detail to the supplement.

`supplementary.tex`: 2pp, compiles clean, sections A–G. Both documents: 0 dangling refs, 0 unused
labels, 0 em-dashes, 0 first-person, 0 undefined citations locally.

### Citation audit (2026-07-28) — F1 CLOSED

Every one of the 20 entries in `p4.bib` was checked against the actual published record. **Eight were
wrong.** The failure was systematic: arXiv IDs, titles, and venues were right, but author lists on
the `@misc` entries had been fabricated. All fixed:

| key | defect | corrected to |
|---|---|---|
| `coldrl` | wrong authors (Sabnis & Sitaraman) | Gupta, A.; Bhayani, A. |
| `deap` | wrong authors (Chakraborttii & Litz) | Mangal, A.; Jain, J.; Guliani, K. K.; Bhalerao, O. |
| `deepref` | wrong authors (Alghamdi, Alkhamees, others) | Alkassab, N.; Huang, C.-T.; Lorido Botran, T. |
| `lightcache` | **placeholder author** `{LightCacheRL Authors}`, no venue | Zhang, K. + 7 co-authors; NPC 2026, Springer |
| `jointhw` | two given names wrong (Sicheng→Samuel, Naveen→Nihal) | Yuan, S.; Saxena, D.; Chen, J.; Sharma, N.; Akella, A. |
| `sober` | title truncated | added ": Pitfalls and Paths to Reproducibility" |
| `mooncake` | two authors omitted | added Cui, J. and Ren, F. |
| `sosd` | wrong year (2020) | 2021, PVLDB 14(1):1–13 |

Verified correct with no change: `belady`, `cao`, `demandmin`, `lrb`, `s3fifo`, `baleen`, `vllm`,
`cachegen`, `pythia`, `kraska`, `interp`, `ccbtree`. All 20 resolve to a retrievable PDF.

### Captions + final length

`\captionsetup{font=small,skip=4pt}` added to both documents. **This alone closed the page overage:
content now ends p7 at 97%, 8pp total, 0 undefined citations.** The saving transfers fully to
Overleaf (caption text is identical there, unlike the figure placeholders), so the real build should
now fit. ⚠️ One caveat: shrinking captions below the template default is a mild deviation from the
AAAI style; it is common practice but is a judgement call.

### Overleaf build (3) findings — 2026-07-28

**BLOCKER FOUND AND FIXED: Table 4 collided with body text on p7.** `tab:index` overflowed its
column by 33.7pt and the right-column prose printed on top of the verdict column. This was the
`Overfull \hbox (33.71152pt too wide)` warning visible in an earlier local compile that was noted
and not acted on. Fixed by `\footnotesize`, `\tabcolsep` 4pt→2pt, dropping the "SOSD "/"IDs: "
prefixes (provenance moved into the caption), and "reach., uncapt."→"reach., unc.".

Two further overfulls cleared: `tab:learnable` (2pt colsep, header "(95\% CI)"→"(CI)",
`\footnotesize`). **The document now compiles with zero overfull boxes.**

**Correctness fix:** Scope and Threats still asserted r6/r7 "fail for timing and redundancy reasons"
— the withdrawn mechanism. Now "fail on precision at that width". Grep confirms 0 remaining
`endogenous`/`placeable` references anywhere.

**Length, still open.** Overleaf build (3): content ends p8 at 23%; local at the same commit read
p7 97%, so the real build runs ~26 points longer (the two remaining figures exceed the local 2.4cm
placeholders). Cuts applied since: ladder 7.6cm→6.6cm, two tables to `\footnotesize`, Scope and
substrate-control prose tightened. Expected recovery ~8–12 points, so **it will probably still be
~10–15% into p8. Verify on Overleaf.** Ranked next cuts:
1. Move "Scope and Threats to Validity" to the supplement (~17 pts) — but reviewers expect a
   limitations section in the main text; least preferred despite being the biggest single win.
2. Cap `Figures/overview` height (~10–15 pts). Note it currently has no height constraint.
3. Move "The sweep rule reverses verdicts in both directions" to the supplement (~10 pts).

⚠️ Unrelated observation: `Figures/ladder` is included with **both** `width` and `height` and no
`keepaspectratio`, so it is being geometrically distorted. If that is unintended, add
`keepaspectratio` — but note the height will then stop constraining it, since width is binding.

### Captions cut + float citation audit (2026-07-28)

Captions were 3,434 chars (~0.53 page). Two carried 62% of that and duplicated the body prose almost
verbatim — the 0.4-point margin, the byte-parity figures (56.8/27.9%), and the 290-point baseline
correction were all already stated in the text. Captions now **2,256 chars**; the detail moved to
supplementary §E. Ladder caption also trimmed.

**Local: content ends p7 at 88%, down from 97%.** Caption text renders identically on Overleaf
(unlike the figure placeholders), so this ~9-point saving transfers in full.

**All floats cited**: fig:overview 1x, tab:gatea 1x, tab:learnable 3x, fig:ladder 1x, tab:decomp 3x,
tab:index 1x. Supplement tables now labelled `tab:s-*` and each cited from its section lead.
Main text points to supplementary §A–B, §C, §D, §E, §F, §G.

**Zero overfull boxes in both documents.** Fixed en route: supplement tables A/C/E overflowed after
the column additions; table D's "mechanism" header could not fit a 1.2cm column; D and F were then
over-narrowed and hyphenating badly, now rebalanced. A residual `Overfull \vbox` warning in the
supplement is benign two-column balancing — the rendered pages were inspected and show no overflow.

### Abstract rewrite + conclusion cut (2026-07-28)

Abstract rewritten to the requested narrative, question-driven form (how much of the gap is real →
can any policy realise it → does it generalise), 1,889 → 2,632 chars. Conclusion cut 1,880 → 1,367
chars (3 paragraphs → 2). **Net +230 chars; page position unchanged at p7 88%.**

⚠️ **The supplied draft abstract stated the withdrawn mechanism** ("the cause is endogenous slack…
survival falls from near 1.0 to 0.05"). Not carried over — replaced with the break-even precision
result (58% required vs 6% achieved). The draft also ended at the LLM domain, dropping the
learned-index *build*; one clause restores it, since the discrimination claim in contribution (v)
depends on the procedure returning more than one verdict.

Every other figure in the draft was checked against the paper and is correct: 14--87%, 12.7--19.7
points, 6% of the gap (r5), 26--40 points (r6), rank correlation 0.60→0.68. "Perfect arbitration
adds under one point" was tightened to the measured $\pm 0.01$.

Both documents: 0 overfull boxes, 0 em-dashes, 0 first-person, 0 `endogenous`/`placeable`.

### Abstract completeness pass + line cut (2026-07-28)

Audited the abstract against what the paper establishes. **One claim was wrong and three things were
missing:**
- "Each falls short" implied every rung fails. **r1 recovers 91–96% of the corridor and r4 clears
  the bar** — the rungs that prove the corridor is reachable *in principle*, which is what makes the
  negative meaningful rather than a null result. Now: "the corridor is reachable in principle; every
  *deployable* ingredient then falls short."
- Added baseline saturation (verdict withheld where a tuned classical structure saturates the ceiling).
- Added the trained-model competence control ("Two trained policies bound the false-negative rate").

Nothing in the abstract is now unsupported: every figure traces to a rung in the text (r1 91–96%,
r2 ±0.01, r5 6%, r6 26–40 and 0.60→0.68, r9 58% vs 6%, Instrument 14–87% / 12.7–19.7).

**Line cut**: the abstract came out the same length once the missing content was added, so the line
was taken from the intro's results preview, which now restated the abstract almost verbatim
(276 chars, ~2.5 lines). Conclusion had already been cut 1,880 → 1,367.

Both documents: 0 overfull, 0 dangling/unused refs, 0 em-dashes, 0 first-person.

### Ladder figure rebuilt — text was illegible (2026-07-28)

Build (4) fits: References begin p8 at 0%, i.e. content ends exactly at the bottom of p7. But
Figure 2's text was unreadable. Diagnosis:

- The diagram was **664 x 788 units** (taller than wide) with 12px fonts. Rendered into an 8.6cm
  column that is a scale of 0.37, so labels printed at **~4.4pt** — under half body size.
- On top of that, `height=6.6cm` with no `keepaspectratio` **squashed it 35% vertically**, since its
  natural height at column width is 10.2cm.
- Widening to `figure*` could not fix it: at text width the old aspect gives a 21cm-tall figure.

The cause was content, not sizing: every rung carried an italic question *and* a full verdict
sentence, all of which the prose already states.

**Rebuilt as a scannable ladder**: one line per rung (label | verdict + numbers), questions dropped,
mechanism band and certificate kept. New extent **420 x 343, aspect 1.22**. Labels now render at
**~7.0pt** (a 59% increase); legend raised 10px -> 11px; the five longest verdict strings shortened
so no box wraps.

`\includegraphics` now `width=\columnwidth,height=7cm,keepaspectratio` — no distortion, height
capped. Paid for the extra 0.4cm by trimming the intro's second preview paragraph (101 chars), which
the rewritten abstract now covers in full. **Local: p7 81%, 0 overfull.**

⚠️ `Figures/ladder.drawio` must be **re-exported** to PDF/PNG for the change to appear.
Note `Figures/overview.drawio` was not touched and may have the same small-text problem — worth
checking its rendered size on the next build.

### Extension: the excluded class (online-updating prediction) — 2026-07-28

Paper names online-updating/content-aware prediction as the class the Instrument excludes and as the
productive direction, then does not test it. This run tests the cheapest member: the predictors
already train on the first half and **freeze**; letting them keep counting adds exactly one
capability.

**Pre-registered before any run**: `SPEC_online_prereg.md`. Thresholds T1 movement (precision
≥ 0.150 vs 0.062 frozen), T2 capture (corridor ≥ +2.5 vs the ONLINE bar, CI clear of zero — reuses
r7's existing gate rather than inventing one), T3 build (T2 plus precision ≥ p\* recomputed online).
Four-way fork fixed in advance.

**Critical design point (I2, iso-capability):** the tuned baseline uses the *same* online predictor.
Scoring an online learned arm against a frozen bar is mechanism **M1**, which this paper indicts in
others. Driving `p4_f7.py --pred markov2_online` gives bar, ceiling and causal arm the capability
together, by construction.

**New file `p4_online.py`** — additive only; the frozen `PREDS` entries are the pre-registered arms
and are untouched. Online counts live in a separate `_add` layer so `reset()` restores the
warm-start table exactly (I4); without this a τ×k sweep would let each cell see the previous cell's
replay. Update rule is the causal transpose of the training loop (I1).

`--leaktest` results: **I1 pass** (serving position i invariant to all positions > i, verified by
rewriting the trace tail), **I4 pass** (two consecutive replays identical), lever live (48% of
suggestions differ from frozen). A `LEVER INERT` branch aborts if online updating changes nothing.

**Bugs fixed en route:** `p4_f7.py --help` crashed with `TypeError: %o format` — two unescaped `%`
in argparse help strings (`"(% done, ...)"` and `"read -60% of F5"`). Pre-existing, unrelated to
this work, and it meant `--help` had been broken for that script.

Reporting rule (SPEC §5): results are a labelled extension and **cannot** modify Table 1/2/3 or any
pre-registered verdict, since they change the definition of "learnable" the Instrument rests on.
