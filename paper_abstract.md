# AAAI-27 — Abstract draft (v1, 2026-07-19)

## Title (recommended)

**Phantom Headroom: How Much Prefetch-Timing Gain in Caching Survives an Honest, Bandwidth-Matched Baseline?**

*Alternates:*
- The Warm Ceiling: Measuring Learnable Prefetch-Timing Headroom in Caches at Iso-Bandwidth
- Most Prefetch Headroom Is a Mirage: A Pre-Registered, Bandwidth-Matched Instrument for Cache Prefetching

## Abstract (~235 words)

Learned and oracle-guided prefetchers are credited with large cache hit-rate gains, but these are
typically measured against under-tuned baselines and at unmatched prefetch bandwidth. We introduce
a pre-registered, bandwidth-matched instrument that measures how much of the apparent headroom a
real scheduler could actually capture. It compares the best *tuned* decoupled prefetcher —
Markov-1/2/3 and an LSTM, fully swept over threshold and degree — against a clairvoyant-timing
ceiling at identical prefetch bandwidth, then deflates that corridor by separating hits a
history-based policy can reach (warm, in-training-vocabulary objects) from the elimination of
compulsory misses reachable only by clairvoyance (out-of-vocabulary objects).

Across 12 production traces spanning CDN, key-value, and block storage, gross corridors of +20 to
+33 hit-rate points appear large, yet 60–90% of each is unreachable compulsory-miss elimination.
Under a pre-registered 8-point bar, genuine *learnable* timing headroom survives on only 3 of 12
traces — one CDN and two key-value — where it measures +12.7 to +19.7 points with bootstrap 95%
confidence intervals bounded away from the bar. The remaining nine die: a tuned predictor already
saturates the reachable ceiling, or the corridor is clairvoyant mirage. A 2×2 interaction study
(n=6) finds eviction and prefetching act as substitutes, not complements. We argue prefetch-timing
gains must be reported against a learnable, bandwidth-matched ceiling — not a gross clairvoyant one
— and release the instrument that does so.

## Contributions (for §1)

1. **A pre-registered, iso-bandwidth instrument** for prefetch-timing headroom that isolates the
   *learnable* corridor via a warm (in-vocabulary) vs. cold (out-of-vocabulary) ceiling split. Bar
   and warm-ceiling cold hits are zero by construction — a verifiable invariant.
2. **A deflationary empirical result on 12 production traces:** gross timing corridors are 60–90%
   unreachable compulsory-miss elimination; honest learnable headroom clears a pre-registered
   8-point bar on only 3/12 traces (bootstrap-confirmed), spanning CDN and KV.
3. **A mechanism:** learnable headroom = exploitable warm reuse × the tuned predictor's failure to
   time it; dead traces are where a tuned predictor already saturates the warm ceiling.
4. **A decoupling result (n=6):** eviction and prefetching are substitutes, not complements —
   evidence against joint eviction+prefetch co-training and for a standalone timing layer.

## Pre-submission checklist (do NOT submit without)

- [ ] Verify Baleen (FAST'24), Demand-MIN (ISCA'18), A-Sober-Look (arXiv 2504.07086) against source.
- [ ] Eyeball cluster53 per-block corridor (bootstrap SE 0.13 — unusually tight).
- [ ] Confirm "when not what" appears nowhere in abstract/intro (Gate C: slogan pre-empted).
- [ ] State n=3 live honestly; lead with the instrument, not the trace count.
