# Lossless preparation throughput — October 8, 2026

The user requested hardware-tuned, lossless throughput without changing the four
methods, one training seed, shared data/evaluation cases or training allowance.
Target: four A800 80 GiB GPUs, 112 visible logical CPUs, software Mesa rendering.

## Implementation

- Test goal collection simulates the complete original seeded policy rollout,
  retaining the same actions/rewards/success flags and last-success goal index.
  It then recreates the same seeded reset and replays recorded actions to render
  the goal. Only initial and selected-goal images are rendered, instead of 201.
- Spawned CPU producer pools collect independent episodes ahead of each GPU.
  Bounded queues cap retained video memory. Completed episodes may arrive out of
  order, but their IDs/seeds/content and final manifest order stay fixed.
- Each GPU batches up to 64 goal images or dense video clips. Dense clip batches
  are materialized incrementally, avoiding full duplicated history/static lists.
  Model weights, input preprocessing, bfloat16 inference and float16 cache precision
  are unchanged. Runtime batch sizes are recorded separately from data provenance.
- One asynchronous writer per GPU compresses and atomically installs files while
  collection/encoding continue. Failed writes propagate; completed files resume
  unchanged and partial files can be regenerated.
- Eight cache readers overlap hashing and per-episode statistics. Global sums
  retain the original episode order and float64 operations. Final file-hash
  verification is parallel without changing comparison semantics.

## Hardware measurements and chosen settings

All comparisons are preprocessing benchmarks, not learned-method evaluations.
See `reports/20261007-joint-flow/throughput-benchmarks.json` for exact outputs.

- 20 full-rollout equivalence cases: all 16 tasks plus four random-action cases.
  Initial/goal RGB, actions, rewards, success flags and goal indices matched
  exactly, including goals selected before the final timestep and failed episodes.
  Goal collection fell from roughly 21 seconds to roughly 1 second per episode.
- Warm CPU goal collection: 8/16/32/48 workers achieved 7.19/12.46/24.41/29.61
  episodes per second. Selected 48 total workers (12 per GPU), with two Mesa
  threads each and 24 prefetched jobs per GPU. This fits the visible CPU budget.
- Encoder sweep: batches 1/2/3/4/5/8/16/32/64/128 all produced identical stored
  float16 features on the tested images. Batch 64 reached 65.74 clips/second,
  versus 21.70 at batch 1 and 38.35 at batch 2. Batch 128 did not improve throughput
  and doubled peak memory. Selected batch 64 (about 7.39 GiB peak in this test).
- End-to-end eight-episode train/validation/test preparation was compared against
  the original cache, both with original dense batches and batch 64. Every cached
  array and logical attribute matched exactly; mean and standard deviation also
  matched bit-for-bit. Protocol metadata differs only in the isolated small test
  configuration. No main-campaign cases were changed by these tests.
- All 43 server unit/integration tests passed. The existing MetaWorld clipping
  warning remains; there were no test failures.

The live configuration is recorded in campaign.json preparation and
throughput-scope-change.json. The existing completed cache is retained. CPU-only
render rates are not claimed as end-to-end rates; live progress after migration
will be recorded in PROGRESS.md. These tests establish equivalence on the checked
cases; they are not a benchmark-success result or a universal performance proof.
