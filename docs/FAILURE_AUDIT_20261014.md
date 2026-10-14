# Failure audit: base planner, predictive score, and refinement

## Scope and conclusion

Audited snapshot: `9f2e440f0b0fe2e6e5783a6c84ca2ef1f5c34dc4`.
This is a read-only review of existing evidence plus CPU numerical/graph checks.
No new model was trained, no new simulator episode was executed, and no sealed
case or active execution tree was changed. The executable audit is
`scripts/analysis/audit_leflow_failures.py`; its findings are summarized in
[the machine-readable receipt](reports/20261014-failure-audit/summary.json).

The recent inference methods have not produced a reliable success-compute gain.
That does not establish that all approaches in the source papers fail: these
experiments all reuse the same retained LeFlow adaptation on each task. The
PushT adaptation itself succeeds on only 12/50 development cases. The original
four-task, author-checkpoint comparison has not been reproduced.

The highest-priority problems are (1) the unestablished base-planner reproduction,
(2) optimization of a predicted terminal distance not yet validated for selecting
physically successful alternatives, and (3) unisolated generator/decoder/score
failure modes. No single causal action-indexing or normalization bug has been
established. A passing software test is not evidence of trained control quality.

## What the actual recent runs say

| Variant | PushT success /50 | Cube success /50 | PushT ms/decision | Cube ms/decision |
|---|---:|---:|---:|---:|
| Retained LeFlow reference |12|40|50.43|57.39|
| More samples (96) |16|37|53.75|61.24|
| More flow steps (24) |11|37|64.51|72.43|
| P&P |7|38|64.46|74.80|
| Gradient guidance |12|36|392.71|396.13|
| Guided P&P |4|37|392.90|386.14|
| Noise CEM |9|41|143.05|156.87|
| Sequential random (matched 3 x 64) |12|40|142.34|158.13|
| Shell noise CEM |10|39|143.12|156.62|
| Shell sequential random |15|39|144.93|157.75|

All 24 P&P/CEM reports account for 1,200 previously executed cases. The audit
recomputed their successes and found zero controller timeouts and unchanged
before/after model fingerprints. Every unsuccessful case exhausted its 50-action
allowance. Thus deadline truncation is not the explanation for these regressions.
The CEM Cube gain is only one case, not a stable general superiority result.

Sources: [P&P analysis](SELF_REFINE_SHARED_FAILURE_ANALYSIS_20261014.md),
[CEM results](NOISE_CEM_SHARED_RESULTS_20261014.md), and
[CEM analysis](NOISE_CEM_SHARED_FAILURE_ANALYSIS_20261014.md).

## Finding 1: early selected heads, not a converged author reproduction

The 20,000-update training budget is not the age of every checkpoint used for
inference. `crossbench/train.py` saves a new best only when success strictly
increases. Equal success keeps the earlier checkpoint.

| Quantity | PushT | Cube |
|---|---:|---:|
| Selected update |5,000|10,000|
| Selected successes |12|40|
| Global batch |64|64|
| Sampled windows by selected checkpoint |320,000|640,000|
| Eligible training start windows |1,439,235|1,408,000|
| Draws / eligible starts |0.22234|0.45455|
| Final 20k draws / eligible starts |0.88936|0.90909|
| Entire completed optimization GPUh |0.76997|0.75616|

The PushT validation curve is 12/8/9/12 at 5k/10k/15k/20k. Cube is
33/40/37/40. Early selection follows the declared tie rule; it is not a hidden
checkpoint-loading error. But it does not establish that the selected generators
are trained comparably to the source paper. The ratios above are draw-count
ratios, NOT epochs or measured unique coverage: the reader samples episodes
uniformly and windows with replacement. Overlapping windows share observations.

For PushT, at most 320,000 of 1,439,235 eligible starts could have been drawn by
selection, and duplicate draws reduce that upper bound. This does not prove that
more training will solve the problem; the later checkpoint also scored 12/50.
It does show why additional sampling techniques should not substitute for
base-model training and decoder diagnostics.

The source paper reports ten-epoch training, and the shared-execution port has
other differences: episode splitting/sampling, optimizer schedule, consistency
history, and a five-primitive execution prefix instead of the native configured
25-primitive interval. Published success rates must not be treated as a matched
control for this adaptation. Author-checkpoint access was unavailable during the
registered transfer; that limitation is preserved in the original run record.

Evidence: `docs/reports/20261013-crossbench-baselines/runs/{pusht,cube}/leflow/`
contains `identity.json`, `best.json`, `complete.json`, and `metrics.jsonl.gz`.

## Finding 2: the consistency term does not train the flow generator

The exact current loss in `crossbench/models.py::LeFlowPolicy.loss` has this
computational structure:

```text
recorded latent path -> flow-matching target -> flow generator loss
recorded latent pairs -> inverse decoder -> predicted actions -> frozen world
                                                        -> consistency loss
```

The consistency branch does not receive a generated path. Consequently its
gradient with respect to the flow parameters is absent.

The audit exposed the three tensors returned by the exact loss computations
using Python AST, without changing any of their expressions, and evaluated
connectivity with a small synthetic frozen world:

| Parameter group | Parameter tensors | Tensors connected to consistency |
|---|---:|---:|
| Flow generator |63|0|
| Inverse decoder |14|14 (all nonzero in this probe)|

This is a graph fact, not a trained-checkpoint performance experiment. The
consistency term regularizes the inverse decoder on recorded paths. It does not
teach generated paths to be reachable. Descriptions of this implementation as
training the flow generator through predictive feedback would be incorrect.

**Important boundary:** the inspected upstream `hsiangwei0903/LeFlow`
`train_latent_planner.py::step_batch` also computes consistency from recorded
paths and inverse-decoded actions. Therefore this is not an ours-only regression
and cannot by itself explain the reproduction gap. Adding a flow consistency
term would be a changed training algorithm, not a proven one-line bug fix.

There IS an implementation difference in the consistency history: our port
flattens transitions into independent single-state examples; upstream
`lewm_consistency_loss` supplies up to three recorded state/action blocks.
Inference in our adapter also uses up to three states after rollout warmup.
This should be included in a fidelity check. Its effect has not been isolated.

## Finding 3: the search is optimizing its score, not demonstrated success

Recorded median within-decision predicted-cost reductions are:

| Search | PushT | Cube |
|---|---:|---:|
| CEM |28.6%|30.3%|
| Independent sequential random |8.8%|13.8%|

Yet CEM succeeds on 9/50 versus random 12/50 on PushT and 41/50 versus40/50 on
Cube. The search is not inert. The issue is that optimizing this score more
strongly has not improved task completion consistently.

The score is terminal MSE between the encoded goal and the world-model rollout
of decoded actions. It is NOT distance to the clamped generated endpoint, and
it is NOT a measured task-success probability. A plan can reduce this score by
exploiting model error, privileging irrelevant visual dimensions, or relying on
an unexecuted suffix. Current data do not distinguish those mechanisms.

Each plan predicts 25 primitive actions but only the first five are executed
before replanning. That is a model-predictive-control choice, not inherently a
bug. It does require a useful predictor and continuation for the terminal score
to rank the executed prefixes correctly. The histories of rejected alternatives
were not executed, so the records do not supply counterfactual outcome labels.

Across-controller prefix-error and terminal-score medians use different visited
states and numbers of decisions. They must not be used to claim a causal increase
in predictive accuracy or proof of verifier hacking.

## Finding 4: P&P can discard a good initial plan

The tested P&P variants replace the full candidate pool. They do not preserve
the original best plan, so prior-only refinement can remove a previously useful
option. Plain P&P regresses without any world-gradient correction.

CEM does retain evaluated incumbents, but is still not reliably better. Thus
keeping the incumbent is a useful control, not a sufficient cure: the scorer
can still replace a physically better plan with a lower-scoring bad one.

## Direct visual example (limited scope)

I inspected the two fixed first archived PushT cases for five controllers each,
using only the six saved frames and goal per trajectory. This is ten saved
trajectories, not visual inspection of all 1,200 outcomes.

Case `pusht/val/0002481/000039`:

| Controller | First-decision selected predicted terminal MSE | Physical outcome |
|---|---:|---|
| Reference |0.01908439|Success after43 actions|
| CEM |0.01763373|Failure at50 actions|
| Sequential random |0.01908439|Success after23 actions|

The CEM trajectory eventually moves the blue pusher away toward the lower left,
while the object barely changes in the later saved frames. The reference reaches
the requested image goal. The goal is the saved target image, not necessarily
the green canonical marker drawn in the scene.

This is a full-policy comparison, not proof that the first CEM choice alone
causes failure: subsequent replanning differs. It illustrates that these are
real control failures, not jobs merely failing to finish or fabricated counters.
Source paths and numerical values are in the audit receipt.

## What the checks do and do not establish

- Verified 856 exported source/evidence file SHA256 hashes; no mismatch.
- Re-ran 185 CPU tests and ten subtests successfully, covering noise CEM,
  refinement, adapters, and protocol regressions.
- Recomputed 24 complete saved reports, 1,200 outcomes, zero timeouts, and frozen
  model fingerprints.
- Inspected action-block ordering, unit/physical/world-normalized conversions,
  endpoint scoring, inference-mode gradient routing, and CEM selection. No
  reversed-score, constant clamped-goal score, or action-block transpose bug was
  demonstrated by this audit.

These checks do not validate learned candidate quality, inverse-decoder accuracy
on generated paths, simulator/source-state fidelity across the real dataset, or
the world score's ability to rank action alternatives. Exact head/HDF5 replay was
not done: those large files were not present in the exported repository evidence.

To rerun the read-only audit on the inspected code and evidence snapshot:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/analysis/audit_leflow_failures.py \
  --evidence-root /path/to/9f2e440-source-and-evidence \
  --output /tmp/leflow-failure-audit.json
```

The script must be run with the audited production source. Later source changes
require reinspection; the script's recorded commit label is not a source checkout.

## Next fault-isolation gates, not another method campaign

1. Replay recorded TRAIN actions from their restored recorded states. Compare
   observed successor states/images and native success to the stored trajectory.
   This isolates reset, action-time alignment, normalization, and camera effects.
2. Feed recorded latent paths through the learned inverse decoder and physically
   execute the decoded prefixes. This bypasses the flow sampler. Failure here
   implicates the decoder/representation or interface before any noise search.
3. From identical starts, compare the same generated candidates with first/random
   selection, existing world-score selection, and diagnostic-only simulator
   outcome selection. This separates missing good proposals from bad ranking.

These are proposed bounded diagnostics, not executed experiments or authorization
for a new GPU queue. Retain the original cases/results and sealed-test separation.
Only after locating the bottleneck should training or the planner be changed.

## Research-process correction

The wrong sequence was to keep adding recent samplers before establishing base
reproduction and decision-quality diagnostics. Counts of passing tests, intact
hashes, and completed GPU jobs establish operational correctness, not a scientific
contribution. The right deliverable now is a trustworthy base pipeline and one
isolated failure mechanism, not another stack of named methods.
