# Native LeFlow self-refinement: implementation and execution status

## Publication update (October 13, 2026)

The archived `self_refine/` implementation, six comparison configurations, and
83-test suite have now been published through the GitHub write connection.
Executable source, test, and configuration files are unchanged from the archive.
The retry reran all 83 CPU tests successfully. Existing `leflow_refinement/`,
`crossbench/`, native runtime files, and completed results are preserved.

**Publication does not mean the GPU comparison ran.** This retry is a source
publication task. No new training or native benchmark evaluations were launched,
and the real-checkpoint/CUDA gates below are still required. See
`SELF_REFINE_PUBLICATION.json` for the retry validation record. The earlier
access-failure description below records the originating session, not current
GitHub write availability.

## Original-session status (historical)

This is an **unpushed implementation**, not a completed experiment. It was
written against inspected repository revision
`8fea74f96c867c9aa19ac763b85b22f11d62d3d0` in `Abecid/LeFlow-`.
The available session could read GitHub, but had no GitHub write actions,
GitHub network/DNS, configured `target_server_2` SSH connection, native
simulator dependencies, or CUDA. A remote push access check failed.

CPU synthetic/unit tests are recorded in `SELF_REFINE_EXECUTION_STATUS.json`.
They are **not** native simulator evaluations or evidence of task success.
The available CPU PyTorch was 2.10.0+cpu; the repository pins 2.8.0. The real
checkpoint/CUDA parity preflight remains required before qualification.
No baseline was retrained, no GPU job was started, no completed report was
modified, and no sealed final benchmark was read.

## Implemented scope

| Component | Implementation |
| --- | --- |
| Native-reference sampler | Original linear-flow Euler arithmetic and terminal-rollout scoring |
| P&P | Full-path fixed-noise-time Predict-and-Perturb inner iterations |
| Guidance-only control | Bounded gradient correction to the estimated clean interior |
| Proposed method | World-feedback correction followed by P&P and another flow prediction |
| Frozen model adapter | Original inverse decoder, action encoder, autoregressive predictor, history size 3 |
| Execution integration | A separate native evaluation entry point using existing WorldModelPolicy and native evaluation helpers |
| Reproducibility | Separate initial/refinement RNG streams, real case IDs, source/data/checkpoint hashes |
| Reporting | Per-case actual episode success, paired wins/regressions, full compute accounting and measured latency |
| Run control | Up to four GPUs inside the visible allocation, finite timeout, locks, no implicit failed-case replay |

This version deliberately does **not** implement selective video masks,
adaptive compute, learned RepFusion conditioning, retrieval, a critic, or
TTT. None is silently included in the comparison.

### No new training phase

The method changes sampled latent plans, not model weights. It reuses one
native LeFlow planner plus its associated LeWM checkpoint for all six variants.
`training_updates=0` is explicit in manifests. The complete model state,
including buffers, is hashed before and after evaluation. Missing checkpoints
cause a visible failure; the runner does not silently retrain a different model.

When a native planner checkpoint truly does not exist, the repository's
existing `train_latent_planner.py` remains the shared-checkpoint training entry
point. Train it once with the original task data and native configuration,
then point **all** inference variants at that same checkpoint. That preparatory
training was not executed in this session and is not a new method-specific loss.

## Formulation and tensor flow

Let B be environments per solver batch, N the number of candidates, H the
number of latent transitions, and D the checkpoint's latent dimension.
The noisy path interior is `[B*N,H-1,D]`. Current and goal states are `[B*N,D]`
and are not optimization variables. Appending them produces `[B*N,H+1,D]`.
The native inverse model maps each adjacent pair to a normalized action block.
No new clipping, scaling, proprioception, reward, or physical-state input is
introduced into the controller.

Flow time t goes from zero (noise) to one (clean):

```
clean = noisy + (1 - t) * flow(noisy, t, current, goal)
re_noised = t * clean + (1 - t) * independent_gaussian_noise
```

P&P performs the reconstruction/perturbation at the same t. It then uses the
last velocity to advance one ordinary Euler interval. Do not reverse the
noise convention when transferring settings from a video-diffusion scheduler.

For the guided method, let `predicted` be the LeWM autoregressive rollout of
actions decoded from `[current, clean, goal]`. The guidance cost is

```
goal_MSE = mean((predicted[:, -1] - goal) ** 2, feature_axis)
consistency_MSE = mean((predicted[:, 1:-1] - clean) ** 2, time_and_feature_axes)
cost = goal_MSE + consistency_weight * consistency_MSE
```

The gradient is taken with respect to `clean` only. Each interior token's
L2 correction is bounded by `radius` per inner iteration. Explicitly disabling
an outer `torch.inference_mode()` and cloning normal tensors allows gradients
through frozen model operations. Differentiable microbatches bound memory.

The corrected clean estimate is re-noised and denoised again for guided P&P.
The guidance-only control instead converts the corrected clean estimate back
to a velocity without drawing new noise. All methods use the same native
**terminal-only** rollout score for final selection: adding the consistency
term to final ranking would be a second, confounded change.

The correction bound is not a bound on final actions after re-noising.
Reducing model cost does not guarantee physical improvement or safety. The
world and inverse decoder can share errors. These are hypotheses to evaluate,
not guarantees reported by this implementation.

## Registered initial comparison

Four original benchmark configurations are loaded without changing their
horizon, action block, receding horizon, goal offset, reset callables or action
normalization. They are Reacher, PushT, TwoRoom, and OGBench Cube. No custom
MetaWorld result is pooled with this table.

The initial configuration uses one evaluation-case seed (42), the same policy
seed (1234), 50 cases per task, and one environment per timed solver batch.
This is a 24-job development comparison (1,200 cases), not a multi-seed SOTA
claim and not the custom 3,200-case sealed final set.

| Method | Candidates | Euler steps | Extra inner loops |
| --- | ---: | ---: | ---: |
| leflow | 64 | 16 | 0 |
| leflow_more_samples | 96 | 16 | 0 |
| leflow_more_steps | 64 | 24 | 0 |
| pnp | 64 | 16 | 8 |
| guidance_only | 64 | 16 | 8 gradient corrections, no extra flow predictions |
| guided_pnp | 64 | 16 | 8 gradient corrections and 8 extra flow predictions |

Extra loops occur twice each at indices 4, 8, 12, and 14 of the 16-step sampler.
The initial radius is 0.05 and consistency weight 0.1; these are fixed starting
hyperparameters, not values selected from measured benchmark outcomes.

The P&P and guided-P&P generators use 1,536 candidate-level flow evaluations,
matching 96x16 and 64x24. This does NOT match total work: the guided method has
additional decoder calls, world transitions, and backward passes, all logged.
Compare success versus actual synchronized controller latency, not flow NFE
alone. The baseline in this runner uses the same audited wrapper as the
refinement variants; its sanity checks introduce overhead versus the raw
upstream solver. Do not substitute published LeFlow timings for measured
wrapper timings or claim a raw-solver systems speedup without measuring it. Batch latency is never divided by batch width and relabeled latency.
There is no wall-time success cap added to the native benchmark; latency is
an observed metric. Each whole evaluation job has an operational 600-second
timeout. A timed-out job is incomplete, never silently scored as a completed
comparison or mixed with successful runs.

The initial random stream matches native single-stream sampling. P&P noise
uses another generator, so extra inner loops do not alter future initial draws
at the same candidate count. Different candidate-count variants are not claimed
to have nested random pools across every subsequent closed-loop decision.

## Native correctness gate

`self_refine.preflight.validate_native_runtime` loads the real model and checks:

1. Zero-refinement sampling is bitwise equal to native `sample_paths`.
2. Action decoding is bitwise equal to the native decoder interface.
3. Differentiable autoregressive rollout agrees with the native rollout.
4. Guidance works inside an outer inference-mode context.
5. Parameters and buffers remain unchanged and no parameter gradients appear.

This gate uses synthetic latent inputs on the real checkpoint. It is an ABI
and numerical check, not a task evaluation. The subsequent evaluation uses
actual dataset-derived starts/goals and native simulator success signals.

The runner verifies the pinned stable-worldmodel commit
`06b1f4c24ccb237a9b2d6bb65bf010a6a5b41e83`. Its exact native success fields are
`episode_successes` and `success_rate` (percentage, not fraction). Missing
reset methods or inconsistent aggregate/per-case outcomes fail visibly.

## Integration into the real repository

These are additive files, not a replacement repository snapshot. In an
execution-enabled checkout of `Abecid/LeFlow-`, add the `self_refine/`,
`tests/self_refine/`, `config/self_refine/`, and this documentation file on a
new branch. Do not replace the existing README, `.gitignore`, `eval.py`,
`latent_planner.py`, requirements, report ledger, or baseline source trees.
Check for path conflicts before adding anything.

Run the tests in that checkout and commit/push before launching compute:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/self_refine

git add self_refine tests/self_refine config/self_refine docs/SELF_REFINEMENT_IMPLEMENTATION.md
git commit -m "Add native LeFlow P&P and world-feedback refinement comparison"
git push -u origin HEAD
```

Use the existing environment satisfying the repository's pinned requirements.
The example checkpoint map uses the released cache aliases. Edit only the map
when the existing shared checkpoint files have different paths. No checkpoint
or dataset is downloaded implicitly.

Inspect the run plan (this command launches nothing):

```bash
python -m self_refine.campaign \
  --config config/self_refine/native_comparison.json \
  --checkpoint-map config/self_refine/checkpoints.example.json \
  --run-root /path/outside/the/source/checkout/native-self-refinement \
  --gpus 0,1,2,3
```

Execute the finite comparison with the same arguments plus `--execute`:

```bash
python -m self_refine.campaign \
  --config config/self_refine/native_comparison.json \
  --checkpoint-map config/self_refine/checkpoints.example.json \
  --run-root /path/outside/the/source/checkout/native-self-refinement \
  --gpus 0,1,2,3 --execute
```

The GPU indices are logical indices within the current visible allocation;
the launcher preserves existing CUDA_VISIBLE_DEVICES restrictions. Use a
single index for one available GPU. Online W&B logging is the default; offline
or disabled logging requires an explicit command-line choice. Never put tokens
or passwords in this repository or in invocation files.

Resume only after inspecting saved status, using `--resume --execute`. This
reuses completed identical jobs and refuses to replay an existing incomplete
job. A source/configuration/checkpoint change invalidates the resume manifest.
A failed job halts further dispatch; other already-started bounded jobs may
finish. Completed result directories are never overwritten.

Generate the measured comparison CSV only after actual evaluations exist:

```bash
python -m self_refine.report /path/to/native-self-refinement \
  --output /path/to/native-self-refinement/comparison.csv
```

A report with no completed results fails rather than printing empty or invented
scores. Protocol mismatches (data, checkpoint, goal protocol, execution settings,
case identity/order, normalization, hardware) prevent paired aggregation.
Existing historical native runs can only be compared after verifying the same
identities. The old custom MetaWorld scores are not compatible just because
both methods contain a flow network or a JEPA encoder.

## Sources and implementation boundaries

- Self-Refining Video Sampling: https://arxiv.org/abs/2601.18577
- Official reference: https://github.com/agwmon/self-refine-video
- Native LeFlow: https://arxiv.org/abs/2608.24855
- Native runtime inspected in Abecid/LeFlow-, blob
  `91bf238994765d67e9b941e138d5da482feafb2a` (`latent_planner.py`).
- Native JEPA prediction graph inspected in blob
  `486fe93a60f7e7af1fce34e80cc3abdf557559f6` (`jepa.py`).
- stable-worldmodel evaluation return contract and reset protocol inspected at
  the pinned commit above, `stable_worldmodel/world.py`.

The code reimplements P&P for native scalar-time latent-path flow and adds a
proposed world-gradient correction. It is not a faithful reproduction of the
video paper's masking/CFG choices, and the effectiveness of the robotics
extension is unmeasured.
