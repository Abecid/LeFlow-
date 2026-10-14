# Self-refinement on retained LeFlow checkpoints — October 14

The current request authorizes execution of the P&P/world-feedback comparison.
The author Hugging Face checkpoint repository currently returns401/not-found via
both public endpoints and the connected Hub tool; model search finds no public
LeFlow model. Existing server/laptop caches contain no author native planner
checkpoint. This is a checkpoint-access dependency, not a lack of SSH, CUDA or
GitHub access. All eight existing A800 GPUs are available.

The retained PushT/Cube LeFlow heads have the released architecture, so twelve
inference-only transfers can be evaluated immediately against the completed
shared-execution baselines. Six methods share each task's selected immutable
LeFlow checkpoint, frozen LeWM,50 original validation cases,50 primitive actions,
10 cumulative controller seconds and original per-case random streams. Each
executes the first five commands of a five-block plan. No training/weight update,
new label, seed, retrieval, critic, TTT or sealed test is introduced. Reacher and
TwoRoom native-author rows remain pending; this transfer is not the requested
four-task author-checkpoint reproduction and does not establish native SOTA.

| Variant | Candidates | Flow steps | Additional refinement |
| --- | ---: | ---: | --- |
| LeFlow reference |64|16|none|
| More candidates |96|16|none|
| More steps |64|24|none|
| Plain P&P |64|16|eight P&P loops|
| Gradient only |64|16|eight world-feedback corrections|
| Guided P&P |64|16|eight combined loops|

Indices4/8/12/14, two rounds each, radius0.05, consistency weight0.1. P&P uses a
separate perturbation generator. Endpoints stay fixed. Inverse decoding, native
actuator bounds, world normalization/history and terminal-only reranking match
the retained reference. World guidance follows the same executable bounded
commands, and factual next-prefix prediction errors are logged when observed.
Rejected actions do not receive invented physical labels.

All six variants use strict FP32 (`NVIDIA_TF32_OVERRIDE=0`). This is also the
retained Cube setting; prior PushT evidence allowed default cuDNN TF32. Its
historical records are preserved and are not silently replaced. The current
reference is reevaluated under the common setting; paired comparisons use that
reference. The adapter matches the legacy controller's action and RNG bitwise
when both run with the same precision.

On actual TRAIN inputs, eight-candidate guidance microbatches took2.5–2.7 seconds
per decision. Batch64 takes0.35–0.37 seconds with approximately359MB peak tensor
allocation. This retains objective, FP32 dtype, noise and candidate count; it is
not a bitwise claim across batch sizes. Strict-FP32 candidate-action differences
are at most1.073e-6 and score differences6.56e-7 in the inspected probes, with
identical selected candidates. Earlier TF32 batching comparisons failed stricter
numerical checks; those diagnostics are recorded as preflight findings, not
physical task results or zero-cost work. No validation outcomes selected the
batching configuration.

The recovered83 CPU tests plus three integration tests pass. Actual trained-head
and released-LeWM GPU checks cover native sampler/decoder/rollout parity,
frozen weights, outer-inference-mode gradients, both primitive dimensions and
all six variants. Final checks run again on the exact launch source. These tests
are not task-success measurements.

Runtime: `/home/mtxu/adam/LeFlow-experiments/20261014-self-refine/runtime`.
Pinned baseline workspace: `/home/mtxu/adam/LeFlow-experiments/20261013-crossbench-baselines`.
The separate twelve-job queue uses one job per GPU, all eight existing GPUs,
finite900-second operational deadlines, full case/checkpoint hashes and no
incomplete-job replay. An operational deadline does not enlarge the50-action/
10-controller-second allowance. All physical reports, failures, paired outcomes,
latencies/world/backward calls, frozen-state checks and online links are published
as each run completes. Original MetaWorld and sixteen completed native-adaptation
training/evaluation records remain immutable.

Method source: [Self-Refining Video Sampling](https://github.com/agwmon/self-refine-video),
[LeFlow](https://github.com/hsiangwei0903/LeFlow). This is a scalar-time latent-flow
P&P adaptation and proposed world-gradient extension; video masking/CFG choices
are not reproduced. DA-SIP, RepFusion and execution-shift TTT are later directions.
