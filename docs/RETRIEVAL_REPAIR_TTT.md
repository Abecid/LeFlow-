# Cross-episode retrieval evidence for residual-flow training

One next candidate, `retrieval_repair_ttt`, is registered under the October 11
sequential authorization. The preceding repair-flow run is complete and remotely
published at `15852d5c4fa550c22c0f6b6d9142bd7435593792`: 80/73/75/75, selecting
5k80/104 and tying the preserved GMM80. Its 146 ledger artifacts and 41 registered
reports were verified on origin/main before this implementation. This candidate
has no evaluation result yet. Training is gated by final-source tests, cached
training-data preflight and distributed checks.

## Specific evidence and hypothesis

The old `FlowReasoningPolicy.deliberate` constructs four generated proposals for
training. Its deployment controller replaces the fourth proposal for seven of
eight targets with a retrieved action chunk, and can shift an earlier plan into
another slot. The residual-flow teacher thus never trains on the retrieved-action
slot that can provide its deployment incumbent. This is a confirmed code-path
difference; it is not a proved explanation of task failure.

Repair-flow's selected model still fails 24/104 cases, all at the action ceiling,
including seven assembly, six faucet-open and six pick-place cases. At 20k,
calibration and teacher action losses improve while success falls to75. Saved
5k traces select retrieved *targets* in4,158/5,173 decisions and exactly match a
shifted prefix in1,206; corresponding failure-only counts are1,411/2,400 and703.
The old logs do not identify the action source of the repair anchor. A retrieved
target is not proof that a retrieved action was selected. The
[saved-record rationale](reports/20261011-retrieval-repair-ttt/rationale.json)
keeps this limitation explicit.

Hypothesis: exposing residual training to a cross-episode retrieved action in
its verified candidate pool reduces one distribution mismatch. This is a single
specified correction, not a diversity/radius/learning-rate sweep. Its result may
be negative. Shifted-plan training coverage and on-policy state distribution
remain unresolved; this is not full deployment-distribution matching.

## Exact correction and causal boundary

For each deep training update, use the same sampled current state, local requested
image target and global image goal. Query only the unchanged training route bank's
five-block routes. Rank an approximate shortlist of256 with exact start/endpoint
cosine distance. **Exclude the entire query episode before shortlisting and exact
reranking**. The lookup receives no teacher action, next-state error label or task
success label. The local requested target is already the existing goal-conditioning
input; no new sensor or outcome field is introduced.

Replace one of four generated proposals with the retrieved five-block action in
seven eighths of global training examples. The remaining eighth has no retrieved
slot, matching the controller's seven retrieved targets plus one direct-goal slot
count; training still uses the original local/global target sampler. Score every
candidate through the frozen world with the existing factual calibration, select
the best incumbent, and train the same clipped action residual:

`delta_target = clamp((recorded_action - stopgrad(incumbent)) / 0.25, -1, 1)`.

The teacher interpolant is used only for the residual imitation loss. Only the
recorded action receives its own recorded outcome-error labels. Generated and
retrieved actions receive model predictions, never inherited physical outcomes.
All learned weights are fresh. No policy/world/encoder/baseline is fine-tuned or
silently inherited. The learned episodic adaptation mechanism is unchanged.

The deployment policy has the same6,466,320 parameters, two rounds, four candidates
per target, exact incumbent preservation, radius.25, eight global plus four repair
flow steps and320 world transitions. It executes one two-primitive-action prefix,
under10 cumulative controller seconds and200 actions. Only diagnostics additionally
record whether the first-round repair anchor was generated, retrieved or shifted.
This provenance labels a construction slot, not causal attribution.

## Literature and implementation grounding

[Residual Policy Learning, Silver et al.](https://arxiv.org/html/1812.06298v1)
adds a learned corrective policy to a base controller and trains the composite
behavior. Its discussion of using the controller on the correction's training
states motivates checking our anchor distribution. Their method is online
model-free RL; this candidate is offline goal-conditioned residual flow imitation,
so it is an adaptation of a general idea, not an RPL reproduction or new claim
to residual control.

[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) and its
[official image-policy implementation](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_image_policy.py)
condition action generation on observations and execute a prefix of the sampled
sequence. We preserve that observation/action timing discipline. This does not
reproduce its architecture or establish that retrieval training is novel. The
existing LeFlow generated-consistency precedent remains acknowledged.

Local code reviewed: `flow_reasoning.py:deliberate`, `flow_controller.py:plan`,
`repair_flow_ttt.py:forward/revise`, `bank.py:RouteBank`, `revision.py` and
`aligned.py` samplers. New isolated code is `execution/retrieval_repair.py`;
historical frozen server checkouts are untouched.

## Fixed contract and completion obligations

- Seed3072 only; same6,222 expert episodes, exact windows/targets/actions and
  fixed104 development cases, global64 and frozen world/encoder/statistics.
- One candidate on four idle GPUs, acquired non-preemptively with advisory locks.
  At most20k updates or28,800 aggregate optimization GPU-seconds. Retrieval queries
  are inside the charged update timer. Bank setup is separately measured and its
  ledger accumulates on resume; prior known campaign compute is90.934490GPUh.
- Four registered evaluations at5k/10k/15k/20k; no extra baseline training,
  seed, inference sweep, new data or sealed final test. Same fixed visual cases.
- Verify source/runtime/config/data/checkpoints, periodic metrics, complete reports,
  process exit and online completion. Preserve failures and costs; publish to
  origin/main and verify remote artifacts before deciding any next correction.
- Compare paired cases to both tied80 references, flow79, progress78 and guided75,
  including whether retrieved incumbents occur and are corrected, saturation,
  actual success, timeouts and action-budget failures. Teacher residual loss and
  latent prediction errors are not physical ground truth.
- Stop new launches for review after an improvement over80 or when no specific
  supported correction remains. Reused development cases cannot establish
  independent generalization, seed robustness, isolated causality or SOTA.

## Verification status before launch

Five new behavior tests pass on the project runtime using CPU only: whole-episode
exclusion with exact reranking, teacher isolation, retrieved candidate world
verification/selection, both-depth gradients with frozen world, and controller
accounting/reset/provenance. The initial test harness incorrectly applied a
proposal-pool assertion to the factual calibration call; that test-only error was
corrected, with no optimizer updates or GPU use. The final coordinator also reruns
inherited behavior checks before the one training-data preflight and no-update
four-rank gradient verification. The cached preflight checks exact old data bytes
for eight examples and10,000 unchanged task/episode draws.

Server record: `/home/mtxu/adam/LeFlow-experiments/20261011-retrieval-repair-ttt`.
Runtime: `/tmp/mtxu-flow-jepa-20261007/env/bin/python`. Current identities and frozen
source are recorded in the active ledger and launch evidence after dispatch.
