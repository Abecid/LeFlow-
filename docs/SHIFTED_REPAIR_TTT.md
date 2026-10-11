# Training verified: causal shifted-plan repair

Readback 2026-10-11T22:21:20.018932+00:00: frozen source `2ae5a6c97bebe56b0b1f889cb34996d34cbf03d6`, all146 runtime
hashes verified. Coordinator905872/start1007760082 and training launcher908541/
start1007773717 own ranks908565–908568 on GPUs0–3. The run had at least
1,134 updates at the 22:22:46 UTC compute readback, with finite saved losses. Online W&B
[8yy6nh9f](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/8yy6nh9f) is running and reports step851,
loss0.623025; collection/upload times differ and this is not an
exact synchronized counter. The first fixed104 evaluation is pending.

All45 final-source tests, unchanged sampled-byte/history/exclusion checks, the
80.18ms full-controller preflight and four-rank no-update synchronized-gradient
check passed. Parameters remain6,466,320 and deployment computation is unchanged.
Bounded preflight0.005931GPUh and DDP check0.004606GPUh,
plus retrieval setup, are preserved separately. Per-update prior planning and
retrieval remain inside the original optimization allowance. Current charged
optimization0.131150GPUh; known cumulative measured
components94.674104GPUh at this partial snapshot. This is not final run cost or a
complete cluster bill. [Launch verification](reports/20261011-shifted-repair-ttt/launch-verification.json)
and [compute accounting](reports/20261011-shifted-repair-ttt/compute-accounting.json)
preserve exact identities and separate costs.

No checkpoint result or success improvement is claimed. The same authorized
seed3072/global64/20k-or28800 GPU-second run continues with four registered
validations, frozen baselines and sealed tests. Do not edit execution source or
dispatch another candidate while it is active.

# Causal previous-plan training correction

One candidate, `shifted_repair_ttt`, is registered under the October11 sequential
iteration authorization. Retrieval-repair is complete and remotely published at
`9a0d2de6866f2b77ad36f4dcea3f684fd474f8d4`; all206 ledger artifacts and45 reports
were verified on origin/main. Its75/77/79/80 curve ties the preserved80, with
7 gains/7 regressions against repair-flow and6/6 against execution-revision.
No improvement or new evaluation result is claimed by this registration.

## Observed gap and bounded hypothesis

The selected retrieval-repair traces contain2,264 shifted,1,548 generated and1,375
retrieved repair anchors across5,187 decisions. Shifted anchors supply43.65%.
All24 unsuccessful cases reach200 actions; no timeout occurs. The implementation
trains on generated/retrieved anchors but never the shifted previous-plan slot.
These counts establish a missing construction path, not a physical failure cause.
Better calibration/action losses have not established greater task success.

The single correction adds that existing deployment transformation to the training
pool. It changes no learning rate, radius, architecture, inference budget or data
sampling. Its hypothesis may be false; the comparison is not a causal ablation.

## Causal construction and explicit approximation

For a sampled decision at index t>0, load the preceding recorded state t-1 and
up to four recorded transitions ending at t-1. Exclude the transition leading to
the current observation, all current/future action labels and outcome-error labels.
Use the same requested local image target and global goal as conditioning. Query
previous-state retrieval with the entire query episode excluded. Generate and
world-score four candidates, then a bounded incumbent-plus-residual second round,
and select a prior plan under that fixed calibrated context. This entire prior
plan construction is detached; it never supplies physical outcome labels.

At the current state, place the prior plan's remaining four action blocks into
candidate slot1, retaining its fresh generated tail. This is the controller's
existing one-block shift, where each block contains two primitive actions. Keep
the previous cross-episode retrieved slot and the other fresh candidates. For
t=0 no shifted slot is inserted. Score the current pool with the unchanged frozen
world; train the same bounded residual imitation target around its best anchor.
Recorded action/outcome pairs alone train physical-outcome calibration.

This is an **offline one-step approximation**, not full deployment-distribution
matching. The current state follows a recorded expert action, not the synthesized
prior action. The prior target is held equal to the current requested target;
actual deployment may switch among eight targets. Its prior planning uses no
recursive warm-start chain or deployment stall history. No on-policy transitions,
new simulation episodes or invented counterfactual labels enter training.

All learned parameters start fresh. Architecture remains6,466,320 parameters;
deployment uses the identical controller class: two rounds,8+4 flow solver steps,
320 world transitions per decision, four candidates per target, radius.25 and
exact incumbent preservation. Its causal episodic adaptation/reset is unchanged.

## Literature and code consulted

[Diffusion Policy's project](https://diffusion-policy.cs.columbia.edu/) and
[official image-policy code](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_image_policy.py)
condition on available observations and execute a slice of a predicted action
sequence. That timing motivates checking the action shift and information boundary;
it does not establish this training construction or its expected effectiveness.
[Residual Policy Learning](https://arxiv.org/html/1812.06298v1) learns corrections
on top of an initial policy with model-free reinforcement learning. Our offline
flow imitation with a world-selected anchor is an adaptation of residual control,
not an RPL reproduction. Neither residual control nor generated consistency is
claimed new; the latter is also described by LeFlow.

Reviewed local `FlowReasoningController.plan`, `RetrievalRepairPolicy.training_contexts`,
`RepairFlowTTTPolicy.forward`, `RevisionSegments`, `AlignedSegments` and episodic
`ProgressTTTPolicy.factual_history/prepare`. Isolated implementation is
`flow_jepa/execution/shifted_repair.py`; all old server checkouts stay frozen.

## Verification and fixed allowance

Six targeted CPU checks cover prior-state/history indexing, exclusion of current
state/teacher/outcomes, exact shifted prefix/fresh tail, cross-episode retrieval,
policy gradients/frozen world and identical deployment for identical weights,
randomness and observations. Final-source inherited checks, unchanged cached
sample checks, full-controller latency and four-rank no-update synchronized
gradient verification gate optimization. The controller latency gate stays95ms.
No task-success claim is inferred from these checks.

One fresh seed3072, same6,222 expert episodes and fixed104 development cases,
unchanged encoder/world/statistics, global64, maximum20k updates OR28,800 aggregate
optimization GPU-seconds. Four idle GPUs are acquired non-preemptively while
another GPU is occupied. Four registered evaluations at5k/10k/15k/20k use the same
10 controller seconds and200 primitive actions. All extra prior planning/retrieval
work occurs inside the optimization timer; preflight, no-update checks and route
setup are measured separately. Known prior cumulative components94.530290GPUh
remain charged. No baseline retraining, extra seed, sweep or sealed final test.

Analyze/publish the full curve, paired cases against all three80 references,
flow79/progress78/guided75, repair provenance/selection/clipping and actual compute.
Keep failed cases, timeouts and model-proxy limitations. Stop further launches for
review if selected development success exceeds80; reused development cases and
one seed do not establish generalization, component causality or SOTA.
