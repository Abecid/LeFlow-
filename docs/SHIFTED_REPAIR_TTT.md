# First evaluation: 79/104 at 5k; training continues

Verified October 11 at 22:42 UTC: the first fixed 104-case development round
scores **79/104**, with **zero controller timeouts**. All 25 failures reach 200
primitive actions. Against the completed selected retrieval-repair reference
(80/104), it gains six cases and loses seven; against repair-flow (80), four/five;
against execution revision (80), five/six. It gains seven/loses three against
guided-TTT (75), seven/six against progress-TTT (78), and six/six against flow
reasoning (79). Exact paired case IDs are retained in the analysis. The registered
10k, 15k and 20k rounds remain pending at this snapshot. This is an interim result;
the unchanged run continues and the threshold for stopping further launches has
not been exceeded.

Shifted anchors are eligible for 98.28% and selected for **54.84%** of deep-update
examples in the final 500-update window before 5k. Retrieved anchors are selected
for 15.00%. These figures establish that the added training path is exercised;
they do not establish that it repairs a physical failure mechanism. Shallow
updates have zero anchor selections by design and are reported separately.
Deep repair-flow loss is .423991, teacher clipping 7.57%, and anchor/teacher action
MSE .037524/.024477. The teacher uses the recorded action label, so its lower MSE
is not independent evidence of action-generation quality.

During deployment, repairs supply 2,631/5,296 decisions (**49.68%**). Anchor
provenance is 1,487 generated, 1,319 retrieved and 2,490 shifted (**47.02%**).
Corrected executed-prefix MAE is .0016209, versus .0026391 before episodic
adaptation. Compared with the predecessor's 5k checkpoint (75/104), corrected
MAE is 3.85% higher while success is four cases higher. This is an observed
cross-run association, not an isolated effect of the new training path. Raw
scores remain optimistic while observed latent progress is nonpositive in the
last 20 decisions of 19/25 failures; corrected scores have that pattern in 12/25.
Latent scores and controller stall counters are not physical ground truth.
Per-task results remain weak for assembly (3/8), faucet-open (1/8) and pick-place
(1/8); reach scores 4/8. The complete task breakdown and negative cases are kept.

[Full 5k report](reports/20261011-shifted-repair-ttt/run/validation/step_0005000.json.gz),
[paired failures and training diagnostics](reports/20261011-shifted-repair-ttt/analysis-round-1.json),
[anchor and stall observations](reports/20261011-shifted-repair-ttt/anchor-outcomes-round-1.json),
and [source/process/online verification](reports/20261011-shifted-repair-ttt/interim-health-round-1.json)
are preserved. Source remains clean at `2ae5a6c`, with all 146 runtime hashes
matching. The coordinator, launcher and four rank PID/start identities are live
and verified. W&B `8yy6nh9f` is running and reports the same 5k score. Mean decision
latency is 85.74 ms. The immutable 5k checkpoint is retained at
`/home/mtxu/adam/LeFlow-experiments/20261011-shifted-repair-ttt/campaign/runs/shifted_repair_ttt_3072/checkpoints/step_0005000.pt`,
SHA256 `901e57c711cb0a1ee8df39892b02b40df4fef10ef12e5c6fef6bfc31912e51f4`.
Report source, protocol, manifest, world and bank identities match the registered
run; the frozen configuration hash and reference case pairing are verified. All 343 periodic training records through
step 8551, plus the registered evaluation record, are preserved losslessly;
finite losses and monotone optimization charges passed verification.

At step 8561, charges are 0.962781 optimization GPUh and
0.472379 registered evaluation GPUh. Including startup checks and retrieval
setup, [known cumulative measured components](reports/20261011-shifted-repair-ttt/compute-accounting-round-1.json)
are **95.978114 GPUh**. This is a partial snapshot, not final run cost or
a full cluster bill. One seed, repeated development use, checkpoint selection,
offline prior-plan approximation, and method-port limitations prevent claims of
independent generalization, component causality or SOTA. Frozen execution,
baselines, original allowances and sealed final tests remain preserved.

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
