# Nonlinear episodic memory on retrieval bounded flow repair

October 12, 2026. The user explicitly authorized this next comparison on the
80/104 retrieval-repair formulation. One fresh `nonlinear_repair_ttt` candidate
is implemented and verified. Frozen execution source is
`d0cae3007c7d678cdc5ddb68ce731e8a2db3005d` (200 checked files);
Training is verified on eight A800s: coordinator1643283, launcher1645550,
W&B [mxujto8j](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/mxujto8j).
At09:52:04UTC the run reached187 updates, with finite metrics and all200 source
hashes matching. No development evaluation has completed yet. No claim of improved success exists before evaluation.

## What changes

Replace the linear ridge fast-weight readout with a nonlinear three-matrix
SwiGLU memory:

M_W(k) = (SiLU(k W0) * (k W2)) W1.

The input is the same goal-conditioned, action-conditioned spatial prefix
feature as retrieval-repair. Keys have dimension 32 and RMS normalization;
W0/W2 are 32x64 and W1 is 64x1. This is 4,160 adaptable entries versus the
previous 32 linear weights. Four explicit differentiable MSE updates fit all
available support patches in batched matrix operations. The learned step size
starts at 0.05 and is bounded by 0.25; gradient norm is capped at 5 and the
existing nonlinear-memory importance-weighted prior protection uses 0.01.

Support consists ONLY of the last four actually observed two-primitive-action
prefixes in the current episode. For support transition (z,a,z_next), the
label is [distance_per_patch(z_next,target) -
distance_per_patch(frozen_world(z,a),target)] / 0.02. At most 128 spatial rows
represent four physical transitions, not 128 independent execution trials.
All goals are fitted together as a batch, with separate goal-relative memory.
Padding is sanitized before nonlinear operations; empty support exactly
returns the learned prior. Each decision refits from that prior and bounded
history, so no weights or evidence leak between evaluation episodes.

A functional adapted-minus-prior readout at the current state/target conditions
the reasoning workspace. The workspace conditions initial action flow and
bounded repair flow. Candidate-specific memory predictions correct prefix
scores using the existing tanh bound and positive-error penalty. One fitted
memory stays fixed during both search rounds. Only the observed executed prefix
enters the next decision's memory; imagined suffixes never supply labels.

Offline training differentiates through all four updates into all three prior
matrices, key features, bounded learning rate, and action/repair conditioning.
The original action-flow, bounded teacher action-difference repair, and next
factual prefix/terminal error losses remain. No generated alternative receives
the demonstration's physical outcome. Training retains alternating planning
depths 1/2; evaluation has two rounds and four memory updates. This run tests a
specific adaptation depth, not an inference-scaling curve.

## What is preserved

Use the retrieval-repair architecture and sampler that reached 80/104, with
fresh policy weights; do not initialize from its selected checkpoint. Frozen
V-JEPA features/world, route bank, 6,222 successful expert TRAIN episodes,
seed 3072, global batch 64, and train/development cases remain identical.
Train to 20,000 updates OR 28,800 aggregate optimizer GPU-seconds. Evaluate the
same 104 development cases at 5k/10k/15k/20k under 200 primitive actions and
10 cumulative controller seconds. All memory computation is inside that time.
Use all eight available A800 GPUs; actual consumption is reported separately.
The original retrieval-repair run used four ranks. Global batch and episode
sampling stay fixed, but rank-local random streams and within-rank shuffled
negative goals differ; optimization is not bitwise identical.

Keep eight routes, four initial candidates, two planning rounds, eight initial
flow steps, four repair steps, radius 0.25, exact retained incumbent, and 320
world transitions per decision. There is no Bayesian acceptance gate, progress
objective replacement, additional training data, baseline retraining or final
test access. Prior baseline and 80/104 references are reused unchanged.

## Literature and precise scope of transfer

- [LaCT / Test-Time Training Done Right](https://tianyuanzhang.com/projects/ttt-done-right/)
  and [official minimal implementation](https://github.com/a1600012888/LaCT/blob/a648340f9798f173227a0626fde66a6e9b65879a/minimal_implementations/causal_lact_with_sliding_window_attn.py):
  nonlinear three-matrix memory and batched differentiable fast-weight updates.
  Both official minimal implementations were inspected at the pinned revision.
  Their associative update/Muon/weight renormalization is not our MSE update.
- The exact MSE derivative and bounded prior-protected update mechanism transfer
  from our earlier `flow_jepa/leflow_ttt/model.py` nonlinear memory. Its original
  values were projected visual residuals for ten-action chunks. Here values are
  factual goal-relative prefix-cost errors for two-action prefixes, preserving
  retrieval-repair's prediction objective. This is an explicit interface change,
  not reuse of unexecuted ten-action outcomes.
- The importance-weighted prior protection was originally motivated by
  [Fast Spatial Memory](https://arxiv.org/html/2604.07350); our implementation is
  bounded replay from a learned prior, not its full streaming architecture.

This is a LaCT-inspired task adaptation, not a full LaCT reproduction: our
128-row memory is far smaller than its large-context experiments; no Muon or
large-context efficiency claim is made. It does not by itself correct a
misleading goal-distance objective or guarantee useful proposals.

## Verification and outcome analysis

Six focused CPU tests pass: exact inner gradients and outer derivatives against
autograd; empty/nonfinite padding and episode isolation; outer gradients into
all fast matrices; future-label exclusion; effects on action generation, repair
and selection; observed-prefix bookkeeping, reset and unchanged controller
budget. The TRAIN-only preflight passed at85.70ms (max97.00ms) per full decision and9.50GiB peak GPU memory. Eight-rank no-update checks passed both planning depths with exact synchronized finite gradients and unchanged parameters. The exact frozen checkout repeats its registered safety checks before optimization; all preparation compute is logged separately.

Log support error before/after, memory parameter/readout changes, learned inner
step size, recorded next-outcome errors, prior/adapted prefix corrections,
proposal/repair diversity, anchor provenance and fixed-budget task success.
Judge against retrieval-repair80 and execution-revision80 with paired successes
and failures. Lower memory-fit loss is not evidence of improved physical success.
Publish every periodic metric/evaluation, including timeouts and negative
results, with source/data/checkpoint hashes and qualitative failure evidence.

Server record: `/home/mtxu/adam/LeFlow-experiments/20261012-nonlinear-repair-ttt`.
Cache: `/tmp/mtxu-nonlinear-repair-ttt-20261012`.

## Verified launch and publication

All45 frozen-source tests and both eight-rank no-update gradient checks passed.
The final full-controller preflight averages87.64ms. All44 owned process/start
identities and source200 hashes are captured in the launch report. Source
installation briefly lacked its root tree object; exact-tree recovery and a
clean full hash check completed before dispatch. The earlier TRAIN preflight
configuration mismatch and its resolution are also recorded; neither performed
an optimizer update. Known preparation components and omissions are in
`reports/20261012-nonlinear-repair-ttt/compute-accounting-launch.json`.

The preceding Bayesian result71/76/77/77 was fully published and remotely verified
at ee1db8c before this fresh candidate was dispatched. Analysis-only updates to
`audit_repair_flow_ttt_run.py` and `repair_flow_ttt_details.py` were added after
source freeze. Use the reporting checkout for audits, preserving execution source.
