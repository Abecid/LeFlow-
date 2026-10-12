# First milestone reached; task-success result pending

At08:45:04UTC the run reached5,000 updates, with1.027975 optimizationGPUh,
no budget overrun or failure. The first registered evaluation has not yet
published its104-case result. Source/launch evidence was published at3082350;
399 ledger artifact blobs and13 distinct report files matched the live GitHub
commit at08:44:35UTC. [Readback](reports/20261012-bayesian-repair-ttt/publication-verification.json)
and [milestone status](reports/20261012-bayesian-repair-ttt/live-milestone-status.json).
Continue the same run; no second candidate or baseline training.

# Verified active training: Bayesian decision-calibrated repair

At 2026-10-12 08:39:58 UTC (01:39 Pacific), fresh training had completed **2,000
updates on GPUs 0–7**, with finite metrics and no evaluation yet. Frozen source
`4d322c90e216702fae2cd29c91e49ad90da713d9`, all 173 runtime hashes, coordinator
1562004, launcher1564888 and eight rank identities were verified. [Live run](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/u1hdmwqr)
is running. The 5k/10k/15k/20k fixed104 evaluations are scheduled under the
original allowance; 80/104 remains the measured reference, not a new result.

The no-update diagnostic used 1,024 TRAIN windows:512 noise-fit examples and467
nonempty-history examples from disjoint TRAIN episodes for reporting (45 windows
had empty history and were excluded from those two groups). With frozen reference
features, factual next-outcome MAE fell **6.01%** (.00148462→.00139539), MSE fell
**17.67%**, and the nominal90% interval covered **90.79%** of reporting outcomes.
These are factual proxy checks, not generated-action success or ranking evidence.
No diagnostic weights/noise estimates initialize the fresh policy.

All45 distinct tests passed (50 executions including five imported repeats).
Real-data full-controller latency averaged **81.12ms**; eight-rank backward
checks passed at all three training depths, with unchanged weights/world and
exact gradient synchronization. At the2k snapshot, optimization used.412279GPUh;
no-update diagnostic/preflight/DDP used.017536GPUh. Route-bank metadata reuse
used22.11CPU seconds; its tensors and6222 episodes are unchanged. Known cumulative
measured components at this partial snapshot are105.714343GPUh, not a cluster bill.

[Launch identity, processes and online readback](reports/20261012-bayesian-repair-ttt/launch-verification.json),
[TRAIN-only diagnostic and factual records](reports/20261012-bayesian-repair-ttt/bayesian-probe.json),
[latency](reports/20261012-bayesian-repair-ttt/preflight.json),
[distributed checks](reports/20261012-bayesian-repair-ttt/ddp-check.json),
[periodic metric snapshot](reports/20261012-bayesian-repair-ttt/run/metrics-at-launch.jsonl.gz)
and [compute accounting](reports/20261012-bayesian-repair-ttt/compute-accounting-launch.json)
are preserved. Metrics with negative predictive NLL are valid continuous densities,
not a sign error; task success is still pending.

The server's initial GitHub fetch failed before any coordinator or optimizer run.
A360KB pack of the exact published source resolved it; all runtime file hashes
matched before dispatch. No active execution source was modified. Reporting
code is maintained in the separate desktop checkout. The15-minute project monitor
is updated for this run and must publish all scheduled results and failures upon
completion. No additional candidate is queued.

# Bayesian decision-calibrated bounded flow repair

October 12 authorization: implement one evidence-grounded candidate and train it
fresh under the existing allowance. This supersedes the prior review hold for
this candidate only. Launch is gated by the TRAIN-only diagnostic, mathematical
and causal tests, full-controller latency and eight-rank gradient checks.

## Published mechanism and implementation mapping

The calibration foundation is [ALPaCA](https://arxiv.org/abs/1807.08912),
Harrison et al., *Meta-Learning Priors for Efficient Online Bayesian Regression*.
Its [official implementation](https://github.com/StanfordASL/ALPaCA/blob/d06391a1bf11beda573078a6a1d62418aac367c4/main/alpaca.py)
was inspected at commit d06391a1bf11beda573078a6a1d62418aac367c4. We adopt its
learned features/prior, analytic posterior and future-observation likelihood.
This is an established method, not a claim that ALPaCA is recent or that its
original experiments establish gains in this task.

With pooled transition features Phi, normalized observed prediction errors y,
learned prior mean w0 and positive prior precision Lambda0:

    Lambda = Lambda0 + Phi.T @ Phi
    w = w0 + solve(Lambda, Phi.T @ (y - Phi @ w0))
    Var[error(phi)] = sigma² * (1 + phi.T @ solve(Lambda, phi))

`flow_jepa/execution/bayesian_repair.py` implements these equations using a
Cholesky solve, without an explicit matrix inverse. Features, prior mean and
precision are trained through the posterior on the NEXT recorded outcome.
Our scalar observation variance is learned offline; this extends the reference's
specified noise covariance. One likelihood observation represents one physical
transition, after pooling spatial features and cost errors. Thirty-two correlated
patches are not counted as thirty-two independent execution trials. A four-step
history is causal and episode-local; online updates never change shared weights.

[World-Coherent Decoding](https://arxiv.org/abs/2609.02159) is a recent precedent
for using observed execution mismatch to improve frozen-world-model decisions.
Our online scope, action generator, labels and controller budget differ. We are
not reproducing that paper's algorithm or using additional physical rollouts.

## Adaptation to the strongest retrieval-repair formulation

The base is the formulation that reached 80/104, initialized from scratch.
The old checkpoint is used ONLY by a no-update TRAIN diagnostic. It never
initializes this candidate. The frozen encoder/world, route bank and recorded
action supervision are retained. The failed weak remaining-expert-time readout
is excluded. Baseline implementations and checkpoints remain unchanged.

The posterior gives SHARED weight samples to all actions for a target.
Deterministic spherical-radial Gaussian cubature preserves posterior first and
second moments; after nonlinear clipping it is an approximation. For each
point, the correction goes through the exact deployed tanh bound and positive
optimism penalty. Candidate cost is the mean of those deployed costs.

A repair replaces the incumbent only if its predicted improvement exceeds
1.644854 times the posterior standard deviation of their COST DIFFERENCE.
The same posterior points are used on both sides, so common error uncertainty
cancels. Among eligible candidates we retain the lowest predicted cost. The
incumbent is always admissible. This is posterior support for the prefix-error
comparison, not a 95% physical-success guarantee. It excludes unmodeled terminal
bias, task semantics and correlated-observation misspecification. Independent
observation noise is used for factual likelihood, not invented as two unrelated
noise draws for the same decision.

The outer objective retains action and bounded-residual flow matching, adds
ALPaCA predictive NLL, and supervises the actual deployed clipped penalties on
recorded outcomes. Generated actions receive no invented outcome/success label.
The terminal point-error head remains factual and is trained against both its
signed error and deployed positive penalty.

Train depths cycle through 1, 2 and 3 under the SAME optimizer allowance, with
loss averaged across active repair depths. Evaluation stays at two rounds for
direct comparison. Further rounds preserve a fixed initial scoring context,
the accepted incumbent, and a TOTAL L-infinity action radius of 0.25 around the
first verified anchor. This enables trained depths; it is not yet a measured
test-time scaling curve, and no extra evaluations are authorized here.

The controller reuses the incumbent's unchanged frozen-world rollout. Two
rounds therefore require 280 rather than 320 predicted transitions, while retaining
the 8-target, 4-candidate pool. This is saved duplicate computation, not a larger
inference allowance. It logs every candidate's action, raw/corrected score,
shared uncertainty margin, acceptance, and later observed selected-prefix error.
Rejected actions still have no measured physical outcome.

## Fixed experiment and gates

One seed 3072; 6,222 successful TRAIN expert episodes; global batch 64;
20,000 optimizer updates OR 28,800 aggregate optimizer GPU-seconds, whichever
binds first. All eight idle A800 80GB GPUs are requested. Four fixed 104-case
development evaluations at the registered milestones; 200 primitive actions
and 10 cumulative controller seconds per episode. No new data, baseline runs,
additional seeds, sweeps or sealed tests. Eight-rank random-stream assignment
differs from the four-rank retrieval run; this is not bitwise identical training.

Before optimization, `scripts/analysis/bayesian_repair_probe.py` uses frozen
retrieval features on 1,024 TRAIN samples: chronological past predicts the next
recorded outcome. Residual-noise estimation and reporting use disjoint TRAIN
episodes. The gate requires lower factual MAE and MSE than the learned prior.
That supports testing the adaptation; it cannot prove repair ranking or task
success. The diagnostic does not train a policy or supply parameters to it.

Tests cover posterior algebra/covariance, common-mode cancellation, padding,
episode isolation, gradients at all depths, future-label isolation, total repair
bounds, acceptance and exact world-call accounting. The full real-data controller
must average <=95ms including encoder/retrieval/diagnostic logging. Eight-rank
checks perform backward passes with no optimizer updates before fresh training.

## Assessment and publication

Task success within the fixed controller allowance is decisive. Compare paired
case outcomes to retrieval-repair 80, execution-revision 80 and preserved baseline
ports. Inspect acceptance/rejection frequencies, uncertainty coverage of factual
errors, repair retention, hard-task failures, proposal spread, score changes and
saved visual trajectories. Lower NLL/MAE alone is not improved control.

Server record: `/home/mtxu/adam/LeFlow-experiments/20261012-bayesian-repair-ttt`.
Source, all periodic metrics, four complete evaluations, checkpoint hashes,
compute and failure findings must be preserved and pushed to origin/main.
Do not edit source beneath an active run or dispatch a speculative next variant.
Current novelty claim is a proposed application: **use execution-calibrated
posterior comparisons to govern bounded generative plan repair**. Benefit,
novelty, causal component contribution and inference scaling remain empirical
questions; prior 80/104 results do not establish this candidate's performance.

## Reporting tools

The reporting checkout adds `scripts/analysis/bayesian_repair_details.py` to
reconstruct candidate scores, preserved incumbents, total bounds and acceptance
margins from saved records, and summarize observed calibration/rejection rates.
`audit_repair_flow_ttt_run.py --method bayesian_repair_ttt` additionally checks
source/data/case/checkpoint identity and 280-call/three-training-depth accounting.
Run these from the reporting checkout, never by modifying frozen execution source.
`repair_flow_ttt_details.py` supplies the same case-paired comparisons and failure
summaries as previous iterations, with the new method's correct accounting.
