# Bayesian repair completed: selected 15k 77/104

Full curve71/76/77/77, below retained 80. Read the [completed findings and evidence](BAYESIAN_REPAIR_TTT_RESULTS.md) and [review hold](BAYESIAN_REPAIR_TTT_REVIEW.md). Historical interim/launch snapshots follow; no run remains active and no next candidate is authorized.

# Bayesian repair: interim curve 71/76; training continues

## First two fixed evaluations: 71/104 and 76/104

The 5k and 10k checkpoints score **71 and 76 successes out of 104**. The provisional
selection is 10k, below the retained 80/104 references. These are interim results;
the registered 15k and 20k evaluations remain part of this same run.

| Checkpoint | Successes | Timeouts | Wins / regressions vs retrieval 80 | Wins / regressions vs execution 80 |
| --- | ---: | ---: | ---: | ---: |
| 5k | 71 | 0 | 2 / 11 | 2 / 11 |
| 10k | 76 | 2 | 5 / 9 | 4 / 8 |

All 33 failures at 5k reach 200 actions. At 10k, 26 failures reach 200 actions;
assembly/00002 and assembly/00006 exhaust the cumulative 10-second controller
allowance at 194 and 192 actions. Their traces include isolated controller calls
of 1.611 and 1.561 seconds, respectively, amid calls usually near 85–90ms.
Recorded overruns are 0.000389 and 0.034504 seconds. These observations do not
identify the source of the spikes or establish that either case would have
succeeded with more time. No evaluation is replayed. Overall mean decision
latencies are 88.59 and 89.26ms; the preflight mean is not a guarantee that every
runtime call stays under that value.

Assembly and faucet each remain 1/8 at both milestones, while reach improves
from 1/8 to 3/8. Pick-place remains 0/8. All eight pick-place failures at both
checkpoints use reach-derived or direct-goal targets throughout their last 20
recorded decisions. This is route provenance, not proof of a physical failure
mechanism or justification for oracle task-ID filtering. Against retrieval 80,
the provisional 10k result gains one assembly, one dial, one faucet and two reach
cases, but regresses four assembly, one door-open, three faucet and one pick-place
case. The full paired IDs and all preserved baseline comparisons are saved.

The method-specific audit reconstructs all candidate costs, shared uncertainty
margins, admissibility and incumbent choices in 5,755 / 5,318 controller decisions.
Every logged decision satisfies the registered 280 world-transition accounting,
40 cached transitions, two reasoning rounds and total action radius <=0.25.
The route-level gate rejects 7,882 of 41,029 positive-mean-gain repairs at 5k and
6,137 of 36,646 at 10k (19.21% / 16.75%). The final executed plan is a repair in
42.52% / 40.05% of decisions; this is distinct from acceptance at all eight
candidate routes. Generated/retrieved/shifted anchor counts are
1,563/1,763/2,429 and 1,426/1,578/2,314. Rejected alternatives have no observed
physical outcomes, so these frequencies cannot establish the gate's causal benefit.

On actually executed prefixes, the deployed correction reduces the world-error
proxy MAE from 0.002648 to 0.001775 at 5k and from 0.002748 to 0.001846 at 10k.
Nominal 90% predictive intervals cover 85.65% / 84.28% of observed prefix errors,
below the nominal level and the TRAIN diagnostic. This uncertainty concerns
encoded prefix prediction error, not terminal-model bias or physical task success.
Better calibration versus uncorrected predictions has not produced an improvement
over the retained 80/104 references. The last-500-update action-flow loss drops
0.162581→0.116172, repair-flow loss 0.261004→0.181118 and teacher clipping
4.50%→2.71%; these selected training summaries do not identify physical causes.

All twenty saved trajectory contact sheets (ten per checkpoint) were rendered
from measured frames and visually inspected at readable resolution. Pick-place00000
leaves the object on the table; faucet00003 approaches the handle without success.
Assembly00000 succeeds at 5k but leaves the ring near its starting location at 10k;
reach00005 changes from failure to success. Both checkpoints fail reach00003.
The saved subset is not visual coverage of every failure, and sparse frames do
not establish contact forces, slippage or counterfactual outcomes.

At the 09:08:16 UTC readback, frozen source 4d322c9, all 173 runtime hashes and
all 44 recorded PID/start identities match; W&B u1hdmwqr is running and reports
the 10k score. The advancing snapshot preserves all 844 periodic training rows
through step 14051, with finite metrics and monotone optimization charges.
Compute at 14051 is 2.906668169 optimization GPUh and 1.086928572 registered
evaluation GPUh, with no update overrun. Diagnostic, preflight, no-update DDP
and retrieval setup are recorded separately; cumulative measured components
total 109.301016643 GPUh; details are in
`reports/20261012-bayesian-repair-ttt/compute-accounting-round-2.json`.
This partial snapshot is not a full cluster bill. The initial health read was at
14048; sequential advancing files are not claimed to be an atomic checkpoint.

Complete reports retain every original byte through lossless gzip parts, with
per-part, combined-stream and original JSON hashes in the validation manifests.
The reporting loader verifies all three levels before decoding. This format is
needed because each compressed report exceeds GitHub's single-file limit; no
candidate records are dropped. Contact sheets, visual notes, paired IDs, posterior
checks, full metrics, checkpoint hashes and transfer-recovery notes are preserved.
All reporting code stays outside the frozen execution tree. One seed and reused
development selection do not establish generalization, component benefit or SOTA.
No further candidate is authorized; finish and publish this run, then stop for review.

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
