# Guided TTT: one fresh action-flow run

October10 Pacific. [Formulation and claim limits](GUIDED_TTT.md).

Preflight passed on recorded training data: seven new behavior tests, finite
gradients with the frozen world unchanged, causal history/window checks, and
eight-GPU exact gradient synchronization at both trained depths. These checks
perform zero optimizer updates and no benchmark evaluation. The registered
two-round controller averages96.97ms over five warmed training-image decisions.
This is a timing check, not measured task success or a guarantee against timeouts.

One fresh `guided_ttt` seed3072 policy is authorized. Global64; stop at20k updates
or28,800 aggregate optimization GPU-seconds; all8 available A800 GPUs. Same6,222
expert episodes, frozen encoder/world, training routes and fixed104 development
cases. Evaluate at5k/10k/15k/20k with10s/200primitive-action limits. The sealed
3,200-case final suite remains unopened. Preserve and reuse existing baselines.

Deployment root:
`/home/mtxu/adam/LeFlow-experiments/20261010-guided-ttt`.
Cache: `/tmp/mtxu-guided-ttt-20261010`.
Launcher: `scripts/operations/launch_guided_ttt.sh`.

Source will be frozen at the deployment commit; all runtime file hashes are
checked before dispatch. The coordinator runs detached, repeats required final
source checks, launches training and dispatches the four evaluations itself.
The run is not yet claimed active in this pre-launch record. Its verified status
and W&B link are added after observing actual optimizer updates.

Verification evidence:
[timing/data checks](reports/20261010-guided-ttt-run/preflight.json),
[eight-GPU no-update check](reports/20261010-guided-ttt-run/ddp-check.json).

Earlier3-round timing checks measured112–115ms; this motivated two rounds before
any benchmark episode or training update. No performance sweep was run. Compute
for checks is reported separately from the optimization budget.

## Verified running

At2026-10-10 16:01:37 Pacific (23:01:37 UTC), step399 had completed on all8 GPUs.
Training metrics were logged through350. No task-success validation was complete.
[Live W&B](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/dw8gnxtj).
Source92fdae6120c568a6e13c7ca2e6013fbc72cb45bf;143 deployed runtime hashes verified.
Coordinator3464557; torchrun3466819. Source and all earlier runs remain frozen.
[Launch evidence](reports/20261010-guided-ttt-run/launch-verification.json).

The final-source coordinator repeated all behavior checks and measured96.94ms
in its training-data preflight. A full Git bundle recovered an incomplete prior
server clone before launch; no optimizer work was duplicated.

Initial measured throughput is about559 examples/second on a deep update.
Provisional ETA: first completed evaluation around13–15 minutes after training
start, full cycle roughly50–60 minutes, assuming stable throughput and evaluation
runtime. These are estimates, not results. A separate CPU-only continuation will
analyze saved milestones and audit the finished run; it cannot launch training,
model inference, additional variants, or sealed tests.

Latest archived readback at2026-10-10T23:04:32.986988+00:00: step2096, no completed validation yet.
CPU analysis continuation PID3470133 is running, with three helper hashes
verified against analysis source82099f229597b032ee864a3ab27a9f0781f771ee.
[Latest health](reports/20261010-guided-ttt-run/latest-health.json),
[analysis launch](reports/20261010-guided-ttt-run/analysis-launch.json).
