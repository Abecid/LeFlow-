# Nonlinear repair completed: selected 5k 73/104

The fixed curve is **73/66/67/72**, below the retained **80/104** references.
All four complete reports, 800 periodic metrics, four controller timeouts and
40 individually inspected saved contact sheets are preserved. Source/data/bank/
checkpoint and controller audits pass; all owned processes exited and W&B finished.
Read the [completed findings](NONLINEAR_REPAIR_TTT_RESULTS.md) and
[review hold](NONLINEAR_REPAIR_TTT_REVIEW.md). No next candidate is authorized.

The interim and launch snapshots below are historical; they do not indicate an active run.

# Nonlinear episodic memory on retrieval bounded flow repair

October 12, 2026. The user explicitly authorized this next comparison on the
80/104 retrieval-repair formulation. One fresh `nonlinear_repair_ttt` candidate
is implemented and verified. Frozen execution source is
`d0cae3007c7d678cdc5ddb68ce731e8a2db3005d` (200 checked files);
Training continues on eight A800 GPUs. The fixed curve is now **73 / 66**:
selected-so-far remains **5k 73/104**, below retained 80/104 references. Ten-k has
four controller timeouts and 34 other failures at 200 actions. No evaluation or
optimizer update is replayed. Two registered evaluations remain.

At the 10:24:59 UTC readback all 200 frozen runtime hashes and all 44 original
process/start identities match; W&B mxujto8j is running (online step 12,701).
There are 517 finite periodic metric rows through 12,901. The copied compute
record is step 12,928; sequential reads advance while training continues.

## Second fixed result: 10k 66/104

| Checkpoint | Successes | Controller timeouts | Other failures at 200 actions |
| --- | ---: | ---: | ---: |
| 5k | 73 | 0 | 31 |
| 10k | 66 | 4 | 34 |

Ten-k has **5 wins/19 regressions versus retrieval-repair 80**, and **4/18 versus
execution-revision 80**. It gains dial-turn 6 and drawer-open 0 versus 5k, but loses
button-press-topdown 4, dial-turn 4/7, door-close 1, door-open 1/4, plate-slide 1 and
reach 1/4. All nine newly regressed cases reach 200 actions. The four timeout cases
already failed at 5k; this does not reveal their outcomes under a longer budget.
See [all paired IDs and timeout traces](reports/20261012-nonlinear-repair-ttt/paired-and-timeouts-10000.json).

| Task | 5k | 10k |
| --- | ---: | ---: |
| assembly | 0 | 0 |
| button-press-topdown | 8 | 7 |
| coffee-button | 8 | 8 |
| dial-turn | 3 | 2 |
| door-close | 8 | 7 |
| door-open | 7 | 5 |
| drawer-close | 8 | 8 |
| drawer-open | 7 | 8 |
| faucet-open | 0 | 0 |
| handle-press | 8 | 8 |
| pick-place | 0 | 0 |
| plate-slide | 8 | 7 |
| reach | 8 | 6 |

The [four timeout records](reports/20261012-nonlinear-repair-ttt/paired-and-timeouts-10000.json)
are assembly 00000 (198 actions), faucet-open 00005 (178), and pick-place 00004/00005
(112/96). Their maximum individual calls are 154.38, 231.98, 235.23 and 252.08 ms;
the two pick-place cases have 44/57 and 41/49 calls above 150 ms. Their controller
means are 175.47/205.36 ms. This is sustained latency in those traces, not merely
one isolated multi-second call. No causal hardware or implementation explanation
is established by these records. The original mean startup gate passed at 87.64 ms;
it does not guarantee every later episode's latency. Overall decision mean/p95 is
97.39/113.20 ms. Recorded budget overruns are 0.014535, 0.043772, 0.001743 and
0.062748 seconds. Work is charged, and actions returned after the shared deadline
are withheld. Pick-place 00004 logs one fully planned but unexecuted final decision;
plan-selection counts therefore do not all denote executed actions.

The [interim trace audit](reports/20261012-nonlinear-repair-ttt/prefix-audit-10000.json)
passes 5,797 logged plans and causal history/reset/four-update/radius accounting.
Normalized support error improves on all 5,693 supported fits, averaging
0.251404→0.127880. Raw/prior/adapted next-observed-prefix MAE is
0.003104/0.004852/0.001935, compared with 0.003162/0.005203/0.002080 at 5k.
Support error and model-distance MAE have different units. Observed trajectories
and support-count groups differ, so these reductions do not isolate a physical
success benefit. The empty-history prior still worsens error versus raw
(0.004212 versus 0.002135).

The selected plan is a repair on 2,515/5,797 logged decisions (43.38%).
Generated/retrieved/shifted anchors count 2,148/1,500/2,149. On matched initial
states with empty histories and verified frozen route costs, initial-pool spread
falls 41.59% and residual-pool spread 29.58%; all 104 cases contract in both pools.
This is observational contraction, not a causal ablation. Last-500-update mixed-depth
action-flow loss falls 0.174954→0.122822, repair loss 0.216217→0.146968 and clipping
4.18%→2.35%, while development success falls. Full depth-specific metrics and
[paired diversity records](reports/20261012-nonlinear-repair-ttt/details-through-10000.json) remain saved.
Seven failed pick-place last-20 tails use only reach/direct-goal provenance; case 5
also includes six pick-place choices. None supplies a physical counterfactual or
justifies task-ID filtering.

All ten newly saved 10k contact sheets were individually inspected, bringing the
preserved total to twenty. Assembly 00000 remains near the object before withdrawing
and times out at 198 actions. Pick-place 00000 leaves the object on the table;
faucet 00003 and dial-turn 00000 finish away from their apparatus without success.
Reach 00003/00005 succeed at 36/40 actions; coffee-button, door-close, drawer-close
and handle-press examples succeed at 35/70/71/13. See the [exact image hashes and
observations](reports/20261012-nonlinear-repair-ttt/visual-review-10000.json). Sparse
frames cover ten saved cases per checkpoint, not all failures or contact forces.

The new complete report and 517 periodic metrics are preserved losslessly with
original-byte hashes in the [round-two readback](reports/20261012-nonlinear-repair-ttt/interim-health-round-2.json).
First-round snapshot bytes remain under `run/snapshots/`, with [path mapping](reports/20261012-nonlinear-repair-ttt/round-1-snapshot-paths.json);
the new metrics preserve the previous snapshot as an exact prefix. The 10k server
checkpoint SHA256 is `b6ddc1ae90b573f637a05d81be9d401ce6726bcd4ef34004d2924e94ccb092a0`.
At compute step 12,928, optimization consumes 3.144676240 GPUh and evaluation
1.193198065 GPUh, with zero optimizer overrun. Known cumulative measured components
are **115.940457326 GPUh**, including separately recorded preparation and retrieval
setup. The failed configuration preflight remains unmeasured; this is not a full
cluster bill. See [round-two compute accounting](reports/20261012-nonlinear-repair-ttt/compute-accounting-round-2.json).

## First fixed result and failure evidence (preserved 5k snapshot)

| Task (eight cases each) | 5k successes |
| --- | ---: |
| assembly | 0 |
| button-press-topdown | 8 |
| coffee-button | 8 |
| dial-turn | 3 |
| door-close | 8 |
| door-open | 7 |
| drawer-close | 8 |
| drawer-open | 7 |
| faucet-open | 0 |
| handle-press | 8 |
| pick-place | 0 |
| plate-slide | 8 |
| reach | 8 |

The same case hashes, reset/model seeds and task identities match every preserved
comparison. Against retrieval-repair 80 there are **8 wins/15 regressions**:
seven reach gains and dial-turn 7, offset by assembly 0/1/5/6, dial-turn 0/1/2/3/6,
door-open 0, drawer-open 0, faucet 0/2/7 and pick-place 4. Against execution-revision
80 there are **5 wins/12 regressions**: door-open 1 and reach 3/4/6/7 improve;
assembly 5/6/7, dial-turn 0/1/2/3, drawer-open 0, faucet 0/4/7 and pick-place 4 regress.
Numbers abbreviate the zero-padded validation IDs. Full paired IDs and comparisons
against all preserved references are in [saved-record details](reports/20261012-nonlinear-repair-ttt/details-5000.json).

The [interim prefix audit](reports/20261012-nonlinear-repair-ttt/prefix-audit-5000.json)
passes all 5,599 decisions: four memory updates, history capped at four actually
observed prefixes, empty-history reset, fixed memory during search, 320 world
transitions, retained incumbent and radius 0.25. This is a saved-trace audit;
the final all-checkpoint CPU audit remains due at completion.

On the 5,495 supported decisions, mean normalized support loss falls from
0.255950 to 0.120733, improving on every supported fit. Across observed next
prefixes, raw/prior/adapted model-distance MAE is 0.003162/0.005203/0.002080.
Support loss and prefix MAE have different units. At empty history, the learned
prior's next-prefix error is 0.004707 versus raw 0.002124; with four support
transitions, adapted error is 0.002066 versus raw 0.003297. Support-count groups
visit different states and are not an adaptation-depth experiment. Failed cases
also show lower adapted error (0.001826 versus raw 0.003249), so this proxy does
not identify physical success. No generated alternative has an observed outcome.

Repairs are executed on 2,221/5,599 decisions (39.67%). Generated/retrieved/shifted
repair anchors count 1,987/1,552/2,060. Mean initial/residual pool spread is
0.128414/0.043388; there is no later-checkpoint diversity trend yet. In the final
500-update window, mixed-depth action-flow loss is 0.174954, residual-flow loss
0.216217 and teacher clipping 4.18% (8.36% on depth-two rows). Lower action or
support error cannot establish a controller benefit on its own.

All failed pick-place tails choose reach/direct-goal provenance in their final
20 decisions. All faucet tails choose direct goal; five failed assembly cases
retain assembly routes throughout. These are associations, not evidence that
oracle task-ID filtering would succeed. The score retains the same controller
allowance: mean/p95 decision latency 93.24/102.80 ms, with no timeout.

All **ten exact saved contact sheets** were visually inspected. Pick-place 00000
leaves the red object on the table while the hand moves toward the goal region.
Assembly 00000 remains near the object then withdraws; faucet 00003 and dial-turn
00000 approach their apparatus and later move away. Each fails at 200 actions.
Reach 00003/00005 succeed at 36/39 actions; coffee-button, door-close, drawer-close
and handle-press examples succeed at 34/72/69/16. See [per-image review and hashes](reports/20261012-nonlinear-repair-ttt/visual-review-5000.json).
Only ten saved cases are shown, not every failure. Sparse frames do not establish
contact force, slip, or outcomes of unchosen actions.

The full 5k report and all 310 periodic metric rows through step 7,750 are preserved
with lossless gzip and original-byte hashes in [the readback](reports/20261012-nonlinear-repair-ttt/interim-health-round-1.json).
The checkpoint remains on the server with SHA256
`2589ed8cf7cfad709fd023c56d5ab3fd8f14ed377014cc11ab2703b03e7f093e`.
At copied compute step 7,749, optimization is 1.869605055 GPUh and evaluation
0.555449387 GPUh, with zero optimizer overrun. Preparation totals 0.032640774 GPUh
and retrieval setup 0.005036058 GPUh, giving **114.027637463 known cumulative GPUh**.
This is a partial measured-component ledger, not a full cluster bill. The failed
configuration preflight remains explicitly unmeasured. See [compute accounting](reports/20261012-nonlinear-repair-ttt/compute-accounting-round-1.json).

At the first snapshot, three fixed evaluations remained. The current status above
now has two pending. Publish all outcomes before stopping for review; no further
candidate is authorized by this monitor.

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
budget. The TRAIN-only preflight passed at 85.70 ms (max 97.00 ms) per full decision and 9.50 GiB peak GPU memory. Eight-rank no-update checks passed both planning depths with exact synchronized finite gradients and unchanged parameters. The exact frozen checkout repeats its registered safety checks before optimization; all preparation compute is logged separately.

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

All 45 frozen-source tests and both eight-rank no-update gradient checks passed.
The final full-controller preflight averages 87.64 ms. All 44 owned process/start
identities and source 200 hashes are captured in the launch report. Source
installation briefly lacked its root tree object; exact-tree recovery and a
clean full hash check completed before dispatch. The earlier TRAIN preflight
configuration mismatch and its resolution are also recorded; neither performed
an optimizer update. Known preparation components and omissions are in
`reports/20261012-nonlinear-repair-ttt/compute-accounting-launch.json`.

The preceding Bayesian result 71/76/77/77 was fully published and remotely verified
at ee1db8c before this fresh candidate was dispatched. Analysis-only updates to
`audit_repair_flow_ttt_run.py` and `repair_flow_ttt_details.py` were added after
source freeze. Use the reporting checkout for audits, preserving execution source.

Launch evidence and all prior comparisons were remotely verified at
`048cd087caf7be1e15da3f22280db5a2cbfec644`: 526 ledger artifact blobs and 18
report/status files matched, including the existing 57 evaluation reports.
[Verification receipt](reports/20261012-nonlinear-repair-ttt/publication-verification.json).
The existing completion monitor follows this registered run and publishes each
new result and the final comparison without scheduling further candidates.

The first fixed evaluation and its complete failure evidence were verified live
on origin/main at `65b4a2c19ee48011d6176e1775509ebec4840d30` on October 12,
10:15:08 UTC: all 548 ledger evidence blobs and 18 report/status files matched,
including 57 preserved completed evaluations and this new 5k report.
[First-evaluation publication receipt](reports/20261012-nonlinear-repair-ttt/publication-round-1-verification.json).

Both fixed evaluations and the complete second-round failure evidence were verified
live on origin/main at `a90440c0475f04c81bcecadb1f4692f345866e13` on October 12,
10:30:41 UTC: all 570 ledger evidence blobs and 18 report/status files matched,
covering 57 prior completed evaluations and the two active fixed reports.
[Second-evaluation publication receipt](reports/20261012-nonlinear-repair-ttt/publication-round-2-verification.json).
