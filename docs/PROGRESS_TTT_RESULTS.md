# Progress TTT: completed run and decision-quality gap

October 10, 2026. The single authorized run finished all 20,000 updates and four
104-case development evaluations. The selected 15k checkpoint scores **78/104
(75.00%)**, below the preserved execution-revision GMM's **80/104 (76.92%)**
and flow-reasoning's **79/104 (75.96%)**. The final checkpoint scores 74/104.
The new adaptation has not demonstrated better control or useful test-time scaling.

The [method](PROGRESS_TTT.md) fits episodic, goal-relative prediction-error weights
from the last four executed transitions. These weights condition the action-flow
workspace and correct candidate costs. The frozen encoder/world and physical JEPA
states remain separate from internal reasoning. Three propose/predict/revise rounds
generate ten-action plans; two primitive actions are executed before replanning.
This is action-flow generation with JEPA verification, not LeFlow's latent-path
flow followed by inverse dynamics. No gradient guidance inside the flow solver
or one-step distillation is implemented.

## Fixed comparison contract

Custom image-goal control on 13 MetaWorld v3 tasks. The same 6,222 successful
expert episodes train the policy and supply retrieval; the full world-training
split has 7,800 episodes. Seed3072, global batch64, 20k-update/28,800 optimization
GPU-second ceilings, frozen V-JEPA2.1 encoder and frozen shared world. Training
samples 1.28M windows with replacement, not a conventional epoch schedule.
The identical 104 development cases (eight per task) are evaluated at each
milestone. Success requires the task criterion within 200 primitive actions and
10 cumulative controller seconds, including adaptation. Highest development
success selects the checkpoint, with earliest ties. The sealed final tests remain
unopened. There was no baseline retraining or additional candidate.

Equal caps do not mean equal realized compute or convergence. These repeatedly
used development cases and one training seed do not establish publication-level
general superiority. Preserved LeFlow21/104, CEM24/104 and HWM5/104 remain the
[fixed shared-world comparisons](RELEASE_BASELINE_RESULTS.md); HWM is a paper port,
and none of these scores is a native published-benchmark SOTA comparison.

## Results and measured failure signals

| Updates | Success /104 | Timeouts /104 | Raw prefix MAE | Prior prefix MAE | Adapted prefix MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 77 | 23 | .002709 | .002721 | .001727 |
| 10,000 | 76 | 23 | .002643 | .002491 | .001622 |
| 15,000, selected | 78 | 21 | .002547 | .002237 | .001656 |
| 20,000 | 74 | 25 | .002681 | .002339 | .001709 |

MAE compares predicted versus observed goal-relative progress on the **same
executed prefixes** of each checkpoint. The prior estimate uses the logged prior
correction and the same [0,2] cost clipping as the adapted diagnostic. At15k,
adaptation lowers this error by26.0% relative to the offline prior. This is a
paired prediction diagnostic, not an unadapted-policy rollout or proof of improved
counterfactual ranking. Only executed actions have observed outcomes. The deployed
ranking applies positive-only correction penalties; signed calibration diagnostics
are not themselves the complete selection objective.

The selected checkpoint wins five cases and loses seven against the80/104
reference. Against prior flow79/104 it wins five and loses six. Case hashes,
reset seeds, model seeds and task identities match. From15k to20k it gains reach2
but loses assembly6, door-open2/4/6 and faucet-open1.

At15k, eight tasks score8/8. Remaining tasks: assembly2/8, door-open7/8,
faucet-open3/8, pick-place0/8 and reach2/8. The hard-task failures persist despite
better local calibration. Successful demonstration training provides limited
evidence for recovery from stalls and contact errors; this is a plausible coverage
limitation, not a demonstrated causal explanation.

Matched initial final-round proposal dispersion falls from.057506 at5k to.022135
at20k (61.5%); all104 cases decrease with the same initial routes and empty
history. This does not measure useful diversity or prove collapse causes failure.
At15k, 20 failed episodes have an optimistic raw prediction with nonpositive
actual progress in their last20 decisions; six remain optimistic after signed
correction. At20k the corresponding counts are20 and9. These are trace summaries,
not counterfactual evidence that another action would succeed.

Runtime also constrains this candidate. Selected mean controller latency is
102.502ms, p95 107.454ms, with21 timeouts; prior selected flow measured99.42ms
and10 timeouts. Two actions per decision require100 decisions to spend the full
200-action allowance, so100ms is the approximate average threshold under10s.
The small latency increase can therefore truncate late attempts. This does not
prove that removing the overhead recovers the lost successes. Component timing
is needed before attributing it to flow, adaptation, world rollout or retrieval.

## Completion and verification

Frozen execution source: `8ae86855738370b06c8f59b96150ac68c389d010`.
Server record: `/home/mtxu/adam/LeFlow-experiments/20261010-progress-ttt`.
Selected checkpoint: `campaign/runs/progress_ttt_3072/best.pt` (15k);
all milestone checkpoints and journals remain on the server.
[W&B](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/to9im8h6) is finished.
The coordinator exited and all eight GPUs were idle at the completion readback.

Optimization used3.542265 GPU-hours; the four validations used2.013268 GPU-hours.
Run preflight adds.005697 GPU-hours, giving5.561230 timed GPU-hours, excluding
process startup and CPU preparation. Earlier implementation-only GPU checks were
reported separately (.010649 GPU-hours); they are not hidden inside optimization.
Training plus validation wall time was42.16min; coordinator wall time44.46min.

The read-only CPU [final audit](reports/20261010-progress-ttt-run/final-audit.json)
verified140 runtime hashes, clean frozen source, config/manifest/world identity,
exact bank tensor identity,6,222 training episode identities, finite best/last
checkpoints, selected milestone tensor identity,401 periodic training records,
four validation records, matching cases, causal history/reset, fixed fast weights
within each search, score reconstruction and480 world transitions per decision.
Prelaunch29 behavior tests are retained; no new model or simulator calls were
needed for this completion analysis.

Reproduce the saved-record analysis with
`scripts/analysis/progress_ttt_details.py` and the server checkpoint audit with
`scripts/analysis/audit_progress_ttt_run.py`. Compressed validation records, logs,
paired outcomes, compute usage and online diagnostics are preserved in
[the evidence directory](reports/20261010-progress-ttt-run/).

The next research recommendation is [execution-calibrated flow guidance](FLOW_RESEARCH_ANGLE_20261010.md).
It is a hypothesis, not a newly implemented or queued experiment. Baselines,
completed execution trees and sealed tests remain fixed.
