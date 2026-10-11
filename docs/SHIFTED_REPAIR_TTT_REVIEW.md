# Post-publication decision: hold further launches for review

The completed shifted-plan result is remotely verified at
[`aab136e797a05feea9b2318fa7512beab332287b`](https://github.com/Abecid/LeFlow-/commit/aab136e797a05feea9b2318fa7512beab332287b):
273 artifact blobs, 49 registered reports, 13 completed comparison entries and
identical 104-case identities. [Verification record](reports/20261011-shifted-repair-ttt/publication-verification.json).
The selected result is 79/104, below the preserved 80; the curve is 79/78/74/78.

**No further candidate is justified by the present evidence at this review.**
Stop launches under the user's explicit stopping rule. This is a research
judgment, not a claim that improvement is impossible, a revoked authorization,
or an operational blocker. Keep the completion monitor available for subsequent
explicitly authorized and registered work. Do not restart any completed run.

## Evidence gate for a next correction

The saved reports, all 800 periodic metrics, 20 visually inspected contact sheets,
causal-source audit and current implementation were reviewed. In particular:
`flow_jepa/execution/shifted_repair.py`, `retrieval_repair.py`, `repair_flow_ttt.py`,
`flow_controller.py`, `aligned.py` and `progress_ttt.py`.

| Possible explanation | What was actually observed | Decision now |
| --- | --- | --- |
| Shifted anchors never participate | They supply 47.02% of selected deployment anchors; deep training selects them 52.03–57.66% near the four milestones. Source and tests cover the shifted path. | No missing-use correction remains to implement. |
| The residual radius is the dominant obstacle | Clipped teacher coordinates fall 7.57%→4.52%, repair loss .424→.240, while success ends lower. This is not a failure-specific causal saturation measurement. | A larger radius or retuned loss would be a speculative parameter change. |
| Reduced proposal spread causes the regression | Matched initial spread contracts 62.75%/50.99%. The preceding retrieval run also contracts while success improves 75→80. | No evidence-supported diversity knob or sweep. |
| A stronger stall penalty will fix the late failures | Final failure episodes have 711 decisions with a positive penalty, versus 49 in successes; only 16 failed-case decisions change the chosen target because of it. These are observational and depend on visited states/lengths. | Existing feedback fires; the records do not show that another candidate would physically succeed. Retuning the penalty is not justified. |
| More recursive previous-plan synthesis matches deployment | Current synthesis omits recursive reuse and prior multi-target/stall choice; current states still come from expert actions, not the generated previous plan. | Adding recursion would not supply its missing physical transition outcomes or establish on-policy matching. No immediate extension is registered. |

Persistent pick-place and faucet failures, assembly regressions and reach
regressions are real development outcomes. Saved images show the object remaining
on the table in the illustrated pick-place case and little late visible motion
in several examples. They do not identify contact forces, precise 3D errors,
which rejected action would work, or a reliable new supervised target. No
counterfactual action inherits the recorded expert action's physical outcome.

## Primary-source cross-check

[DAgger, Ross et al., 2011, §3 and Algorithm 3.1](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)
collects learner-induced states and obtains expert actions for those states.
The present fixed expert corpus and synthesized prior plans do not reproduce
that procedure. New rollout/expert labels would change this run family's data
contract and are not collected here. This distinction does not prove dataset
shift caused these particular failures.

[Residual Policy Learning, Silver et al., 2018](https://arxiv.org/html/1812.06298v1)
uses reinforcement learning to improve an existing controller. Our bounded
residual-flow imitation and world-based selection are an adaptation/hypothesis,
not a faithful RPL reproduction or evidence for its guarantees. RPL's result
does not justify relabeling model-scored actions with unobserved physical rewards.

[Diffusion Policy's primary project](https://diffusion-policy.cs.columbia.edu/)
uses action-sequence generation and receding-horizon execution. It provides
context for plan reuse, not evidence that the current offline shifted-anchor
approximation, further recursion, or a new diversity setting will improve these
repeatedly used cases. No novelty or controlled component benefit is claimed.

## Preserved outcome and resumption condition

Retain the 80/104 reference, all negative results, frozen source/data/baselines
and sealed final tests. Known measured cumulative components are 98.593936 GPUh;
no new training, inference sweeps, added data or budget was spent on this decision.

A future candidate needs a specific new evidence-to-correction argument, causal
labels/accounting, meaningful tests, source freeze and pre-launch registration
within the standing limits; otherwise it needs a new explicit user scope decision.
The monitor should check for newly authorized registered work or publication
problems, stay quiet when unchanged, and never infer an experiment from idle GPUs.
