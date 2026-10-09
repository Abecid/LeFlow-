# Latent revision: completed results and failure analysis

October 9, 2026. **The one authorized run is complete: 20,000 updates and all four
registered validation rounds.** The selected 20,000-update checkpoint scores
**77/104 (74.04%)**, below the previous
controller-grounded method's **79/104 (75.96%)**. Selection follows the registered
highest-success, earliest-tie rule. Task success within the 10-second controller
allowance and 200 primitive actions is the primary outcome.

## What was implemented

A fresh 4,531,782-parameter policy uses a separate four-slot recurrent workspace
to condition action proposals on up to four already observed execution errors.
Four shared refinement iterations run at inference. The workspace is rebuilt
from causal history each decision; it is not a physical JEPA state or an
indefinitely persistent hidden state. Factual recorded actions/outcomes supervise
prefix and terminal cost corrections. No generated action receives a fabricated
physical-outcome label.

This candidate retains the strongest previous pipeline's **GMM action proposer,
retrieved targets and bounded CEM search**. It does not add a diffusion action
head, VLM, GRPO stage or simulator-generated training data. Candidate-specific
readouts use the workspace; the recurrent workspace itself does not iterate on
CEM feedback. The actual selection penalty uses only positive terminal-window
corrections; the signed executed-prefix correction is diagnostic.

The [registered method](LATENT_REVISION_RUN.md) gives the exact objective and
implementation. [Prior literature and code analysis](LARC_APPLICATION_20261009.md)
records LARC, RD-VLA, MPCoT and relevant overlap. This is an evidence-motivated
candidate, not a demonstrated novelty or latent-reasoning contribution.

## Same registered comparison, saved references

| Update | New successes | New success rate | Previous method at same update | New controller timeouts |
| --- | ---: | ---: | ---: | ---: |
| 5,000 | 73/104 | 70.19% | 79/104 | 0 |
| 10,000 | 75/104 | 72.12% | 78/104 | 0 |
| 15,000 | 74/104 | 71.15% | 75/104 | 0 |
| 20,000 | 77/104 | 74.04% | 78/104 | 0 |

| Method | Saved/selected successes |
| --- | ---: |
| Previous controller-grounded | 79/104 |
| New latent revision | 77/104 |
| Historical LeFlow adaptation | 27/104 |
| Same-world CEM | 22/104 |
| Repaired flow | 17/104 |
| Historical HWM adaptation | 8/104 |

Against the 79/104 reference, **4 cases improve and
6 regress** (-1.92 percentage
points). Case IDs, reset seeds, episode hashes and model seed 3072 match across
all rounds and references. The descriptive paired-reset interval is
[-7.69,
3.85] percentage points; it
does not cover model-seed variation, repeated development or checkpoint selection.

The task is a custom 13-task MetaWorld v3 image-goal benchmark, not standard
MT10/MT50. The unchanged training split contains 7,800 episodes (6,240 expert,
1,560 random); this policy and its route bank use the same **6,222 successful
expert episodes** as the previous candidate. Main sampling and global batch 64
are preserved, with additional causal history from those same episodes. The
fixed validation subset is **104 cases, eight per task**, from a 650-case pool.
The 3,200-case final-test set, including three held-out tasks, stays sealed.

The encoder/cache, normalization and fine-world checkpoint match the previous
79/104 candidate and same-world CEM. Historical LeFlow/HWM results retain the
previously documented world/representation/sampler differences. They are saved
adaptations, not a controlled claim of beating corrected published SOTA.
No baseline was rerun. Reused validation supports development decisions,
not an unbiased final-test or long-horizon generalization claim.

| Task | New selected /8 | Previous selected /8 |
| --- | ---: | ---: |
| assembly | 2 | 3 |
| button-press-topdown | 8 | 8 |
| coffee-button | 8 | 8 |
| dial-turn | 8 | 7 |
| door-close | 8 | 8 |
| door-open | 7 | 8 |
| drawer-close | 8 | 8 |
| drawer-open | 8 | 8 |
| faucet-open | 3 | 4 |
| handle-press | 8 | 8 |
| pick-place | 0 | 1 |
| plate-slide | 6 | 6 |
| reach | 3 | 2 |

At the selected checkpoint, all 27 failures reach the action
cap, with 0 controller timeouts. Mean decision latency is
85.21 ms, p95 is
88.94 ms, and mean cumulative controller time
is 4.398 seconds per case.
71 cases succeed within 100 actions.
Return is logged but is not the primary comparison: episode lengths differ.

## Better prediction does not establish better control

| Update | Raw prefix MAE | Corrected prefix MAE | Positive terminal penalty | Changed best anchor in sampled pool |
| --- | ---: | ---: | ---: | ---: |
| 5,000 | 0.002690 | 0.001923 | 9.06% | 0.194% |
| 10,000 | 0.002907 | 0.002141 | 12.78% | 0.455% |
| 15,000 | 0.002584 | 0.001861 | 15.04% | 0.516% |
| 20,000 | 0.002665 | 0.001916 | 15.41% | 0.428% |

For the selected checkpoint, corrections reduce error on 5,264
observed executed prefixes by **28.12%**.
Raw predictions classify 1267/5216
positive-progress prefixes incorrectly; corrected predictions classify
878/4619 incorrectly.
These denominators differ. Correlation changes from
0.712 to
0.731. All are dependent, chosen-action traces,
not observed outcomes or ranking accuracy for rejected candidates.

The actual terminal penalty is positive in
827/5368 decisions and averages
0.0001840 cosine-cost units. It changes the best
anchor within the sampled pool in only
23/5368 decisions
(0.428%). This excludes action
changes within the same anchor and changes to the sampled pool; it is not the
complete causal effect of the new model.

Mean selected prefix correction is +0.001441,
while mean terminal correction is -0.002089.
The conservative score clips negative terminal corrections to zero. This exposes
a concrete alignment weakness: the useful correction to the executed prefix
is diagnostic, while ranking uses a different, unexecuted horizon. Improved
forecasting therefore need not improve the action that is actually executed.
The run does not isolate whether recurrence, memory, changed proposals or
calibration causes its task-level changes; no extra ablation was run.

## Failure modes and coverage limits

In the last 20 decisions, 18
failed cases retain positive mean raw predicted target progress while mean
observed target progress is nonpositive. The corrected forecast still does so
in 12 cases.
19 failures end that
window no closer to the final goal in JEPA distance; among them,
3
show positive mean progress toward their changing local targets. These are
latent-distance diagnostics between pre-decision observations, not physical
reward or proof that every necessary intermediate setback is wrong.

A code/manifest audit found a shared coverage restriction: both candidates
retain the old 60-control-step sampling window, although the action head
supervises only five steps. All eligible demonstrations have 100 control
blocks. Main starts are at most block 40, and supervised chunks end by primitive
action 90 of 200. Later goal/context states are available; later action targets
are omitted. Of the 6,222 eligible episodes, 1,152 record first success after
primitive index 80. This includes all 480 assembly and all 480 drawer-open
episodes. Drawer-open still succeeds consistently, so this is an inherited
coverage limitation, not a sufficient explanation of failure or of the new
method's regression. See [training coverage](reports/20261009-latent-revision/training-coverage.json).

Expert-history calibration also differs from policy-history recovery and
arbitrary generated actions. The available traces cannot supply true outcomes
for unexecuted chunks. These gaps are candidates for the next design, not
license to treat world predictions as physical labels.

## Saved visual evidence

All ten complete contact sheets at the selected checkpoint were visually
inspected at readable resolution; labels/panels are intact. Each contains five
uniformly sampled recorded frames and the registered image goal. Initial-frame
identity, trajectory lengths and published bytes were checked. Rendering made
no new model or simulator calls. The fixed cases are illustrative, not a
representative success/failure sample.

- [assembly/00000](reports/20261009-latent-revision/contact-sheets-step-20000/assembly-00000.png): success, 84 primitive actions.
- [coffee-button/00000](reports/20261009-latent-revision/contact-sheets-step-20000/coffee-button-00000.png): success, 36 primitive actions.
- [dial-turn/00000](reports/20261009-latent-revision/contact-sheets-step-20000/dial-turn-00000.png): success, 93 primitive actions.
- [door-close/00000](reports/20261009-latent-revision/contact-sheets-step-20000/door-close-00000.png): success, 64 primitive actions.
- [drawer-close/00004](reports/20261009-latent-revision/contact-sheets-step-20000/drawer-close-00004.png): success, 70 primitive actions.
- [faucet-open/00003](reports/20261009-latent-revision/contact-sheets-step-20000/faucet-open-00003.png): failure, 200 primitive actions.
- [handle-press/00000](reports/20261009-latent-revision/contact-sheets-step-20000/handle-press-00000.png): success, 15 primitive actions.
- [pick-place/00000](reports/20261009-latent-revision/contact-sheets-step-20000/pick-place-00000.png): failure, 200 primitive actions.
- [reach/00003](reports/20261009-latent-revision/contact-sheets-step-20000/reach-00003.png): failure, 200 primitive actions.
- [reach/00005](reports/20261009-latent-revision/contact-sheets-step-20000/reach-00005.png): success, 50 primitive actions.

Pick-place0 approaches the red object, then shows very similar gripper/object
poses from action 50 through 200. The object remains near its starting table
location, different from the registered goal. Faucet-open3 approaches the
faucet region, but its later gripper positions drift right of it and the final
arrangement differs from the goal. Reach3 remains offset from its goal with
little visible change in the later sampled frames. All three fail at 200 actions.
The views do not establish contact forces, a precise grasp failure mechanism,
or the causal contribution of the latent workspace.

Assembly0 succeeds in 84 actions, recovering from its 10k failure. Reach5 also
succeeds, in 50 actions, after failing at 5k. Coffee-button0, dial-turn0,
door-close0, drawer-close4 and handle-press0 are the other five saved successes.
Task success follows the simulator criterion, not exact pixel equality with
the goal image. All ten complete contact sheets were inspected at 1536×326;
panels, case labels, action indices and goal captions are intact and readable.


## Compute, verification and next direction

All eight A800 GPUs were used with global batch 64 and seed 3072. Optimization
used **3.243509 aggregate GPU-hours**, below the same
8-GPU-hour ceiling, and stopped at the 20,000-update limit. Registered validation
used **2.047476 GPU-hours**; training-only preflight used
0.005534 GPU-hours. The new measured subtotal is
**5.296518 GPU-hours**. Other held-GPU occupancy has a separate
0.144280 GPU-hour upper bound.
These are accounting components, not a complete cluster bill. Equal ceilings
and update counts do not imply equal realized FLOPs: the previous candidate
used 2.421563 optimization GPU-hours. No shared world/cache cost is counted twice.
Known cumulative campaign components now total
52.377571 GPU-hours.

All four previous and six new behavioral tests passed. Full-size preflight
verified causal inputs, identical main samples and unchanged world parameters.
The final CPU-only audit checks immutable source, data-manifest, world and bank identities,
finite best/final checkpoints, 401 unique scheduled training metric records,
all four validation metric records, causal history lengths, the 480-world-step
per-decision contract and the optimization ceiling. The
[source](https://github.com/Abecid/LeFlow-/tree/899f8f2564e011ae87065a70acf28c50dade9997)
was unchanged throughout this run. W&B synced and all owned workers exited;
all eight GPUs were idle at the [final health check](reports/20261009-latent-revision/final-health.json).

The evidence supports prioritizing **alignment between executed-prefix evidence
and actual decision selection**, while preserving route-level goal value and
necessary intermediate setbacks. Coverage of late recorded action chunks is a
second concrete issue to address within the existing data. Better calibration
alone is insufficient. Neither fix is claimed to work yet; no further method,
seed, ablation, baseline rerun or final test has been launched.

Reproduce the analysis without model or simulator calls:

```sh
python scripts/analysis/controller_grounded.py --repo . \
  --run-dir docs/reports/20261009-latent-revision/run \
  --output docs/reports/20261009-latent-revision/validation-review.json
python scripts/analysis/latent_revision_details.py --repo .
```

Evidence: [all-case review](reports/20261009-latent-revision/validation-review.json),
[extra failure diagnostics](reports/20261009-latent-revision/extra-diagnostics.json),
[compute ledger](reports/20261009-latent-revision/compute-accounting.json),
[postrun audit](reports/20261009-latent-revision/postrun-audit.json), and
[W&B](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/5thxkk6y).
