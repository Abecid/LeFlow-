# Execution-aligned revision: interim results

October 9, 2026. The first three registered validations are complete. The selected 5,000-update
checkpoint scores **80/104 (76.92%)**, with zero controller timeouts. Training
continues toward the unchanged 20,000-update/8-GPU-hour ceiling; this is an
interim result, not the completed run or a final-test result.

| Checkpoint | New execution revision | Prior controller-grounded | Prior latent revision |
| --- | ---: | ---: | ---: |
| 5,000 | 80/104 | 79/104 | 73/104 |
| 10,000 | 78/104 | 78/104 | 75/104 |
| 15,000 | 79/104 | 75/104 | 74/104 |

The selected historical references remain controller-grounded **79/104** and
latent revision **77/104**. Against 79/104, seven cases improve and six regress:
net +1 success (+0.96 percentage points). The descriptive paired-reset interval
is [-5.77, 7.69] points and does not cover model-seed variation or repeated
development. Against 77/104, seven improve and four regress. One model seed and
reused validation do not establish reliable superiority or a faithful-SOTA win.

All 104 case IDs, reset seeds, episode hashes and training seeds match the saved
references. The same-world CEM22, repaired flow17, historical LeFlow27 and HWM8
records are preserved; no baseline was rerun. Their documented implementation
and world differences remain. [Registered method and limits](EXECUTION_REVISION_RUN.md).

## What the first round shows

Mean decision latency is86.74ms; p95 is90.65ms. All24 failures reach the200-action
cap, and no case exhausts the10-second controller allowance. Seventy-seven cases
succeed within100 primitive actions.

| Task | New /8 |
| --- | ---: |
| assembly | 3 |
| button-press-topdown | 8 |
| coffee-button | 8 |
| dial-turn | 7 |
| door-close | 8 |
| door-open | 6 |
| drawer-close | 8 |
| drawer-open | 8 |
| faucet-open | 3 |
| handle-press | 8 |
| pick-place | 1 |
| plate-slide | 8 |
| reach | 4 |

The implementation now uses prefix corrections in both inner search and outer
selection. Among119,184 evaluated target/iteration groups, correction changes
the lowest-cost action7,421 times (6.23%). It changes the best target within the
sampled pool41/4,966 times (0.83%); stall penalties also change target choice
41 times. These are same-pool diagnostics, not causal ablations or known outcomes
for rejected actions. Do not compare the6.23% action-group statistic with the old
0.43% target-choice statistic as if they measured the same thing.

The selected four-block prefix exactly matches the shifted previous plan in
11.68% of decisions. This identity check is not unique candidate provenance or
a causal estimate of warm-start benefit. Corrections lower MAE on observed chosen prefixes from0.002765 to
0.002003 (27.56%). Seventeen failures retain optimistic raw forecasts despite
nonpositive actual local progress in their last20 decisions; corrected forecasts
still do so in13. Nineteen failures end that window no closer to the final goal
in JEPA distance. Pick-place, assembly and faucet-open remain major weaknesses.

The short-window sampler is repaired, but the change is modest in frequency:
147/10,000 preflight draws start after control block40, and the last500-update
training window logs1.56% late samples. This supplies previously omitted labels;
it does not establish that missing late labels caused the observed failures.

## Evidence and next checkpoint

[Paired validation review](reports/20261009-execution-revision/validation-review.json),
[decision and training diagnostics](reports/20261009-execution-revision/extra-diagnostics.json),
and [compressed raw records](reports/20261009-execution-revision/run/) are saved.
This numerical review made no model/simulator calls. All ten complete 5k contact sheets were visually inspected at their full
1536×326 resolution. Captions, sampled action indices, frames and goal panels
are intact and readable; copied bytes were checked. The final source/checkpoint/compute
audit remains pending completion.

The frozen source stays `8be2808eaa09e539d30d07ee3760fc7b59717793`. The next registered
validation is at 20,000 updates; no settings are changed in response to these results.
[Live metrics](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/tyytbnn1).

## Second validation and saved visual failures

The 10k checkpoint scores **78/104**, with zero timeouts; 5k80/104 remains selected.
This is a mixed result, not monotonic improvement. Both completed rounds and
paired case changes are retained in the review above.

All ten [5k contact sheets](reports/20261009-execution-revision/contact-sheets-step-5000/)
show actual saved rollout frames and the registered goal. Assembly0 shows little
visible change after approaching the ring. Pick-place0 approaches the red object
and then moves the gripper toward the goal side while the object remains on the
table. Faucet-open3 moves past/right of the handle; reach3 remains offset from
its target. These cases fail at200 actions. Reach5 succeeds at45 actions.
The views do not establish contact forces, precise grasp mechanics or which
repair caused a behavior. [Visual verification](reports/20261009-execution-revision/visual-review-5000.json).

## Third validation

The 15k checkpoint scores **79/104**, with zero controller timeouts. The registered
sequence is80/78/79, and the5k checkpoint remains selected. The20k validation is
the last scheduled round. No source, scoring constant or training setting has
changed during the run. All three raw reports and paired changes are preserved.
