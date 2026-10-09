# Execution-aligned revision: interim results

October 9, 2026. The first registered validation is complete. The 5,000-update
checkpoint scores **80/104 (76.92%)**, with zero controller timeouts. Training
continues toward the unchanged 20,000-update/8-GPU-hour ceiling; this is an
interim result, not the completed run or a final-test result.

| Checkpoint | New execution revision | Prior controller-grounded | Prior latent revision |
| --- | ---: | ---: | ---: |
| 5,000 | 80/104 | 79/104 | 73/104 |

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

A shifted previous plan supplies the exact selected four-block prefix in11.68%
of decisions. Corrections lower MAE on observed chosen prefixes from0.002765 to
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
This numerical review made no model/simulator calls. Visual failure inspection
and the final source/checkpoint/compute audit remain pending completion.

The frozen source stays `8be2808eaa09e539d30d07ee3760fc7b59717793`. The next registered
validation is at10,000 updates; no settings are changed in response to these results.
[Live metrics](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/tyytbnn1).
