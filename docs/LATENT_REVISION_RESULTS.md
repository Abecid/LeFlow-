# Latent revision: results and failure analysis

October 9, 2026. **Training is ongoing; two registered checkpoints are complete.**
The 10k model scores **75/104 (72.12%)**, up from 73/104 at 5k and still below
the previous controller-grounded method's 79/104. The 15k and 20k evaluations
remain pending. Source and the registered algorithm remain unchanged.

## Second checkpoint: 10,000 updates

The same 104 cases produce seven improvements and five regressions relative
to 5k. Reach rises from 1/8 to 4/8, while assembly falls from 2/8 to 0/8.
Compared with the saved 79/104 reference, five cases improve and nine regress.
All 29 failures reach the 200-action cap; none exhausts the controller clock.

Across 5,396 observed prefixes, corrections reduce MAE from 0.002907 to
0.002141 (26.35%). Terminal penalties are positive in 12.78% of decisions and
change the best anchor in the actual sampled pool in 25/5,500 decisions
(0.455%). These are chosen-action diagnostics, not evidence that rejected
candidates would execute better.

All ten [10k contact sheets](reports/20261009-latent-revision/contact-sheets-step-10000/index.json)
were visually inspected at full resolution and checked against the exported
bytes. Assembly0 now stalls after its initial approach; reach5 now succeeds
in 44 actions. Pick-place0 moves toward the target region but leaves the red
object on the table, visibly different from the registered goal. Faucet-open3
and reach3 still fail. The other five saved cases succeed. These observations
are descriptive; sparse views do not establish contact forces or exact grasp
failure causes.

The earlier checkpoint evidence below is retained explicitly as the **5k**
analysis. The machine-readable review contains both rounds and the currently
selected 10k comparison.

## First checkpoint comparison: 5,000 updates

| Method | Selected/saved validation successes |
| --- | ---: |
| Previous controller-grounded | 79/104 |
| New latent revision,5k | 73/104 |
| Historical LeFlow adaptation | 27/104 |
| Same-world CEM | 22/104 |
| Repaired flow | 17/104 |
| Historical HWM adaptation | 8/104 |

All104 case IDs, reset seeds, episode hashes and model seed3072 match. Relative
to79/104, four cases become successful and ten regress, a−5.77 percentage-point
difference. The same-world reference is the important iteration comparison;
beating the historical adaptations does not establish improvement over our
strongest method. Historical world/representation/sampler confounds persist.
No baseline or final test was rerun. Validation is reused for development and
selection; this is not an unbiased final-test or multi-seed estimate.

| Task | New5k /8 | Previous selected5k /8 |
| --- | ---: | ---: |
| assembly | 2 | 3 |
| button-press-topdown | 8 | 8 |
| coffee-button | 8 | 8 |
| dial-turn | 6 | 7 |
| door-close | 8 | 8 |
| door-open | 7 | 8 |
| drawer-close | 8 | 8 |
| drawer-open | 8 | 8 |
| faucet-open | 4 | 4 |
| handle-press | 8 | 8 |
| pick-place | 0 | 1 |
| plate-slide | 5 | 6 |
| reach | 1 | 2 |

All31 failures reach200 primitive actions. None exhausts the10-second controller
allowance. Mean decision latency is85.55ms, p95 is89.08ms, and mean cumulative
controller time is4.675s per episode. The regression is not explained by clock
exhaustion. Sixty-seven cases succeed within100 primitive actions.

## Prediction improves, but its effect on selection is limited

Across5,579 observed executed prefixes, the corrected progress estimate reduces
mean absolute error from0.002690 to0.001923 (28.52%). Predicted-positive prefixes
with nonpositive observed progress decrease from25.84% of raw-positive estimates
to19.80% of corrected-positive estimates. The denominators differ:5,537 raw and
4,667 corrected positive predictions. Correlation rises from0.714 to0.728.
These are paired diagnostic measurements on the same chosen actions and
dependent decisions, not counterfactual candidate ranking or task success.

The controller's actual conservative penalty applies to the **terminal-window**
correction. It is positive for515/5,683 decisions (9.06%) and averages only
0.0000595 cosine-cost units. Corrections change the best anchor in the actual
sampled candidate pool in11/5,683 decisions (0.194%). This metric does not count
changes to actions within the same anchor or changes to the generated pool;
it is not a complete causal effect estimate.

Mean predicted prefix cost correction is+0.001601, while mean terminal correction
is−0.002461; the conservative rule clips negative terminal corrections to zero.
Thus the useful prefix estimate is largely diagnostic in this formulation,
while decision penalties operate on a different, unexecuted horizon. This is
a concrete objective-alignment weakness to investigate after the run. It does
not prove that a particular alternative score would recover the ten regressions.

On the last500 logged training updates before5k, deep and shallow calibration
losses average0.004892 and0.005013. Their small difference does not establish
a task-level benefit from recurrence or memory. This training diagnostic is
already part of the objective; no extra depth ablation was run.

## Stalling and visual evidence

Twenty of31 failures have positive mean raw predicted progress but nonpositive
observed progress in their last20 decisions. Late stalling remains. The new
mean action likelihood loss improves during training, but the task-success
result is worse; a better supervised proxy alone is insufficient.

Ten preselected saved-rollout contact sheets were rendered without new model
or simulator calls, checked against registered initial images, inspected at
full resolution, and byte-verified in the published copy. All panels and labels
are readable. The [image index](reports/20261009-latent-revision/contact-sheets-step-5000/index.json)
records exact case IDs, steps and source archives.

- **Pick-place0 fails:** sampled frames show an initial approach, followed by
  very similar poses from50 to200 actions, different from the registered goal.
  Its last20 decisions use retrieved anchors; mean observed target progress is
  +0.000193 versus raw predicted+0.002068. The views do not establish a precise
  grasp/contact failure mechanism.
- **Reach3 and reach5 fail:** late poses remain offset from the goal. Their last
  20 decisions all choose the direct goal. Mean observed progress is−0.000204
  and−0.000245, versus raw predicted+0.000616 and+0.000487, respectively.
- **Faucet-open3 fails:** late frames show little visible change. The final
  retrieved anchor is from `train/reach/00599`, with a60-step route span. Its
  tail mean observed progress is−0.000410 versus predicted+0.001722. Cross-task
  retrieval is permitted; this example motivates inspecting object-specific
  support, not declaring every cross-task anchor invalid.
- The other six inspected cases succeed: assembly0(86 actions), coffee-button0
  (38), dial-turn0(89), door-close0(66), drawer-close4(78), handle-press0(15).
  These are the fixed saved cases, not a representative success sample.

No observed outcomes exist for rejected candidates or unexecuted full chunks.
Training corrections on factual expert transitions remains different from
calibrating arbitrary generated actions and recovery states. This distribution
gap, the limited use of prefix evidence in ranking, proposal changes, and route
support are hypotheses for further work. No new method, ablation or training
restart has been launched based on this checkpoint.

## Evidence and continuation

See the [implemented formulation](LATENT_REVISION_RUN.md),
[all-case analysis and paired regressions](reports/20261009-latent-revision/validation-review.json),
[raw compressed checkpoint report](reports/20261009-latent-revision/run/validation/step_0005000.json.gz),
and [W&B](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/5thxkk6y).
The5k optimization charge is0.752739GPUh; its registered validation costs
0.567791GPUh. One-GPU preflight is separately0.005534GPUh. The source is frozen
at899f8f2564e011ae87065a70acf28c50dade9997, with the same20k/28,800GPU-second ceiling.
Training continues to its registered endpoint and periodic evaluations. The
10k optimization charge is 1.573701 GPUh; the live compute snapshot may also
include updates after that checkpoint.
