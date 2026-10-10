# Execution-aligned revision: completed comparison and failure analysis

October 9, 2026. **The authorized fresh run is complete: 20,000 updates and all
four fixed 104-case validations.** Successes were **80, 78, 79 and 76**. The
registered selection rule retains the **5,000-update checkpoint at 80/104
(76.92%)**, compared with the previous controller-grounded **79/104 (75.96%)**
and latent-revision **77/104 (74.04%)** references.

This is a small best-checkpoint gain, not reliable superiority or a SOTA claim.
Against 79/104, seven cases improve and six regress: **+0.96 percentage points**.
The descriptive paired-reset interval is [-5.77, 7.69] points; it does not cover
model-seed variation, repeated development or checkpoint selection. Against
77/104, seven cases improve and four regress. All case IDs, reset seeds, episode
hashes and training seeds match the saved references.

## What model and task these results describe

The current method is a 4.53M-parameter GMM action policy with an execution-error
workspace, training-route retrieval and bounded CEM search. It is trained from
scratch over a frozen V-JEPA 2.1 ViT-L image encoder and a separately trained,
now-frozen action-conditioned latent predictor. It is not a fine-tuned LeFlow
checkpoint, the published Flow-JEPA dynamics model, or V-JEPA2-AC used as-is.
There is no flow-matching or diffusion objective in the current policy.

The project began with a LeWM/PushT flow-versus-BTM setup and CPU fixture checks.
The user then removed BTM and authorized V-JEPA 2.1/MetaWorld joint state-action
flow planning. That is the major early task/backbone change. The original
MetaWorld run and its representation repair precede the three recent policies:
controller-grounded, latent revision and execution revision. All three recent
policies share the same frozen world, features, normalization, episode inventory
and validation cases. The flow-to-GMM change happened at controller-grounded;
the latest iteration changes within-episode sampling, not tasks or episodes.
Older causal video-history features were replaced by repeated-current-image
features during the earlier world/goal-alignment repair. Historical LeFlow/HWM
scores therefore do not share the current world/representation.

The custom image-goal task uses current RGB and one final goal image to control
13 MetaWorld v3 manipulation tasks, including assembly and pick-place. The full
offline training split is 7,800 episodes (600/task): 6,240 scripted-expert and
1,560 random episodes, each with 200 primitive actions, totaling 1,560,000
recorded transitions. The world uses both modes; current policy and retrieval
use the same 6,222 successful expert episodes. No privileged state or expert
test-time subgoal sequence enters the policy. There are 650 validation episodes;
the same fixed 104 are used for these four checkpoint evaluations. Goals are
screened for constructibility with the scripted expert before model training.
These results are not an unconditional reset-distribution success rate.

The selected method is evaluated by task-macro environment success under the
200-action/10-second cumulative-controller allowance. Latent costs, returns and
calibration are diagnostics, not substitutes for task success. The reserved
3,200-case final test covers the 13 seen tasks and window-open, handle-pull and
push; it was not run or read in this iteration. Reusing these development cases
does not establish a long-horizon generalization or published-SOTA result.

[LeFlow](https://arxiv.org/html/2608.24855v1) learns a goal-conditioned flow over
latent path interiors, decodes transitions into actions and ranks frozen-world
rollouts. Its main backbone is LeWM, and its reported benchmarks are TwoRoom,
PushT, Reacher and OGBench-Cube, not this MetaWorld split. The separate
[Flow-JEPA paper](https://arxiv.org/abs/2608.29029) instead uses flow matching for
action-conditioned future-state prediction. Neither name accurately describes
the current GMM policy's learning objective.

Our current hypothesis is that observed prediction-execution discrepancies can
improve selection of the action prefix a bounded controller will actually
execute. The recent mechanism makes prefix corrections affect inner action
search and outer target choice. Its numerical calibration benefit is measured;
a robust task-success benefit attributable to latent reasoning or this mechanism
is not. Retrieval, GMM proposals and plan reuse are supporting choices, not
independent novelty claims. A return to a flow-specific claim would require an
actual flow component and evidence for its benefit.

## Implemented changes and preserved budget

The [registered candidate](EXECUTION_REVISION_RUN.md) makes four fixed changes:

- Score both the executed prefix and terminal window, including positive learned
  error corrections, in inner action search and outer target selection.
- Reuse the shifted previous plan after observation, retaining fresh alternatives
  and removing the duplicated direct-goal proposal.
- Apply a bounded, decaying target penalty after repeated observed optimistic
  failures near the same state and target.
- Sample all valid five-step starts up to first success, then choose a feasible
  future goal, repairing the inherited 60-step-window restriction.

The unchanged 4,531,782-parameter policy was initialized from scratch. The encoder,
fine world, normalization and retrieval-bank tensors are unchanged. The full
training split has 7,800 episodes; this policy/bank use the same 6,222 successful
experts. Global batch 64 and 20,000 updates give 1,280,000 sampled windows, not a
conventional epoch count. The start/goal distribution changes explicitly; the
episode set does not. Only one model seed, 3072, was run.

The same 104 validation cases are used at each checkpoint. Every controller has
200 primitive actions and 10 seconds of cumulative controller computation per
case. Search remains 480 world transitions per decision, and executes the chosen
first block of two primitive actions. All eight A800 GPUs accelerated this run.
No baseline rerun, extra seed, ablation, automatic next variant or final test was
launched. The reserved 3,200-case final test stays sealed.

## Success comparisons

| Update | Execution revision | Previous controller-grounded | Previous latent revision |
| --- | ---: | ---: | ---: |
| 5,000 | 80/104 | 79/104 | 73/104 |
| 10,000 | 78/104 | 78/104 | 75/104 |
| 15,000 | 79/104 | 75/104 | 74/104 |
| 20,000 | 76/104 | 78/104 | 77/104 |

| Saved/selected reference | Successes /104 |
| --- | ---: |
| New execution revision, 5k | **80** |
| Previous controller-grounded, 5k | 79 |
| Previous latent revision, 20k | 77 |
| Historical LeFlow adaptation | 27 |
| Same-world CEM | 22 |
| Repaired flow | 17 |
| Historical HWM adaptation | 8 |

Historical LeFlow/HWM implementations retain documented sampler, architecture,
world and representation differences. Their saved scores do not establish a
win over faithful corrected published implementations. This custom 13-task
MetaWorld v3 image-goal benchmark is not standard MT10/MT50. Reused development
validation is not an unbiased held-out result or a long-horizon generalization
claim. Equal update/compute ceilings do not mean identical realized FLOPs.

At the selected checkpoint, all 24 failures reach the 200-action cap; there are
**zero controller timeouts**. There are also zero timeouts in the other three
rounds. Mean decision latency is **86.74 ms**,
p95 is **90.65 ms**, and cumulative controller
time averages **4.142 seconds per case**.
Seventy-seven cases succeed within 100 primitive actions. Task success under the
allowance is primary; return and latent-distance diagnostics are secondary.

| Task | Selected checkpoint /8 |
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

## Do the corrections affect decisions?

Yes, within the actual sampled candidate pools. At the selected checkpoint:

- Learned corrections change the preferred action in **7,421/119,184 target/
  iteration groups (6.23%)**.
- Corrections change the best target in **41/4,966 decisions (0.83%)**. Stall
  penalties separately change the best target in 41 decisions.
- The selected four-block prefix exactly matches the shifted previous plan in
  **11.68%** of decisions. This does not establish unique candidate provenance
  or the causal benefit of reuse.
- On 4,862 observed chosen prefixes, correction reduces MAE from
  **0.002765 to 0.002003 (27.56%)**.

These are decision/calibration diagnostics, not observed physical outcomes for
rejected actions. The 6.23% action-group statistic must not be compared with the
previous 0.43% target-choice statistic as if they measured the same effect.
The bundled run cannot isolate which repair caused the one-case net improvement.

## Remaining failure modes

**More training did not improve task success.** Assembly scores 3/8 at 5k and
0/8 at each later checkpoint. Pick-place scores 1/8, 0/8, 2/8 and 0/8. Lower
training loss does not establish better recovery or executable planning.

An additional read-only review pairs 5k with 20k: seven cases regress and three
improve, giving the net four-success decline (3.85 percentage points). Regressions
are assembly5/6/7, dial-turn2, door-open7, pick-place4 and reach1; improvements
are dial-turn6 and door-open0/1. This is not a monotonic decline at every saved
checkpoint, and a small selected-development-set difference does not establish
a general training-length effect.

| Checkpoint | Success /104 | Training action NLL | Initial proposal diversity | Online corrected-prefix MAE |
| --- | ---: | ---: | ---: | ---: |
| 5k | 80 | -1.40190 | 0.07522 | 0.002003 |
| 10k | 78 | -1.67952 | 0.04815 | 0.002028 |
| 15k | 79 | -1.78656 | 0.03756 | 0.002112 |
| 20k | 76 | -1.82393 | 0.03666 | 0.002120 |

Training NLL is the mean of ten logged minibatches in the final 500 updates
before each checkpoint; lower is better and continuous-density NLL may be
negative. Training calibration loss similarly falls from 0.004948 to 0.003809.
These are sampled training statistics, not a fixed held-out action-likelihood
test. Online MAE uses executed prefixes from different policy-induced states;
its change is not a same-state causal comparison. Values come from the saved
[diagnostics](reports/20261009-execution-revision/extra-diagnostics.json) and
[initial-state analysis](reports/20261009-execution-revision/initial-proposal-diversity.json).

The leading explanation is a mismatch between better fitting expert actions
and maintaining useful alternatives/recovery under a small candidate budget.
The policy's Gaussian scales are learned within [0.05,0.5], but the saved
diversity statistic does not separate scale shrinkage, mixture weights and
component means. Training histories/calibration use successful expert episodes;
evaluation can enter stalled or failed-contact states that those histories do
not cover. The objective supervises action likelihood and latent prediction
error, not environment success. A hand approaching the goal while leaving an
object behind is compatible with a misleading visual-distance objective.
These are supported hypotheses, not proven causes or evidence that mixture
entropy collapsed. No new model or simulator calls were used for this review.

Numeric instability, changed validation identities, different world weights and
controller time exhaustion are not supported explanations: the final audit
verified finite checkpoints and matching identities, and every round has zero
controller timeouts. We have no measured basis for solving this by simply
training longer. The selected 5k checkpoint remains preserved.

A matched-initial-state diagnostic provides a concrete lead: initial proposal
diversity falls from **0.07522 at 5k to 0.03666 at 20k**, about **51%**, and is
lower in **101/104 cases**. These are the same reset/goal pairs with no prior
history or warm start, and the same per-case evaluation seeds. The statistic
includes fixed recorded proposals; it is not a direct measurement of mixture
entropy or proof that loss of exploration caused the success regression.
[Initial-state evidence](reports/20261009-execution-revision/initial-proposal-diversity.json).

At 5k, **17/24 failures** retain positive mean raw predicted local progress but
nonpositive observed progress over their last 20 decisions; corrected forecasts
still do so in 13 cases. Nineteen failures end that window no closer to the final
goal in JEPA distance. At 20k, 20/28 failures have the raw optimism pattern and
23 end no closer to the goal. State-distance forecasts remain imperfect proxies
for task completion, especially outside expert-like recovery states.

The coverage repair is real but modest in frequency: 147/10,000 training-only
preflight draws start after control block 40. This adds formerly omitted labels;
it does not establish that missing late labels caused the main manipulation
failures. Later recorded actions were already available in the retrieval bank.

## Inspected visual evidence

Every one of the ten complete [selected-checkpoint contact sheets](reports/20261009-execution-revision/contact-sheets-step-5000/)
was inspected at its original 1536×326 resolution. Captions, five sampled frames,
action indices and registered goal panels are intact and readable. Initial RGB,
trajectory length and copied/published file hashes were checked. Rendering used
saved frames only, with zero new model or simulator calls.

- Assembly0 shows little visible change after approaching the ring and fails.
- Pick-place0 approaches the object, then moves the gripper toward the goal side
  while the red object remains near its initial table location; it fails.
- Faucet-open3 moves past/right of the handle; reach3 remains offset from its
  target. Both fail at 200 actions.
- Reach5 succeeds at 45 actions. Coffee-button0, dial-turn0, door-close0,
  drawer-close4 and handle-press0 are the other five saved successes.

These fixed illustrative cases do not establish contact forces, precise grasp
mechanics or the causal contribution of any repair. Robot motion without object
completion and reduced proposal diversity are leads for future investigation,
not confirmed causal explanations. No follow-up experiment was run.

## Compute, verification and saved artifacts

Optimization used **3.233375 aggregate GPU-hours**, below the
same 8-GPU-hour ceiling, stopping at the 20,000-update limit. Registered validation
used **1.945972 GPU-hours** and training-only
preflight used **0.005719 GPU-hours**. Their measured
subtotal is **5.185067 GPU-hours**. Other held GPU
occupancy, including loading and upload shutdown, has a separate
**0.429479 GPU-hour upper bound**; it is
not part of the measured subtotal or a complete cluster bill. Known cumulative
campaign components total **57.562638 GPU-hours**.

All 14 behavioral checks passed on the exact execution source. The final CPU-only
audit verifies unchanged source/data/world identities and retrieval-bank tensors,
finite best/final checkpoints, 401 unique training metric records, four validation
records, case pairing, causal history/reuse, score reconstruction within float32
rounding tolerance and the 480-transition/optimization limits. W&B finished
syncing, all owned workers exited, and all eight GPUs were empty/idle at the
2026-10-09T23:50:16.141415+00:00 health check. This campaign has no further runs queued.

Frozen execution source: `8be2808eaa09e539d30d07ee3760fc7b59717793`.
Selected checkpoint: `/home/mtxu/adam/LeFlow-experiments/20261009-execution-revision/campaign/runs/execution_revision_3072/best.pt`.
SHA256: `74a3f6db1d59d79f2e3a45ca5086bb18f1f6a0f9d6c6dc865b88c84187af7e2c`.

[Paired review](reports/20261009-execution-revision/validation-review.json),
[all-round diagnostics](reports/20261009-execution-revision/extra-diagnostics.json),
[raw compressed records](reports/20261009-execution-revision/run/),
[final audit](reports/20261009-execution-revision/postrun-audit.json),
[compute ledger](reports/20261009-execution-revision/compute-accounting.json),
[shutdown evidence](reports/20261009-execution-revision/final-health.json), and
[W&B](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/tyytbnn1).
