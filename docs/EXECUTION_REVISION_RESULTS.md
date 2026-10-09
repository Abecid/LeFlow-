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
