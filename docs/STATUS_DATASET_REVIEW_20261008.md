# Live status and evaluation interpretation

Observed October 8, 2026 at **10:28 PM America/Los_Angeles**
(October 9, 05:28:05 UTC). This is a read-only status review; no experiment,
configuration, checkpoint or queue was changed.

## Active run, queue and ETA

The compatible static-state world is at **16,933 / 20,000 updates**, with three
registered validations completed. Its charged optimization is 2,269.677 seconds
(2.521863 GPU-hours) and completed validation is 1,262.401 seconds (1.402667
GPU-hours). Four A800 80GB GPUs, indices 0–3, are occupied; 4–7 are idle.
Ours-only continuation 807392 is waiting for world completion. Cancelled parent
778158 remains stopped, and cancelled coordinator 795290 is absent. The only
run directory is world_3072; no repaired planner or repeated baseline has begun.

Next is exactly one fresh joint_flow_consistent head, seed 3072. LeFlow and HWM
are frozen; CEM has no learned head. Existing CEM world diagnostics continue.
There are no additional seeds, ablations, variants or final tests queued. After
our four validations, save selected-validation comparisons/failure analysis
and stop for review.

At the observed speed, world completion including its last validation is
estimated around **10:40–10:50 PM October 8**. Our head then needs roughly
**2–3 hours including evaluation**: first head results approximately
11:15–11:40 PM, full review roughly **12:45–1:45 AM October 9**. These are
estimates, not deadlines. The old head took 91.6 minutes optimization plus
35.0 minutes validation; the repaired consistency path is more expensive.
All learned models retain the original ceiling: 7,200 optimization seconds OR
20,000 updates, whichever comes first, on four GPUs, global batch64/microbatch4.
Validation is separately metered. Equal ceilings do not mean identical actual
GPU time or FLOPs, and cumulative development costs remain separately recorded.

## Exact dataset

Custom **MetaWorld v3 image-goal manipulation** protocol with frozen V-JEPA2.1
ViT-L features. It is not the standard MT10/MT50 or ML benchmark protocol.
The same original selected episode entries and goals are reused.

| Partition/use | Exact size |
|---|---:|
| World training | 7,800 episodes: 600 for each of 13 tasks |
| Training collection composition | 6,240 scripted expert + 1,560 random |
| Planner training eligibility | 6,222 successful expert episodes |
| Validation pool | 650 episodes: 50 for each of 13 training tasks |
| Each periodic closed-loop validation | 104 fixed cases: 8 per task |
| World prediction diagnostic | 512 fixed sampled trajectory windows from validation |
| Reserved final test | 3,200 reset/goal cases: 200 for each of 16 tasks |

Each training episode has 100 control steps with action repeat2: 200 primitive
actions. Thus world training contains **780,000 control transitions / 1,560,000
primitive actions**. All 6,222 planner-eligible episodes also have 100 control
steps. Only door-open loses expert episodes to the success filter: 462 instead
of480; all other 12 tasks retain480. Planner windows span60 control steps
(12 segments of5); sampling balances tasks. Test entries store initial/goal
annotations and have zero trajectory steps in the manifest; they are 3,200
online evaluation cases, not 3,200 training trajectories.

The 13 training/validation tasks are assembly, button-press-topdown,
coffee-button, dial-turn, door-close, door-open, drawer-close, drawer-open,
faucet-open, handle-press, pick-place, plate-slide and reach. Final testing adds
window-open, handle-pull and push: 2,600 cases on seen tasks and600 on unseen
tasks. No held-out task is used for model selection. Evaluation goals were
screened for scripted-expert constructibility before training; selected-case
success is not an unconditional MetaWorld reset-distribution estimate.

## Results available now

All numbers below are on the same fixed104 validation cases. Historical learned
heads were selected by validation success; current CEM numbers are interim world
diagnostics. The new world is selected by prediction loss, not CEM success.

| Method/reference | Successes | Success rate |
|---|---:|---:|
| Historical LeFlow adaptation | 27/104 | 25.96% |
| Historical HWM adaptation | 8/104 | 7.69% |
| Historical CEM / old selected world | 7/104 | 6.73% |
| Original proposed method | 23/104 | 22.12% |
| CEM / new world at5k | 23/104 | 22.12% |
| CEM / new world at10k | 24/104 | 23.08% |
| CEM / new world at15k | 24/104 | 23.08% |
| Repaired proposed planner | Not started | No result |

At15k, world prediction loss is0.0100957 versus persistence0.0281005. Action
identification among16 candidates is87.70%, chance6.25%. These diagnose
short-horizon prediction/action dependence; they do not establish successful
planning. Cross-representation latent losses are not directly comparable.

## Metric for the research claim

The primary metric is **task-balanced closed-loop environment success at the
fixed training and inference allowances**, on the reserved final test. Each
case has at most200 primitive actions and10 seconds cumulative controller
compute, including encoding/preprocessing/planning/action preparation; simulator
rendering and physics are excluded. Actions finished after the deadline are
not executed. Timeouts stay in the denominator as failures.

Report overall success, seen-task and held-out-task success, every task, and
paired percentage-point differences with uncertainty. Use identical reset/goal
cases. Paired reset confidence intervals do not estimate training-seed
variability: this is intentionally a one-seed study. The104 repeatedly reused
validation cases select models and guide repairs, so selected validation scores
are development evidence rather than an unbiased final estimate.

MetaWorld's [official evaluation documentation](https://github.com/Farama-Foundation/Metaworld/blob/main/docs/evaluation/evaluation.md)
uses environment success as the benchmark endpoint. The importance of retaining
uncertainty in few-run RL comparisons is discussed by
[Agarwal et al., NeurIPS2021](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html).
Our exact action/time/reset protocol remains custom and must be disclosed.

Other logged diagnostics: success within50/100/200 actions, first-success steps,
capped steps, return, mean/p95 full-controller latency, controller timeouts and
clock overruns, world-model calls, predicted plan cost and observed subgoal
distance, plus world prediction/persistence loss and action identification.
Training metrics are logged every50 updates; four evaluations occur at25/50/
75/100% of the update/time ceiling. A publication about executable subgoals also
needs evidence that scores predict observed progress; low training or consistency
loss alone cannot support that claim.

## Fairness and failure findings

The episode IDs, reset seeds, goal hashes, model seed, data split, budget ceilings
and evaluation limits match. This review verified all three new-world reports
against the saved historical LeFlow104-case identity signature; all matched.
Losses are finite and training steps/charged time are monotone.

However, the historical-reference comparison changes world and state
representation, and historical LeFlow has the confirmed restricted-noise sampler
defect. LeFlow/HWM are adaptations rather than exact published SOTA replicas.
This can assess full-pipeline development progress but cannot isolate our
planner's contribution or establish superiority over a faithful corrected SOTA
baseline. Include the already-recorded CEM diagnostic for the selected new world
alongside the historical references; this needs no extra GPU work. No baseline
reruns are authorized or launched by this review.

At15k, coffee-button and handle-press each succeed8/8, drawer-close6/8 and reach2/8;
the other nine tasks are0/8. All80 failures exhaust controller compute after
68–70 primitive actions, before the200-action cap. Mean controller call time is
280.7ms, p95 289.7ms. This establishes a compute-limited rollout bottleneck, not
whether the underlying physical cause is bad grasping, collision or inaccurate
subgoals; those causes need trajectory evidence. Do not relax the budget during
this comparison.

World prediction loss improves8.97% from10k to15k while CEM success stays24/104.
One coffee-button reset improves and one drawer-close reset regresses. This is
another direct warning that better latent prediction need not yield better
control. New-world CEM already improving from old-world7/104 to24/104 also makes
the shared-world confound substantial. We still have no repaired-planner result
or evidence that its repaired ranking predicts executable long-horizon progress.

[Raw timestamped evidence](reports/20261009-repaired-comparison/status-dataset-review-20261009T052805Z.json)
contains the manifest counts, process scope, training ledger and complete three
validation reports. [Live world metrics](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/8vwksovs)
continue online. The old partial final-test outcomes were not read for this review.
