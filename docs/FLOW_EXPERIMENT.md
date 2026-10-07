# Joint flow planning on V-JEPA 2.1 / MetaWorld

This is the active experiment. BTM is not part of its model, training, or campaign.
Historical BTM files remain for reproducibility; do not launch `run_comparison.py`
for this experiment.

## Fixed scientific target

Primary outcome: closed-loop MetaWorld v3 success within 200 primitive actions,
averaged equally over 16 tasks. Report the 13 training-task and 3 task-held-out
groups separately. The task list follows *The Planning Limits of Latent World
Models* (arXiv:2609.39235), but this code is an independent implementation, not
the authors' released experiment or a claim to reproduce their scores.

The official V-JEPA 2.1 ViT-L encoder is frozen. Causal 16-frame image histories
are encoded and pooled to a 2 x 4 x 4 spatial-temporal token grid: [32,1024].
Training-only feature statistics standardize every method's inputs and targets.
The downstream dynamics model uses one such history embedding as its state.
This differs from the paper's four-embedding history predictor; comparisons in
this campaign share the same implementation. No simulator state enters a learned
planner; privileged state is used only by the scripted data-collection expert.

The joint planner predicts intermediate visual embeddings and action chunks.
For M=12, K=5 control steps, and two primitive actions per control step, outputs
are subgoals [B,11,32,1024] and actions [B,12,5,8]. Endpoints are observations.
Generated-plan consistency compares each chunk's frozen-world-model rollout to
its next proposed subgoal; both endpoint and action gradients reach the planner.
This regularizer is not a certificate of simulator reachability.

## Data and evaluation rules

- 600 independent training resets per training task (80% expert / 20% random).
  World dynamics sees both collection modes. Planners use successful expert
  episodes with hindsight goals. No training episodes from the three held-out
  tasks. The success rate of the collecting expert is logged, not assumed.
- 50 validation resets per training task, separate from all training resets.
  Intermediate evaluation uses a fixed 32 per task. Only validation outcomes
  select checkpoints or tune hyperparameters.
- 200 **distinct test resets per task**, including all expert failures. A goal
  image is the last successful expert frame, or final expert frame if the expert
  never succeeds. Failed expert goal construction is flagged and reported;
  episodes are never replaced after seeing method outcomes.
- Three training seeds (3072/3073/3074). All methods share reset IDs and goal
  images. 3,200 test episodes/model, 9,600 executions/method over three seeds;
  repeated model-seed evaluations are **not** 9,600 independent environments.
- Report environment success, first-success time, episode return, full-controller
  mean/p95 latency, world-model calls, and predicted/observed subgoal distance.
  Goal images are allowed task specifications; expert subgoals are never exposed.
- The primary comparison is joint_flow_consistent versus leflow_adapted. Report
  all comparisons even if the method loses. Confidence intervals resample training
  seeds and paired resets within tasks; correct secondary comparisons for multiple
  comparisons. With 200 resets, a per-task success CI can still be about +/-6.9 pp
  near 50%; do not infer small per-task improvements from point estimates.

## Comparators and attribution

1. CEM over the same JEPA dynamics, horizons 5 and 30.
2. LeFlow-style flow paths plus a separate inverse model and rollout selection
   (arXiv:2608.24855), adapted to the shared V-JEPA token representation.
3. HWM-style learned macro-actions and coarse/fine CEM (arXiv:2604.03208), adapted
   to the shared representation and data.
4. A matched 2 x 2: deterministic / flow, each with / without generated consistency.

The LeFlow/HWM ports are method-family comparisons, **not exact reproductions of
published SOTA checkpoints**. Published results use different data and protocols
and must not be inserted into this measured comparison table. The campaign logs
parameter counts, training updates, controller settings and actual latency.
Equal candidate counts alone do not establish equal compute. A separate
validation-calibrated latency-matched comparison is required for efficiency claims.

## Execution status

SSH from the development runtime currently fails because `target_server_2` is
not configured there. No GPU training or online evaluation has started. The code
and launch path are being prepared and incrementally pushed to GitHub.
