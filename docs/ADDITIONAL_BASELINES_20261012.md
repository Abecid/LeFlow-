# Authorized GPU batch now launched

The user has explicitly authorized actual training/evaluation on all eight
server GPUs. The [registered batch record](ADDITIONAL_BASELINE_RUNS_20261012.md)
supersedes the historical implementation-only/one-method-at-a-time status below.
Two four-GPU lanes run the five methods under the unchanged single-seed budget.
The optimized dataset uses the exact AlignedSegments sampling/state/action
fields but omits unused TTT-history tensors; real TRAIN byte checks precede training.

Historical implementation specification follows.

# Five additional MetaWorld controller baselines (implementation, not scores)

**State:** code added to main; no GPU training, 104-case validation, sealed-test
access or W&B run has been performed or claimed by this change. CPU synthetic
tests are attached. Source is NOT a bit-for-bit SAGE or Feedback-WM robotics
reproduction. Save every result, including negative outcomes.

## Fixed comparison protocol

Each candidate uses the existing `flow_jepa.execution.train` budgeted runner,
`scripts.operations.controller_grounded` isolated coordinator and
`flow_jepa.evaluate.evaluate` physical simulation loop. The baseline configs
inherit the exact existing seed 3072, global batch 64, 20,000-update OR 28,800
optimizer GPU-second cap, frozen V-JEPA2.1 ViT-L encoder, frozen shared
spatial world checkpoint SHA256, recorded 6,222 successful training expert
episodes, and the 104 identical fixed *development* cases (13 tasks × 8 resets).
Each validation episode is limited to 200 primitive actions AND 10 cumulative
controller seconds (including encoding/inference). The standard four evaluations
are triggered at progress milestones, select the best development checkpoint,
and retain per-case records. The sealed 3,200-case final benchmark is not enabled.
Equal **caps** do not ensure equal optimizer FLOPs, useful training updates,
number of imagined world transitions, or inference-time occupancy.

For new policies the training input is the **same** `AlignedSegments` sampler
(except SAGE, which additionally reads one factual intermediate goal state from
the *same* successful trajectory). Output is always [B,5,8] normalized action
chunks: 5 blocks × 2 four-dimensional simulator commands; inference executes
only the first block [2,4] then replans.

## Registered method identities

| Method | Train / model | Deployment algorithm | Conformity |
| --- | --- | --- | --- |
| `vanilla_action_flow` | Conditional rectified flow on expert [5,8], vision tokens current+final goal only | 8 Euler steps, one 40-D chunk, execute first [8], **no world search or retrieval** | Controlled BC baseline, no TTT/workspace |
| `action_flow_search` | Same flow architecture, additionally conditioned on factual local subgoal | Seven retrieved demonstrations + goal; 8×4 candidates ×3 independent batches; frozen world scores prefix/terminal; 480 world transitions; no workspace revision/calibration/repair | Fixed-world search, NOT an exact no-workspace ablation of the stronger trained reasoner |
| `sage_port` | Factual duration-conditioned intermediate V-JEPA subgoal supervised by MSE+cosine; 8-mode GMM on *predicted* subgoal | Eight candidate subgoal durations (5,5,10,10,20,20,40,60); 4 GMM proposals ×3 CEM rounds, 480 world transitions, first two actions execute | Paper-inspired adapted SAGE, not native trained author checkpoints |
| `diffusion_policy_port` | DDPM epsilon-prediction on [B,5,8] action sequence with temporal conditional 1D U-Net; 100-step cosine schedule | 16-step deterministic DDIM, 2-action receding execution, no model-based candidate ranking | Image-goal / V-JEPA adaptation of Diffusion Policy, no original visual encoder/proprio input |
| `feedback_wm_port` | **Exactly the same diffusion network and DDPM loss** as above | Last four DDIM denoising steps use corrected one-block world prediction, 4 counterfactual action probes to weight action-sensitive visual features, and frozen-world gradient guidance; actual prior prefix residual updates once observation arrives | Independent Feedback-WM-inspired port, not official robotics implementation |

One important design principle: neither SAGE nor Feedback-WM is given
MetaWorld's privileged state, task ID, or expert/test-time counterfactual
outcomes. Their learned components train from the same episode inventory.

The SAGE subgoal generator sees duration choices only from the predefined set,
rather than oracle remaining-to-goal duration. It trains against an actually
recorded future visual feature state; the GMM learns actual expert action chunks
conditioned on the generated subgoal. The subgoal predictor here is a deterministic
spatial residual Transformer port because the exact published robot generator
has not been ported. Its scores MUST be marked `sage_port`.

The Feedback-WM method paper uses an observer-style feedback state and
action-controllable guidance. This port uses EMA of the factual preceding
prediction residual and four local action probes to approximate controllability.
Its diffusion policy can load a matching `diffusion_policy_port` model state
strictly since they have the same parameter layout. For a clean guidance-only
comparison, initialize feedback from the saved diffusion model instead of
training two independent stochastic initializations; preserve that provenance.
Do not claim theoretical convergence guarantees for this independently chosen
feedback law.

## Reproduce on the established A800 MetaWorld GPU server

Run one method at a time from a **clean GitHub main checkout** on the GPU server.
Do **not** run the launcher from a Mac or from this chat's CPU container.
The script clones the exact fetched Git revision into an isolated record
directory, verifies the same registered data hashes, runs CPU tests, a bounded
train-only GPU preflight, and (only if it passes) trains plus evaluates at four
milestones. It does not silently reuse weights or change frozen earlier runs.

```bash
cd /home/mtxu/adam/LeFlow-  # or wherever the clean Git checkout is on your server
git pull --ff-only origin main
export LEFLOW_RUNTIME=/tmp/mtxu-flow-jepa-20261007/env
export LEFLOW_ROUTE_BANK=/tmp/mtxu-flow-reasoning-20261010/route_bank.pt
PREVIOUS=/home/mtxu/adam/LeFlow-experiments/20261009-repaired-comparison

for method in vanilla_action_flow action_flow_search sage_port diffusion_policy_port feedback_wm_port; do
    bash scripts/operations/run_additional_baseline.sh "$method" \
      "/home/mtxu/adam/LeFlow-experiments/additional-$method" \
      "$PREVIOUS" "/tmp/mtxu-additional-$method" 4
done
```

This performs **five sequential** runs; choose a fresh record/cache for each
baseline. The script inherits the coordinator's 95/120 ms
preflight gate and does not relax the 10s episode budget. Verify performance on
train-only examples before committing to 20k training steps.

The existing coordinator copies the frozen data via verified manifest
adoption. If the prior frozen world path, route bank or environment differs,
it fails rather than silently rebuilding a different experimental comparison.

### Test commands

```bash
python -m unittest discover -s tests -p "test_additional_action_flow.py" -v
python -m unittest discover -s tests -p "test_published_policy_ports.py" -v
python -m unittest discover -s tests -p "test_feedback_wm_port.py" -v
```

The tests check pure PyTorch shape, finite losses/gradients, fixed-search
world-call accounting, and FWM's factual-only update. They do NOT establish
trainable MetaWorld performance or reproduce the authors' checkpoints.

## Literature and responsible naming

- SAGE, Cheng et al., 2026, https://arxiv.org/abs/2607.17973
- Feedback World Model, An et al., 2026, https://arxiv.org/abs/2605.15705
  (public repository currently a project page, not executable robot code).
- Diffusion Policy, Chi et al., RSS 2023, https://diffusion-policy.cs.columbia.edu/
  and https://github.com/real-stanford/diffusion_policy
- Our comparison protocol and frozen references: `docs/RELEASE_BASELINES.md`,
  `docs/RELEASE_BASELINE_RESULTS.md`, `docs/RESULTS_INDEX.md`.

These additions preserve prior result ledgers and do not backfill success
numbers. New benchmark claims must await actual training and case-level rollout
records on the same frozen environment.
