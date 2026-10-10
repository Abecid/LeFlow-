# Progress TTT: one fresh training run and periodic evaluation

October 10, 2026. The user explicitly directs immediate training and evaluation.
This supersedes the implementation-only hold. The method and its 29 passing
behavior tests are documented in [PROGRESS_TTT.md](PROGRESS_TTT.md).

- One fresh `progress_ttt` policy, seed 3072. No inherited policy checkpoint.
- All eight available A800 GPUs; global batch 64. Same 6,222 successful expert
  episodes, window sampling, frozen V-JEPA encoder, frozen world and bank tensors.
- Maximum 20,000 optimizer updates OR 28,800 aggregate optimization GPU-seconds.
  All new learned modules and adaptation meta-training count within this budget.
- Evaluate the identical 104 development cases at 5k/10k/15k/20k. If the compute
  cap binds earlier, evaluate the last checkpoint without extending optimization.
- Task success within 200 primitive actions and 10 cumulative controller seconds
  is primary. All within-episode adaptation counts on the controller clock.
- Select highest development success, earliest checkpoint for ties, unchanged.
  Compare preserved execution-revision80/104, flow-reasoning79/104 and frozen
  corrected LeFlow/HWM/CEM records. No baseline is retrained.
- No new seed, automatic next variant, ablation, scaling sweep or sealed final test.

Record root: `/home/mtxu/adam/LeFlow-experiments/20261010-progress-ttt`.
Cache: `/tmp/mtxu-progress-ttt-20261010`.
Launcher: `scripts/operations/launch_progress_ttt.sh`.
Coordinator runs detached and includes all four evaluations with online W&B logs.
Source is frozen at the deployment commit and runtime hashes are verified before
launch. Prior running/completed source trees and artifacts remain untouched.

Report success by checkpoint and task, controller timeouts/runtime, raw/prior/
adapted executed-prefix error, fast-weight magnitude, proposal dispersion and
selected reasoning round. A lower calibration error is not evidence of improved
control. Analyze the first complete result before any further method changes.
