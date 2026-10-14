# Registered noise-space CEM transfer

Run the six explicitly requested variants on the retained frozen PushT/Cube
LeFlow heads. There are12 new inference-only jobs and600 development cases,
not a new training campaign. Baselines are retained; no optimizer updates.

Each method uses seed3072, the exact original50 validation case IDs and resets,
the task-specific retained LeFlow checkpoint/world, the existing action units,
H=5 latent transitions, five-primitive executed prefixes,50 total primitives and
10 cumulative controller seconds. All use strict FP32, NVIDIA_TF32_OVERRIDE=0.
The completed strict-FP32 reference is12/50 PushT and40/50 Cube; it is reused.

Methods: noise_cem, noise_cem_shell, noise_random, noise_random_shell,
noise_bon192, noise_bon192_shell. Each scores192 candidates/decision at16 flow
steps (3072 candidate flow evaluations;960 world-transition predictions).
CEM uses3 rounds of64 and8 elites, old-moment weight0.25, standard-deviation
floor0.1. Shell radius is sqrt((H-1)*192). Defaults are fixed before evaluation.
No gradients, adaptation, invented outcome labels, action averaging or final
rescoring. The incumbent is the best evaluated candidate across completed rounds.
Actual latency is measured; equal operation counts are not equal wall time.

Use all8 GPUs, one job per GPU, with fixed immutable source and900 operational
seconds/job. CPU numerical/adapter checks and real-TRAIN-checkpoint gates must
pass before simulator evaluation. Retain factual selected-prefix prediction
errors without another world rollout; observations never affect the frozen
search. Preserve per-case success, timing, work counts, round margins/noise norms,
reset audits, full reports, W&B links and trajectory archives. Do not replay
consumed jobs after a failure. Reacher/TwoRoom and original author-checkpoint
results remain pending because their original planner weights are unavailable.

Server: /home/mtxu/adam/LeFlow-experiments/20261014-noise-cem.
Weights/cache: /home/mtxu/adam/LeFlow-experiments/20261013-crossbench-baselines.
Config: [noise_comparison_20261014.json](../config/leflow_refinement/noise_comparison_20261014.json).
This transfer cannot establish native-author SOTA, held-out generalization or
statistical superiority: it is one seed on previously reused development cases.
