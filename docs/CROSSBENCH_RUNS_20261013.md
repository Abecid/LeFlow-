# Sixteen authorized PushT/Cube training and evaluation runs

The user requested actual execution of the implemented eight-controller matrix.
Completed MetaWorld evidence is reused. The new controllers are LeFlow, vanilla
action flow, execution-revision GMM, fixed-world action-flow search, SAGE port,
Diffusion Policy port, Feedback-WM port and retrieval-repair TTT on each of
PushT and Cube. No ninth duplicate flow baseline is added.

All eight A80080GB GPUs were idle. Two four-GPU lanes take prepared jobs from
the same registered matrix. PushT runs may fill both lanes while Cube downloads;
when Cube data is ready, one lane prepares its cache on four GPUs, with bounded
float32 batch512 encoding and ordered writes. Every new policy starts fresh at
seed3072/global64 with the20k-update OR8 optimizer-GPUh cap. Native50-case
validation runs at four progress milestones, with50 primitives AND10 controller
seconds. These are screening/development comparisons, not a sealed final study.

## Real-data corrections before formal training

The old PushT source and latent cache are available at
`/tmp/mtxu-leflow-20261004/data`. It has18685 source episodes and2336736 frames;
source splits are14948 TRAIN,1868 validation and1869 reserved test episodes.
Its source actions contain2318 out-of-box outgoing TRAIN commands among1855322
(about0.125%). The pinned PushT `step` does not clip these. Blind clipping would
attach the recorded outcome to a different action, so raw commands and all cached
features are retained. Protocol`crossbench-shared-lewm-v2` selects only native-safe
factual training prefixes/history and goal spans; retrieval uses the same bounded
factual routes. Validation case generation and the declared action limits are
unchanged. Eligible episode/window counts are recorded in each run identity.
This is a shared data-eligibility correction, not a claimed lossless sampling change.

The compact cache is read into RAM without changing any cached value, sampled
field, dtype or model loss. Byte-equality checks cover both action dimensions.
The server lacks a hardware EGL display; its previously verified Mesa/llvmpipe
stack renders both native environments with two CPU rendering threads per rank.
GPU encoding/planning remains on the assigned rank. The isolated runtime inherits
the pinned working packages and installs the exact SWM commit with VCS metadata;
it does not modify the older experiment environments. Native reset/render/action/
self-goal-success and all18 CPU/native tests pass.

Cube's official compressed archive is 46,184,624,478 bytes. The interrupted
8,021,606,400-byte prefix was preserved and recovery used four parallel HTTP
range workers. The complete archive matches the official SHA256
`3725d6a01abd492164441ef0a27e588f52b94a118fab56b96987b1a34a6c2600`.
Extraction and source metadata are complete: 10,000 episodes and 2,010,000 frames,
with 8,000 TRAIN, 1,000 validation and 1,000 reserved test episodes. One four-GPU
lane now prepares the TRAIN/validation cache while the other trains PushT.
No reserved test images, actions or outcomes are read by the policy cache.

The persistent GPU queue is PID3160064/start1020541048. Formal training stays
frozen at `747f635ad36af2cbe3e2bbb0b7997efe9caccaa1` (245 runtime files).
A separate CPU publication watcher audits completed processes/checkpoints and
preserves all four physical evaluations, periodic logs, failures and selected
measured trajectory archives. It publishes from its own checkout and verifies
remote bytes. It does not launch GPU work, edit frozen source or add experiments.
The exact [operations scripts](reports/20261013-crossbench-baselines/operations/)
are preserved. Raw trajectory archives have not been turned into visually
reviewed figures; their presence is not a visual-inspection claim.

Server record: `/home/mtxu/adam/LeFlow-experiments/20261013-crossbench-baselines`.
Runtime: `/tmp/mtxu-crossbench-20261013/env/bin/python`.
Exact eight-method specifications: [cross-benchmark runbook](CROSSBENCH_BASELINES.md).
Current launch/check/evidence records: [batch archive](reports/20261013-crossbench-baselines/).
Source is frozen immediately before formal launch after the server checks.
Native50 results and MetaWorld104 results are reported separately; backbones and
native success predicates differ. SAGE/Feedback-WM retain paper-port labels.
