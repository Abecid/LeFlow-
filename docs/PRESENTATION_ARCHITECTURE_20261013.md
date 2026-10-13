# Baseline architecture teaching brief

Prepared 2026-10-13. Source audit: official papers and released code; the current Abecid/LeFlow- implementation in `audit-snapshot`. Seven candidate slides, intended to precede all success-rate charts. The visible slide copy is separated from presenter notes so the deck can teach one relationship at a time.

## Direct answer: is our baseline LeFlow or V-JEPA?

**Our current MetaWorld platform is a frozen V-JEPA 2.1 ViT-L visual encoder plus our own trained, then frozen, action-conditioned spatial world model. LeFlow is a planning reference and a separately evaluated port. Our configured primary experimental control is the execution-revision GMM controller.** These answer three different questions: representation backbone, planning literature baseline, and matched experimental comparator.

The current code does **not** load Meta's pretrained V-JEPA 2-AC robot predictor or native LeWorldModel dynamics. The older PushT pilot used LeWM; that history must not be silently carried into the current MetaWorld architecture.

**Bibliographic correction:** arXiv **2603.19312v1** is **LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels**, submitted March 13, 2026. **LeFlow: Generative Latent Flow Planning for World Models** is **2608.24855**, submitted August 25, 2026. The LeFlow README cites both, which can easily cause confusion. [P1, P3, C1]

---

## Candidate slide 1 — Three jobs: represent the scene, predict consequences, choose actions

### Visible explanatory body

The robot receives a **current image** and a **goal image**. An encoder turns each into a numerical description of the scene. A world model predicts how that description changes under a proposed action. A controller generates several action plans, compares their predicted consequences with the goal, and chooses what to execute.

In our current system, **V-JEPA 2.1 supplies the representation; our trained dynamics supply the consequences; a GMM or flow-based controller supplies the candidate actions.** LeFlow is one particular design for that third part, coupled to a world model.

### Native visual / shape labels

Use three editable module boxes and a feedback loop, not a paper-name-only diagram:

- **Encoder E — frozen:** RGB current/goal images → `Zcurrent, Zgoal: [B, 32, 1024]`.
- **World F — trained once, then frozen:** `Zcurrent + candidate A[B,5,8]` → `Zpred[B,5,32,1024]`.
- **Controller — the comparison changes this:** generate candidates → score predicted outcomes → execute first block `A[:,0]: [B,8] = [B,2,4]` → new RGB image.

### Presenter notes

A latent is a learned vector, not a ground-truth simulator state or a guaranteed object pose. An image encoder alone cannot answer “what happens if I choose this action?” The dynamics and the action-selection procedure must be named separately. Our four action coordinates are the MetaWorld simulator action interface; the eight-dimensional block groups two consecutive four-dimensional actions, not eight independent robot joints.

**Sources:** current encoder [C4]; dynamics [C5]; controller configuration including `primary_baseline: execution_revision` [C6].

---

## Candidate slide 2 — What V-JEPA 2 and 2.1 contribute

### Visible explanatory body

**V-JEPA 2 learns visual representations from video before robot control is introduced.** It hides video patches, encodes the visible patches, and trains a predictor to recover the hidden patches' target embeddings. It predicts features rather than reconstructing RGB pixels.

**V-JEPA 2.1 strengthens spatially detailed features:** it also supervises visible context tokens, supervises multiple encoder depths, and supports separate image/video tokenizers in one encoder. Larger teachers are distilled into smaller backbones, including the ViT-L we use.

Robot control is a separate stage: **V-JEPA 2-AC** trains an action-conditioned predictor on a frozen encoder. Our system follows this broad separation, but implements its own predictor.

### Native visual / shape labels

Two rows are easier than a dense architecture diagram:

| Stage | Input and output | What is learned? |
|---|---|---|
| Visual pretraining | Visible patch embeddings + positions → target patch embeddings | Encoder and pretraining predictor; an EMA target encoder supplies stopped-gradient targets |
| Action-conditioned post-training | Image features + candidate actions + robot state → future image features | A different predictor, with the encoder frozen |

Small concrete comparison: original V-JEPA 2-AC uses per-image **`16×16×1408`** ViT-g features, **7-D** action/state signals, and a **24-layer, ~300M** predictor. Our spatial state and predictor are different; do not transplant those dimensions into our diagram.

### Presenter notes

V-JEPA 2 pretraining is action-free; its robot post-training uses under 62 hours of DROID videos with end-effector signals, without task/reward/success labels. The AC predictor uses frame-causal attention, teacher-forcing plus a two-step rollout loss, and CEM action search. V-JEPA 2.1's manipulation experiment reuses the AC approach with a new visual encoder. Neither pretraining mask tokens nor an AC predictor are a flow sampler.

Do not call V-JEPA 2.1 “V-JEPA 2 with a diffusion head.” Its paper separately studies a navigation-specific Conditional Diffusion Transformer with clean-latent prediction and DDIM; that is a downstream experiment, not the encoder we instantiate.

**Sources:** V-JEPA 2 §§2–3 [P2]; V-JEPA 2.1 §§2.3, 3.3–3.4 [P4]; official model factory [C2].

---

## Candidate slide 3 — The exact encoder and world model in our MetaWorld runs

### Visible explanatory body

We load the official **V-JEPA 2.1 ViT-L/16, distilled from ViT-G**, and keep it frozen. We repeat each current or goal image 16 times, run the video encoder, and pool its features into 32 tokens. Thus the current visual input contains **one observed moment**, not a 16-frame motion history.

Our smaller dynamics transformer then predicts the next 32-token state from the current tokens and a two-action block. Applying it five times produces the imagined consequences of a ten-action plan. It predicts latent features directly; there is no pixel decoder in this planning loop.

### Native visual / shape labels

Show the following transformation with explicit intermediate dimensions:

1. RGB image: `[B,256,256,3]`.
2. Repeat and arrange channels: `[B,3,16,256,256]`.
3. Patch/tubelet encoder: `2×16×16` tubelets → `[B,8×16×16,1024] = [B,2048,1024]`.
4. Pool over `(time,height,width)` to `(2,4,4)` → **`Z[B,32,1024]`**.
5. Dynamics: `Z[B,32,1024] + a[B,8]` → **`Znext[B,32,1024]`**.

Inside F: feature projection `1024→256`; action projection `8→256` broadcast over tokens; **4 transformer layers / 4 heads**; projection `256→1024` added as a residual to Z. Five recurrent calls yield `[B,5,32,1024]`.

### Presenter notes

Exact checkpoint: `https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt`, key `ema_encoder`. The checkpoint is named for a 384-pixel model; our configured runtime input is **256 pixels**, using the encoder's interpolated RoPE support. Only the encoder's final-layer 1024-dimensional tensor is loaded, not the official pretraining predictor or teacher-sized projected representation. The `(2,4,4)` grid is technically spatiotemporal; the repeated-frame convention does not supply true motion observations.

Our world-training implementation minimizes **feature MSE + mean tokenwise cosine distance** on autoregressive recorded-action rollouts. The world is hash-checked and frozen for the controller comparisons. This is a custom action-conditioned model, not an exact reproduction of Meta's AC architecture or loss.

**Sources:** `vision.py` lines 46–78, 80–117, 124–129 [C4]; `models.py` lines 48–79, 181–218 [C5]; frozen-world loading [C7]; checkpoint/config [C2, C6].

---

## Candidate slide 4 — LeWorldModel and LeFlow solve different parts of the stack

### Visible explanatory body

**LeWorldModel (LeWM)** learns a compact visual representation and action-conditioned dynamics together from pixels. Its small model trains with a next-embedding prediction term and a Gaussian-distribution regularizer. Its original controller searches for actions with CEM.

**LeFlow freezes that learned world and learns a reusable planner on top.** Its flow model generates possible intermediate latent states between the current and goal embeddings. An inverse-dynamics network translates each neighboring latent pair into an action block. The frozen world predicts what those actions would actually do in the model.

### Native visual / shape labels

Use two nested stages:

- **LeWM, learned first:** image → global latent `z[B,D]`; latent/action history → next latent.
- **LeFlow, learned second:** fixed `z0,zgoal` → flow-generated interior `[B,H−1,D]` → full path `[B,H+1,D]` → inverse dynamics → actions `[B,H,K·da]`.

For the native main configuration, **H=5** and **K=5 primitive actions per block**. The LeWM training config sets **D=192**; the loader obtains actual dimensions from each checkpoint. Keep D symbolic when discussing all checkpoints.

### Presenter notes

LeFlow's learned components are the flow and IDM; rollout ranking is an algorithm, not a separately trained third neural network. Native LeWM is a different backbone from V-JEPA 2.1. Both belong to the JEPA family, but sharing the family name does not make their dimensions, losses, checkpoints, or action interfaces interchangeable.

The current released flow is a four-layer, width-512 transformer with eight heads; the IDM is a three-hidden-layer, width-512 MLP. The dataset keys can include proprioception, but native `encode_current_and_goal` reads pixels/goal, and the native flow conditions on their encoded visual latents. Do not infer that all loaded metadata are model inputs.

**Sources:** LeWM abstract [P1]; LeFlow method [P3]; `LatentPathFlow` and `InverseDynamics` [C3]; native `config/train/lewm.yaml` [C8].

---

## Candidate slide 5 — What LeFlow generates, fixes, and verifies

### Visible explanatory body

For a five-block plan, LeFlow samples **four noisy intermediate latent states**. The start and goal are fixed conditions; they are never part of the noise being evolved. Sixteen Euler updates turn the noise into a proposed path. An IDM maps each pair `[zt, zt+1, zt+1−zt]` into an action block.

Every generated path already ends at the goal by construction. Therefore, LeFlow ranks candidates using the **world-model rollout of decoded actions**, not the generated endpoint. A lower score means the model predicts a closer endpoint; physical task success still has to be observed.

### Native visual / worked tensor example

With one query, 64 candidates and the 192-D native configuration:

| Operation | Shape |
|---|---|
| Gaussian interior | `[64,4,192]` |
| Flow velocity at each internal solver time τ | `[64,4,192]` |
| Append fixed start and goal | `[1,64,6,192]` |
| Decode five transitions | `[1,64,5,K·da]` |
| Roll out decoded actions from the real current embedding | Final predictions `[1,64,192]` |
| Mean squared distance to encoded goal | 64 scores; select the smallest |

**τ is internal generation time. H indexes future physical action blocks. An Euler update does not execute a robot action.**

### Presenter notes / scientific precision

The implementation does not use an attention mask to clamp endpoints: the flow input simply excludes them. Start, goal, and sinusoidal time projections are added to each interior token. No causal mask is passed to this flow transformer, so its interior tokens can attend jointly. This is different from a JEPA pretraining mask and different from the causal dynamics predictor.

Training uses straight-line conditional flow matching on recorded latent interiors, IDM MSE on normalized recorded action blocks, and 0.1× one-step world consistency. **Paper/code discrepancy:** the paper describes consistency on generated transitions; released `step_batch` actually uses recorded `z_path` and IDM predictions. With the frozen world/encoded targets, this term trains the IDM, not the flow. The scalar smoothness diagnostic is also computed on recorded paths, with default weight zero.

Default selection uses 64 candidates, 16 Euler steps, and final-rollout MSE; optional smoothness/consistency ranking weights are zero. The released solver discards `init_action`, so previous-plan reuse is not part of its default mechanism. Calling rollout selection a “projection onto a controllable manifold” is the paper's interpretation; it is not a proof of real-system reachability.

**Sources:** native sampler/IDM/rollout [C3]; released training computation [C9]; default solver config [C10]; paper §§3.2–3.5 [P3].

---

## Candidate slide 6 — Our LeFlow port is a specific adaptation to the shared world

### Visible explanatory body

The LeFlow reference expects one compact vector per image. Our world retains 32 V-JEPA feature tokens. Our port therefore learns a spatial adapter: it compresses the tokens to **512 dimensions**, then freezes the adapter and trains the released flow/IDM modules on that compact representation.

The IDM outputs the action format our shared world understands. We score those actions by rolling them through the **original spatial world** and comparing its predicted endpoint with the full goal tokens. We do not score the compressed path's goal-clamped endpoint or use reconstructed pixels.

### Native visual / shape labels

Use a branch diagram to make the compressed proposal and full-space verification visibly distinct:

- Spatial state `Z[B,32,1024]` → adapter `c[B,512]`.
- Flow candidates `[64,4,512]` → paths `[64,6,512]` → IDM actions `[64,5,8]`.
- The **same actions**, with the original `Z`, enter F → endpoints `[64,32,1024]`.
- MSE to full spatial goal → selected five-block sequence.

The release-port configuration executes **five blocks = ten primitive actions** per completed decision. Our main GMM/action-flow controllers execute **one block = two actions** before observing again.

### Presenter notes / fair comparison boundaries

This borrows LeFlow Appendix E's spatial-adapter idea, but is not its unchanged DINO-WM experiment. Appendix E uses DINOv2 patch features compressed to 512 plus a 10-D proprioceptive embedding, for a 522-D planner state. Ours uses V-JEPA features, a 512-D state, a different dynamics model, no proprioception, a different task/action interface, and different data/budgets.

Our adapter uses positioned tokens and a summary token in a two-layer transformer. Its training-only decoder and its exact reconstruction implementation are ours; the source explicitly says the paper did not release that decoder. The adapter is trained for 2,000 steps before being frozen. The port retains the released recorded-path consistency computation, normalized IDM actions, 64 proposals, and 16 Euler steps.

The original LeFlow paper evaluates TwoRoom, PushT, Reacher and OGBench Cube, with a DINO-WM PushT transfer appendix. Our custom MetaWorld results should be presented as **a controlled port on our platform**, not the authors' reported LeFlow performance. Different executed chunk lengths also change observation frequency and controller-time demands; equal candidate counts are not equal closed-loop protocols.

**Sources:** our adapter/controller [C11]; port configuration [C12]; native paper Appendix E [P3].

---

## Candidate slide 7 — Our main comparison changes the action generator

### Visible explanatory body

**GMM means Gaussian mixture model:** a network predicts four possible action-plan centers, their probabilities, and their spreads. It samples a complete 40-number plan in one pass. The surrounding controller still retrieves local targets, predicts consequences, selects candidates, and observes execution.

**Our action flow learns to move noise directly into that same 40-number action space.** It does not first invent a JEPA state path and then ask an IDM to realize it. The world model produces the imagined state path after the candidate actions have been generated.

### Native visual / shape labels

| Generator | Generated quantity | Conversion to simulator actions |
|---|---|---|
| Native LeFlow / our LeFlow port | Intermediate latent states | Separate IDM |
| Our GMM controllers | Four-component distribution over `[5,8]` plans | Reshape first block to `[2,4]` |
| Our action-flow controllers | Velocity transporting noise to `[5,8]` plans | Reshape first block to `[2,4]` |

For the GMM, the head returns logits `[B,4]`, means `[B,4,40]`, and diagonal scales `[B,4,40]`. Means use tanh; scales are bounded to 0.05–0.5. “Four mixtures” is not the same as “four candidates”: candidates are samples, with the highest-weight component mean retained in the first slot.

### One worked flow-matching example — illustrative, not measured

Take one coordinate of a recorded action plan: target **0.4**, sampled noise **−0.6**. At internal time **τ=0.25**, the training input is `0.75×(−0.6) + 0.25×0.4 = −0.35`; the target velocity is `0.4−(−0.6)=1.0`. The network learns this velocity conditioned on current/target/goal features and controller context. At inference it starts from noise and integrates its learned velocities; only the final bounded plan is submitted to the world model.

### Presenter notes

The simplest GMM head trains by mixture negative log-likelihood on demonstrated plans. Execution-revision variants add their own workspace/calibration/controller changes; do not describe all three GMM rows as identical one-pass policies. In the current retained action-flow family, the generator is a small residual MLP over 40 action coordinates, conditioned by a learned workspace. This workspace is computational memory, not a physically reachable latent scene.

The retained repair version samples initial actions with eight Euler steps, then samples bounded action residuals with four Euler steps, preserving the incumbent. It shares the frozen encoder and world with the GMM control, but differs in more than the generative distribution. The result charts should appear only after these mechanisms and common evaluation rules have been introduced.

**Sources:** `ChunkPolicy` [C13]; `ActionFlow` and flow matching [C14]; retained architecture definition [C15].

---

## Presenter guardrails and unresolved details

1. **Do not call our world “the V-JEPA 2.1 world model checkpoint.”** Only the official encoder weights are imported; F is ours. Its concrete formula and training loss are above.
2. **Do not claim 16 observed frames.** The current static configuration repeats a single image. The encoder class's generic “past-only clips” docstring does not override `state_clip(..., representation="static")`.
3. **Do not infer native LeFlow checkpoint dimensions from the default config alone.** D=192 is the checked configuration, while the loader reads actual dimensions from checkpoint tensors. I did not download or inspect every released binary checkpoint for this brief.
4. **Do not equate model rollout with a physical outcome.** Only executed actions have factual outcomes; generated path endpoints are clamped design choices.
5. **No invented mask or ControlNet mechanism.** The released flow transformer jointly processes interior tokens, conditioned on endpoint projections. Native world-predictor causality, visual pretraining masking, and our factual-history padding masks serve different purposes.
6. **Do not describe the shared world as a universally calibrated success verifier.** The deployed scores are feature distances/corrections. No task-success semantics follows from the representation's name.
7. **The robot's low-level internal actuator implementation is not specified by the neural tensor shapes.** Describe the externally visible MetaWorld action interface unless the environment implementation is separately being taught.

## Primary sources and exact code pointers

### Papers

- **P1 — LeWorldModel:** https://arxiv.org/abs/2603.19312v1 . Title/identity and two-loss learning contribution verified against arXiv.
- **P2 — V-JEPA 2:** https://arxiv.org/html/2506.09985v1 , especially §§2, 3.1, 3.2. Official source: https://github.com/facebookresearch/vjepa2 .
- **P3 — LeFlow:** https://arxiv.org/abs/2608.24855 and https://arxiv.org/html/2608.24855v1 . Full paper read through https://huggingface.co/papers/2608.24855.md using the Hugging Face papers workflow when arXiv HTML fetch failed. Local source copy: `/workspace/scratch/0c9d30bdbf59/research-deck-rebuild/sources/leflow-paper.md`. Official repo independently confirms the arXiv identity.
- **P4 — V-JEPA 2.1:** https://arxiv.org/html/2603.14482v1 , especially §§2.3, 3.3, 3.4 and 3.10.

### Upstream code, pinned to the inspected revision

- **C1 — LeFlow README and citations:** https://github.com/hsiangwei0903/LeFlow/blob/f1fe192e41ec6f20de25cd1054ede40a8cbdcfa5/README.md .
- **C2 — Official V-JEPA model mapping and distilled ViT-L factory:** https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/src/hub/backbones.py#L20-L24 and #L303-L315.
- **C3 — Native LeFlow modules and runtime:** https://github.com/hsiangwei0903/LeFlow/blob/f1fe192e41ec6f20de25cd1054ede40a8cbdcfa5/latent_planner.py . Flow L200–262; IDM L265–293; input encoding L340–353; sampling L356–384; decoding/rollout/ranking L386–478; previous-plan discard L556–558; losses L589–628.
- **C8 — Native LeWM dimensions/config:** https://github.com/hsiangwei0903/LeFlow/blob/f1fe192e41ec6f20de25cd1054ede40a8cbdcfa5/config/train/lewm.yaml .
- **C9 — Released consistency computation:** https://github.com/hsiangwei0903/LeFlow/blob/f1fe192e41ec6f20de25cd1054ede40a8cbdcfa5/train_latent_planner.py#L161-L208 . Training defaults: https://github.com/hsiangwei0903/LeFlow/blob/f1fe192e41ec6f20de25cd1054ede40a8cbdcfa5/config/train/latent_planner.yaml . Note README main-run commands override default epochs to 10.
- **C10 — Native evaluation defaults:** https://github.com/hsiangwei0903/LeFlow/blob/f1fe192e41ec6f20de25cd1054ede40a8cbdcfa5/config/eval/solver/latent_flow.yaml .

### Our code, pinned to the audited snapshot

- **C4 — Encoder/static input:** https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/vision.py#L46-L129 .
- **C5 — Dynamics and losses:** https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/models.py#L48-L79 and #L181-L218.
- **C6 — Current encoder/controller config:** https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/config/flow_metaworld_retrieval_repair_ttt.json . Encoder L39–63; primary comparator L110–111; controller sizes L121–147.
- **C7 — Frozen shared-world loading:** https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/train.py#L76-L84 .
- **C11 — Our LeFlow adapter, training, and controller:** https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/baselines/leflow.py#L14-L129 .
- **C12 — Our LeFlow port configuration:** https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/config/flow_metaworld_leflow_release.json .
- **C13 — GMM distribution and sampling:** https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/model.py#L33-L82 . Basic controller: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/controller.py#L34-L99 . Execution-aware extension: `flow_jepa/execution/aligned.py`, L38–95 at the same commit.
- **C14 — Action-flow implementation:** https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/flow_reasoning.py#L15-L100 and #L148-L163.
- **C15 — Retained repair-flow formulation:** local `docs/PAPER_BLUEPRINT.md`, formulation section; code `flow_jepa/execution/repair_flow_ttt.py` and `retrieval_repair.py` at the audited commit. The local blueprint is a teaching/checking aid; cite the implementation and original experiment artifacts in the final deck notes.

## Verification scope

This was a read-only architecture audit. The actual pretrained encoder binary and every native LeFlow binary were not loaded in this turn; naming, architecture and loading behavior are established from the audited config and source. No experiment, remote Slides mutation, training, or deployment was performed. The scalar flow example is deliberately illustrative and is not a reported model prediction.


## Presentation standard for subsequent reports

Develop the research argument before showing results. Introduce the task, literature, baseline architecture and each changed algorithm before its score. A method name is never an explanation: state its purpose, exact inputs and outputs, learned parameters, training loss and inference steps. Define every variable and tensor at first use, and distinguish the published method, our port, our internal comparator and proposed additions. Use diagrams and worked examples where they explain the mechanism. Put source and implementation citations next to the claims and in speaker notes. Keep results and proposed work separate, and do not use a success number as the opening explanation.
