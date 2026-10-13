# Completed baseline batch: the paper must confront vanilla flow 98

All five additional methods are now complete and audited. Vanilla action-flow
BC scores 98/104 versus world search 84, diffusion74 and feedback76; the SAGE port
scores 77/104. Our retained proposed repair method remains80. The
[full report](ADDITIONAL_BASELINE_RUNS_20261012.md) preserves exact curves and limits.
The intended thesis—execution-calibrated bounded flow repair—remains a research
question, with implementation and negative results. Current evidence does not
support claiming that our TTT/reasoning/repair improves the strongest controller.
The plain BC baseline cannot be relabeled as our novel mechanism. Future authorized
work should establish a useful correction of a strong action-flow incumbent and
measure real success per controller allowance; additional complexity alone is
not a contribution. No new experiment is authorized by this blueprint.

Historical formulation/design notes follow; their strongest-reference framing
is superseded by the measured baseline comparison.

# New baseline evidence supersedes the80 strongest-reference premise

October12: vanilla conditional action flow completes77/84/89/98, selecting98/104
under the same registered development protocol. It uses no world search, TTT,
workspace revision or residual repair. [Audited findings](VANILLA_ACTION_FLOW_RESULTS.md)
include19 wins/one regression versus retrieval80 and20/two versus GMM80.
The proposed execution-calibrated repair controller80 does not outperform this
standard BC control. Do not relabel the baseline as our novel method or claim
our extra mechanisms improve the strongest measured controller. The thesis below
remains a research question; its contribution/effectiveness evidence must be
reassessed against the98 baseline after the remaining comparisons complete.

Historical formulation/design blueprint follows; the retained method equations
remain accurate, but its strongest-flow-reference framing is superseded.

# Paper blueprint: execution-calibrated bounded flow repair

October 12, 2026. This is the paper's proposed framing, the implemented
formulation, and an evidence map. It does not authorize another run. The
[experiment index](RESULTS_INDEX.md) and [ledger](results-ledger.json) preserve
all results, including unsuccessful iterations.

## Central thesis and title

**Execution-calibrated bounded flow repair for visual world-model planning.**

The research question is whether a frozen visual world model can support a
generative controller that learns from its actual execution errors to revise
the next plan within a fixed compute allowance. The scientific target is
**better executed decisions per unit of controller computation**. Prediction
accuracy is a diagnostic and an intermediate learning target.

The proposed mechanism connects three operations: fit a small episode-local
correction from observed action prefixes; use corrected candidate consequences
to revise a latent workspace; generate bounded action repairs around a retained
incumbent. JEPA supplies the visual representation and predictive geometry;
flow matching supplies action proposals and local revisions; execution supplies
the factual signal that can correct the planner's judgment.

The distinction to pursue is **how observed errors change generation and action
selection**, with factual supervision and a bounded replacement mechanism.
“JEPA + flow + reasoning + TTT” is an ingredient list, not a sufficient novelty
claim. Existing work already studies feedback correction and diffusion guidance.
Novelty of this particular mechanism remains to be established against that work.

The best measured flow formulation is retrieval-repair at **80/104**. Its linear
episodic memory is the current technical anchor. The more expressive nonlinear
memory reaches **73/104** and should appear as a documented unsuccessful
extension, not the main method selected for its sophistication.

Retained execution source is `3bdfb49e7c9d4952da7494e50c8dc2530197641e`,
selected checkpoint 20,000, run `6wpx3jcv`. The source/configuration and
checkpoint hash are preserved in its [results report](RETRIEVAL_REPAIR_TTT_RESULTS.md).

## Contributions and their evidence status

| Contribution to develop | Implemented content | Current evidence |
| --- | --- | --- |
| Execution-calibrated generative plan revision | Factual episodic correction conditions generation, candidate scoring and a consequence-conditioned workspace | Mechanism implemented; causal success benefit not isolated |
| Bounded residual flow repair | Second action flow conditions on the selected incumbent and revised workspace; exact incumbent remains eligible | Bound and controller accounting audited; retaining an incumbent guarantees only surrogate-cost availability |
| Training through factual adaptation | Differentiate through a fit on preceding recorded transitions; supervise the next recorded action/outcome; exclude own-episode retrieval | Causal boundaries tested; generated alternatives have no physical outcome labels |
| Empirical characterization of the prediction–execution gap | Full learning curves, paired failures, calibration, candidate spread, timeouts and measured image sequences | Available descriptive evidence; single seed and repeated development selection limit inference |
| Reproducible comparison artifact | Source/config/checkpoint identities, full periodic metrics, all registered evaluations and negative results | Audited and published; supporting artifact rather than stand-alone algorithmic novelty |

Retrieval, shifted-plan reuse, stall penalties, clipping and the solver are
supporting implementation choices. Describe the problem each addresses. Do not
present each inherited component or engineering optimization as a new contribution.

## Formulation matching the retained implementation

### Setting and variables

Let the current RGB observation be $o_t$, the goal image be $g$, the frozen
encoder be $E$, and the frozen action-conditioned world be $F$. Write
$z_t=E(o_t)$, $z_g=E(g)$. Current and goal images use the same static-image
input convention: repeat the frame over the encoder's 16-frame input and pool
to 32 spatial tokens of dimension 1,024. This is a frozen V-JEPA 2.1 representation
plus our trained predictor, not an unchanged native LeWM or released JEPA controller.

An action plan $A\in[-1,1]^{5\times8}$ contains five blocks. Each block contains
two four-dimensional simulator actions. Roll out
$\hat z_{1:5}=F(z_t,A)$, select a plan, and execute only its first block.
The next observation reveals the outcome of those two actions; the unexecuted
suffix remains a prediction. Episode-local memory resets between cases.

Retrieve seven distinct local targets from TRAIN routes and add the direct
goal, giving eight target options $q$. A retrieved route includes a recorded
start, local next target, farther route endpoint and first five-block action
chunk. Its route score is the sum of current-to-recorded-start and
route-endpoint-to-final-goal cosine distances. Direct-goal route cost is
current-to-goal distance. These are geometric proxies, not measured task values.

### Factual episodic correction

Retain at most four observed prefix transitions
$\mathcal H_t=\{(z_i,a_i,\hat z_i^+,z_i^+)\}$. For spatial patch $p$, define
$d_p(z,q)=1-\cos(z_p,q_p)$ and factual error

$$
y_{ip}(q)=\frac{d_p(z_i^+,q)-d_p(\hat z_i^+,q)}{s},\qquad s=0.02.
$$

The learned, unit-normalized key
$k_{ip}=K_\eta(z_i,a_i,\hat z_i^+,q,p)\in\mathbb R^{32}$ uses pre-outcome
features and a constant coordinate. Fit a target-specific correction about a
learned prior $w_0$:

$$
w_t(q)=\arg\min_w\sum_{i\in\mathcal H_t}\frac1{32}
\sum_{p=1}^{32}(k_{ip}^{\mathsf T}w-y_{ip})^2
+\lambda\|w-w_0\|_2^2,\qquad\lambda=0.125.
$$

The batched Cholesky solve is differentiable and fit separately for each target.
Its update $w_t-w_0$ conditions the latent workspace. Four transitions supply
128 patch rows; these are **four physical observations**, not 128 independent trials.

For a candidate, predict a bounded patch correction

$$
c_p(A,q)=b\tanh\!\left(\frac{s\,k_p(A,q)^{\mathsf T}w_t(q)}b\right),
\qquad b=0.1,
$$

and use $e_{\rm pre}=\frac1{32}\sum_p c_p$. A separate bounded learned head
predicts terminal-window error $e_{\rm term}$. The retained variant uses the
positive parts as penalties on optimistic predictions; a negative fitted error
does not directly reward a candidate. Match this deployed transformation during
training rather than describing an unbounded residual corrector.

### Candidate score and latent revision

Let $d(z,q)=\frac1{32}\sum_p d_p(z,q)$. The local candidate cost is

$$
J_t(A,q)=\alpha d(\hat z_1,q)
+\frac{1-\alpha}{2}[d(\hat z_4,q)+d(\hat z_5,q)]
+\alpha[e_{\rm pre}]_++(1-\alpha)[e_{\rm term}]_+,
\quad\alpha=0.5.
$$

The final route/action choice minimizes route cost + local cost + the inherited
observed-stall penalty. The latter requires repeated locally similar optimistic
failures, has recency decay, and is capped at 0.02. It is not a task-success critic.

A four-slot latent workspace attends to current state, local target, final goal
and factual execution history. After the initial pool is predicted and scored,
it attends to candidate tokens preserving the correspondence among the action,
ordered spatial cost maps, corrections and total score. Write this revision as

$$
h^{(1)}=R_\theta(h^{(0)},\{(A_j,M_{1:5,j},e_j,J_t(A_j,q))\}_j).
$$

Here $M_{1:5,j}$ denotes ordered spatial cost maps derived from predicted
states, and $e_j$ denotes prefix/terminal corrections. These are the actual
candidate-token contents, alongside the action and cost. Rollout evidence and discrete
incumbent selection are detached. Gradients train the workspace and conditioned
flows; they do not backpropagate through the complete imagined solver. Imagined
evidence uses a distinct token type from observed history. The workspace is an
internal computational state, not an asserted physically reachable JEPA state.

### Initial flow and bounded repair flow

Train an initial conditional action flow on recorded $A^*$. With
$\epsilon\sim\mathcal N(0,I)$, $\tau\sim U[0,1]$, and
$x_\tau=(1-\tau)\epsilon+\tau A^*$, use

$$
\mathcal L_{\rm action}=
\mathbb E\|v_\psi(x_\tau,\tau\mid z_t,q,z_g,h^{(0)})-(A^*-\epsilon)\|_2^2.
$$

At deployment, sample four plans per target with eight Euler steps. For retrieved
targets, replace one slot with a recorded route action; when available, reuse a
shifted previous plan in another slot. Predict and score the pool to obtain
incumbent $A_0$. Revise the workspace from this candidate evidence.

Train a second flow on the normalized bounded teacher residual

$$
U^*=\operatorname{clip}((A^*-A_0)/\rho,-1,1),\qquad \rho=0.25.
$$

Its condition includes $A_0$ and $h^{(1)}$. With
$u_\tau=(1-\tau)\epsilon_r+\tau U^*$, use

$$
\mathcal L_{\rm repair}=\mathbb E
\|v_\omega(u_\tau,\tau\mid z_t,q,z_g,h^{(1)},A_0)-(U^*-\epsilon_r)\|_2^2.
$$

Four Euler steps generate three residual repairs. Form
$A_j=\operatorname{clip}(A_0+\rho\operatorname{clip}(U_j,-1,1),-1,1)$,
and keep $A_0$ as the fourth exact candidate. Thus
$\|A_j-A_0\|_\infty\le\rho$, apart from floating-point roundoff. All
completed proposals are rolled out and rescored under the same fitted correction
and fixed score context. Across both rounds the deployed configuration uses
$8\times4\times5\times2=320$ predicted world transitions per full decision.

Keeping the incumbent makes its surrogate score available. The lowest retained
score cannot increase under an unchanged scoring context. This is **not** a
guarantee of physical improvement, calibrated uncertainty or safe task execution.

### Offline learning and supervision boundary

The objective is

$$
\mathcal L=\mathcal L_{\rm action}+\mathcal L_{\rm repair}
+0.1\mathcal L_{\rm factual\ calibration}.
$$

The calibration term averages
Smooth-L1 errors for predicted versus recorded prefix patch error and
terminal-window error, scaled by 0.02. Its computational graph differentiates
through the causal fit on preceding transitions into the learned keys/prior
and conditioning. Empty history returns the prior.

Updates alternate depths one and two. Depth one trains initial flow and
calibration; depth two additionally trains the repair flow. On deep updates,
cross-episode TRAIN retrieval supplies one candidate slot for seven out of eight
batch positions. The entire query episode is excluded before shortlisting and
reranking. Retrieval uses the current state/requested target and exclusion ID,
not the teacher action or a future outcome label. Shifted-plan deployment remains
an incompletely matched training distribution.

The teacher residual uses a recorded action as an imitation target. It does
not establish the physical outcome of a generated alternative. World predictions
are evidence for planning, never fabricated environment labels. There is no
task-reward critic, GRPO stage, chain-of-thought annotation, one-step distillation,
or general inference-depth scaling result in this retained implementation.

Exact sources: [memory and scoring](../flow_jepa/execution/progress_ttt.py),
[candidate-conditioned workspace and initial flow](../flow_jepa/execution/flow_reasoning.py),
[bounded repair](../flow_jepa/execution/repair_flow_ttt.py),
[cross-episode retrieval](../flow_jepa/execution/retrieval_repair.py),
[controller](../flow_jepa/execution/flow_controller.py),
[route bank](../flow_jepa/execution/bank.py),
[configuration](../config/flow_metaworld_retrieval_repair_ttt.json).

## Related work and the distinction that needs testing

| Primary reference | Relevant foundation | Our relationship and claim boundary |
| --- | --- | --- |
| [LeFlow](https://arxiv.org/html/2608.24855v1), [official code](https://github.com/hsiangwei0903/LeFlow) | Latent-path flow, inverse action decoding and frozen-world reranking | Our retained flow generates and repairs actions using JEPA consequence evidence; it is an architectural extension of the planning direction, not a small unchanged LeFlow add-on |
| [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) | Generative action chunks | Action generation is established; the proposed contribution concerns execution-conditioned revision |
| [Feedback World Model](https://arxiv.org/html/2605.15705) | Observation-error feedback plus diffusion-policy guidance | Closest overlap: feedback with generative control already exists. We must establish a distinct correction/replacement mechanism and its decision benefit |
| [QGF](https://arxiv.org/html/2606.11087), [code](https://github.com/zhouzypaul/qgf) | Critic-gradient steering of flow policies using a clean-action estimate | Useful guidance foundation; our retained repair has no critic gradient and must not be labeled QGF |
| [LaCT](https://tianyuanzhang.com/projects/ttt-done-right/), [code](https://github.com/a1600012888/LaCT) | Nonlinear fast memory and efficient batched updates | Our small-context, MSE-updated transfer scores 73; it is not full LaCT/Muon or a demonstrated scaling benefit |
| [LARC](https://rootyjeon.github.io/latent-reasoning-umm/) | Latent reasoning for unified multimodal generation | Motivation for an internal workspace; our consequence-conditioned control is a task adaptation, not a BAGEL/LARC reproduction |
| [ALPaCA](https://arxiv.org/abs/1807.08912), [code](https://github.com/StanfordASL/ALPaCA) | Meta-learned features/priors for Bayesian online regression | The tested posterior/gating variant scores 77; no trustworthy physical-success confidence guarantee follows |
| [HWM](https://arxiv.org/html/2604.03208v2) | Hierarchical latent planning | Relevant baseline; current robotics result is a documented paper port with severe time limits |

This table records a positioning review, not an exhaustive novelty certification.
Before submission, compare mechanisms directly with the closest feedback/planning
work and complete a targeted novelty review. Cite adopted components explicitly.

## Results to portray honestly

### Experimental setup to disclose

| Item | Registered setting |
| --- | --- |
| Domain | Custom image-goal MetaWorld-v3 manipulation, 13 seen tasks; not standard MT10/MT50 |
| Inputs | Current RGB and completed-task goal RGB; frozen V-JEPA 2.1 ViT-L features, 32 x 1,024 spatial tokens |
| Shared-world training inventory | 7,800 episodes of 200 primitive actions: 6,240 scripted expert and 1,560 random; 1.56 million transitions |
| Policy/retrieval training | Same 6,222 successful expert episodes; no development episodes in policy training |
| Optimization | Fresh policy, seed 3072, global batch 64, AdamW, initial peak learning rate 0.0001, weight decay 0.05, gradient clip 1; 1k warmup and cosine decay |
| Ceiling / sampling | 20,000 updates or 28,800 aggregate optimizer GPU-seconds; 1.28 million replacement-sampled windows at the update cap; no conventional epoch count |
| Hardware / actual compute | A800 GPUs; retrieval reference uses four ranks and 1.749 optimization GPU-hours; nonlinear uses eight and 4.813. Global batch stays 64 but rank-dependent random streams/negatives differ |
| Development | 650 reserved episodes; the same 104 screened goal cases, eight per task, evaluated at 5k/10k/15k/20k; highest success selects checkpoint with earliest tie |
| Controller allowance | 200 primitive actions and ten cumulative seconds including encoding/planning; simulator rendering/physics excluded; all online adaptation is charged |
| Reserved evaluation | 3,200 cases, 200 per each of 16 tasks, including three unseen tasks; not a current reported test result |

Seen tasks are assembly, button-press-topdown, coffee-button, dial-turn,
door-close, door-open, drawer-close, drawer-open, faucet-open, handle-press,
pick-place, plate-slide and reach. Reserved unseen tasks are window-open,
handle-pull and push. Goal-image screening uses the fixed collection expert's
success before method training; describe the screened distribution explicitly.
Common encoder/world/preparation costs are separate from policy optimization.

| Selected development model | Success /104 | Interpretation |
| --- | ---: | --- |
| Execution revision, GMM | 80 | Strongest non-flow internal reference |
| Repair flow / retrieval repair | 80 / 80 | Strongest flow candidates; neither exceeds GMM |
| Reasoning-conditioned flow | 79 | One-case difference cannot establish superiority |
| Goal-progress objective | 78 | Additional supervision does not improve the best result |
| Bayesian acceptance | 77 | Uncertainty mechanism does not improve the best result |
| Nonlinear memory | 73 | Better prediction proxy with worse task completion |
| LeFlow port / HWM port / released CEM | 21 / 5 / 24 | Shared-backbone, fixed-budget comparison; not native benchmark scores |

Use all learning curves and per-task results, not only selected maxima. The
retrieval80 and GMM80 checkpoints have six paired wins and six losses. Nonlinear73
has eight wins and fifteen losses versus retrieval80. Its assembly and pick-place
results stay 0/8, while reach falls from 8/8 to 4/8. At 20k its adapted next-prefix
MAE is 0.001833 versus raw 0.002935, yet success is 72. These measurements motivate
the problem; they do not isolate a causal reason for failure.

The main metric is environment task success within 200 actions and ten cumulative
controller seconds. Report latency, timeout rate, world calls, success by action
count and paired failures alongside it. Calibration, diversity and imitation
losses answer diagnostic questions. Eight cases per task, one training seed,
checkpoint selection and repeated development use limit the comparison.
LeFlow's loss was still improving at the common update ceiling; HWM/CEM are
strongly time-limited. A common ceiling is not matched convergence or equal
consumed compute. See [baseline audit](RELEASE_BASELINES.md).

## Paper narrative and section design

1. **Introduction:** present the prediction-to-execution mismatch in visual
   generative planning; identify observed execution as a causal feedback source;
   state the proposed bounded repair mechanism and its concrete evaluation question.
2. **Related work:** distinguish latent-path planning, action generation,
   feedback correction, fast adaptation and inference guidance. Focus the novelty
   discussion on the closest mechanisms rather than a long list of fashionable methods.
3. **Method:** setting and factual history; episode-local calibration;
   consequence-conditioned workspace; initial and residual flows; deployed
   score/selection; offline training and exact supervision boundaries.
4. **Experiments:** benchmark/provenance; primary task success and runtime;
   strongest internal comparator; paired failures; calibration versus decisions.
   Put unsuccessful extensions and full curves in the appendix with links from the main text.
5. **Limitations:** offline behavior coverage, progress representation, absent
   counterfactual labels, single-seed development, baseline ports and missing scaling evidence.

An eight-page main text could allocate roughly 1.5 pages to motivation/related
work, 2.5 to method, 3 to experiments/analysis and one to discussion. This is a
layout proposal, not a claim that the present evidence satisfies a venue's bar.
Keep equations and measured failure images close to the claims they support.

### Current-evidence abstract draft

Frozen visual world models provide a predictive substrate for generative
planning, but their candidate scores can remain misaligned with executed task
progress. We study execution-calibrated bounded flow repair: an episodic
correction fitted to observed action prefixes conditions candidate evaluation
and latent plan revision, while a residual flow proposes bounded changes around
a retained incumbent. Offline training differentiates through factual adaptation
without assigning demonstration outcomes to generated alternatives. In a custom
thirteen-task MetaWorld development protocol, the strongest flow controller
achieves 80/104 successes, matching a strong GMM controller and exceeding the
tested shared-backbone baseline ports under the specified budget. Nonlinear
memory and Bayesian acceptance extensions improve aspects of calibration but
fail to improve success. These results expose the remaining gap between more
accurate prediction and useful generative decisions; independent evaluation and
mechanism attribution remain necessary.

## Figure design grounded in inspected references

I read LeFlow’s formulation, LaCT’s project page and paper abstract, and the
closest feedback/guidance formulations,
and visually inspected **LeFlow Figure 1**, **LeFlow Figure 3**, and **LaCT
Figure 2** from the actual reference images. Figure identities, inspected-image
hashes and applicable lessons are recorded in the
[reference review](reports/20261012-paper-blueprint/reference-figure-review.json).
No new empirical figure or synthetic success trajectory is created here.

| Proposed figure / venue position | Communication gap | Data and faithful design |
| --- | --- | --- |
| Main Fig. 1: observed error changes the next repair | What the mechanism adds to ordinary sample-and-rank | One actual episode with current/goal RGB, factual predicted-versus-observed prefix cost, and a small causal schematic separating observation, workspace and action. Any schematic geometry must be labeled analytic |
| Main Fig. 2: paired task success and runtime | Is the controller better, and under which allowance? | Measured per-task success/paired outcomes and latency/timeout panels; label development, checkpoint selection and baseline ports. Borrow the separate outcome/cost presentation lesson from LeFlow Fig. 1 |
| Main Fig. 3: calibration versus execution | Why more accurate prediction is insufficient | Recorded selected-prefix errors plus actual task success across checkpoints. Rejected candidates have no factual physical labels; do not invent ranking accuracy for them |
| Main Fig. 4: same-reset qualitative cases | What succeeds or stalls physically? | Goal plus matched-frame sequences from preserved rollouts, fixed sampling times, same cases across methods, and both success and failure examples. Do not infer forces or slip from sparse single-view images |
| Future main Fig. 5: success versus inference budget | Does extra inference provide useful capability? | Prospective measured curves at frozen checkpoints and common allowances; currently missing. Training-checkpoint curves cannot substitute for this experiment |
| Appendix: evidence update/apply detail | What is fitted, and when does its outcome become known? | Distinguish observed support, imagined queries and applied corrections. LaCT Fig. 2 supplies the update/apply distinction; display our actual four-transition context size |

LeFlow Fig. 3 motivates separating planned states from predicted rollouts; our
visuals must additionally separate both from observed execution. Use real
samples and measured plots for empirical claims, visible captions, and accessible
color/line encodings. Avoid making a generic diffusion/noise tutorial or dense
boxes-and-arrows chart the headline contribution. Do not export/share figures
or a deck until every assembled page/slide has been rendered and inspected.

## Evidence needed before stronger claims

| Intended claim | Required evidence still missing | Compute-conscious route |
| --- | --- | --- |
| Selected method generalizes beyond development | Frozen method/checkpoint on independent cases with paired uncertainty and per-task coverage | Reuse saved checkpoints on the registered reserved cases after explicit evaluation authorization |
| Correction/repair causes the gain | Focused mechanism comparison against the strongest internal controller | Choose the minimum comparison that tests the final claim; inference-only disabling can reuse weights but must disclose the train/deploy mismatch |
| Inference scaling is useful | Success/latency versus actual inference allowance or depth at fixed weights | Validate supported depths/implementation, then a small preregistered curve; no fictitious scaling claims |
| TTT is beneficial | Adapted-versus-prior decision outcomes under identical action/time accounting | Reuse the final trained model where feasible; calibration MAE alone is insufficient |
| Advantage is not only the deadline | A small common-runtime comparison | Evaluate saved weights without retraining, with timeouts and completed actions |
| Robustness across retraining | Independent training replication | Not covered by current one-seed authorization; disclose this limit unless separately authorized |

These are prospective evidence requirements, not a queue of authorized runs.
The immediate research priority is to connect factual correction to useful
decisions or identify that representation/proposal coverage is the limiting
factor. More expressive memory, a smaller surrogate loss or an additional
auxiliary term is not sufficient evidence of a better method.

If final experiments establish both independent task gains and a distinct
decision mechanism, the method paper should lead with that result. If they
do not, the current defensible output is a controlled empirical study of the
prediction–execution gap and the limits of these adaptations. Retain the intended
method thesis while letting measured results determine the final claims.
