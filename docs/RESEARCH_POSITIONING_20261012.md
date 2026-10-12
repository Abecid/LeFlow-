# Research positioning after the LeFlow / V-JEPA experiments

**Assessment date:** October 12, 2026.  
**Audited main snapshot:** d42d76c1cf2f39a56de189148482122775c3962e.  
**Evidence:** [independently derived audit](reports/20261012-research-positioning/audit.json), [complete results index](RESULTS_INDEX.md), [original run ledger](results-ledger.json).

## Recommendation

Keep **retrieval-conditioned bounded action-flow repair** as the engineering base. The strongest paper direction to pursue is **learning and adapting the value of the action prefix that will actually execute, so additional flow proposals improve real decisions**.

The current proposed title, “execution-calibrated bounded flow repair,” describes a substantial implementation. Its main empirical and novelty claims remain unproven. Three controllers tie at 80/104: the earlier GMM, bounded repair flow, and retrieval-conditioned repair. Neither flow nor TTT has an isolated improvement over the strongest internal control. Recent papers also already cover feedback-guided diffusion, flow plus JEPA, action-discriminative world models, retrieval-guided flow, and JEPA adaptation.

The best separate direction is **temporal reachability/value learning**, starting from MQE and optionally Dual Goal Representations. This changes the quantity the planner scores. Another auxiliary head predicting expert timestamps, a larger fast memory, or more denoising steps would repeat the current pattern without resolving the main uncertainty.

This report proposes research and experiments. It does not launch training, branch the simulator, change the registered data protocol, or open final-test outcomes.

## 1. Audit coverage

The audit covers all 16 registered current shared-world comparisons and 61 current checkpoint evaluations. Success and timeout counts were independently recomputed from 57 current full evaluation files. Their 104 case IDs, reset seeds, episode hashes, and model seed match. An additional 24 historical MetaWorld files were recounted: **81 full files, 8,424 episode records**.

The four Bayesian evaluations are large chunk-manifest archives. Their manifests and saved final numerical analysis were inspected; the complete chunked traces were not reconstructed. The derived audit distinguishes these four from the raw recount. All recounted counts match the published records.

The older PushT branch was also inspected. Its last scientific results precede the current MetaWorld campaign; its flow run remains incomplete. That historical status is separate from the completed current main campaign.

Known measured main-campaign components total approximately **118.74 GPU-hours**, with disclosed missing preparation costs. This is not a complete cluster bill. All 16 current comparisons are recorded complete.

## 2. What the current system actually does

This is a custom MetaWorld-v3 image-goal controller using a frozen official V-JEPA 2.1 encoder and a separately trained frozen spatial dynamics model. It is not an unchanged LeFlow reproduction.

| Item | Actual setting |
| --- | --- |
| World training | 7,800 episodes: 6,240 scripted expert and 1,560 random; 200 primitive actions each; 1.56 million transitions |
| Policy and route-bank training | 6,222 successful expert episodes |
| Development | 13 seen tasks, eight fixed cases each, 104 total; repeatedly reused at 5k/10k/15k/20k |
| Other reserved validation data | 650 episodes originally reserved; the 104 controller cases are a fixed subset |
| Reserved final evaluation | 3,200 cases across 16 tasks, including three unseen tasks; no current final-test result |
| Optimization | One seed, 3072; global batch 64; 20,000 charged updates or eight optimizer GPU-hours |
| Execution allowance | 200 primitive actions and ten cumulative controller seconds, including encoding and online adaptation; rendering/physics excluded |
| Hardware | A800 GPUs; recent runs use four or eight ranks, so same seed/global batch does not imply identical random streams |

Let B be the batch size. A current RGB image and a goal RGB image each have shape [B, 256, 256, 3]. Each image is repeated into a 16-frame clip. The frozen encoder receives [B, 3, 16, 256, 256], then pools its output to **[B, 32, 1024]**. The configured pooling grid is [2, 4, 4]. These are static-image features, not sixteen distinct observed frames.

Call the resulting current and goal tensors Z and G. The planner retrieves seven local targets from training trajectories and adds the direct goal, giving eight targets. For each target, the initial action flow proposes four plans:

- Per target: [4, 5, 8].
- All targets: [8, 4, 5, 8].
- Five blocks per plan; each eight-vector block contains two four-dimensional MetaWorld primitive actions.
- Each full candidate therefore contains ten primitive actions, but **only the first two execute before replanning**.

The frozen world rolls each plan forward for five blocks, producing predicted states [8, 4, 5, 32, 1024]. A four-slot internal workspace consumes candidate actions, predicted spatial goal costs, fitted error corrections, and scores. It conditions a second flow that proposes local repairs around an incumbent. The workspace is a learned computational state, not a claim that an imagined latent is physically reachable.

The retained method uses eight initial Euler steps, then four repair Euler steps. For each target, the repair pool contains three bounded residual repairs and the unchanged incumbent. The complete decision makes 320 predicted world transitions. Incumbent rollouts are included in this accounting.

### Actual learning objectives

Let A* in [-1,1]^(5 x 8) be a recorded expert action chunk, epsilon an identically shaped Gaussian sample, and tau in [0,1] flow time. The initial flow velocity v is trained by

$$
x_\tau=(1-\tau)\epsilon+\tau A^*,\qquad
L_{\rm flow}=E\|v(x_\tau,\tau,\text{context})-(A^*-\epsilon)\|^2.
$$

Let A0 be the selected initial plan and rho = 0.25. The repair target is

$$
U^*=\operatorname{clip}((A^*-A_0)/\rho,-1,1).
$$

A second flow learns to generate U*, and a sampled repair U gives

$$
A_{\rm repair}=\operatorname{clip}(A_0+\rho\operatorname{clip}(U,-1,1),-1,1).
$$

Thus the action change is bounded by 0.25 in every coordinate. Retaining A0 guarantees availability of its **surrogate score**, not preservation of real success.

The episodic correction uses at most four already executed two-action transitions. It fits 32-dimensional learned keys by ridge regression, separately for each local target, against observed-minus-predicted patchwise goal-cosine error. Four transitions yield 128 patch rows, but only four physical action interventions. The fit is differentiable, and its fast weights condition generation and scoring.

The selected method's outer loss is initial-flow loss + repair-flow loss + 0.1 times factual calibration loss. Generated alternatives do not have physical outcome labels. They receive frozen-world predictions, and their repair targets come from recorded expert actions.

Exact implementation: [encoder](../flow_jepa/vision.py), [initial flow and workspace](../flow_jepa/execution/flow_reasoning.py), [factual adaptation](../flow_jepa/execution/progress_ttt.py), [repair](../flow_jepa/execution/repair_flow_ttt.py), [retrieval training](../flow_jepa/execution/retrieval_repair.py), [controller](../flow_jepa/execution/flow_controller.py), [retained configuration](../config/flow_metaworld_retrieval_repair_ttt.json). The [existing blueprint](PAPER_BLUEPRINT.md) supplies the complete implemented equations.

## 3. All current comparisons

Curves are successes out of 104 at 5k / 10k / 15k / 20k charged updates. “Selected” is the best reused-development checkpoint, with earliest ties. CEM has no extra head training.

| Method | Complete curve | Selected | Selected timeouts |
| --- | --- | ---: | ---: |
| Controller-grounded GMM | 79 / 78 / 75 / 78 | 79 | 0 |
| Latent revision GMM | 73 / 75 / 74 / 77 | 77 | 0 |
| Execution revision GMM | 80 / 78 / 79 / 76 | **80** | 0 |
| Reasoning-conditioned action flow | 79 / 75 / 76 / 76 | 79 | 10 |
| Flow plus episodic adaptation | 77 / 76 / 78 / 74 | 78 | 21 |
| Guided action-flow TTT | 72 / 70 / 72 / 75 | 75 | 29 |
| Bounded repair-flow TTT | 80 / 73 / 75 / 75 | **80** | 0 |
| Retrieval-conditioned bounded repair | 75 / 77 / 79 / 80 | **80** | 0 |
| Shifted-plan repair training | 79 / 78 / 74 / 78 | 79 | 0 |
| Goal-progress objective and TTT | 72 / 72 / 78 / 77 | 78 | 0 |
| Bayesian posterior repair | 71 / 76 / 77 / 77 | 77 | 0 |
| Nonlinear episodic repair | 73 / 66 / 67 / 72 | 73 | 0 |
| Direct LeFlow plus nonlinear TTT | 27 / 21 / 27 / 29 | 29 | 0 |
| LeFlow release-module/spatial port | 14 / 21 / 19 / 14 | 21 | 0 |
| HWM paper port | 2 / 3 / 5 / 5 | 5 | 99 |
| Released CEM | 24 | 24 | 80 |

The main improvement over the external ports already exists in the controller-grounded GMM at 79. The flow-based additions currently supply **zero percentage-point improvement over the strongest GMM**.

Direct LeFlow TTT improves its port from 21 to 29, with 12 paired wins and four regressions. This is a legitimate development observation, but the TTT variant consumes 2.437 optimizer GPU-hours versus 0.852 for the port, and the contribution of adaptation alone is not isolated.

### Historical campaigns must remain separate

| Earlier MetaWorld run | Four checkpoint scores /104 |
| --- | --- |
| Original joint latent/action flow | 23 / 17 / 16 / 19 |
| Original LeFlow adaptation | 26 / 20 / 25 / 27 |
| Original HWM adaptation | 7 / 2 / 6 / 8 |
| Original world CEM diagnostics | 10 / 9 / 7 / 7 |
| Repaired joint flow | 17 / 17 / 16 / 15 |
| Repaired world CEM diagnostics | 23 / 24 / 24 / 22 |

The original samplers had a confirmed 1,024-dimensional-state / 256-dimensional-output-subspace restriction and inconsistent static-goal versus causal-state feature conventions. The repair changed the world and representation. These scores cannot substitute for current shared-world baselines. World checkpoint selection used prediction loss, explaining historical reported CEM references of 7 and 22 rather than their curve maxima.

A historical 26-condition, 208-candidate intervention diagnostic found predicted-versus-actual goal-progress Spearman correlations of 0.031 for ours and 0.099 for LeFlow, despite correlations of 0.899 and 0.875 for distance to each model's own proposed subgoal. This is an earlier, different-world diagnostic, not a current-controller result. It illustrates why “accurate about the selected waypoint” need not mean useful progress toward the real task.

The older PushT pilot used frozen LeWM and 18,685 episodes (14,948 / 1,868 / 1,869 train / validation / test). At selected 1k checkpoints, BTM versus 16-step flow scored 40/40/25% versus 50/30/20% at offsets 25/50/100. The primary difference was five versus four successes out of twenty, with a reported paired interval spanning -15 to +25 percentage points. At the last matched 23k checkpoint, the corresponding scores were 5/0/10% versus 45/15/5%. BTM completed 23,830 updates; flow remained 23,000 saved / 23,040 logged, without its final cycle. One versus sixteen generator calls yielded only a descriptive 1.38x solver timing ratio under different hardware/load conditions. This does not justify reviving BTM.

Sources: [repair history](REPAIR_RESULTS.md), [repaired results](REPAIRED_VALIDATION_RESULTS.md), [PushT assessment at its branch snapshot](https://github.com/Abecid/LeFlow-/blob/bedb8bc9b84828ca105457be7b3437fded30ab3e/docs/research/pusht_pair_20261004/reports/20261006T050931Z.md).

## 4. Findings that should drive the research decision

### The tied controllers have complementary capabilities

Repair-flow80 versus GMM80 has six wins and six regressions. Retrieval80 versus GMM80 also has six and six; retrieval80 versus repair80 has seven and seven.

Independently combining their completed rollout outcomes gives:

- Three tied80 controllers: **89 distinct solved cases (85.58%)**, 70 successes shared by all three.
- Add nonlinear73: **92 distinct solved cases (88.46%)**, 64 shared by all four.

This is a descriptive oracle that retrospectively chooses the controller for each entire episode. It is not candidate-level oracle performance, a deployable ensemble, or a prediction that a learned router will achieve 89 or 92. It does show that some failures are not universal across policies using the same frozen features.

| Task, eight cases | GMM80 | Repair80 | Retrieval80 | Nonlinear73 |
| --- | ---: | ---: | ---: | ---: |
| Assembly | 3 | 1 | 4 | 0 |
| Dial turn | 7 | 7 | 7 | 3 |
| Door open | 6 | 8 | 8 | 7 |
| Faucet open | 3 | 2 | 3 | 0 |
| Pick and place | 1 | 2 | 1 | 0 |
| Reach | 4 | 4 | 1 | 8 |

Retrieval80 saturates eight tasks; **23 of its 24 failures** occur in assembly, faucet-open, pick-place, and reach. Nonlinear memory solves all eight reach cases at its selected checkpoint but loses assembly, faucet, and pick-place. An aggregate mean hides these capability exchanges.

### Better factual prediction has not delivered better decisions

These errors are MAE in the encoded goal-distance prediction, not meters, forces, success probabilities, or labels for unexecuted alternatives.

| Selected method | Raw MAE | Prior-corrected MAE | Adapted MAE | Success |
| --- | ---: | ---: | ---: | ---: |
| Progress TTT | .002547 | .002237 | .001656 | 78 |
| Guided TTT | .002416 | .001773 | .001632 | 75 |
| Repair flow | .002707 | .002637 | .001688 | 80 |
| Retrieval repair | .002554 | .002350 | .001624 | 80 |
| Nonlinear repair | .003162 | .005203 | .002080 | 73 |

The nonlinear model's empty-history learned prior is worse than the raw world. Adaptation repairs part of that error, but does not beat the retained controllers. At 20k, nonlinear adapted MAE is .001833 versus raw .002935, a 37.56% reduction, yet success is 72.

The Bayesian model's nominal 90% coverage becomes 83.45% on its selected-policy trajectories, compared with 90.79% in the train-only check. Its uncertainty concerns encoded prefix error, not the probability that replacing an action preserves eventual task success.

### Shrinking proposal spread is not an established cause

The GMM and repair-flow runs lose success as proposal spread contracts. However, retrieval repair improves 75 to 80 while its initial spread contracts 62.03%; Bayesian repair improves 71 to 77 while spread also contracts. Finite candidate spread is not policy entropy. These observations reject a simple “more noise fixes training degradation” conclusion.

### Retrieval task labels do not explain all failures

Late pick-place failures often select reach or direct-goal routes. But some progress-objective and Bayesian failures retain pick-place routes throughout their late decision windows. Matching the route's source task does not establish physical feasibility or correct contact. An oracle task-ID filter would change the policy information and would not resolve this evidence.

### The clock is already part of the scientific result

Two primitive actions per decision imply at most 100 decisions for a 200-action episode. The ten-second allowance therefore permits an average of

$$
10/100=0.1\text{ second}=100\text{ ms}
$$

per decision, including encoding and adaptation.

Selected mean decision times are 99.42 ms for flow reasoning, 102.50 for progress TTT, 103.46 for guided TTT, 86.46 for repair flow, and 85.72 for retrieval repair. Their selected timeout counts are 10, 21, 29, zero, and zero respectively. Extra inference is not free: it can remove physical interaction steps.

HWM and CEM are particularly deadline-limited: their 99 and 80 failures all time out. LeFlow21 has no timeouts, so timing does not explain that entire gap. Equal update/compute ceilings also do not imply equal consumed compute or convergence.

Sources: [retrieval results](RETRIEVAL_REPAIR_TTT_RESULTS.md), [nonlinear results](NONLINEAR_REPAIR_TTT_RESULTS.md), [Bayesian results](BAYESIAN_REPAIR_TTT_RESULTS.md), [baseline fidelity and timing](RELEASE_BASELINE_RESULTS.md). The derived audit records the new oracle calculations and all selected latencies.

## 5. Position against recent work

The references below were checked against primary papers, author projects, and official code where available. Reported metrics belong to each paper's own benchmark and are not directly comparable with our 104-case protocol.

| Work | Established mechanism and relevance | Recommended role |
| --- | --- | --- |
| [LeFlow, Aug 2026](https://arxiv.org/abs/2608.24855), [code](https://github.com/hsiangwei0903/LeFlow) | Rectified flow generates intermediate latent states; inverse dynamics emits actions; frozen-world rollouts rank candidates. Our retained flow generates actions directly. | Baseline and origin, not a novelty claim |
| [Flow-JEPA, Aug/Sep 2026](https://arxiv.org/abs/2608.29029) | Flow models future latent trajectories conditioned on current observations and actions, including visual-noise robustness. | Rules out generic “replace JEPA regression with flow” novelty |
| [V-JEPA Policy, Sep 29 2026](https://arxiv.org/abs/2609.37250), [code](https://github.com/breez3young/VJEPA-Policy) | Frozen V-JEPA2.1 plus future predictor and flow action expert. Released README distinguishes code from planned checkpoint releases. | Rules out generic JEPA-plus-action-flow novelty |
| [Feedback World Model, May 2026](https://arxiv.org/abs/2605.15705) | Factual observation-error feedback corrects predicted futures and supports action-aware diffusion guidance. Linked GitHub contains the project website, not a verified runnable method release. | Closest feedback baseline; label any implementation a reproduction from the paper |
| [AD-WM, Sep 2026](https://arxiv.org/abs/2609.30264), [code](https://github.com/ad-wm/ad-wm-code) | Predictor-level action recovery preserves distinctions between candidate actions; elite regret is more informative than average factual MSE. Auxiliary heads are discarded at deployment. | Closest action-discrimination baseline |
| [DA-LeWM, Aug 2026](https://arxiv.org/abs/2608.18746) | Diagnoses latent/real ranking and CEM-stage alignment; uses inverse and goal-action objectives. | Prevents claiming the prediction-versus-ranking gap itself is new |
| [QGF, June 2026](https://arxiv.org/abs/2606.11087), [code](https://github.com/zhouzypaul/qgf) | A behavioral-cloning flow is guided by a separately trained value critic, evaluated at an estimated clean action. | Best guidance implementation once a useful value exists |
| [TraceFlow, Sep 2026](https://arxiv.org/abs/2609.20646), [code](https://github.com/zhangjiaxuan-Xuan/TraceFlow) | Successful/failed trace retrieval supplies bounded attraction/repulsion to a frozen flow; terminal outcome bits update the bank. Simulation gains are selective. | Strong cheap memory baseline and close novelty overlap |
| [AdaJEPA, June 2026](https://arxiv.org/abs/2606.32026), [code](https://github.com/agentic-learning-ai-lab/adajepa); [JEPA-TTT, Sep 2026](https://arxiv.org/abs/2610.00722) | Episodic or persistent predictor adaptation inside planning. JEPA-TTT accumulates dense replay across episodes; it is not a four-transition adaptation protocol. | Relevant for genuine dynamics shifts; protocol must match |
| [Planning Limits, Sep 2026](https://arxiv.org/abs/2609.39235) | Even perfect short-horizon prediction can fail for distant goals; target horizon/value matters. | Direct motivation for continuation value |
| [MQE, ICLR 2026](https://arxiv.org/abs/2511.07730), [code](https://github.com/WJ2003B/mqe-release) | Multistep temporal backups learn asymmetric goal reachability; visual OGBench and Bridge task stitching. | Strongest separate axis to implement |
| [Dual Goal Representations, ICLR 2026](https://arxiv.org/abs/2510.06714), [code](https://github.com/deepindermann/dual-goal-representations) | Goal embeddings represent dynamical reachability; release includes goal-conditioned flow BC. | Alternative goal representation |
| [GRASP, ICML 2026](https://proceedings.mlr.press/v306/psenka26a.html) | Joint virtual-state/action optimization with soft dynamics constraints parallelizes long-horizon planning. | Longer-term direction for genuinely non-greedy, long-horizon tasks |

Additional close work includes [Aim Short to Reach Far](https://arxiv.org/abs/2609.30036), which retrieves nearby intermediate targets from recorded trajectories; [HWM](https://arxiv.org/abs/2604.03208), which already supplies temporal hierarchy; and [ATLAS](https://arxiv.org/abs/2609.36333), which transfers useful latent geometry and aligns marginals. Simply adding a subgoal hierarchy or a geometric regularizer is not sufficient differentiation.

For lower inference cost, [Coda](https://arxiv.org/abs/2609.21216) trains a correction to a frozen flow's sampled endpoint, and [A2A](https://arxiv.org/abs/2602.07322) changes the flow source using encoded proprioceptive history. A2A's actual code uses historical agent positions; applying it would add information beyond our current image-only interface. [DA-SIP](https://github.com/ryaninkook/DASIP) already studies adaptive solver/step allocation. These are engineering controls or alternative task changes, not the main recommended contribution.

## 6. Proposed primary paper: execution-conditioned advantage verification

A working title is **Learning Which Repairs Help: Execution-Conditioned Flow Planning in JEPA World Models**.

The target claim is:

> Real execution outcomes teach the evaluator which local action changes improve continuation, allowing additional flow candidates to raise physical task success instead of merely lowering a model score.

This is a hypothesis. The novelty would have to lie in the decision-focused adaptation mechanism and demonstrated scaling, not in defining a critic or adding an existing RL loss.

### Why factual fitting is insufficient

Consider a toy scalar action a. Let the true next state be f*(a)=a, while a model predicts f(a)=-a+b. After executing a0, choose b=2a0. Then f(a0)=a0, exactly matching the observed transition.

Nevertheless, the true action derivative is +1 and the model derivative is -1. For a goal g, the true squared-cost derivative at a0 is 2(a0-g), while the model gives -2(a0-g). Gradient guidance points in the opposite direction despite zero factual error. This is an illustrative calculation, not a claim that the current neural model literally has this form.

There is a second problem: even correctly predicted latent distance can be a poor proxy for whether a task can subsequently finish. An object may need to be approached, grasped, and lifted before its position becomes similar to the goal. More accurate prediction does not repair a wrong task objective.

### First decisive experiment: separate coverage from selection

Use a prespecified set of TRAIN-derived states, never the sealed test cases. Restore the same complete simulator state and random state for each alternative. Also copy the factual controller history, fast weights, retained incumbent or cached plan, controller RNG, and remaining-budget ledger. Compare candidate prefixes from a common nested pool containing the incumbent, flow repairs, and the existing route/action proposals.

For each candidate plan A_j, execute only u_j=A_j[0], the eight-vector containing the first two primitive actions. Discard its unexecuted suffix. Update the copied base controller through its normal factual-observation update and declared cache-handling rule, then continue that fixed controller pi0 under the same remaining action/time allowance. Record true task completion and time to completion.

The estimand is explicitly policy- and budget-dependent:

$$
Q^{\pi_0}(Z,G,H,u,b)=\Pr(\text{task completion within }b\mid Z,G,H,u;\pi_0),
\qquad b=(b_{\rm actions},b_{\rm controller\ seconds}).
$$

Here H is the factual controller history and b is the allowance before executing the prefix. Prefix execution and subsequent controller work are charged to that allowance. This estimates the value of the action intervention that actually executes, not optimal task value. It does not assign the subsequent outcome to A_j's eight primitive actions that were never executed. The continuation controller is fixed because otherwise labels confound the first decision with a changed downstream policy. Repeatedly deploying a new selector changes that downstream policy; fixed-pi0 labels do not automatically become values of the new policy.

For state i and candidate j, let Y_ij be the observed binary success of this controlled continuation. If j_hat(i) is the selector's choice, measure

$$
C_K=\frac1M\sum_{i=1}^M\max_{j\le K}Y_{ij},
\qquad
S_K=\frac1M\sum_{i=1}^M Y_{i,j_{\rm hat}(i)}.
$$

Here M is the number of restored states and K is the candidate count. C_K is coverage for these fixed continuations; S_K is achieved selection success. Their difference is binary selection regret. One rollout supplies one realized outcome, not an exact success probability. Use matched continuation randomness when comparing alternatives and distinguish this finite-sample diagnostic from expected policy value. Also measure how often a repair turns a successful incumbent continuation into a failure.

A bounded pilot could use ten TRAIN-derived states per task: 13 x 10 = 130 states, one incumbent plus eight alternatives, or 1,170 continuations. At an absolute 200-primitive-action cap, this is at most 234,000 primitive actions. This is a proposed diagnostic allocation, not an observed runtime estimate or a registered run.

The result decides the direction:

- High coverage and poor selection: train the evaluator.
- Low coverage: improve action/contact/recovery coverage; another score cannot choose a nonexistent successful action.
- Correct latent-distance ranking but poor real continuation: change the value/goal objective.
- Good offline ranking but poor closed-loop deployment: investigate distribution change induced by repeated selection and the remaining controller budget.

Candidate evaluation is extra interaction data. Any model trained with it must be compared with controls receiving the same additional data and compute. It cannot be folded silently into the original 6,222-episode comparison.

If branch outcomes become training labels, partition complete source episodes before fitting the evaluator. Keep a disjoint set of restored states for outcome-ranking evaluation and replacement-margin calibration; do not report fit on the training branches as evidence of improved selection. The final reserved cases remain untouched until the method and selection rules are fixed.

### Minimal new mechanism if the diagnostic supports selection

Keep the current flow and world fixed initially. A small score head receives Z, G, the first action block u, predicted prefix features, a representation of the four factual transitions, and the remaining allowances b in [B,2]. These budgets are known controller metadata. Its output is a fixed-policy continuation score, not another cosine-error estimate. The same image and prefix can have different prospects with 20 versus 180 primitive actions remaining.

For a better-versus-worse pair j and k, train

$$
L_{\rm pair}=\log(1+\exp(-(s_j-s_k))),
$$

where s_j and s_k are the predicted continuation scores and the pair ordering comes from the actual controlled outcomes. Tied outcomes do not supply an ordering. Time-discounted completion can provide an additional declared ordering, rather than inventing dense progress labels.

The stronger research extension is to train the existing differentiable memory fit through this **outer decision loss**. Its inner update still uses only already observed transitions. Offline branch outcomes teach which features and adaptations improve subsequent ranking. At deployment, the model cannot observe rejected alternatives or future success; those labels exist only in training and evaluation.

Use an independently validated replacement margin if necessary, and measure its false-improvement rate as the candidate pool grows. A Gaussian posterior over latent error is not itself a confidence bound on action advantage.

Only after a useful ranker is established should its score guide the flow. QGF's clean-endpoint estimate is

$$
\widehat A_1=\operatorname{clip}(A_\tau+(1-\tau)v(A_\tau,\tau),-1,1).
$$

The critic gradient is evaluated at this estimated clean action. The inspected implementation supports omitting the sampler Jacobian. If the learned value describes only the executed first block, its gradient must target that block or a clearly defined full-plan value; eventual success after replanning does not label an unrealized plan.

## 7. Best separate direction: MQE-style temporal reachability

Our progress-objective experiment already predicts remaining expert time, temporal order, and soft stage compatibility. It scored 78. The new direction must use different learning semantics.

MQE learns a quasimetric: a nonnegative distance with identity and triangle inequality, while allowing d(x,y) to differ from d(y,x). Its learned distances encode temporal reachability. Variable-length temporal backups relate a current state/action to an intermediate state and then the goal, enabling composition beyond one recorded timestamp.

A concrete transfer keeps Z,G in [B,32,1024] and u in [B,8]. Learned visual adapters feed a state-action encoder phi(Z,u) and goal encoder psi(G). The inspected release defaults to a two-member ensemble and 512-dimensional embeddings, producing [2,B,512]; matched-goal distances have shape [2,B], while all-pairs training distances have shape [2,B,B]. The default networks use three 512-unit hidden layers. Its JAX implementation can guide a small PyTorch port rather than introducing a second training runtime.

Our two-primitive-action block is a new temporally extended action interface relative to the native primitive-action implementation. Sample temporal offsets in consistent units. Preserving a primitive-step discount gamma requires a block discount gamma squared; otherwise explicitly define the discount per block.

Use negative learned distance as a score. The exponential of negative distance is related to a discounted reachability value; it is not automatically a calibrated physical-success probability. Start by changing ranking only, using the same candidate bank. This tests the objective before adding guidance, a new encoder, and a new memory simultaneously.

Reachability of an exact complete goal image is also not automatically the MetaWorld task-success predicate. Robot or gripper posture can differ while the task is already complete. Quasimetric structure does not remove this mismatch; validate rankings against factual task outcomes or define a task-relevant goal equivalence before claiming that the objective problem is solved.

The current 1,560 random world-training episodes may improve coverage for value learning, but they were excluded from policy-head training. Including them in a critic is a declared data-contract change, not free extra information.

Sources: [MQE project and formulation](https://mqe-paper.github.io/), [actual implementation](https://github.com/WJ2003B/mqe-release/blob/master/impls/agents/mqe.py). Dual Goal Representations is the most relevant alternative if goal encoding rather than value backup is the bottleneck. Transitive RL is a more ambitious composition option, but its released large-data regime makes it a lower immediate priority.

## 8. What would make the paper strong

The main figure should show **real task success versus actual inference allowance**, with a frozen checkpoint. A training-checkpoint curve or lower flow-matching loss is not an inference-scaling curve.

The required evidence is compact but decisive:

1. **A strong internal comparison.** Beat or explain a tradeoff against the retained GMM80 and retrieval80, not only the low-scoring external ports.
2. **Decision evidence.** Show candidate coverage, outcome-based selection regret, false-improvement rate, and actual closed-loop success.
3. **Mechanism evidence.** Hold proposals fixed while changing the score; compare factual-error adaptation with outcome-trained adaptation. Both receive the same branch data if used.
4. **Compute evidence.** Use nested candidate pools and common total controller allowances. Count encoding, adaptation, world calls, guidance, and rejected timed-out decisions. Current training covers one/two reasoning depths; arbitrary deeper inference is an untested extrapolation.
5. **Independent evaluation.** Freeze the selected mechanism, thresholds, and checkpoint before the reserved evaluation. The 104 development cases have supported 61 current evaluations and numerous design decisions; they are not 6,344 independent test samples.
6. **A recognizable benchmark.** A native visual OGBench-Cube comparison would be especially relevant to LeFlow and AD-WM. The custom 13-task protocol is useful but does not establish standard MT10/MT50 or native LeFlow SOTA.
7. **Flow attribution.** Establish better candidate coverage or repairability than GMM at matched data and actual controller time. If the new evaluator helps GMM equally and flow adds no advantage, frame the contribution as generator-agnostic JEPA planning rather than forcing flow into the central claim.
8. **An honest seed statement.** Current authorization and evidence use seed 3072. Independent case evaluation cannot estimate training-seed variability.

I would develop one mechanism after the same-state diagnostic. If selection is the bottleneck, use execution-conditioned continuation ranking. If the objective is the bottleneck, use MQE-style reachability. If proposal coverage is poor, change the training support for recovery/contact actions. The existing archive does not justify choosing a more sophisticated module before resolving that distinction.

The current evidence supports a useful controlled empirical account of why factual calibration has failed to improve these controllers. A stronger methods paper requires an attributable outcome improvement and a distinct mechanism against the close 2026 references.

