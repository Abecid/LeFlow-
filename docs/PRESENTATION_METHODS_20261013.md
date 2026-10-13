# What our controllers actually do: a teaching brief for the research deck

**Code and evidence audited:** `Abecid/LeFlow-` at the published source snapshot `b2bc294fe5a3de99f195aba90c99481a4852a401` (underlying experiment-code snapshot `d42d76c1cf2f39a56de189148482122775c3962e`). This brief explains the implementation and saved experiments; it introduces no new experiment. Every source link below is pinned to that published commit.

## 1. The distinction the presentation must establish first

The current comparison contains **three internal Gaussian-mixture controllers, nine internal action-flow controllers, two LeFlow latent-path controllers, one HWM hierarchy, and one released CEM controller**. A Gaussian source distribution inside a flow does not make that flow a GMM. Likewise, inheriting a Python class called `RevisionPolicy` does not preserve its GMM output: `FlowReasoningPolicy` explicitly deletes that output head. [Implementation dispatcher][dispatcher] · [Flow replacement][flow-architecture]

The strongest internal control is **Execution revision GMM, 80/104 development successes**. It already has training-trajectory retrieval, a learned multimodal action proposal, frozen-world search, an observed-history workspace, prefix-aware scoring, and previous-plan reuse. The later best flow variants also achieve 80/104. Explaining this strong baseline is essential: the large gap to the external ports is already present before adding action flow or online fast-weight adaptation. [Current results][results]

The phrase “our changes” should refer to identifiable modules or algorithms, not imply that every run adds a module to the previous run. The actual branches are:

- **Internal GMM branch:** short learned action-chunk controller → observed-history revision → execution-aligned scoring and plan reuse.
- **Internal action-flow branch:** replace GMM proposals and CEM refitting with a learned flow and recurrent imagined evidence → add factual fast-weight adaptation → explore guidance or bounded residual repair.
- **Repair-training branches:** expose repair training to retrieved candidates, then optionally to shifted previous plans. Deployment retrieval and plan reuse already existed.
- **Objective/memory branches:** change the progress objective, change ridge memory to a Bayesian posterior, or change it to a nonlinear fast learner. These are alternatives built on retrieval repair.
- **Separate LeFlow branch:** keep latent-path generation and inverse dynamics; optionally add nonlinear memory at its full-chunk interface. This is not the same controller as our direct action-flow branch. [Dispatcher][dispatcher] · [LeFlow implementation][leflow]

## 2. Shared physical task, tensors, and control loop

The current task is image-goal MetaWorld control. The model receives the current RGB image and a target RGB image; simulator state and task identity are not policy inputs. A successful episode can use at most 200 primitive actions and ten cumulative seconds of controller work, including encoding and adaptation. The 104 development cases cover 13 seen tasks, eight cases each, and were reused across the 5k/10k/15k/20k checks. The reserved 3,200-case final evaluation has not supplied a current final-test result. [Protocol][protocol]

| Object | Actual shape | Meaning |
| --- | --- | --- |
| Current or goal image | `[B,256,256,3]` | One RGB observation |
| Encoder input | `[B,3,16,256,256]` | The same image repeated 16 times in the current static representation |
| Encoded state `Z` | `[B,32,1024]` | Frozen V-JEPA 2.1 features pooled to a `[2,4,4]` temporal/spatial grid |
| Primitive command | `[B,4]` | One environment action |
| Action block | `[B,8]` | Two consecutive four-dimensional commands concatenated |
| Internal candidate chunk `A` | `[B,5,8]` | Five blocks, hence ten primitive actions, represented jointly by 40 numbers |
| One decision's candidate pool | `[8,4,5,8]` | Eight local targets, four chunks per target |
| Frozen-world predictions | `[8,4,5,32,1024]` | Predicted spatial state after each of the five blocks |
| Internal executed prefix | `[8]` reshaped to `[2,4]` | Only the first block of the selected chunk is executed before observing again |

The configuration name `action_repeat=2` is easy to misread. Here the cache and evaluator **group two consecutive commands**; they need not be identical. The five-block planning horizon is therefore different from the one-block execution horizon. The external LeFlow, HWM, and CEM ports execute all five blocks per completed decision. [Encoder][encoder] · [State construction][state-clip] · [Action cache][action-cache] · [Execution][execution] · [Baseline configurations][baseline-config]

### What is frozen and what is learned?

The official V-JEPA encoder is frozen. A spatial action-conditioned world model is trained separately on the common data, then hash-verified and frozen for these controller comparisons. Its input is `Z:[B,32,1024]` and `a:[B,8]`. Linear projections map both to width 256; the action vector is broadcast over the 32 tokens; a four-layer, four-head Transformer predicts a residual in the original feature space:

\[
\widehat Z_{t+1}=Z_t+W_o\,T(W_z Z_t+W_a a_t).
\]

Autoregressing this model over five blocks produces the imagined future. It preserves a 32-token spatial state; it does not use the actor's pooled 256-dimensional vector as its physical state. Its training prediction loss is MSE plus mean tokenwise cosine distance. Controller training does not retrain this world, and the internal TTT methods do not update its parameters online. [World model and loss][world] · [Frozen loading][world-freeze]

The internal visual distance is

\[
d(Z,Q)=\frac1{32}\sum_{p=1}^{32}\left[1-\cos(Z_p,Q_p)\right].
\]

This averages aligned patchwise distances. It is not the cosine distance between two globally averaged features, a physical distance, a reward, or a success probability. Features are also standardized with the cached training statistics before use. [Distance][distance] · [Route-state normalization][route-bank]

### Where do eight local targets come from?

A bank stores observations and action chunks from **6,222 successful expert training episodes**. Each stored route has a start, its immediate five-block successor, and a farther endpoint. Retrieval first uses a fixed random projection: each normalized 1024-dimensional token is projected to eight dimensions, then all 32 projected tokens are flattened into a 256-dimensional key. Spatial locations remain distinct. A shortlist of 256 routes is reranked by exact `d(route_start,current)+d(route_endpoint,global_goal)`. The seven distinct immediate successors become local targets; the global goal is the eighth target. Each retrieved target comes with the recorded five-block action chunk that reached it from its source start. [Route-bank construction][bank-build] · [Retrieval algorithm][route-bank]

Thus “retrieval” supplies both **a reachable-in-the-demonstration local image** and **an action proposal**. Whether the chunk is reachable from the current state is only predicted by the frozen world; the demonstration does not prove it will work from the current state. The outer planner adds route cost to the locally optimized response cost, so the final selected route is allowed to differ from the nearest retrieved route. [GMM controller][gmm-controller]

## 3. What exactly is the Gaussian mixture controller?

### Plain-language explanation

It is a neural network that looks at **where the robot is, a nearby visual target, and the eventual goal**, and predicts four plausible distributions over short action sequences. One distribution can put probability around one way of moving; another can represent a different way. The components have no assigned task names or hand-coded meanings. The network learns their weights, their centers, and how broadly to sample around those centers from expert action chunks. World-model search then decides which proposed chunk to execute. [GMM head][gmm-head]

This is a **mixture-density action policy**. It is not a mixture of four V-JEPA encoders, four world models, four retrieved routes, or four task-specific controllers. “Four mixture components” and “four sampled candidates per target” are separate choices, even though both happen to equal four here.

### Network input → output, exactly

1. Current state, local target, and global goal each have shape `[B,32,1024]`.
2. A shared linear projection maps each token from 1024 to 256 dimensions. Three learned type embeddings distinguish current, local-target, and global-goal tokens.
3. Concatenation gives `[B,96,256]`. A three-layer, four-head Transformer with sinusoidal positions processes all 96 tokens, followed by token mean-pooling to `[B,256]`.
4. A linear head emits **324 numbers** per example: `4 × (1 + 40 + 40)`, reshaped to `[B,4,81]`.
5. The result is logits `[B,4]`, means `[B,4,40]`, and standard deviations `[B,4,40]`. Means use `tanh`, and standard deviations use `0.05 + 0.45 sigmoid(raw)`, so the means are bounded by ±1 and scales lie between 0.05 and 0.5. Each component is a diagonal Gaussian in the full 40-dimensional action-chunk space. [GMM head][gmm-head] · [Transformer trunk][trunk]

For a flattened chunk `A∈R^40`, the learned conditional distribution is

\[
p_\theta(A\mid Z,Q,G)=\sum_{k=1}^{4}\pi_k(Z,Q,G)\prod_{j=1}^{40}
\mathcal N(A_j;\mu_{kj}(Z,Q,G),\sigma_{kj}^{2}(Z,Q,G)).
\]

There is no learned full covariance matrix. Coordinates are conditionally independent within a component; mixing components still represents multiple coordinated modes of complete chunks. A component is not separately supervised or manually assigned. Joint maximum likelihood determines component responsibilities. [GMM loss][gmm-loss]

### What is its training loss?

The primary objective is negative log-likelihood of the recorded expert chunk, divided by 40:

\[
L_{\rm NLL}=-\frac1{40}\log\sum_k\pi_k\mathcal N(A^*;\mu_k,\operatorname{diag}\sigma_k^2).
\]

An auxiliary term selects the component with highest posterior responsibility for that recorded chunk, rolls out its **mean** under the frozen world, and asks it to match the observed state after the first block and after the fifth block. The objective adds half of each prediction loss. Its multiplier is zero through step 1000, ramps to 0.1 over steps 1001–2000, then remains 0.1. The selection index is detached; gradients can pass through the frozen world's action input into the selected policy mean. [GMM loss][gmm-loss] · [Weight schedule][training-schedule]

This auxiliary term is an expert-endpoint consistency regularizer. The generated mean was not actually executed, so the expert endpoint is not a measured counterfactual outcome for that generated action. The later internal flow implementations explicitly remove this inherited generated-action consistency term. [GMM loss][gmm-loss] · [Flow loss][flow-loss]

### How is an action selected at inference?

For each of eight targets:

1. Sample four component indices from `softmax(logits)`, with replacement; sample a 40-dimensional Gaussian action chunk from each selected component.
2. Override slot 0 with the **mean of the highest-weight component**. This is neither the weighted mixture mean nor an assertion that the component is the best action.
3. For the seven retrieved targets, replace slot 3 with their recorded expert action chunks. In the original and latent-revision GMM, the direct-goal slot 3 duplicates slot 0; execution revision removes that duplication.
4. Roll all 32 candidates through five frozen-world transitions. The original local score is the average visual distance to the local target after blocks 4 and 5.
5. Keep the two best candidates per target; fit a new diagonal Gaussian to those two elites; floor its standard deviation at 0.05; resample four chunks, preserving the best-so-far chunk in slot 0. Repeat for three total scored rounds.
6. Add each target's retrieval route cost to its best local score, choose the smallest total, and execute **only its first block**, i.e. two primitive actions. Observe again and replan. [GMM sampling][gmm-sampling] · [GMM controller][gmm-controller] · [Aligned initialization][aligned-init]

This uses `8 × 4 × 3 × 5 = 480` predicted world transitions per completed decision. Four GMM components do not imply four separate world models. Four candidates per round do not mean only four candidates are ever considered over the whole decision.

### How is this related to the Cross-Entropy Method (CEM)?

| Learned GMM | CEM-style local refinement | Released CEM comparison |
| --- | --- | --- |
| A conditional density learned offline from expert actions | A temporary search distribution fitted to the best sampled chunks at the current decision | The unchanged stable-worldmodel CEM solver, behind a shared-world interface adapter |
| Four mixture logits, means, and scales depend on images | One diagonal Gaussian per target is refitted from two elites | 300 samples, 30 iterations, 30 elites; five-block horizon |
| Has persistent trained neural parameters | Refitting does not train or mutate network parameters | No extra policy-head training; optimizes normalized actions from its initial Gaussian |
| Seeds a very small informed search | Three scored rounds; best candidate preserved | Endpoint spatial MSE objective; executes all ten primitive actions |

The internal baseline is therefore **learned GMM proposals + retrieved expert proposals + short CEM-style refinement + route selection**. It is not the released CEM baseline with a different name. The baseline comparison also changes execution horizon, candidate budget, scoring objective, and retrieval, so its gap cannot be attributed to the density family alone. [GMM controller][gmm-controller] · [Released CEM][released-cem] · [CEM configuration][cem-config]

**Observed evidence:** original controller-grounded GMM peaks at 79/104; latent-revision GMM at 77; execution-revision GMM at 80. Their selected checks have no timeouts. These are internal controller implementations, not published stand-alone GMM baselines from an external paper. [Current results][results]

## 4. The two GMM revisions: history conditioning versus execution alignment

### Latent revision: add an observed-history workspace and error heads

This keeps the GMM head and the three-round elite-Gaussian search. It adds a learned workspace `[B,4,256]`, four slots initialized from the current/target/goal context. Up to four prior **executed** transitions supply observed-minus-predicted feature residuals, actual motion, actual local progress, predicted local progress, and model error. Projected history adds `4 × 32` tokens to the 96 image tokens. A recurrent cell performs self-attention among workspace slots, cross-attention into that context, and an MLP; the same cell is applied four times at inference. Its pooled thought `[B,256]` modifies the GMM conditioning vector through a learned projection. [Workspace][workspace]

Two new small heads predict bounded signed cost residuals:

- **Prefix head:** `[thought 256, first action block 8, first-block patch costs 32]`, input size 296 → 256 → 1.
- **Terminal head:** `[thought 256, all actions 40, prefix map 32, terminal map 32]`, input size 360 → 256 → 1.

Both use `0.1 tanh(raw)`. The terminal map averages blocks 4 and 5. The prefix head cannot see the unexecuted action suffix. The loss fits observed-minus-predicted factual costs from recorded actions, with target relabeling to demonstration endpoint, global goal, or another training endpoint. Only target images are relabeled; the physical action/outcome pairing is preserved. Shallow-reference and non-regression terms accompany the main calibration loss. [Correction heads][workspace] · [Revision training][revision-loss]

In `latent_revision`, inference adds only the **positive terminal correction** to the old terminal score. This penalizes predicted optimism; negative predicted errors do not reward a candidate. The prefix correction is trained and logged, but does not enter that version's selection objective. No online gradient or ridge fit occurs: history changes the neural forward pass. Calling this version TTT would blur the distinction between memory input and online fitting. [Revision controller][revision-controller]

**Problem addressed:** the world may be overoptimistic and its recent errors may reveal how the current scene differs from the training distribution. **Evidence:** 77/104 is below the 79/104 initial GMM. Adding this workspace did not establish a control improvement.

### Execution revision: change sampling, scoring, and reuse

This uses the same `RevisionPolicy` architecture. It changes the training sampler from the old 60-block start restriction to any feasible five-block window, with a feasible future-goal offset. At deployment it changes the local objective to

\[
C(A,Q)=0.5\,d(\widehat Z_1,Q)+0.25[d(\widehat Z_4,Q)+d(\widehat Z_5,Q)]
+0.5[\widehat e_{\rm prefix}]_++0.5[\widehat e_{\rm terminal}]_+.
\]

That explicitly scores the first block that will actually execute. It also shifts the remaining four blocks of the previous selected plan into slot 1, retaining a freshly generated tail. An eight-event stall memory penalizes a nearby state/target after at least three recent cases in which predicted progress was positive but actual progress was nonpositive. Matching uses cosine radii 0.005; penalties decay by 0.8 and cap at 0.02. [Aligned sampler and initialization][aligned-init] · [Aligned scoring and search][aligned-score] · [Aligned configuration][aligned-config]

**Problem addressed:** mismatch between optimizing the far endpoint and executing only a short prefix; forgetting usable previous plans; repeatedly selecting optimistic stalled routes. **Changed parameters/modules:** no new policy architecture beyond `RevisionPolicy`; new data-window distribution and controller logic. **Evidence:** best 80/104, versus 79 and 77. Multiple things changed together, so one extra success does not identify the causal contribution of prefix scoring, window coverage, warm-starting, or stall memory. [Current results][results]

## 5. Action flow and recurrent proposal revision

### What replaces the GMM?

`FlowReasoningPolicy` removes the 324-output mixture head. It uses an `ActionFlow` to map Gaussian noise to a complete action chunk. The action-flow network receives a noisy `[B,N,5,8]` chunk, scalar flow time, and a context vector; it flattens the 40 action coordinates, projects them to 256, adds time/context embeddings, passes through three residual MLP blocks, and predicts a 40-coordinate **velocity** reshaped to the original chunk. The blocks are LayerNorm → 256-to-512 linear → SiLU → 512-to-256 linear with residual addition. This action-velocity network is an MLP over the whole chunk; it is not a Transformer over separate action tokens. The upstream visual context and workspace still use attention. [Action-flow network][flow-architecture]

For a recorded chunk `A*`, independent standard Gaussian noise `ε`, and uniform flow time `τ`:

\[
X_\tau=(1-\tau)\epsilon+\tau A^*,\qquad
L_{\rm flow}=\mathbb E\|v_\theta(X_\tau,\tau,c)-(A^*-\epsilon)\|^2.
\]

Inference begins with full Gaussian noise and applies eight Euler updates `X ← X + v(X,i/8,c)/8`. Only the final clean proposal is clipped to action bounds. This integrates a learned vector field; it does not sample component labels or predict mixture variances. [Sampling][flow-sampling] · [Flow loss][flow-loss]

### What is “reasoning” in this implementation?

The controller prepares a causal workspace from the current images and observed history. After generating and world-scoring four candidates per target, each candidate becomes one evidence token containing **40 action coordinates + 160 patch costs across five predicted states + two corrections + one score = 203 values**. These values are detached from world prediction, projected to width 256, and added to the workspace's attention context. A recurrent update changes the next proposal context. Each action stays attached to its predicted consequence; the candidate set does not have an artificial positional ordering. [Flow revision][flow-revision]

`flow_reasoning` uses three generate–predict–revise rounds. It reuses the same initial Gaussian noise across rounds, so proposal changes are driven by changed evidence/context rather than merely a different noise draw. It preserves the best candidate separately. The initial causal scoring thought remains fixed throughout the search, even though the generation thought changes; this avoids moving the scoring criterion each round. The initial pool includes retrieved and shifted-plan proposals through the inherited controller, but the original flow training deliberation does not include those deployment slots. [Flow controller][flow-controller]

The loss averages flow errors over the trained depths and adds a 0.25-weight non-regression term relative to the first context, plus 0.1 factual calibration loss. Training depths cycle 1, 2, 3 across optimizer updates. Candidate outcomes used as imagined evidence are frozen-world predictions, not real counterfactual outcome labels. The inherited GMM endpoint-consistency objective is zero for all internal flow variants. [Flow loss][flow-loss] · [Training schedule][training-schedule]

**Changed modules:** GMM output removed; initial action flow and a 203-to-256 evidence encoder added; workspace now revises from candidate consequences. **Problem addressed:** spend test-time computation generating a more informed next proposal, instead of refitting a Gaussian to elites. **Evidence:** best 79/104, ten timeouts, 99.42 ms mean decision time. It does not exceed execution GMM80. Three trained/evaluated rounds are a fixed setting; this run is not a measured success-versus-inference-budget scaling curve. [Current results][results] · [Timing][timing]

## 6. Factual ridge TTT: online fitting in the cost-error interface

### What is adapted?

`progress_ttt` inherits action-flow planning but replaces the learned prefix-error head with a small **target-conditioned fast-weight regression**. It does not update the visual encoder, world model, Transformer, or flow weights online. The offline-trained feature network and learned prior remain fixed; an ephemeral 32-dimensional weight vector is fitted separately for each current target from up to four completed transitions in this episode. Fits reset at the episode boundary. [Ridge memory][ridge-memory] · [Factual support controller][ridge-controller]

For each spatial patch, the key network receives normalized prior state, frozen-world predicted next state, local target, the actually executed eight-dimensional action block, and a 16-dimensional positional code. Input width is `3×1024+8+16=3096`; an MLP maps 3096 → 64 → 31, adds a constant coordinate, then normalizes to a unit 32-dimensional key. The label is

\[
y_{ip}=\frac{d_p(Z^{\rm observed}_{i+1},Q)-d_p(\widehat Z_{i+1},Q)}{0.02}.
\]

For the current target `Q`, the fit solves

\[
w_Q=\arg\min_w\sum_{i,p}\frac{m_i}{32}(x_{ip}^{\top}w-y_{ip})^2+0.125\|w-w_0\|^2.
\]

The code uses a batched Cholesky solve about learned prior `w0`. Four transitions yield at most 128 patch rows but only **four physical interventions**, not 128 independent robot experiences. Invalid padded history is zeroed before feature nonlinearities. The fit is differentiable during offline outer training. [Ridge solve][ridge-solve] · [Key/label interface][ridge-memory]

| Ridge-fit object | Shape | Interpretation |
| --- | --- | --- |
| Historical current / predicted / observed states | `[B,4,32,1024]` each | Four prior blocks, each with spatial features |
| Historical actions | `[B,4,8]` | Exactly the two-command blocks that were executed |
| Learned patch keys before flattening | `[B,4,32,32]` | History × patch × key width |
| Design matrix `Φ` | `[B,128,32]` | Last two row axes flattened; masked when fewer transitions exist |
| Residual targets `y` | `[B,128]` | Signed patch goal-cost errors divided by 0.02 |
| Gram system | `[B,32,32]` | `ΦᵀΦ/32 + 0.125 I`, with masked rows |
| Fitted weights `w` | `[B,32]` | One regression per requested target |

At a single physical deployment decision, this batch axis is **eight different local targets for the same observed history**, not eight additional environment rollouts. Each target defines different error labels and therefore a different fitted correction function. The world rollouts for proposed actions remain separate imagined computations. [Controller context batching][flow-controller] · [Key/label interface][ridge-memory]

The fitted change `w−w0` projects from 32 to 256 and modifies the workspace used for action generation. For candidate ranking, the same keys use only the first action block and its predicted state; a bounded patch correction is `0.1 tanh(0.02 xᵀw / 0.1)`. The patch mean replaces the prefix correction. Only positive correction enters the selection cost; the inherited terminal-error head remains. Fast weights are fitted once per decision and remain fixed throughout its search. [TTT conditioning and scoring][ridge-scoring]

### How is it trained to adapt?

The offline outer objective differentiates through a fit on **earlier recorded transitions** and supervises the **next recorded action** and its observed prefix/terminal cost errors. It retains flow imitation and the non-regression term. The current query's outcome is excluded from support. At deployment, a proposed block enters memory only after the next actual observation arrives. Imagined candidate outcomes never become support labels. [TTT outer loss][ridge-loss] · [Factual support controller][ridge-controller]

**Problem addressed:** correct local model optimism from new experience cheaply, and let the corrected information influence both proposals and their score. **Evidence:** best 78/104, 21 timeouts, 102.50 ms mean. For the same executed prefixes at that selected check, adapted MAE is 0.001656 versus raw 0.002547 and prior 0.002237. Prediction calibration improved; a separate adaptation-on/off control rollout was not run, so this does not establish that fitting caused better actions or higher success. [Calibration][calibration] · [Timing][timing]

## 7. Two ways to use extra computation: guidance versus bounded repair

### Guidance: differentiate a cheap action-sensitive prefix energy

`guided_ttt` replaces the original TTT key network with an action-sensitive energy model. State/target/difference plus patch position gives 3088 inputs → 64 cached state features. An 8-to-64 action projection is added, followed by a small nonlinear feature network. Separate heads predict a base prefix cost and 32-dimensional fast keys. These keys no longer need a predicted future state, which makes energy evaluation cheaper than repeatedly differentiating the full world. [Guidance architecture][guidance-architecture]

The base energy is trained to imitate the frozen world's prefix patch costs **on recorded actions**; the fitted residual predicts observed-minus-predicted errors. A prequential trust heuristic fits earlier support, predicts the most recent observed residual, and compares with the learned prior. A positive relative MSE improvement gates the full-history adaptation; with fewer than two transitions, trust is zero. This is a reliability heuristic, not calibrated uncertainty. [Guidance trust][guidance-trust]

During the last four of eight Euler steps, half of each four-candidate pool is guided. The clean estimate is `Â = clip(Xτ+(1−τ)vτ)`. The action gradient of mean energy is evaluated at its **first block only**, detached from the velocity/ODE chain (identity-Jacobian approximation). A strength-1 drift subtracts the gradient from that block's velocity, with gradient RMS capped at 0.25. The other two candidates receive ordinary flow generation. Two reasoning rounds are used, so this run spends 16 flow evaluations plus eight guided gradient evaluations, rather than the 24 flow evaluations of three-round `progress_ttt`. [Guided sampler][guidance-sampler] · [Guidance config][guidance-config]

Additional offline losses are 0.1 base-energy distillation, 0.25 guided-action imitation, 0.25 adapted-versus-prior guided non-regression, and 0.01 expert-action gradient penalty. The guided imitation target is the recorded **action**, not an invented physical outcome for a new action. The energy is not a learned Q-function: there is no Bellman target, IQL objective, or success-label supervision here. [Guided losses][guidance-loss]

**Problem addressed:** steer generation toward a useful prefix before paying for full-world verification. **Evidence:** 75/104 selected, 29 timeouts, 103.46 ms mean; same-prefix adapted MAE 0.001632 versus prior 0.001773. This implementation did not turn its predictive correction into a better completed-task result. Comparing it with repair also changes solver budget and training losses, not guidance alone. [Results][results] · [Calibration][calibration]

### Bounded repair: a second flow proposes neighbors of a verified incumbent

`repair_flow_ttt` branches from ridge TTT, not from the guided energy model. After eight initial-flow steps and one full-world scoring round, it selects the lowest-cost candidate per target as incumbent `A0`. A new 40-to-256 anchor encoder and a separate three-block `ActionFlow` generate residuals conditioned on that incumbent and the revised workspace. Slot 0 preserves the incumbent exactly. Three other slots use four Euler steps and produce

\[
A'=\operatorname{clip}\left(A_0+0.25\operatorname{clip}(U,-1,1),-1,1\right).
\]

The coordinatewise action change is at most 0.25. The full world scores all four candidates again, under the same causal fast weights and scoring thought. The ordinary repair controller re-evaluates the incumbent, giving 320 predicted world transitions and 12 flow evaluations per completed decision. It makes no surrogate-gradient calls. [Repair architecture/sampling][repair-sampling] · [Controller][flow-controller]

Training uses the bounded expert residual `U*=clip((A*−A0)/0.25,−1,1)` as an action-space flow-matching target. The full objective is initial flow loss + residual flow loss + 0.1 factual calibration loss. The generated incumbent and its world score are detached evidence; repair directions are supervised by recorded action differences. The residual limit can truncate a large correction required to reach an expert chunk. Training alternates depth 1 and 2. [Repair loss][repair-loss] · [Training schedule][training-schedule]

**Problem addressed:** replace repeated global generation or gradient guidance with cheap local edits to a strong existing chunk. **Changed modules:** one anchor encoder and a separate residual-flow network. **Evidence:** 80/104 at 5k, no timeouts, 86.46 ms mean, then 73/75/75. It ties GMM80, exchanging six wins and six regressions. Retaining the incumbent guarantees that the previous **model score** remains available; it does not guarantee preservation or improvement of actual task success. [Results][results] · [Paired cases][paired]

## 8. Retrieval and shifted-plan training: exposing the repair model to its deployed pool

### Retrieval repair changes the training pool

All internal controllers already retrieve seven targets and their recorded action chunks at deployment. The new change in `retrieval_repair_ttt` is that offline repair training now also constructs a first-round pool containing **three generated candidates plus one retrieved action chunk**, with a deterministic 1/8 direct-goal fraction where no retrieved slot is inserted. The retrieval query uses current state and requested target, excludes the current demonstration's episode, and searches five-block routes from other successful training episodes. Shortlisting is 256 followed by exact spatial-distance reranking. [Training retrieval][retrieval-training]

The inherited repair flow is trained around the winner of this mixed pool, rather than only winners from generated candidates. No new learned parameter module is introduced by the retrieval subclass; it changes the evidence/anchor distribution used by training and records candidate provenance. The recorded expert chunk still supplies action imitation labels; the retrieved candidate's actual outcome from the query state is unknown. [Training pool][retrieval-pool]

**Problem addressed:** generated-only training anchors differ from the retrieved incumbents the repair policy sees at deployment. **Evidence:** 75 → 77 → 79 → 80/104, selected 20k, no timeouts, 85.72 ms mean. It ties GMM80 with six wins and six regressions and ties repair80 with seven wins and seven regressions. It has better later-checkpoint results than basic repair, but the reused development curve is not an isolated causal estimate of retrieval exposure. [Results][results] · [Paired cases][paired]

### Shifted-plan repair changes the training pool again

Execution revision and all internal flow descendants already reuse the previous selected plan at deployment: drop the executed block, reuse the four remaining blocks, and fill the fifth from a fresh proposal. Retrieval repair did not expose its training to that slot. `shifted_repair_ttt` adds it offline. [Aligned initialization][aligned-init] · [Flow controller][flow-controller]

At the preceding **recorded** state, using only history ending there, the current policy synthesizes a two-round generated-plus-retrieval plan. Its suffix becomes slot 1 at the current recorded state. On eligible examples the current training pool is two generated + one shifted + one retrieved. The previous rollout is detached and nonrecursive. It uses the same requested local/global targets as the current training example; it does not reproduce deployment's full previous eight-target selection or the on-policy state distribution. On the first decision there is no shifted slot. [Shifted data][shifted-data] · [Shifted planning and training][shifted-training]

**Changed parameters/modules:** no new learned module or deployment algorithm; `ShiftedRepairController` is an alias of `RetrievalRepairController`. **Problem addressed:** offline/deployment mismatch in reused-plan incumbents. **Evidence:** 79/78/74/78, best 79, no timeouts, 85.74 ms. This is a plausible training-distribution correction that has not exceeded retrieval80. [Shifted deployment identity][shifted-training] · [Results][results]

## 9. Other axes explored on retrieval repair

### A. Goal-progress objective: replace part of the semantic target, not the dynamics

`progress_objective_ttt` adds a goal-conditioned scalar `GoalProgress` readout. State and global-goal tokens are each normalized and projected 1024→32. Their concatenated projected state, goal, and absolute difference preserve all 32 locations, producing 3072 inputs to a 128-hidden MLP and sigmoid output `[B]`. The scalar estimates normalized expert time remaining, not success probability. [Progress readout][progress-readout]

Training labels are capped remaining blocks divided by 60, using expert trajectories up to their first success; separated early/later states receive regression and temporal-order losses. Hindsight current/local/global triplets also supply time-offset labels. A second 32-dimensional ridge fit calibrates observed-minus-predicted changes in this learned progress value, with one pooled key per actual transition. Its learned prior and 32-to-256 conditioning projection are separate from the inherited patch-cosine TTT. Calibration labels stop gradients through the progress readout to discourage flattening the objective just to erase its residuals. [Progress data and readout][progress-readout] · [Progress calibration][progress-fit]

The candidate cost adds weight 0.05 times `0.5 progress(Ẑ1,G)+0.25 progress(Ẑ4,G)+0.25 progress(Ẑ5,G)`, plus half the positive prefix correction. All local targets use the same **global goal** for this term. Route scoring adds a stage penalty for backward progress or an implausibly large jump beyond the five-block local horizon. [Progress objective/controller][progress-score]

**Problem addressed:** nearest latent-image targets may reward locally easy movement without completing the intended task stage. **Changed modules:** scalar progress readout, second learned prior/conditioning projection, temporal training data/objective, route/candidate cost terms. **Evidence:** 72/72/78/77, best 78, no timeouts, 93.24 ms. Its progress-proxy errors have different units from cosine-cost errors; selected corrected proxy MAE 0.016436 versus raw 0.016240 is not a cosine MAE or a demonstrated improvement in task semantics. [Results][results] · [Progress results][progress-results]

### B. Bayesian repair: quantify uncertainty in the executed-prefix cost error

`bayesian_repair_ttt` replaces fixed-ridge fitting with learned-feature Bayesian linear regression in the ALPaCA form. **One executed transition is one likelihood observation**: the 32 patch features and signed errors are averaged before fitting, giving `Φ:[B,4,32]` and `y:[B,4]`. In addition to the inherited 32-dimensional mean prior, it learns a full lower-triangular 32×32 prior precision factor and a scalar observation-noise variance. The posterior mean `[B,32]` and Cholesky factor `[B,32,32]` are solved exactly; its 32-dimensional mean change still conditions the proposal workspace. [Bayesian posterior][bayesian-posterior]

Every candidate is evaluated under the same 64 symmetric posterior cubature weight points. The bounded positive prefix penalties are averaged for its deployed cost. Sharing posterior samples preserves correlation when comparing a candidate to its incumbent; sampling independent candidate posteriors would invent uncertainty even for an action compared with itself. The bounded nonlinear moments use cubature, not an exact Gaussian transformation. [Posterior comparison math][bayesian-math] · [Bayesian cost][bayesian-cost]

In a repair round, a candidate can replace slot 0 only if its estimated cost improvement minus `1.644854 × std(improvement)` is positive. This uncertainty only describes the learned prefix-cost residual. It excludes global task semantics, terminal-model bias, and other sources of control failure. The number 1.644854 does not make the decision a certified 90% safe action improvement. [Acceptance rule][bayesian-math] · [Controller uncertainty scope][bayesian-controller]

Training covers depths 1/2/3, while the retained evaluation uses two rounds. Residuals are bounded relative to the original incumbent across every round. The outer calibration includes Gaussian NLL, a loss for the actually deployed bounded positive penalty, and terminal residual fitting. The controller caches the incumbent's full world rollout, reducing two-round world transitions from 320 to **280**; eight initial plus four repair flow steps remain. [Bayesian training][bayesian-training] · [Incumbent caching][bayesian-controller]

**Problem addressed:** accept a repair according to uncertainty in its improvement over a known incumbent, rather than the smallest point estimate alone. **Evidence:** 71/76/77/77, earliest selected 15k, no timeouts, 88.63 ms. Nominal 90% coverage is 83.45% on selected-policy trajectories versus 90.79% in the train-only check. The four large Bayesian reports were inspected via saved compact analysis and chunk manifests, not independently reconstructed in the full raw recount. [Results][results] · [Bayesian coverage][calibration] · [Audit scope][audit-scope]

### C. Nonlinear memory: adapt a small function rather than a linear readout

`nonlinear_repair_ttt` transfers the gated memory update used in our direct LeFlow-TTT branch to the internal **two-action goal-cost error** interface. It keeps retrieval repair, frozen-world scoring, and its initial/repair flows. It removes the 32-vector ridge prior and adds three learned fast matrices: `W0:[32,64]`, `W1:[64,1]`, `W2:[32,64]`. The readout is

\[
M(k)=\big[\operatorname{SiLU}(kW_0)\odot(kW_2)\big]W_1.
\]

The inherited 32-dimensional unit keys are multiplied by √32. Up to four completed transitions provide keys `[B,128,32]` and scalar residual targets `[B,128,1]`, with the same factual cosine-cost residual target as ridge TTT. Four explicit MSE-gradient updates adapt ephemeral matrix copies, with total gradient norm capped at 5, a learned learning rate bounded by 0.25 and initialized to 0.05, and elastic anchoring to the learned initial matrices. Offline training differentiates through all four updates. [Memory function/gradients][memory-math] · [Nonlinear fitting][nonlinear-fit]

To condition proposals, the fitted-versus-prior function difference is evaluated at a synthetic current-state query with zero action and predicted state equal to current state. It produces 32 patch readouts, projected 32→256 into the workspace. This query has no outcome label. For candidate scoring the memory instead reads the candidate's actual first-block action and its frozen-world prediction, and outputs the same bounded patch-cost correction used by ridge repair. [Nonlinear conditioning/scoring][nonlinear-score]

**Changed modules:** ridge prior/closed-form solver replaced by three-matrix fast memory and four gradient steps; workspace conditioning uses change in function values rather than a 32-vector parameter difference. **Problem addressed:** linear residual fitting may be too restrictive for scene-dependent error structure. **Evidence:** 73/66/67/72, selected 5k, no timeouts, 93.24 ms. At 20k raw→adapted same-prefix MAE improves 0.002935→0.001833 (37.56%), yet success is 72. At the selected check its learned prior is worse than raw, so adaptation partly repairs a poor prior. It solves 8/8 reach but 0/8 assembly, faucet-open, and pick-place; this is a capability trade, not overall progress. Four transitions are not a demonstration of large-context scaling. [Calibration][calibration] · [Task outcomes][paired]

## 10. The separate latent-path and search baselines

### LeFlow release-module spatial port

The spatial adapter maps `[B,32,1024]` into one compact `[B,512]` state through positioned patch tokens plus a summary token and a two-layer Transformer. A training-only decoder reconstructs the spatial input; the adapter is trained first and frozen for flow/inverse training. The code discloses that this decoder is an implementation choice because the paper's decoder was not released. This is an adapted port, not an unchanged native benchmark result. [LeFlow adapter][leflow]

A latent-path flow generates **four intermediate 512-dimensional latent states** between encoded start and goal, with five transitions total. At inference 64 candidates use 16 Euler steps, forming paths `[64,6,512]`. An inverse-dynamics MLP turns each consecutive latent pair into an eight-dimensional action block, giving `[64,5,8]`. The frozen spatial world rolls out those actions; endpoint MSE to the original spatial goal ranks candidates. All ten primitive actions execute. There is no route bank, internal four-slot workspace, or short action-CEM refinement. [LeFlow sampling and controller][leflow-sampling] · [LeFlow config][baseline-config]

Its loss is latent-path flow matching + standardized inverse-action MSE + 0.1 frozen-world compact-state consistency on **recorded paths**. **Evidence:** 14/21/19/14, best 21, no timeouts. The lower result cannot be interpreted as proof that action flow is intrinsically better than latent flow: the internal pipeline also changes retrieval, hierarchy, scoring, control frequency, learned history, and proposal refinement. [LeFlow loss][leflow] · [Results][results]

### Direct LeFlow plus nonlinear TTT

This retains that latent-path and inverse-dynamics architecture. Its fast memory keys use prior spatial state, predicted full-chunk endpoint, the complete 40-dimensional action chunk, and patch position. The value is a learned **32-dimensional projection of the full 1024-dimensional state prediction residual**, not the scalar local-goal cost used by internal repair. Memory matrices are `[32,64]`, `[64,32]`, `[32,64]`; up to four completed ten-action chunks support four updates. A 32→1024 output corrects the spatial endpoint, and additional 32→512 projections condition latent-flow generation and inverse dynamics. [Direct LeFlow memory][direct-memory] · [Direct LeFlow adaptation][direct-adaptation]

Its outer loss adds full-endpoint prediction and residual-reconstruction objectives to the inherited flow/inverse/consistency losses. Candidate selection uses corrected endpoint MSE. At the simulator boundary actions are clipped; factual support records the clipped actually executed actions and recomputes their world prediction. Model parameters remain unchanged online; only episode-local fast tensors are fitted. [Direct LeFlow loss][direct-adaptation] · [Direct LeFlow controller][direct-controller]

**Evidence:** 27/21/27/29, selected 29, no timeouts, 104.19 ms per ten-action decision. It improves over the LeFlow port's selected 21 with 12 paired wins and four regressions, but consumes more optimization compute, and no adaptation-only rollout ablation isolates the cause. It still remains far below the stronger internal controllers. Its ten-action endpoint MSE is not comparable numerically with the internal two-action cosine-cost MAE. [Results and direct-TTT comparison][results]

### HWM paper-based spatial port

This is explicitly a paper-based port, not released HWM robotics source. A two-layer action-sequence Transformer maps variable primitive-action segments into four-dimensional macro actions. A 10-layer, 12-head DINO-WM-style predictor at width 768 predicts coarse spatial waypoints, with input/output projections for the common 1024-dimensional features. Training uses L1 teacher-forced waypoint prediction. [HWM model][hwm-model]

High-level CEM searches four macro steps (150 candidates, ten iterations, ten elites, 0.4 standard-deviation momentum). The first state predicted under its optimized mean plan becomes the subgoal. Low-level CEM searches five action blocks under the common frozen world (150 candidates, ten iterations, ten elites, zero momentum), minimizing L1 spatial endpoint error. All ten primitive actions execute. It does not receive the internal route bank or controller corrections. [HWM controller][hwm-controller] · [HWM configuration][hwm-config]

**Evidence:** 2/3/5/5, selected 5; all 99 failed selected episodes time out, mean reported decision time 3,537.46 ms. This is evidence about this port under the strict common episode clock, not unrestricted HWM capability. [Results][results] · [Timing][timing]

### Released CEM on the common world

No learned action proposal or extra policy head is added. The stable-worldmodel 0.0.6 solver is hash-checked, optimizes a five-block normalized action plan with 300 candidates ×30 iterations, scores original spatial endpoint MSE, and executes all five blocks. **Evidence:** 24/104 with 80 timeouts; all failed episodes time out, mean decision 1,011.75 ms. Its larger open-loop action search makes an especially different comparison from the internal retrieved GMM's 32 candidates and two-action replanning. [Released CEM][released-cem] · [CEM config][cem-config] · [Timing][timing]

## 11. Exact 16-method mapping and observed outcomes

The numerical table below is generated from the saved audit data. “Selected” means highest success on repeatedly reused development cases, earliest tie. Timeouts are episode counts at that same selected checkpoint. Mean milliseconds include controller work, not simulation/rendering. Mean time is descriptive across visited decisions; it is not a hard per-decision cap. The GMM prefix and flow examples above explain what each label means.

### Implementation mapping

| Method | Family | Change relative to its parent | Fitted online | Complete decision |
| --- | --- | --- | --- | --- |
| [`controller_grounded`][gmm-head] | GMM | 4 logits + 4×40 means + 4×40 scales | None | 3 elite-search rounds; 480 world transitions; 2 actions |
| [`latent_revision`][workspace] | GMM | Same mixture head + observed-history workspace + error heads | None; history is forward-pass context | 4 workspace passes, 3 elite rounds; 480 world transitions; 2 actions |
| [`execution_revision`][aligned-score] | GMM | Same RevisionPolicy; new prefix objective, windows, stalls, plan reuse | None; history is forward-pass context | 4 workspace passes, 3 elite rounds; 480 world transitions; 2 actions |
| [`flow_reasoning`][flow-architecture] | Action flow | GMM head removed; 40-D velocity network and candidate-evidence encoder | None | 3×8 flow steps; 480 world transitions; 2 actions |
| [`progress_ttt`][ridge-memory] | Action flow | Learned patch keys replace prefix MLP; learned fast prior/conditioning | 32-vector ridge weights for each target | 3×8 flow steps; 480 world transitions; 2 actions |
| [`guided_ttt`][guidance-sampler] | Action flow | Action-sensitive energy features, trust gating and clean-prefix gradient | 32-vector gated ridge weights; two logical fits | 2×8 flow steps + 8 gradient calls; 320 world transitions; 2 actions |
| [`repair_flow_ttt`][repair-sampling] | Action + residual flows | New anchor encoder and separate bounded residual flow | 32-vector ridge weights | 8 initial +4 repair steps; 320 world transitions; 2 actions |
| [`retrieval_repair_ttt`][retrieval-pool] | Action + residual flows | No new learned module; mixed generated/retrieved offline training pool | Same ridge weights | 8+4 flow steps; 320 world transitions; 2 actions |
| [`shifted_repair_ttt`][shifted-training] | Action + residual flows | No new learned module; add causal previous-plan slot during training | Same ridge weights | Same as retrieval repair; 320 world transitions; 2 actions |
| [`progress_objective_ttt`][progress-score] | Action + residual flows | Global-goal time-to-go readout, second prior/conditioning, scoring terms | Patch ridge + separate scalar-progress ridge | 8+4 flow steps; 320 world transitions; 2 actions |
| [`bayesian_repair_ttt`][bayesian-controller] | Action + residual flows | Learned Bayesian prior precision/noise, shared-posterior acceptance | 32-D posterior mean +32×32 Cholesky factor | 8+4 flow steps; 280 world transitions through incumbent caching; 2 actions |
| [`nonlinear_repair_ttt`][nonlinear-fit] | Action + residual flows | Replace ridge readout with gated three-matrix memory | 32×64, 64×1, 32×64 fast tensors, four updates | 8+4 flow steps; 320 world transitions; 2 actions |
| [`leflow_ttt`][direct-adaptation] | Latent-path flow + inverse dynamics | Add full-chunk residual memory and flow/inverse conditioning | 32×64, 64×32, 32×64 fast tensors, four updates | 64 paths ×16 flow steps; 5-block execution; 10 actions |
| [`leflow_release`][leflow-sampling] | Latent-path flow + inverse dynamics | Spatial adapter; released latent-flow/inverse modules ported to dimensions | None | 64 paths ×16 flow steps; 5-block execution; 10 actions |
| [`hwm_paper`][hwm-controller] | Hierarchical world-model + CEM | Paper-based macro action encoder and coarse predictor | None | 150×10 high CEM and 150×10 low CEM; 10 actions |
| [`cem_release`][released-cem] | Action-space CEM | Shared-world interface to released solver; no policy head | None | 300 samples×30 CEM iterations; 10 actions |

### Exact current development ledger

| Method | Successes at 5k /10k /15k /20k | Selected update | Selected successes /104 | Timeouts /104 | Mean decision ms |
| --- | --- | ---: | ---: | ---: | ---: |
| [controller_grounded](https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/CONTROLLER_GROUNDED_RESULTS.md) | 79 / 78 / 75 / 78 | 5,000 | 79 | 0 | 81.42 |
| [latent_revision](https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/LATENT_REVISION_RESULTS.md) | 73 / 75 / 74 / 77 | 20,000 | 77 | 0 | 85.21 |
| [execution_revision](https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/EXECUTION_REVISION_RESULTS.md) | 80 / 78 / 79 / 76 | 5,000 | 80 | 0 | 86.74 |
| [flow_reasoning](https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/FLOW_REASONING_RESULTS.md) | 79 / 75 / 76 / 76 | 5,000 | 79 | 10 | 99.42 |
| [progress_ttt](https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/PROGRESS_TTT_RESULTS.md) | 77 / 76 / 78 / 74 | 15,000 | 78 | 21 | 102.50 |
| [guided_ttt](https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/GUIDED_TTT_RESULTS.md) | 72 / 70 / 72 / 75 | 20,000 | 75 | 29 | 103.46 |
| [repair_flow_ttt](https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/REPAIR_FLOW_TTT_RESULTS.md) | 80 / 73 / 75 / 75 | 5,000 | 80 | 0 | 86.46 |
| [retrieval_repair_ttt](https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/RETRIEVAL_REPAIR_TTT_RESULTS.md) | 75 / 77 / 79 / 80 | 20,000 | 80 | 0 | 85.72 |
| [shifted_repair_ttt](https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/SHIFTED_REPAIR_TTT_RESULTS.md) | 79 / 78 / 74 / 78 | 5,000 | 79 | 0 | 85.74 |
| [progress_objective_ttt](https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/PROGRESS_OBJECTIVE_TTT_RESULTS.md) | 72 / 72 / 78 / 77 | 15,000 | 78 | 0 | 93.24 |
| [bayesian_repair_ttt](https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/BAYESIAN_REPAIR_TTT_RESULTS.md) | 71 / 76 / 77 / 77 | 15,000 | 77 | 0 | 88.63 |
| [nonlinear_repair_ttt](https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/NONLINEAR_REPAIR_TTT_RESULTS.md) | 73 / 66 / 67 / 72 | 5,000 | 73 | 0 | 93.24 |
| [leflow_ttt](https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/LEFLOW_TTT_RESULTS.md) | 27 / 21 / 27 / 29 | 20,000 | 29 | 0 | 104.19 |
| [leflow_release](https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/RELEASE_BASELINE_RESULTS.md) | 14 / 21 / 19 / 14 | 10,000 | 21 | 0 | 91.11 |
| [hwm_paper](https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/RELEASE_BASELINE_RESULTS.md) | 2 / 3 / 5 / 5 | 15,000 | 5 | 99 | 3537.46 |
| [cem_release](https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/RELEASE_BASELINE_RESULTS.md) | 24 | No policy training | 24 | 80 | 1011.75 |

Source: the source reports linked in each row, independently recounted audit records where available, and [the pinned derived audit](https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/reports/20261012-research-positioning/audit.json). The four Bayesian checkpoints use manifest/compact-analysis evidence. No values in this table are from the sealed final test.

## 12. What the evidence establishes, and what each diagram should avoid implying

1. **The strongest existing control is the internal execution GMM.** The 80/104 tie cannot be presented as a flow or TTT improvement over that control. The original GMM79 already explains most of the gap to the external ports. [Results][results]
2. **Adaptation can improve a factual prediction without proving a better choice among actions.** Raw/prior/adapted MAE compares different predictions for the same actions actually taken by one policy. Different policies and checkpoints visit different states. Do not graph their MAEs as though they predict a common fixed test set, or equate a percentage MAE reduction with percentage success improvement. [Calibration][calibration]
3. **Only a short executed prefix is physically labeled in the internal controllers.** The remaining eight commands in a proposed chunk are imagined and normally discarded or reused after a fresh observation. Calling the whole ten-action chunk “verified” must mean verified by the model, not observed in the environment. [Execution][execution] · [GMM diagnostics][gmm-controller]
4. **The common deadline matters.** With two actions per decision, using the whole 200-action allowance requires at most 100 decisions, so ten seconds corresponds to a 100 ms average allowance along a full-length episode. It is not a 100 ms timeout applied to every decision. For the ten-action external ports, the corresponding arithmetic is 20 decisions and 500 ms average. Early success, episode-to-episode variation, and the cumulative budget explain why an overall mean does not determine timeout count. [Execution budget][execution] · [Timing][timing]
5. **Incumbent preservation is surrogate preservation.** A retained initial action can have a lower frozen-world score and still be physically worse, and a lower predicted local-target cost need not imply eventual task success. Bayesian uncertainty covers only one error interface. [Repair sampling][repair-sampling] · [Bayesian scope][bayesian-controller]
6. **Complementarity is real at whole-episode level, but its exploitability is unknown.** The three selected 80-success controllers solve 89 distinct cases in their union, 70 shared; adding nonlinear73 gives 92 distinct, 64 shared. This oracle retrospectively chooses an entire completed rollout per case. It is not a candidate oracle, deployed ensemble, or attainable router result. [Paired cases and oracle scope][paired]
7. **Reduced diversity is not a demonstrated single cause.** Retrieval repair improves 75→80 while initial candidate spread contracts 62.03%. Spread of four generated samples is not a density entropy estimate. [Diversity evidence][diversity]
8. **The reused development checks are not a final generalization claim.** Single-seed, repeatedly selected curves and mismatched consumed compute cannot establish general superiority or standard MT10/MT50 SOTA. Older latent-flow/state-rank and representation defects changed the world and representation; historical curves must remain in a separate section. [Protocol][protocol] · [Historical separation][history]

## 13. Suggested teaching order and names for the rebuilt deck

Begin with the physical loop and show the actual tensors. Then introduce the learned GMM, followed by one numeric example of proposal → five imagined states → prefix/terminal score → first two commands executed. Only then show which modules are swapped by flow, memory, guidance, and repair. Results should follow these definitions.

| Code label | Audience-facing name | One sentence that should appear before showing its result |
| --- | --- | --- |
| `controller_grounded` | GMM action proposals with local search | Predict four Gaussian modes over ten-action chunks, test a small mixed generated/retrieved pool in the world model, execute two actions. |
| `latent_revision` | GMM with observed-error context | Use the last four outcomes to condition proposals and predict the world's cost bias. |
| `execution_revision` | GMM with prefix scoring and plan reuse | Score the part that executes, reuse the previous suffix, and avoid repeatedly stalled routes. |
| `flow_reasoning` | Flow action proposals with repeated model feedback | Replace the GMM and elite-Gaussian refit with a flow conditioned on candidate consequences. |
| `progress_ttt` | Flow with online ridge error correction | Fit a small cost-error function to this episode's executed transitions. |
| `guided_ttt` | Flow guided by a fitted prefix energy | Use the fitted cost gradient to steer the first action block during generation. |
| `repair_flow_ttt` | Flow that edits a selected action chunk | Generate three bounded residual edits around a world-scored incumbent and keep the incumbent available. |
| `retrieval_repair_ttt` | Repair trained with retrieved proposals | Expose offline repair training to the retrieved action slot already used at deployment. |
| `shifted_repair_ttt` | Repair trained with reused-plan proposals | Also expose offline training to a causally synthesized previous-plan suffix. |
| `progress_objective_ttt` | Repair with learned task-stage progress | Add a global-goal time-to-go proxy and factual correction to candidate and route costs. |
| `bayesian_repair_ttt` | Repair with uncertainty-aware acceptance | Replace fixed ridge memory with a learned Bayesian posterior and compare repairs to the incumbent under shared uncertainty. |
| `nonlinear_repair_ttt` | Repair with nonlinear online memory | Replace the linear fast readout with four updates to a small gated function of executed-prefix error. |

For architecture figures, visually separate **physical state** (`32×1024`), **computational workspace** (`4×256`), and **fast memory parameters** (32-vector, Bayesian posterior, or three matrices). They serve different purposes. Mark the encoder and world as frozen, offline-learned controller modules as trained, and only the ephemeral fast tensors as fitted online. A box simply titled “TTT” or “reasoning” would conceal the very distinction the user asked to understand.

## Pinned sources

[dispatcher]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/train.py#L26-L84
[protocol]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/RESEARCH_POSITIONING_20261012.md#L27-L40
[encoder]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/vision.py#L77-L117
[state-clip]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/vision.py#L124-L130
[action-cache]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/data.py#L104-L129
[execution]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/evaluate.py#L369-L439
[world]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/models.py#L53-L79
[world-freeze]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/train.py#L76-L84
[distance]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/models.py#L48-L54
[bank-build]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/bank.py#L53-L102
[route-bank]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/bank.py#L105-L152
[gmm-head]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/model.py#L33-L49
[trunk]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/models.py#L11-L45
[gmm-loss]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/model.py#L51-L71
[gmm-sampling]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/model.py#L73-L82
[gmm-controller]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/controller.py#L9-L99
[training-schedule]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/train.py#L235-L259
[workspace]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/revision.py#L51-L141
[revision-loss]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/revision.py#L143-L202
[revision-controller]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/revision_controller.py#L10-L107
[aligned-init]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/aligned.py#L17-L84
[aligned-score]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/aligned.py#L86-L182
[aligned-config]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/config/flow_metaworld_aligned.json
[flow-architecture]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/flow_reasoning.py#L15-L51
[flow-sampling]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/flow_reasoning.py#L53-L115
[flow-revision]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/flow_reasoning.py#L117-L146
[flow-loss]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/flow_reasoning.py#L148-L188
[flow-controller]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/flow_controller.py#L20-L121
[ridge-solve]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/progress_ttt.py#L23-L42
[ridge-memory]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/progress_ttt.py#L45-L89
[ridge-scoring]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/progress_ttt.py#L91-L128
[ridge-loss]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/progress_ttt.py#L133-L175
[ridge-controller]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/progress_ttt.py#L178-L208
[guidance-architecture]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/guided_ttt.py#L17-L49
[guidance-trust]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/guided_ttt.py#L51-L114
[guidance-sampler]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/guided_ttt.py#L116-L137
[guidance-loss]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/guided_ttt.py#L139-L184
[guidance-config]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/config/flow_metaworld_guided_ttt.json
[repair-sampling]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/repair_flow_ttt.py#L17-L62
[repair-loss]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/repair_flow_ttt.py#L71-L131
[retrieval-training]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/retrieval_repair.py#L13-L65
[retrieval-pool]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/retrieval_repair.py#L68-L121
[shifted-data]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/shifted_repair.py#L17-L61
[shifted-training]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/shifted_repair.py#L64-L137
[progress-readout]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/progress_objective.py#L18-L56
[progress-fit]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/progress_objective.py#L58-L89
[progress-score]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/progress_objective.py#L91-L143
[progress-results]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/PROGRESS_OBJECTIVE_TTT_RESULTS.md
[bayesian-math]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/bayesian_repair.py#L21-L64
[bayesian-posterior]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/bayesian_repair.py#L67-L121
[bayesian-cost]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/bayesian_repair.py#L123-L176
[bayesian-training]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/bayesian_repair.py#L178-L265
[bayesian-controller]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/bayesian_controller.py#L23-L137
[memory-math]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/leflow_ttt/model.py#L15-L30
[nonlinear-fit]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/nonlinear_repair.py#L19-L80
[nonlinear-score]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/execution/nonlinear_repair.py#L82-L135
[leflow]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/baselines/leflow.py#L14-L95
[leflow-sampling]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/baselines/leflow.py#L97-L129
[baseline-config]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/config/flow_metaworld_leflow_release.json
[direct-memory]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/leflow_ttt/model.py#L33-L97
[direct-adaptation]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/leflow_ttt/model.py#L100-L179
[direct-controller]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/leflow_ttt/controller.py#L6-L79
[hwm-model]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/baselines/hwm.py#L1-L78
[hwm-controller]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/baselines/hwm.py#L81-L153
[hwm-config]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/config/flow_metaworld_hwm_paper.json
[released-cem]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/flow_jepa/baselines/cem.py#L16-L63
[cem-config]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/config/flow_metaworld_cem_release.json
[audit-scope]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/RESEARCH_POSITIONING_20261012.md#L15-L25
[results]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/RESEARCH_POSITIONING_20261012.md#L84-L109
[history]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/RESEARCH_POSITIONING_20261012.md#L111-L128
[paired]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/RESEARCH_POSITIONING_20261012.md#L132-L152
[calibration]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/RESEARCH_POSITIONING_20261012.md#L154-L168
[diversity]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/RESEARCH_POSITIONING_20261012.md#L170-L176
[timing]: https://github.com/Abecid/LeFlow-/blob/b2bc294fe5a3de99f195aba90c99481a4852a401/docs/RESEARCH_POSITIONING_20261012.md#L178-L192
