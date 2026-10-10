# Project identity, late-training regression and research contribution

Read-only experiment audit, October 9 Pacific / October 10 UTC, 2026. Audited
experiment head `22a695aab2e39cc5fe5faee96a903107752ab8c8`, including the compressed
training logs and all four validation reports. Also reviewed the subsequent
documentation-only commit `e59760911b4a1c73d573b65cc4be29a466fd04fc`; it does not
change those results. This review makes zero model,
simulator, training or final-test calls. The independently recomputed counts,
episode pairing, source hashes and loss/diversity summaries are in
[the audit record](reports/20261010-project-audit.json).

The latest completed method is **a Gaussian-mixture/retrieval controller using
frozen V-JEPA 2.1 features**, not a flow-matching or diffusion planner. Its
selected 5k checkpoint achieves 80/104; the final 20k checkpoint achieves 76/104.
The project retains the flow-jepa repository/package name, but the successful
recent training path does not optimize a flow-matching objective.

## What actually trained

| Item | Current experiment |
|---|---|
| Task | Custom MetaWorld v3 image-goal manipulation; not standard MT10/MT50 |
| Inputs | Current RGB observation and an RGB image of the completed task |
| Encoder | Official frozen V-JEPA 2.1 ViT-L distilled checkpoint |
| State view | Current image repeated over a 16-frame input, pooled to 32 tokens of 1,024 features; goals use the same convention |
| Dynamics | Our separately trained action-conditioned predictor, frozen during policy training; not the released V-JEPA action-conditioned controller |
| New learned model | Fresh 4,531,782-parameter RevisionPolicy with a four-component Gaussian-mixture action distribution and a recurrent execution-error workspace |
| Actions | Five control blocks, each containing two four-dimensional simulator actions: 40 coordinates per candidate |
| Training data | 7,800 episodes: 6,240 scripted expert and 1,560 random; world uses both, policy/retrieval use the 6,222 successful experts |
| Training size | 20,000 updates, global batch 64, 1,280,000 sampled windows, one seed 3072 |
| Recent hardware | Eight A800 GPUs under the later recorded authorization and unchanged aggregate optimization ceiling |
| Development validation | Same 104 resets at 5k/10k/15k/20k: eight per training task, drawn from the 650-case validation pool |
| Reserved final evaluation | 3,200 cases: 200 per task over 16 tasks; no new final-test results examined here |

The 13 training tasks are assembly, button-press-topdown, coffee-button, dial-turn,
door-close, door-open, drawer-close, drawer-open, faucet-open, handle-press,
pick-place, plate-slide and reach. The additional task-held-out cases are
window-open, handle-pull and push. They do not contribute to the 104-case score.
Evaluation cases were filtered for expert-constructible goal annotations before
training; the score describes that selected reset distribution.

Primary evaluation is task-balanced environment success within **200 primitive
actions and 10 seconds of cumulative controller computation**. The latter includes
encoding and planning, excludes simulator physics/rendering, and counts timeout
as failure. Secondary metrics include success by 50/100/200 actions, capped steps,
return, latency, world calls and observed/predicted progress. Action likelihood,
latent consistency and calibration loss are training/proxy diagnostics, not task
success. The latest losses fit recorded actions, match predicted local outcomes
to recorded states, and calibrate cost errors using factual recorded transitions;
there is no RL reward objective or flow-matching loss.

At inference, seven retrieved targets from training routes plus the final goal
form eight options. For each, the policy proposes short actions; the frozen world
scores three batches of four candidates. The target and its selected actions are
chosen together, and only the first two primitive actions execute. Recent observed
prediction errors condition the policy and score corrections. The latest version
also reuses a shifted previous plan and temporarily penalizes repeatedly optimistic
targets. This requires 480 candidate world transitions per decision.

## Which things changed

1. The original upstream LeFlow setting uses LeWorldModel with TwoRoom, PushT,
   Reacher and OGBench Cube. We moved to the custom V-JEPA 2.1/MetaWorld setting.
   LeFlow and V-JEPA are different layers of a system, not mutually exclusive
   model families: the former is a planner, the latter supplies representations.
2. The original MetaWorld implementation used causal-history states but static
   image goals. The repair aligned both to static-image features and trained a
   compatible new world. Old LeFlow/HWM scores therefore do not share the new world.
3. After the repaired joint-flow run, controller-grounded planning replaced
   generated long latent paths with retrieved observed targets and a short GMM
   action policy. Latent revision then added execution-error memory/corrections.
4. The latest execution revision retains the same episodes, encoder, world and
   route tensors as those two GMM runs. Its sampler permits all valid five-block
   starts up to first success, replacing the inherited requirement to fit a
   60-block window. Goal offsets must fit the remaining episode. Thus the
   sampling distribution changes, but no new trajectories or tasks are added.
5. The initial three-seed/full-test plan was narrowed in subsequent recorded
   instructions to one seed and fixed development validations. Do not describe
   the current result as a three-seed, 3,200-case test result.

## Regression measured directly from saved records

| Updates | Success | Last-500-update action NLL | Initial proposal diversity | Timeouts |
|---:|---:|---:|---:|---:|
| 5,000 | 80/104 (76.92%) | -1.402 | 0.07522 | 0 |
| 10,000 | 78/104 (75.00%) | -1.680 | 0.04815 | 0 |
| 15,000 | 79/104 (75.96%) | -1.787 | 0.03756 | 0 |
| 20,000 | 76/104 (73.08%) | -1.824 | 0.03666 | 0 |

All reset IDs, seeds, episode hashes and model seeds match. From 5k to 20k,
seven cases are lost and three gained. Assembly changes 3/8 to 0/8, pick-place
1/8 to 0/8, reach 4/8 to 3/8, and door-open 6/8 to 7/8. Other task totals are
unchanged, although individual dial-turn/door-open cases exchange outcomes.
The four-case net drop is descriptive; a post-hoc exact paired McNemar calculation
gives p=0.34375. Checkpoint selection, repeated development and one training seed
prevent treating this as a confirmatory generalization result.

The strongest diagnostic lead is **better demonstration fitting with narrower
proposals**, rather than numerical divergence or timeout. At identical initial
reset/goal pairs, with no previous-plan or execution history, proposal diversity
falls 51.27% and is lower in 101/104 cases. This statistic averages action-coordinate
standard deviations across the proposed pool, including fixed recorded actions.
It is not mixture entropy and does not identify whether mixture weights, component
means, scales or clipping caused the reduction. The implementation has a positive
scale floor. Calling this proven mode collapse would overstate the evidence.

The plausible mechanism is that fitting successful expert actions more tightly
narrows useful recovery alternatives once execution departs from demonstrations.
The policy/calibration data contain recorded expert behavior, not true outcomes
for all newly proposed actions. Current logs do not include held-out action NLL
or counterfactual outcome labels that would prove this mechanism or distinguish
memorization from other objective/distribution effects. More training or replacing
the GMM with a flow model is not an established remedy.

Prediction/selection mismatch also persists. At 5k, 17/24 failures show positive
raw predicted but nonpositive observed local progress in their last 20 decisions;
at 20k this is 20/28. Twenty-three final-checkpoint failures end that window no
closer to the image goal in latent distance. The corrected chosen-prefix MAE is
0.002003 at 5k and 0.002120 at 20k, measured on different policy-induced trajectories.
Better calibration on factual training actions need not improve ranking of new
actions. At 5k, corrections change within-target candidate choices in 6.23% of
evaluated groups but the selected target in only 0.83% of decisions. These are
different denominators, neither a causal attribution experiment.

I inspected the saved 5k pick-place/00000, assembly/00000 and faucet-open/00003
contact sheets. Pick-place shows the gripper moving toward the goal side while
the red object remains on the table. Assembly shows little change after approach.
Faucet-open shows the arm moving past/right of the handle. These support a failure
of object-task completion, not a force/contact diagnosis. No 20k frame comparison
or new simulator execution was performed. See the
[saved contact sheets](reports/20261009-execution-revision/contact-sheets-step-5000/).

## What the available baselines establish

All internal rows below use the same fixed 104 validation cases and seed.

| Selected reference | Success | Interpretation |
|---|---:|---|
| Execution revision, 5k | 80/104 (76.92%) | Current best; GMM, not flow |
| Controller-grounded, 5k | 79/104 (75.96%) | Strongest earlier pipeline; same encoder/world/data |
| Latent revision, 20k | 77/104 (74.04%) | Same encoder/world/data |
| Historical LeFlow adaptation | 27/104 (25.96%) | Different world/state convention; known restricted-noise sampler defect |
| Original joint flow | 23/104 (22.12%) | Historical representation and sampler defect |
| Same-world CEM | 22/104 (21.15%) | Same fine world; all 82 failures timed out |
| Repaired joint flow | 17/104 (16.35%) | Same fine world; all 87 failures timed out |
| Historical HWM adaptation | 8/104 (7.69%) | Simplified adaptation on older representation/world |

The latest improvement over the strongest prior method is **one case**, seven
paired gains against six losses. The saved descriptive paired interval spans
-5.77 to +7.69 percentage points. The substantial improvement over earlier slow
pipelines is a full-system result under a wall-time allowance, not an isolated
effect of the new correction module, evidence for flow matching, or a corrected
published-SOTA comparison. No faithful corrected LeFlow or strong matched
retrieval/GMM ablation has been run under this custom protocol.

For external context, [Planning Limits](https://arxiv.org/html/2609.39235v1)
reports 30% for five-step V-JEPA 2.1 MPC, 47% for thirty-step MPC at matched
rollout-compute budget, 52% with its larger budget, and 76% with same-episode
expert subgoals. Those use 16 tasks, 20 episodes/task and a different predictor
and protocol. Our selected 13-task validation number cannot be ranked against them.
The paper motivates supplying useful nearby targets without an evaluation-time
expert trajectory; our retrieval uses only training episodes.

[LeFlow's published results](https://arxiv.org/pdf/2608.24855), Table 1, are
100.0% TwoRoom, 95.2% PushT, 86.8% Reacher and 100.0% OGBench Cube (five evaluation
runs). They are not MetaWorld comparator scores. [Anchored Planning](https://arxiv.org/pdf/2609.30036)
already uses recorded intermediate targets, and [H-JEPA](https://arxiv.org/html/2610.06805v1)
already studies learned hierarchical abstractions. Neither paper supplies a
directly interchangeable result for our custom protocol.

## LeFlow's role and the unresolved contribution

The [official LeFlow implementation](https://github.com/hsiangwei0903/LeFlow)
uses a frozen LeWM encoder/predictor, a flow model generating latent-path interiors
conditioned on the current/goal states, an inverse decoder producing actions, and
world-rollout reranking. It learns a reusable planning prior; it does not replace
JEPA prediction with pixel generation. The paper also already describes
generated-transition dynamics consistency in Section 3.5. Our earlier framing
cannot claim that consistency alone as a new contribution.

The original project hypothesis was joint latent-subgoal/action flow generation
to supply executable local targets for a short-range JEPA world. The current
hypothesis has become **execution-feedback-conditioned local control**: evaluate
the response the deployed controller will actually execute, observe its prediction
errors, and use them in the next decision. The implementation is a useful
engineering baseline, but the benefit of its recurrent correction mechanism has
not been established over the simpler 79/104 reference. Borrowed retrieval,
mixture proposals, CEM and plan reuse do not themselves establish novelty.

If the intended research claim remains flow matching improving JEPA planning,
keep the current GMM system as a strong control and formulate a flow mechanism
that changes useful multimodal proposal coverage or execution-conditioned revision
under the same inference allowance. A generic GMM-to-flow substitution, more
training, or another consistency term is not enough. The eventual evidence must
isolate that mechanism with unchanged representations, controller, data and
supported targets; correction must respect the lack of observed counterfactual
outcomes for rejected actions. This is a research recommendation only. No variant,
baseline rerun, extra seed or final evaluation is launched by this audit.
