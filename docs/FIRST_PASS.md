# Four-method first comparison — October 7, 2026

The user limited this campaign to three main baselines and one proposed method,
one training seed, the same data and compute allowance, and periodic evaluation.
Ablations are deferred until first results and failure cases justify follow-up.
This document records the evidence reviewed before any full model training.

## Selected experiment

| Entry | Role and reason |
| --- | --- |
| Joint flow + generated-plan consistency | Our strongest motivated candidate: jointly propose actions/subgoals, teach generated local transitions to agree with frozen JEPA dynamics, then execute through short-range MPC. Its advantage remains a hypothesis. |
| LeFlow adaptation | Closest learned generative-planning baseline; separates latent path generation from inverse action decoding and checks action outcomes. |
| HWM adaptation | Strong hierarchical comparison; separates any benefit of our learned joint planner from hierarchy itself. |
| JEPA/CEM, horizon 30 | Strong direct-planning reference under the shared world model and controller-time cap. Avoid spending a second run on short CEM. |

These are controlled method-family implementations. The study does not reproduce
published checkpoint scores or establish an across-benchmark SOTA ranking.

## Evidence and design decisions

- [Planning Limits, September 30](https://arxiv.org/html/2609.39235v1): nearby
  oracle subgoals improve the reported MetaWorld controller from 30% to 76%,
  versus 47% for longer rollouts at matched computation. The oracle result is
  motivation for learning subgoals, not a predicted score for our method.
- [HWM](https://arxiv.org/html/2604.03208v2): globally plausible plans can propose
  subgoals the local controller cannot execute. Keep short local transitions and
  explicitly evaluate observed progress toward the proposed targets.
  The [official code](https://github.com/kevinghst/HWM_PLDM) implements its PLDM
  setting; our V-JEPA/MetaWorld comparison is an adaptation.
- [LeFlow](https://arxiv.org/html/2608.24855v1) and its
  [implementation](https://github.com/hsiangwei0903/LeFlow/blob/main/latent_planner.py):
  generative paths plus inverse dynamics and rollout ranking are established.
  Its horizon-scaling results deteriorate strongly, with a changing evaluation
  regime acknowledged by the authors. Our distinguishing hypothesis is generated
  action/subgoal consistency, not generic flow-based latent planning.
- [FF-JEPA](https://arxiv.org/html/2606.09311v1): both deterministic and diffusion
  hierarchies substantially outperform flat planning in its long-horizon PushT
  setting. This supports hierarchy but does not isolate a flow-specific benefit.
  Its goal-free protocol differs from our image-goal task; do not add another
  first-pass adaptation or infer cross-paper superiority.
- [Qantara](https://arxiv.org/html/2607.04978v1) and
  [code](https://github.com/corl-team/qantara): joint state/action modeling is
  viable, and matching training inputs to inference queries matters. Our regularizer
  operates on generated plans rather than only observed state pairs. Qantara's
  bridge objective and end-to-end backbone are not implemented or claimed here.
- [Flow-JEPA](https://arxiv.org/html/2608.29029v3): reports improved robustness but
  explicitly does not separate joint trajectory prediction from stochastic flow
  training. [LeWAM](https://arxiv.org/html/2609.27455v2) reports direct future-feature
  prediction slightly above its flow alternative. Keep the shared deterministic
  predictor; use flow for alternative executable plans rather than adding another
  expensive generative world model or a preference-training stage now.

## Fixed resources and evaluation

- One seed: 3072; at most four GPUs; sequential models on the same allocation.
- One shared frozen encoder/cache and one shared fine world model. Every learned
  head sees identical task-balanced successful-expert episodes, start positions,
  action windows and goals for a given training sample index. HWM trains all its
  local macro transitions from those windows. The fine world model additionally
  uses the same registered random-action data for every method's common backbone.
- Each learned model: at most 7,200 optimization seconds or 20,000 updates,
  whichever occurs first. Three heads plus the shared world give up to 32 GPU-hours
  of optimization on four GPUs, with reported last-update overruns. Shared
  preparation and validation are recorded separately. CEM needs no learned head.
- The generated consistency coefficient warms up over the first 10% and ramps
  over the next 10% of the earlier budget cap, so it is exercised in a short run.
- Four periodic validation rounds on the same 104 episodes (8 x 13 training
  tasks); checkpoint selection uses validation only. The fixed 3,200-reset test
  (200 x 16 tasks) is run once per selected method after all four models finish.
- Every episode: at most 200 primitive actions and 10 cumulative seconds of
  controller computation. Late actions are discarded and timeouts count as
  unsuccessful episodes. Log actual time and any safe-boundary overrun, because
  identical resource allowances do not imply identical FLOPs or consumed time.
- Log macro/per-task success, success within 50/100/200 actions, time to first
  success, return, mean/p95 controller latency, model calls, observed subgoal
  distance, budget exhaustion, gradient/loss statistics, and train/validation
  GPU-hours. Preserve paired per-episode records, including failures.
- Confidence intervals reflect paired-reset uncertainty for one trained model,
  not variation across training seeds. No baseline scores are imported from papers.

Review the first learning curves and failure cases before proposing any follow-up
ablation or new method. There is no automatically queued sweep or extension.
