# Self-refinement transfer: failure and behavior analysis

All12 runs completed600 physical cases with zero parameter updates. No controller timed out; every unsuccessful case reached50 actions. The retained reference reproduces all50 prior success/failure flags on each task under the explicit common FP32 setting. Original reports are unchanged.

| Variant | PushT /50 | Cube /50 | PushT ms/decision | Cube ms/decision |
| --- | ---: | ---: | ---: | ---: |
|leflow|12|40|50.43|57.39|
|leflow_more_samples|16|37|53.75|61.24|
|leflow_more_steps|11|37|64.51|72.43|
|pnp|7|38|64.46|74.80|
|guidance_only|12|36|392.71|396.13|
|guided_pnp|4|37|392.90|386.14|

Guided P&P adds1 success and loses9 versus LeFlow on PushT, and adds1/loses4 on Cube. Plain P&P adds2/loses7 on PushT and2/4 on Cube. Guidance alone swaps4/4 PushT cases and loses4 Cube cases without gaining any. More candidates gains4 PushT cases without a regression, but swaps1/4 on Cube. This is a single-seed development screening result; these counts do not establish final generalization.

The regression is not explained by the registered controller deadline: all runs have zero timeouts. The guided planners use roughly7–8 times the observed per-decision latency of the reference, yet complete the same finite action allowance. There is no evidence here that gradient work buys a better success–compute tradeoff.

Selected surrogate cost and factual next-prefix prediction error do not track success monotonically. Across their own executed trajectories, PushT guided P&P has median selected predicted terminal MSE0.02550 versus reference0.03124, and median observed prefix prediction MSE0.00683 versus0.00847, but succeeds on4 rather than12 cases. These medians use436 versus411 observed prefixes and different trajectories; they do not prove that guidance causally reduced model error or reveal rejected-action outcomes. Cube guidance-only similarly reduces median predicted terminal MSE0.01409→0.00811 while success drops40→36. The diagnostic establishes that a lower prediction/distance proxy is insufficient to establish task completion.

Frozen-policy/world fingerprints remain equal before and after each full rollout run. TRAIN-only real-checkpoint tests preserve sampler/decoder behavior and match the legacy reference’s first action and RNG bitwise under matched precision. Deadline, physical action bounds, source/cache/checkpoint identities and inference-mode gradients pass. No additional sampler, representation or indexing bug is confirmed by these checks; they are scoped evidence, not proof that every implementation detail is infallible.

Both stochastic P&P and guided P&P replace the entire generated candidate pool; they do not retain the original selected trajectory as an incumbent. The regression therefore occurs even without gradient correction. It does not distinguish proposal quality from reranking error. We did not execute rejected candidates or isolate which guidance term causes regressions, so harmful override, manifold drift and objective mismatch remain explanations to test rather than established physical causes.

The useful next diagnostic would compare original/refined candidates from the same decision inputs and preserve the original option, then validate which score changes predict actual executable progress. This run does not authorize another variant, retraining, expanded benchmark, DA-SIP allocation or TTT. Reacher/TwoRoom author-checkpoint results remain unavailable.

[Full physical reports and paired case IDs](SELF_REFINE_SHARED_RESULTS_20261014.md).
