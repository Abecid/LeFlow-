# Authorized repairs after baseline training

> Latest authorization: the user has now explicitly requested training from
> scratch and a matched final comparison. See [REPAIRED_COMPARISON](REPAIRED_COMPARISON.md).
> The no-training boundary described below was valid during the repair stage;
> the original campaign remains held, while the new campaign is authorized.


The user's October 8 instruction supersedes the earlier automatic final-test
queue. All four learned runs (shared world plus three heads) completed 20,000
updates and four registered validations. No additional training is authorized.
The original final test had already begun; its supervisor/evaluator were stopped
at 02:45:55 UTC, October 9. Existing test records remain preserved and their
outcomes will not be used for these repairs.

## Latest completed validation

| Method | Selected step | Success / 104 |
| --- | --- | --- |
| Ours | 5,000 | 23 (22.12%) |
| LeFlow adaptation | 20,000 | 27 (25.96%) |
| HWM adaptation | 20,000 | 8 (7.69%) |
| CEM using selected shared world | world 20,000 | 7 (6.73%) |

The shared world and three heads consumed 21.112076 optimization GPU-hours in
total; registered training-time validation consumed 9.050318 GPU-hours separately.
These are adaptation results subject to [the audit](EVAL_AUDIT_20261009.md).

## Fixed constraints

- No optimizer updates to any saved model, no retraining, new seeds, or ablation
  campaigns. Repair code and evaluation-only diagnostics are authorized.
- Retain seed 3072, the same four methods, original train/validation/test reset
  IDs, image goals, task lists, cache hashes, and split boundaries.
- Any future training comparison needs separate authorization and must retain
  the same per-method ceiling: 7,200 optimization seconds or 20,000 updates on
  at most four GPUs, whichever comes first. Never give ours extra fine-tuning
  before comparing it with untouched baselines. Charge validation separately.
- Preserve the 10-second cumulative controller and 200-action episode limits
  for actual comparisons. Short diagnostic rollouts are not benchmark scores.
- Keep frozen execution SHA 56419ed, checkpoints, ledgers, and cache unchanged.
  Publish from the separate reporting checkout to origin/main.

## Work now

1. Implement a clean-endpoint state parameterization with full-dimensional
   start/goal bridge anchors and learned residuals, plus a terminal Euler update
   that can remove all sampled state noise, keeping the transformer width fixed.
   Apply the same state sampler/target convention to both flow methods. Test at
   the production 1,024-dimensional state and 256-dimensional hidden width.
   Preserve legacy behavior for reading historical checkpoints. New semantics
   require their own protocol; do not reinterpret legacy velocity weights as
   trained endpoint predictors.
2. Align the state/goal feature convention using the static-image feature view
   already present in the lossless cache. Use it consistently for dynamics,
   planner states, local inverse targets, goals, and live observations. This is
   a protocol change shared by all methods, not a conversion of existing trained
   dynamics. Check exact cache/live-input correspondence without new training.
3. Add common continuous-rollout ranking for the two flow methods: begin at the
   actual observation, roll out proposed actions continuously, score final-goal
   distance and waypoint agreement. Never restart the score's predicted state
   at an imagined waypoint. Keep the original scoring selectable for historical
   checkpoint diagnostics.
4. Check candidate scores against simulator execution using existing selected
   checkpoints only. Preregister one existing validation reset per each of the
   13 training tasks (index 0), eight candidates, and both flow methods: 208 short
   candidate rollouts. Each executes one five-control-step action chunk, with
   shared short CEM refinement. Report rank correlation and selected-candidate
   regret for actual subgoal distance and static-goal progress, retaining all
   failures. The test split is excluded. Bound this diagnostic to at most four
   GPUs and ten minutes; no sweep or automatic expansion follows.

The diagnostic evaluates original trained proposals and alternative scoring; it
cannot establish the success of an untrained repaired architecture. A negative
or inconclusive calibration result is a valid result and must be reported. The
repaired configuration must reject training and full-test launch by default.
Only a later explicitly authorized, equally budgeted comparison could establish
whether trained repaired models improve success.

## Publication and completion

Push current findings and this plan to origin/main now, then push tested repairs,
diagnostic results and remaining limitations there. Verify the remote commits.
The existing monitor is updated to respect the hold, avoid duplicate repair
edits, and never restart training or final tests automatically.
