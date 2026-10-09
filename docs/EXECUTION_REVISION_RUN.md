# Execution-aligned revision: registered candidate

October 9, 2026. The user authorized implementing the diagnosed improvements and
one fresh training run. This registration precedes its first validation result.
Method name: `execution_revision`; configuration: `config/flow_metaworld_aligned.json`.

## Evidence and scope

The previous latent-revision run scored77/104, below the controller-grounded79/104.
It improved chosen-prefix prediction MAE28.12%, but used only a positive terminal
correction for decisions. That correction changed target choice in0.428% of
recorded candidate pools. Eighteen of27 failures retained optimistic, actually
nonprogressing late local behavior. An inherited60-control-step sampling window
omitted supervised action targets after primitive action90 despite a five-step
action head. The direct-goal candidate set also duplicated its mean action.
See [completed evidence](LATENT_REVISION_RESULTS.md).

The new candidate retains the same4,531,782-parameter RevisionPolicy, initialized
from scratch, four-step latent workspace, factual calibration objective, cached
V-JEPA2.1 encoder, normalization and frozen fine world. No extra teacher,
simulator training, diffusion head, new training episodes or generated-action
physical labels are introduced. This is a bundled first repair, not an ablation
or proof that any individual mechanism causes improvement.

## Fixed changes

1. **Executed-prefix scoring.** For each candidate, letC1 be predicted distance
   after the first control block andC5 the average terminal distance at blocks4/5.
   LetE1,E5 be their factual-error-trained signed corrections. Inner CEM uses
   `J = 0.5*(C1 + relu(E1)) + 0.5*(C5 + relu(E5))`.
   Outer selection uses unchanged route cost+J+the stall penalty below.
   Equal weights are fixed before evaluation, not selected by a validation sweep.
   Negative corrections cannot reward unsupported generated actions. The full
   route and terminal terms retain incentives beyond one-step goal proximity.

2. **Shifted-plan reuse.** After observing the executed block, preserve the
   remaining four blocks of the selected plan as one proposal for every target,
   with that target's fresh sampled tail. It replaces stochastic slot1. Mean,
   recorded-expert and fresh alternatives remain; the direct goal's last slot
   stays independently sampled instead of duplicating its mean. This follows
   the established reuse principle in [iCEM](https://github.com/martius-lab/iCEM),
   not a novelty claim. No additional candidates or model transitions are used.

3. **Causal, temporary stall feedback.** Retain eight observed transitions.
   A transition counts only when predicted target progress>0 and observed
   progress<=0. For a candidate target, match past start and target within
   token-mean cosine distance0.005 of the current start/target. At least three
   matching failures are required. Sum their observed optimism gaps, weighted
   by0.8^age, capped at0.02 cost units. This penalty decays and expires, is local
   to state/target, never uses task labels, and cannot be triggered by imagined
   outcomes. These are explicit first-candidate engineering constants, not
   empirically optimized or certified confidence bounds.

4. **Local training-window coverage.** Keep the same task/episode random draws
   and successful expert episode inventory. Sample start uniformly through
   `min(episode_steps-5, first_success_action//2)`. Then choose among5/10/20/40/60
   goal offsets that fit the remaining episode. Five-step labels and causal
   four-transition history remain factual. Starts do not move past first
   success; a chunk may straddle it. There is no new padding or synthetic data.
   The goal/start distribution changes explicitly; it is not identical-window
   training to previous candidates. Late recorded actions were already present
   in the shared retrieval bank; this repairs the learned head's supervision.

## Preserved comparison contract

- Exactly one fresh seed3072; global batch64 across8 A800 GPUs, physical batch8.
- Stop at20,000 updates or28,800 aggregate optimization GPU-seconds, whichever
  occurs first. All learned components and factual-history work are charged.
  This is1,280,000 sampled windows at the update cap, not conventional epochs.
- Same7,800-episode training split; policy/bank use6,222 successful experts.
  Same cached features and frozen world SHA256
  `ae3f910da43f0a43dd65945394444a0b11719ed042cdacffacc2faaf824ec5a3`.
- Same104 validation cases at5k/10k/15k/20k. Highest task-macro success selects
  the checkpoint; earliest wins ties. Restore training RNG after validation.
- Same200 primitive actions and10-second cumulative controller allowance per
  case, including encoding, retrieval, latent reasoning, feedback and search.
  Eight targets×four candidates×three CEM rounds×five blocks=480 world
  transitions per decision. Execute only the selected first block (two actions).
- Preserve saved previous79/104 and latent-revision77/104, same-world CEM22,
  repaired flow17, historical LeFlow27 and HWM8. Historical implementation and
  world confounds remain: this is not a faithful-SOTA superiority claim.
- No baseline retraining, extra seeds, automatic variants, ablations or final
  test. Reserved3,200-case test remains sealed. Repeated development validation
  is not an unbiased final-test estimate. Equal caps are not equal realized FLOPs.

## Verification, diagnostics and accounting

Behavioral checks cover late action99/state100 supervision, bounded future goals,
same episode draws, causal histories, prefix-induced ranking changes, negative
correction conservatism, observed-before-reuse, exact executed action handoff,
480 transitions, local repeated-stall activation, expiration and episode reset.
All existing factual-label and frozen-world tests remain required.

Bounded preflight uses training data only: check10,000 sample specifications,
eight actual late windows, finite forward/backward and unchanged world weights.
Time the full encoder/controller with populated training-derived history, plan
reuse and stall memory; no parameter update or policy simulator rollout occurs.
Retain the prior120ms preflight gate rather than relaxing execution limits.

Log previous metrics plus late-window fraction, mean start/goal offset, prefix
and terminal penalties, candidate stall counts/penalties, warm-start availability,
correction-induced within-target candidate changes and target changes. Chosen
prefix observations support calibration diagnostics; rejected candidates have
no true physical-outcome labels. Capture the same ten trajectory case IDs.

Report optimization, validation, preparation/preflight and other reserved GPU
occupancy separately. Keep execution source frozen and publish progress/results
from a separate checkout. Compare actual success and paired failures; a repair
is not presumed to help.
