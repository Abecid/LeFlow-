# Next step: diagnose and learn goal-conditioned task progress

October11,2026,21:06 Pacific. No training is active. This is a research review
and a saved-record diagnostic, not a newly registered full training run.
The prior review's launch hold remains while the proposed scoring mechanism
needs a training-data diagnostic and concrete validated implementation.

## The evidence has changed the priority

The first flow-repair run tied80/104 (80/73/75/75); retrieval-conditioned repair
tied80 (75/77/79/80); shifted-plan repair selected79 (79/78/74/78). All twelve
registered evaluations of these three runs have zero controller timeouts.
Their failures exhaust200 actions. More speed is no longer the first bottleneck.
Lower teacher loss, less clipping and better prefix calibration have not reliably
increased task success. Further shifted-plan synthesis, a larger radius, more
solver steps or an entropy penalty is not yet supported by these observations.

New read-only analysis follows selected route provenance through all104 cases
for four frozen selected models. It verifies identical reset/hash/seed identities
and makes zero model/simulator calls, optimizer updates or final-test reads.
[Full case inventory](reports/20261011-progress-review/route-provenance.json).

| Selected model | Pick-place success | Failed pick-place cases whose final20 decisions contain no pick-place route |
| --- | ---: | ---: |
| Execution revision80 |1/8|6/7|
| Repair flow80 |2/8|6/6|
| Retrieval repair80 |1/8|7/7|
| Shifted repair79 |1/8|7/7|

For retrieval repair, all seven failed pick-place cases initially select a
pick-place route. Their final140 decisions select only reach routes or the direct
goal. The one successful case selects pick-place routes for all32 decisions.
Across all eight cases,155/732 selected targets come from reach episodes and
315/732 are direct-goal targets. Saved pick-place0 images show the hand moving
while the cube remains on the table; its last20 choices are all direct-goal.
The saved reach3 failure was also inspected and appears nearly stationary late.
These images do not measure exact3D distances or contact forces.

This is an association and a concrete target-selection warning, not proof that
cross-task transfer or the direct-goal candidate is inherently wrong. Some reach
routes may help a legitimate approaching phase; there is no counterfactual
physical outcome for rejected choices. Provenance names the selected target route,
not necessarily the source of the final repaired action. Do not filter routes by
benchmark task ID or count such an oracle filter as method improvement.

## The implementation gap worth testing

`models.distance` averages spatial-token cosine costs. `bank.RouteBank.query`
ranks routes by start matching plus endpoint-to-goal distance. The controller
adds local prefix/terminal distance and conservative prediction-error corrections.
The direct goal is always an option at the same five-block horizon. The episodic
fit learns observed-minus-predicted COSINE COST, not a task-completion value.

Consequently, accurate calibration can make the wrong proxy more accurate.
Moving a hand closer to the final image is not sufficient evidence of grasping,
lifting or manipulating the relevant object. The code has no explicit learned
progress ordering or stage compatibility criterion to resolve that ambiguity.
This is a verified objective/design limitation; its causal contribution to the
remaining failures has not been measured. It is not a confirmed indexing or
sampler bug. Assembly/reach errors may have additional causes.

## Recommended sequence, with one central change

Use retrieval-repair80 as the development starting formulation: its generated/
retrieved anchor distribution is covered in training, it ties the best result,
and its complete curve improves through20k. Keep execution-revision80 as a frozen
reference. Do not transfer a pretrained policy into a supposedly fresh-budget run.

1. Audit the planning metric on existing TRAIN trajectories before another full
   run. Use temporally separated recorded states with the same future goal; check
   whether visual cost correctly orders coarse progress and separates visually
   similar stages. Inspect spatial-token contributions. Temporal order is weak
   supervision, not strict monotonic physical progress at every transition.
   Reserve disjoint episodes within TRAIN for a diagnostic probe; leave the104
   comparison and final suites unchanged. Charge any learned probe to the method
   budget, rather than hiding it as free pretraining.
2. If the features contain useful progress information, add one small goal-
   conditioned progress/stage readout on frozen JEPA tokens. Train with recorded
   temporal relations and existing action/outcome data. Use it to score target
   compatibility and the full-world consequences of flow repairs. A direct-goal
   option should need evidence of local completion compatibility, not win merely
   on averaged visual similarity. Learn this from observations/history/goals,
   not task IDs, phase annotations, simulator coordinates or an oracle filter.
3. Make episodic correction update this progress estimate using observed
   transitions. Keep the exact incumbent and bounded flow repairs; when support
   is weak, retain conservative behavior. Validate observed-state ordering before
   relying on predicted states; factual prediction-error checks remain necessary.
   Generated actions never inherit a demonstration's physical outcome label.
4. Only after that diagnostic/design gate, register one fresh candidate under
   the same6222/global64/20k/28800 GPU-second/104-case/controller contract. Compare
   task success, paired gains/losses, stage/target choices, object-interaction
   failures and actual compute. More inference is a later controlled test after
   the score can distinguish useful from misleading plans.

This is a proposed direction, not an implemented or proven improvement. The
small readout might fail because the frozen representation lacks the needed
information; that outcome should prevent another blind policy run. No numerical
threshold, phase count, new optimizer setting or solver sweep is selected here.

## Primary sources and faithful adaptation limits

[ReGuide](https://reguide-project.github.io/) organizes guidance around task phases
and gates it when dynamics extrapolation is unreliable. Its full method collects
successful guided rollouts and adds them to training, which violates our unchanged
expert-data comparison if copied directly. Our candidate would use only existing
trajectories and retain our world/flow controller. It would be an adaptation.
Its [official clustering code](https://github.com/tzuhsiangl/reguide/blob/89babcf51244b5a5ec4e0bebf24278f8fb41db40/pcg_data_generation/latent_state_clustering.py)
was inspected: visual/proprioceptive temporal differences, PCA/k-means, clusters
ordered by median normalized trajectory time, phase prototype sets. Our visual-
only goal-conditioned setting lacks its proprioception/task-specific pipeline;
do not copy a hard phase count or treat cluster order as physical ground truth.

[ProWorld](https://arxiv.org/html/2608.01926) explicitly learns goal-conditioned
progress ordering using temporally separated pairs as weak supervision and
scores sustained progress. It retrains a hyperbolic world representation; we
would initially test a small readout on frozen features, without claiming that
its geometry or reported gains transfer. Its paper states code is to be released
after acceptance; implementation equivalence has not been verified.

[H-JEPA](https://h-jepa.com/) provides evidence that changing the space used for
planning costs can improve planning, and that predictive features can neglect
agent information. It does not establish that our frozen features preserve the
object/contact distinctions needed for these MetaWorld failures.

## Research angle

**Execution-calibrated progress reasoning for flow planning:** learn which
predicted changes advance the goal, use observed execution to correct that
judgment, and condition bounded generative repair on it. The central change is
the decision objective and its effect on action selection. Generic phase guidance,
progress learning, feedback and TTT already exist in the sources above; neither
combining them nor an80/104 tie establishes novelty. A distinctive mechanism and
reliable success improvement still need demonstration.

The diagnostic and this recommendation are to be published to origin/main.
Baselines, completed sources, data, training budgets and sealed tests stay intact.
