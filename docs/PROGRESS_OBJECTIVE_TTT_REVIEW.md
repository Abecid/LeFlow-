# Post-publication review: retain the 80/104 reference

The complete progress-objective result is published at
[`82f67878254accfc5aa970c5200574c7b79aea99`](https://github.com/Abecid/LeFlow-/commit/82f67878254accfc5aa970c5200574c7b79aea99).
Live remote verification checked 382 artifact blobs, 53 registered evaluation
reports and 14 completed entries, including all 107 artifacts for this run.
[Verification record](reports/20261011-progress-objective-ttt/publication-verification.json).
The curve is 72/72/78/77; selected 15k scores 78/104 with zero timeouts.
The [complete findings](PROGRESS_OBJECTIVE_TTT_RESULTS.md) preserve every outcome.

**No further specific correction is justified by the present evidence.**
Stop launches for review under the user's stopping rule. This is a research
judgment, not a claim that improvement is impossible or an operational blocker.
No run is active or queued. Keep monitoring available for later explicitly
authorized, registered work; do not restart completed runs or infer permission
from an idle GPU or another heartbeat.

## What the completed run resolves

The review uses all 800 periodic records, four fixed evaluations, paired case
outcomes, all 40 visually inspected saved contact sheets, the causal scoring
implementation and the final source/data/checkpoint audit. It introduces no
model call, simulator call, additional training or final-test access.

| Proposed explanation | Observed evidence | Decision |
| --- | --- | --- |
| The new progress score or calibration never participates | Positive optimism penalties occur in 2,769/5,181 selected-checkpoint decisions; stage penalties are nonzero. Exact score reconstruction, past-prefix fit and episodic reset pass. | No missing-use or score-accounting failure was found to repair. |
| A stage score removes pick-place route drift and restores success | Pick-place scores 0/0/1/0. Reach/direct-only late routes remain common, but one selected and two final failed cases retain pick-place routes for all 20 final decisions. | Matching route provenance is not sufficient. Oracle task-ID filtering and a stronger stage penalty are not supported corrections. |
| Better calibration predicts better physical success | Applied proxy error improves at 5k and 10k, which both score 72. At selected 15k78 it slightly worsens raw error, 0.016240 to 0.016436. At 20k77 it is nearly unchanged. The readout changes across checkpoints. | Neither disabling calibration nor changing its weight is justified by these observational proxy comparisons. |
| The teacher bound or inadequate optimization is the main obstacle | Action-flow loss falls 0.168696 to 0.080395; deep clipping is 7.25/4.05/3.72/4.07%. The run finishes its 20k update allowance. | Do not enlarge the radius, extend the budget or repeat training based on these aggregate losses. |
| Proposal contraction causes the remaining failures | Initial global/residual spread contracts 57.77%/47.23% by selected 15k while success improves 72 to 78; it contracts further at 20k77. Improving predecessor runs also contracted. | No specific diversity intervention or sweep is established. |
| A timeout or process failure explains the negative result | All 416 episodes have zero timeouts; all 117 failures reach 200 actions. Source/data/checkpoint audit passes, all 44 recorded owned processes exit, W&B finishes. | No operational recovery, evaluation replay or optimizer restart is needed. |

Weak remaining-expert-time supervision and the inherited shifted-plan coverage
approximation remain limitations. Combining prior variants, adding recursion,
changing weights or fitting another probe would be a new hypothesis, not a fix
demonstrated by these records. The failed shifted variant already showed that
covering one previous-plan approximation did not beat 80. This does not rule out
future work, but it does not supply the missing evidence for an immediate run.

The inspected pick-place example leaves the object on the table; faucet and
reach failures show little late visible motion, and assembly0 regresses after
its selected-checkpoint success. Single-view frames cannot establish precise
3D contacts, forces or the outcome of rejected actions. Recorded expert actions
and executed prefixes retain their own labels; generated alternatives receive
no invented physical outcomes. The method's adaptations and primary-source
distinctions remain documented in [the method report](PROGRESS_OBJECTIVE_TTT.md).

## Preserved result and future scope

Retain the 80/104 references. Known measured cumulative components total
105.284527341 GPU-hours, not a full cluster bill. One seed, repeated development
selection, custom baseline ports and unequal consumed compute do not establish
generalization, component causality, novelty or SOTA.

A future candidate needs a concrete new evidence-to-correction argument,
causal supervision/accounting, meaningful checks, a frozen source and pre-launch
registration under the standing finite budget. A changed research/data/budget
scope needs explicit user authorization. No speculative queue is created here.
The monitor remains active to publish later authorized completed work and detect
publication failures, staying quiet while nothing actionable changes.
