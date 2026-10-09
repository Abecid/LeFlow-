# Contribution audit: what is actually new?

October 9, 2026, following the user's request to distinguish a central research
contribution from a collection of useful existing methods. Reviewed against
project revision `4c4298b34f6760e78f3eac9146043d27de4de1c2`.

## Verdict on the previous formulation

The [previous formulation](NEXT_METHOD_FORMULATION.md) is primarily an
engineering synthesis. It has a coherent practical motivation, but it does
**not yet establish a distinct algorithmic contribution**. Calling it the
best-supported next engineering hypothesis should not have implied that it
met the user's requirement for a new research direction and mechanism.

The ingredients and their roles are already substantially represented in
prior work:

| Proposed ingredient | Closest reviewed precedent | Contribution status |
|---|---|---|
| Recorded observations as intermediate targets | [Anchored Planning](https://arxiv.org/html/2609.30036v1) | Borrowed support mechanism. Variable-span retrieval without oracle duration is an unvalidated adaptation, not an established contribution. |
| Action-conditioned coarse transitions | [HWM](https://arxiv.org/html/2604.03208v2), [H-JEPA](https://arxiv.org/html/2610.06805v1) | Existing temporal-abstraction idea. |
| Empirical action support for coarse search | [Hi-LeWM](https://arxiv.org/html/2607.12547v1) | Existing response to unsupported coarse-model search. |
| One-pass mixture action proposals | [SAGE](https://arxiv.org/html/2607.17973v1) | Existing proposal mechanism, adapted to our observation setting. |
| Short world-model verification and CEM refinement | [LeFlow](https://arxiv.org/html/2608.24855v1), SAGE and conventional MPC | Existing planning pattern. |

Useful combinations can support a research contribution when they establish a
new principle, mechanism or compelling empirical finding. We have not shown
that here. The proposed combination is unimplemented and untested. The completed
repaired joint-flow run remains 17/104, below saved same-world CEM22/104 and
historical LeFlow27/104. Those results do not validate the proposed combination.

## A sharper problem to organize the research around

The most relevant hypothesis is that **subgoal quality depends on what the
deployed controller can actually accomplish within its compute and execution
limits**. A plausible state or demonstrated route does not by itself specify
that capability.

There is an unresolved mismatch even in the proposed engineering design:

1. The coarse predictor evaluates recorded action chunks from the live state.
2. Route selection chooses a waypoint based on those predicted transitions.
3. The mixture policy and CEM can produce a different action sequence toward
   that waypoint.
4. Only one control block executes; the rest can change after the next
   observation and high-level replan.

Thus the coarse route score is not a prediction of the complete deployed
controller's behavior. The fine verifier checks the selected local proposal in
the model, which is helpful, but does not establish the actual outcome of future
replanning. Receding-horizon execution is standard and is not itself a bug.
Whether this mismatch materially causes our failures is still unmeasured.

Our observed five-step training / 60-step scoring mismatch, long decision
latency and weak earlier proxy correlations motivate examining this question.
They do not prove a causal explanation for the repaired run's failures.
All timeouts are stopping conditions, not evidence of a particular physical
failure. The earlier ranking diagnostic used an older checkpoint.

A precise research question is:

> Can a lightweight adaptation of a frozen visual world model select better
> subgoals by estimating the consequences of the actual bounded controller,
> under the same offline data, training allowance and action cadence?

This is a proposed focus, not a claim that the general idea is new or that the
required mechanism has already been designed.

## Novelty boundaries beyond the recent JEPA literature

A targeted primary-source search found important overlaps that prevent simply
renaming this focus as a new method:

- [HAC / Learning Multi-Level Hierarchies with Hindsight](https://cs.brown.edu/people/gdk/pubs/multi_level_her.pdf),
  Section 4.4, already distinguishes goals an optimal lower-level policy could
  reach from goals the current lower-level policy can reach. Subgoal testing
  penalizes failures. Controller-relative reachability is not our invention.
- [Strict Subgoal Execution](https://arxiv.org/html/2506.21039v1), Sections 4.1–4.3,
  uses low-level failures to refine graph costs and enforces subgoal completion.
  Its removal of fixed subgoal step limits and online experience differ from
  our unchanged controller allowance and fixed offline data. Failure-aware
  routing alone would not be new either.
- [Learning Multi-Timescale Abstractions](https://arxiv.org/html/2605.17058v1),
  Sections 4.1–4.4, models post-subgoal outcomes with variable duration and a
  resource allocation policy. Its combinatorial setting and primitive-resource
  budget differ from our visual manipulation and inference-time budget, but
  merely adding a budget argument to a high-level transition model is not a
  sufficient novelty claim.
- [EA-WM](https://arxiv.org/html/2606.13053v1), Sections 4.1–4.3, adds a task-event
  predictor/verifier to a visual-feature world model using simulator-derived
  task labels. A generic progress-verification head is also not a new direction
  by itself. Those privileged labels are not part of our proposed image-only
  formulation.
- Hi-LeWM already identifies compatibility between coarse search and the fixed
  low-level controller as central. The distinction must therefore go beyond
  identifying that compatibility problem or using empirical action support.

These additional sources were inspected for their relevant formulations;
their implementations were not audited or reproduced in this turn. This
targeted search is not an exhaustive proof of novelty or non-novelty.

## What a central mechanism would need to change

The target should concern the controller's result, rather than only the
transition induced by an arbitrary recorded macro-action. Let `C_K(z,q;g)` be
the deployed local action search toward target `q`, conditioned on final goal
`g`, with its fixed inference work allowance `K`, and let `c` be the committed
prefix length. Its immediate
model-predicted effect is

`hat_z_exec = F^c(z, prefix_c[C_K(z,q;g)])`.

In our protocol `c=1` control block, or two primitive actions. A multidecision
prediction would additionally have to model observations, replanning, target
changes and the remaining cumulative controller allowance. Physical execution
horizon and optimizer/inference budget are distinct variables.

This equation only specifies the missing relationship. It is a composition
of existing operations, **not a proposed novel algorithm or proof of physical
reachability**. A defensible method contribution would require a new efficient
way to estimate or correct this relationship, a valid supervision source and a
clear distinction from controller-aware HRL, model-based planning and planning
distillation. Adding a standard critic or distilling this composition alone
does not automatically meet that standard.

If such a mechanism is developed, it should replace the central coarse target
or scoring objective, not become another unexplained weighted auxiliary loss.
Retrieval would supply supported targets, the mixture would supply efficient
actions, and local CEM would implement control. Their necessity would follow
from the central mechanism. No new learned component receives an extra budget.

## The fixed-data constraint is substantive

Our offline trajectories record outcomes for the actions actually taken. They
do not contain true outcomes for every new controller-generated action or
candidate subgoal at each state. A synthetic rollout through the same frozen
world can teach agreement with that model; it cannot independently establish
real executability or correct unknown model errors. Hindsight success on a
recorded trajectory also does not prove the new controller can reproduce it.

Any reformulation must explain what can be learned from the allowed data,
which support/generalization assumptions it needs, and what remains a model
proxy. It must not quietly obtain counterfactual labels from validation resets,
collect new simulator trajectories, or add online corrective training. No such
data collection is authorized or scheduled. The original data, single seed,
aggregate training ceiling and registered evaluation protocol remain fixed.

The eventual primary evidence must still be task success within the controller
allowance. The first results can include failure logging from the same episodes.
Later attribution experiments might be needed for a publication, but the user
has explicitly deferred ablations; none is queued by this audit.

## Current decision

Retain the previous document as an engineering reference, not the settled
paper-level formulation. The research focus should be the controller's actual
achievable progress under bounded inference, but **a distinct, feasible central
mechanism remains unresolved**. Do not claim a new method merely by naming this
focus or assembling more components around it.

This audit changes research positioning, not experiment state. No implementation,
training, baseline rerun, GPU benchmark, simulator diagnostic or test was run.
Verification: re-read this saved audit, the positioning update in the previous
formulation and the progress entry; checked local links and the repository diff.
