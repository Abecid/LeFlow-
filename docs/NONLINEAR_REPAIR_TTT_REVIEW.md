# Nonlinear memory review: retain 80 and hold further launches

The single authorized nonlinear candidate completed 73/66/67/72, selecting 5k 73/104 below retrieval-repair 80 and execution-revision 80. All four fixed evaluations are retained. This does not support replacing either reference.

The memory is used and learns: every supported fit improves, and aggregate next-factual-prefix error improves after adaptation. Yet fitted support error, next-prefix model distance and physical task success are different targets. The empty-history learned prior worsens error relative to raw prediction. The current evidence does not identify why that occurs or demonstrate that removing it would improve physical success.

Later action/repair losses and teacher clipping decrease, while initial proposal spread contracts and success stays below 5k. Proposal contraction also occurred in earlier improving runs; it is not a causal label or a warrant for a diversity/radius/weight sweep. Shifted-plan training coverage and prefix-only supervision remain inherited approximations, not newly established physical failure causes.

The four 10k timeouts are preserved without replay. Sustained slow calls in two pick-place episodes are observed; their cause is unknown. All four cases already failed at 5k. No timing evidence supports inventing their success under a longer deadline. The saved trajectories show failures and successes at sparse sampled times, not counterfactual contact dynamics.

This is a LaCT-inspired small-context adaptation, not full LaCT/Muon reproduction. Eight-rank training versus the four-rank retrieval reference changes random streams and within-rank negatives despite fixed data/global batch. One seed and reused development selection do not establish generalization, component benefit, novelty or SOTA. No oracle task-ID filtering or generated-action success labels are justified.

The October 12 authorization covers this one candidate. No next run, extra seed, new data, baseline retraining, inference/ablation sweep, budget extension or sealed final test is authorized. Preserve the 80/104 reference, source/baselines and all negative results. Keep the existing monitor active for subsequently explicitly authorized and registered work; never restart this completed coordinator or its workers.

See [complete findings](NONLINEAR_REPAIR_TTT_RESULTS.md), [full diagnostics](reports/20261012-nonlinear-repair-ttt/analysis/final-details.json) and [CPU identity/accounting audit](reports/20261012-nonlinear-repair-ttt/analysis/final-audit.json).
