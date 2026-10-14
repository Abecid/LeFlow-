# JEPA-scored noise-space CEM for LeFlow

## Scope

The user requested implementation and publication of noise-space CEM and matched
independent-sampling controls. This is an inference-only extension of the existing
`leflow_refinement` framework, not a new trained model or a LeWAM reproduction.
The encoder, flow, inverse dynamics and world predictor remain frozen. Candidate
quality is the existing predicted terminal latent MSE; no new critic is added.
Existing P&P and world-guided P&P implementations, immutable execution checkouts,
completed results and sealed tests must remain unchanged.

This change does not launch GPU campaigns, retrain models or expand the separately
registered October 14 self-refinement campaign. Benchmark gains are not established.

## Implementation plan

Search the initial noise tensor of shape `[B,N,H-1,D]`. Initialize independent
per-environment diagonal Gaussian search distributions with zero mean and unit
variance. Evaluate complete flow-generated paths and the actions decoded from
those paths. Fit elite moments with smoothing and a positive variance floor.
Retain the best *evaluated* candidate across all rounds, never average actions.

Provide both unprojected Gaussian and radius-`sqrt((H-1)*D)` shell-projected
variants. Shell normalization applies to the complete path-interior noise vector,
not separately to temporal tokens. Resample zero-norm proposals. Preserve raw
proposal/cost pairs for the Gaussian moment fit; the projected noise is what the
flow receives. A projected independent-sampling control isolates radial effects.

The comparison uses 3 rounds x 64 candidates x 16 flow steps versus 192 independent
candidates x 16 steps: 3,072 candidate-level flow evaluations and 960 predicted
world transitions per environment at horizon 5. Actual latency must still be
measured; equal operation counts do not imply equal latency.

## Status

Implementation in progress. No new benchmark results are claimed. This document
will be updated with executable entry points, tests and publication evidence.
