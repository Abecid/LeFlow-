# Conditional BTM subgoal model

For a batch size B, latent dimension D=192, H=5 path transitions, and physical
spacing q in {1,2,4} action blocks, the cached target is
`z_path: [B,H+1,D]`. With five primitive actions per block, the target endpoints
are 25, 50, or 100 actions apart. The interiors `y=z_path[:,1:-1]` have shape
`[B,4,192]`. Conditions are the current latent `z0: [B,192]`, final-goal latent
`zg: [B,192]`, and `q: [B]`.

Draw independent Gaussian noise `epsilon` of the same shape as y and an
interpolation coordinate `s: [B]` uniformly in [0,1]. Define

\[
x_s=(1-s)\epsilon+sy,\qquad d=y-\epsilon.
\]

Flow matching predicts a velocity `v(x_s,s,z0,zg,q)` with target d. Euler sampling
uses 2/4/8/16 calls as configured. The BTM model predicts a terminal map
`T(x_s,z0,zg,q)` with **no s input**. Its objective is

\[
L_{BTM}=\operatorname{MSE}\left(T(x_s),
  \operatorname{stopgrad}[T(x_s)+J_T(x_s)d]\right)
  +\lambda\operatorname{MSE}(T(y),y),\qquad\lambda=1.
\]

All appearances of T keep the conditions fixed; the JVP differentiates only with
respect to the interior-state tensor. The implementation detaches the entire
right-hand side. Differentiating `||J_T d||²` instead would change the algorithm.
`tests/test_core.py` checks the gradient analytically on a linear map. Sampling
is one call `T(epsilon,z0,zg,q)` followed by concatenation of the two endpoints.
The physical duration q remains a condition: eliminating the artificial
generative time does not eliminate how long a robot transition takes.

The map uses residual parameterization `T(x,c)=x+f(x,c)` with a zero-initialized
output layer. No pretrained teacher, source regression term, pixel decoder, or
new representation-learning loss is required for this first comparison.

Both generators share a short-step inverse model. Its input concatenates
`[z_t,z_next,z_next-z_t]: [B,K,576]`; its output is
`[B,K,5*A]`, where A is the primitive action dimension (2 for PushT).
Training uses observed short transitions (`K=4` by default), action MSE, and a
0.1-weighted frozen-LeWM consistency loss. This consistency term updates inverse
dynamics through the frozen predictor; it is not a loss on generated paths.

At deployment:

1. Sample 64 conditional waypoint chains `[B,64,6,192]`.
2. Decode short action blocks toward successive waypoints and roll them through
   the action-conditioned model. For coarse spacing, local targets interpolate
   toward the next waypoint; this is a proposal heuristic, not a learned
   macro-action model or a guarantee of physical reachability.
3. Rank using actual model-rolled actions: final-goal MSE plus mean waypoint MSE
   in hierarchical mode. A clamped generated endpoint is never the score.
4. Refine the chosen first segment with short-horizon CEM and execute one action
   block. Observe the real environment and replan.

The frozen LeWM and CEM still cost compute. One generator call is not one call
for the complete robot controller. All reachability estimates remain predictions;
closed-loop environment success is the primary downstream metric.

References: [BTM, Eq. (20)](https://arxiv.org/html/2608.01692v3),
[LeFlow](https://arxiv.org/abs/2608.24855),
[HWM](https://arxiv.org/abs/2604.03208), and
[Planning Limits](https://arxiv.org/abs/2609.39235).
The latter two motivate the long-horizon problem; this release does not reproduce
their V-JEPA robot implementations. BTM's population theorem has support and
regularity assumptions; it is not an empirical guarantee for these JEPA latents.
