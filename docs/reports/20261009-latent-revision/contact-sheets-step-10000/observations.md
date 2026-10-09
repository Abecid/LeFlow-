# Visual observations: selected 10k checkpoint

All ten complete 1536×326 contact sheets were inspected. These are saved model
rollouts, with five uniformly sampled frames and the registered image goal.
Captions, action indices and all panels are readable; no clipping, overlap or
missing glyphs was observed.

Assembly0 approaches the object region but makes little visible change in the
later sampled frames and fails at 200 actions. Pick-place0 moves the gripper
toward the target region while the red object remains near its original table
location, unlike the goal view. Faucet-open3 fails to reach the goal arrangement;
the later gripper positions move past the faucet region. Reach3 ends offset from
the goal, whereas reach5 succeeds in 44 actions after failing at the 5k checkpoint.
The remaining five saved cases succeed: coffee-button0, dial-turn0, door-close0,
drawer-close4 and handle-press0. Sparse images do not establish contact forces,
precise grasp failures or a causal effect of the latent module.
