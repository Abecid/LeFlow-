# Visual observations: selected 20k checkpoint

Pick-place0 approaches the red object, then shows very similar gripper/object
poses from action 50 through 200. The object remains near its starting table
location, different from the registered goal. Faucet-open3 approaches the
faucet region, but its later gripper positions drift right of it and the final
arrangement differs from the goal. Reach3 remains offset from its goal with
little visible change in the later sampled frames. All three fail at 200 actions.
The views do not establish contact forces, a precise grasp failure mechanism,
or the causal contribution of the latent workspace.

Assembly0 succeeds in 84 actions, recovering from its 10k failure. Reach5 also
succeeds, in 50 actions, after failing at 5k. Coffee-button0, dial-turn0,
door-close0, drawer-close4 and handle-press0 are the other five saved successes.
Task success follows the simulator criterion, not exact pixel equality with
the goal image. All ten complete contact sheets were inspected at 1536×326;
panels, case labels, action indices and goal captions are intact and readable.
