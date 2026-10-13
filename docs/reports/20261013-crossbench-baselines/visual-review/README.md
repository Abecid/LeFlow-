# Native saved-trajectory visual review

All 16 finished contact sheets were visually inspected in full on October 13, 2026. They contain all saved frames and recorded goals from the 160 archived selected-checkpoint trajectories: the first ten registered case-plan indices per run. No experiment, model inference or simulator rollout was repeated.

The 50-case reports remain the quantitative evidence. This fixed ten-case subset is not representative: Cube SAGE has ten displayed successes despite one failure in its full report; PushT diffusion has ten displayed failures despite eleven successes in its full report. Success labels use recorded native outcomes, not visual judgment. Source-camera occlusion and frame boundaries remain intact. Blank columns on short trajectories are intentional.

The evaluator stores NPZ files by case-plan index and sorts report rows by case ID. The composition reconstructs the original order from seeds and verifies the entire case-plan hash before applying labels. Initial and goal pixels match exactly across methods for every displayed case. Original 224×224 pixels remain in the NPZs; sheets display them at 176×176, without cropping or overlays.

| Dataset | Method | Displayed successes | Finished image |
| --- | --- | ---: | --- |
| cube | action_flow_search | 4/10 | [Contact sheet](cube-action_flow_search.png) |
| cube | diffusion_policy_port | 8/10 | [Contact sheet](cube-diffusion_policy_port.png) |
| cube | execution_revision | 5/10 | [Contact sheet](cube-execution_revision.png) |
| cube | feedback_wm_port | 9/10 | [Contact sheet](cube-feedback_wm_port.png) |
| cube | leflow | 8/10 | [Contact sheet](cube-leflow.png) |
| cube | retrieval_repair_ttt | 5/10 | [Contact sheet](cube-retrieval_repair_ttt.png) |
| cube | sage_port | 10/10 | [Contact sheet](cube-sage_port.png) |
| cube | vanilla_action_flow | 7/10 | [Contact sheet](cube-vanilla_action_flow.png) |
| pusht | action_flow_search | 6/10 | [Contact sheet](pusht-action_flow_search.png) |
| pusht | diffusion_policy_port | 0/10 | [Contact sheet](pusht-diffusion_policy_port.png) |
| pusht | execution_revision | 7/10 | [Contact sheet](pusht-execution_revision.png) |
| pusht | feedback_wm_port | 4/10 | [Contact sheet](pusht-feedback_wm_port.png) |
| pusht | leflow | 3/10 | [Contact sheet](pusht-leflow.png) |
| pusht | retrieval_repair_ttt | 7/10 | [Contact sheet](pusht-retrieval_repair_ttt.png) |
| pusht | sage_port | 9/10 | [Contact sheet](pusht-sage_port.png) |
| pusht | vanilla_action_flow | 3/10 | [Contact sheet](pusht-vanilla_action_flow.png) |

[Source and image hashes](index.json) and [per-sheet inspection findings](inspection.json) record coverage and limitations. No missing panels, unreadable labels, overlapping text or introduced clipping were observed in the exact PNGs identified by those hashes. This visual review does not establish a causal method advantage or generalization.
