"""Released-world and native-architecture path-gradient integration.

No benchmark dataset, optimizer or success-rate claim. Physical actions are
bounded and actually executed. Outputs are explicitly interface-check records.
"""
import json
from pathlib import Path
import sys
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from huggingface_hub import hf_hub_download, HfApi
from crossbench.environment import NativeEnv
from crossbench.world import ActionUnits, LeWMWorld
from leflow_refinement.runtime import NativeRuntime, variants


def main():
    torch.set_num_threads(1)
    output = Path('refinement-checks'); output.mkdir(exist_ok=True)
    rows = []
    availability = []
    for task in ('pusht', 'cube'):
        world_path = hf_hub_download('quentinll/lewm-'+task, 'weights.pt')
        policy_path = None
        try:
            policy_path = hf_hub_download('hsiangwei0903/LeFlow', task+'/latent_planner.pt')
            availability.append(dict(task=task, pretrained_head='available'))
        except Exception as error:
            from huggingface_hub.errors import HfHubHTTPError
            if not isinstance(error, HfHubHTTPError): raise
            availability.append(dict(task=task, pretrained_head='blocked', error=str(error)))
            print('PRETRAINED_HEAD_BLOCKED', task, str(error), flush=True)
        (output/'pretrained_head_status.json').write_text(json.dumps(availability, indent=2))
        env = NativeEnv(task)
        try:
            env.env.reset(seed=3072); image = env.render()
            env.env.reset(seed=3073); goal_image = env.render()
            low, high = env.action_space.low, env.action_space.high
            # Synthetic interface statistics, NOT benchmark action normalization.
            units = ActionUnits(low, high, (low+high)/2, (high-low)/2, block=5)
            from latent_planner import load_lewm, LatentPlannerRuntime, LatentPathFlow, InverseDynamics
            lewm = load_lewm(world_path)
            world = LeWMWorld(lewm, units)
            z = world.encode(image[None])[:, 0]
            g = world.encode(goal_image[None])[:, 0]
            ad = len(low)*5
            if policy_path is not None:
                adapter = NativeRuntime.load(policy_path, world_checkpoint=world_path, device='cpu')
                head_source = 'released_trained_head'
            else:
                # Explicit architecture-only fallback, NEVER a performance run.
                torch.manual_seed(19)
                flow = LatentPathFlow(z.shape[-1])
                inverse = InverseDynamics(z.shape[-1], ad)
                adapter = NativeRuntime(LatentPlannerRuntime(lewm, flow, inverse, action_block=5))
                head_source = 'randomly_initialized_head_interface_check_only'
            assert adapter.model.inverse_dynamics.action_dim == ad
            adapter.lower = torch.full((ad,), -1.)
            adapter.upper = torch.full((ad,), 1.)
            original = {k:v.clone() for k,v in adapter.model.state_dict().items()}
            for name in ('leflow', 'pnp', 'world_gradient', 'world_pnp'):
                cfg = replace(variants()[name], samples=2, steps=4, refine_at=(1,2), repeats=1, consistency_weight=0.)
                with torch.inference_mode():
                    result = adapter.plan(z, g, cfg, seed=7)
                action = result['actions'][0, 0]
                assert action.shape == (ad,) and torch.isfinite(action).all()
                assert action.abs().max() <= 1
                if 'world' in name:
                    assert result['telemetry']['gradient_norm_sum'] > 0
                assert all(p.grad is None for p in adapter.model.parameters())
                for k,v in adapter.model.state_dict().items():
                    torch.testing.assert_close(v, original[k], atol=0, rtol=0)
                physical = units.physical(action).reshape(5, len(low)).numpy()
                env.env.reset(seed=3072)
                for primitive in physical:
                    rgb, _, _, done = env.step(primitive)
                    if done: break
                assert rgb.shape == (224,224,3)
                row = dict(task=task, variant=name, interface_check_only=True, head_source=head_source,
                    latent_dim=z.shape[-1], action_dim=ad, telemetry=result['telemetry'],
                    checkpoint=adapter.provenance)
                rows.append(row)
                print('WORLD_REFINEMENT_INTERFACE_OK', json.dumps(row), flush=True)
        finally:
            env.close()
    (output/'released_interfaces.json').write_text(json.dumps(rows, indent=2))
    sizes = []
    for repo in ('quentinll/lewm-pusht','quentinll/lewm-reacher',
                 'quentinll/lewm-tworooms','quentinll/lewm-cube'):
        info = HfApi().dataset_info(repo, files_metadata=True)
        sizes.append(dict(repo=repo, revision=info.sha,
            files=[dict(path=s.rfilename, size=s.size) for s in info.siblings]))
    (output/'dataset_sizes.json').write_text(json.dumps(sizes, indent=2))
    print('DATASET_ARCHIVES', json.dumps(sizes), flush=True)


if __name__ == '__main__':
    main()
