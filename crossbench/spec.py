"""Immutable scope for the three-benchmark comparison.

MetaWorld is reused. New PushT/Cube jobs use compact released LeWM features.
They are shared-backbone, matched-execution ports, not native reproductions.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

METHODS = (
    'leflow', 'vanilla_action_flow', 'execution_revision', 'action_flow_search',
    'sage_port', 'diffusion_policy_port', 'feedback_wm_port', 'retrieval_repair_ttt',
)


@dataclass(frozen=True)
class Spec:
    dataset: str
    method: str
    dim: int = 192
    patches: int = 1
    primitive_dim: int = 2
    action_block: int = 5
    horizon: int = 5
    width: int = 256
    seed: int = 3072
    global_batch: int = 64
    updates: int = 20000
    optimization_gpu_seconds: float = 28800.0
    eval_episodes: int = 50
    goal_offset: int = 25
    action_budget: int = 50
    controller_seconds: float = 10.0
    flow_steps: int = 8
    repair_steps: int = 4
    repair_radius: float = 0.25
    history: int = 4
    candidates: int = 4
    targets: int = 8
    rounds: int = 3
    learning_rate: float = 1.e-4
    weight_decay: float = .05
    protocol: str = 'crossbench-shared-lewm-v1'

    def __post_init__(self):
        if self.dataset not in ('pusht', 'cube') or self.method not in METHODS:
            raise ValueError('Only the two new datasets and eight registered methods are allowed')
        if self.patches != 1 or self.horizon != 5:
            raise ValueError('Released compact LeWM state and five-block planning are required')
        if self.primitive_dim != {'pusht': 2, 'cube': 4}[self.dataset]:
            raise ValueError('Native primitive action dimension mismatch')
        if min(self.dim, self.action_block, self.eval_episodes, self.updates,
               self.global_batch, self.action_budget, self.goal_offset) <= 0:
            raise ValueError('Dimensions, budgets and episode counts must be positive')
        if self.width % 4 or self.width < 16 or self.controller_seconds <= 0:
            raise ValueError('Invalid width or controller allowance')

    @property
    def ad(self):
        return self.primitive_dim * self.action_block

    @property
    def flat(self):
        return self.horizon * self.ad

    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def matrix():
    return {
        'schema': 1,
        'metaworld': {'status': 'reuse_completed', 'source_commit':
            'a5ffc726c23fe6069a4e4a95239c4c08d2c81005',
            'results': 'docs/results-ledger.json', 'retrain': False},
        'jobs': [asdict(Spec(dataset=d, method=m, primitive_dim=2 if d=='pusht' else 4))
                 for d in ('pusht','cube') for m in METHODS],
        'status': 'implementation_only_until_native_preflight_and_server_launch',
        'sealed_final_tests_enabled': False,
    }


if __name__ == '__main__':
    print(json.dumps(matrix(), indent=2))
