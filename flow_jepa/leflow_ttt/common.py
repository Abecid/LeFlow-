"""Reuse exactly the baseline adapter and charge its original training prefix."""
import json
from pathlib import Path
import torch
from ..common import file_hash


def load_common_adapter(model,c):
    cfg=c['leflow_ttt']; source=Path(cfg['adapter_source'])
    registry=json.loads(Path(cfg['adapter_source_registry']).read_text())
    expected=registry['files'][source.name]
    assert file_hash(source)==expected
    saved=torch.load(source,map_location='cpu',weights_only=False)
    assert saved['method']=='leflow_release' and saved['seed']==3072 and not saved['fixture']
    native={k.removeprefix('adapter.'):v for k,v in saved['model'].items() if k.startswith('adapter.')}
    model.adapter.load_state_dict(native,strict=True)
    model.set_stage(False)
    logs=[json.loads(x) for x in (source.parent/'metrics.jsonl').read_text().splitlines()]
    rows=[x for x in logs if x.get('step')==cfg['common_adapter_steps'] and 'budget/optimization_gpu_hours' in x]
    assert len(rows)==1 and 'train/adapter/loss' in rows[0]
    gpu_seconds=3600*rows[0]['budget/optimization_gpu_hours']
    assert gpu_seconds>0
    return dict(source_checkpoint_sha256=expected,adapter_steps=cfg['common_adapter_steps'],
                adapter_optimization_gpu_seconds=gpu_seconds,baseline_source=saved['code'],
                policy_weights_inherited=False)
