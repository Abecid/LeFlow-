"""Evaluate a saved comparison checkpoint on its exact registered validation cases."""
import argparse
import json
from pathlib import Path
import torch
from flow_jepa.common import file_hash
from .spec import Spec
from .models import build
from .world import load_world
from .data import RouteBank
from .evaluate import evaluate
from .train import RETRIEVAL, code_hash


def main():
    from .environment import configure_renderer_device
    configure_renderer_device()
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache','checkpoint','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args();root=Path(a.cache)
    device=torch.device(a.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    payload=torch.load(a.checkpoint,map_location=device,weights_only=False)
    identity=payload['identity'];s=Spec(**identity['spec'])
    meta=json.loads((root/'manifest.json').read_text())
    if identity.get('fixture') or meta.get('fixture'):raise ValueError('No fixture scores in formal evaluation')
    if identity['source']!=code_hash():raise ValueError('Evaluate from the trained frozen source revision')
    if identity['cache']!=file_hash(root/'cache.h5') or identity['world']!=meta['checkpoint_sha256']:
        raise ValueError('Evaluation cache/world differs from training')
    model=build(s).to(device);model.load_state_dict(payload['model'],strict=True);model.eval()
    world=load_world(meta,device)
    bank=RouteBank(root,s,device) if s.method in RETRIEVAL else None
    metrics=evaluate(model,world,bank,s,root,a.output,file_hash(a.checkpoint),device)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
