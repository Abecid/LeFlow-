"""Bounded, no-optimizer TRAIN-only stage diagnostic. Never a task policy."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F


def main(a):
    began = time.perf_counter(); torch.set_num_threads(8)
    bank = torch.load(a.bank, map_location='cpu', mmap=True, weights_only=False)
    manifest = json.loads(Path(a.manifest).read_text())
    rows = {r['id']: r for r in manifest['entries'] if r['split'] == 'train'}
    assert all(i.startswith('train/') for i in bank['episode_ids'])
    groups = {}
    for i, identity in enumerate(bank['episode_ids']):
        row = rows[identity]
        if row['first_success_action'] >= 30:
            groups.setdefault(row['task'], []).append(i)
    rng = np.random.default_rng(3072)
    projection = torch.randn(bank['raw'].shape[-1], 16,
                             generator=torch.Generator().manual_seed(3072)) / 4
    features, stages, tasks, splits, costs, ids = [], [], [], [], [], []
    for task, episodes in sorted(groups.items()):
        selected = rng.choice(episodes, 60, replace=False)
        for j, ep in enumerate(selected):
            row = rows[bank['episode_ids'][ep]]
            indices = (bank['episode'] == int(ep)).nonzero()[:, 0]
            success = row['first_success_action'] // 2
            goal_index = min(round(success / 5), len(indices)-1)
            phase_indices = [min(round(success*f/5), goal_index-1) for f in (.15,.5,.85)]
            states = (bank['raw'][indices[phase_indices + [goal_index]]].float()-bank['mean'])/bank['std']
            normalized = F.normalize(states, dim=-1)
            goal = normalized[-1:].expand(3,-1,-1)
            x = normalized[:3]
            # Fixed random projection preserves spatial positions; neither time,
            # task identity nor stage labels enter the query feature.
            feat = torch.cat((x@projection, goal@projection, (x-goal)@projection), -1).flatten(1)
            features.append(feat); stages.extend(range(3)); tasks.extend([task]*3)
            splits.extend([j >= 40]*3); ids.append(bank['episode_ids'][ep])
            costs.append(1-(x*goal).sum(-1).mean(-1))
    x=torch.cat(features); y=torch.tensor(stages); test=torch.tensor(splits)
    d=torch.cdist(x[test], x[~test]); neighbors=d.topk(5,largest=False).indices
    predicted=y[~test][neighbors].mode(1).values
    correct=predicted==y[test]; test_tasks=np.array(tasks)[test.numpy()]
    ordering=torch.stack(costs)
    pair_order=torch.stack((ordering[:,0]>ordering[:,1],ordering[:,1]>ordering[:,2],ordering[:,0]>ordering[:,2]),1)
    per_task={t:dict(heldout_stage_accuracy=float(correct[torch.from_numpy(test_tasks==t)].float().mean()),
                     heldout_examples=int((test_tasks==t).sum())) for t in sorted(groups)}
    result=dict(training_only=True,optimizer_updates=0,gpu_seconds=0,
        protocol='40 reference +20 held-out TRAIN episodes per task; 3 coarse temporal stages; global cross-task 5NN; fixed spatial random projection; no fit',
        temporal_labels_are_weak_not_physical_completion=True,task_ids_used_only_for_sampling_and_reporting=True,
        training_episode_ids=ids,reference_examples=int((~test).sum()),heldout_examples=int(test.sum()),
        stage_chance=1/3,heldout_stage_accuracy=float(correct.float().mean()),
        raw_distance_pair_order_accuracy=float(pair_order.float().mean()),per_task=per_task,
        bank_source=a.bank,bank_sha256=json.loads(Path(a.bank+'.json').read_text())['sha256'],
        cpu_seconds=time.perf_counter()-began)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='training_episode_ids'}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('bank','manifest','output'):p.add_argument('--'+key,required=True)
    main(p.parse_args())
