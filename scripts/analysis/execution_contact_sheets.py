"""Render saved validation frames only; never invokes a model or simulator."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main(a):
    root, run, dest = Path(a.root), Path(a.run_dir), Path(a.output)
    dest.mkdir(parents=True, exist_ok=True)
    report=json.loads((run/'validation'/f'step_{a.step:07d}.json').read_text())
    records={r['id']:r for r in report['records']}
    manifest=json.loads((root/'manifest.json').read_text())
    rows={r['id']:r for r in manifest['entries'] if r['split']=='validation'}
    try:
        font=ImageFont.truetype('DejaVuSans.ttf',16)
    except OSError:
        font=ImageFont.load_default()
    written=[]
    for path in sorted((run/'trajectories'/f'step_{a.step:07d}').glob('*/*.npz')):
        ident=f'validation/{path.parent.name}/{path.stem}'
        r=records[ident]; row=rows[ident]
        with np.load(path) as data:
            frames=data['frames']
        assert len(frames)==r['steps']+1 and frames.dtype==np.uint8
        with h5py.File(root/row['path'],'r') as f:
            goal=f['goal_rgb'][:]
            assert np.array_equal(frames[0],f['initial_rgb'][:])
        indices=np.linspace(0,len(frames)-1,5,dtype=int).tolist()
        sheet=Image.new('RGB',(6*256,326),'white');draw=ImageDraw.Draw(sheet)
        title=f"{ident} | checkpoint {a.step} | {'SUCCESS' if r['success'] else 'FAILURE'} | {r['steps']} primitive actions"
        draw.text((8,8),title,fill='black',font=font)
        for i,t in enumerate(indices):
            sheet.paste(Image.fromarray(frames[t]),(i*256,42))
            draw.text((i*256+8,302),f'Action {t}',fill='black',font=font)
        sheet.paste(Image.fromarray(goal),(5*256,42))
        draw.text((5*256+8,302),'Registered image goal',fill='black',font=font)
        output=dest/f'{path.parent.name}-{path.stem}.png'
        sheet.save(output)
        written.append(dict(id=ident,success=r['success'],steps=r['steps'],
                            sampled_primitive_indices=indices,file=output.name,
                            source=str(path),registered_episode_sha256=row['sha256']))
    (dest/'index.json').write_text(json.dumps(dict(checkpoint=a.step,
        kind='Measured saved rollout frames, uniformly sampled; final panel is registered goal image',
        simulator_calls=0,model_calls=0,images=written),indent=2)+'\n')
    print(json.dumps(dict(images=len(written),output=str(dest))))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','run-dir','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--step',type=int,required=True)
    main(p.parse_args())
