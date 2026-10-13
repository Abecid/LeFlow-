"""Compose native rollout evidence from saved pixels; no model or simulator calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from result_io import read_record

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
a = p.parse_args()
r = a.repo
base = r / 'docs/reports/20261013-crossbench-baselines'
out = base / 'visual-review'
out.mkdir(exist_ok=True)
ledger = json.loads((r / 'docs/crossbench-results-ledger.json').read_text())
font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
font = ImageFont.truetype(font_path, 14)
title_font = ImageFont.truetype(font_path, 22)
sources, sheets, shared = [], [], {}
size, gap, margin, row_height = 176, 6, 12, 232
width = 2 * margin + 7 * size + 6 * gap
for run in ledger['completed_runs']:
    dataset, method = run['dataset'], run['method']
    report_path = next(x['report'] for x in run['curve'] if x['step'] == run['selected_step'])
    report = read_record(r / report_path)
    # The evaluator saves NPZs by unsorted case-plan index, then sorts report rows by ID.
    ordered = sorted(report['records'], key=lambda x: x['seed'])
    fields = ['id', 'episode', 'source_offset', 'start', 'goal_offset', 'seed']
    plan = [{k: x[k] for k in fields} for x in ordered]
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert digest == report['identity']['cases']
    assert [x['seed'] for x in ordered] == list(range(103072, 103122))
    files = sorted((r / run['evidence'] / 'selected-trajectories').glob('*.npz'))
    assert [x.stem for x in files] == [f'{i:05d}' for i in range(10)]
    sheet = Image.new('RGB', (width, 96 + 10 * row_height + 34), '#f4f5f6')
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 12), f'{dataset.upper()} / {method} | selected {run["selected_step"]:,} updates', fill='#15212c', font=title_font)
    draw.text((margin, 44), 'Measured saved frames: first 10 registered cases, all archived samples. Final column: recorded goal.', fill='#23394a', font=font)
    draw.text((margin, 66), 'Labels use native success reports. These images do not replace the full 50-case evaluation or its success predicate.', fill='#23394a', font=font)
    entries = []
    for row, path in enumerate(files):
        record = ordered[int(path.stem)]
        with np.load(path, allow_pickle=False) as z:
            frames, indices, goal = z['frames'], z['action_indices'], z['goal']
        assert frames.dtype == goal.dtype == np.uint8
        assert frames.shape[1:] == goal.shape == (224, 224, 3)
        assert len(frames) == len(indices) <= 6
        assert indices.tolist() == np.linspace(0, record['steps'], min(6, record['steps'] + 1), dtype=int).tolist()
        key = (dataset, record['id'])
        initial_goal = (hashlib.sha256(frames[0].tobytes()).hexdigest(), hashlib.sha256(goal.tobytes()).hexdigest())
        assert shared.setdefault(key, initial_goal) == initial_goal, key
        top = 96 + row * row_height
        label = f'{path.stem} | {record["id"]} | {"SUCCESS" if record["success"] else "FAILURE"} | {record["steps"]} actions | {record["failure_reason"]}'
        draw.text((margin, top), label, fill='#164e3b' if record['success'] else '#862d24', font=font)
        for col, (frame, action) in enumerate(zip(frames, indices)):
            x = margin + col * (size + gap)
            sheet.paste(Image.fromarray(frame).resize((size, size), Image.Resampling.LANCZOS), (x, top + 24))
            draw.text((x + 4, top + 204), f'Action {int(action)}', fill='#172832', font=font)
        x = margin + 6 * (size + gap)
        sheet.paste(Image.fromarray(goal).resize((size, size), Image.Resampling.LANCZOS), (x, top + 24))
        draw.text((x + 4, top + 204), 'Recorded goal', fill='#172832', font=font)
        entry = dict(case_index=int(path.stem), id=record['id'], success=record['success'], steps=record['steps'], failure_reason=record['failure_reason'], action_indices=indices.tolist(), npz=str(path.relative_to(r)), npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), initial_rgb_sha256=initial_goal[0], goal_rgb_sha256=initial_goal[1])
        entries.append(entry)
        sources.append(entry)
    draw.text((margin, sheet.height - 27), 'Saved evidence only. No rollout rerun, model inference, interpolation of motion, or fabricated frames.', fill='#23394a', font=font)
    dest = out / f'{dataset}-{method}.png'
    sheet.save(dest)
    sheets.append(dict(dataset=dataset, method=method, selected_step=run['selected_step'], report=report_path, report_sha256=hashlib.sha256((r / report_path).read_bytes()).hexdigest(), file=str(dest.relative_to(r)), sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), width=sheet.width, height=sheet.height, cases=entries))
index = dict(kind='Measured native validation frames, composed from all 160 archived selected-checkpoint NPZ files', sheets=sheets, sheet_count=len(sheets), archived_case_count=len(sources), all_archived_frames_displayed=True, same_initial_and_goal_pixels_across_methods=True, source_frame_size=[224, 224], display_frame_size=[176, 176], resize='Lanczos, no cropping or pixel overlays', source='747f635ad36af2cbe3e2bbb0b7997efe9caccaa1', model_calls=0, simulator_calls=0, optimizer_updates=0, sealed_test_reads=0, visual_inspection='Record separately after viewing every finished sheet')
(out / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
print(json.dumps({'sheets':len(sheets),'archived_cases':len(sources),'output':str(out)}))
