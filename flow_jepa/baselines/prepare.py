"""Register a new method protocol while preserving the existing dataset."""
import argparse
import json
from pathlib import Path

from ..common import config, digest, file_hash, save_json
from ..budget import verify_data_compatibility


def prepare(c, source, root, world, source_config):
    source, root = Path(source), Path(root)
    verify_data_compatibility(config(source_config), c)
    old = json.loads((source / 'manifest.json').read_text())
    assert not old.get('fixture', True)
    assert file_hash(world) == c['baseline']['world_sha256']
    if root.exists() and (root / 'manifest.json').exists():
        now = json.loads((root / 'manifest.json').read_text())
        assert now['protocol'] == digest(c)
    else:
        root.mkdir(parents=True, exist_ok=True)
        for name in ('episodes', 'weights', 'vendor'):
            (root / name).symlink_to((source / name).resolve(), target_is_directory=True)
        now = dict(old, protocol=digest(c))
        save_json(root / 'manifest.json', now)
        save_json(root / 'configuration.json', c)
    for key in ('entries', 'encoder', 'mean', 'std', 'goal_screening', 'expert_success'):
        assert now[key] == old[key], key
    save_json(root / 'reuse.json', dict(source_root=str(source),
        source_manifest=file_hash(source / 'manifest.json'), entries_sha256=digest(old['entries']),
        world_sha256=file_hash(world), encoder_sha256=old['encoder'],
        unchanged_fields=['entries', 'encoder', 'mean', 'std', 'goal_screening', 'expert_success'],
        method=c['primary_method'], test_enabled=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('config', 'source', 'root', 'world', 'source-config'): p.add_argument('--' + k, required=True)
    a = p.parse_args(); prepare(config(a.config), a.source, a.root, a.world, a.source_config)
