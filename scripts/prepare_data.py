#!/usr/bin/env python
"""Download released LeWM data/weights, safely extract, inspect, and split.

No images or credentials are uploaded. Archives are kept for resumable downloads;
use --remove-archive to remove a verified archive after successful extraction.
"""

import argparse
import json
import shutil
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import HfApi, hf_hub_download
import zstandard as zstd
from btm_jepa.data import (
    TASKS,
    atomic_json,
    cache_root,
    episode_split,
    inspect_source,
    file_sha256,
)


def extract(archive, target, expected_name):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".h5.partial")
    try:
        with (
            open(archive, "rb") as raw,
            zstd.ZstdDecompressor().stream_reader(raw) as stream,
        ):
            if str(archive).endswith(".tar.zst"):
                found = False
                with tarfile.open(fileobj=stream, mode="r|") as tar:
                    for member in tar:
                        # Extract exactly one known regular file; never trust archive paths.
                        if (
                            member.isfile()
                            and Path(member.name).name
                            == Path(expected_name).name + ".h5"
                        ):
                            if found:
                                raise ValueError(
                                    "Archive contains multiple matching datasets"
                                )
                            if (
                                shutil.disk_usage(target.parent).free
                                < member.size + 1024**3
                            ):
                                raise OSError(
                                    f"Need {member.size / 1024**3:.1f} GiB plus 1 GiB free for extraction"
                                )
                            with (
                                tar.extractfile(member) as source,
                                open(tmp, "wb") as out,
                            ):
                                shutil.copyfileobj(source, out, 8 * 1024 * 1024)
                            found = True
                if not found:
                    raise ValueError(f"No {expected_name}.h5 in archive")
            else:
                with open(tmp, "wb") as out:
                    shutil.copyfileobj(stream, out, 8 * 1024 * 1024)
        inspect_source(tmp)
        tmp.replace(target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument(
        "--source",
        type=Path,
        help="Use an existing uncompressed SWM HDF5 instead of downloading data",
    )
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--remove-archive", action="store_true")
    args = p.parse_args()
    spec, root, api = TASKS[args.task], cache_root(), HfApi()
    target = root / "datasets" / (spec["name"] + ".h5")
    model = api.model_info(spec["repo"])
    data = api.dataset_info(spec["repo"], files_metadata=True)
    size = next(s.size for s in data.siblings if s.rfilename == spec["archive"])
    disk_root = root
    while not disk_root.exists():
        disk_root = disk_root.parent
    plan = dict(
        task=args.task,
        dataset=spec["repo"],
        dataset_revision=data.sha,
        archive=spec["archive"],
        compressed_bytes=size,
        target=str(target),
        model_revision=model.sha,
        root=str(root),
        free_bytes=shutil.disk_usage(disk_root).free,
    )
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    root.mkdir(parents=True, exist_ok=True)
    model_dir = root / "checkpoints" / args.task
    for name in ("weights.pt", "config.json"):
        hf_hub_download(spec["repo"], name, revision=model.sha, local_dir=model_dir)
    if args.source:
        target = args.source.expanduser().resolve()
    elif not target.exists():
        # Conservative compressed-size check, with exact tar member check above.
        if shutil.disk_usage(root).free < size * 2 + 1024**3:
            raise OSError(
                f"At least {2 * size / 1024**3 + 1:.1f} GiB needed for archive and extraction; actual raw data can be larger. Set STABLEWM_HOME to a large disk."
            )
        archive = Path(
            hf_hub_download(
                spec["repo"],
                spec["archive"],
                repo_type="dataset",
                revision=data.sha,
                local_dir=root / "downloads" / args.task,
            )
        )
        extract(archive, target, spec["name"])
        if args.remove_archive:
            archive.unlink()
    source = inspect_source(target)
    split = episode_split(source, seed=args.seed)
    metadata = dict(
        **plan,
        source=source,
        split=split,
        source_mode="provided_hdf5" if args.source else "official_download",
        lewm_checkpoint=str(model_dir / "weights.pt"),
        checkpoint_sha256=file_sha256(model_dir / "weights.pt"),
    )
    path = root / "prepared" / args.task / "source.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["source"]["signature"] != source["signature"] or old["split"] != split:
            raise ValueError(
                "Existing preparation uses different data/split. Use a separate STABLEWM_HOME."
            )
    atomic_json(path, metadata)
    print(
        f"Prepared {source['episodes']} episodes, {source['frames']} frames: {path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
