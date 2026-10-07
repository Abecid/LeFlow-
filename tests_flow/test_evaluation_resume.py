import pytest

from flow_jepa.common import save_json
from flow_jepa.evaluate import EpisodeJournal, checked_report


def test_resume_keeps_completed_episodes_and_rejects_changed_inputs(tmp_path):
    identity = dict(protocol="test", seed=3072, checkpoint_hash="original")
    row = dict(id="test/reach/00001", task="reach", seed=72, sha256="episode-original")
    journal = EpisodeJournal(tmp_path / "episodes", identity)
    assert journal.load(row) is None
    record = dict(
        id=row["id"],
        task=row["task"],
        reset_seed=row["seed"],
        episode_sha256=row["sha256"],
        model_seed=identity["seed"],
        success=False,
    )
    journal.save(record)
    resumed = EpisodeJournal(tmp_path / "episodes", identity)
    assert resumed.load(row) == record  # Keep failures, not just successful episodes.
    with pytest.raises(ValueError, match="episode mismatch"):
        resumed.load(dict(row, sha256="different-episode"))
    with pytest.raises(ValueError, match="identity mismatch"):
        EpisodeJournal(tmp_path / "episodes", dict(identity, checkpoint_hash="changed"))
    path = tmp_path / "report.json"
    report = dict(**identity, fixture=False, wandb_synced=False, records=[record])
    save_json(path, report)
    assert checked_report(path, identity) == report
    with pytest.raises(ValueError, match="identity mismatch"):
        checked_report(path, dict(identity, seed=3073))
