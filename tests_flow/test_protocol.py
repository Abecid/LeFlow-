import copy

import pytest

from flow_jepa.common import digest, episode_seed
from flow_jepa.compare import compare
from flow_jepa.data import episode_plan, select_valid_goals


def test_goal_screening_never_replaces_based_on_model_outcomes(small_config):
    rows = episode_plan(small_config)
    for row in rows:
        row["expert_success"] = row["index"] != 0
    selected, screening = select_valid_goals(rows, small_config)
    test = [x for x in selected if x["split"] == "test" and x["task"] == "reach"]
    assert len(test) == 4
    assert [r["index"] for r in test] == [0, 1, 2, 3]
    assert [r["source_index"] for r in test] == [1, 2, 3, 4]
    assert screening["test/reach"]["failed_ids"] == ["test/reach/00000"]
    for row in rows:
        row["expert_success"] = False
    with pytest.raises(ValueError, match="Insufficient valid"):
        select_valid_goals(rows, small_config)


def test_split_disjoint_and_holdout_is_test_only(small_config):
    rows = episode_plan(small_config)
    assert len({r["id"] for r in rows}) == len(rows)
    for task in small_config["training_tasks"] + small_config["heldout_tasks"]:
        seeds = [r["seed"] for r in rows if r["task"] == task]
        assert len(seeds) == len(set(seeds))
    assert all(r["split"] == "test" for r in rows if r["task"] == "push")
    assert episode_seed("reach", "test", 0) != episode_seed("reach", "train", 0)


def reports(c):
    out = []
    for method in c["methods"]:
        for seed in c["seeds"]:
            rows = []
            for task in c["training_tasks"] + c["heldout_tasks"]:
                for i in range(c["data"]["test_episodes_per_task"]):
                    rows.append(
                        dict(
                            id=f"test/{task}/{i}",
                            task=task,
                            reset_seed=episode_seed(task, "test", i),
                            episode_sha256=f"{task}{i}",
                            success=method == c["primary_method"],
                            model_seed=seed,
                        )
                    )
            out.append(
                dict(
                    method=method,
                    seed=seed,
                    records=rows,
                    split="test",
                    fixture=False,
                    protocol=digest(c),
                    manifest="synthetic-unit-test",
                    code="test-code",
                    world_hash=f"world{seed}",
                )
            )
    return out


def test_paired_comparison_and_duplicate_guard(small_config):
    data = reports(small_config)
    result = compare(small_config, data)
    base = small_config["primary_baseline"]
    assert result["comparisons"][base]["delta_percentage_points"] == 100
    assert result["comparisons"][base]["paired_ci95"] == [100, 100]
    assert result["independent_resets"] == 8
    assert result["model_seeds"] == 3
    duplicate = copy.deepcopy(data)
    duplicate[0]["records"][1] = duplicate[0]["records"][0]
    with pytest.raises(ValueError, match="Duplicate"):
        compare(small_config, duplicate)


@pytest.mark.parametrize(
    "change", ["fixture", "manifest", "world_hash", "code", "protocol"]
)
def test_comparison_rejects_unmatched_experiments(small_config, change):
    data = reports(small_config)
    data[0][change] = True if change == "fixture" else "different"
    with pytest.raises(ValueError):
        compare(small_config, data)
