from copy import deepcopy

from flow_jepa.failure_analysis import analyze, failure_category


def test_stopping_conditions_and_paired_failure_inventory(small_config):
    base = dict(id="test/reach/0", task="reach", model_seed=3072, reset_seed=17,
                episode_sha256="same", success=False, steps=200,
                controller_budget_exhausted=False, controller_latency_ms=[20, 30],
                predicted_plan_cost=None, observed_subgoal_cosine=[])
    assert failure_category(base, 200) == "primitive_action_limit"
    assert failure_category(dict(base, steps=2), 200) == "early_environment_end"
    assert failure_category(dict(base, controller_budget_exhausted=True), 200) == "controller_time_exhausted"
    assert failure_category(dict(base, success=True, controller_budget_exhausted=True), 200) == "success"
    reports = [dict(method=m, records=[deepcopy(base)]) for m in small_config["methods"]]
    reports[1]["records"][0].update(success=True, steps=20)
    result = analyze(small_config, reports)
    paired = result["paired"][small_config["primary_baseline"]]
    assert paired["outcomes"] == {"baseline_only_success": 1}
    assert paired["failed_cases"][0]["id"] == base["id"]
    assert paired["failed_cases"][0]["ours_observed_subgoal_distance"]["mean"] is None
    assert result["methods"][small_config["primary_method"]]["by_outcome"]["failed"]["controller_seconds"]["mean"] == .05
