"""Post-evaluation failure inventory from recorded outcomes, without extra rollouts."""
from collections import Counter

import numpy as np


def failure_category(row, primitive_limit):
    if row["success"]:
        return "success"
    if row.get("controller_budget_exhausted", False):
        return "controller_time_exhausted"
    if row["steps"] >= primitive_limit:
        return "primitive_action_limit"
    return "early_environment_end"


def distribution(values):
    values = [float(x) for x in values if x is not None and np.isfinite(x)]
    if not values:
        return dict(count=0, mean=None, median=None, p95=None)
    return dict(count=len(values), mean=float(np.mean(values)),
                median=float(np.median(values)), p95=float(np.quantile(values, .95)))


def analyze(c, reports):
    """Call after compare() has verified completeness and paired identities."""
    groups = {}
    for report in reports:
        groups.setdefault(report["method"], []).extend(report["records"])
    limit = c["evaluation"]["budget_primitive"]
    result = dict(
        interpretation=(
            "Operational failure categories describe recorded stopping conditions, not physical causes. "
            "Latent distances and plan costs are proxies, not proof of executable progress. "
            "No additional episodes or optimization were used. Test cases are for future hypotheses; "
            "do not tune this registered comparison or rerun it selectively."
        ), methods={}, paired={},
    )
    for method, rows in groups.items():
        tasks = {}
        for task in sorted({r["task"] for r in rows}):
            selected = [r for r in rows if r["task"] == task]
            tasks[task] = dict(
                episodes=len(selected), successes=sum(r["success"] for r in selected),
                stopping_conditions=dict(Counter(failure_category(r, limit) for r in selected)),
            )
        by_outcome = {}
        for label, success in (("successful", True), ("failed", False)):
            selected = [r for r in rows if bool(r["success"]) == success]
            by_outcome[label] = dict(
                episodes=len(selected),
                primitive_steps=distribution([r["steps"] for r in selected]),
                controller_seconds=distribution([sum(r["controller_latency_ms"]) / 1000 for r in selected]),
                predicted_plan_cost=distribution([r.get("predicted_plan_cost") for r in selected]),
                observed_subgoal_distance=distribution([v for r in selected for v in r["observed_subgoal_cosine"]]),
            )
        result["methods"][method] = dict(tasks=tasks, by_outcome=by_outcome)
    ours = {(r["model_seed"], r["id"]): r for r in groups[c["primary_method"]]}
    for baseline, rows in groups.items():
        if baseline == c["primary_method"]:
            continue
        outcomes, cases = Counter(), []
        for row in rows:
            own = ours[(row["model_seed"], row["id"])]
            outcome = ("both_success" if row["success"] else "ours_only_success") if own["success"] else (
                "baseline_only_success" if row["success"] else "both_fail")
            outcomes[outcome] += 1
            if not own["success"]:
                cases.append(dict(
                    id=own["id"], model_seed=own["model_seed"], task=own["task"],
                    reset_seed=own["reset_seed"], episode_sha256=own["episode_sha256"],
                    outcome=outcome, ours_stop=failure_category(own, limit),
                    baseline_stop=failure_category(row, limit),
                    ours_steps=own["steps"], baseline_steps=row["steps"],
                    ours_plan_cost=own.get("predicted_plan_cost"),
                    ours_observed_subgoal_distance=distribution(own["observed_subgoal_cosine"]),
                ))
        result["paired"][baseline] = dict(outcomes=dict(outcomes), failed_cases=cases)
    return result
