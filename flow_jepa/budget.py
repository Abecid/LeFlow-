"""Explicit, shared resource limits for the first comparison."""

import time


class PlanningBudgetExceeded(RuntimeError):
    pass


class PlanningBudget:
    def __init__(self, seconds, *, clock=time.perf_counter):
        self.clock = clock
        self.deadline = clock() + seconds

    def check(self):
        if self.clock() >= self.deadline:
            raise PlanningBudgetExceeded("Episode controller-time budget exhausted")


def training_progress(step, steps, elapsed, limit):
    """Stop at either common cap; schedule validation at quarter-budget points."""
    return min(1.0, max(step / steps, elapsed / limit if limit else 0.0))


def verify_data_compatibility(source, execution):
    # Collection and encoding depend on exactly these fields. Training heads,
    # optimizer budgets and evaluation controllers never change cached episodes.
    for key in ("schema", "training_tasks", "heldout_tasks", "data", "encoder"):
        if source[key] != execution[key]:
            raise ValueError(f"Cannot reuse cache after changing {key}")
