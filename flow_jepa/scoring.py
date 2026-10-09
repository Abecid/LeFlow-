"""Score a connected action rollout; generated states never reset dynamics."""
from .models import distance


def continuous_plan_score(world, start, goal, paths, actions, *, waypoint_weight=0.1,
                          budget=None):
    if waypoint_weight < 0:
        raise ValueError("Waypoint weight cannot be negative")
    n, segments, chunk, action_dim = actions.shape
    if paths.shape[:2] != (n, segments + 1):
        raise ValueError("Path and action segment counts differ")
    rollout = world.rollout(start.expand(n, -1, -1),
                            actions.clamp(-1, 1).reshape(n, segments * chunk, action_dim),
                            budget=budget)
    goal_cost = distance(rollout[:, -1], goal.expand(n, -1, -1))
    waypoint_cost = distance(rollout[:, chunk - 1::chunk], paths[:, 1:]).mean(1)
    return goal_cost + waypoint_weight * waypoint_cost
