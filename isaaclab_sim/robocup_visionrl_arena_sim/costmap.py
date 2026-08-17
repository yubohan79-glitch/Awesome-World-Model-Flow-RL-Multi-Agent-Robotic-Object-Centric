from __future__ import annotations

import math


from ._bootstrap import (
    ARENA_SIZE,
    BASE_TARGET_CONTACT_RADIUS,
    COSTMAP_HARD_MARGIN,
    COSTMAP_LAST_WARN,
    COSTMAP_MAX_REPULSE_STEP,
    COSTMAP_SOFT_INFLATION,
    COSTMAP_WARN_INTERVAL_S,
    MATCH_STATE,
    NAV_BLOCKERS,
    PLANNER_GRID_RESOLUTION,
    PUSHABLE_OBSTACLES,
    PUSH_OBSTACLE_STEP_M,
    ROBOT_COLLISION_RADIUS,
    ROBOT_PUSHABLE_RENDER_CLEARANCE_RADIUS,
    ROBOT_PUSHABLE_VISUAL_HALF_EXTENTS,
    ROUTE_CLEARANCE,
    TARGET_CONTACT_RADIUS,
    TARGET_REGISTRY
)
from .replay import point_blocked, pushable_position_valid, segment_blocked
from .transforms import quat_from_euler, set_xform

def circle_aabb_collision(
    point: tuple[float, float],
    center: tuple[float, float],
    half_size: tuple[float, float],
    radius: float,
) -> tuple[bool, tuple[float, float], float]:
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    closest_x = max(center[0] - half_size[0], min(point[0], center[0] + half_size[0]))
    closest_y = max(center[1] - half_size[1], min(point[1], center[1] + half_size[1]))
    vx = point[0] - closest_x
    vy = point[1] - closest_y
    distance = math.hypot(vx, vy)
    if distance > 1e-8:
        penetration = radius - distance
        if penetration <= 0.0:
            return False, (0.0, 0.0), 0.0
        return True, (vx / distance, vy / distance), penetration

    inside_x = half_size[0] - abs(dx)
    inside_y = half_size[1] - abs(dy)
    if inside_x < 0.0 or inside_y < 0.0:
        return False, (0.0, 0.0), 0.0
    if inside_x <= inside_y:
        normal = (1.0 if dx >= 0.0 else -1.0, 0.0)
        penetration = inside_x + radius
    else:
        normal = (0.0, 1.0 if dy >= 0.0 else -1.0)
        penetration = inside_y + radius
    return True, normal, penetration


def oriented_rect_aabb_collision(
    rect_center: tuple[float, float],
    yaw: float,
    rect_half: tuple[float, float],
    box_center: tuple[float, float],
    box_half: tuple[float, float],
) -> tuple[bool, tuple[float, float], float]:
    ux = (math.cos(yaw), math.sin(yaw))
    uy = (-math.sin(yaw), math.cos(yaw))
    delta = (rect_center[0] - box_center[0], rect_center[1] - box_center[1])
    axes = (ux, uy, (1.0, 0.0), (0.0, 1.0))
    best_axis = (1.0, 0.0)
    best_overlap = math.inf

    for axis in axes:
        rect_radius = rect_half[0] * abs(ux[0] * axis[0] + ux[1] * axis[1]) + rect_half[1] * abs(
            uy[0] * axis[0] + uy[1] * axis[1]
        )
        box_radius = box_half[0] * abs(axis[0]) + box_half[1] * abs(axis[1])
        distance = abs(delta[0] * axis[0] + delta[1] * axis[1])
        overlap = rect_radius + box_radius - distance
        if overlap <= 0.0:
            return False, (0.0, 0.0), 0.0
        if overlap < best_overlap:
            best_overlap = overlap
            sign = 1.0 if delta[0] * axis[0] + delta[1] * axis[1] >= 0.0 else -1.0
            best_axis = (axis[0] * sign, axis[1] * sign)

    norm = math.hypot(best_axis[0], best_axis[1])
    if norm <= 1e-8:
        return True, (1.0, 0.0), float(best_overlap)
    return True, (best_axis[0] / norm, best_axis[1] / norm), float(best_overlap)


def robot_pushable_collision(
    robot_xy: tuple[float, float],
    yaw: float,
    box_center: tuple[float, float],
    box_half: tuple[float, float],
) -> tuple[bool, tuple[float, float], float]:
    return oriented_rect_aabb_collision(
        robot_xy,
        yaw,
        ROBOT_PUSHABLE_VISUAL_HALF_EXTENTS,
        box_center,
        box_half,
    )


def push_pushable_obstacle(path: str, yaw: float, source: str) -> bool:
    obstacle = PUSHABLE_OBSTACLES.get(path)
    if obstacle is None:
        return False
    xy = obstacle["xy"]
    z = float(obstacle["z"])
    size = obstacle["size"]
    assert isinstance(xy, list)
    assert isinstance(size, tuple)
    direction = (math.cos(yaw), math.sin(yaw))
    accepted = None
    for multiplier in (1.0, 1.7, 2.4, 3.1, 4.0, 5.0):
        candidate = (
            float(xy[0]) + direction[0] * PUSH_OBSTACLE_STEP_M * multiplier,
            float(xy[1]) + direction[1] * PUSH_OBSTACLE_STEP_M * multiplier,
        )
        if pushable_position_valid(path, candidate):
            accepted = candidate
            break
    if accepted is None:
        return False
    xy[0], xy[1] = accepted
    set_xform(path, (accepted[0], accepted[1], z), quat_from_euler(0.0, 0.0, 0.0))
    warn_costmap(source, f"pushable box {path.rsplit('/', 1)[-1]} moved to ({accepted[0]:.2f}, {accepted[1]:.2f})")
    return True


def snap_to_grid(point: tuple[float, float]) -> tuple[int, int]:
    res = PLANNER_GRID_RESOLUTION
    return (round(point[0] / res), round(point[1] / res))


def grid_to_world(cell: tuple[int, int]) -> tuple[float, float]:
    res = PLANNER_GRID_RESOLUTION
    return (cell[0] * res, cell[1] * res)


def warn_costmap(source: str, message: str):
    now = float(MATCH_STATE.get("current_time", 0.0))
    key = f"{source}:{message}"
    if now - COSTMAP_LAST_WARN.get(key, -999.0) < COSTMAP_WARN_INTERVAL_S:
        return
    COSTMAP_LAST_WARN[key] = now
    MATCH_STATE["last_event"] = message
    print(f"[COSTMAP]: {source} {message}")


def nearest_free_point(point: tuple[float, float]) -> tuple[float, float]:
    if not point_blocked(point):
        return point
    for radius_step in range(1, 10):
        radius = radius_step * PLANNER_GRID_RESOLUTION
        samples = max(12, radius_step * 8)
        for sample in range(samples):
            angle = math.tau * float(sample) / float(samples)
            candidate = (point[0] + math.cos(angle) * radius, point[1] + math.sin(angle) * radius)
            if not point_blocked(candidate):
                return candidate
    limit = ARENA_SIZE * 0.5 - ROUTE_CLEARANCE
    return (max(-limit, min(limit, point[0])), max(-limit, min(limit, point[1])))


def clamp_to_arena(point: tuple[float, float]) -> tuple[float, float]:
    limit = ARENA_SIZE * 0.5 - ROUTE_CLEARANCE
    return (max(-limit, min(limit, point[0])), max(-limit, min(limit, point[1])))


def aabb_costmap_repel(
    point: tuple[float, float],
    center: tuple[float, float],
    half_size: tuple[float, float],
) -> tuple[tuple[float, float], bool, float]:
    x, y = point
    cx, cy = center
    hx, hy = half_size
    sx = x - cx
    sy = y - cy
    inside_x = hx - abs(sx)
    inside_y = hy - abs(sy)
    if inside_x >= 0.0 and inside_y >= 0.0:
        sign_x = 1.0 if sx >= 0.0 else -1.0
        sign_y = 1.0 if sy >= 0.0 else -1.0
        if inside_x <= inside_y:
            return (sign_x * (inside_x + COSTMAP_HARD_MARGIN), 0.0), True, 1.0
        return (0.0, sign_y * (inside_y + COSTMAP_HARD_MARGIN)), True, 1.0

    closest_x = max(cx - hx, min(x, cx + hx))
    closest_y = max(cy - hy, min(y, cy + hy))
    dx = x - closest_x
    dy = y - closest_y
    distance = math.hypot(dx, dy)
    if distance <= 1e-6 or distance >= COSTMAP_SOFT_INFLATION:
        return (0.0, 0.0), False, 0.0
    strength = (COSTMAP_SOFT_INFLATION - distance) / COSTMAP_SOFT_INFLATION
    step = min(COSTMAP_MAX_REPULSE_STEP, strength * COSTMAP_MAX_REPULSE_STEP)
    return (dx / distance * step, dy / distance * step), False, strength


def circle_costmap_repel(
    point: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> tuple[tuple[float, float], bool, float]:
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-6:
        dx, dy, distance = 1.0, 0.0, 1.0
    if distance < radius:
        step = radius - distance + COSTMAP_HARD_MARGIN
        return (dx / distance * step, dy / distance * step), True, 1.0
    soft_distance = radius + COSTMAP_SOFT_INFLATION
    if distance >= soft_distance:
        return (0.0, 0.0), False, 0.0
    strength = (soft_distance - distance) / COSTMAP_SOFT_INFLATION
    step = min(COSTMAP_MAX_REPULSE_STEP, strength * COSTMAP_MAX_REPULSE_STEP)
    return (dx / distance * step, dy / distance * step), False, strength


def dynamic_target_costmap() -> list[tuple[str, tuple[float, float], float]]:
    blockers = []
    for target_path, target in TARGET_REGISTRY.items():
        if target["knocked"]:
            continue
        xy = target["xy"]
        kind = str(target["kind"])
        assert isinstance(xy, tuple)
        radius = ROBOT_COLLISION_RADIUS + (BASE_TARGET_CONTACT_RADIUS if kind.startswith("base_") else TARGET_CONTACT_RADIUS)
        blockers.append((target_path, xy, radius))
    return blockers


def dynamic_pushable_costmap() -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
    blockers = []
    for path, obstacle in PUSHABLE_OBSTACLES.items():
        xy = obstacle["xy"]
        half = obstacle["half"]
        assert isinstance(xy, list)
        assert isinstance(half, tuple)
        blockers.append(
            (
                path,
                (float(xy[0]), float(xy[1])),
                (
                    float(half[0]) + ROBOT_PUSHABLE_RENDER_CLEARANCE_RADIUS,
                    float(half[1]) + ROBOT_PUSHABLE_RENDER_CLEARANCE_RADIUS,
                ),
            )
        )
    return blockers


def aabb_clearance(point: tuple[float, float], center: tuple[float, float], half_size: tuple[float, float]) -> float:
    dx = max(abs(point[0] - center[0]) - half_size[0], 0.0)
    dy = max(abs(point[1] - center[1]) - half_size[1], 0.0)
    if dx <= 0.0 and dy <= 0.0:
        return -min(half_size[0] - abs(point[0] - center[0]), half_size[1] - abs(point[1] - center[1]))
    return math.hypot(dx, dy)


def costmap_potential(point: tuple[float, float]) -> float:
    potential = 0.0
    for _blocker_path, center, half_size in NAV_BLOCKERS:
        clearance = aabb_clearance(point, center, half_size)
        if clearance < 0.0:
            return 1e6
        if clearance < COSTMAP_SOFT_INFLATION:
            strength = (COSTMAP_SOFT_INFLATION - clearance) / COSTMAP_SOFT_INFLATION
            potential += 8.0 * strength * strength

    for _target_path, center, radius in dynamic_target_costmap():
        distance = math.hypot(point[0] - center[0], point[1] - center[1])
        clearance = distance - radius
        if clearance < 0.0:
            return 1e6
        if clearance < COSTMAP_SOFT_INFLATION:
            strength = (COSTMAP_SOFT_INFLATION - clearance) / COSTMAP_SOFT_INFLATION
            potential += 6.0 * strength * strength

    for _obstacle_path, center, half_size in dynamic_pushable_costmap():
        clearance = aabb_clearance(point, center, half_size)
        if clearance < 0.0:
            potential += 2.0
        elif clearance < COSTMAP_SOFT_INFLATION:
            strength = (COSTMAP_SOFT_INFLATION - clearance) / COSTMAP_SOFT_INFLATION
            potential += 1.5 * strength * strength
    return potential


def apply_costmap_recovery(
    point: tuple[float, float],
    source: str,
    *,
    passes: int = 3,
) -> tuple[tuple[float, float], bool, bool]:
    corrected = point
    touched = False
    hard_touched = False
    for _ in range(passes):
        total_x = 0.0
        total_y = 0.0
        strongest = 0.0
        touched_name = ""
        hard_touch = False

        for blocker_path, center, half_size in NAV_BLOCKERS:
            push, hard, strength = aabb_costmap_repel(corrected, center, half_size)
            if hard or strength > 0.0:
                total_x += push[0]
                total_y += push[1]
                if hard or strength > strongest:
                    strongest = max(strength, strongest)
                    touched_name = blocker_path.rsplit("/", 1)[-1]
                    hard_touch = hard_touch or hard

        for target_path, center, radius in dynamic_target_costmap():
            push, hard, strength = circle_costmap_repel(corrected, center, radius)
            if hard or strength > 0.0:
                total_x += push[0]
                total_y += push[1]
                if hard or strength > strongest:
                    strongest = max(strength, strongest)
                    touched_name = target_path.rsplit("/", 1)[-1]
                    hard_touch = hard_touch or hard

        for obstacle_path, center, half_size in dynamic_pushable_costmap():
            push, hard, strength = aabb_costmap_repel(corrected, center, half_size)
            if hard or strength > 0.0:
                total_x += push[0]
                total_y += push[1]
                if hard or strength > strongest:
                    strongest = max(strength, strongest)
                    touched_name = obstacle_path.rsplit("/", 1)[-1]
                    hard_touch = hard_touch or hard

        if abs(total_x) < 1e-7 and abs(total_y) < 1e-7:
            break
        touched = True
        hard_touched = hard_touched or hard_touch
        corrected = clamp_to_arena((corrected[0] + total_x, corrected[1] + total_y))
        if touched_name:
            mode = "hard contact" if hard_touch else "near obstacle"
            warn_costmap(source, f"{mode} near {touched_name}; repulsive costmap recovery")

    return corrected, touched, hard_touched


def plan_safe_path(start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]]:
    if point_blocked(start):
        warn_costmap("planner", f"start inside obstacle at ({start[0]:.2f}, {start[1]:.2f}); using nearest free cell")
        start = nearest_free_point(start)
    if point_blocked(goal):
        warn_costmap("planner", f"goal inside obstacle at ({goal[0]:.2f}, {goal[1]:.2f}); using nearest free cell")
        goal = nearest_free_point(goal)

    start_cell = snap_to_grid(start)
    goal_cell = snap_to_grid(goal)
    min_cell = math.floor((-ARENA_SIZE * 0.5 + ROUTE_CLEARANCE) / PLANNER_GRID_RESOLUTION)
    max_cell = math.ceil((ARENA_SIZE * 0.5 - ROUTE_CLEARANCE) / PLANNER_GRID_RESOLUTION)
    neighbors = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    open_set: set[tuple[int, int]] = {start_cell}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start_cell: 0.0}
    f_score: dict[tuple[int, int], float] = {
        start_cell: math.hypot(goal_cell[0] - start_cell[0], goal_cell[1] - start_cell[1])
    }

    while open_set:
        current = min(open_set, key=lambda cell: f_score.get(cell, float("inf")))
        if current == goal_cell:
            grid_path = [current]
            while current in came_from:
                current = came_from[current]
                grid_path.append(current)
            grid_path.reverse()
            path = [start]
            path.extend(grid_to_world(cell) for cell in grid_path[1:-1])
            path.append(goal)
            return smooth_path(path)

        open_set.remove(current)
        current_world = grid_to_world(current)
        for dx, dy in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            if nxt[0] < min_cell or nxt[0] > max_cell or nxt[1] < min_cell or nxt[1] > max_cell:
                continue
            nxt_world = grid_to_world(nxt)
            if point_blocked(nxt_world) or segment_blocked(current_world, nxt_world):
                continue
            local_cost = costmap_potential(nxt_world)
            if local_cost >= 1e5:
                continue
            tentative_g = g_score[current] + math.hypot(dx, dy) * (1.0 + local_cost)
            if tentative_g >= g_score.get(nxt, float("inf")):
                continue
            came_from[nxt] = current
            g_score[nxt] = tentative_g
            f_score[nxt] = tentative_g + math.hypot(goal_cell[0] - nxt[0], goal_cell[1] - nxt[1])
            open_set.add(nxt)

    warn_costmap(
        "planner",
        f"A* failed from ({start[0]:.2f}, {start[1]:.2f}) to ({goal[0]:.2f}, {goal[1]:.2f}); falling back to reactive costmap",
    )
    return [start, goal]


def smooth_path(path: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(path) <= 2:
        return path
    smoothed = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        nxt = len(path) - 1
        while nxt > anchor + 1 and segment_unsafe_for_robot(path[anchor], path[nxt]):
            nxt -= 1
        smoothed.append(path[nxt])
        anchor = nxt
    return smoothed


def segment_unsafe_for_robot(p0: tuple[float, float], p1: tuple[float, float]) -> bool:
    if segment_blocked(p0, p1):
        return True
    length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    samples = max(2, math.ceil(length / (PLANNER_GRID_RESOLUTION * 0.55)))
    for index in range(1, samples):
        alpha = index / samples
        sample = (p0[0] + (p1[0] - p0[0]) * alpha, p0[1] + (p1[1] - p0[1]) * alpha)
        if costmap_potential(sample) > 4.0:
            return True
    return False


def interpolate_path(
    path: list[tuple[float, float]],
    distance: float,
) -> tuple[tuple[float, float, float], float, bool]:
    if len(path) < 2:
        return (path[0][0], path[0][1], 0.0), 0.0, True

    walked = 0.0
    for p0, p1 in zip(path, path[1:]):
        segment_length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if distance <= walked + segment_length:
            alpha = 0.0 if segment_length <= 1e-9 else (distance - walked) / segment_length
            x = p0[0] + (p1[0] - p0[0]) * alpha
            y = p0[1] + (p1[1] - p0[1]) * alpha
            yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            return (x, y, 0.0), yaw, False
        walked += segment_length

    final = path[-1]
    previous = path[-2]
    yaw = math.atan2(final[1] - previous[1], final[0] - previous[0])
    return (final[0], final[1], 0.0), yaw, True


def path_length(path: list[tuple[float, float]]) -> float:
    return sum(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) for p0, p1 in zip(path, path[1:]))


def demo_policy_corridor(team: str, start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> list[tuple[float, float]]:
    """Wide, regulation-safe staging waypoints for the portfolio self-play replay.

    The high-level policy still decides which opponent target to attack. These
    waypoints emulate the low-level Nav2 corridor preference that keeps the
    differential-drive base away from start rails, inner fences, and armor.
    """
    sx, sy = start_xy
    gx, gy = goal_xy
    if team == "yellow":
        staging = []
        if sy < -0.84:
            staging.append((max(0.36, sx), -0.78))
        if sy < -0.42:
            staging.append((0.30, -0.58))
        if sy < 0.24:
            staging.append((0.30, -0.26))
        if gy >= 0.24:
            staging.append((0.30, 0.42))
            staging.append((0.0 if gx < 0.0 else 0.34, max(0.42, gy)))
        if gx < -0.34:
            staging.append((-0.34, max(0.40, gy)))
        elif gx > 0.34 and gy < 0.24:
            staging.append((0.34, min(-0.28, gy)))
    else:
        staging = []
        if sy > 0.84:
            staging.append((min(-0.36, sx), 0.78))
        if sy > 0.42:
            staging.append((-0.30, 0.58))
        if sy > -0.24:
            staging.append((-0.30, 0.26))
        if gy <= -0.24:
            staging.append((-0.30, -0.42))
            staging.append((0.0 if gx > 0.0 else -0.34, min(-0.42, gy)))
        if gx > 0.34:
            staging.append((0.34, min(-0.40, gy)))
        elif gx < -0.34 and gy > -0.24:
            staging.append((-0.34, max(0.28, gy)))

    route = [start_xy]
    for waypoint in staging:
        waypoint = nearest_free_point(clamp_to_arena(waypoint))
        if math.hypot(waypoint[0] - route[-1][0], waypoint[1] - route[-1][1]) > 0.08:
            route.append(waypoint)
    if math.hypot(goal_xy[0] - route[-1][0], goal_xy[1] - route[-1][1]) > 0.04:
        route.append(goal_xy)
    return route


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def slew_rate(current: float, target: float, max_delta: float) -> float:
    if target > current + max_delta:
        return current + max_delta
    if target < current - max_delta:
        return current - max_delta
    return target
