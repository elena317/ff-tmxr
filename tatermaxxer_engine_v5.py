import math
from dataclasses import dataclass, asdict
from typing import Optional

EPS = 1e-10
NEG_INF = -10**9

@dataclass(frozen=True)
class OptimizerConfig:
    diameter: float = 39.0
    clearance: float = 0.40
    basket_wall_thickness: float = 0.25
    heater_height: float = 5.00
    minimum_3_bag_basket_width: float = 29.75
    minimum_2_bag_basket_width: float = 19.8
    minimum_1_bag_basket_width: float = 9.9
    inside_height_3: float = 2.0
    inside_height_2: float = 2.0
    inside_height_1: float = 2.0
    simplify_manufacturing: bool = True
    manufacturing_increment: float = 0.25
    row_mode: str = "Optimize automatically"
    requested_rows: Optional[int] = None


def optimize(config: OptimizerConfig | dict):
    """Run the v4.0 2D optimizer math without Streamlit/UI dependencies."""
    if isinstance(config, dict):
        config = OptimizerConfig(**config)

    diameter = float(config.diameter)
    clearance = float(config.clearance)
    basket_wall_thickness = float(config.basket_wall_thickness)
    heater_height = float(config.heater_height)
    minimum_3_bag_basket_width = float(config.minimum_3_bag_basket_width)
    minimum_2_bag_basket_width = float(config.minimum_2_bag_basket_width)
    minimum_1_bag_basket_width = float(config.minimum_1_bag_basket_width)
    inside_height_3 = float(config.inside_height_3)
    inside_height_2 = float(config.inside_height_2)
    inside_height_1 = float(config.inside_height_1)
    simplify_manufacturing = bool(config.simplify_manufacturing)
    manufacturing_increment = float(config.manufacturing_increment)
    row_mode = config.row_mode
    requested_rows = config.requested_rows

    if diameter <= 0:
        raise ValueError("Vessel diameter must be positive.")
    if clearance < 0:
        raise ValueError("Clearance cannot be negative.")
    if basket_wall_thickness < 0:
        raise ValueError("Basket wall thickness cannot be negative.")
    if any(h <= 0 for h in (inside_height_1, inside_height_2, inside_height_3)):
        raise ValueError("Inside heights must be positive.")
    if manufacturing_increment <= 0:
        raise ValueError("Manufacturing increment must be positive.")
    if heater_height > diameter:
        raise ValueError("The internal heater height cannot exceed the circle diameter.")
    if not (minimum_1_bag_basket_width <= minimum_2_bag_basket_width <= minimum_3_bag_basket_width):
        raise ValueError("Basket width thresholds must satisfy 1-bag <= 2-bag <= 3-bag.")

    R = diameter / 2.0
    usable_R = R - clearance
    lower_bottom = max(clearance, heater_height)
    top_limit = diameter - clearance
    if usable_R <= 0:
        raise ValueError("Clearance must be smaller than half the circle diameter.")

    class_specs = {
        1: {"min_width": minimum_1_bag_basket_width, "inside_height": inside_height_1,
            "height": inside_height_1 + basket_wall_thickness},
        2: {"min_width": minimum_2_bag_basket_width, "inside_height": inside_height_2,
            "height": inside_height_2 + basket_wall_thickness},
        3: {"min_width": minimum_3_bag_basket_width, "inside_height": inside_height_3,
            "height": inside_height_3 + basket_wall_thickness},
    }
    min_class_height = min(spec["height"] for spec in class_specs.values())
    available_vertical_span = max(0.0, top_limit - lower_bottom)
    max_rows = int(math.floor(available_vertical_span / min_class_height + 1e-12))
    if max_rows < 1:
        return None

    target_rows = int(requested_rows) if row_mode == "Specify row count" and requested_rows is not None else None
    if target_rows is not None and (target_rows < 1 or target_rows > max_rows):
        return None

    def max_width_for_height(bottom_z, row_height):
        bottom_distance = abs(bottom_z - R)
        top_distance = abs(bottom_z + row_height - R)
        vertical_extent = max(bottom_distance, top_distance)
        q = usable_R**2 - vertical_extent**2
        if q < -EPS:
            return None
        return 2.0 * math.sqrt(max(0.0, q))

    def standard_width_ceiling(threshold, increment):
        return math.ceil((threshold - 1e-12) / increment) * increment

    def discrete_available_width(raw_width, increment):
        return math.floor((raw_width + EPS) / increment) * increment

    def class_is_feasible(bottom_z, bag_class):
        spec = class_specs[bag_class]
        row_height = spec["height"]
        if bottom_z < lower_bottom - EPS or bottom_z + row_height > top_limit + EPS:
            return None
        raw_width = max_width_for_height(bottom_z, row_height)
        if raw_width is None:
            return None
        if simplify_manufacturing:
            available = discrete_available_width(raw_width, manufacturing_increment)
            minimum = standard_width_ceiling(spec["min_width"], manufacturing_increment)
            if available + EPS < minimum:
                return None
            return raw_width, available
        if raw_width + EPS < spec["min_width"]:
            return None
        return raw_width, raw_width

    def reachable_height_offsets():
        limit = max(0.0, top_limit - lower_bottom)
        seen = {0.0}
        frontier = [0.0]
        while frontier:
            current = frontier.pop()
            for bag_class in (1, 2, 3):
                nxt = current + class_specs[bag_class]["height"]
                if nxt <= limit + EPS:
                    key = round(nxt, 10)
                    if key not in seen:
                        seen.add(key)
                        frontier.append(key)
        return sorted(seen)

    def threshold_bottom_boundaries(row_height, width_target):
        if width_target <= 0:
            return []
        q = usable_R**2 - (width_target / 2.0) ** 2
        if q < -EPS:
            return []
        d = math.sqrt(max(0.0, q))
        return [R - d, R + d - row_height]

    def stack_start_feasibility_boundaries():
        offsets = reachable_height_offsets()
        candidates = {round(lower_bottom, 10)}
        for total_height in offsets:
            if total_height <= EPS:
                continue
            upper_start = top_limit - total_height
            if upper_start >= lower_bottom - EPS:
                candidates.add(round(upper_start, 10))
                centered = R - total_height / 2.0
                centered = min(max(centered, lower_bottom), upper_start)
                candidates.add(round(centered, 10))
        for offset in offsets:
            for bag_class in (1, 2, 3):
                spec = class_specs[bag_class]
                targets = [spec["min_width"]]
                if simplify_manufacturing:
                    targets = [standard_width_ceiling(spec["min_width"], manufacturing_increment)]
                for target in targets:
                    for row_bottom in threshold_bottom_boundaries(spec["height"], target):
                        stack_start = row_bottom - offset
                        if lower_bottom - EPS <= stack_start <= top_limit - min_class_height + EPS:
                            candidates.add(round(max(lower_bottom, stack_start), 10))
        return sorted(candidates)

    def initial_stack_start_candidates():
        boundaries = stack_start_feasibility_boundaries()
        candidates = set(boundaries)
        for a, b in zip(boundaries, boundaries[1:]):
            if b - a > 1e-8:
                candidates.add(round((a + b) / 2.0, 10))
        return sorted(candidates)

    def best_bags_from(start_z, target_rows=None):
        memo = {}
        def solve(z, rows_used):
            key = (round(z, 10), rows_used)
            if key in memo:
                return memo[key]
            if target_rows is not None and rows_used == target_rows:
                memo[key] = 0
                return 0
            best = 0 if target_rows is None else NEG_INF
            for bag_class in (3, 2, 1):
                feasible = class_is_feasible(z, bag_class)
                if feasible is None:
                    continue
                row_height = class_specs[bag_class]["height"]
                suffix = solve(z + row_height, rows_used + 1)
                if suffix > -10**8:
                    best = max(best, bag_class + suffix)
            memo[key] = best
            return best
        return solve(start_z, 0), memo

    def manufacturing_refinement_candidates(global_best_bags, bag_results_by_start):
        if not simplify_manufacturing:
            return []
        boundaries = stack_start_feasibility_boundaries()
        best_intervals = []
        for a, b in zip(boundaries, boundaries[1:]):
            if b - a <= 1e-8:
                continue
            midpoint = round((a + b) / 2.0, 10)
            result = bag_results_by_start.get(midpoint)
            if result is None:
                bags, memo = best_bags_from(midpoint)
                bag_results_by_start[midpoint] = (bags, memo)
            else:
                bags = result[0]
            if bags == global_best_bags:
                best_intervals.append((a, b))
        if not best_intervals:
            return []
        offsets = reachable_height_offsets()
        candidates = set()
        def inside_best_interval(value):
            return any(a - EPS <= value <= b + EPS for a, b in best_intervals)
        for offset in offsets:
            for bag_class in (1, 2, 3):
                row_height = class_specs[bag_class]["height"]
                minimum = standard_width_ceiling(class_specs[bag_class]["min_width"], manufacturing_increment)
                centered_q = usable_R**2 - (row_height / 2.0) ** 2
                if centered_q < -EPS:
                    continue
                centered_max_width = 2.0 * math.sqrt(max(0.0, centered_q))
                first_k = max(1, int(math.ceil((minimum - EPS) / manufacturing_increment)))
                last_k = int(math.floor((centered_max_width + EPS) / manufacturing_increment))
                for k in range(first_k, last_k + 1):
                    width_target = k * manufacturing_increment
                    for row_bottom in threshold_bottom_boundaries(row_height, width_target):
                        stack_start = row_bottom - offset
                        if (lower_bottom - EPS <= stack_start <= top_limit - min_class_height + EPS
                                and inside_best_interval(stack_start)):
                            candidates.add(round(max(lower_bottom, stack_start), 10))
        ordered = sorted(candidates)
        local_mids = []
        for a, b in zip(ordered, ordered[1:]):
            if b - a > 1e-8 and inside_best_interval((a + b) / 2.0):
                local_mids.append(round((a + b) / 2.0, 10))
        candidates.update(local_mids)
        return sorted(candidates)

    def finalize_sequence(sequence):
        if not sequence:
            return None
        if simplify_manufacturing:
            common_widths = {}
            for bag_class in (1, 2, 3):
                class_rows = [r for r in sequence if r["bag_class"] == bag_class]
                if not class_rows:
                    continue
                width = min(r["available_width"] for r in class_rows)
                minimum = standard_width_ceiling(class_specs[bag_class]["min_width"], manufacturing_increment)
                width = math.floor((width + EPS) / manufacturing_increment) * manufacturing_increment
                if width + EPS < minimum:
                    return None
                common_widths[bag_class] = width
        else:
            common_widths = None
        rows_out = []
        for row in sequence:
            bag_class = row["bag_class"]
            actual_width = common_widths[bag_class] if common_widths is not None else row["raw_width"]
            rows_out.append({
                "bottom_z": row["bottom_z"], "width": actual_width, "height": row["height"],
                "inside_height": class_specs[bag_class]["inside_height"], "bags": bag_class,
            })
        total_bags_local = sum(r["bags"] for r in rows_out)
        three_bag_area_local = sum(r["height"] * r["width"] for r in rows_out if r["bags"] == 3)
        basket_count_local = len(rows_out)
        two_bag_area_local = sum(r["height"] * r["width"] for r in rows_out if r["bags"] == 2)
        total_area_local = sum(r["height"] * r["width"] for r in rows_out)
        return rows_out, (total_bags_local, three_bag_area_local, -basket_count_local,
                          two_bag_area_local, total_area_local)

    def best_sequence_for_start(start_z, required_bags, bag_memo, target_rows=None):
        best = None
        sequence = []
        def suffix_best(z, rows_used):
            return bag_memo.get((round(z, 10), rows_used), NEG_INF)
        def dfs(z, rows_used, bags_so_far):
            nonlocal best
            if target_rows is not None and rows_used == target_rows:
                if bags_so_far == required_bags:
                    final = finalize_sequence(sequence)
                    if final is not None and (best is None or final[1] > best[1]):
                        best = final
                return
            if target_rows is None and suffix_best(z, rows_used) == 0:
                if bags_so_far == required_bags:
                    final = finalize_sequence(sequence)
                    if final is not None and (best is None or final[1] > best[1]):
                        best = final
                return
            for bag_class in (3, 2, 1):
                feasible = class_is_feasible(z, bag_class)
                if feasible is None:
                    continue
                row_height = class_specs[bag_class]["height"]
                next_z = z + row_height
                next_rows = rows_used + 1
                suffix = suffix_best(next_z, next_rows)
                if suffix <= -10**8:
                    continue
                if bags_so_far + bag_class + suffix != required_bags:
                    continue
                raw_width, available_width = feasible
                sequence.append({"bottom_z": z, "height": row_height, "bag_class": bag_class,
                                 "raw_width": raw_width, "available_width": available_width})
                dfs(next_z, next_rows, bags_so_far + bag_class)
                sequence.pop()
        dfs(start_z, 0, 0)
        return best

    starts = initial_stack_start_candidates()
    result_cache = {}
    global_best_bags = NEG_INF
    for start_z in starts:
        bags, memo = best_bags_from(start_z, target_rows=target_rows)
        result_cache[start_z] = (bags, memo)
        if bags > -10**8:
            global_best_bags = max(global_best_bags, bags)
    if global_best_bags <= 0:
        return None
    if simplify_manufacturing and target_rows is None:
        for start_z in manufacturing_refinement_candidates(global_best_bags, result_cache):
            if start_z not in result_cache:
                result_cache[start_z] = best_bags_from(start_z, target_rows=target_rows)
    best = None
    for start_z, (bags, memo) in result_cache.items():
        if bags != global_best_bags:
            continue
        candidate = best_sequence_for_start(start_z, global_best_bags, memo, target_rows=target_rows)
        if candidate is None:
            continue
        rows_candidate, score = candidate
        full_score = (*score, -start_z)
        if best is None or full_score > best[0]:
            best = (full_score, rows_candidate)
    if best is None:
        return None

    score, row_details = best
    total_bags = sum(row["bags"] for row in row_details)
    total_area = sum(row["height"] * row["width"] for row in row_details)
    usable_area = math.pi * usable_R**2
    heater_vertical_distance = abs(heater_height - R)
    heater_chord_q = usable_R**2 - heater_vertical_distance**2
    heater_chord = 2.0 * math.sqrt(max(0.0, heater_chord_q)) if heater_chord_q >= -EPS else 0.0
    heater_area = heater_height * heater_chord
    usable_area -= heater_area
    utilization = 100.0 * total_area / usable_area if usable_area > 0 else 0.0
    physical_circle_area = math.pi * R**2
    physical_utilization = 100.0 * total_area / physical_circle_area if physical_circle_area else 0.0

    return {
        "config": asdict(config),
        "rows": row_details,
        "basket_count": len(row_details),
        "total_bags": total_bags,
        "total_area": total_area,
        "clearance_adjusted_utilization": utilization,
        "physical_utilization": physical_utilization,
        "score": score,
        "max_rows": max_rows,
    }
