import math
from dataclasses import asdict, dataclass
from typing import Optional

EPS = 1e-10
NEG_INF = -10**9


@dataclass(frozen=True)
class TrayOptimizerConfig:
    diameter: float = 39.0
    clearance: float = 0.40
    heater_height: float = 5.00
    tray_inside_width: float = 9.9
    tray_inside_height: float = 2.0
    tray_thickness: float = 0.25
    inter_tray_gap: float = 0.0
    row_mode: str = "Optimize automatically"
    requested_rows: Optional[int] = None


def optimize_trays(config: TrayOptimizerConfig | dict):
    """Optimize exact-width rows containing one, two, or three individual trays."""
    if isinstance(config, dict):
        config = TrayOptimizerConfig(**config)

    diameter = float(config.diameter)
    clearance = float(config.clearance)
    heater_height = float(config.heater_height)
    inside_width = float(config.tray_inside_width)
    inside_height = float(config.tray_inside_height)
    thickness = float(config.tray_thickness)
    gap = float(config.inter_tray_gap)
    row_mode = config.row_mode
    requested_rows = config.requested_rows

    if diameter <= 0:
        raise ValueError("Vessel diameter must be positive.")
    if clearance < 0:
        raise ValueError("Clearance cannot be negative.")
    if heater_height > diameter:
        raise ValueError("The internal heater height cannot exceed the circle diameter.")
    if inside_width <= 0 or inside_height <= 0:
        raise ValueError("Tray inside width and height must be positive.")
    if thickness < 0 or gap < 0:
        raise ValueError("Tray thickness and inter-tray gap cannot be negative.")

    radius = diameter / 2.0
    usable_radius = radius - clearance
    lower_bottom = max(clearance, heater_height)
    top_limit = diameter - clearance
    if usable_radius <= 0:
        raise ValueError("Clearance must be smaller than half the circle diameter.")

    outside_tray_width = inside_width + 2.0 * thickness
    row_height = inside_height + thickness
    class_specs = {
        tray_count: {
            "width": tray_count * outside_tray_width + (tray_count - 1) * gap,
            "height": row_height,
            "inside_height": inside_height,
        }
        for tray_count in (1, 2, 3)
    }

    available_span = max(0.0, top_limit - lower_bottom)
    max_rows = int(math.floor(available_span / row_height + EPS))
    if max_rows < 1:
        return None

    target_rows = (
        int(requested_rows)
        if row_mode == "Specify row count" and requested_rows is not None
        else None
    )
    if target_rows is not None and not 1 <= target_rows <= max_rows:
        return None

    def available_chord(bottom_z):
        vertical_extent = max(
            abs(bottom_z - radius),
            abs(bottom_z + row_height - radius),
        )
        q = usable_radius**2 - vertical_extent**2
        if q < -EPS:
            return None
        return 2.0 * math.sqrt(max(0.0, q))

    def class_fits(bottom_z, tray_count):
        if bottom_z < lower_bottom - EPS or bottom_z + row_height > top_limit + EPS:
            return False
        chord = available_chord(bottom_z)
        return chord is not None and class_specs[tray_count]["width"] <= chord + EPS

    def width_boundaries(width):
        q = usable_radius**2 - (width / 2.0) ** 2
        if q < -EPS:
            return []
        d = math.sqrt(max(0.0, q))
        return [radius - d, radius + d - row_height]

    candidates = {round(lower_bottom, 10)}
    for rows in range(1, max_rows + 1):
        upper_start = top_limit - rows * row_height
        if upper_start >= lower_bottom - EPS:
            candidates.add(round(upper_start, 10))
            centered = radius - rows * row_height / 2.0
            candidates.add(round(min(max(centered, lower_bottom), upper_start), 10))
    for offset_rows in range(max_rows):
        offset = offset_rows * row_height
        for tray_count in (1, 2, 3):
            for row_bottom in width_boundaries(class_specs[tray_count]["width"]):
                start = row_bottom - offset
                if lower_bottom - EPS <= start <= top_limit - row_height + EPS:
                    candidates.add(round(max(lower_bottom, start), 10))

    boundaries = sorted(candidates)
    for left, right in zip(boundaries, boundaries[1:]):
        if right - left > 1e-8:
            candidates.add(round((left + right) / 2.0, 10))

    def best_from(start_z):
        memo = {}

        def solve(z, rows_used):
            key = (round(z, 10), rows_used)
            if key in memo:
                return memo[key]
            if target_rows is not None and rows_used == target_rows:
                memo[key] = 0
                return 0
            best = 0 if target_rows is None else NEG_INF
            for tray_count in (3, 2, 1):
                if class_fits(z, tray_count):
                    suffix = solve(z + row_height, rows_used + 1)
                    if suffix > NEG_INF // 2:
                        best = max(best, tray_count + suffix)
            memo[key] = best
            return best

        return solve(start_z, 0), memo

    best = None
    for start_z in sorted(candidates):
        total_trays, memo = best_from(start_z)
        if total_trays <= 0:
            continue

        sequence = []

        def build(z, rows_used, trays_so_far):
            nonlocal best
            key = (round(z, 10), rows_used)
            suffix_here = memo.get(key, NEG_INF)
            if target_rows is not None and rows_used == target_rows:
                finished = trays_so_far == total_trays
            else:
                finished = target_rows is None and suffix_here == 0
            if finished:
                rows = [
                    {
                        "bottom_z": row["bottom_z"],
                        "width": class_specs[row["trays"]]["width"],
                        "height": row_height,
                        "inside_height": inside_height,
                        "bags": row["trays"],
                        "trays": row["trays"],
                    }
                    for row in sequence
                ]
                three_area = sum(
                    r["trays"] * outside_tray_width * r["height"]
                    for r in rows if r["trays"] == 3
                )
                two_area = sum(
                    r["trays"] * outside_tray_width * r["height"]
                    for r in rows if r["trays"] == 2
                )
                total_area = sum(
                    r["trays"] * outside_tray_width * r["height"] for r in rows
                )
                score = (
                    total_trays,
                    three_area,
                    -len(rows),
                    two_area,
                    total_area,
                    -start_z,
                )
                if best is None or score > best[0]:
                    best = (score, rows)
                return

            for tray_count in (3, 2, 1):
                if not class_fits(z, tray_count):
                    continue
                next_key = (round(z + row_height, 10), rows_used + 1)
                suffix = memo.get(next_key, NEG_INF)
                if suffix <= NEG_INF // 2:
                    continue
                if trays_so_far + tray_count + suffix != total_trays:
                    continue
                sequence.append({"bottom_z": z, "trays": tray_count})
                build(z + row_height, rows_used + 1, trays_so_far + tray_count)
                sequence.pop()

        build(start_z, 0, 0)

    if best is None:
        return None

    score, rows = best
    total_area = sum(
        row["trays"] * outside_tray_width * row["height"] for row in rows
    )
    usable_area = math.pi * usable_radius**2
    heater_distance = abs(heater_height - radius)
    heater_q = usable_radius**2 - heater_distance**2
    heater_chord = 2.0 * math.sqrt(max(0.0, heater_q)) if heater_q >= -EPS else 0.0
    usable_area -= heater_height * heater_chord
    physical_area = math.pi * radius**2

    return {
        "config": asdict(config),
        "rows": rows,
        "basket_count": len(rows),
        "row_count": len(rows),
        "total_bags": sum(row["trays"] for row in rows),
        "total_trays": sum(row["trays"] for row in rows),
        "total_area": total_area,
        "clearance_adjusted_utilization": (
            100.0 * total_area / usable_area if usable_area > 0 else 0.0
        ),
        "physical_utilization": (
            100.0 * total_area / physical_area if physical_area > 0 else 0.0
        ),
        "score": score,
        "max_rows": max_rows,
        "outside_tray_width": outside_tray_width,
        "outside_tray_height": row_height,
    }
