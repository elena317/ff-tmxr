import math
import random
from pathlib import Path
from tatermaxxer_basketengine_v5 import OptimizerConfig, optimize

SOURCE = Path('/mnt/data/circular_rectangle_optimizer_v4.0.py').read_text()
lines = SOURCE.splitlines()
block = '\n'.join(lines[175:549])  # exact v4.0 function definitions only


def run_reference(cfg: OptimizerConfig):
    g = {'math': math}
    # Exact pre-function setup performed by v4.0 UI/geometry section.
    g.update({
        'diameter': float(cfg.diameter),
        'clearance': float(cfg.clearance),
        'basket_wall_thickness': float(cfg.basket_wall_thickness),
        'heater_height': float(cfg.heater_height),
        'minimum_3_bag_basket_width': float(cfg.minimum_3_bag_basket_width),
        'minimum_2_bag_basket_width': float(cfg.minimum_2_bag_basket_width),
        'minimum_1_bag_basket_width': float(cfg.minimum_1_bag_basket_width),
        'inside_height_3': float(cfg.inside_height_3),
        'inside_height_2': float(cfg.inside_height_2),
        'inside_height_1': float(cfg.inside_height_1),
        'simplify_manufacturing': bool(cfg.simplify_manufacturing),
        'manufacturing_increment': float(cfg.manufacturing_increment),
        'row_mode': cfg.row_mode,
        'requested_rows': cfg.requested_rows,
    })
    R = cfg.diameter / 2.0
    usable_R = R - cfg.clearance
    lower_bottom = max(cfg.clearance, cfg.heater_height)
    top_limit = cfg.diameter - cfg.clearance
    class_specs = {
        1: {'min_width': float(cfg.minimum_1_bag_basket_width), 'inside_height': float(cfg.inside_height_1), 'height': float(cfg.inside_height_1 + cfg.basket_wall_thickness)},
        2: {'min_width': float(cfg.minimum_2_bag_basket_width), 'inside_height': float(cfg.inside_height_2), 'height': float(cfg.inside_height_2 + cfg.basket_wall_thickness)},
        3: {'min_width': float(cfg.minimum_3_bag_basket_width), 'inside_height': float(cfg.inside_height_3), 'height': float(cfg.inside_height_3 + cfg.basket_wall_thickness)},
    }
    min_class_height = min(spec['height'] for spec in class_specs.values())
    available_vertical_span = max(0.0, top_limit - lower_bottom)
    max_rows = int(math.floor(available_vertical_span / min_class_height + 1e-12))
    g.update({'R': R, 'usable_R': usable_R, 'lower_bottom': lower_bottom, 'top_limit': top_limit,
              'class_specs': class_specs, 'min_class_height': min_class_height, 'max_rows': max_rows})
    exec(block, g)
    optimized = g['optimize_variable_height_stack']()
    if optimized is None:
        return None
    score, rows = optimized
    total_bags = sum(r['bags'] for r in rows)
    total_area = sum(r['height'] * r['width'] for r in rows)
    usable_area = math.pi * usable_R**2
    heater_vertical_distance = abs(cfg.heater_height - R)
    heater_chord_q = usable_R**2 - heater_vertical_distance**2
    heater_chord = 2.0 * math.sqrt(max(0.0, heater_chord_q)) if heater_chord_q >= -1e-10 else 0.0
    usable_area -= cfg.heater_height * heater_chord
    utilization = 100.0 * total_area / usable_area if usable_area > 0 else 0.0
    physical_area = math.pi * R**2
    physical_utilization = 100.0 * total_area / physical_area if physical_area else 0.0
    return {'score': score, 'rows': rows, 'basket_count': len(rows), 'total_bags': total_bags,
            'total_area': total_area, 'clearance_adjusted_utilization': utilization,
            'physical_utilization': physical_utilization}


def assert_close(a, b, tol=1e-8):
    if abs(a-b) > tol:
        raise AssertionError(f'{a} != {b}')


def compare(cfg):
    ref = run_reference(cfg)
    new = optimize(cfg)
    if (ref is None) != (new is None):
        raise AssertionError('None mismatch')
    if ref is None:
        return
    if ref['basket_count'] != new['basket_count'] or ref['total_bags'] != new['total_bags']:
        raise AssertionError((ref['basket_count'], ref['total_bags'], new['basket_count'], new['total_bags']))
    assert_close(ref['total_area'], new['total_area'])
    assert_close(ref['clearance_adjusted_utilization'], new['clearance_adjusted_utilization'])
    assert_close(ref['physical_utilization'], new['physical_utilization'])
    if len(ref['rows']) != len(new['rows']):
        raise AssertionError('row length mismatch')
    for i, (rr, nr) in enumerate(zip(ref['rows'], new['rows']), start=1):
        if rr['bags'] != nr['bags']:
            raise AssertionError(f'row {i} bag mismatch')
        for key in ('bottom_z','width','height','inside_height'):
            assert_close(rr[key], nr[key])

cases = [
    OptimizerConfig(),
    OptimizerConfig(simplify_manufacturing=False),
    OptimizerConfig(minimum_1_bag_basket_width=10.0, inside_height_1=1.5, inside_height_2=1.5, simplify_manufacturing=True),
    OptimizerConfig(minimum_1_bag_basket_width=10.0, inside_height_1=1.5, inside_height_2=1.5, simplify_manufacturing=False),
    OptimizerConfig(inside_height_1=1.5),
    OptimizerConfig(inside_height_1=1.5, inside_height_2=1.75, inside_height_3=2.25, manufacturing_increment=0.5),
]

passed = 0
for idx, cfg in enumerate(cases, 1):
    compare(cfg)
    passed += 1
    print(f'PASS {idx:02d}: simplify={cfg.simplify_manufacturing}')

print(f'REGRESSION_PASS {passed}/{len(cases)}')
