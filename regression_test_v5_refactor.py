import hashlib
import json

from tatermaxxer_basketengine_v5 import OptimizerConfig, optimize


CASES = [
    OptimizerConfig(),
    OptimizerConfig(simplify_manufacturing=False),
    OptimizerConfig(
        minimum_1_bag_basket_width=10.0,
        inside_height_1=1.5,
        inside_height_2=1.5,
        simplify_manufacturing=True,
    ),
    OptimizerConfig(
        minimum_1_bag_basket_width=10.0,
        inside_height_1=1.5,
        inside_height_2=1.5,
        simplify_manufacturing=False,
    ),
    OptimizerConfig(inside_height_1=1.5),
    OptimizerConfig(
        inside_height_1=1.5,
        inside_height_2=1.75,
        inside_height_3=2.25,
        manufacturing_increment=0.5,
    ),
]

# Golden fingerprints were recorded after the v4.0-to-v5 refactor comparison
# passed. Normalization makes the check portable across supported Python versions.
EXPECTED = [
    "ef1aea3c4efb8d383ac25d6b5be00dcf560a807dc168cd9c5067b0607efbf571",
    "1772c0615dd9127f909681c2b325b12b324b3a112b3d3140ff13410bf2c6d907",
    "32d3cde57addde1fea90094a7047d991a23c371ea0e223b4348fec01ffb3705a",
    "e50ba9375caee4c01d7a98d36d57dd660a1bd3681ccef0302e283c5737f1ddd3",
    "b91010f6402b498e528f3dbdc9425f88a30cb2ba6d2f2f3d194ddc9c9b54d595",
    "0316db6f7081f45fb56ae502d36719ea5ef4207ed92fa601bfcd0a7f5f594e78",
]


def normalized_result(result):
    if result is None:
        return None
    return {
        "basket_count": result["basket_count"],
        "total_bags": result["total_bags"],
        "total_area": round(result["total_area"], 8),
        "clearance_adjusted_utilization": round(
            result["clearance_adjusted_utilization"], 8
        ),
        "physical_utilization": round(result["physical_utilization"], 8),
        "rows": [
            {
                key: round(value, 8) if isinstance(value, float) else value
                for key, value in row.items()
            }
            for row in result["rows"]
        ],
    }


def fingerprint(result):
    payload = json.dumps(
        normalized_result(result),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


passed = 0
for index, (config, expected) in enumerate(zip(CASES, EXPECTED), start=1):
    result = optimize(config)
    actual = fingerprint(result)
    if actual != expected:
        summary = (
            None
            if result is None
            else {
                "basket_count": result["basket_count"],
                "total_bags": result["total_bags"],
                "total_area": result["total_area"],
            }
        )
        raise AssertionError(
            f"Case {index:02d} changed: expected {expected}, got {actual}; "
            f"summary={summary}"
        )
    passed += 1
    print(
        f"PASS {index:02d}: simplify={config.simplify_manufacturing}, "
        f"baskets={result['basket_count']}, bags={result['total_bags']}"
    )

print(f"REGRESSION_PASS {passed}/{len(CASES)}")

