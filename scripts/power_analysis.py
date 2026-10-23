from __future__ import annotations

import argparse
import json
from numbers import Integral
from pathlib import Path

import numpy as np

from novavision.eval.metrics import probability as _probability
from novavision.eval.reports import read_report
from novavision.outputs import validate_output_paths, write_output

CHANCE = 1.0 / 7.0
ALPHA = 0.05
TARGET_POWER = 0.80
EFFECTS = (0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
_SIM = 4000


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _critical_count(n: int, chance: float, rng: np.random.Generator) -> int:
    null = rng.binomial(n, chance, _SIM)
    return int(np.quantile(null, 1 - ALPHA)) + 1


def power_at(n: int, p1: float, chance: float, rng: np.random.Generator) -> float:
    n = _positive_integer(n, "n")
    p1 = _probability(p1, "p1")
    chance = _probability(chance, "chance")
    crit = _critical_count(n, chance, rng)
    return float(np.mean(rng.binomial(n, p1, _SIM) >= crit))


def min_n_for_power(
    p1: float, chance: float, rng: np.random.Generator, n_max: int = 5000
) -> int | None:
    p1 = _probability(p1, "p1")
    chance = _probability(chance, "chance")
    n_max = _positive_integer(n_max, "n_max")
    if p1 <= chance:
        return None
    for n in range(10, n_max + 1, 10):
        if power_at(n, p1, chance, rng) >= TARGET_POWER:
            return n
    return None


def analyze(ceiling: float, planned_n: int) -> dict:
    ceiling = _probability(ceiling, "ceiling")
    planned_n = _positive_integer(planned_n, "planned_n")
    rng = np.random.default_rng(0)
    rows = []
    for effect_strength in EFFECTS:
        recovery_accuracy = effect_strength * ceiling + (1 - effect_strength) * CHANCE
        rows.append(
            {
                "effect_strength": effect_strength,
                "recovery_accuracy": round(recovery_accuracy, 4),
                "n_for_80pct_power": min_n_for_power(recovery_accuracy, CHANCE, rng),
                "power_at_planned_n": round(power_at(planned_n, recovery_accuracy, CHANCE, rng), 3),
            }
        )
    return {
        "method": "binomial_mixture_simulation",
        "protocol_power": False,
        "assumptions": {
            "independent_trials": True,
            "non_conveyed_accuracy": "uniform chance (1/7)",
            "conveyed_accuracy": "supplied real-image validation accuracy",
            "mixture": "p1 = s * ceiling + (1 - s) * chance",
            "null": "Binomial(n, chance), not the benchmark's fixed-label permutation null",
            "validation_transfer": "validation accuracy is assumed to transfer to generated images",
        },
        "chance": round(CHANCE, 4),
        "probe_ceiling": round(ceiling, 4),
        "alpha": ALPHA,
        "target_power": TARGET_POWER,
        "planned_n": planned_n,
        "rows": rows,
    }


def _format(report: dict) -> str:
    lines = [
        "Illustrative binomial planning simulation; not power of the benchmark's permutation test.",
        f"Assumed ceiling {report['probe_ceiling']} from supplied validation; "
        f"chance {report['chance']}; alpha {report['alpha']}; "
        f"target power {report['target_power']}.",
        "Assumes validation accuracy transfers to generated images and trials are independent.",
        "",
        "| Effect s | Recovery acc | n for 80% power | Power at planned n={} |".format(
            report["planned_n"]
        ),
        "|---|---|---|---|",
    ]
    for row in report["rows"]:
        sample_size = row["n_for_80pct_power"]
        lines.append(
            f"| {row['effect_strength']:.1f} | {row['recovery_accuracy']:.3f} | "
            f"{sample_size if sample_size is not None else 'unreachable'} | "
            f"{row['power_at_planned_n']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Illustrative binomial sample-size simulation")
    parser.add_argument(
        "--probe-validation",
        required=True,
        help="probe validation report; assumes its accuracy is a ceiling for generated images",
    )
    parser.add_argument("--planned-n", type=int, default=420, help="per-tier n of the planned run")
    parser.add_argument("--out", default="outputs/generated/power_analysis.json")
    args = parser.parse_args()

    validation_path = Path(args.probe_validation)
    out = Path(args.out)
    try:
        validate_output_paths([out], [validation_path])
        ceiling = read_report(validation_path)["accuracy"]
        report = analyze(ceiling, args.planned_n)
    except ValueError as exc:
        parser.error(str(exc))
    payload = json.dumps(report, indent=2).encode("utf-8")
    write_output(out, payload)
    print(_format(report))


if __name__ == "__main__":
    main()
