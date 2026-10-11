"Estimate sample sizes with an illustrative binomial simulation."

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CHANCE = 1.0 / 7.0
ALPHA = 0.05
TARGET_POWER = 0.80
EFFECTS = (0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
_SIM = 4000


def _critical_count(n: int, chance: float, rng: np.random.Generator) -> int:
    """Smallest correct-count that clears the one-sided chance null at ALPHA."""
    null = rng.binomial(n, chance, _SIM)
    return int(np.quantile(null, 1 - ALPHA)) + 1


def power_at(n: int, p1: float, chance: float, rng: np.random.Generator) -> float:
    crit = _critical_count(n, chance, rng)
    return float(np.mean(rng.binomial(n, p1, _SIM) >= crit))


def min_n_for_power(
    p1: float, chance: float, rng: np.random.Generator, n_max: int = 5000
) -> int | None:
    """Smallest tested n reaching TARGET_POWER, or None within n_max."""
    if p1 <= chance:
        return None
    for n in range(10, n_max + 1, 10):
        if power_at(n, p1, chance, rng) >= TARGET_POWER:
            return n
    return None


def analyze(ceiling: float, planned_n: int) -> dict:
    rng = np.random.default_rng(0)
    rows = []
    for s in EFFECTS:
        p1 = s * ceiling + (1 - s) * CHANCE
        rows.append(
            {
                "effect_strength": s,
                "recovery_accuracy": round(p1, 4),
                "n_for_80pct_power": min_n_for_power(p1, CHANCE, rng),
                "power_at_planned_n": round(power_at(planned_n, p1, CHANCE, rng), 3),
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
        f"Assumed ceiling {report['probe_ceiling']} from real-scene validation; "
        f"chance {report['chance']}; alpha {report['alpha']}; "
        f"target power {report['target_power']}.",
        "Assumes validation accuracy transfers to generated images and trials are independent.",
        "",
        "| Effect s | Recovery acc | n for 80% power | Power at planned n={} |".format(
            report["planned_n"]
        ),
        "|---|---|---|---|",
    ]
    for r in report["rows"]:
        n = r["n_for_80pct_power"]
        lines.append(
            f"| {r['effect_strength']:.1f} | {r['recovery_accuracy']:.3f} | "
            f"{n if n is not None else 'unreachable'} | {r['power_at_planned_n']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Illustrative binomial sample-size simulation")
    parser.add_argument(
        "--probe-validation",
        default="outputs/results/probe_validation_scene_l14.json",
        help="probe validation JSON; assumes its accuracy is a ceiling for generated images",
    )
    parser.add_argument("--planned-n", type=int, default=420, help="per-tier n of the planned run")
    parser.add_argument("--out", default="outputs/generated/power_analysis.json")
    args = parser.parse_args()

    ceiling = json.loads(Path(args.probe_validation).read_text())["accuracy"]
    report = analyze(ceiling, args.planned_n)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(_format(report))


if __name__ == "__main__":
    main()
