"Turn a run's results.json into a schema-valid benchmark submission."

from __future__ import annotations

import argparse
import json
from pathlib import Path

from novavision.experiments.run import ALL_CONDITIONS


def build_submission(results: dict, system: str) -> dict:
    manifest = results["manifest"]
    config = manifest["config"]
    metrics = results["metrics"]

    conditions = {}
    for tier in ALL_CONDITIONS:
        row = metrics.get(tier)
        if not row:
            continue
        control = row.get("shuffled_control") or {}
        conditions[tier] = {
            "accuracy": row["accuracy"],
            "accuracy_ci": row["accuracy_ci"],
            "shuffled_p": control.get("p_value"),
            "n": row["n"],
        }

    submission = {
        "system": system,
        "track": config.get("track", "content"),
        "generator": config.get("diffusion_model", "unknown"),
        "probe": config.get("probe", "unknown"),
        "conditions": conditions,
        "provenance": {
            "git_sha": manifest.get("git_sha", "unknown"),
            "seeds": config.get("seeds", 1),
            "base_seed": config.get("base_seed", 0),
            "benchmark_sha256": config.get("benchmark_sha256"),
        },
    }

    health = metrics.get("probe_health") or {}
    if health.get("distinct_labels") and health.get("n_labels"):
        submission["probe_health"] = {
            "distinct_labels": health["distinct_labels"],
            "n_labels": health["n_labels"],
        }
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a benchmark submission from a run")
    parser.add_argument("--results", default="outputs/generated/run/results.json")
    parser.add_argument("--system", required=True, help="name of the submitted system")
    parser.add_argument("--out", default="outputs/generated/submission.json")
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    submission = build_submission(results, args.system)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(submission, indent=2), encoding="utf-8")
    print(
        f"Wrote {args.out}. "
        "Validate against docs/benchmark/submission.schema.json before submitting."
    )


if __name__ == "__main__":
    main()
