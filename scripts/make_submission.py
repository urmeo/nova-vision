from __future__ import annotations

import argparse
import json
from pathlib import Path

from novavision.experiments.run import ALL_CONDITIONS
from novavision.outputs import validate_output_paths, write_output


def build_submission(results: dict, system: str) -> dict:
    manifest = results["manifest"]
    config = manifest["config"]
    backend = config.get("backend", "").lower()
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
        "generator": ("null" if backend == "null" else config.get("diffusion_model", "unknown")),
        "probe": config.get("probe_model") or config.get("probe", "unknown"),
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

    results_path = Path(args.results)
    out = Path(args.out)
    try:
        validate_output_paths([out], [results_path])
    except ValueError as exc:
        parser.error(str(exc))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    submission = build_submission(results, args.system)
    try:
        encoded = json.dumps(submission, indent=2, allow_nan=False).encode("utf-8")
    except ValueError as exc:
        parser.error(f"Cannot serialize submission: {exc}")
    write_output(out, encoded)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
