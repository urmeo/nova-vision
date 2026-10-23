from __future__ import annotations

import argparse
import json
from pathlib import Path

from novavision.experiments import run as run_mod
from novavision.experiments.manifest import git_sha, package_version
from novavision.taxonomy import EMOTIONS

_MEASUREMENTS = (
    "intended_valence",
    "intended_arousal",
    "recovered_valence",
    "recovered_arousal",
)


def _validate_record(record: dict, position: int) -> None:
    context = f"Record {position}"
    if not isinstance(record, dict):
        raise ValueError(f"{context} must be an object")
    required = {"tier", "content", "intended", "seed", "predicted", *_MEASUREMENTS}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"{context} is missing fields: {', '.join(sorted(missing))}")
    if not isinstance(record["tier"], str) or record["tier"] not in run_mod.ALL_CONDITIONS:
        raise ValueError(f"{context} has an unknown tier")
    if not isinstance(record["content"], str):
        raise ValueError(f"{context} content must be a string")
    for field in ("intended", "predicted", "classified"):
        if field == "classified" and record.get(field) is None:
            continue
        if not isinstance(record[field], str) or record[field] not in EMOTIONS:
            raise ValueError(f"{context} {field} must be a known emotion label")
    for field in ("seed", "index"):
        if field not in record:
            continue
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{context} {field} must be a nonnegative integer")
    for field in _MEASUREMENTS:
        value = record[field]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"{context} {field} must be numeric or null")


def _conditions_for(records: list[dict]) -> tuple[str, ...]:
    if not isinstance(records, list):
        raise ValueError("Results records must be a list")
    tiers = set()
    pairings = set()
    for position, record in enumerate(records, start=1):
        _validate_record(record, position)
        key = (
            record["tier"],
            record.get("index", record["content"]),
            record["intended"],
            record["seed"],
        )
        if key in pairings:
            raise ValueError(f"Record {position} duplicates a tier/item/emotion/seed pairing")
        pairings.add(key)
        tiers.add(record["tier"])
    if {"scene", "shuffled"} <= tiers:
        raise ValueError("Results cannot mix content-track scene and text-track shuffled records")
    track = "text" if "shuffled" in tiers else "content"
    return run_mod.CONDITIONS[track]


def resummarize(
    results_path: str | Path,
    *,
    figures_dir: str | Path | None = None,
    out: str | Path | None = None,
) -> dict:
    path = Path(results_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError("Results must be an object containing records")
    if "manifest" in payload and not isinstance(payload["manifest"], dict):
        raise ValueError("Results manifest must be an object")
    records = payload["records"]
    conditions = _conditions_for(records)

    payload["metrics"] = run_mod._summarize(records, conditions)
    payload["contrasts"] = run_mod._contrasts(records)
    payload.setdefault("manifest", {})["reanalysis"] = {
        "git_sha": git_sha(),
        "note": "metrics/diagnostics recomputed from the original records; no images regenerated",
        "packages": {"numpy": package_version("numpy")},
    }
    destination = Path(out) if out is not None else path
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_mod.dump_results(payload, destination)
    run_mod._write_figures(
        destination.parent, records, payload["metrics"], conditions, figures_dir=figures_dir
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute summary from existing records")
    parser.add_argument("--results", default="outputs/generated/run/results.json")
    parser.add_argument("--out", default="outputs/generated/reanalysis/results.json")
    parser.add_argument("--figures", default=None, help="override the figure directory")
    args = parser.parse_args()
    payload = resummarize(args.results, figures_dir=args.figures, out=args.out)
    health = payload["metrics"].get("probe_health", {})
    print(json.dumps({"reanalyzed": args.results, "probe_health": health}, indent=2))


if __name__ == "__main__":
    main()
