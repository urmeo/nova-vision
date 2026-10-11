"Recompute metrics, contrasts, and figures from an existing run's records."

from __future__ import annotations

import argparse
import json
from pathlib import Path

from novavision.experiments import run as run_mod
from novavision.experiments.manifest import git_sha, package_version


def _conditions_for(records: list[dict]) -> tuple[str, ...]:
    tiers = {r["tier"] for r in records}
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
    records = payload["records"]
    conditions = _conditions_for(records)

    payload["metrics"] = run_mod._summarize(records, conditions)
    payload["contrasts"] = run_mod._contrasts(records)
    payload.setdefault("manifest", {})["reanalysis"] = {
        "git_sha": git_sha(),
        "note": "metrics/diagnostics recomputed from the original records; no images regenerated",
        "packages": {pkg: package_version(pkg) for pkg in ("numpy",)},
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
    parser.add_argument("--results", default="outputs/results/results.json")
    parser.add_argument("--out", default="outputs/generated/reanalysis/results.json")
    parser.add_argument("--figures", default=None, help="override the figure directory")
    args = parser.parse_args()
    payload = resummarize(args.results, figures_dir=args.figures, out=args.out)
    health = payload["metrics"].get("probe_health", {})
    print(json.dumps({"reanalyzed": args.results, "probe_health": health}, indent=2))


if __name__ == "__main__":
    main()
