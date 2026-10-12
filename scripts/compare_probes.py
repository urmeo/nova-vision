"Paired comparison of two probe-validation reports (exact McNemar)."

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from math import comb
from pathlib import Path

from novavision.eval.metrics import accuracy


def mcnemar_exact(gold: Sequence[str], pred_a: Sequence[str], pred_b: Sequence[str]) -> dict:
    """Two-sided exact McNemar on per-item correctness of two paired runs."""
    if len(gold) != len(pred_a) or len(gold) != len(pred_b):
        raise ValueError("Paired gold and prediction arrays must have equal lengths")
    correct_a = [g == p for g, p in zip(gold, pred_a)]
    correct_b = [g == p for g, p in zip(gold, pred_b)]
    only_a = sum(a and not b for a, b in zip(correct_a, correct_b))
    only_b = sum(b and not a for a, b in zip(correct_a, correct_b))
    discordant = only_a + only_b
    if discordant == 0:
        return {"only_a_correct": 0, "only_b_correct": 0, "p_value": 1.0}
    tail_count = min(only_a, only_b)
    p_value = min(1.0, 2 * sum(comb(discordant, i) for i in range(tail_count + 1)) / 2**discordant)
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": round(p_value, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact McNemar between two probe reports")
    parser.add_argument("report_a", help="probe_validation JSON (with per-item outcomes)")
    parser.add_argument("report_b", help="probe_validation JSON from the same sample")
    args = parser.parse_args()

    report_a = json.loads(Path(args.report_a).read_text(encoding="utf-8"))
    report_b = json.loads(Path(args.report_b).read_text(encoding="utf-8"))
    if report_a["gold"] != report_b["gold"]:
        raise SystemExit("reports are not from the same sample; a paired test is invalid")
    for report in (report_a, report_b):
        if len(report["predictions"]) != len(report["gold"]):
            raise SystemExit("report predictions/gold length mismatch; refusing a paired test")
        if report["n"] != len(report["gold"]):
            raise SystemExit(
                "report sample count disagrees with gold labels; refusing a paired test"
            )
        digests = report.get("image_digests")
        if (
            not isinstance(digests, list)
            or len(digests) != report["n"]
            or not digests
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                for digest in digests
            )
        ):
            raise SystemExit(
                "report lacks valid image fingerprints; rerun validation before a paired test"
            )
    if report_a["image_digests"] != report_b["image_digests"]:
        raise SystemExit("reports use different images or image order; a paired test is invalid")
    out = {
        "a": {
            "probe": report_a["probe"],
            "accuracy": round(accuracy(report_a["gold"], report_a["predictions"]), 4),
        },
        "b": {
            "probe": report_b["probe"],
            "accuracy": round(accuracy(report_b["gold"], report_b["predictions"]), 4),
        },
        "n": report_a["n"],
        **mcnemar_exact(report_a["gold"], report_a["predictions"], report_b["predictions"]),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
