from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from novavision.eval.metrics import accuracy
from novavision.eval.metrics import mcnemar_exact as binary_mcnemar_exact
from novavision.eval.reports import read_report
from novavision.taxonomy import EMOTIONS


def mcnemar_exact(gold: Sequence[str], pred_a: Sequence[str], pred_b: Sequence[str]) -> dict:
    if len(gold) != len(pred_a) or len(gold) != len(pred_b):
        raise ValueError("Paired gold and prediction arrays must have equal lengths")
    correct_a = [g == p for g, p in zip(gold, pred_a, strict=True)]
    correct_b = [g == p for g, p in zip(gold, pred_b, strict=True)]
    return binary_mcnemar_exact(correct_a, correct_b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact McNemar between two probe reports")
    parser.add_argument("report_a", help="probe validation report with per-item outcomes")
    parser.add_argument("report_b", help="probe validation report from the same sample")
    args = parser.parse_args()

    report_a = read_report(Path(args.report_a))
    report_b = read_report(Path(args.report_b))
    for report in (report_a, report_b):
        gold, predictions = report.get("gold"), report.get("predictions")
        if (
            not isinstance(gold, list)
            or not isinstance(predictions, list)
            or not gold
            or not predictions
        ):
            raise SystemExit("report gold and predictions must be nonempty arrays")
        if any(
            not isinstance(label, str) or label not in EMOTIONS
            for values in (gold, predictions)
            for label in values
        ):
            raise SystemExit("report gold and predictions must contain canonical emotion labels")
        n = report.get("n")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise SystemExit("report sample count must be a positive integer")
        if len(predictions) != len(gold):
            raise SystemExit("report predictions/gold length mismatch; refusing a paired test")
        if n != len(gold):
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
    if report_a["gold"] != report_b["gold"]:
        raise SystemExit("reports are not from the same sample; a paired test is invalid")
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
