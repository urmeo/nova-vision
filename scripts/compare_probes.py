"Paired comparison of two probe-validation reports (exact McNemar)."

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


def mcnemar_exact(gold, pred_a, pred_b) -> dict:
    """Two-sided exact McNemar on per-item correctness of two paired runs."""
    correct_a = [g == p for g, p in zip(gold, pred_a)]
    correct_b = [g == p for g, p in zip(gold, pred_b)]
    only_a = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)
    only_b = sum(1 for x, y in zip(correct_a, correct_b) if not x and y)
    n = only_a + only_b
    if n == 0:
        return {"only_a_correct": 0, "only_b_correct": 0, "p_value": 1.0}
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": round(p, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact McNemar between two probe reports")
    parser.add_argument("report_a", help="probe_validation JSON (with per-item outcomes)")
    parser.add_argument("report_b", help="probe_validation JSON from the same sample")
    args = parser.parse_args()

    a = json.loads(Path(args.report_a).read_text())
    b = json.loads(Path(args.report_b).read_text())
    if a["gold"] != b["gold"]:
        raise SystemExit("reports are not from the same sample; a paired test is invalid")
    for r in (a, b):
        if len(r["predictions"]) != len(r["gold"]):
            raise SystemExit("report predictions/gold length mismatch; refusing a paired test")
    out = {
        "a": {"probe": a["probe"], "accuracy": a["accuracy"]},
        "b": {"probe": b["probe"], "accuracy": b["accuracy"]},
        "n": a["n"],
        **mcnemar_exact(a["gold"], a["predictions"], b["predictions"]),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
