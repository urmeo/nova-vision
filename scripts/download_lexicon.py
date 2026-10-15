"Download and verify the Warriner (2013) affect norms."

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

WARRINER_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.3758%2Fs13428-012-0314-x/MediaObjects/13428_2012_314_MOESM1_ESM.zip"
)
WARRINER_SHA256 = "5a0db0437ef234219fb899caced5eb3195dbf9f84396637ab4db74c50ca441aa"
CSV_MEMBER = "BRM-emot-submit.csv"


class _HTTPSOnlyRedirect(urllib.request.HTTPRedirectHandler):
    "Reject any redirect that would downgrade the transport off HTTPS."

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.lower().startswith("https://"):
            raise ValueError(f"refusing a non-HTTPS redirect to {newurl!r}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_https(url: str) -> bytes:
    "Fetch over HTTPS; reject transport downgrades."
    if not url.lower().startswith("https://"):
        raise ValueError("refusing a non-HTTPS URL; the norms back research results")
    opener = urllib.request.build_opener(_HTTPSOnlyRedirect)
    with opener.open(url, timeout=30) as resp:  # noqa: S310 - HTTPS enforced on every hop above
        return resp.read()


def verify_sha256(blob: bytes, expected: str) -> str:
    """Return the digest; raise if it does not match a non-empty ``expected``."""
    got = hashlib.sha256(blob).hexdigest()
    if expected and got != expected:
        raise ValueError(
            f"checksum mismatch: expected {expected}, got {got}. "
            "Refusing a tampered or silently updated file; pass --sha256 to accept it."
        )
    return got


def extract_csv(zip_bytes: bytes, member: str = CSV_MEMBER) -> str:
    """Read one CSV member out of the norms zip."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        return zf.read(member).decode("utf-8", errors="replace")


def convert(raw: str) -> list[tuple[str, float, float]]:
    """Warriner 1..9 valence/arousal to NovaVision valence (-1..1)/arousal (0..1)."""
    reader = csv.DictReader(io.StringIO(raw.lstrip("\ufeff")))
    required = {"Word", "V.Mean.Sum", "A.Mean.Sum"}
    if reader.fieldnames is None or required - set(reader.fieldnames):
        raise ValueError("Norms CSV must contain Word, V.Mean.Sum and A.Mean.Sum columns")
    rows = []
    for row in reader:
        try:
            word = (row["Word"] or "").strip().lower()
            valence = (float(row["V.Mean.Sum"]) - 5.0) / 4.0
            arousal = (float(row["A.Mean.Sum"]) - 1.0) / 8.0
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Norms CSV line {reader.line_num} has invalid affect values") from exc
        if not word or not (-1 <= valence <= 1 and 0 <= arousal <= 1):
            raise ValueError(f"Norms CSV line {reader.line_num} needs a word and scores in [1, 9]")
        rows.append((word, round(valence, 4), round(arousal, 4)))
    if not rows:
        raise ValueError("Norms CSV contains no entries")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the Warriner affect norms")
    parser.add_argument("--out", default="outputs/generated/warriner.tsv")
    parser.add_argument("--url", default=WARRINER_URL, help="norms zip URL (HTTPS)")
    parser.add_argument("--member", default=CSV_MEMBER, help="CSV filename inside the zip")
    parser.add_argument(
        "--sha256",
        default=WARRINER_SHA256,
        help="expected SHA-256 of the downloaded zip (pinned; override for a mirror)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the integrity check (not recommended)",
    )
    args = parser.parse_args()

    blob = fetch_https(args.url)
    if args.no_verify:
        print("WARNING: --no-verify set; skipping integrity check.", file=sys.stderr)
    else:
        if not args.sha256:
            parser.error("empty --sha256; pass a digest or use --no-verify to skip explicitly")
        verify_sha256(blob, args.sha256)
    rows = convert(extract_csv(blob, args.member))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("word\tvalence\tarousal\n")
        for word, valence, arousal in rows:
            fh.write(f"{word}\t{valence}\t{arousal}\n")

    print(f"Wrote {out} ({len(rows)} words). Set NOVAVISION_LEXICON={out} to use it.")


if __name__ == "__main__":
    main()
