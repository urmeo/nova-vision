from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

from novavision.affect.lexicon import _normalize_text
from novavision.outputs import write_output

WARRINER_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.3758%2Fs13428-012-0314-x/MediaObjects/13428_2012_314_MOESM1_ESM.zip"
)
WARRINER_SHA256 = "5a0db0437ef234219fb899caced5eb3195dbf9f84396637ab4db74c50ca441aa"
CSV_MEMBER = "BRM-emot-submit.csv"


class _HTTPSOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.lower().startswith("https://"):
            raise ValueError(f"refusing a non-HTTPS redirect to {newurl!r}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_https(url: str) -> bytes:
    if not url.lower().startswith("https://"):
        raise ValueError("refusing a non-HTTPS URL")
    opener = urllib.request.build_opener(_HTTPSOnlyRedirect)
    with opener.open(url, timeout=30) as resp:
        return resp.read()


def verify_sha256(blob: bytes, expected: str) -> str:
    got = hashlib.sha256(blob).hexdigest()
    if expected and got != expected:
        raise ValueError(f"checksum mismatch: expected {expected}, got {got}")
    return got


def extract_csv(zip_bytes: bytes, member: str = CSV_MEMBER) -> str:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        try:
            return zf.read(member).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"Norms CSV {member!r} is not valid UTF-8") from error


def convert(raw: str) -> list[tuple[str, float, float]]:
    reader = csv.DictReader(io.StringIO(raw.lstrip("\ufeff")))
    required = {"Word", "V.Mean.Sum", "A.Mean.Sum"}
    if reader.fieldnames is None or required - set(reader.fieldnames):
        raise ValueError("Norms CSV must contain Word, V.Mean.Sum and A.Mean.Sum columns")
    rows = []
    words: set[str] = set()
    for row in reader:
        try:
            word = _normalize_text(row["Word"] or "").strip()
            valence = (float(row["V.Mean.Sum"]) - 5.0) / 4.0
            arousal = (float(row["A.Mean.Sum"]) - 1.0) / 8.0
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Norms CSV line {reader.line_num} has invalid affect values") from exc
        if not word or not (-1 <= valence <= 1 and 0 <= arousal <= 1):
            raise ValueError(f"Norms CSV line {reader.line_num} needs a word and scores in [1, 9]")
        if any(separator in word for separator in "\t\r\n"):
            raise ValueError(f"Norms CSV line {reader.line_num} has a word with a tab or newline")
        if word.startswith("#"):
            raise ValueError(f"Norms CSV line {reader.line_num} has a comment-prefixed word")
        if word in words:
            raise ValueError(f"Norms CSV line {reader.line_num} has duplicate word {word!r}")
        words.add(word)
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
    content = "word\tvalence\tarousal\n" + "".join(
        f"{word}\t{valence}\t{arousal}\n" for word, valence, arousal in rows
    )
    encoded = content.encode("utf-8")

    out = Path(args.out)
    write_output(out, encoded)

    print(f"Wrote {out} ({len(rows)} words). Set NOVAVISION_LEXICON={out} to use it.")


if __name__ == "__main__":
    main()
