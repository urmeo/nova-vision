import hashlib
import io
import zipfile

import download_lexicon as dl
import pytest

_CSV = "Word,V.Mean.Sum,A.Mean.Sum\nhappy,7.0,5.0\ngloom,2.0,3.0\n"


def _zip_bytes(name=dl.CSV_MEMBER, body=_CSV) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, body)
    return buf.getvalue()


def test_verify_sha256_accepts_match():
    blob = b"abc"
    digest = hashlib.sha256(blob).hexdigest()
    assert dl.verify_sha256(blob, digest) == digest


def test_verify_sha256_rejects_tampering():
    with pytest.raises(ValueError, match="checksum mismatch"):
        dl.verify_sha256(b"tampered", hashlib.sha256(b"original").hexdigest())


def test_verify_sha256_empty_expected_skips():
    assert dl.verify_sha256(b"anything", "")


def test_https_only_redirect_rejects_downgrade():
    handler = dl._HTTPSOnlyRedirect()
    with pytest.raises(ValueError, match="non-HTTPS redirect"):
        handler.redirect_request(None, None, 302, "Found", {}, "http://evil.example/x")


def test_fetch_https_refuses_plaintext_url():
    with pytest.raises(ValueError, match="non-HTTPS URL"):
        dl.fetch_https("http://crr.ugent.be/x.zip")


def test_extract_and_convert_roundtrip():
    raw = dl.extract_csv(_zip_bytes())
    rows = dl.convert(raw)
    words = {w: (v, a) for w, v, a in rows}
    assert words["happy"] == (0.5, 0.5)
    assert words["gloom"][0] < 0


def test_convert_normalizes_range():
    rows = dl.convert("Word,V.Mean.Sum,A.Mean.Sum\nX,9.0,9.0\nY,1.0,1.0\n")
    vals = {w: (v, a) for w, v, a in rows}
    assert vals["x"] == (1.0, 1.0)
    assert vals["y"] == (-1.0, 0.0)
