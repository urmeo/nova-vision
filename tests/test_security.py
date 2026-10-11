"""Regression checks for unsafe deserialization and default public binding."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "novavision", ROOT / "scripts", ROOT / "server.py"]


_UNSAFE = re.compile(
    r"\bpickle\.loads?\b|\btorch\.load\s*\(|weights_only\s*=\s*False|"
    r"yaml\.load\s*\((?!.*SafeLoader)|\bos\.system\s*\(|shell\s*=\s*True"
)


def _py_files():
    for source in SOURCES:
        if source.is_dir():
            yield from source.rglob("*.py")
        elif source.exists():
            yield source


def test_no_unsafe_deserialization_or_shell():
    offenders = []
    for path in _py_files():
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if _UNSAFE.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}")
    assert not offenders, "unsafe deserialization/shell call introduced:\n" + "\n".join(offenders)


def test_no_default_public_bind():
    server = (ROOT / "server.py").read_text()
    bind = re.compile(r"""getenv\(\s*["']HOST["']\s*,\s*["']0\.0\.0\.0["']""")
    assert not bind.search(server)
    assert "resolve_host()" in server
