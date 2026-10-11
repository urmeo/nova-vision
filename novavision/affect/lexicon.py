"""Valence/arousal scoring of text from an affect lexicon."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from novavision.data import LEXICON_PATH

_TOKEN = re.compile(r"[a-z][a-z']+")


STOPWORDS = frozenset(
    """
a an the and or but if because as of at by for with about into onto over under
again then once here there when where why how all any both each few more most
other some such than too very just own same so that this these those me my mine
we us our ours you your yours he him his she her hers it its they them their
theirs what which who whom am is are was were be been being have has had having
do does did doing done will would shall should can could may might must ought
to from in on out off it's i'm i've i'll you're we're they're he's she's that's
there's
""".split()
)

NEGATORS = frozenset(
    "not no never nor cannot can't don't doesn't didn't isn't wasn't aren't "
    "weren't won't wouldn't couldn't shouldn't ain't without".split()
)


@dataclass(frozen=True)
class AffectScore:
    valence: float
    arousal: float
    coverage: float


def _variants(token: str) -> Iterator[str]:
    "Candidate lemmas in priority order."
    yield token
    if token.endswith(("ies", "ied")) and len(token) > 4:
        yield token[:-3] + "y"
        yield token[:-1]
    elif token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        yield stem + "e"
        yield stem
        if len(stem) > 2 and stem[-1] == stem[-2]:
            yield stem[:-1]
    elif token.endswith("ed") and len(token) > 4:
        stem = token[:-2]
        yield stem + "e"
        yield stem
        if len(stem) > 2 and stem[-1] == stem[-2]:
            yield stem[:-1]
    elif token.endswith("es") and len(token) > 4:
        yield token[:-2]
        yield token[:-1]
    elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        yield token[:-1]


class AffectLexicon:
    """Maps words to valence (-1..1) and arousal (0..1)."""

    def __init__(self, entries: dict[str, tuple[float, float]]):
        if not entries:
            raise ValueError("Lexicon is empty")
        for word, (valence, arousal) in entries.items():
            if not (-1 <= valence <= 1 and 0 <= arousal <= 1):
                raise ValueError(f"Invalid affect for {word!r}: valence [-1,1], arousal [0,1]")
        self._entries = dict(entries)

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, word: str) -> tuple[float, float] | None:
        for variant in _variants(word.lower()):
            if variant in STOPWORDS or variant in NEGATORS:
                continue
            if variant in self._entries:
                return self._entries[variant]
        return None

    def score(self, text: str) -> AffectScore:
        tokens = _TOKEN.findall(text.lower().replace("’", "'"))
        matched: list[tuple[float, float]] = []
        content_count = 0
        for index, token in enumerate(tokens):
            if token in STOPWORDS or token in NEGATORS:
                continue
            content_count += 1
            affect = self.lookup(token)
            if affect is None:
                continue
            valence, arousal = affect

            if any(previous in NEGATORS for previous in tokens[max(0, index - 2) : index]):
                valence = -valence
            matched.append((valence, arousal))
        if not matched:
            return AffectScore(0.0, 0.5, 0.0)

        valence = sum(value for value, _ in matched) / len(matched)
        arousal = sum(value for _, value in matched) / len(matched)
        coverage = len(matched) / content_count
        return AffectScore(round(valence, 4), round(arousal, 4), round(coverage, 4))

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> AffectLexicon:
        path = Path(path or os.getenv("NOVAVISION_LEXICON") or LEXICON_PATH)
        entries: dict[str, tuple[float, float]] = {}
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [field.strip() for field in line.split("\t")]

                if [field.lower() for field in parts[:3]] == ["word", "valence", "arousal"]:
                    continue
                if len(parts) < 3:
                    raise ValueError(f"{path}:{line_number} has fewer than 3 tab-separated columns")
                word, valence, arousal = parts[:3]
                try:
                    entries[word.lower()] = (float(valence), float(arousal))
                except ValueError:
                    raise ValueError(
                        f"{path}:{line_number} has non-numeric valence/arousal"
                    ) from None
        return cls(entries)
