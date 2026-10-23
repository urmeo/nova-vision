from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

from novavision.data.loaders import LEXICON_PATH

# Contractions
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)*")
_NEGATION_BOUNDARY = re.compile(r"[.!?,;:\r\n\v\f\x85\u2028\u2029…–—]|--|\s-\s")


STOPWORDS = frozenset(
    """
i a an the and or but if because as of at by for with about into onto over under
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


def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")
    return text.lower().replace("‘", "'").replace("’", "'")


def _validated_entry(word: str, values: Sequence[float]) -> tuple[str, tuple[float, float]]:
    if not isinstance(word, str) or not word.strip():
        raise ValueError("Lexicon words must be nonempty strings")
    word = _normalize_text(word).strip()
    message = f"Invalid affect for {word!r}: valence [-1,1], arousal [0,1]"
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 2:
        raise ValueError(f"Invalid affect for {word!r}: expected two numeric values")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise ValueError(message)
    try:
        valence, arousal = (float(value) for value in values)
    except OverflowError as error:
        raise ValueError(message) from error
    if not (-1 <= valence <= 1 and 0 <= arousal <= 1):
        raise ValueError(message)
    return word, (valence, arousal)


def _variants(token: str) -> Iterator[str]:
    yield token
    if token.endswith(("ies", "ied")) and len(token) > 4:
        yield token[:-3] + "y"
        yield token[:-1]
    elif token.endswith(("ing", "ed")):
        suffix_length = 3 if token.endswith("ing") else 2
        stem = token[:-suffix_length]
        if len(stem) > 2:
            yield stem + "e"
            yield stem
            if stem[-1] == stem[-2]:
                yield stem[:-1]
    elif token.endswith("es") and len(token) > 4:
        yield token[:-1]
        yield token[:-2]
    elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        yield token[:-1]


class AffectLexicon:
    def __init__(self, entries: Mapping[str, Sequence[float]]) -> None:
        if not isinstance(entries, Mapping):
            raise ValueError("Lexicon entries must map words to affect pairs")
        if not entries:
            raise ValueError("Lexicon is empty")
        self._entries: dict[str, tuple[float, float]] = {}
        for word, values in entries.items():
            word, affect = _validated_entry(word, values)
            if word in self._entries:
                raise ValueError(f"Duplicate lexicon word: {word!r}")
            self._entries[word] = affect

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, word: str) -> tuple[float, float] | None:
        for variant in _variants(_normalize_text(word).strip()):
            if variant in STOPWORDS or variant in NEGATORS:
                continue
            if variant in self._entries:
                return self._entries[variant]
        return None

    def score(self, text: str) -> AffectScore:
        text = _normalize_text(text)
        matched: list[tuple[float, float]] = []
        content_count = 0
        previous_tokens: list[str] = []
        previous_end = 0
        for match in _TOKEN.finditer(text):
            token = match.group()
            contrast = token in {"but", "however"} or (
                token == "yet" and bool(previous_tokens) and previous_tokens[-1] not in NEGATORS
            )
            if contrast or _NEGATION_BOUNDARY.search(text[previous_end : match.start()]):
                previous_tokens.clear()
            negated = any(previous in NEGATORS for previous in previous_tokens)
            previous_tokens = [*previous_tokens[-1:], token]
            previous_end = match.end()
            if token in STOPWORDS or token in NEGATORS:
                continue
            content_count += 1
            affect = self.lookup(token)
            if affect is None:
                continue
            valence, arousal = affect

            if negated:
                valence = -valence
            matched.append((valence, arousal))
        if not matched:
            return AffectScore(0.0, 0.5, 0.0)

        valence = sum(value for value, _ in matched) / len(matched)
        arousal = sum(value for _, value in matched) / len(matched)
        coverage = len(matched) / content_count
        return AffectScore(round(valence, 4), round(arousal, 4), round(coverage, 4))

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> AffectLexicon:
        path = Path(path if path is not None else os.getenv("NOVAVISION_LEXICON") or LEXICON_PATH)
        entries: dict[str, tuple[float, float]] = {}
        with path.open(encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = [field.strip() for field in line.rstrip("\r\n").split("\t")]

                if [field.lower() for field in parts[:3]] == ["word", "valence", "arousal"]:
                    continue
                if len(parts) < 3:
                    raise ValueError(f"{path}:{line_number} has fewer than 3 tab-separated columns")
                word, valence, arousal = parts[:3]
                try:
                    values = (float(valence), float(arousal))
                except ValueError as error:
                    raise ValueError(
                        f"{path}:{line_number} has non-numeric valence/arousal"
                    ) from error
                try:
                    word, affect = _validated_entry(word, values)
                except ValueError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                if word in entries:
                    raise ValueError(f"{path}:{line_number}: Duplicate lexicon word: {word!r}")
                entries[word] = affect
        if not entries:
            raise ValueError(f"{path} contains no lexicon entries")
        return cls(entries)
