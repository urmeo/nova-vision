from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from PIL import Image

from novavision.eval.probes import image_digest as _image_digest
from novavision.experiments.validation import _check_checkpoint_record
from novavision.outputs import atomic_output, json_safe


def _open_checkpoint(path: Path, mode: str):
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("Resume checkpoint cannot use a symbolic link")
    flags = {
        "rb": os.O_RDONLY,
        "r+b": os.O_RDWR,
        "x": os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        "a": os.O_WRONLY | os.O_APPEND,
    }[mode]
    flags |= (
        getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    )
    descriptor = os.open(path, flags, 0o666)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("Resume checkpoint must be a regular file without other hard links")
        return os.fdopen(descriptor, mode, encoding=None if "b" in mode else "utf-8")
    except BaseException:
        os.close(descriptor)
        raise


class _Checkpoint:
    def __init__(self, path: Path | None, manifest: dict, *, images_dir: Path | None = None):
        self.path = path
        self.images_dir = images_dir
        self.done: dict[tuple[str, str, int, int], dict] = {}
        if path and path.exists():
            truncate_at = None
            with _open_checkpoint(path, "rb") as fh:
                header = fh.readline()
                try:
                    matches = json.loads(header.decode("utf-8")) == {"manifest": manifest}
                except (json.JSONDecodeError, UnicodeDecodeError):
                    matches = False
                if not matches:
                    raise ValueError(
                        "Resume checkpoint does not match this run's provenance; "
                        "use a fresh output directory."
                    )
                last_line = header
                for line_number, line in enumerate(fh, start=2):
                    if not line.strip():
                        last_line = line
                        continue
                    try:
                        record = json.loads(line.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        if line.endswith(b"\n"):
                            raise ValueError(
                                f"Resume checkpoint contains invalid JSON on line {line_number}"
                            ) from exc
                        truncate_at = fh.tell() - len(line)
                        break
                    _check_checkpoint_record(
                        record, f"Invalid checkpoint record in {path} on line {line_number}"
                    )
                    key = (record["tier"], record["intended"], record["index"], record["seed"])
                    if key in self.done and self.done[key] != record:
                        raise ValueError("Resume checkpoint contains conflicting records")
                    self.done[key] = record
                    last_line = line
            if images_dir is not None:
                for record in self.done.values():
                    self._verify_original(record)
            if truncate_at is not None or not last_line.endswith(b"\n"):
                with _open_checkpoint(path, "r+b") as fh:
                    if truncate_at is not None:
                        fh.truncate(truncate_at)
                    if not last_line.endswith(b"\n"):
                        fh.seek(0, os.SEEK_END)
                        fh.write(b"\n")
        elif path:
            with _open_checkpoint(path, "x") as fh:
                fh.write(json.dumps({"manifest": manifest}, allow_nan=False) + "\n")

    def cached(self, tier: str, intended: str, index: int, seed: int) -> dict | None:
        return self.done.get((tier, intended, index, seed))

    def _verify_original(self, rec: dict) -> None:
        assert self.images_dir is not None
        relative = rec.get("image_path")
        digest = rec.get("image_pixel_sha256")
        if not relative or not digest:
            raise ValueError("Resume checkpoint is missing a saved original image reference/digest")
        source = (self.images_dir.parent / relative).resolve()
        if self.images_dir.resolve() not in source.parents or not source.is_file():
            raise ValueError("Saved original image is missing from the resumed run")
        try:
            with Image.open(source) as image:
                actual_digest = _image_digest(image)
        except OSError as exc:
            raise ValueError("Saved original image could not be decoded") from exc
        if actual_digest != digest:
            raise ValueError("Saved original image pixel digest differs from the checkpoint")

    def record(self, rec: dict, *, image: Image.Image | None = None) -> dict:
        _check_checkpoint_record(json_safe(rec))
        if self.images_dir is not None:
            if image is None or _image_digest(image) != rec.get("image_pixel_sha256"):
                raise ValueError("Saved original must match the pixels recorded by the probe")
            self.images_dir.mkdir(parents=True, exist_ok=True)
            name = (
                f"{rec['tier']}-{rec['index']:05d}-{rec['intended']}-{rec['seed']:03d}-"
                f"{rec['image_pixel_sha256']}.png"
            )
            target = self.images_dir / name
            with atomic_output(target) as stream:
                image.save(stream, format="PNG")
            rec = {**rec, "image_path": f"images/{name}"}
        if self.path:
            with _open_checkpoint(self.path, "a") as fh:
                fh.write(json.dumps(json_safe(rec)) + "\n")
        return rec
