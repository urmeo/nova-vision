from pathlib import Path
from shutil import copy2, copytree, rmtree
from tempfile import mkdtemp

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "build" / "pages"


def main() -> None:
    html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    marker = '<html lang="en" data-preview="true">'
    if html.count(marker) != 1:
        raise ValueError("Expected one HTML root element for preview mode")

    if DESTINATION.is_symlink() or (DESTINATION.exists() and not DESTINATION.is_dir()):
        raise ValueError("Preview destination must be a directory")
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(mkdtemp(prefix=".pages-", dir=DESTINATION.parent))
    previous = temporary / "previous"
    try:
        staged = temporary / "pages"
        staged.mkdir()
        for filename in ("index.html", "styles.css", "app.js", "favicon.svg"):
            copy2(ROOT / "docs" / filename, staged / filename)
        copytree(ROOT / "docs" / "assets", staged / "assets")
        (staged / ".nojekyll").touch()
        try:
            if DESTINATION.exists():
                DESTINATION.rename(previous)
            staged.rename(DESTINATION)
        except BaseException:
            if previous.exists() and not DESTINATION.exists():
                try:
                    previous.rename(DESTINATION)
                except OSError as exc:
                    raise OSError(f"Previous preview is preserved at {previous}") from exc
            raise
        if previous.exists():
            rmtree(previous)
    finally:
        if not previous.exists():
            rmtree(temporary)
    print(f"Built preview: {DESTINATION.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
