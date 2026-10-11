"""Build the GitHub Pages preview without models or project dependencies."""

from pathlib import Path
from shutil import copy2, copytree, rmtree

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "build" / "pages"


def main() -> None:
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    marker = '<html lang="en">'
    if html.count(marker) != 1:
        raise ValueError("Expected one HTML root element for preview mode")

    if DESTINATION.exists():
        rmtree(DESTINATION)
    DESTINATION.mkdir(parents=True)
    (DESTINATION / "index.html").write_text(
        html.replace(marker, '<html lang="en" data-preview="true">'), encoding="utf-8"
    )
    for filename in ("styles.css", "app.js", "favicon.svg"):
        copy2(ROOT / "static" / filename, DESTINATION / filename)
    copytree(ROOT / "docs" / "assets", DESTINATION / "assets")
    (DESTINATION / ".nojekyll").touch()
    print(f"Built preview: {DESTINATION.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
