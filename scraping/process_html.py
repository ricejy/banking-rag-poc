from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


class _PlainTextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._block_tags = {
            "p",
            "div",
            "section",
            "article",
            "header",
            "footer",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "li",
            "ul",
            "ol",
            "br",
        }

    # if hit tags identified, append to \n to indicate next line
    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._block_tags:
            self._chunks.append("\n")

    # only handle text
    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self._chunks.append(text)

    # same for endtags
    def handle_endtag(self, tag: str) -> None:
        if tag in self._block_tags:
            self._chunks.append("\n")

    # join them to put into markdown
    def markdown_text(self) -> str:
        raw = " ".join(self._chunks)
        lines = [line.strip() for line in raw.splitlines()]
        lines = [line for line in lines if line]
        return "\n\n".join(lines)


def html_to_markdown(html: str) -> str:
    try:
        from markdownify import markdownify

        return markdownify(html, heading_style="ATX")
    except Exception:
        pass

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text("\n")
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        return "\n\n".join(lines)
    except Exception:
        parser = _PlainTextHTMLParser()
        parser.feed(html)
        return parser.markdown_text()


URL_PREFIXES: list[str] = [
    "https://www.ocbc.com/personal-banking/security/secure-banking-ways/",
    "https://www.ocbc.com/personal-banking/investments/precious-metals-account"
]


def url_prefix_to_filename_prefix(url_prefix: str) -> str | None:
    parsed = urlparse(url_prefix)
    if not parsed.netloc:
        return None
    path = parsed.path.strip("/") or "index"
    safe_path = re.sub(r"[^A-Za-z0-9._-]+", "_", path)
    return f"{parsed.netloc}_{safe_path}"


def iter_matching_html_files(
    scraped_dir: Path, filename_prefixes: Iterable[str]
) -> Iterable[Path]:
    prefixes = [p for p in filename_prefixes if p]
    if not prefixes:
        return []
    for path in sorted(scraped_dir.glob("*.html")):
        if any(path.name.startswith(prefix) for prefix in prefixes):
            yield path


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    poc_dir = base_dir / "poc_html_files"
    output_dir = base_dir / "processed_markdown_files"
    output_dir.mkdir(parents=True, exist_ok=True)

    url_prefixes = URL_PREFIXES
    if not url_prefixes:
        print("No URL prefixes configured in process_html.py.")
        return

    filename_prefixes = [
        p for p in (url_prefix_to_filename_prefix(u) for u in url_prefixes) if p
    ]

    matched_files = list(iter_matching_html_files(poc_dir, filename_prefixes))
    if not matched_files:
        print("No HTML files matched. Add URL prefixes to poc_html_files.")
        return

    for html_path in matched_files:
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        markdown = html_to_markdown(html)
        output_path = output_dir / f"{html_path.stem}.md"
        output_path.write_text(markdown, encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
