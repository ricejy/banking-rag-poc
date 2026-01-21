from __future__ import annotations

from pathlib import Path


def pdf_to_markdown(pdf_path: Path) -> str:
    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    pages.append(text)
            return "\n\n".join(pages).strip()
    except Exception:
        pass

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append(text)
        return "\n\n".join(pages).strip()
    except Exception:
        return ""


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    pdf_dir = base_dir / "scraped_pdf_files"
    output_dir = base_dir / "processed_markdown_files"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_dir.exists():
        print(f"Missing directory: {pdf_dir}")
        return

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in scraped_pdf_files.")
        return

    for pdf_path in pdf_files:
        markdown = pdf_to_markdown(pdf_path)
        if not markdown:
            print(f"No extractable text in {pdf_path.name}")
            continue
        output_path = output_dir / f"{pdf_path.stem}.md"
        output_path.write_text(markdown, encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
