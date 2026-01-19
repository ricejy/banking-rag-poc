from __future__ import annotations

import re
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb import Documents, EmbeddingFunction, Embeddings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator
from config import MAX_WORDS, WORD_OVERLAP

# for metadata
TOPIC_RULES = {
    "products": {"investments", "deposits", "insurance", "cards", "loans"},
    "services": {"secure-banking-ways"},
    "faq": {"help-and-support"},
}

# chunking strategy
@dataclass
class Chunk:
    text: str
    metadata: dict[str, str | int]

embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return embedder.encode(sentences=input)

fn = MyEmbeddingFunction()

# get topic and put into metadata
def infer_topic_from_filename(filename: str) -> str:
    lower = filename.lower()
    for topic, keywords in TOPIC_RULES.items():
        if any(keyword in lower for keyword in keywords):
            return topic
    return "other"

# splits into chunks with headings for each section + overlap
def split_markdown_sections(markdown: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in markdown.splitlines():
        # detect headings using regex
        heading_match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if heading_match:
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
                current_lines = []
            current_heading = heading_match.group(2).strip()
            continue
        current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))
    # for last section , so we dont lose final chunk
    # note that if the markdown has no headings, it will return entire docu as content.
    if not sections:
        sections = [("", markdown.strip())]
    return sections

# split into word chunks after section chunking using sliding window
# NOTE: Tune max_words to find sweet spot for our RAG
def word_chunks(text: str, max_words: int, overlap_words: int) -> Iterator[str]:
    words = text.split()
    if not words:
        return
    # sliding window to chunk words and overlap, so semantic meaning is not lost
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        yield " ".join(chunk_words)
        if end == len(words):
            break
        start = max(0, end - overlap_words)

# intake the markdown and use the above section and word chunking
def build_chunks(
    markdown_text: str,
    base_metadata: dict[str, str | int],
    max_words: int,
    overlap_words: int,
) -> Iterable[Chunk]:
    for section_index, (heading, body) in enumerate(
        split_markdown_sections(markdown_text)
    ):
        text = body.strip()
        if not text:
            continue
        for chunk_index, chunk_text in enumerate(
            word_chunks(text, max_words=max_words, overlap_words=overlap_words)
        ):
            # now append meta data for heading, section, and chunk indices
            metadata = dict(base_metadata)
            metadata["section"] = heading or "root"
            metadata["section_index"] = section_index
            metadata["chunk_index"] = chunk_index
            # output chunk class
            yield Chunk(text=chunk_text, metadata=metadata)


def load_markdown_files(md_dir: Path) -> list[Path]:
    if not md_dir.exists():
        return []
    return sorted(md_dir.glob("*.md"))

# metadata builder
def build_metadata(md_path: Path) -> dict[str, str | int]:
    scraped_at = datetime.fromtimestamp(md_path.stat().st_mtime, tz=timezone.utc)
    return {
        "source_file": md_path.name,
        "source_path": str(md_path),
        "scraped_at": scraped_at.isoformat(),
        "topic": infer_topic_from_filename(md_path.name),
    }

# upsert into chromadb with chunked text
def upsert_chromadb(
    chunks: list[Chunk],
    persist_dir: Path,
    collection_name: str,
) -> None:
    
    client = chromadb.PersistentClient(path=str(persist_dir))
    # note: if want to use diff embedding model, need to delete existing collection first to prevent conflict.
    # client.delete_collection("poc_collection")
    collection = client.get_or_create_collection(name=collection_name, embedding_function=fn)
    ids = [f"{chunk.metadata['source_file']}::{i}" for i, chunk in enumerate(chunks)]
    documents = [chunk.text for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Upserted {len(chunks)} chunks into Chroma collection '{collection_name}'.")
    print(collection.peek())
    print(collection.count())


def main() -> None:

    base_dir = Path(__file__).resolve().parent
    md_dir = (base_dir / "processed_markdown_files").resolve()
    md_files = load_markdown_files(md_dir)
    if not md_files:
        print(f"No markdown files found in {md_dir}")
        return

    all_chunks: list[Chunk] = []
    for md_path in md_files:
        markdown_text = md_path.read_text(encoding="utf-8", errors="ignore")
        base_metadata = build_metadata(md_path)
        for chunk in build_chunks(
            markdown_text,
            base_metadata,
            max_words= MAX_WORDS,
            overlap_words= WORD_OVERLAP,
        ):
            all_chunks.append(chunk)

    chroma_dir = (base_dir / "../chromadb/data").resolve()
    upsert_chromadb(all_chunks, chroma_dir, "poc_collection")


if __name__ == "__main__":
    main()
