from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer


# This query uses vector similarity search (ANN) over embeddings.
# Use the exact same embedding model you used when creating the collection.
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
COLLECTION_NAME = "poc_collection"
PERSIST_DIR = (Path(__file__).resolve().parent / "data").resolve()

# define embedding function again with init this time.
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self) -> None:
        self._model = SentenceTransformer(EMBEDDING_MODEL)

    def __call__(self, input: Documents) -> Embeddings:
        return self._model.encode(sentences=input)

#
def main() -> None:
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=MyEmbeddingFunction(),
    )

    query_text = "How do I dispute a card transaction?"
    results = collection.query(
        query_texts=[query_text],
        n_results=10,
        include=["documents", "metadatas", "distances"],
    )

    print(f"Query: {query_text}\n")
    for rank, (doc, meta, dist) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        print(f"Rank {rank} | distance={dist:.4f} | topic={meta.get('topic')}")
        print(f"Source: {meta.get('source_file')}")
        print(doc[:400])
        print("-" * 80)


if __name__ == "__main__":
    main()
