import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np

from src.retrieval.embedding_service import generate_embedding

INDEX_PATH = Path("data/vector_store/contract_clauses.index")
METADATA_PATH = Path("data/vector_store/contract_clauses.json")


class ClauseVectorStore:
    """
    Local vector store for contract clauses.

    FAISS stores the embedding vectors while a JSON file
    stores the corresponding clause metadata.
    """

    def __init__(
        self,
        index_path: Path = INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path

        self.index = None
        self.metadata: List[Dict] = []

    def build(self, clauses: List[Dict]):
        """
        Build a FAISS index from contract clauses.

        Each clause should contain:
            article
            clause_number
            title
            text
        """

        if not clauses:
            raise ValueError("No clauses provided to build vector store.")
        texts = [
            (
                f"Clause {clause['clause_number']}. "
                f"{clause['title']}. "
                f"{clause['text']}"
            )
            for clause in clauses
            if clause.get("text", "").strip()
        ]

        if not texts:
            raise ValueError("No valid clause text found.")

        from src.retrieval.embedding_service import generate_embeddings

        embeddings = generate_embeddings(texts)
        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.metadata = [clause for clause in clauses if clause.get("text", "").strip()]
        self.save()

        print("\nVector store built successfully.")
        print(f"Clauses indexed: {len(self.metadata)}")
        print(f"Embedding dimension: {dimension}")

    def save(self):
        """Save FAISS index and metadata to disk."""

        if self.index is None:
            raise ValueError("Vector index has not been built.")

        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(self.index_path),
        )

        with open(
            self.metadata_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.metadata,
                file,
                indent=4,
                ensure_ascii=False,
            )

        print(f"Vector index saved to: {self.index_path}")
        print(f"Metadata saved to: {self.metadata_path}")

    def load(self):
        """Load an existing FAISS index and metadata."""

        if not self.index_path.exists():
            raise FileNotFoundError(f"Vector index not found at: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Vector metadata not found at: {self.metadata_path}"
            )

        self.index = faiss.read_index(str(self.index_path))

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.metadata = json.load(file)

        return self

    def search(
        self,
        query: str,
        top_k: int = 5,
    ):
        """
        Perform semantic similarity search.
        Returns the top-k most similar clauses.
        """

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")
        if self.index is None:
            self.load()

        query_embedding = generate_embedding(query)
        query_vector = np.asarray(
            [query_embedding],
            dtype="float32",
        )

        scores, indices = self.index.search(
            query_vector,
            min(top_k, self.index.ntotal),
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            if index < 0:
                continue

            clause = self.metadata[index].copy()
            clause["similarity"] = float(score)
            results.append(clause)
        return results
