from pathlib import Path

from src.data_processing.clause_processor import (
    load_ocr_text,
    extract_clauses,
)

from src.retrieval.vector_store import ClauseVectorStore

OCR_PATH = Path("data/processed/ocr_sample.txt")


def main():

    print("Contract Semantic Search Test")

    text = load_ocr_text(OCR_PATH)
    clauses = extract_clauses(text)
    print(f"\nClauses extracted: {len(clauses)}")

    if not clauses:
        raise ValueError("No clauses extracted.")
    store = ClauseVectorStore()
    store.build(clauses)

    # Test semantic search
    query = "Either party may terminate the agreement " "by providing written notice."

    print("\n Semantic search query:")
    print(query)

    results = store.search(
        query=query,
        top_k=5,
    )

    print("\n Top semantic matches:")

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(f"Result: {index}")
        print(f"Clause: {result['clause_number']}")
        print(f"Title: {result['title']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Text: {result['text'][:500]}")

    print("\n Semantic search test completed successfully.")


if __name__ == "__main__":
    main()
