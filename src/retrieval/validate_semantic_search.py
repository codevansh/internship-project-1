from src.retrieval.vector_store import ClauseVectorStore

TEST_QUERIES = [
    {
        "name": "Termination",
        "query": "Can either party terminate the agreement?",
    },
    {
        "name": "Confidentiality",
        "query": "Which clause contains confidentiality obligations?",
    },
    {
        "name": "Governing Law",
        "query": "What law governs this agreement?",
    },
    {
        "name": "Force Majeure",
        "query": "What happens if an event outside the parties' control prevents performance?",
    },
    {
        "name": "Insurance",
        "query": "What insurance obligations are required?",
    },
    {
        "name": "Assignment",
        "query": "Can the agreement be assigned to another party?",
    },
]


def run_validation():

    store = ClauseVectorStore()
    store.load()

    print("\nVector store loaded successfully.")
    print(f"Indexed clauses: {store.index.ntotal}")
    print(f"Metadata records: {len(store.metadata)}")

    if store.index.ntotal != len(store.metadata):
        raise ValueError("Vector index count does not match metadata count.")

    print("\nRunning semantic search tests...")

    for test_number, test in enumerate(TEST_QUERIES, start=1):
        print(f"TEST {test_number}: {test['name']}")
        print(f"\nQuery:")
        print(test["query"])

        results = store.search(
            query=test["query"],
            top_k=5,
        )

        if not results:
            print("\nNo results returned.")
            continue

        print("\nTop 5 semantic matches:")

        for rank, result in enumerate(results, start=1):
            print(f"Rank: {rank}")
            print(f"Clause: {result['clause_number']}")
            print(f"Title: {result['title']}")
            print(f"Similarity: {result['similarity']:.4f}")
            print(f"Text: {result['text'][:300]}")


if __name__ == "__main__":
    run_validation()
