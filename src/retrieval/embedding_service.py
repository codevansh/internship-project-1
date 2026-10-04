from typing import List
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def load_embedding_model():
    """
    Load the sentence-transformer embedding model.

    The model is loaded once and reused for subsequent
    embedding operations.
    """

    global _model

    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        print("Embedding model loaded successfully.")

    return _model


def generate_embeddings(texts: List[str]):
    """
    Generate vector embeddings for a list of texts.

    Args:
        texts: List of clause/query strings.

    Returns:
        Embeddings as a numpy array.
    """

    if not texts:
        raise ValueError("No texts provided for embedding generation.")

    model = load_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings


def generate_embedding(text: str):
    """
    Generate a single embedding vector.
    """

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")
    embeddings = generate_embeddings([text])

    return embeddings[0]
