
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

DOCUMENTS = [
    {
        "id": "doc1",
        "text": (
            "To change your password, open Settings, "
            "select Security, and choose Change Password."
        ),
    },
    {
        "id": "doc2",
        "text": (
            "If you forget your password, click Forgot Password "
            "on the login page and follow the reset link sent to your email."
        ),
    },
    {
        "id": "doc3",
        "text": (
            "You can update your email address from Account Settings. "
            "You may need to verify the new email address."
        ),
    },
    {
        "id": "doc4",
        "text": (
            "For billing questions, open the Billing section "
            "to view invoices and payment history."
        ),
    },
]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping character-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunk = text[start : start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def _build_chunks() -> list[dict[str, str]]:
    """Split each document and retain its original ID."""

    chunks = []

    for document in DOCUMENTS:
        for index, text in enumerate(chunk_text(document["text"])):
            chunks.append(
                {
                    "id": document["id"],
                    "chunk_id": f"{document['id']}_chunk{index}",
                    "text": text,
                }
            )

    return chunks


_model: SentenceTransformer | None = None
_document_chunks: list[dict[str, str]] | None = None
_chunk_embeddings: np.ndarray | None = None


def _load_index() -> tuple[
    SentenceTransformer,
    list[dict[str, str]],
    np.ndarray,
]:
    """Load the model and build chunk embeddings once."""

    global _model, _document_chunks, _chunk_embeddings

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    if _document_chunks is None:
        _document_chunks = _build_chunks()

    if _chunk_embeddings is None:
        chunk_texts = [chunk["text"] for chunk in _document_chunks]

        _chunk_embeddings = _model.encode(
            chunk_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    return _model, _document_chunks, _chunk_embeddings


def search_documents(
    query: str,
    top_k: int = 3,
) -> list[dict[str, str | float]]:
    """Find semantically similar document chunks."""

    query = query.strip()

    if not query:
        raise ValueError("query cannot be empty")

    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10")

    model, chunks, embeddings = _load_index()

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    scores = np.dot(embeddings, query_embedding)
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results: list[dict[str, str | float]] = []

    for index in ranked_indices:
        chunk = chunks[int(index)]

        results.append(
            {
                "id": chunk["id"],
                "text": chunk["text"],
                "score": round(float(scores[index]), 4),
            }
        )

    return results


if __name__ == "__main__":
    question = "I cannot log in because I forgot my password"

    print(f"Question: {question}\n")

    for result in search_documents(question):
        print(
            f"[{result['score']:.4f}] "
            f"{result['id']}: {result['text']}"
        )
