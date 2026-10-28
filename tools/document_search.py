
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# نمونه اسناد پایگاه دانش
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

# مدل و بردارهای اسناد به‌صورت تنبل بارگذاری می‌شوند.
# یعنی تا زمان اولین جست‌وجو، مدل بارگذاری نخواهد شد.
_model: SentenceTransformer | None = None
_document_embeddings: np.ndarray | None = None


def _load_index() -> tuple[SentenceTransformer, np.ndarray]:
    """Load the embedding model and build the document vectors once."""
    global _model, _document_embeddings

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    if _document_embeddings is None:
        document_texts = [doc["text"] for doc in DOCUMENTS]

        _document_embeddings = _model.encode(
            document_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    return _model, _document_embeddings


def search_documents(
    query: str,
    top_k: int = 3,
) -> list[dict[str, str | float]]:
    """Find documents semantically similar to a query."""

    query = query.strip()

    if not query:
        raise ValueError("query cannot be empty")

    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10")

    model, document_embeddings = _load_index()

    # تبدیل پرسش به بردار
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # برای بردارهای نرمال‌شده، ضرب داخلی برابر cosine similarity است.
    scores = np.dot(document_embeddings, query_embedding)

    # مرتب‌سازی از بیشترین شباهت به کمترین
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results: list[dict[str, str | float]] = []

    for index in ranked_indices:
        document = DOCUMENTS[int(index)]

        results.append(
            {
                "id": document["id"],
                "text": document["text"],
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
