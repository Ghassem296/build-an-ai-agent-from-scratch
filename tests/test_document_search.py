import pytest

from tools.document_search import chunk_text


def test_chunk_text_returns_empty_list_for_empty_text():
    assert chunk_text("") == []


def test_chunk_text_splits_text_with_overlap():
    text = "".join(chr(0x400 + i) for i in range(1200))

    chunks = chunk_text(text, chunk_size=500, overlap=80)

    assert len(chunks) == 3
    assert [len(chunk) for chunk in chunks] == [500, 500, 360]
    assert chunks[0][-80:] == chunks[1][:80]
    assert chunks[1][-80:] == chunks[2][:80]


def test_chunk_text_rejects_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=0)


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=10, overlap=10)



from unittest.mock import patch

import numpy as np

from tools.document_search import search_documents


def test_search_documents_returns_expected_fields():
    fake_chunks = [
        {
            "id": "doc1",
            "chunk_id": "doc1_chunk0",
            "text": "How to reset a password",
        },
        {
            "id": "doc2",
            "chunk_id": "doc2_chunk0",
            "text": "How to update billing details",
        },
    ]

    fake_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    fake_model = type(
        "FakeModel",
        (),
        {
            "encode": lambda self, *args, **kwargs: np.array(
                [1.0, 0.0],
                dtype=np.float32,
            )
        },
    )()

    with patch(
        "tools.document_search._load_index",
        return_value=(fake_model, fake_chunks, fake_embeddings),
    ):
        results = search_documents("reset password", top_k=2)

    assert len(results) == 2
    assert results[0]["id"] == "doc1"
    assert results[0]["text"] == "How to reset a password"
    assert results[0]["score"] == 1.0
    assert set(results[0].keys()) == {"id", "text", "score"}
    assert results[0]["score"] >= results[1]["score"]
