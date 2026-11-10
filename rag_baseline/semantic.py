"""CPU sentence embeddings feeding the existing exact cosine retriever."""

import math
import re

from .core import retrieve


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# Official commit: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/commit/c9745ed1d9f207416be6d2e6f8de32d1f16199bf
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def load_encoder(model_id=MODEL_ID, revision=MODEL_REVISION):
    """Load only an immutable revision on CPU. Call in managed Colab only."""
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full immutable lowercase commit SHA")
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_id, revision=revision, device="cpu", trust_remote_code=False,
        model_kwargs={"use_safetensors": True},
    )


def encode_texts(encoder, texts):
    """Return finite, nonzero normalized rows with one consistent dimension."""
    if isinstance(texts, str):
        raise ValueError("texts must be a sequence of strings, not a single string")
    texts = list(texts)
    if not all(isinstance(text, str) for text in texts):
        raise ValueError("each text must be a string")
    if not texts:
        return []
    encoded = encoder.encode(
        texts, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    if len(encoded) != len(texts):
        raise ValueError("encoder returned a different number of rows than texts")
    vectors = []
    dimension = None
    for row in encoded:
        vector = [float(value) for value in row]
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("encoder must return finite nonempty vectors")
        if dimension is None:
            dimension = len(vector)
        if len(vector) != dimension:
            raise ValueError("encoder dimensions must be consistent")
        norm = math.hypot(*vector)
        if norm == 0 or not math.isfinite(norm):
            raise ValueError("encoder must return nonzero vectors with finite norms")
        vectors.append([value / norm for value in vector])
    return vectors


def semantic_retrieve(encoder, query, chunks, k):
    """Encode query and chunks together, then reuse core cosine and ID ties."""
    if not isinstance(k, int) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    chunks = list(chunks)
    if not query.strip() or not chunks or k == 0:
        return []
    vectors = encode_texts(encoder, [query] + [chunk.text for chunk in chunks])
    return retrieve(vectors[0], [chunk.chunk_id for chunk in chunks], vectors[1:], k)
