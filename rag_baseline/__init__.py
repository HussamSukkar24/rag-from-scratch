"""Deterministic stdlib baseline. No model integration is implied."""

from .core import (
    Document, Chunk, Hit, chunk_document, embed_texts, encode_query, retrieve,
    assemble_context, retrieval_metrics, answer_baseline, validate_citations,
)

__all__ = [
    "Document", "Chunk", "Hit", "chunk_document", "embed_texts", "encode_query",
    "retrieve", "assemble_context", "retrieval_metrics", "answer_baseline",
    "validate_citations",
]
