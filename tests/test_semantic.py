"""Adapter contracts only. Fake vectors do not verify a real embedding model."""

import unittest

from rag_baseline.core import Chunk
from rag_baseline.semantic import encode_texts, semantic_retrieve


class Rows(list):
    """Minimal ndarray-like return, without importing or installing numpy."""

    def tolist(self):
        return list(self)


class FakeEncoder:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True,
               show_progress_bar=False):
        self.calls.append((list(texts), normalize_embeddings,
                           convert_to_numpy, show_progress_bar))
        return Rows(self.rows)


class SemanticAdapterTests(unittest.TestCase):
    def test_encode_preserves_text_vector_alignment_and_options(self):
        encoder = FakeEncoder([[1., 0.], [0., 1.]])
        vectors = encode_texts(encoder, ["north", "south"])
        self.assertIsInstance(vectors, list)
        self.assertEqual(vectors, [[1., 0.], [0., 1.]])
        self.assertEqual(encoder.calls, [(["north", "south"], True, True, False)])

    def test_encode_rejects_missing_or_extra_rows(self):
        for rows in [[[1., 0.]], [[1., 0.], [0., 1.], [1., 1.]]]:
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    encode_texts(FakeEncoder(rows), ["north", "south"])

    def test_encode_rejects_inconsistent_or_empty_dimensions(self):
        for rows in [[[1., 0.], [1.]], [[], []]]:
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    encode_texts(FakeEncoder(rows), ["north", "south"])

    def test_encode_rejects_nonfinite_vectors(self):
        for value in [float("nan"), float("inf"), -float("inf")]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    encode_texts(FakeEncoder([[value, 0.]]), ["north"])

    def test_cosine_ties_use_chunk_id_and_preserve_score_alignment(self):
        # Key by text so batching order cannot accidentally satisfy the oracle.
        class LookupEncoder:
            def encode(self, texts, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False):
                lookup = {"question": [1., 0.], "east": [0., 1.],
                          "west": [0., -1.], "answer": [1., 0.]}
                return Rows([lookup[text] for text in texts])

        chunks = [Chunk("z", "d1", 0, 4, "east"),
                  Chunk("b", "d2", 0, 4, "west"),
                  Chunk("a", "d3", 0, 6, "answer")]
        hits = semantic_retrieve(LookupEncoder(), "question", chunks, 3)
        self.assertEqual([hit.chunk_id for hit in hits], ["a", "b", "z"])
        self.assertEqual([hit.rank for hit in hits], [1, 2, 3])
        for hit, expected in zip(hits, [1., 0., 0.]):
            self.assertAlmostEqual(hit.score, expected)
        top = semantic_retrieve(LookupEncoder(), "question", chunks, 1)
        self.assertEqual([hit.chunk_id for hit in top], ["a"])

    def test_empty_query_returns_no_hits(self):
        chunks = [Chunk("c", "d", 0, 4, "text")]
        self.assertEqual(semantic_retrieve(FakeEncoder([]), "", chunks, 1), [])


if __name__ == "__main__":
    unittest.main()
