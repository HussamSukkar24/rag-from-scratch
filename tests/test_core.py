"""Independent baseline acceptance fixtures; execute in managed Colab only."""

import math
import unittest

from rag_baseline.core import (
    Document,
    chunk_document,
    embed_texts,
    encode_query,
    retrieval_metrics,
    retrieve,
)


class ChunkTests(unittest.TestCase):
    def test_exact_spans_and_no_redundant_terminal_chunk(self):
        chunks = chunk_document(Document("d1", "abcdefghij"), 4, 1)
        self.assertEqual(
            [(c.doc_id, c.start, c.end, c.text) for c in chunks],
            [("d1", 0, 4, "abcd"), ("d1", 3, 7, "defg"),
             ("d1", 6, 10, "ghij")],
        )
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(set(ids)), 3)
        self.assertEqual(ids, [c.chunk_id for c in
                              chunk_document(Document("d1", "abcdefghij"), 4, 1)])

    def test_unicode_newlines_use_character_offsets(self):
        doc = Document("unicode", "A😀\né文Z")
        chunks = chunk_document(doc, 3, 1)
        self.assertEqual([(c.start, c.end, c.text) for c in chunks],
                         [(0, 3, "A😀\n"), (2, 5, "\né文"), (4, 6, "文Z")])

    def test_empty_short_and_nonoverlapping_documents(self):
        self.assertEqual(chunk_document(Document("empty", ""), 4, 1), [])
        short = chunk_document(Document("short", "hi"), 4, 1)
        self.assertEqual([(c.start, c.end, c.text) for c in short], [(0, 2, "hi")])
        chunks = chunk_document(Document("plain", "abcdef"), 3, 0)
        self.assertEqual([c.text for c in chunks], ["abc", "def"])

    def test_invalid_configuration(self):
        for size, overlap in [(0, 0), (-1, 0), (4, -1), (4, 4), (4, 5)]:
            with self.subTest(size=size, overlap=overlap):
                with self.assertRaises(ValueError):
                    chunk_document(Document("d", "abc"), size, overlap)


class EmbeddingTests(unittest.TestCase):
    def test_lowercase_normalized_counts_and_query_alignment(self):
        vocabulary, vectors = embed_texts(["Red red BLUE", "blue green"])
        self.assertEqual(set(vocabulary), {"red", "blue", "green"})
        self.assertEqual(len(vocabulary), 3)
        self.assertEqual(len(vectors), 2)
        self.assertTrue(all(len(v) == 3 for v in vectors))
        red = encode_query("red", vocabulary)
        blue = encode_query("BLUE", vocabulary)
        green = encode_query("green", vocabulary)
        for v in [*vectors, red, blue, green]:
            self.assertTrue(all(math.isfinite(x) for x in v))
            self.assertAlmostEqual(sum(x * x for x in v), 1.0)
        dot = lambda a, b: sum(x * y for x, y in zip(a, b))
        self.assertAlmostEqual(dot(red, blue), 0.0)
        self.assertAlmostEqual(dot(vectors[0], red), 2 / math.sqrt(5))
        self.assertAlmostEqual(dot(vectors[0], blue), 1 / math.sqrt(5))
        self.assertAlmostEqual(dot(vectors[0], green), 0.0)
        self.assertAlmostEqual(dot(vectors[1], blue), 1 / math.sqrt(2))
        self.assertAlmostEqual(dot(vectors[1], green), 1 / math.sqrt(2))


class RetrievalTests(unittest.TestCase):
    def test_cosine_order_scores_ranks_and_id_ties(self):
        hits = retrieve([1, 0, 0], ["c3", "c2", "c1"],
                        [[0, 0, 1], [0, 2, 0], [3, 0, 0]], 3)
        self.assertEqual([h.chunk_id for h in hits], ["c1", "c2", "c3"])
        self.assertEqual([h.rank for h in hits], [1, 2, 3])
        for hit, expected in zip(hits, [1.0, 0.0, 0.0]):
            self.assertAlmostEqual(hit.score, expected)
        top = retrieve([1, 0, 0], ["c3", "c2", "c1"],
                       [[0, 0, 1], [0, 1, 0], [1, 0, 0]], 2)
        self.assertEqual([h.chunk_id for h in top], ["c1", "c2"])

    def test_negative_cosine_is_not_clamped(self):
        hits = retrieve([1, 0], ["opposite", "orthogonal"], [[-2, 0], [0, 1]], 2)
        self.assertEqual([h.chunk_id for h in hits], ["orthogonal", "opposite"])
        self.assertAlmostEqual(hits[1].score, -1.0)

    def test_k_boundaries_and_empty_index(self):
        self.assertEqual(retrieve([1, 0], ["a"], [[1, 0]], 0), [])
        self.assertEqual(len(retrieve([1, 0], ["a"], [[1, 0]], 8)), 1)
        self.assertEqual(retrieve([1, 0], [], [], 3), [])
        with self.assertRaises(ValueError):
            retrieve([1, 0], ["a"], [[1, 0]], -1)

    def test_invalid_vectors_and_misaligned_rows(self):
        cases = [
            ([0, 0], ["a"], [[1, 0]]),
            ([1, 0], ["a"], [[0, 0]]),
            ([1, 0], ["a"], [[1, 0, 0]]),
            ([1, 0], ["a", "b"], [[1, 0], [1]]),
            ([float("nan"), 0], ["a"], [[1, 0]]),
            ([1, 0], ["a"], [[float("inf"), 0]]),
            ([1, 0], ["a", "b"], [[1, 0]]),
        ]
        for query, ids, vectors in cases:
            with self.subTest(query=query, ids=ids, vectors=vectors):
                with self.assertRaises(ValueError):
                    retrieve(query, ids, vectors, 1)


class MetricTests(unittest.TestCase):
    def test_hand_calculated_fixture(self):
        first = retrieval_metrics(["c1", "c2"], {"c1", "c3"}, 2)
        second = retrieval_metrics(["c2", "c3"], {"c2"}, 2)
        for key, expected in {"recall": .5, "precision": .5,
                              "hit_rate": 1., "rr": 1.}.items():
            self.assertAlmostEqual(first[key], expected)
        for key, expected in {"recall": 1., "precision": .5,
                              "hit_rate": 1., "rr": 1.}.items():
            self.assertAlmostEqual(second[key], expected)
        self.assertAlmostEqual((first["recall"] + second["recall"]) / 2, .75)

    def test_rank_two_and_cutoff_miss(self):
        at_two = retrieval_metrics(["wrong", "right"], {"right"}, 2)
        self.assertEqual(at_two["rr"], .5)
        self.assertEqual(at_two["recall"], 1.)
        self.assertEqual(at_two["precision"], .5)
        at_one = retrieval_metrics(["wrong", "right"], {"right"}, 1)
        for key in ["recall", "precision", "hit_rate", "rr"]:
            self.assertEqual(at_one[key], 0.)


if __name__ == "__main__":
    unittest.main()
