"""Whole-evidence boundaries and narrow lexical extraction, not entailment."""

import unittest

from rag_baseline.core import (
    Chunk, Hit, answer_baseline, assemble_context, retrieval_metrics,
    validate_citations,
)


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            Chunk("c1", "d1", 0, 27, "The library opens at 09:00."),
            Chunk("c2", "d2", 0, 27, "The museum closes at 18:00."),
            Chunk("c3", "d3", 0, 33, "The library is closed on Sundays."),
        ]
        self.hits = [Hit("c1", 1.0, 1), Hit("c2", .5, 2)]

    def test_context_exact_budget_includes_provenance(self):
        expected = "[c1] d1 [0,27)\nThe library opens at 09:00."
        self.assertEqual(assemble_context(self.hits[:1], self.chunks, len(expected)),
                         expected)
        self.assertEqual(assemble_context(self.hits[:1], self.chunks, len(expected) - 1),
                         "")
        self.assertEqual(assemble_context(self.hits, self.chunks, 0), "")

    def test_context_hit_order_separator_and_unretrieved_exclusion(self):
        expected = ("[c2] d2 [0,27)\nThe museum closes at 18:00.\n\n"
                    "[c1] d1 [0,27)\nThe library opens at 09:00.")
        self.assertEqual(assemble_context(self.hits[::-1], self.chunks, len(expected)),
                         expected)

    def test_context_skips_oversized_chunk_but_keeps_later_whole_chunk(self):
        chunks = [Chunk("big", "d", 0, 100, "x" * 100),
                  Chunk("s", "d", 0, 1, "Y")]
        hits = [Hit("big", 1., 1), Hit("s", .5, 2)]
        self.assertEqual(assemble_context(hits, chunks, 20), "[s] d [0,1)\nY")

    def test_context_rejects_unknown_duplicate_and_negative_budget(self):
        cases = [([Hit("missing", 1., 1)], self.chunks, 2000),
                 ([self.hits[0], self.hits[0]], self.chunks, 2000),
                 (self.hits, self.chunks + [self.chunks[0]], 2000),
                 (self.hits, self.chunks, -1)]
        for hits, chunks, budget in cases:
            with self.subTest(hits=hits, budget=budget):
                with self.assertRaises(ValueError):
                    assemble_context(hits, chunks, budget)

    def test_narrow_lexical_answer_is_exact_retrieved_chunk(self):
        # No stemming or semantic paraphrase is claimed by this baseline.
        answer = answer_baseline("library opens", self.hits, self.chunks)
        self.assertEqual(answer, {"text": "The library opens at 09:00.",
                                  "cited_chunk_ids": ["c1"], "abstained": False})
        self.assertTrue(validate_citations(answer, self.hits, self.chunks))

    def test_salary_empty_query_missing_hits_and_budget_abstain(self):
        cases = [("library salary", self.hits, 2000),
                 ("", self.hits, 2000),
                 ("library opens", [], 2000),
                 ("library opens", self.hits, 0)]
        for query, hits, budget in cases:
            with self.subTest(query=query, budget=budget):
                answer = answer_baseline(query, hits, self.chunks, max_chars=budget)
                self.assertTrue(answer["abstained"])
                self.assertEqual(answer["cited_chunk_ids"], [])
                self.assertTrue(validate_citations(answer, hits, self.chunks))

    def test_differing_matching_chunks_abstain(self):
        alternative = Chunk("c4", "d4", 0, 27, "The library opens at 10:00.")
        answer = answer_baseline("library opens", [self.hits[0], Hit("c4", .9, 2)],
                                 self.chunks + [alternative])
        self.assertTrue(answer["abstained"])
        self.assertEqual(answer["cited_chunk_ids"], [])

    def test_citations_reject_invented_unretrieved_swapped_and_edited_text(self):
        cases = [
            {"text": self.chunks[0].text, "cited_chunk_ids": ["invented"], "abstained": False},
            {"text": self.chunks[2].text, "cited_chunk_ids": ["c3"], "abstained": False},
            {"text": self.chunks[0].text, "cited_chunk_ids": ["c2"], "abstained": False},
            {"text": "The library opens at 10:00.", "cited_chunk_ids": ["c1"], "abstained": False},
            {"text": self.chunks[0].text, "cited_chunk_ids": [], "abstained": False},
            {"text": self.chunks[0].text, "cited_chunk_ids": ["c1", "c2"], "abstained": False},
            {"text": "Insufficient evidence.", "cited_chunk_ids": ["c1"], "abstained": True},
        ]
        for answer in cases:
            with self.subTest(answer=answer):
                self.assertFalse(validate_citations(answer, self.hits, self.chunks))


class MetricBoundaryTests(unittest.TestCase):
    def test_no_gold_is_explicitly_excluded_not_scored_as_zero(self):
        metrics = retrieval_metrics(["c1"], set(), 1)
        self.assertTrue(metrics["excluded"])
        for key in ["recall", "precision", "hit_rate", "rr"]:
            self.assertIsNone(metrics[key])

    def test_duplicate_ranked_ids_are_rejected_even_beyond_cutoff(self):
        with self.assertRaises(ValueError):
            retrieval_metrics(["c1", "c2", "c1"], {"c1"}, 1)

    def test_precision_uses_requested_k_and_empty_results_are_zero(self):
        metrics = retrieval_metrics(["c1"], {"c1"}, 4)
        self.assertEqual(metrics["precision"], .25)
        self.assertEqual(metrics["recall"], 1.)
        for ranked, k in [([], 2), (["c1"], 0)]:
            metrics = retrieval_metrics(ranked, {"c1"}, k)
            self.assertFalse(metrics["excluded"])
            for key in ["recall", "precision", "hit_rate", "rr"]:
                self.assertEqual(metrics[key], 0.)


if __name__ == "__main__":
    unittest.main()
