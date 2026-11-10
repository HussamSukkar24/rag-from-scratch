"""Grounded span mechanics only; fake QA does not establish semantic truth."""

import copy
import unittest

from rag_baseline.core import Chunk, Hit
from rag_baseline.grounded import answer_question, validate_grounded_citations


class GroundedTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [Chunk("a", "doc-a", 10, 17, "Key 47."),
                       Chunk("b", "doc-b", 0, 10, "Door blue.")]
        self.hits = [Hit("a", 1., 1)]
        self.calls = []

    def qa(self, *, question, context, handle_impossible_answer, top_k=1):
        self.calls.append((question, context, handle_impossible_answer))
        start = context.index("47")
        return dict(answer="47", start=start, end=start + 2, score=.9)

    def assert_abstained(self, answer):
        self.assertTrue(answer["abstained"])
        self.assertEqual(answer["cited_chunk_ids"], [])
        self.assertEqual(answer["citations"], [])
        self.assertIsInstance(answer["reason"], str)
        self.assertTrue(answer["reason"])

    def test_exact_extraction_maps_to_absolute_source_offsets(self):
        answer = answer_question("Which key?", self.hits, self.chunks, self.qa)
        self.assertFalse(answer["abstained"])
        self.assertEqual(answer["text"], "47")
        self.assertEqual(answer["cited_chunk_ids"], ["a"])
        self.assertEqual(answer["citations"], [dict(chunk_id="a", doc_id="doc-a",
                                                   start=14, end=16, quote="47")])
        self.assertTrue(validate_grounded_citations(answer, self.hits, self.chunks))
        question, context, impossible = self.calls[0]
        self.assertEqual(question, "Which key?")
        self.assertIs(impossible, True)
        self.assertIn("Key 47.", context)
        self.assertNotIn("Door blue.", context)

    def test_budget_preserves_only_whole_evidence_and_empty_budget_abstains(self):
        answer_question("key", self.hits, self.chunks, self.qa)
        budget = len(self.calls[0][1])
        accepted = answer_question("key", self.hits, self.chunks, self.qa, max_chars=budget)
        self.assertFalse(accepted["abstained"])
        for limit in [budget - 1, 0]:
            self.assert_abstained(answer_question("key", self.hits, self.chunks,
                                                 self.qa, max_chars=limit))

    def test_empty_query_or_no_evidence_abstains_without_qa(self):
        def forbidden(**kwargs):
            self.fail("QA must not run without a question and evidence")
        for query, hits in [("", self.hits), ("  ", self.hits), ("key", [])]:
            self.assert_abstained(answer_question(query, hits, self.chunks, forbidden))

    def test_low_nonfinite_or_empty_backend_answers_abstain(self):
        for score, text in [(.49, "47"), (float("nan"), "47"),
                            (float("inf"), "47"), (-float("inf"), "47"), (.9, "")]:
            with self.subTest(score=score, text=text):
                def backend(**kwargs):
                    context = kwargs["context"]
                    start = context.index("47")
                    return dict(answer=text, start=start, end=start + len(text), score=score)
                self.assert_abstained(answer_question("key", self.hits, self.chunks, backend))

    def test_score_equal_to_threshold_is_accepted(self):
        def backend(**kwargs):
            start = kwargs["context"].index("47")
            return dict(answer="47", start=start, end=start + 2, score=.5)
        self.assertFalse(answer_question("key", self.hits, self.chunks, backend)["abstained"])

    def test_invalid_offsets_and_wrong_answer_text_abstain(self):
        for start, end, text in [(-1, 1, "47"), (0, 10000, "47"), (5, 3, "47"),
                                 (1.5, 3.5, "47"), (0, 0, "47")]:
            with self.subTest(start=start, end=end):
                def backend(**kwargs):
                    return dict(answer=text, start=start, end=end, score=.9)
                self.assert_abstained(answer_question("key", self.hits, self.chunks, backend))
        def wrong_quote(**kwargs):
            start = kwargs["context"].index("47")
            return dict(answer="99", start=start, end=start + 2, score=.9)
        self.assert_abstained(answer_question("key", self.hits, self.chunks, wrong_quote))

    def test_header_and_cross_chunk_extractions_abstain(self):
        hits = self.hits + [Hit("b", .8, 2)]
        def header(**kwargs):
            return dict(answer=kwargs["context"][:1], start=0, end=1, score=.9)
        self.assert_abstained(answer_question("key", hits, self.chunks, header))
        def crossing(**kwargs):
            context = kwargs["context"]
            start, end = context.index("47"), context.index("Door") + 4
            return dict(answer=context[start:end], start=start, end=end, score=.9)
        self.assert_abstained(answer_question("key", hits, self.chunks, crossing))

    def ranked(self, extra_spans, chunks=None, hits=None):
        """Return a top answer of "fourteen" plus extra (text, score) spans."""
        chunks = chunks or [Chunk("l", "doc-l", 5, 46,"The archive lends maps for fourteen days.")]
        hits = hits or [Hit("l", 1., 1)]
        requested = []

        def backend(*, question, context, handle_impossible_answer, top_k=1):
            requested.append(top_k)
            spans = [("fourteen", .53)] + extra_spans if top_k > 1 else [("fourteen", .31)]
            results = []
            for text, score in spans:
                start = context.index(text)
                results.append(dict(answer=text, score=score, start=start, end=start + len(text)))
            return results if top_k > 1 else results[0]
        return answer_question("How many days?", hits, chunks, backend), hits, chunks, requested

    def test_containing_candidate_extends_the_answer_with_its_own_score(self):
        answer, hits, chunks, requested = self.ranked([("maps", .7), ("fourteen days", .5)])
        self.assertFalse(answer["abstained"])
        self.assertEqual(answer["text"], "fourteen days")
        self.assertEqual(answer["citations"][0]["start"], 5 + 27)
        self.assertEqual(answer["citations"][0]["end"], 5 + 40)
        self.assertTrue(validate_grounded_citations(answer, hits, chunks))
        self.assertGreater(requested[0], 1)

    def test_widest_validated_containing_candidate_wins(self):
        answer, *_ = self.ranked([("fourteen days", .6), ("maps for fourteen days", .55)])
        self.assertEqual(answer["text"], "maps for fourteen days")

    def test_low_or_non_containing_candidates_keep_the_top_answer(self):
        for extra in [[("fourteen days", .49)], [("maps", .9)], [("", .9)]]:
            with self.subTest(extra=extra):
                answer, *_ = self.ranked(extra)
                self.assertEqual(answer["text"], "fourteen")

    def test_candidate_in_another_chunk_is_ignored(self):
        chunks = [Chunk("l", "doc-l", 0, 23, "Maps go for fourteen.  "),
                  Chunk("m", "doc-m", 0, 23, "Books fourteen days ago")]
        hits = [Hit("l", 1., 1), Hit("m", .9, 2)]
        answer, *_ = self.ranked([("fourteen days", .6)], chunks, hits)
        self.assertEqual(answer["text"], "fourteen")
        self.assertEqual(answer["cited_chunk_ids"], ["l"])

    def test_empty_candidate_list_abstains(self):
        self.assert_abstained(answer_question(
            "key", self.hits, self.chunks, lambda **kwargs: []))

    def test_validator_rejects_invented_unretrieved_swapped_and_edited_citations(self):
        valid = answer_question("key", self.hits, self.chunks, self.qa)
        for field, value in [("chunk_id", "invented"), ("chunk_id", "b"),
                              ("doc_id", "doc-b"), ("start", 13), ("end", 17),
                              ("quote", "99")]:
            with self.subTest(field=field, value=value):
                invalid = copy.deepcopy(valid)
                invalid["citations"][0][field] = value
                self.assertFalse(validate_grounded_citations(invalid, self.hits, self.chunks))
        invalid = copy.deepcopy(valid)
        invalid["cited_chunk_ids"] = ["b"]
        self.assertFalse(validate_grounded_citations(invalid, self.hits, self.chunks))
        invalid = copy.deepcopy(valid)
        invalid["text"] = "99"
        self.assertFalse(validate_grounded_citations(invalid, self.hits, self.chunks))


if __name__ == "__main__":
    unittest.main()
