"""Independent per-source consistency mechanics, without model quality claims."""

import unittest

from rag_baseline.core import Chunk, Hit
from rag_baseline.grounded import answer_question, validate_grounded_citations


class GroundedConflictTests(unittest.TestCase):
    def evidence(self, second="The Iris vault code is 73."):
        first = "The Iris vault code is 41."
        chunks = [Chunk("a", "d1", 0, len(first), first),
                  Chunk("b", "d2", 0, len(second), second)]
        return chunks, [Hit("a", .9, 1), Hit("b", .8, 2)]

    def backend(self, chunks, second_result="valid"):
        calls = []

        def qa(*, question, context, handle_impossible_answer, top_k=1):
            self.assertEqual(question, "What is the vault code?")
            self.assertIs(handle_impossible_answer, True)
            calls.append(context)
            first_present = chunks[0].text in context
            second_present = chunks[1].text in context
            # Combined-context answer remains the first code. Only independent
            # per-source answers expose disagreement for the conflict policy.
            if first_present:
                text, score = "41", .9
            elif second_present:
                text, score = "73", .9
                if second_result == "low":
                    score = .49
                elif second_result == "empty":
                    return dict(answer="", score=.9, start=0, end=0)
                elif second_result == "invalid":
                    return dict(answer="73", score=.9, start=-1, end=1)
                elif second_result == "exception":
                    raise RuntimeError("QA_SOURCE_FAILURE")
            else:
                self.fail("QA was given no selected evidence")
            start = context.index(text)
            return dict(answer=text, score=score, start=start, end=start + len(text))

        return qa, calls

    def test_different_valid_per_source_answers_abstain(self):
        chunks, hits = self.evidence()
        qa, calls = self.backend(chunks)
        answer = answer_question("What is the vault code?", hits, chunks, qa)
        self.assertTrue(answer["abstained"])
        self.assertEqual(answer["reason"], "conflicting_evidence")
        self.assertEqual(answer["citations"], [])
        self.assertEqual(answer["cited_chunk_ids"], [])
        for source in chunks:
            self.assertTrue(any(source.text in context and
                                all(other.text not in context for other in chunks if other != source)
                                for context in calls))

    def test_identical_answers_from_different_sources_remain_answered(self):
        chunks, hits = self.evidence("The west vault code is 41.")
        def qa(*, question, context, handle_impossible_answer, top_k=1):
            start = context.index("41")
            return dict(answer="41", score=.9, start=start, end=start + 2)
        answer = answer_question("What is the vault code?", hits, chunks, qa)
        self.assertFalse(answer["abstained"])
        self.assertEqual(answer["text"], "41")
        self.assertTrue(validate_grounded_citations(answer, hits, chunks))

    def test_low_empty_and_invalid_source_spans_do_not_create_conflict(self):
        for response in ["low", "empty", "invalid"]:
            with self.subTest(response=response):
                chunks, hits = self.evidence()
                qa, _ = self.backend(chunks, response)
                answer = answer_question("What is the vault code?", hits, chunks, qa)
                self.assertFalse(answer["abstained"])
                self.assertEqual(answer["text"], "41")
                self.assertTrue(validate_grounded_citations(answer, hits, chunks))

    def test_unretrieved_or_budget_skipped_sources_cannot_veto_answer(self):
        chunks, hits = self.evidence()
        qa, _ = self.backend(chunks)
        answer = answer_question("What is the vault code?", hits[:1], chunks, qa)
        self.assertFalse(answer["abstained"])
        # Budget exactly covers the first complete provenance block and body.
        budget = len(f"[a] d1 [0,{len(chunks[0].text)})\n{chunks[0].text}")
        qa, calls = self.backend(chunks)
        answer = answer_question("What is the vault code?", hits, chunks, qa, max_chars=budget)
        self.assertFalse(answer["abstained"])
        self.assertEqual(answer["text"], "41")
        self.assertTrue(all(chunks[1].text not in context for context in calls))

    def test_per_source_backend_exception_propagates(self):
        chunks, hits = self.evidence()
        qa, _ = self.backend(chunks, "exception")
        with self.assertRaisesRegex(RuntimeError, "QA_SOURCE_FAILURE"):
            answer_question("What is the vault code?", hits, chunks, qa)


if __name__ == "__main__":
    unittest.main()
