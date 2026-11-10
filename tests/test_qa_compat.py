"""Cloud compatibility regression for the removed Transformers QA task."""

import unittest


class QaCompatibilityTests(unittest.TestCase):
    def test_extractive_question_answering_task_is_registered(self):
        from transformers.pipelines import SUPPORTED_TASKS

        self.assertIn("question-answering", SUPPORTED_TASKS)


if __name__ == "__main__":
    unittest.main()
