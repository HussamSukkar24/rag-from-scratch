"""Regression for CLI aggregation and artifact writing, without model loading."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rag_baseline import semantic_cli


class SemanticCliTests(unittest.TestCase):
    def test_cli_writes_hand_calculated_macro_metrics_and_excludes_no_gold(self):
        class Encoder:
            max_seq_length = 256

            def encode(self, texts, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False):
                vectors = {"apple": [1., 0.], "orange": [0., 1.],
                           "apple query": [0., 1.], "orange query": [0., 1.],
                           "salary": [1., 0.]}
                return [vectors[text] for text in texts]

        fixture = {
            "fixture_version": 1,
            "documents": [{"doc_id": "d1", "text": "apple"},
                          {"doc_id": "d2", "text": "orange"}],
            "queries": [
                {"query_id": "q1", "text": "apple query", "relevant_doc_ids": ["d1"]},
                {"query_id": "q2", "text": "orange query", "relevant_doc_ids": ["d2"]},
                {"query_id": "q3", "text": "salary", "relevant_doc_ids": []},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            fixture_path = Path(directory) / "fixture.json"
            output = Path(directory) / "manifest.json"
            fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
            stdout = io.StringIO()
            with patch.object(semantic_cli, "load_encoder", return_value=Encoder()) as load:
                with patch.object(semantic_cli, "_git", return_value=None):
                    with patch.object(semantic_cli, "distributions", return_value=[]):
                        with patch("sys.argv", ["semantic_cli", "--fixture", str(fixture_path),
                                                "--output", str(output)]):
                            with redirect_stdout(stdout):
                                semantic_cli.main()
            load.assert_called_once_with(semantic_cli.MODEL_ID, semantic_cli.MODEL_REVISION)
            self.assertTrue(output.is_file())
            manifest = json.loads(output.read_text(encoding="utf-8"))
            # Semantic gold ranks are 2 and 1; lexical gold ranks are both 1.
            self.assertEqual(manifest["aggregate"]["semantic"],
                             {"recall_at_1": .5, "recall_at_2": 1., "mrr": .75})
            self.assertEqual(manifest["aggregate"]["lexical"],
                             {"recall_at_1": 1., "recall_at_2": 1., "mrr": 1.})
            self.assertEqual(manifest["gold_query_count"], 2)
            self.assertEqual(manifest["no_gold_query_count"], 1)
            no_gold = next(row for row in manifest["results"] if row["query_id"] == "q3")
            for method in ["semantic", "lexical"]:
                metrics = no_gold[method]["metrics"]
                self.assertTrue(metrics["excluded"])
                for key in ["recall_at_1", "recall_at_2", "rr"]:
                    self.assertIsNone(metrics[key])
            # A semantic nearest neighbor for salary is not evidence of an answer.
            self.assertEqual(len(no_gold["semantic"]["hits"]), 2)
            receipt = json.loads(stdout.getvalue())
            self.assertEqual(receipt["output"], str(output))
            self.assertEqual(receipt["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
