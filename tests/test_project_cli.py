"""CLI orchestration oracles only; fake services do not verify cloud models."""

from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from rag_baseline import project_cli, persistent, semantic, grounded
from rag_baseline.core import Chunk, Hit


PERSIST_SIGNATURE = inspect.signature(persistent.persist_index)
SOURCE_HASH_POLICY = "sha256 of supplied chunk records per document"


class Encoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return [[1., 0.] for _ in texts]


class ProjectCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def patches(self, stack, name, target_module, replacement):
        # Support either explicit module calls or imported aliases, without
        # imposing a new private helper interface on the implementation.
        stack.enter_context(patch.object(target_module, name, replacement))
        stack.enter_context(patch.object(project_cli, name, replacement, create=True))

    def invoke(self, argv):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = project_cli.main(argv)
        self.assertIn(result, (None, 0))
        return json.loads(stdout.getvalue())

    def assert_rejected(self, argv):
        try:
            result = project_cli.main(argv)
        except SystemExit as error:
            self.assertNotEqual(error.code, 0)
        except (ValueError, OSError):
            pass
        else:
            self.assertIsInstance(result, int)
            self.assertNotEqual(result, 0)

    def test_invalid_documents_and_chunk_configuration_fail_before_encoder_load(self):
        invalid = [dict(documents=[]), dict(documents=[dict(doc_id="", text="fact")]),
                   dict(documents=[dict(doc_id="d", text="")]),
                   dict(documents=[dict(doc_id="d", text="one"), dict(doc_id="d", text="two")]),
                   dict(documents=[dict(doc_id=7, text="fact")]),
                   dict(documents=[dict(doc_id="d", text=7)]),
                   dict(documents=[dict(doc_id="d", text="fact", source=7)]),
                   dict(documents=[dict(doc_id="d")])]
        documents = self.root / "documents.json"
        load = Mock(side_effect=AssertionError("Invalid input must precede model loading"))
        with ExitStack() as stack:
            self.patches(stack, "load_encoder", semantic, load)
            for payload in invalid:
                with self.subTest(payload=payload):
                    documents.write_text(json.dumps(payload), encoding="utf-8")
                    self.assert_rejected(["index", "--documents", str(documents),
                                          "--index", str(self.root / "new-index")])
            documents.write_text(json.dumps(dict(documents=[dict(doc_id="d", text="fact")])))
            for flags in [["--size", "0"], ["--size", "4", "--overlap", "4"], ["--overlap", "-1"]]:
                with self.subTest(flags=flags):
                    self.assert_rejected(["index", "--documents", str(documents),
                                          "--index", str(self.root / "new-index")] + flags)
        load.assert_not_called()

    def test_index_accepts_unlabelled_long_text_and_preserves_chunk_spans(self):
        text = "x" * 205 + "FIN"
        documents = self.root / "documents.json"
        payload = dict(documents=[dict(doc_id="long", text=text, source="original:long")])
        documents.write_text(json.dumps(payload), encoding="utf-8")
        encoder = Encoder()
        persist = Mock(return_value={"dimension": 2, "source_hash_policy": SOURCE_HASH_POLICY})
        with ExitStack() as stack:
            self.patches(stack, "load_encoder", semantic, Mock(return_value=encoder))
            self.patches(stack, "persist_index", persistent, persist)
            self.invoke(["index", "--documents", str(documents), "--index", str(self.root / "index"),
                         "--size", "80", "--overlap", "10"])
        persist.assert_called_once()
        bound = PERSIST_SIGNATURE.bind(*persist.call_args.args, **persist.call_args.kwargs).arguments
        chunks = bound["chunks"]
        self.assertEqual([(chunk.doc_id, chunk.start, chunk.end) for chunk in chunks],
                         [("long", 0, 80), ("long", 70, 150), ("long", 140, 208)])
        self.assertEqual([chunk.text for chunk in chunks], [text[0:80], text[70:150], text[140:208]])
        self.assertEqual(encoder.calls, [[text[0:80], text[70:150], text[140:208]]])
        self.assertEqual(bound["vectors"], [[1., 0.]] * 3)
        self.assertEqual(bound["model_id"], semantic.MODEL_ID)
        self.assertEqual(bound["model_revision"], semantic.MODEL_REVISION)

    def test_query_existing_index_encodes_only_question_and_keeps_langchain_selection(self):
        chunks = [Chunk("a", "d1", 10, 17, "Key 47."), Chunk("b", "d2", 0, 7, "Key 82.")]
        manifest = dict(chunks=[asdict(chunk) for chunk in chunks], dimension=2,
                        model_id=semantic.MODEL_ID, model_revision=semantic.MODEL_REVISION,
                        source_hash_policy=SOURCE_HASH_POLICY)
        index = self.root / "index"
        index.mkdir()
        encoder = Encoder()
        # LC chooses b despite a/b score ties: scores must map by ID, not position.
        from langchain_core.documents import Document
        selected = Document(page_content="Key 82.", metadata=dict(chunk_id="b", doc_id="d2", start=0, end=7))
        def qa(**kwargs):
            start = kwargs["context"].index("82")
            return dict(answer="82", start=start, end=start + 2, score=.9)
        with ExitStack() as stack:
            self.patches(stack, "_load", persistent, Mock(return_value=(None, None, manifest)))
            self.patches(stack, "load_encoder", semantic, Mock(return_value=encoder))
            self.patches(stack, "load_qa", grounded, Mock(return_value=qa))
            self.patches(stack, "retrieve_index", persistent,
                         Mock(return_value=[Hit("a", .8, 1), Hit("b", .8, 2)]))
            self.patches(stack, "langchain_retrieve", persistent, Mock(return_value=[selected]))
            persist = Mock(side_effect=AssertionError("Query must not create an index"))
            self.patches(stack, "persist_index", persistent, persist)
            output = self.invoke(["query", "--index", str(index), "--question", "Which key?", "--k", "1"])
        self.assertTrue(encoder.calls)
        self.assertTrue(all(texts == ["Which key?"] for texts in encoder.calls))
        persist.assert_not_called()
        self.assertEqual(output["question"], "Which key?")
        self.assertEqual(output["answer"]["text"], "82")
        self.assertEqual(output["answer"]["cited_chunk_ids"], ["b"])
        evidence = output["evidence"]
        self.assertEqual([(row["chunk_id"], row["doc_id"], row["start"], row["end"],
                           row["text"], row["score"]) for row in evidence],
                         [("b", "d2", 0, 7, "Key 82.", .8)])

    def test_inspect_unknown_id_rejected_without_model_loading(self):
        chunk = Chunk("a", "d1", 10, 17, "Key 47.")
        manifest = dict(chunks=[asdict(chunk)], dimension=2, model_id=semantic.MODEL_ID,
                        model_revision=semantic.MODEL_REVISION, source_hash_policy=SOURCE_HASH_POLICY)
        index = self.root / "index"
        index.mkdir()
        with ExitStack() as stack:
            self.patches(stack, "_load", persistent, Mock(return_value=(None, None, manifest)))
            load = Mock(side_effect=AssertionError("Inspect must not load models"))
            self.patches(stack, "load_encoder", semantic, load)
            self.patches(stack, "load_qa", grounded, load)
            self.assert_rejected(["inspect", "--index", str(index), "--chunk-id", "missing"])
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
