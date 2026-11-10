"""Cloud-only integration tests using real Chroma and LangChain storage paths.

Known vectors isolate persistence and adapter behavior from model quality.
No replacement Chroma client or LangChain retriever is used in these tests.
"""

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from langchain_core.documents import Document as LangChainDocument

from rag_baseline.core import Chunk
from rag_baseline.persistent import persist_index, retrieve_index, langchain_retrieve


MODEL = "fixture/orthogonal"
REVISION = "a" * 40


class QueryEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True,
               show_progress_bar=False):
        self.calls.append(list(texts))
        if list(texts) != ["question"]:
            raise AssertionError("Stored corpus must never be reembedded")
        return [[1., 0., 0.]]


class PersistentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "index"
        self.chunks = [Chunk("c3", "d3", 0, 5, "gamma"),
                       Chunk("c2", "d2", 4, 8, "beta"),
                       Chunk("c1", "d1", 2, 7, "alpha")]
        self.vectors = [[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]]

    def persist(self, chunks=None, vectors=None):
        persist_index(self.path, self.chunks if chunks is None else chunks,
                      self.vectors if vectors is None else vectors,
                      model_id=MODEL, model_revision=REVISION)

    def retrieve(self, vector, k, model=MODEL, revision=REVISION):
        return retrieve_index(self.path, vector, k,
                              model_id=model, model_revision=revision)

    def test_cosine_conversion_topk_and_deterministic_ties(self):
        self.persist()
        # Inserted in reverse ID order, so insertion order cannot break ties.
        for k, ids in [(1, ["c1"]), (2, ["c1", "c2"]), (8, ["c1", "c2", "c3"])]:
            with self.subTest(k=k):
                hits = self.retrieve([1., 0., 0.], k)
                self.assertEqual([hit.chunk_id for hit in hits], ids)
                self.assertEqual([hit.rank for hit in hits], list(range(1, len(ids) + 1)))
                for hit, score in zip(hits, [1., 0., 0.]):
                    self.assertAlmostEqual(hit.score, score, delta=1e-5)

    def test_reload_persists_vectors_text_metadata_and_langchain_query(self):
        self.persist()
        # Every API call opens from the persisted path, without a client handle.
        first = self.retrieve([1., 0., 0.], 1)
        again = retrieve_index(Path(str(self.path)), [1., 0., 0.], 1,
                               model_id=MODEL, model_revision=REVISION)
        self.assertEqual(first, again)
        encoder = QueryEncoder()
        documents = langchain_retrieve(Path(str(self.path)), "question", 1, encoder,
                                       model_id=MODEL, model_revision=REVISION)
        self.assertEqual(len(documents), 1)
        self.assertIsInstance(documents[0], LangChainDocument)
        self.assertEqual(documents[0].page_content, "alpha")
        for key, value in {"chunk_id": "c1", "doc_id": "d1", "start": 2, "end": 7}.items():
            self.assertEqual(documents[0].metadata[key], value)
        self.assertEqual(encoder.calls, [["question"]])

    def test_rejects_incompatible_reload_dimensions_model_and_revision(self):
        self.persist()
        for vector, model, revision in [([1., 0.], MODEL, REVISION),
                                        ([1., 0., 0.], "other/model", REVISION),
                                        ([1., 0., 0.], MODEL, "b" * 40)]:
            with self.subTest(model=model, revision=revision, vector=vector):
                with self.assertRaises(ValueError):
                    self.retrieve(vector, 1, model, revision)

    def test_zero_k_empty_query_and_negative_k(self):
        self.persist()
        self.assertEqual(self.retrieve([1., 0., 0.], 0), [])
        encoder = QueryEncoder()
        self.assertEqual(langchain_retrieve(self.path, "", 1, encoder,
                                           model_id=MODEL, model_revision=REVISION), [])
        self.assertEqual(langchain_retrieve(self.path, "question", 0, encoder,
                                           model_id=MODEL, model_revision=REVISION), [])
        self.assertEqual(encoder.calls, [])
        with self.assertRaises(ValueError):
            self.retrieve([1., 0., 0.], -1)

    def test_duplicate_chunk_ids_rejected(self):
        with self.assertRaises(ValueError):
            self.persist([self.chunks[0], self.chunks[0]], [[1., 0., 0.], [0., 1., 0.]])

    def test_malformed_or_nonfinite_stored_vectors_rejected(self):
        cases = [([[1., 0., 0.]], self.chunks),
                 ([[1., 0.], [0., 1., 0.], [1., 0., 0.]], self.chunks),
                 ([[0., 0., 0.]], self.chunks[:1]),
                 ([[float("nan"), 0., 1.]], self.chunks[:1]),
                 ([[float("inf"), 0., 1.]], self.chunks[:1])]
        for number, (vectors, chunks) in enumerate(cases):
            with self.subTest(vectors=vectors):
                with self.assertRaises(ValueError):
                    persist_index(Path(self.temporary.name) / f"invalid-{number}", chunks, vectors,
                                  model_id=MODEL, model_revision=REVISION)

    def test_nonfinite_and_zero_query_vectors_rejected(self):
        self.persist()
        for vector in [[0., 0., 0.], [float("nan"), 0., 1.], [float("inf"), 0., 1.]]:
            with self.subTest(vector=vector):
                with self.assertRaises(ValueError):
                    self.retrieve(vector, 1)

    def test_missing_completion_manifest_rejected(self):
        self.persist()
        (self.path / "manifest.json").unlink()
        with self.assertRaises(ValueError):
            self.retrieve([1., 0., 0.], 1)

    def test_manifest_checksum_source_and_chunk_tamper_rejected(self):
        self.persist()
        path = self.path / "manifest.json"
        original = path.read_text(encoding="utf-8")
        # Recompute the public checksum for the latter two cases so a simple
        # checksum guard alone cannot satisfy source/provenance validation.
        for mutation in ["checksum", "source", "chunk"]:
            with self.subTest(mutation=mutation):
                envelope = json.loads(original)
                manifest = envelope["manifest"]
                if mutation == "checksum":
                    envelope["sha256"] = "0" * 64
                else:
                    if mutation == "source":
                        manifest["source_hashes"]["d1"] = "0" * 64
                    else:
                        manifest["chunks"][0]["text"] = "delta"
                    canonical = json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                                           allow_nan=False).encode()
                    envelope["sha256"] = hashlib.sha256(canonical).hexdigest()
                path.write_text(json.dumps(envelope), encoding="utf-8")
                try:
                    with self.assertRaises(ValueError):
                        self.retrieve([1., 0., 0.], 1)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_non_axis_vectors_survive_writer_exit_and_reload(self):
        from dataclasses import asdict

        # Hand-calculated unit vectors: match, orthogonal, and opposite.
        vectors = [[-.6, -.8, 0.], [-.8, .6, 0.], [.6, .8, 0.]]
        request = dict(path=str(self.path.resolve()),
                       chunks=[asdict(chunk) for chunk in self.chunks], vectors=vectors,
                       model_id=MODEL, model_revision=REVISION)
        writer = (
            "import json,sys; from rag_baseline.core import Chunk; "
            "from rag_baseline.persistent import persist_index; data=json.load(sys.stdin); "
            "data['chunks']=[Chunk(**record) for record in data['chunks']]; "
            "persist_index(**data)"
        )
        # Run from the package root so subprocess imports the reviewed checkout.
        # subprocess.run waits for writer termination before any reader opens.
        result = subprocess.run([sys.executable, "-c", writer], input=json.dumps(request),
                                cwd=Path(__file__).resolve().parents[1], text=True,
                                capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        hits = self.retrieve([.6, .8, 0.], 3)
        self.assertEqual([hit.chunk_id for hit in hits], ["c1", "c2", "c3"])
        for hit, expected in zip(hits, [1., 0., -1.]):
            self.assertAlmostEqual(hit.score, expected, delta=1e-5)

        class NonAxisEncoder(QueryEncoder):
            def encode(self, texts, **kwargs):
                super().encode(texts, **kwargs)
                return [[.6, .8, 0.]]

        documents = langchain_retrieve(self.path, "question", 3, NonAxisEncoder(),
                                       model_id=MODEL, model_revision=REVISION)
        self.assertEqual([doc.metadata["chunk_id"] for doc in documents],
                         ["c1", "c2", "c3"])
        expected_chunks = {chunk.chunk_id: chunk for chunk in self.chunks}
        for document in documents:
            chunk = expected_chunks[document.metadata["chunk_id"]]
            self.assertEqual(document.page_content, chunk.text)
            self.assertEqual(document.metadata,
                             dict(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id,
                                  start=chunk.start, end=chunk.end))

    def test_actual_stored_vector_corruption_rejected(self):
        import chromadb
        from chromadb.config import Settings

        self.persist()
        manifest_bytes = (self.path / "manifest.json").read_bytes()
        client = chromadb.PersistentClient(path=str(self.path),
                                          settings=Settings(anonymized_telemetry=False))
        collection = client.get_collection("rag_chunks", embedding_function=None)
        # c1 originally points along x. Replace with a valid but different unit
        # vector so finiteness and dimension validation cannot mask corruption.
        collection.update(ids=["c1"], embeddings=[[0., 1., 0.]])
        self.assertEqual((self.path / "manifest.json").read_bytes(), manifest_bytes)
        with self.assertRaises(ValueError):
            self.retrieve([1., 0., 0.], 1)

    def test_stored_rotation_accepts_within_tolerance_and_rejects_outside(self):
        import chromadb
        from chromadb.config import Settings

        self.persist()
        manifest = (self.path / "manifest.json").read_bytes()
        client = chromadb.PersistentClient(path=str(self.path),
                                          settings=Settings(anonymized_telemetry=False))
        collection = client.get_collection("rag_chunks", embedding_function=None)
        # This policy deliberately accepts tiny coordinate changes. It does not
        # establish exact bit identity or cryptographic integrity of stored rows.
        angle = 5e-7
        collection.update(ids=["c1"], embeddings=[[math.cos(angle), math.sin(angle), 0.]])
        actual = collection.get(ids=["c1"], include=["embeddings"])["embeddings"][0]
        self.assertGreater(abs(float(actual[1])), 0.)
        self.assertLess(abs(float(actual[1])), 1e-6)
        hits = self.retrieve([1., 0., 0.], 1)
        self.assertEqual([hit.chunk_id for hit in hits], ["c1"])
        self.assertAlmostEqual(hits[0].score, 1., delta=1e-5)
        self.assertEqual((self.path / "manifest.json").read_bytes(), manifest)

        angle = 2e-6
        collection.update(ids=["c1"], embeddings=[[math.cos(angle), math.sin(angle), 0.]])
        actual = collection.get(ids=["c1"], include=["embeddings"])["embeddings"][0]
        self.assertGreater(abs(float(actual[1])), 1e-6)
        self.assertEqual((self.path / "manifest.json").read_bytes(), manifest)
        with self.assertRaises(ValueError):
            self.retrieve([1., 0., 0.], 1)


if __name__ == "__main__":
    unittest.main()
