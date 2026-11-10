"""Colab only: persistent retrieval comparison, never answer generation.

python -m rag_baseline.persistent_cli --index /content/rag-index --output /content/persistent.json
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

from .cli import _git
from .core import Document, chunk_document, retrieve
from .persistent import retrieve_index, langchain_retrieve, _load, _vector, _vector_hash
from .semantic import MODEL_ID, MODEL_REVISION, load_encoder, encode_texts
from .semantic_cli import _scores


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=root / "fixtures/semantic.json")
    args = parser.parse_args()
    archive = args.output.with_suffix(".index.zip")
    persist_log = args.output.with_suffix(".persist.log")
    outputs = (args.output, archive, persist_log)
    if (args.index.exists() or not args.index.parent.is_dir()
            or any(path.exists() or not path.parent.is_dir() for path in outputs)
            or len(set(path.resolve() for path in outputs)) != len(outputs)):
        parser.error("index and output artifacts must be new paths with existing parents")
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    raw = args.fixture.read_bytes()
    fixture = json.loads(raw)
    docs = [Document(**item) for item in fixture["documents"]]
    doc_ids = {doc.doc_id for doc in docs}
    if (not docs or len(doc_ids) != len(docs)
            or any(not doc.text.strip() or len(doc.text) > 200 for doc in docs)):
        raise ValueError("fixture must have unique nonempty one-chunk documents")
    queries = fixture["queries"]
    if len({q["query_id"] for q in queries}) != len(queries):
        raise ValueError("query IDs must be unique")
    if any(not set(q["relevant_doc_ids"]).issubset(doc_ids) for q in queries):
        raise ValueError("gold must reference existing documents")
    chunks = [chunk for doc in docs for chunk in chunk_document(doc, 200, 0)]
    nonblank = [q for q in queries if q["text"].strip()]
    encoder = load_encoder(MODEL_ID, MODEL_REVISION)
    vectors = encode_texts(encoder, [c.text for c in chunks] + [q["text"] for q in nonblank])
    corpus = vectors[:len(chunks)]
    query_vectors = {q["text"]: row for q, row in zip(nonblank, vectors[len(chunks):])}

    # A real model produced these vectors above. Reuse them in the LangChain
    # callback so all paths compare identical embeddings without extra inference.
    class CachedQueries:
        def encode(self, texts, **kwargs):
            if any(text not in query_vectors for text in texts):
                raise ValueError("query cache refuses unknown text or corpus reembedding")
            return [query_vectors[text] for text in texts]

    request = dict(path=str(args.index.resolve()), chunks=[asdict(c) for c in chunks], vectors=corpus,
                   model_id=MODEL_ID, model_revision=MODEL_REVISION)
    # The writer exits before the first reader is created. This proves disk
    # reopen across processes rather than Chroma's in-process client cache.
    writer = (
        "import json,sys; from rag_baseline.core import Chunk; "
        "from rag_baseline.persistent import persist_index; p=json.load(sys.stdin); "
        "p['chunks']=[Chunk(**c) for c in p['chunks']]; persist_index(**p)"
    )
    with persist_log.open("x", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-c", writer], input=json.dumps(request),
                       text=True, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=root)
    # Snapshot only after writer termination and before any reader is opened.
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(args.index.rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(args.index))
    saved_manifest = json.loads((args.index / "manifest.json").read_text(encoding="utf-8"))
    ids = [c.chunk_id for c in chunks]
    expected = {c.chunk_id: c for c in chunks}
    _, collection, validated_manifest = _load(args.index, MODEL_ID, MODEL_REVISION)
    stored = collection.get(ids=ids, include=["embeddings"])
    if len(stored["ids"]) != len(ids) or set(stored["ids"]) != set(ids):
        raise ValueError("readback diagnostic IDs differ from validated index")
    readback_chunks = []
    for chunk_id, values in zip(stored["ids"], stored["embeddings"], strict=True):
        row = _vector(values, validated_manifest["dimension"])
        reference = validated_manifest["reference_vectors"][chunk_id]
        reference_hash = validated_manifest["vector_hashes"][chunk_id]
        stored_hash = _vector_hash(row)
        readback_chunks.append(dict(
            chunk_id=chunk_id,
            max_abs_error=max(abs(actual - wanted) for actual, wanted in zip(row, reference, strict=True)),
            reference_float32_sha256=reference_hash, stored_float32_sha256=stored_hash,
            bit_hash_match=reference_hash == stored_hash,
            stored_norm=math.hypot(*row), reference_norm=math.hypot(*reference),
        ))
    readback_chunks.sort(key=lambda item: item["chunk_id"])
    readback_diagnostics = dict(
        scope="current readback after writer process exit; not evidence of the historical failure cause",
        policy=validated_manifest["vector_comparison"], chunks=readback_chunks,
        max_abs_error=max(item["max_abs_error"] for item in readback_chunks),
        bit_mismatch_count=sum(not item["bit_hash_match"] for item in readback_chunks),
    )
    if readback_diagnostics["max_abs_error"] > validated_manifest["vector_comparison"]["absolute_tolerance"]:
        raise ValueError(f"readback diagnostic exceeded validated tolerance: {readback_diagnostics}")
    results = []
    tolerance = 1e-5
    for q in queries:
        gold = [c.chunk_id for c in chunks if c.doc_id in q["relevant_doc_ids"]]
        exact = retrieve(query_vectors[q["text"]], ids, corpus, len(chunks)) if q["text"].strip() else []
        chroma = retrieve_index(args.index, query_vectors[q["text"]], len(chunks),
                                model_id=MODEL_ID, model_revision=MODEL_REVISION) if q["text"].strip() else []
        same_ids = [h.chunk_id for h in chroma] == [h.chunk_id for h in exact]
        scores_match = len(chroma) == len(exact) and all(
            math.isclose(a.score, b.score, abs_tol=tolerance, rel_tol=0)
            for a, b in zip(chroma, exact))
        lc_checks = []
        for k in (1, 2):
            lc_docs = langchain_retrieve(args.index, q["text"], k, CachedQueries(),
                                         model_id=MODEL_ID, model_revision=MODEL_REVISION)
            lc_ids = [doc.metadata["chunk_id"] for doc in lc_docs]
            correct = lc_ids == [h.chunk_id for h in exact[:k]]
            provenance = all(
                doc.page_content == expected[doc.metadata["chunk_id"]].text
                and doc.metadata == {key: asdict(expected[doc.metadata["chunk_id"]])[key]
                                     for key in ("chunk_id", "doc_id", "start", "end")}
                for doc in lc_docs)
            lc_checks.append(dict(k=k, ids=lc_ids, ids_match=correct, provenance_match=provenance,
                                 documents=[dict(text=doc.page_content, metadata=doc.metadata) for doc in lc_docs]))
        results.append(dict(
            query_id=q["query_id"], text=q["text"], gold_chunk_ids=gold,
            exact=[asdict(h) for h in exact], chroma=[asdict(h) for h in chroma],
            chroma_ids_match=same_ids, chroma_scores_match=scores_match,
            exact_metrics=_scores(exact, gold, len(chunks)),
            chroma_metrics=_scores(chroma, gold, len(chunks)), langchain=lc_checks,
            passed=same_ids and scores_match and all(c["ids_match"] and c["provenance_match"] for c in lc_checks),
        ))
    aggregate = {}
    for method in ("exact", "chroma"):
        scores = [r[f"{method}_metrics"] for r in results if not r[f"{method}_metrics"]["excluded"]]
        aggregate[method] = {label: sum(row[key] for row in scores) / len(scores) if scores else None
                             for label, key in (("recall_at_1", "recall_at_1"),
                                                ("recall_at_2", "recall_at_2"), ("mrr", "rr"))}
    status = _git(root, "status", "--porcelain")
    receipt = dict(
        schema_version=1, purpose="persistent retrieval and LangChain integration only",
        started_at_utc=started_at, duration_seconds=time.perf_counter() - started,
        source_sha=_git(root, "rev-parse", "HEAD"), source_dirty=None if status is None else bool(status),
        source_file_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(root.joinpath("rag_baseline").glob("*.py"))},
        requirements_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (root / "requirements-persistent.txt", root / "requirements-semantic-lock.txt")},
        fixture_sha256=hashlib.sha256(raw).hexdigest(), fixture=fixture,
        model_id=MODEL_ID, model_revision=MODEL_REVISION, device="cpu", seed=None,
        query_encoding_policy="one real corpus/query encoder batch; LangChain query callback reuses those actual vectors",
        python=sys.version, platform=platform.platform(),
        dependency_versions={d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]},
        config=dict(size=200, overlap=0, score_abs_tolerance=tolerance, langchain_k=[1, 2]),
        index_manifest=saved_manifest, writer_process_exited=True,
        readback_diagnostics=readback_diagnostics,
        artifacts={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (archive, persist_log)},
        metric_policy="Macro gold-query recall@1,@2 and full-corpus MRR; no-gold excluded, no answer or abstention evaluation.",
        aggregate=aggregate, results=results, passed=all(r["passed"] for r in results),
    )
    payload = json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    print(json.dumps(dict(output=str(args.output), sha256=hashlib.sha256(payload.encode()).hexdigest(),
                         passed=receipt["passed"])))
    if not receipt["passed"]:
        raise SystemExit("retrieval parity failed; inspect the saved full receipt")


if __name__ == "__main__":
    main()
