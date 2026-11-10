"""Colab only: python -m rag_baseline.semantic_cli --output /content/semantic.json.

Compare real sentence embeddings with lexical retrieval on the identical tiny
original corpus. No answer generation, semantic support or abstention claim.
Output must be new.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import sys
import time

from .cli import _git
from .core import Document, chunk_document, embed_texts, encode_query, retrieve, retrieval_metrics
from .semantic import MODEL_ID, MODEL_REVISION, load_encoder, encode_texts


def _scores(hits, gold_ids, count):
    ranked = [hit.chunk_id for hit in hits]
    at_one = retrieval_metrics(ranked, gold_ids, 1)
    at_two = retrieval_metrics(ranked, gold_ids, 2)
    all_ranks = retrieval_metrics(ranked, gold_ids, count)
    return dict(recall_at_1=at_one["recall"], recall_at_2=at_two["recall"],
                rr=all_ranks["rr"], excluded=at_one["excluded"])


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=root / "fixtures/semantic.json")
    args = parser.parse_args()
    # Fail before loading/downloading a model if the destination already exists.
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("output must be a new file in an existing directory")
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    fixture_bytes = args.fixture.read_bytes()
    fixture = json.loads(fixture_bytes)
    documents = [Document(**item) for item in fixture["documents"]]
    size, overlap = 200, 0
    chunks = [chunk for doc in documents for chunk in chunk_document(doc, size, overlap)]
    doc_ids = {doc.doc_id for doc in documents}
    if (not documents or len(doc_ids) != len(documents)
            or any(not doc.text.strip() or len(doc.text) > size for doc in documents)):
        raise ValueError("fixture needs unique nonempty documents fitting one chunk each")
    queries = fixture["queries"]
    if len({item["query_id"] for item in queries}) != len(queries):
        raise ValueError("fixture query IDs must be unique")
    if any(not set(item["relevant_doc_ids"]).issubset(doc_ids) for item in queries):
        raise ValueError("gold must name existing fixture documents")
    vocabulary, lexical_vectors = embed_texts([chunk.text for chunk in chunks])
    encoder = load_encoder(MODEL_ID, MODEL_REVISION)
    # Batch corpus and all nonblank queries together for compatible dimensions.
    nonblank = [item for item in queries if item["text"].strip()]
    encoded = encode_texts(encoder, [c.text for c in chunks] + [q["text"] for q in nonblank])
    corpus_vectors = encoded[:len(chunks)]
    query_vectors = {item["query_id"]: row for item, row in zip(nonblank, encoded[len(chunks):])}
    ids = [chunk.chunk_id for chunk in chunks]
    results = []
    for item in queries:
        gold = [c.chunk_id for c in chunks if c.doc_id in item["relevant_doc_ids"]]
        semantic_hits = (retrieve(query_vectors[item["query_id"]], ids, corpus_vectors, len(chunks))
                         if item["text"].strip() else [])
        lexical_query = encode_query(item["text"], vocabulary)
        lexical_hits = retrieve(lexical_query, ids, lexical_vectors, len(chunks)) if any(lexical_query) else []
        results.append(dict(
            query_id=item["query_id"], query=item["text"], gold_chunk_ids=gold,
            semantic=dict(hits=[asdict(h) for h in semantic_hits], metrics=_scores(semantic_hits, gold, len(chunks))),
            lexical=dict(hits=[asdict(h) for h in lexical_hits], metrics=_scores(lexical_hits, gold, len(chunks))),
        ))
    aggregate = {}
    for method in ("semantic", "lexical"):
        gold_scores = [r[method]["metrics"] for r in results if not r[method]["metrics"]["excluded"]]
        aggregate[method] = {
            label: sum(row[key] for row in gold_scores) / len(gold_scores) if gold_scores else None
            for label, key in (("recall_at_1", "recall_at_1"),
                               ("recall_at_2", "recall_at_2"), ("mrr", "rr"))
        }
    status = _git(root, "status", "--porcelain")
    manifest = dict(
        schema_version=1, purpose="retrieval comparison only", started_at_utc=started_at,
        duration_seconds=time.perf_counter() - started,
        source_sha=_git(root, "rev-parse", "HEAD"), source_dirty=None if status is None else bool(status),
        source_file_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in sorted((root / "rag_baseline").glob("*.py"))},
        requirements_sha256=hashlib.sha256((root / "requirements-semantic.txt").read_bytes()).hexdigest(),
        python=sys.version, platform=platform.platform(), device="cpu",
        dependency_versions={dist.metadata["Name"]: dist.version for dist in distributions() if dist.metadata["Name"]},
        model_id=MODEL_ID, model_revision=MODEL_REVISION, dimension=len(corpus_vectors[0]),
        normalization="L2", precision="float32 encoder output, Python float cosine",
        model_max_seq_length=encoder.max_seq_length,
        seed=None, randomness="no sampling or training; encoder inference only",
        config=dict(size=size, overlap=overlap, batch_size=32, ranking="exact cosine then chunk ID"),
        fixture=fixture, fixture_sha256=hashlib.sha256(fixture_bytes).hexdigest(),
        document_sha256={d.doc_id: hashlib.sha256(d.text.encode()).hexdigest() for d in documents},
        chunks=[asdict(c) for c in chunks],
        metric_policy="Macro gold-query recall@1,@2 and full-corpus MRR; no-gold excluded, no answer evaluation.",
        gold_query_count=sum(bool(q["relevant_doc_ids"]) for q in queries),
        no_gold_query_count=sum(not q["relevant_doc_ids"] for q in queries),
        aggregate=aggregate, results=results,
    )
    payload = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    print(json.dumps(dict(output=str(args.output), sha256=hashlib.sha256(payload.encode()).hexdigest())))


if __name__ == "__main__":
    main()
