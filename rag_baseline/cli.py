"""Run the synthetic baseline and emit an auditable JSON manifest in Colab."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from .core import (
    Document, chunk_document, embed_texts, encode_query, retrieve,
    assemble_context, answer_baseline, validate_citations, retrieval_metrics,
)


def _git(root, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL,
            text=True, timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=root / "fixtures/baseline.json")
    parser.add_argument("--size", type=int, default=80)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--output", type=Path, help="New JSON file; refuses to overwrite")
    args = parser.parse_args()
    if args.size <= 0 or not 0 <= args.overlap < args.size:
        parser.error("require size > 0 and 0 <= overlap < size")
    if args.k < 0 or args.max_chars < 0:
        parser.error("k and max-chars must be nonnegative")
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    raw = args.fixture.read_bytes()
    fixture = json.loads(raw)
    documents = [Document(**item) for item in fixture["documents"]]
    if len({doc.doc_id for doc in documents}) != len(documents):
        raise ValueError("document IDs must be unique")
    chunks = [chunk for doc in documents
              for chunk in chunk_document(doc, args.size, args.overlap)]
    vocabulary, vectors = embed_texts([chunk.text for chunk in chunks])
    # Punctuation-only chunks cannot participate in cosine retrieval.
    index = [(chunk.chunk_id, row) for chunk, row in zip(chunks, vectors) if any(row)]
    rows = []
    for item in fixture["queries"]:
        query = encode_query(item["text"], vocabulary)
        hits = retrieve(query, [cid for cid, _ in index], [v for _, v in index], args.k) if any(query) else []
        gold_docs = set(item["relevant_doc_ids"])
        gold_ids = [chunk.chunk_id for chunk in chunks if chunk.doc_id in gold_docs]
        answer = answer_baseline(item["text"], hits, chunks, args.max_chars)
        rows.append(dict(
            query_id=item["query_id"], query=item["text"], hits=[asdict(hit) for hit in hits],
            relevant_chunk_ids=gold_ids,
            context=assemble_context(hits, chunks, args.max_chars), answer=answer,
            citation_valid=validate_citations(answer, hits, chunks),
            expected_abstain=item["expect_abstain"],
            abstention_matches=answer["abstained"] == item["expect_abstain"],
            metrics=retrieval_metrics([hit.chunk_id for hit in hits], gold_ids, args.k),
        ))
    included = [row["metrics"] for row in rows if not row["metrics"]["excluded"]]
    no_gold = [row for row in rows if row["metrics"]["excluded"]]
    status = _git(root, "status", "--porcelain")
    manifest = dict(
        schema_version=1, baseline="normalized-word-count-extractive-v1",
        started_at_utc=started_at,
        duration_seconds=time.perf_counter() - started,
        source_sha=_git(root, "rev-parse", "HEAD"),
        source_dirty=None if status is None else bool(status),
        source_file_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in sorted((root / "rag_baseline").glob("*.py"))},
        python=sys.version, platform=platform.platform(), device="cpu", dependencies="stdlib",
        seed=None, randomness="none", fixture_version=fixture["fixture_version"],
        fixture_sha256=hashlib.sha256(raw).hexdigest(),
        document_sha256={doc.doc_id: hashlib.sha256(doc.text.encode()).hexdigest() for doc in documents},
        config=dict(size=args.size, overlap=args.overlap, k=args.k, max_chars=args.max_chars),
        dimension=len(vocabulary), vocabulary=vocabulary, chunks=[asdict(c) for c in chunks],
        indexed_chunk_ids=[cid for cid, _ in index],
        metric_policy="Macro mean over gold queries; precision denominator k; no-gold excluded.",
        aggregate={key: sum(m[key] for m in included) / len(included) if included else None
                   for key in ("recall", "precision", "hit_rate", "rr")},
        gold_query_count=len(included), no_gold_query_count=len(no_gold),
        no_gold_abstention_rate=(sum(row["answer"]["abstained"] for row in no_gold) / len(no_gold)
                                 if no_gold else None),
        results=rows,
    )
    payload = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        print(json.dumps(dict(output=str(args.output), sha256=hashlib.sha256(payload.encode()).hexdigest())))
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
