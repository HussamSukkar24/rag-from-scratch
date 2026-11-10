"""Index unlabelled JSON documents, query an index, or inspect stored spans.

UTF-8 JSON text only; PDFs and network scraping are unsupported. Models are
loaded lazily. Run functional commands in the authorised Colab environment.
Redirect stdout to retain JSON receipts; no output files are overwritten here.
"""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
import math
from pathlib import Path
import platform
import sys
import time

from .cli import _git
from .core import Document, Chunk, Hit, chunk_document, assemble_context
from .grounded import (QA_MODEL_ID, QA_MODEL_REVISION, load_qa,
                       answer_question, validate_grounded_citations)
from .persistent import persist_index, retrieve_index, langchain_retrieve, _load, _bytes
from .semantic import MODEL_ID, MODEL_REVISION, load_encoder, encode_texts


def _documents(path):
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("documents"), list) or not data["documents"]:
        raise ValueError("documents must be a nonempty JSON list")
    documents, ids = [], set()
    for record in data["documents"]:
        if (not isinstance(record, dict) or set(record) - {"doc_id", "text", "source"}
                or not isinstance(record.get("doc_id"), str) or not record["doc_id"].strip()
                or not isinstance(record.get("text"), str) or not record["text"].strip()
                or not isinstance(record.get("source", ""), str)):
            raise ValueError("each document needs a nonempty doc_id and text, and optional string source")
        if record["doc_id"] in ids:
            raise ValueError("document IDs must be unique")
        ids.add(record["doc_id"])
        documents.append(Document(**record))
    return documents, hashlib.sha256(raw).hexdigest()


def _index(args):
    documents, input_hash = _documents(args.documents)
    if args.index.exists() or args.index.is_symlink() or not args.index.parent.is_dir():
        raise ValueError("index must be a new path with an existing parent directory")
    chunks = [c for doc in documents for c in chunk_document(doc, args.size, args.overlap)]
    encoder = load_encoder(MODEL_ID, MODEL_REVISION)
    vectors = encode_texts(encoder, [c.text for c in chunks])
    manifest = persist_index(args.index, chunks, vectors, model_id=MODEL_ID, model_revision=MODEL_REVISION)
    return dict(
        index=str(args.index), input_sha256=input_hash, document_count=len(documents),
        chunk_count=len(chunks), embedding_dimension=manifest["dimension"],
        index_manifest_sha256=hashlib.sha256(_bytes(manifest)).hexdigest(),
        config=dict(size=args.size, overlap=args.overlap, offset_unit="python_character"),
        documents=[dict(doc_id=d.doc_id, source=d.source,
                        text_sha256=hashlib.sha256(d.text.encode()).hexdigest()) for d in documents],
        chunks=[asdict(c) for c in chunks], source_hash_policy=manifest["source_hash_policy"],
    )


def _query(args, manifest):
    chunks = [Chunk(**record) for record in manifest["chunks"]]
    hits, evidence = [], []
    if args.k:
        encoder = load_encoder(MODEL_ID, MODEL_REVISION)
        vector = encode_texts(encoder, [args.question])[0]
        # Score every row so actual LangChain tie choices can be mapped by ID.
        scored = retrieve_index(args.index, vector, len(chunks), model_id=MODEL_ID, model_revision=MODEL_REVISION)
        scores = {hit.chunk_id: hit.score for hit in scored}
        selected = langchain_retrieve(args.index, args.question, args.k, encoder,
                                      model_id=MODEL_ID, model_revision=MODEL_REVISION)
        for rank, document in enumerate(selected, 1):
            chunk_id = document.metadata["chunk_id"]
            if chunk_id not in scores:
                raise ValueError("LangChain selected an ID absent from validated Chroma scores")
            hit = Hit(chunk_id, scores[chunk_id], rank)
            hits.append(hit)
            evidence.append(dict(document.metadata, text=document.page_content, score=hit.score, rank=rank))
    qa = None

    def ask_qa(**kwargs):
        nonlocal qa
        if qa is None:
            qa = load_qa(QA_MODEL_ID, QA_MODEL_REVISION)
        return qa(**kwargs)

    answer = answer_question(args.question, hits, chunks, ask_qa,
                             min_score=args.min_score, max_chars=args.max_chars)
    valid = validate_grounded_citations(answer, hits, chunks)
    if not valid:
        raise ValueError("answer failed mechanical citation validation")
    return dict(
        index=str(args.index), question=args.question, evidence=evidence,
        context=assemble_context(hits, chunks, args.max_chars), answer=answer, citation_valid=valid,
        qa_model=dict(model_id=QA_MODEL_ID, revision=QA_MODEL_REVISION),
        config=dict(k=args.k, min_score=args.min_score, max_chars=args.max_chars),
        source_hash_policy=manifest["source_hash_policy"],
        citation_policy="exact stored quotes, spans and retrieved membership; not semantic truth",
    )


def _inspect(args, manifest):
    chunk = next((record for record in manifest["chunks"] if record["chunk_id"] == args.chunk_id), None)
    if chunk is None:
        raise ValueError("unknown chunk ID")
    return dict(index=str(args.index), chunk=chunk,
                source_span=dict(doc_id=chunk["doc_id"], start=chunk["start"], end=chunk["end"],
                                 quote=chunk["text"], offset_unit="python_character"),
                source_hash_policy=manifest["source_hash_policy"],
                inspection_scope="validated stored source span, not an unseen original full-document hash")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="create a new index from unlabelled JSON documents")
    index.add_argument("--documents", type=Path, required=True)
    index.add_argument("--index", type=Path, required=True)
    index.add_argument("--size", type=int, default=200)
    index.add_argument("--overlap", type=int, default=0)
    query = commands.add_parser("query", help="ask a question against an existing validated index")
    query.add_argument("--index", type=Path, required=True)
    query.add_argument("--question", required=True)
    query.add_argument("--k", type=int, default=2)
    query.add_argument("--min-score", type=float, default=0.5)
    query.add_argument("--max-chars", type=int, default=2000)
    inspect = commands.add_parser("inspect", help="show a stored chunk and its source span without loading models")
    inspect.add_argument("--index", type=Path, required=True)
    inspect.add_argument("--chunk-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "index" and (args.size <= 0 or not 0 <= args.overlap < args.size):
        parser.error("require size>0 and 0<=overlap<size")
    if args.command == "query" and (not args.question.strip() or args.k < 0 or args.max_chars < 0
            or not math.isfinite(args.min_score) or not 0 <= args.min_score <= 1):
        parser.error("require nonempty question, k/budget>=0 and finite min-score in [0,1]")
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    # Third-party progress output belongs on stderr; successful stdout is JSON.
    with redirect_stdout(sys.stderr):
        if args.command == "index":
            result = _index(args)
        else:
            _, _, manifest = _load(args.index, MODEL_ID, MODEL_REVISION)
            result = _query(args, manifest) if args.command == "query" else _inspect(args, manifest)
            result["index_manifest_sha256"] = hashlib.sha256(_bytes(manifest)).hexdigest()
    root = Path(__file__).resolve().parent.parent
    lock = root / "requirements-grounded-lock.txt"
    status = _git(root, "status", "--porcelain")
    result.update(
        schema_version=1, command=args.command, started_at_utc=started_at,
        duration_seconds=time.perf_counter() - started, source_sha=_git(root, "rev-parse", "HEAD"),
        source_dirty=None if status is None else bool(status), python=sys.version,
        platform=platform.platform(), device="cpu", seed=None,
        embedding_model=dict(model_id=MODEL_ID, revision=MODEL_REVISION),
        dependency_versions={d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]},
        source_file_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((root / "rag_baseline").glob("*.py"))},
        requirements_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    main()
