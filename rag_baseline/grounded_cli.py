"""Colab only: python -m rag_baseline.grounded_cli --index NEW --output NEW.json.

Extractive QA only. Report retrieval, exact-answer and no-answer outcomes
separately. Fixed threshold is never selected or tuned on held-out evaluation.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

from .cli import _git
from .core import Document, Hit, chunk_document, retrieval_metrics, _context_entries
from .grounded import (QA_MODEL_ID, QA_MODEL_REVISION, load_qa,
                       answer_question, validate_grounded_citations)
from .persistent import retrieve_index, langchain_retrieve
from .semantic import MODEL_ID, MODEL_REVISION, load_encoder, encode_texts


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=root / "fixtures/grounded.json")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--k", type=int, default=2)
    args = parser.parse_args()
    if not 0 <= args.min_score <= 1 or args.max_chars < 0 or args.k < 1:
        parser.error("require score in [0,1], nonnegative budget and k>=1")
    archive = args.output.with_suffix(".index.zip")
    persist_log = args.output.with_suffix(".persist.log")
    outputs = (args.output, archive, persist_log)
    if (args.index.exists() or not args.index.parent.is_dir()
            or any(path.exists() or not path.parent.is_dir() for path in outputs)
            or len({path.resolve() for path in outputs}) != len(outputs)):
        parser.error("index and receipt artifacts must be new paths with existing parents")
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    raw = args.fixture.read_bytes()
    fixture = json.loads(raw)
    documents = [Document(**row) for row in fixture["documents"]]
    doc_ids = {d.doc_id for d in documents}
    splits = {name: fixture[f"{name}_queries"] for name in ("development", "evaluation")}
    queries = [q for rows in splits.values() for q in rows]
    if (not documents or len(doc_ids) != len(documents)
            or any(not d.text.strip() or len(d.text) > 200 for d in documents)):
        raise ValueError("fixture requires unique nonempty one-chunk documents")
    if len({q["query_id"] for q in queries}) != len(queries):
        raise ValueError("query IDs must be unique across development/evaluation")
    for query in queries:
        if not set(query["relevant_doc_ids"]).issubset(doc_ids):
            raise ValueError("gold document ID is absent from source fixture")
        if query["expected_answer"] is not None and not isinstance(query["expected_answer"], str):
            raise ValueError("expected_answer must be a string or null")
    chunks = [c for d in documents for c in chunk_document(d, 200, 0)]
    encoder = load_encoder(MODEL_ID, MODEL_REVISION)
    nonblank = [q for q in queries if q["text"].strip()]
    vectors = encode_texts(encoder, [c.text for c in chunks] + [q["text"] for q in nonblank])
    corpus_vectors = vectors[:len(chunks)]
    query_vectors = {q["text"]: row for q, row in zip(nonblank, vectors[len(chunks):], strict=True)}

    class CachedQueries:
        def encode(self, texts, **kwargs):
            if any(text not in query_vectors for text in texts):
                raise ValueError("embedding cache refuses unknown text")
            return [query_vectors[text] for text in texts]

    writer = (
        "import json,sys; from rag_baseline.core import Chunk; "
        "from rag_baseline.persistent import persist_index; p=json.load(sys.stdin); "
        "p['chunks']=[Chunk(**c) for c in p['chunks']]; persist_index(**p)"
    )
    request = dict(path=str(args.index.resolve()), chunks=[asdict(c) for c in chunks], vectors=corpus_vectors,
                   model_id=MODEL_ID, model_revision=MODEL_REVISION)
    with persist_log.open("x", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-c", writer], input=json.dumps(request),
                       text=True, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=root)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(args.index.rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(args.index))
    qa = load_qa(QA_MODEL_ID, QA_MODEL_REVISION)
    results = []
    sources = {d.doc_id: d.text for d in documents}
    for split, split_queries in splits.items():
        for query in split_queries:
            text = query["text"]
            chroma_hits = (retrieve_index(args.index, query_vectors[text], args.k,
                                          model_id=MODEL_ID, model_revision=MODEL_REVISION)
                           if text.strip() else [])
            lc_docs = langchain_retrieve(args.index, text, args.k, CachedQueries(),
                                         model_id=MODEL_ID, model_revision=MODEL_REVISION)
            score_by_id = {hit.chunk_id: hit.score for hit in chroma_hits}
            lc_ids = [doc.metadata["chunk_id"] for doc in lc_docs]
            if set(lc_ids) != set(score_by_id):
                raise ValueError("LangChain and Chroma selected different evidence")
            hits = [Hit(chunk_id, score_by_id[chunk_id], rank) for rank, chunk_id in enumerate(lc_ids, 1)]
            trace = []
            selected = [chunk for chunk, _ in _context_entries(hits, chunks, args.max_chars)]

            def recorded_qa(**kwargs):
                response = qa(**kwargs)
                source_index = len(trace) - 1
                trace.append(dict(phase="combined_context" if not trace else "source_evidence",
                                  chunk_id=selected[source_index].chunk_id if trace else None,
                                  context=kwargs["context"], response=response))
                return response

            answer = answer_question(text, hits, chunks, recorded_qa,
                                     min_score=args.min_score, max_chars=args.max_chars)
            valid = validate_grounded_citations(answer, hits, chunks)
            original_valid = all(sources[c["doc_id"]][c["start"]:c["end"]] == c["quote"]
                                 for c in answer["citations"])
            expected = query["expected_answer"]
            correct = answer["abstained"] if expected is None else not answer["abstained"] and answer["text"] == expected
            gold_ids = [c.chunk_id for c in chunks if c.doc_id in query["relevant_doc_ids"]]
            retrieved_docs = {doc.metadata["doc_id"] for doc in lc_docs}
            # A conflict case needs all labelled evidence; an answerable question
            # needs at least one gold source. Neither rule proves answer support.
            evidence_present = (bool(retrieved_docs.intersection(query["relevant_doc_ids"])) if expected is not None
                                else set(query["relevant_doc_ids"]).issubset(retrieved_docs))
            if not valid or not original_valid:
                outcome = "citation_failure"
            elif not evidence_present:
                outcome = "retrieval_failure"
            elif correct:
                outcome = "correct_no_answer" if expected is None else "exact_answer"
            else:
                outcome = "answer_stage_failure"
            results.append(dict(
                split=split, query_id=query["query_id"], query=text,
                expected_answer=expected, gold_chunk_ids=gold_ids, hits=[asdict(h) for h in hits],
                langchain_documents=[dict(text=d.page_content, metadata=d.metadata) for d in lc_docs],
                retrieval_metrics=retrieval_metrics(lc_ids, gold_ids, args.k), evidence_present=evidence_present,
                qa_trace=trace, answer=answer, citation_valid=valid, original_source_valid=original_valid,
                desired_behavior_matches=correct, outcome=outcome,
            ))
    summary = {}
    for split in splits:
        rows = [r for r in results if r["split"] == split]
        answerable = [r for r in rows if r["expected_answer"] is not None]
        unanswerable = [r for r in rows if r["expected_answer"] is None]
        summary[split] = dict(
            query_count=len(rows),
            exact_answer_rate=sum(r["desired_behavior_matches"] for r in answerable) / len(answerable) if answerable else None,
            no_answer_accuracy=sum(r["desired_behavior_matches"] for r in unanswerable) / len(unanswerable) if unanswerable else None,
            retrieval_failure_count=sum(r["outcome"] == "retrieval_failure" for r in rows),
            answer_stage_failure_count=sum(r["outcome"] == "answer_stage_failure" for r in rows),
            citation_failure_count=sum(r["outcome"] == "citation_failure" for r in rows),
        )
    status = _git(root, "status", "--porcelain")
    receipt = dict(
        schema_version=1, purpose="extractive grounding evaluation, no generative model or entailment guarantee",
        source_sha=_git(root, "rev-parse", "HEAD"), source_dirty=None if status is None else bool(status),
        started_at_utc=started_at, duration_seconds=time.perf_counter() - started,
        python=sys.version, platform=platform.platform(), device="cpu", seed=None,
        embedding_model=dict(model_id=MODEL_ID, revision=MODEL_REVISION),
        qa_model=dict(model_id=QA_MODEL_ID, revision=QA_MODEL_REVISION, safetensors=True),
        dependency_versions={d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]},
        source_file_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((root / "rag_baseline").glob("*.py"))},
        requirements_sha256=hashlib.sha256((root / "requirements-grounded.txt").read_bytes()).hexdigest(),
        fixture=fixture, fixture_sha256=hashlib.sha256(raw).hexdigest(),
        document_sha256={d.doc_id: hashlib.sha256(d.text.encode()).hexdigest() for d in documents},
        config=dict(size=200, overlap=0, k=args.k, min_score=args.min_score, max_chars=args.max_chars),
        threshold_policy="fixed before execution; development reported separately; no evaluation-driven selection",
        ambiguity_policy="After a valid combined extraction, independently QA every retrieved budget-selected whole body with the same query/threshold; two distinct valid answers after strip/casefold/whitespace normalization abstain as conflicting_evidence. Numbers, units and punctuation retained. Ambiguity signal, not contradiction proof or a safety guarantee.",
        embedding_policy="one real corpus/query batch reused by actual LangChain query embedding callback",
        metric_policy="strict expected string match or desired abstention; null/conflict behavior is not assumed; no-gold retrieval metrics excluded",
        index_manifest=json.loads((args.index / "manifest.json").read_text(encoding="utf-8")),
        artifacts={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (archive, persist_log)},
        summary=summary, results=results,
    )
    payload = json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    print(json.dumps(dict(output=str(args.output), sha256=hashlib.sha256(payload.encode()).hexdigest(), summary=summary)))
    if any(not r["citation_valid"] or not r["original_source_valid"] for r in results):
        raise SystemExit("citation mechanics failed; inspect saved receipt")


if __name__ == "__main__":
    main()
