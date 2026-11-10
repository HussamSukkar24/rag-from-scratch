"""Extractive QA with mechanical provenance, not semantic entailment checks.

Invalid caller configuration raises ValueError. Missing or invalid evidence and
malformed backend answers abstain conservatively. Backend exceptions propagate
so execution failures cannot be reported as successful no-answer decisions.
"""

import math
import re

from .core import _context_entries


QA_MODEL_ID = "deepset/minilm-uncased-squad2"
# Official metadata: https://huggingface.co/api/models/deepset/minilm-uncased-squad2
QA_MODEL_REVISION = "934656cdda79824eabf503ed56e15c01ddbdbe3f"
# Ranked spans requested from the combined-context call. Only the backend's own
# scored spans are considered; no text is ever added to an extracted answer.
ANSWER_CANDIDATES = 5


def load_qa(model_id=QA_MODEL_ID, revision=QA_MODEL_REVISION):
    """Lazy CPU-only candidate loader. Actual installed-version check is in Colab."""
    if not isinstance(revision, str) or not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full immutable commit SHA")
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=False, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("QA needs a fast tokenizer for exact character offsets")
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_id, revision=revision, trust_remote_code=False, use_safetensors=True)
    model.to("cpu")
    return pipeline("question-answering", model=model, tokenizer=tokenizer,
                    device=-1, framework="pt")


def _abstain(reason):
    return dict(text="", cited_chunk_ids=[], abstained=True, reason=reason, citations=[])


def answer_question(query, hits, chunks, qa, *, min_score=0.5, max_chars=2000):
    """Accept one exact answer wholly inside one retrieved chunk's text body.

    Whole chunks use core's budget/header policy. Oversized chunks are skipped.
    Returned citation offsets are absolute Python characters in the supplied
    original chunk's document. Headers and cross-chunk answers cannot be cited.
    A model score is uncalibrated confidence, not a truth or safety guarantee.
    The combined call requests ranked candidates. If a lower-ranked candidate
    also passes every check, lies in the same chunk and strictly contains the
    top span, the widest such candidate is returned with its own score. This
    recovers phrases such as "fourteen days" when the top span is "fourteen".
    After a valid combined extraction, independently check each selected body.
    Two different valid above-threshold source answers cause conservative
    abstention. This can reject compatible answers and miss real contradictions.
    """
    if (isinstance(min_score, bool) or not isinstance(min_score, (int, float))
            or not math.isfinite(min_score) or not 0 <= min_score <= 1):
        raise ValueError("min_score must be finite and in [0,1]")
    if type(max_chars) is not int or max_chars < 0:
        raise ValueError("max_chars must be a nonnegative integer")
    if query is None or (isinstance(query, str) and not query.strip()):
        return _abstain("empty_query")
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    if hits is None or chunks is None:
        return _abstain("no_evidence")
    hits, chunks = list(hits), list(chunks)
    if not hits or not chunks:
        return _abstain("no_evidence")
    if any(type(c.start) is not int or type(c.end) is not int or c.start < 0
           or c.end <= c.start or len(c.text) != c.end - c.start for c in chunks):
        return _abstain("invalid_evidence")
    try:
        entries = _context_entries(hits, chunks, max_chars)
    except ValueError:
        return _abstain("invalid_evidence")
    if not entries:
        return _abstain("context_budget")
    context = "\n\n".join(block for _, block in entries)
    ranges = []
    position = 0
    for chunk, block in entries:
        body_start = position + len(block) - len(chunk.text)
        ranges.append((body_start, body_start + len(chunk.text), chunk))
        position += len(block) + 2
    result = qa(question=query, context=context, handle_impossible_answer=True,
                top_k=ANSWER_CANDIDATES)
    # The pipeline returns a bare dict when it finds a single candidate.
    candidates = result if isinstance(result, list) else [result]
    if not candidates:
        return _abstain("invalid_backend_response")
    combined = _validated_answer(candidates[0], context, ranges, min_score)
    if combined["abstained"]:
        return combined
    combined = _widest_containing(combined, candidates[1:], context, ranges, min_score)
    source_answers = set()
    for chunk, _ in entries:
        source = _validated_answer(
            qa(question=query, context=chunk.text, handle_impossible_answer=True),
            chunk.text, [(0, len(chunk.text), chunk)], min_score)
        if not source["abstained"]:
            source_answers.add(" ".join(source["text"].strip().casefold().split()))
    # Different confident extractions are an ambiguity signal, not proof of a
    # semantic contradiction. Only the already budget-selected evidence votes.
    if len(source_answers) >= 2:
        return _abstain("conflicting_evidence")
    return combined


def _widest_containing(best, candidates, context, ranges, min_score):
    """Prefer a validated candidate span that strictly contains the top span."""
    top = best["citations"][0]
    for candidate in candidates:
        answer = _validated_answer(candidate, context, ranges, min_score)
        if answer["abstained"]:
            continue
        cited = answer["citations"][0]
        current = best["citations"][0]
        if (cited["chunk_id"] == top["chunk_id"]
                and cited["start"] <= top["start"] and top["end"] <= cited["end"]
                and cited["end"] - cited["start"] > current["end"] - current["start"]):
            best = answer
    return best


def _validated_answer(result, context, ranges, min_score):
    """Apply identical confidence, exact-span and source mapping to each call."""
    if not isinstance(result, dict):
        return _abstain("invalid_backend_response")
    answer, score = result.get("answer"), result.get("score")
    if (not isinstance(answer, str) or isinstance(score, bool)
            or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
        return _abstain("invalid_backend_response")
    if not answer.strip():
        return _abstain("no_answer")
    if score < min_score:
        return _abstain("below_threshold")
    start, end = result.get("start"), result.get("end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(context):
        return _abstain("invalid_offsets")
    if context[start:end] != answer:
        return _abstain("answer_mismatch")
    for body_start, body_end, chunk in ranges:
        if body_start <= start < end <= body_end:
            source_start = chunk.start + start - body_start
            source_end = chunk.start + end - body_start
            citation = dict(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id,
                            start=source_start, end=source_end, quote=answer)
            return dict(text=answer, cited_chunk_ids=[chunk.chunk_id], abstained=False,
                        reason="answered", citations=[citation])
    return _abstain("outside_chunk")


def validate_grounded_citations(answer, hits, chunks):
    """Validate exact quoted spans and retrieved IDs only, never semantic truth."""
    try:
        if answer["abstained"]:
            return not answer["cited_chunk_ids"] and not answer["citations"] and answer["text"] == ""
        if len(answer["citations"]) != 1 or len(answer["cited_chunk_ids"]) != 1:
            return False
        citation = answer["citations"][0]
        chunk_id = citation["chunk_id"]
        by_id = {c.chunk_id: c for c in chunks}
        if len(by_id) != len(chunks) or chunk_id not in by_id or chunk_id not in {h.chunk_id for h in hits}:
            return False
        chunk = by_id[chunk_id]
        start, end = citation["start"], citation["end"]
        return (type(chunk.start) is int and type(chunk.end) is int and chunk.start >= 0
                and len(chunk.text) == chunk.end - chunk.start
                and type(start) is int and type(end) is int and chunk.start <= start < end <= chunk.end
                and citation["doc_id"] == chunk.doc_id and answer["cited_chunk_ids"] == [chunk_id]
                and answer["text"] == citation["quote"] == chunk.text[start - chunk.start:end - chunk.start])
    except (KeyError, TypeError, AttributeError):
        return False
