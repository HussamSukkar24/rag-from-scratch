"""Small deterministic retrieval primitives, without semantic-model claims."""

from collections import Counter
from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    source: str = ""


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float
    rank: int


def chunk_document(doc, size, overlap):
    """Split at Python character offsets [start,end), without a redundant tail."""
    if not isinstance(size, int) or not isinstance(overlap, int):
        raise ValueError("size and overlap must be integers")
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("require size > 0 and 0 <= overlap < size")
    chunks = []
    start = 0
    while start < len(doc.text):
        end = min(start + size, len(doc.text))
        chunks.append(Chunk(f"{doc.doc_id}:{start}:{end}", doc.doc_id,
                            start, end, doc.text[start:end]))
        if end == len(doc.text):
            break
        start = end - overlap
    return chunks


def _tokens(text):
    return re.findall(r"\w+", text.lower())


def encode_query(query, vocabulary):
    """Unknown words are ignored; an empty or unseen query encodes to zero."""
    counts = Counter(_tokens(query))
    vector = [float(counts[word]) for word in vocabulary]
    norm = math.hypot(*vector)
    return [value / norm for value in vector] if norm else vector


def embed_texts(texts):
    texts = list(texts)
    vocabulary = sorted({word for text in texts for word in _tokens(text)})
    return vocabulary, [encode_query(text, vocabulary) for text in texts]


def _unit_vector(vector, dimension):
    if len(vector) != dimension or not all(math.isfinite(x) for x in vector):
        raise ValueError("vectors must have matching dimensions and finite values")
    norm = math.hypot(*vector)
    if not norm or not math.isfinite(norm):
        raise ValueError("cosine requires nonzero vectors with finite norms")
    return [x / norm for x in vector]


def retrieve(query_vector, chunk_ids, vectors, k):
    """Exact cosine, descending scores, then ascending chunk IDs for ties."""
    if not isinstance(k, int) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    if len(chunk_ids) != len(vectors) or len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("index needs one unique chunk ID per vector")
    if k == 0 or not chunk_ids:
        return []
    query = _unit_vector(query_vector, len(query_vector))
    rows = [_unit_vector(row, len(query)) for row in vectors]
    scores = [(chunk_id, max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(query, row)))))
              for chunk_id, row in zip(chunk_ids, rows)]
    scores.sort(key=lambda pair: (-pair[1], pair[0]))
    return [Hit(chunk_id, score, rank)
            for rank, (chunk_id, score) in enumerate(scores[:k], 1)]


def _context_entries(hits, chunks, max_chars):
    if not isinstance(max_chars, int) or max_chars < 0:
        raise ValueError("max_chars must be a nonnegative integer")
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if len(by_id) != len(chunks):
        raise ValueError("duplicate chunk IDs")
    entries = []
    used = 0
    seen = set()
    for hit in hits:
        if hit.chunk_id not in by_id or hit.chunk_id in seen:
            raise ValueError("hits must identify unique known chunks")
        seen.add(hit.chunk_id)
        chunk = by_id[hit.chunk_id]
        block = f"[{chunk.chunk_id}] {chunk.doc_id} [{chunk.start},{chunk.end})\n{chunk.text}"
        cost = len(block) + (2 if entries else 0)
        if used + cost <= max_chars:
            entries.append((chunk, block))
            used += cost
    return entries


def assemble_context(hits, chunks, max_chars):
    """Include whole chunks in hit order; skip any that do not fit the budget."""
    return "\n\n".join(block for _, block in _context_entries(hits, chunks, max_chars))


def retrieval_metrics(ranked_ids, relevant_ids, k):
    """One-query metrics. No-gold queries return null metrics and excluded=True."""
    if not isinstance(k, int) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    ranked_ids = list(ranked_ids)
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("ranked IDs must be unique")
    gold = set(relevant_ids)
    if not gold:
        return dict(recall=None, precision=None, hit_rate=None, rr=None, excluded=True)
    selected = ranked_ids[:k]
    count = len(gold.intersection(selected))
    rr = next((1 / rank for rank, item in enumerate(selected, 1) if item in gold), 0.0)
    return dict(recall=count / len(gold), precision=count / k if k else 0.0,
                hit_rate=float(count > 0), rr=rr, excluded=False)


# shortcut: lexical coverage is a deliberately narrow fixture policy, not entailment.
_QUESTION_WORDS = set("a an the is are was were do does did what when where which who how at on in of to for please tell me".split())


def answer_baseline(query, hits, chunks, max_chars=2000):
    """Extract one whole chunk only when all content words occur in it.

    Multiple different matching chunks cause abstention. Synonyms, negation,
    conflicting evidence outside those matches and instructions are not understood.
    """
    required = set(_tokens(query)) - _QUESTION_WORDS
    entries = _context_entries(hits, chunks, max_chars)
    matches = [chunk for chunk, _ in entries
               if required and required.issubset(set(_tokens(chunk.text)))]
    if not matches or len({chunk.text for chunk in matches}) != 1:
        return dict(text="Insufficient evidence.", cited_chunk_ids=[], abstained=True)
    selected = matches[0]
    return dict(text=selected.text, cited_chunk_ids=[selected.chunk_id], abstained=False)


def validate_citations(answer, hits, chunks):
    """Check retrieved membership and exact extraction, not claim entailment."""
    cited = answer["cited_chunk_ids"]
    if answer["abstained"]:
        return not cited
    retrieved = {hit.chunk_id for hit in hits}
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    return (len(cited) == 1 and cited[0] in retrieved and cited[0] in by_id
            and answer["text"] == by_id[cited[0]].text)
