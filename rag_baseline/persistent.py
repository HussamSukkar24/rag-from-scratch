"""Isolated local Chroma persistence and actual LangChain retrieval, Colab only."""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import struct

from .core import Chunk, Hit
from .semantic import encode_texts


COLLECTION = "rag_chunks"
VECTOR_COMPARISON = dict(absolute_tolerance=1e-6, relative_tolerance=0.0,
                         reference="input rounded to IEEE754 float32")


def _bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _identity(model_id, model_revision):
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be nonempty")
    if not isinstance(model_revision, str) or not re.fullmatch("[0-9a-f]{40}", model_revision):
        raise ValueError("model_revision must be an immutable full commit SHA")


def _vector(row, dimension=None):
    try:
        row = [float(value) for value in row]
    except (TypeError, ValueError) as exc:
        raise ValueError("vector must contain numbers") from exc
    norm = math.hypot(*row)
    if (not row or (dimension is not None and len(row) != dimension)
            or not all(math.isfinite(x) for x in row) or not math.isfinite(norm) or norm == 0):
        raise ValueError("vectors must be finite, nonzero and dimension-compatible")
    if not math.isclose(norm, 1.0, abs_tol=1e-5):
        raise ValueError("persistent index requires normalized vectors")
    return row


def _vector_hash(row):
    return _sha(struct.pack(f"<{len(row)}f", *row))


def _provenance(chunks):
    ids = [chunk.chunk_id for chunk in chunks]
    if not chunks or len(set(ids)) != len(ids):
        raise ValueError("index requires nonempty chunks and unique chunk IDs")
    sources = {}
    for chunk in chunks:
        if (not isinstance(chunk.chunk_id, str) or not chunk.chunk_id
                or not isinstance(chunk.doc_id, str) or not chunk.doc_id
                or not isinstance(chunk.start, int) or not isinstance(chunk.end, int)
                or chunk.start < 0 or chunk.end <= chunk.start
                or not isinstance(chunk.text, str) or len(chunk.text) != chunk.end - chunk.start):
            raise ValueError("chunk must preserve nonempty IDs, text and exact character span")
        sources.setdefault(chunk.doc_id, []).append(asdict(chunk))
    # Full original documents are not supplied by this API. Hash exact supplied
    # source spans and text, never imply a hash of unseen original document text.
    source_hashes = {doc_id: _sha(_bytes(sorted(rows, key=lambda r: r["chunk_id"])))
                     for doc_id, rows in sources.items()}
    config = dict(offset_unit="python_character", spans=[
        dict(chunk_id=c.chunk_id, doc_id=c.doc_id, start=c.start, end=c.end) for c in chunks])
    return config, source_hashes


def _client(path):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


def persist_index(path, chunks, vectors, *, model_id, model_revision):
    """Create a new directory only; incomplete writes stay failed, never rebuilt."""
    _identity(model_id, model_revision)
    chunks, vectors = list(chunks), list(vectors)
    config, source_hashes = _provenance(chunks)
    if len(chunks) != len(vectors):
        raise ValueError("one vector is required per chunk")
    rows = [_vector(row) for row in vectors]
    dimension = len(rows[0])
    rows = [_vector(row, dimension) for row in rows]
    references = {c.chunk_id: list(struct.unpack(f"<{dimension}f", struct.pack(f"<{dimension}f", *row)))
                  for c, row in zip(chunks, rows)}
    manifest = dict(
        schema_version=2, collection=COLLECTION, distance="cosine", model_id=model_id,
        model_revision=model_revision, dimension=dimension, normalization="L2",
        chunk_config=config, source_hash_policy="sha256 of supplied chunk records per document",
        source_hashes=source_hashes, chunks=[asdict(c) for c in chunks],
        vector_hashes={c.chunk_id: _vector_hash(row) for c, row in zip(chunks, rows)},
        reference_vectors=references, vector_comparison=VECTOR_COMPARISON,
    )
    digest = _sha(_bytes(manifest))
    path = Path(path)
    path.mkdir(parents=False, exist_ok=False)
    collection = _client(path).create_collection(
        name=COLLECTION, embedding_function=None,
        metadata={"hnsw:space": "cosine", "manifest_sha256": digest},
    )
    collection.add(ids=[c.chunk_id for c in chunks], embeddings=rows,
                   documents=[c.text for c in chunks], metadatas=[
                       dict(chunk_id=c.chunk_id, doc_id=c.doc_id, start=c.start, end=c.end)
                       for c in chunks])
    # Completion marker written last. Missing marker is always a hard failure.
    with (path / "manifest.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(manifest=manifest, sha256=digest), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return manifest


def _load(path, model_id, model_revision):
    _identity(model_id, model_revision)
    path = Path(path)
    if not (path / "manifest.json").is_file() or not (path / "chroma.sqlite3").is_file():
        raise ValueError("missing completed persistent index or manifest")
    try:
        envelope = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        manifest, digest = envelope["manifest"], envelope["sha256"]
        if _sha(_bytes(manifest)) != digest:
            raise ValueError("corrupt manifest checksum")
        if (manifest["schema_version"] != 2 or manifest["collection"] != COLLECTION
                or manifest["distance"] != "cosine" or manifest["normalization"] != "L2"
                or manifest["vector_comparison"] != VECTOR_COMPARISON):
            raise ValueError("unsupported index configuration")
        if manifest["model_id"] != model_id or manifest["model_revision"] != model_revision:
            raise ValueError("index model identity or revision mismatch")
        chunks = [Chunk(**record) for record in manifest["chunks"]]
        config, source_hashes = _provenance(chunks)
        if config != manifest["chunk_config"] or source_hashes != manifest["source_hashes"]:
            raise ValueError("index chunk configuration or source hash mismatch")
        client = _client(path)
        collection = client.get_collection(COLLECTION, embedding_function=None)
        if collection.metadata != {"hnsw:space": "cosine", "manifest_sha256": digest}:
            raise ValueError("index collection configuration or manifest mismatch")
        stored = collection.get(include=["documents", "metadatas", "embeddings"])
        ids = stored["ids"]
        if len(ids) != len(chunks) or set(ids) != {c.chunk_id for c in chunks}:
            raise ValueError("stored index IDs differ from manifest")
        by_id = {c.chunk_id: c for c in chunks}
        for index, chunk_id in enumerate(ids):
            chunk = by_id[chunk_id]
            expected = dict(chunk_id=chunk_id, doc_id=chunk.doc_id, start=chunk.start, end=chunk.end)
            if stored["documents"][index] != chunk.text or stored["metadatas"][index] != expected:
                raise ValueError("stored index text or provenance differs from manifest")
            row = _vector(stored["embeddings"][index], manifest["dimension"])
            reference = _vector(manifest["reference_vectors"][chunk_id], manifest["dimension"])
            if _vector_hash(reference) != manifest["vector_hashes"][chunk_id]:
                raise ValueError("manifest embedding reference checksum mismatch")
            # Cosine storage may normalize float32 values. Verify against exact,
            # checksummed input references without requiring returned bit identity.
            # Differences <=1e-6 per component cannot be distinguished from
            # rounding by this policy. Larger changes must fail, even if unit norm.
            max_error = max(abs(actual - wanted) for actual, wanted in zip(row, reference))
            if max_error > VECTOR_COMPARISON["absolute_tolerance"]:
                raise ValueError(
                    f"stored embedding differs from manifest: chunk_id={chunk_id!r} "
                    f"max_abs_error={max_error:.12g} abs_tolerance=1e-6 rel_tolerance=0 "
                    f"stored_norm={math.hypot(*row):.12g} reference_norm={math.hypot(*reference):.12g}"
                )
        return client, collection, manifest
    except (KeyError, IndexError, TypeError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("corrupt persistent index or manifest") from exc


def retrieve_index(path, query_vector, k, *, model_id, model_revision):
    """Reload and validate, convert cosine distance to score, then use ID ties."""
    if not isinstance(k, int) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    if k == 0:
        return []
    _, collection, manifest = _load(path, model_id, model_revision)
    query = _vector(query_vector, manifest["dimension"])
    # Tiny original corpus only: request all rows so a cutoff cannot hide ID ties.
    result = collection.query(query_embeddings=[query], n_results=len(manifest["chunks"]), include=["distances"])
    ids, distances = result["ids"][0], result["distances"][0]
    if len(ids) != len(manifest["chunks"]) or set(ids) != {c["chunk_id"] for c in manifest["chunks"]}:
        raise ValueError("Chroma query returned incomplete or unknown IDs")
    if len(distances) != len(ids) or not all(math.isfinite(d) for d in distances):
        raise ValueError("Chroma returned malformed cosine distances")
    scored = sorted(zip(ids, distances), key=lambda item: (item[1], item[0]))
    return [Hit(chunk_id, max(-1.0, min(1.0, 1.0 - distance)), rank)
            for rank, (chunk_id, distance) in enumerate(scored[:k], 1)]


def langchain_retrieve(path, query, k, encoder, *, model_id, model_revision):
    """Use actual Chroma.as_retriever().invoke with query-only embeddings."""
    if not isinstance(k, int) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    if k == 0 or not query.strip():
        return []
    client, _, manifest = _load(path, model_id, model_revision)
    from langchain_core.embeddings import Embeddings
    from langchain_chroma import Chroma

    class QueryEmbedding(Embeddings):
        def embed_documents(self, texts):
            raise ValueError("corpus reembedding is forbidden for a persisted index")

        def embed_query(self, text):
            return _vector(encode_texts(encoder, [text])[0], manifest["dimension"])

    store = Chroma(collection_name=COLLECTION, client=client, embedding_function=QueryEmbedding(),
                   create_collection_if_not_exists=False)
    documents = store.as_retriever(search_kwargs={"k": min(k, len(manifest["chunks"]))}).invoke(query)
    expected = {c["chunk_id"]: c for c in manifest["chunks"]}
    seen = set()
    for doc in documents:
        chunk_id = doc.metadata.get("chunk_id")
        chunk = expected.get(chunk_id)
        if (chunk is None or chunk_id in seen or doc.page_content != chunk["text"]
                or doc.metadata != {key: chunk[key] for key in ("chunk_id", "doc_id", "start", "end")}):
            raise ValueError("LangChain returned corrupt or unknown provenance")
        seen.add(chunk_id)
    if len(documents) != min(k, len(expected)):
        raise ValueError("LangChain returned an unexpected document count")
    return documents
