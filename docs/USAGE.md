# Use the RAG project

The index/query/inspect workflow was verified in a clean isolated Colab CPU
environment at `6a9420d`: four CLI tests, three unlabelled documents,
cross-process reload, two questions, exact original source slices, stored-chunk
inspection and unknown-ID rejection. See `../results/project-6a9420d-receipt.json`.
That run added a longer third document to the two examples below; this guide
does not claim every snippet was executed literally.

The development loan answer remains `fourteen` rather than the expected
`fourteen days`. Source offsets and citations are correct; this is a
completeness limitation, not evidence that citation checks prove answer
quality. See `EXTRACTION_LIMITATION.md`.

## Prepare a clean isolated environment

Clone the repository into a fresh Linux environment (the recorded runs used
Colab at `/content`) and record the revision:

```python
%cd /content/rag-from-scratch
!git rev-parse HEAD
```

The QA lock was captured on Python 3.13.18. A different runtime still requires
compatibility checks. Do not combine this lock with the older semantic lock.

```python
import subprocess
import sys

subprocess.run([sys.executable, "-m", "venv", "--without-pip", "/content/rag-env"], check=True)
python = "/content/rag-env/bin/python"
pip = [sys.executable, "-m", "pip", "--python", python]
subprocess.run(pip + ["install", "torch==2.8.0", "--index-url",
                     "https://download.pytorch.org/whl/cpu"], check=True)
subprocess.run(pip + ["install", "--extra-index-url", "https://download.pytorch.org/whl/cpu",
                     "-r", "requirements-grounded-lock.txt"], check=True)
subprocess.run(pip + ["check"], check=True)
with open("/content/observed-freeze.txt", "x") as log:
    subprocess.run(pip + ["freeze"], stdout=log, check=True)
```

The snippets below are interactive commands. Preserve their output and any
failure logs. A successful import alone does not verify models, persistence,
LangChain or answer quality.

## Add original documents and create a new index

Create unlabelled JSON outside the repository. These are original example facts;
replace them with text you may process. Questions and gold answers are not needed.

```python
%%writefile /content/custom-documents.json
{
  "documents": [
    {"doc_id": "orchard", "text": "The orchard pavilion closes at 17:40.", "source": "original:orchard"},
    {"doc_id": "studio", "text": "The pottery studio lends aprons for three days.", "source": "original:studio"}
  ]
}
```

Run the package's project CLI using the isolated interpreter:

```python
! /content/rag-env/bin/python -m rag_baseline.project_cli index --documents /content/custom-documents.json --index /content/custom-index --size 200 --overlap 0
```

`index` requires a new directory. Documents need unique nonempty string IDs,
nonempty text and an optional string `source`. Longer documents are supported:
chunk size and overlap count Python characters, with half-open source offsets.
For example, choose `--size 200 --overlap 20` when intentionally testing overlapping
chunks. Small character chunks can split facts; they are not token-aware chunks.

The command uses the pinned real embedding model, finite normalised vectors and
persistent Chroma. Keep exact source text so citation offsets can be inspected.
The manifest hashes supplied chunk records, not an unseen original document.
`source` is an input description; inspect output identifies the stored chunk's
source document and offsets, not a fetched external resource.

## Reopen, query and inspect evidence

Each invocation below starts a new process and reloads the existing index.
`query` encodes the question, not the stored corpus. Actual LangChain-selected
chunks supply the answer context; cosine scores are matched by chunk ID even
when tied retrieval order differs.

```python
from pathlib import Path
import json
import subprocess

base = [python, "-m", "rag_baseline.project_cli"]
question = "When does the orchard pavilion close?"
result = subprocess.run(base + ["query", "--index", "/content/custom-index",
    "--question", question, "--k", "2", "--min-score", "0.5", "--max-chars", "2000"],
    text=True, capture_output=True)
with Path("/content/custom-query.stderr.log").open("x", encoding="utf-8") as log:
    log.write(result.stderr)
with Path("/content/custom-query.json").open("x", encoding="utf-8") as output:
    output.write(result.stdout)
result.check_returncode()
record = json.loads(result.stdout)
print(json.dumps(record, indent=2))
```

Inspect the selected text, document IDs, original character offsets, cosine
scores and returned citations. JSON also records the model revisions and command
configuration. A command failure must return nonzero; do not treat partial stdout
as a successful answer.

For an answered question, inspect each cited stored chunk and independently slice
the original source:

```python
sources = {row["doc_id"]: row["text"] for row in
           json.loads(Path("/content/custom-documents.json").read_text())["documents"]}
for citation in record["answer"]["citations"]:
    subprocess.run(base + ["inspect", "--index", "/content/custom-index",
                           "--chunk-id", citation["chunk_id"]], check=True)
    quoted = sources[citation["doc_id"]][citation["start"]:citation["end"]]
    assert quoted == citation["quote"]
    print(citation["doc_id"], citation["start"], citation["end"], repr(quoted))
```

`inspect` validates the stored index and rejects unknown IDs without downloading
models. It reports stored source spans; it cannot authenticate an external full
document that was never supplied to this API.

Try an absent fact in a separate process and keep its actual result:

```python
with Path("/content/custom-missing.json").open("x", encoding="utf-8") as output:
    subprocess.run(base + ["query", "--index", "/content/custom-index",
        "--question", "Who designed the orchard pavilion?", "--k", "2",
        "--min-score", "0.5"], stdout=output, check=True)
```

Nearest-neighbour retrieval always has candidates in a nonempty index. It does
not establish answerability. Invalid spans, empty answers and low confidence
abstain; differing valid per-source answers can also cause abstention. None of
these rules proves semantic truth or detects every contradiction. The system
returns extracted source text rather than appending an unsupported unit.

## Change the corpus and rebuild

Edit or add documents in a copy of the JSON. Keep document IDs stable when they
still identify the same source, and retain the original JSON version separately.
Build a **new** index, then point subsequent queries at it:

```python
! /content/rag-env/bin/python -m rag_baseline.project_cli index --documents /content/custom-documents-v2.json --index /content/custom-index-v2 --size 200 --overlap 0
```

Create `custom-documents-v2.json` before this command. Rebuilding is the supported
update workflow; there is no incremental add/delete operation or overwrite flag.
An old index remains an old corpus snapshot even if the external source file
changes. Do not claim its citations refer to the edited source. Model identity,
revision, dimension or integrity mismatches must fail rather than silently rebuild.

The project CLI accepts UTF-8 JSON, not PDFs, directory ingestion or network
scraping. The separate `grounded_cli --fixture` is for labelled development and
evaluation data and currently restricts each document to one chunk of at most
200 characters. That evaluation restriction does not apply to `project_cli index`.
Meaningful evaluation of longer text needs independent evidence-span labels.

Exercise: edit the studio duration in a new corpus version. Compare source slices
from the old and rebuilt indexes, then explain why a valid citation is not a check
that its answer is complete.
