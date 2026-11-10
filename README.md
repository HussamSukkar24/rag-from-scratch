# RAG from Scratch

An original implementation following [TensorTonic's RAG project](https://www.tensortonic.com/projects/rag-from-scratch).

## The process

This was my first RAG project. The goal was simple: give it some documents, ask a question, and get an answer back that points to exactly where in the documents it came from. If the answer isn't in there, it should say so instead of making something up.

**What it does**

1. Splits documents into small chunks and remembers where each chunk sits in the original text.
2. Turns each chunk into an embedding, a list of numbers that captures its meaning, so similar ideas end up close together even when the wording is different.
3. Stores those embeddings in Chroma, so the index can be saved and reopened later.
4. When a question comes in, finds the closest chunks and passes them to a question-answering model.
5. The model picks an exact piece of text as the answer, and the project checks that the citation really matches the original source, character for character.

You can point it at your own documents and use three commands: `index` to build it, `query` to ask a question and `inspect` to see the source behind a citation.

**What stood out**

- A correct citation doesn't mean a correct answer. It only proves where the text came from. The loan example shows this: the citation is valid, but the answer is "fourteen" instead of "fourteen days".
- Saying "I don't know" is a feature. When the evidence is missing, or two sources disagree, it abstains instead of guessing.
- I didn't just lower the threshold to make that example pass. The model does see "fourteen days" as an option, but only scores it 0.15. Lowering the bar for one test question would just be cheating the test.
- Small test sets prove the pipeline works. They don't prove it's accurate in general.

**What I'd do next**

- Try a generative model that writes a full answer, and compare it with this extractive approach.
- Build a bigger evaluation set so the accuracy numbers actually mean something.
- Add PDF support and let the index update without rebuilding it from scratch.

Everything below is the detailed record: the exact runs, test results and evidence behind each step.

## Details

### Setup

Use a clean Linux Python environment. The recorded runs used an isolated Colab CPU runtime with Python 3.13.18. Install CPU torch 2.8.0 from the official CPU wheel index, then the pinned lock file, then run `pip check`:

```sh
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-grounded-lock.txt
pip check
```

`requirements-grounded.txt` lists the direct dependencies and `requirements-grounded-lock.txt` pins the full observed environment for the current workflow. The semantic and persistent requirement and lock files pin earlier stages and should not be combined with the grounded lock. A different Python version or operating system still needs its own compatibility check. [docs/USAGE.md](docs/USAGE.md) gives the full walkthrough.

### Commands

```sh
python -m rag_baseline.project_cli index --documents /content/custom-documents.json --index /content/custom-index --size 200 --overlap 0
python -m rag_baseline.project_cli query --index /content/custom-index --question "When does the orchard pavilion close?" --k 2 --min-score 0.5
python -m rag_baseline.project_cli inspect --index /content/custom-index --chunk-id "orchard:0:37"
```

The input is `{"documents":[{"doc_id":"orchard","text":"The orchard pavilion closes at 17:40."}]}`. Use your own permitted source text and inspect the chunk IDs returned by indexing rather than assuming this example ID. `index` requires a new directory; adding or changing documents means building a new index.

### Verified results

Verification ran in an isolated Colab CPU runtime. Each result below comes from a recorded receipt.

- Workflow at `6a9420d`: four CLI tests passed. Three documents, one longer than 200 characters, produced four chunks. Fresh processes reloaded the index, answered `17:40` with a matching original source slice, abstained on an absent designer fact, inspected a stored chunk and rejected an unknown ID. See [the workflow receipt](results/project-6a9420d-receipt.json).
- Question answering at `699b06b`: 19 targeted tests passed (1 QA compatibility, 13 grounding, 5 conflict policy), followed by the unchanged six-query evaluation with threshold 0.5 and retrieval k 2. All four held-out queries matched their desired outcome: both answers and both abstentions. Citation checks passed for all six queries. The development loan answer is `fourteen` rather than `fourteen days`; the model ranks `fourteen days` second at 0.151, below the 0.5 threshold, so the shorter span is kept. See [the QA receipt](results/grounded-699b06b-receipt.json).

Commit IDs in this README, the docs and the receipts refer to the development history, which is not part of this repository.

These are tiny fixtures. They show that the pipeline works end to end; they do not establish general retrieval quality or answer accuracy.

### Limitations

- Extractive answers can be incomplete even when the citation is valid. [docs/EXTRACTION_LIMITATION.md](docs/EXTRACTION_LIMITATION.md) traces the loan example in detail.
- Input is UTF-8 JSON text only: no PDFs, directory ingestion or web scraping.
- There are no incremental updates; changing the corpus means rebuilding the index.
- Character-based chunks can split facts across chunk boundaries.
- Abstention rules reduce unsupported answers but do not prove semantic truth or detect every contradiction.
