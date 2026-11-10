# Incomplete development extraction

The recorded run at `fb17bc13972c2f64d5a58c310f7e9a699eb4e00d`
returned `fourteen` for `dev-loan`, whose unchanged expected answer is
`fourteen days`. Evidence: [full receipt](../results/grounded-fb17bc1-receipt.json),
under `manifest.results`, query ID `dev-loan`.

The source is `The Cedar archive lends maps for fourteen days.` Retrieval placed
the correct source first, with recall 1 and reciprocal rank 1. The recorded QA
responses were:

| Context | Answer | Score | Backend span |
| --- | --- | --- | --- |
| Combined retrieved context | `fourteen` | 0.5306715965270996 | [53,61) |
| Original loan source body | `fourteen` | 0.311175137758255 | [33,41) |
| Other retrieved source body | empty | 0.9971749782562256 | [0,0) |

The combined context's source body starts at character 20. Subtracting that
header offset maps [53,61) to the unchanged original source span [33,41), exactly
`fourteen`. The fuller expected phrase occupies [33,46). Both citation checks
passed; strict answer matching correctly recorded an answer-stage failure.

The combined score exceeds the fixed 0.5 threshold. The isolated loan response
is below it, and the other response is empty, so neither supplies a competing
validated answer. The disagreement check does not replace the combined answer.

No application defect is supported by this trace. `load_qa` uses the official
extractive pipeline; `answer_question` and `_validated_answer` preserve the
backend's exact text and offsets. They do not trim units or set a custom
answer-length cap; backend defaults remain in effect.
The trace establishes a short decoded pipeline span. It does not isolate model
logits from the pipeline's decoding contribution. Appending nearby words would
change the model-selected span and attach its score to different text without
evidence. Product code, threshold, k, model revisions and gold labels remain
unchanged. The incomplete development answer remains a documented limitation.

## Ranked candidate follow-up at `699b06b`

`answer_question` now requests five ranked spans from the combined-context call.
A lower-ranked span replaces the top span only if it passes the same threshold,
offset and source checks, lies in the same chunk and strictly contains the top
span. Its own model score travels with it. No text is appended.

The Colab gate at `699b06b83db2486ad420fa5aa007ad5500abb4e4` passed 19 targeted
tests (1 QA compatibility, 13 grounding, 5 conflict) and the unchanged six-query
evaluation. Evidence: [receipt](../results/grounded-699b06b-receipt.json).

| Combined-context candidate | Score | Span |
| --- | --- | --- |
| `fourteen` | 0.5306715965270996 | [53,61) |
| `fourteen days` | 0.15134599804878235 | [53,66) |
| empty | 0.02333342656493187 | [0,0) |
| `fourteen days.` | 0.004181886091828346 | [53,67) |
| `fourteen days.` plus the next header | 0.0019210769096389413 | [53,77) |

The model does rank the full phrase second, but well below the fixed 0.5
threshold, so the answer correctly remains `fourteen`. The top span and score are
identical to the `fb17bc1` run. Held-out outcomes are unchanged at 4/4 and
citation checks passed for every query. Lowering the threshold or adding a
ratio rule for this one development example would be tuning to the fixture, so
neither was done. The limitation stands; the candidate rule only helps when the
model is itself confident in the longer span.

## Verification and practical input limits

This diagnosis reads existing evidence and source only. If extraction code
later changes, the minimum relevant checks are `test_grounded.py`,
`test_grounded_conflicts.py` and the unchanged fixed six-query CLI evaluation.
Preserve raw spans and responses; do not tune held-out labels or append units to
obtain a match.

The existing persistent and QA CLIs accept an external `--fixture` JSON path,
but these are evaluation inputs. They require relevance labels, and QA also
requires development/evaluation queries and expected answers. Documents must be
nonempty and at most 200 characters, giving one chunk each. Both CLIs create a
new index and refuse an existing index path. Their JSON receipts expose stored
chunk text, IDs, source offsets, retrieved documents and QA traces.

The library already supported validated index reload, vector queries, actual
LangChain retrieval and exact-span QA. The practical gap identified during this
diagnosis was unlabelled document input plus an independent question against an
existing index, with explicit source-span inspection. The subsequent
`rag_baseline.project_cli` implements that small contract without changing the
extractor or adding a service or model architecture. Its index/query/inspect
workflow was verified at `6a9420d`; see `docs/USAGE.md` and the recorded workflow
receipt. It supports UTF-8 JSON text, not arbitrary external file formats.
