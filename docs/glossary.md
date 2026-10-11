# Glossary — RAG lab (in your words)

- **Knowledge base**: the pile of files you drag-drop that the app can later chat about.
- **Ingest**: taking a dropped file, pulling text + structure out of it, cutting it into chunks, embedding it, storing it.
- **Structure-aware chunking**: cutting on real boundaries (headings, pages, sections) instead of blind fixed-size slices, keeping title + heading path with each chunk.
- **Embed**: turning a chunk into numbers (vector) so similar meanings sit near each other.
- **Hybrid search (BM25 + vector)**: keyword search plus meaning search combined (with reciprocal-rank fusion) so exact terms and paraphrases both hit.
- **Rerank**: pulling a wide cheap candidate set (e.g. top 30–80), then re-scoring with a slower accurate model down to the best 5–10 before the LLM sees them.
- **Citations**: the answer points at the chunks it used, so you can tell retrieval from hallucination.
- **RAGAS eval**: a small scored test set over your own docs that measures whether the right chunks were retrieved and the answer stayed grounded.
- **Observability**: traces per question — query, retrieved chunks + scores, rerank order, final prompt — so you can see where it broke.
- **Blueprint (NVIDIA)**: the enterprise reference (NIMs, Helm, Milvus/Elasticsearch) we borrow ideas from, not deploy as-is.
