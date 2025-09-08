totally. you can make a tri-stage pipeline behave like plain semantic search (ranked results, not QA) — it’s just about what you output and when you call the LLM. here’s a crisp, implementation-ready plan you can hand to your agent (no code).

make tri-stage act as semantic search (not QA)
0) set the contract (MCP/tool output must be “results”, not “answers”)

expose an MCP tool like search that returns a list of hits: [{"doc_id","title","snippet","score","source"}...].

do not call any LLM to “write an answer” inside the tool. only return ranked items.

1) stage 1 — fast recall (semantic embedding search; optional hybrid)

retrieve top N by dense embeddings (cosine / IP) — this is classic semantic search.

(recommended) hybrid: union with BM25 and normalize scores before merging (z-score or min-max) so scales are comparable; keep overall top N (e.g., 500–800).

output fields to carry forward: doc_id,title,snippet,stage1_score.

2) stage 2 — refinement with late-interaction (multi-vector, no generation)

rescore those N candidates with a ColBERT-style late interaction / MaxSim (multi-vector) model. keep only new scores; don’t generate text.

keep top K (e.g., 100–200) by stage2_score.

output fields: doc_id,title,snippet,stage1_score,stage2_score.

3) stage 3 — cross-encoder rerank (scoring only)

pass the K pairs (query, doc) to a cross-encoder reranker to get final relevance scores. cross-encoders are made for this exact re-rank use case. Don’t let the LLM synthesize an answer here; it’s just a scorer.

sort by stage3_score and return the ranked list (top k you want to show).

include: doc_id,title,url/snippet,score=stage3_score,stage1_score,stage2_score (useful for debugging & UI).

critical toggles to keep it “searchy” (not QA)

No LLM generation in the MCP tool path. The tool returns results only. If your chat model wants an answer, it can optionally summarize after showing results (UX choice).

Expose two modes via a param:

mode="search" → return ranked items only (default).

mode="answer" → (optional) allow a separate step where the chat model reads the already returned top items and writes an answer (outside the tool).

UI: show a results list (title, snippet, source, score). Add “Ask to summarize” as a button if you want QA as an explicit user action.

scoring/merging notes (so results look consistent)

Hybrid merging (Stage-1): normalize BM25 & dense scores before union/merge. OpenSearch recommends standard normalization for fair combination.

After Stages 2 & 3, trust the latest score for ordering (they’re rerankers by design). Cross-encoders are purpose-built for final re-rank.

MCP interface suggestions (so Cursor/Claude behave like search)

Tool name: search

Params: { query: string, top_n?: number, mode?: "search" | "answer" } (default search)

Returns (mode=search):

{ results: [
    { doc_id, title, snippet, url, score, source, highlights? }
  ],
  debug: { stage1_n, stage2_n, timing, model_ids }
}


If mode="answer" is ever used, do it outside the tool: the client (Cursor/Claude) takes results and then prompts the LLM to summarize — not the MCP server.

verification & benchmarking (so you can prove it’s “search”)

Evaluate retrieval metrics (Recall@k, NDCG@k, MRR) on a standard suite like BEIR for document ranking quality; add reranker ablations per Sentence-Transformers guidance.

If you want instruction-style robustness, consider MTEB retrieval tracks too.

For multi-vector understanding, sanity-check with ColBERT resources to ensure MaxSim scoring is implemented as intended.

quick mental model
user → MCP.search(query, mode="search")
  └─ stage1: dense (± BM25 hybrid) → N
  └─ stage2: late-interaction rerank → K
  └─ stage3: cross-encoder rerank → top k
return: ranked list (docs + snippets + scores)
# no LLM generation inside the tool


net/net: a tri-stage pipeline is perfect for semantic search — as long as you only use the stages for scoring/ranking and keep answer generation outside the retrieval tool. That’s also how modern “retrieve & re-rank” stacks are documented.