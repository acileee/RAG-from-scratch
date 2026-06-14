"""
WHAT: Provides a structured debugging utility for inspecting retrieval results.
WHY: When a retriever returns wrong or unexpected chunks it is hard to diagnose
     the problem from raw lists alone. A structured report with ranks, previews,
     scores, and a pass/fail flag makes debugging much faster.
HOW: Defines debug_retrieval() which runs any retriever function, formats the
     results into ranked rows with text previews and scores, checks whether the
     expected document was retrieved, and returns a single report dict.
EXAMPLE: report = debug_retrieval("what is RAG?", semantic_search,
             embedded_chunks, expected_doc_id="rag", top_k=3)
         # report["success"] is True if the rag chunk appeared in the top 3,
         # and report["results"] shows rank/score for each returned chunk.
"""


def debug_retrieval(query: str, retriever_fn, chunks: list[dict], expected_doc_id: str | None = None, top_k: int = 3) -> dict:
    """
    WHAT: Runs a retriever and returns a structured debug report for one query.
    WHY: During development it is common to find that a retriever silently returns
         the wrong chunks for a specific query. This function surfaces the exact
         rank, chunk_id, doc_id, score, and text preview so the failure is
         immediately visible without writing ad-hoc print statements.
    HOW: 1. Calls retriever_fn(query, chunks, top_k) to get results.
         2. Iterates over results, building a result_rows list with rank, chunk_id,
            doc_id, a 120-char text preview, and the relevance score.
         3. Determines success: if expected_doc_id is provided, checks whether it
            appears in the returned doc_ids; if None, checks that nothing was
            returned (correct behaviour for unanswerable queries).
         4. Returns a report dict containing all of the above.
    EXAMPLE: report = debug_retrieval(
                 "what do knowledge graphs store?",
                 semantic_search, embedded_chunks,
                 expected_doc_id="kg", top_k=3)
             for row in report["results"]:
                 print(f"Rank {row['rank']}: {row['doc_id']} (score={row['score']:.3f})")
             # Should show kg at rank 1 with the highest score.
    """
    results = retriever_fn(query, chunks, top_k=top_k)
    returned_doc_ids = []
    returned_chunk_ids = []
    result_rows = []
    for i, chunk in enumerate(results):
        returned_doc_ids.append(chunk["doc_id"])
        returned_chunk_ids.append(chunk["chunk_id"])
        rank = i + 1
        text_preview = chunk["text"][:120] + "..." if len(chunk["text"]) > 120 else chunk["text"]
        result_rows.append({"rank": rank, "chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"], "text_preview": text_preview, "score": chunk.get("score", None)})
    if expected_doc_id is not None:
        success = expected_doc_id in returned_doc_ids
    else:
        success = len(returned_doc_ids) == 0
    report = {"query": query,
        "expected_doc_id": expected_doc_id,
        "success": success,
        "top_k": top_k,
        "returned_doc_ids": returned_doc_ids,
        "returned_chunk_ids": returned_chunk_ids,
        "results": result_rows,}
    return report


if __name__ == "__main__":
    test_chunks = [
        {"chunk_id": "1", "text": "Python is awesome", "start_char": 0, "end_char": 17, "doc_id": "doc1", "filename": "f1.txt", "score": 0.9},
        {"chunk_id": "2", "text": "Java is also great", "start_char": 17, "end_char": 36, "doc_id": "doc2", "filename": "f2.txt", "score": 0.7},
    ]

    def dummy_retriever(q, chunks, top_k=3):
        return chunks[:top_k]

    print("Testing debug_retrieval:")
    query = "Python test"
    report = debug_retrieval(query, dummy_retriever, test_chunks, expected_doc_id="doc1", top_k=2)

    print(f"  Query: '{query}'")
    print(f"  Success: {report['success']}")
    print(f"  Found doc_ids: {report['returned_doc_ids']}")
    print(f"  Results:")
    for r in report['results']:
        print(f"    - Rank {r['rank']}: {r['chunk_id']} (score={r['score']})")
