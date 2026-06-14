"""
WHAT: Module for improving retrieval recall by generating multiple variants of a user
      query and merging the results into a single deduplicated ranked list.
WHY: A single query phrasing may miss relevant chunks because the user's words do not
     exactly match the document's vocabulary. Generating several semantically equivalent
     variants and merging their results increases the chance that every relevant chunk
     surfaces at least once.
HOW:
  1. Take the original query and produce up to three variants: the original, an
     abbreviation-expanded form (via query_rewriting), and a stopword-stripped form.
  2. Run the retriever against every variant and collect scored chunk results.
  3. For each chunk, keep only the highest score seen across all variants.
  4. Sort the deduplicated pool by score and return the top ``final_top_k`` chunks.
EXAMPLE: A user asks "how does LLM generation work?". The three variants become:
         (1) "how does LLM generation work?" — original
         (2) "how does large language model generation work?" — expanded
         (3) "llm generation work" — stopwords removed
         Running all three against the corpus and merging their hits gives broader
         coverage than any single variant alone.
"""

from query_rewriting import rewrite_query


def generate_query_variants(query: str) -> list[str]:
    """
    WHAT: Produce up to three distinct phrasings of a query — original, abbreviation-
          expanded, and stopword-stripped — deduplicated so no variant appears twice.
    WHY: Different phrasings retrieve different chunks. The original preserves user
         intent; the expanded form matches documents that spell out abbreviations; the
         stripped form focuses on content words and can match chunks where filler words
         differ. Together they widen retrieval coverage.
    HOW:
      1. Validate that the query is not empty or whitespace-only.
      2. Strip leading/trailing whitespace to get the canonical original form.
      3. Run ``rewrite_query`` on the original to get the abbreviation-expanded variant.
      4. Build the stopword-stripped variant by lower-casing the original, splitting on
         spaces, filtering out a predefined set of common English stopwords, and
         re-joining the remaining words.
      5. Collect all three variants into a list, then iterate through them in order and
         append each to a new list only if it has not been added yet (preserves order,
         removes exact duplicates).
      6. Return the deduplicated list.
    EXAMPLE:
      >>> generate_query_variants("What is RAG?")
      ["What is RAG?",
       "what is retrieval augmented generation?",
       "rag"]
      # The first two differ, so both are kept.
      # "rag" is the stopword-stripped form with only content words.
    """
    if query.strip() == "":
        raise ValueError("Query cannot be empty or whitespace")
    original_query = query.strip()
    rewritten_query = rewrite_query(original_query)
    stopwords = {"the", "is", "in", "and", "to", "of", "a", "or", "an", "for", "with", "on", "as", "by", "does", "do", "what", "how"}
    words = original_query.lower().split()
    simple_words = []
    for word in words:
        if word not in stopwords:
            simple_words.append(word)
    simple_query = " ".join(simple_words)
    variants = [original_query, rewritten_query, simple_query]
    unique_variants = []
    for variant in variants:
        if variant not in unique_variants:
            unique_variants.append(variant)
    return unique_variants


def multi_query_search(query: str, retriever_fn, chunks: list[dict], top_k_per_query: int = 3, final_top_k: int = 5,) -> list[dict]:
    """
    WHAT: Search a chunk collection using multiple query variants and return a merged,
          deduplicated list of the highest-scoring chunks across all variants.
    WHY: Any single query variant can produce false negatives — chunks that are relevant
         but happen not to match that particular phrasing. Searching with multiple
         variants and keeping the best score per chunk maximises recall without
         returning duplicates.
    HOW:
      1. Call ``generate_query_variants`` to get up to three query phrasings.
      2. For each variant, call ``retriever_fn(variant, chunks, top_k=top_k_per_query)``
         to get that variant's top results.
      3. Maintain a dict keyed by ``chunk_id``. For each returned chunk, store it only
         if its score exceeds whatever score is already recorded for that chunk id
         (keeps the best-scoring version of each chunk across all variants).
      4. Sort the deduplicated dict values by ``rerank_score`` descending and slice
         to ``final_top_k``.
      5. Return the final ranked list.
    EXAMPLE:
      >>> from semantic_search import semantic_search
      >>> results = multi_query_search(
      ...     "How does RAG retrieve context?",
      ...     semantic_search,
      ...     embedded_chunks,
      ...     top_k_per_query=3,
      ...     final_top_k=5
      ... )
      # Variant 1 ("How does RAG retrieve context?") finds chunks 1, 4, 7
      # Variant 2 ("how does retrieval augmented generation retrieve context?") finds chunks 2, 4, 9
      # Variant 3 ("rag retrieve context") finds chunks 4, 7, 11
      # Merged + deduped + sorted: chunks [4, 1, 9, 7, 2] (best scores win)
    """
    query_variants = generate_query_variants(query)
    best_by_chunk_id = {}
    for variant in query_variants:
        results = retriever_fn(variant, chunks, top_k=top_k_per_query)
        for chunk in results:
            chunk_id = chunk["chunk_id"]
            score = chunk.get("score", 0)
            if chunk_id not in best_by_chunk_id:
                best_by_chunk_id[chunk_id] = chunk
            elif score > best_by_chunk_id[chunk_id].get("score", 0):
                best_by_chunk_id[chunk_id] = chunk
    sorted_chunks = sorted(best_by_chunk_id.values(), key=lambda x: x.get("score", 0), reverse=True)
    return sorted_chunks[:final_top_k]


def debug_multi_query_search(query: str, retriever_fn, chunks: list[dict], expected_doc_id: str | None = None, top_k_per_query: int = 3, final_top_k: int = 5) -> dict:
    """
    WHAT: Run multi-query search and return a structured report that includes the query
          variants used, the merged results, and a success flag indicating whether an
          expected document appeared in the output.
    WHY: When tuning retrieval it is useful to see exactly which variants were generated
         and which chunks each one contributed. The success flag lets automated tests
         assert that a known-relevant document was retrieved.
    HOW:
      1. Call ``generate_query_variants`` to capture the variants that will be used.
      2. Call ``multi_query_search`` with the same parameters to get merged results.
      3. Extract ``doc_id`` and ``chunk_id`` from every result for easy inspection.
      4. Set ``success`` to True if ``expected_doc_id`` appears in the returned doc ids,
         or to True vacuously (no expectation, no results) when no expectation is given.
      5. Return all of the above as a single dict.
    EXAMPLE:
      >>> report = debug_multi_query_search(
      ...     "explain LLM grounding",
      ...     semantic_search,
      ...     embedded_chunks,
      ...     expected_doc_id="doc_rag_overview",
      ...     top_k_per_query=3,
      ...     final_top_k=5
      ... )
      >>> report["variants"]
      ["explain LLM grounding",
       "explain large language model grounding",
       "explain llm grounding"]
      >>> report["success"]
      True   # "doc_rag_overview" was in the top-5 merged results
    """
    variants = generate_query_variants(query)
    results = multi_query_search(query, retriever_fn, chunks, top_k_per_query=top_k_per_query, final_top_k=final_top_k)
    returned_doc_ids = [result["doc_id"] for result in results]
    returned_chunk_ids = [result["chunk_id"] for result in results]
    if expected_doc_id is not None:
        success = expected_doc_id in returned_doc_ids
    else:
        success = len(returned_doc_ids) == 0
    return {
        "query": query,
        "variants": variants,
        "results": results,
        "returned_doc_ids": returned_doc_ids,
        "returned_chunk_ids": returned_chunk_ids,
        "success": success
    }

if __name__ == "__main__":
    print("Testing generate_query_variants:")
    test_queries = [
        "what is Python",
        "large language models",
    ]

    for query in test_queries:
        variants = generate_query_variants(query)
        print(f"  Query: '{query}'")
        print(f"  Variants ({len(variants)}): {variants}")

    print("\nTesting error handling:")
    try:
        generate_query_variants("")
        print("  ERROR: Should raise ValueError for empty query")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
