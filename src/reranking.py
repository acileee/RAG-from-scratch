"""
WHAT: Module for re-scoring and reordering retrieved chunks using a combination of the
      original retrieval score and query-term overlap, then returning only the top-k
      after reranking.
WHY: First-stage retrieval (e.g. cosine similarity over embeddings) is fast but
     imprecise. Some high-scoring chunks may be thematically close but lack the exact
     concepts the user asked about. Reranking adds a lightweight second signal —
     term overlap — to surface chunks that are both semantically similar and literally
     relevant.
HOW:
  1. Tokenise the query into a set of meaningful words (stopwords stripped).
  2. For each retrieved chunk, tokenise its text and count how many query tokens appear.
  3. Add the overlap count to the original retrieval score to get a combined score.
  4. Sort all chunks by combined score descending and return the top-k.
EXAMPLE: A user asks "Python data structures". First-stage retrieval returns a chunk
         about "Python web frameworks" (score 0.82) and a chunk about "Python lists and
         dicts" (score 0.75). The reranker counts query-term overlaps: the web-frameworks
         chunk has 1 overlap (Python), the lists-and-dicts chunk has 3 overlaps (Python,
         data, structures). Combined scores flip the ranking so the more precisely
         relevant chunk rises to the top.
"""


def rerank_chunks(query: str, retrieved_chunks: list[dict], top_k: int = 3) -> list[dict]:
    """
    WHAT: Re-score a list of already-retrieved chunks by combining their original
          retrieval score with a query-term overlap count, then return the top-k.
    WHY: Initial retrieval (keyword or vector search) might not be perfect.
         Some chunks might be false positives (high score but not relevant).
         Adding an overlap signal penalises chunks that merely happen to be close
         in embedding space but do not actually mention the query concepts.
    HOW:
      1. Validate the query is non-empty.
      2. Tokenise the query with ``simple_tokenize`` to get a set of content words.
      3. For each chunk, tokenise ``chunk["text"]`` the same way.
      4. Compute ``overlap_score`` as the size of the intersection between query tokens
         and chunk tokens.
      5. Read ``chunk["score"]`` as the ``retrieval_score`` (defaults to 0 if absent).
      6. Set ``combined_score = overlap_score + retrieval_score``.
      7. Store all three scores on a copy of each chunk dict for transparency.
      8. Sort by ``rerank_score`` descending and return the first ``top_k`` entries.
    EXAMPLE:
      Query: "Python programming language"
      Query tokens: {python, programming, language}

      Chunk A text: "Python" — original score 0.95
        overlap: 1 token matched  →  combined = 1 + 0.95 = 1.95

      Chunk B text: "Learn Python programming efficiently" — original score 0.70
        overlap: 2 tokens matched  →  combined = 2 + 0.70 = 2.70  ← ranked higher

      Even though Chunk A had a stronger embedding score, Chunk B wins because it
      actually discusses "programming", which the query explicitly asked about.
    """
    if query.strip() == "":
        raise ValueError("Query cannot be empty or whitespace")

    # Extract tokens that matter from query
    query_tokens = simple_tokenize(query)

    scored_chunks = []
    for chunk in retrieved_chunks:
        # Extract meaningful tokens from chunk
        chunk_tokens = simple_tokenize(chunk["text"])

        # Signal 1: How many query tokens appear in this chunk?
        overlap_score = len(query_tokens.intersection(chunk_tokens))

        # Signal 2: What was the original retrieval score?
        retrieval_score = chunk.get("score", 0)

        # COMBINE: Higher is better - need both signals to score high
        combined_score = overlap_score + retrieval_score

        # Prepare result with all signals visible
        chunk_copy = chunk.copy()
        chunk_copy["original_score"] = retrieval_score
        chunk_copy["overlap_score"] = overlap_score
        chunk_copy["rerank_score"] = combined_score
        scored_chunks.append(chunk_copy)

    # Sort by combined score (highest first)
    sorted_chunks = sorted(scored_chunks, key=lambda chunk: chunk["rerank_score"], reverse=True)

    return sorted_chunks[:top_k]


def retrieve_then_rerank(query: str, retriever_fn, chunks: list[dict], retrieve_k: int = 10, final_k: int = 3,) -> list[dict]:
    """
    WHAT: Run first-stage retrieval to get a broad candidate set, then rerank those
          candidates and return a tighter, higher-quality top-k list.
    WHY: Reranking over a large initial pool (e.g. top-10) is more effective than
         reranking over a small one (e.g. top-3), because the better chunk might rank
         4th by embedding similarity but 1st after overlap scoring. This two-stage
         approach balances recall (broad retrieval) with precision (tight reranking).
    HOW:
      1. Call ``retriever_fn(query, chunks, top_k=retrieve_k)`` to get a broad candidate
         set of ``retrieve_k`` chunks sorted by embedding similarity.
      2. Pass those candidates to ``rerank_chunks`` which applies the overlap signal and
         returns the best ``final_k`` chunks.
      3. Return the reranked list directly.
    EXAMPLE:
      >>> from semantic_search import semantic_search
      >>> results = retrieve_then_rerank(
      ...     "knowledge graph entities",
      ...     semantic_search,
      ...     embedded_chunks,
      ...     retrieve_k=10,
      ...     final_k=3
      ... )
      # Stage 1: semantic_search returns the 10 most embedding-similar chunks.
      # Stage 2: rerank_chunks scores each for overlap with {"knowledge","graph","entities"}
      #          and returns the 3 chunks that best match on both signals.
    """
    retrieved_chunks = retriever_fn(query, chunks, top_k=retrieve_k)
    return rerank_chunks(query, retrieved_chunks, top_k=final_k)


if __name__ == "__main__":
    test_chunks = [
        {"chunk_id": "1", "text": "Python is a programming language", "start_char": 0, "end_char": 32, "doc_id": "doc1", "filename": "file1.txt", "score": 0.7},
        {"chunk_id": "2", "text": "Python uses indentation for code blocks", "start_char": 32, "end_char": 70, "doc_id": "doc1", "filename": "file1.txt", "score": 0.6},
        {"chunk_id": "3", "text": "Learning Python improves programming skills", "start_char": 70, "end_char": 113, "doc_id": "doc2", "filename": "file2.txt", "score": 0.5},
        {"chunk_id": "4", "text": "Java is also a programming language", "start_char": 113, "end_char": 148, "doc_id": "doc3", "filename": "file3.txt", "score": 0.4},
    ]

    print("Testing simple_tokenize:")
    tokens = simple_tokenize("Python programming skills")
    print(f"  Tokens: {tokens}")

    print("\nTesting rerank_chunks:")
    query = "Python programming"
    results = rerank_chunks(query, test_chunks, top_k=2)
    print(f"  Query: '{query}'")
    print(f"  Found {len(results)} top results")
    for r in results:
        print(f"    - {r['chunk_id']}: rerank_score={r['rerank_score']}, overlap={r.get('overlap_score', 0)}, original={r.get('original_score', 0)}")

    print("\nTesting error handling:")
    try:
        rerank_chunks("", test_chunks)
        print("  ERROR: Should raise ValueError for empty query")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
