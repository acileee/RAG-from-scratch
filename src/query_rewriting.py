"""
WHAT: Module for rewriting user queries before retrieval to improve the chance of finding
      relevant document chunks.
WHY: Users often write abbreviated or informal queries (e.g. "what is RAG?") while the
     documents use full forms (e.g. "retrieval augmented generation"). Expanding
     abbreviations before searching bridges that vocabulary gap and increases recall.
HOW:
  1. Define a lookup table of common RAG/AI abbreviations mapped to their full forms.
  2. Normalise the raw query (lowercase, strip extra whitespace).
  3. Replace each word that matches an abbreviation with its expansion.
  4. Expose helper functions that combine rewriting with retrieval and optional debugging.
EXAMPLE: A user asks "how do LLMs use KGs?". After rewriting the query becomes
         "how do large language models use knowledge graphs?" which matches document
         chunks that were written with the full terminology.
"""


def query_rewriting(query: str) -> str:
    """
    WHAT: Expand common domain abbreviations in a query string and normalise its
          whitespace and casing so downstream retrieval can match more document chunks.
    WHY: Retrieval systems compare query text against indexed chunk text. When users
         type abbreviations like "LLM" but documents contain "large language model",
         the two strings will not match even though they mean the same thing. Expanding
         abbreviations before retrieval closes this gap.
    HOW:
      1. Reject empty or whitespace-only input immediately with a ValueError.
      2. Strip leading/trailing whitespace and convert the query to lowercase so
         comparisons are case-insensitive.
      3. Collapse any runs of multiple spaces into a single space.
      4. Split the normalised query into individual words.
      5. For each word, check the abbreviation lookup table; replace it with the full
         form if found, or keep the original word otherwise.
      6. Re-join the words and return the expanded query string.
    EXAMPLE:
      Input:  "What is  LLM and RAG?"
      Step 2: "what is  llm and rag?"
      Step 3: "what is llm and rag?"
      Step 5: "what is large language model and retrieval augmented generation?"

      Searching with the expanded form now retrieves chunks about
      "large language model" and "retrieval augmented generation" that the
      original short query would have missed.
    """
    expand_common_abbreviations = {
        "llm": "large language model",
        "rag": "retrieval augmented generation",
        "kg": "knowledge graph",
        "ai": "artificial intelligence",
        "llms": "large language models",
        "kgs": "knowledge graphs"
    }

    if query.strip() == "":
        raise ValueError("Query cannot be empty or whitespace")

    # Normalize the query
    rewritten_query = query.strip().lower()

    # Remove extra whitespace
    if "  " in rewritten_query:
        rewritten_query = " ".join(rewritten_query.split())

    # Split into words to check each one
    rewritten_query_split = rewritten_query.split()
    rewritten_words = []

    # Replace abbreviations with full forms
    for word in rewritten_query_split:
        expanded_word = expand_common_abbreviations.get(word, word)
        rewritten_words.append(expanded_word)

    return " ".join(rewritten_words)


def rewritten_search(query: str, retriever_fn, chunks: list[dict], top_k: int = 3) -> list[dict]:
    """
    WHAT: Rewrite a query to expand abbreviations, then run it through the provided
          retriever function to return the top-k matching chunks.
    WHY: Bundling rewriting and retrieval into one call makes it easy to drop
         query-rewriting into any existing RAG pipeline without changing the call site —
         just swap ``retriever_fn(query, ...)`` for ``rewritten_search(query, retriever_fn, ...)``.
    HOW:
      1. Pass the raw query to ``rewrite_query`` to get an expanded version.
      2. Hand the expanded query to ``retriever_fn`` along with ``chunks`` and ``top_k``.
      3. Return whatever the retriever returns (a list of scored chunk dicts).
    EXAMPLE:
      >>> from semantic_search import semantic_search
      >>> results = rewritten_search(
      ...     "explain RAG",
      ...     semantic_search,
      ...     embedded_chunks,
      ...     top_k=3
      ... )
      # Internally searches for "explain retrieval augmented generation"
      # and returns the 3 most relevant chunks.
    """
    rewritten = rewrite_query(query)
    return retriever_fn(rewritten, chunks, top_k=top_k)


from retrieval_debug import debug_retrieval


def debug_rewritten_search(query: str, retriever_fn, chunks: list[dict], expected_doc_id: str | None = None, top_k: int = 3) -> dict:
    """
    WHAT: Rewrite a query, run retrieval, and produce a structured debug report showing
          the original query, the rewritten query, and the full retrieval diagnostics.
    WHY: During development it is hard to know whether a retrieval failure is caused by
         the query itself or by poor embeddings. Surfacing the rewritten form alongside
         the debug report lets a developer see exactly what was searched and why results
         ranked the way they did.
    HOW:
      1. Expand abbreviations in the raw query using ``rewrite_query``.
      2. Call ``debug_retrieval`` with the expanded query; it runs retrieval and
         computes diagnostic metrics (scores, rank of the expected doc, etc.).
      3. Return a dict with three keys: the original query, the rewritten query,
         and the full debug report from step 2.
    EXAMPLE:
      >>> report = debug_rewritten_search(
      ...     "what is KG?",
      ...     semantic_search,
      ...     embedded_chunks,
      ...     expected_doc_id="doc_knowledge_graphs",
      ...     top_k=5
      ... )
      >>> print(report["rewritten_query"])
      "what is knowledge graph?"
      >>> print(report["debug_report"]["success"])
      True   # the expected document appeared in the top-5 results
    """
    rewritten = rewrite_query(query)
    report = debug_retrieval(
        rewritten,
        retriever_fn,
        chunks,
        expected_doc_id=expected_doc_id,
        top_k=top_k,
    )
    return {
        "original_query": query,
        "rewritten_query": rewritten,
        "debug_report": report,
    }

if __name__ == "__main__":
    print("Testing rewrite_query:")
    test_queries = [
        "what is RAG?",
        "LLMs and KGs",
        "AI and ML basics",
        "  multiple   spaces  ",
    ]

    for query in test_queries:
        rewritten = rewrite_query(query)
        print(f"  Original: '{query}'")
        print(f"  Rewritten: '{rewritten}'")

    print("\nTesting error handling:")
    try:
        rewrite_query("")
        print("  ERROR: Should raise ValueError for empty query")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

    try:
        rewrite_query("   ")
        print("  ERROR: Should raise ValueError for whitespace-only query")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
