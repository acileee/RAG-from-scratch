"""
WHAT: Module implementing two keyword-based retrieval strategies for RAG:
      keyword_search() (set-intersection scoring) and keyword_search_v2()
      (term-frequency scoring).
WHY: Keyword search is the simplest possible retriever and serves as a baseline
     to understand what semantic search improves upon. It requires no model, no
     GPU, and no pre-computed embeddings — making it ideal for learning the
     retrieval concept in isolation.
HOW: Both functions tokenise the query and each chunk into lowercase words,
     remove common English stopwords, and score chunks by how many query tokens
     they contain. They differ in scoring: v1 counts unique matching tokens
     (set intersection), v2 counts total occurrences (term frequency). Both
     return the top-k highest-scoring chunks sorted descending.
EXAMPLE: Given a corpus of AI textbook chunks and a query "attention mechanism
         transformer", keyword_search returns the chunks that literally mention
         "attention" and "transformer" most — but would miss a chunk that uses
         the synonym "self-attention layer" if those exact tokens aren't in the
         query. That limitation motivates the semantic_search module.
"""

import re


def keyword_search(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    WHAT: Score and rank document chunks against a query using set-intersection
          keyword matching, returning the top_k highest-scoring chunks.

    WHY: Before investing in embeddings and vector search, it is useful to
         understand what plain word-overlap retrieval can and cannot do. This
         function makes that baseline concrete and measurable. It is also a
         fast, zero-dependency fallback when embedding infrastructure is
         unavailable.

    HOW:
        1. Define an inner tokenize() function that lower-cases, strips
           punctuation, splits on whitespace, and removes stopwords, returning
           a set of meaningful tokens.
        2. Tokenise the query into query_tokens.
        3. For each chunk, tokenise its text and compute the score as the size
           of the intersection between query_tokens and chunk_tokens
           (i.e. how many distinct query words appear in the chunk).
        4. Discard chunks with score == 0 (no matching tokens).
        5. Sort surviving chunks by score descending and return the first top_k.

    EXAMPLE: Query "retrieval augmented generation" tokenises to
             {"retrieval", "augmented", "generation"}.
             A chunk containing "Retrieval-Augmented Generation (RAG) combines…"
             scores 3 (all three tokens match) and would be ranked first.
             A chunk about "neural network training" scores 0 and is discarded.
    """
    def tokenize(text: str) -> set:
        """
        WHAT: Convert a string into a set of meaningful lowercase tokens with
              punctuation and stopwords removed.

        WHY: Set-based intersection only works reliably when both sides use the
             same normalisation. Removing stopwords prevents common words like
             "the" or "is" from inflating match scores for irrelevant chunks.

        HOW:
            1. Split text on whitespace to get raw word tokens.
            2. Lowercase each token and strip punctuation via regex.
            3. Discard empty strings and words in the stopword set.
            4. Return remaining tokens as a set (deduplication is implicit).

        EXAMPLE: tokenize("RAG retrieves the most relevant chunks!")
                 returns {"rag", "retrieves", "most", "relevant", "chunks"}
                 — "the" is removed as a stopword, punctuation is stripped.
        """
        # Common English stopwords that don't carry meaning
        remove_stopwords = set(["the", "is", "in", "and", "to", "of", "a", "or", "an", "for", "with", "on", "as", "by"])
        tokens = set()
        for word in text.split():
            # Convert to lowercase for case-insensitive matching
            cleaned_word = word.lower()
            # Remove punctuation
            cleaned_word = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@[\\]^_`{|}~]', '', cleaned_word)
            # Keep only non-empty tokens that aren't stopwords
            if cleaned_word and cleaned_word not in remove_stopwords:
                tokens.add(cleaned_word)
        return tokens

    # Extract tokens from query
    query_tokens = tokenize(query)

    # Helper to extract key fields from chunk (for output)
    copied_chunk = lambda chunk: {
        "chunk_id": chunk["chunk_id"],
        "text": chunk["text"],
        "start_char": chunk["start_char"],
        "end_char": chunk["end_char"],
        "doc_id": chunk["doc_id"],
        "filename": chunk["filename"]
    }

    # Score each chunk
    scored_chunks = []
    score_chunk = lambda chunk: len(query_tokens.intersection(tokenize(chunk["text"])))

    for chunk in chunks:
        # Count how many query tokens appear in this chunk
        score = score_chunk(chunk)
        # Only keep chunks with at least one match
        if score > 0:
            chunk_copy = copied_chunk(chunk)
            chunk_copy["score"] = score
            scored_chunks.append(chunk_copy)

    # Sort by score (highest first)
    sorted_chunks = sorted(scored_chunks, key=lambda x: x["score"], reverse=True)

    # Return only top-k results
    return sorted_chunks[:top_k]


def keyword_search_v2(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    WHAT: Score and rank document chunks using term-frequency keyword matching —
          a chunk scores higher each time a query word appears in it, not just
          once per unique word.

    WHY: keyword_search() treats each matched token equally regardless of how
         many times it appears. keyword_search_v2() rewards chunks that repeat
         query terms, reflecting the intuition that a passage heavily discussing
         "retrieval" is more relevant to a retrieval query than one that merely
         mentions it once. This mirrors classic TF-style scoring from information
         retrieval.

    HOW:
        1. Define a local tokenize_list() that returns a list (not a set) so
           repeated words are preserved.
        2. Tokenise the query into query_tokens (list).
        3. For each chunk, tokenise its text into chunk_tokens (list).
        4. For every query token, add chunk_tokens.count(token) to the score —
           so a token appearing 3 times in a chunk contributes 3 to the score.
        5. Discard chunks with score == 0, sort descending, return top_k.

    EXAMPLE: Query "embedding embedding vector" tokenises to
             ["embedding", "embedding", "vector"].
             A chunk saying "Embedding models produce embedding vectors for each
             token" scores count("embedding")=2 + count("embedding")=2 +
             count("vector")=1 = 5, ranking it above a chunk that only says
             "an embedding exists" (score = 2 + 2 + 0 = 4).
    """
    def tokenize_list(text: str) -> list:
        """
        WHAT: Convert a string into a list of meaningful lowercase tokens,
              preserving duplicates so term frequency can be counted.

        WHY: Unlike the set-based tokenize() in keyword_search(), this version
             keeps repeated words because keyword_search_v2 needs to call
             list.count() on the result — which requires duplicates to exist.

        HOW:
            1. Split on whitespace, lowercase, strip punctuation via regex.
            2. Skip empty strings and stopwords.
            3. Append surviving tokens to a list (order and repetition preserved).

        EXAMPLE: tokenize_list("RAG uses retrieval and retrieval is fast")
                 returns ["rag", "uses", "retrieval", "retrieval", "fast"]
                 — "and" and "is" are stopwords and are removed; "retrieval"
                 appears twice, which allows count("retrieval") == 2.
        """
        remove_stopwords = set(["the", "is", "in", "and", "to", "of", "a", "or", "an", "for", "with", "on", "as", "by"])
        tokens = []
        for word in text.split():
            cleaned_word = word.lower()
            cleaned_word = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@[\\]^_`{|}~]', '', cleaned_word)
            if cleaned_word and cleaned_word not in remove_stopwords:
                tokens.append(cleaned_word)
        return tokens

    query_tokens = tokenize_list(query)
    scored_chunks = []
    for chunk in chunks:
        chunk_tokens = tokenize_list(chunk["text"])
        score = 0
        for token in query_tokens:
            if token in chunk_tokens:
                score += chunk_tokens.count(token)
        if score > 0:
            chunk_copy = {"chunk_id": chunk["chunk_id"], "text": chunk["text"], "start_char": chunk["start_char"], "end_char": chunk["end_char"], "doc_id": chunk["doc_id"], "filename": chunk["filename"], "score": score}
            scored_chunks.append(chunk_copy)
    sorted_chunks = sorted(scored_chunks, key=lambda x: x["score"], reverse=True)
    return sorted_chunks[:top_k]

if __name__ == "__main__":
    test_chunks = [
        {"chunk_id": "1", "text": "Python is useful for machine learning", "start_char": 0, "end_char": 40, "doc_id": "doc1", "filename": "file1.txt"},
        {"chunk_id": "2", "text": "Machine learning uses neural networks", "start_char": 40, "end_char": 80, "doc_id": "doc1", "filename": "file1.txt"},
        {"chunk_id": "3", "text": "Deep learning improves model accuracy", "start_char": 80, "end_char": 120, "doc_id": "doc2", "filename": "file2.txt"},
    ]

    print("Testing keyword_search:")
    results = keyword_search("python machine learning", test_chunks, top_k=2)
    print(f"  Found {len(results)} results for 'python machine learning'")
    for r in results:
        print(f"    - {r['chunk_id']}: score={r['score']}")

    print("\nTesting keyword_search_v2:")
    results = keyword_search_v2("python machine learning", test_chunks, top_k=2)
    print(f"  Found {len(results)} results for 'python machine learning'")
    for r in results:
        print(f"    - {r['chunk_id']}: score={r['score']}")

    print("\nTesting edge cases:")
    print(f"  Empty query v1: {len(keyword_search('', test_chunks))} results")
    print(f"  Empty query v2: {len(keyword_search_v2('', test_chunks))} results")
    print(f"  Nonexistent term: {len(keyword_search('xyz123', test_chunks))} results")
