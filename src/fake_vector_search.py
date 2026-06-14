"""
WHAT: Module implementing a simple in-memory vector search (vector_search()) that
      scores pre-embedded document chunks against a query embedding using cosine
      similarity and returns the top-k matches.
WHY: Real-world RAG systems use dedicated vector databases (FAISS, Pinecone, Weaviate)
     for approximate nearest-neighbour search at scale. This "fake" implementation
     performs the same operation with a plain Python loop so learners can see
     exactly what a vector database does internally — without the abstraction of
     an external library.
HOW: For each chunk, cosine_similarity_numpy() computes the similarity between the
     query embedding and the chunk's stored embedding. All chunks with an embedding
     are scored, sorted descending, and the top_k are returned with their scores
     attached.
EXAMPLE: After calling embed_chunks() on 30 chunks from a machine-learning
         textbook, vector_search(query_embedding, embedded_chunks, top_k=3)
         iterates all 30, scores them, and surfaces the 3 paragraphs most
         semantically aligned with the query — the exact output that gets
         injected into an LLM's context window in the generation step.
"""

from cosine_similarity import cosine_similarity_numpy


def vector_search(query_embedding: list[float], chunks: list[dict], top_k: int = 3) -> list[dict]:
    """
    WHAT: Score every chunk in a list against a query embedding using cosine
          similarity, then return the top_k chunks sorted by descending score.

    WHY: This is the retrieval core of a RAG system. Given a user query that has
         been converted to a vector, we need to find the stored text chunks whose
         meaning is closest to that query. Cosine similarity over dense embeddings
         captures semantic closeness far better than any keyword heuristic, which
         is why this step is central to making RAG useful.

    HOW:
        1. Iterate over every chunk in the input list.
        2. Read the "embedding" field from each chunk. If it is missing, raise a
           ValueError immediately — every chunk must be embedded before this
           function is called (see local_embeddings.embed_chunks).
        3. Call cosine_similarity_numpy(query_embedding, chunk_embedding) to
           compute a float in [-1, 1] representing semantic alignment (in practice
           embedding vectors are non-negative, so scores land in [0, 1]).
        4. Build a copy of each chunk dict that contains only the fields needed
           downstream (chunk_id, text, start_char, end_char, doc_id, filename)
           plus the computed "score". This keeps the output clean.
        5. Sort all scored chunks by "score" descending.
        6. Return the first top_k entries.

    EXAMPLE: A query "How does the transformer attention mechanism work?" is
             embedded into a 384-dim vector. vector_search() scores 40 textbook
             chunks against it. Chunk #17, "The scaled dot-product attention
             computes …", scores 0.91. Chunk #3, "Python syntax for loops", scores
             0.04. The function returns chunk #17 and the next two highest-scoring
             chunks as the context for the LLM to answer from.
    """
    scored_chunks = []

    for chunk in chunks:
        chunk_embedding = chunk.get("embedding")

        if chunk_embedding is not None:
            # Calculate similarity between query and this chunk
            score = cosine_similarity_numpy(query_embedding, chunk_embedding)

            # Prepare result with metadata
            chunk_copy = {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "doc_id": chunk["doc_id"],
                "filename": chunk["filename"],
                "score": score  # Higher score = better match
            }
            scored_chunks.append(chunk_copy)
        else:
            # Every chunk must have an embedding to use vector search
            raise ValueError(f"Chunk with id {chunk['chunk_id']} does not have an embedding")

    # Sort by similarity score (highest first)
    sorted_chunks = sorted(scored_chunks, key=lambda x: x["score"], reverse=True)

    # Return only top-k most similar chunks
    return sorted_chunks[:top_k]

if __name__ == "__main__":
    test_chunks = [
        {"chunk_id": "1", "text": "Python programming", "start_char": 0, "end_char": 18, "doc_id": "doc1", "filename": "file1.txt", "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]},
        {"chunk_id": "2", "text": "Machine learning models", "start_char": 18, "end_char": 42, "doc_id": "doc1", "filename": "file1.txt", "embedding": [0.15, 0.25, 0.35, 0.45, 0.55]},
        {"chunk_id": "3", "text": "Deep neural networks", "start_char": 42, "end_char": 62, "doc_id": "doc2", "filename": "file2.txt", "embedding": [0.2, 0.3, 0.4, 0.5, 0.6]},
    ]

    query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

    print("Testing vector_search:")
    results = vector_search(query_embedding, test_chunks, top_k=2)
    print(f"  Found {len(results)} results")
    for r in results:
        print(f"    - {r['chunk_id']}: score={r['score']:.4f}, text='{r['text']}'")

    print("\nTesting with top_k=5 (more than available):")
    results = vector_search(query_embedding, test_chunks, top_k=5)
    print(f"  Found {len(results)} results (max available)")

    print("\nTesting error handling (missing embedding):")
    bad_chunk = {"chunk_id": "4", "text": "No embedding", "start_char": 0, "end_char": 12, "doc_id": "doc3", "filename": "file3.txt"}
    try:
        vector_search(query_embedding, test_chunks + [bad_chunk], top_k=3)
        print("  ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
