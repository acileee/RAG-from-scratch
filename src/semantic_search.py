"""
WHAT: Module providing the top-level semantic_search() entry point for the RAG
      pipeline — the function a caller uses when they have a natural-language
      query and a list of pre-embedded chunks and want the most relevant results.
WHY: Semantic search is the core retrieval mechanism that makes RAG powerful.
     Unlike keyword search it understands meaning, so "feline rest" matches
     "how do cats sleep" even with no shared words. This module wires together
     the embedding step (local_embeddings) and the vector comparison step
     (fake_vector_search) behind one clean interface.
HOW: semantic_search() validates the query, converts it to an embedding with
     embed_texts(), confirms every chunk already has an embedding, then delegates
     scoring and ranking to vector_search() from fake_vector_search.py.
EXAMPLE: A user asks "How does attention work in transformers?". semantic_search()
         embeds that question, then vector_search() compares it against 40
         pre-embedded document chunks and returns the 3 chunks whose embeddings
         are geometrically closest — even if none of them contain the word
         "attention" verbatim.
"""

from local_embeddings import embed_texts
from fake_vector_search import vector_search


def semantic_search(query: str, embedded_chunks: list[dict], top_k: int = 3) -> list[dict]:
    """
    WHAT: Find the top_k most semantically relevant chunks for a natural-language
          query by embedding the query and comparing it against pre-embedded chunks
          using cosine similarity.

    WHY: Keyword search fails when the user's wording differs from the document's
         wording. Semantic search solves this by working in embedding space where
         meaning, not surface form, determines similarity. This is what lets a RAG
         system answer "What are feline sleeping habits?" using a chunk that says
         "Cats sleep 16 hours a day" — zero shared content words, yet the vectors
         are close.

    HOW:
        1. Reject blank queries immediately with a ValueError — an empty query
           would produce a meaningless embedding and waste compute.
        2. Embed the query by calling embed_texts([query])[0], using the same
           sentence-transformer model that was used to embed the chunks, so the
           resulting vectors live in the same geometric space.
        3. Validate that every chunk in embedded_chunks carries an "embedding" key;
           raise ValueError with the offending chunk_id if one is missing. A chunk
           without an embedding cannot be scored.
        4. Delegate to vector_search(query_embedding, embedded_chunks, top_k),
           which computes cosine similarity for each chunk and returns the top_k
           ranked results.

    EXAMPLE: After running embed_chunks() on 20 chunks from a RAG tutorial PDF,
             semantic_search("how do embeddings capture meaning", embedded_chunks,
             top_k=3) embeds the question, scores all 20 chunks, and returns the 3
             whose content is closest in meaning — typically chunks that discuss
             vector spaces, word representations, or model encoding — even if they
             never use the word "capture".
    """
    if query.strip() == "":
        raise ValueError("Query cannot be empty or whitespace")

    # Step 1: Convert query to embedding (dense vector of meaning)
    query_embedding = embed_texts([query])[0]

    # Step 2: Validate all chunks have embeddings
    for embedded_chunk in embedded_chunks:
        if "embedding" not in embedded_chunk:
            raise ValueError(f"Chunk with id {embedded_chunk['chunk_id']} does not have an embedding")

    # Step 3: Find most similar chunks using vector search
    results = vector_search(query_embedding, embedded_chunks, top_k)

    return results

if __name__ == "__main__":
    print("Testing semantic_search:")
    print("  This function requires embedded chunks with embeddings from local_embeddings module.")

    test_chunks = [
        {"chunk_id": "1", "text": "Python programming", "start_char": 0, "end_char": 18, "doc_id": "d1", "filename": "f1.txt", "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]},
        {"chunk_id": "2", "text": "Machine learning models", "start_char": 18, "end_char": 42, "doc_id": "d1", "filename": "f1.txt", "embedding": [0.15, 0.25, 0.35, 0.45, 0.55]},
    ]

    print("\n  Test data prepared with embedded chunks")
    print(f"  Sample chunk embeddings available: {len(test_chunks)} chunks")

    print("\nTesting error handling:")
    try:
        semantic_search("", test_chunks)
        print("  ERROR: Should raise ValueError for empty query")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

    try:
        semantic_search("test", [{"chunk_id": "1", "text": "text"}])
        print("  ERROR: Should raise ValueError for missing embedding")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
