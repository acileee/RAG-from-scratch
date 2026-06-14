"""
WHAT: Module for searching pre-computed embeddings stored on disk without re-embedding at query time.
WHY: Generating embeddings is slow and costly; saving them once and reusing them makes the
     RAG pipeline fast and reproducible across multiple query sessions.
HOW:
  1. Load a JSON file containing chunk objects that each carry a pre-computed embedding vector.
  2. Validate the structure so errors are caught early.
  3. Delegate the actual similarity computation to the semantic_search module.
  4. Return the top-k most similar chunks to the caller.
EXAMPLE: A user runs the RAG pipeline twice. The first run embeds 500 document chunks and
         saves them to ``data/embeddings.json``. The second run calls
         ``search_saved_embeddings("What is RAG?", "data/embeddings.json", top_k=5)``
         and gets results instantly without re-calling the embedding model.
"""

from storage import load_json, save_json
from semantic_search import semantic_search


def search_saved_embeddings(query: str, embeddings_path: str, top_k: int = 3) -> list[dict]:
    """
    WHAT: Load pre-computed embeddings from a JSON file and return the top-k chunks
          most semantically similar to the given query.
    WHY: Re-embedding every chunk on every query is wasteful. Loading saved embeddings
         lets us skip the embedding step entirely at query time, making retrieval fast
         even over large document collections.
    HOW:
      1. Read the JSON file at ``embeddings_path`` into memory as a list of dicts.
      2. Confirm the loaded object is actually a list; raise ValueError otherwise.
      3. Confirm every chunk dict contains an ``"embedding"`` key; raise ValueError
         for any that are missing one (a partial save would silently corrupt retrieval).
      4. Call ``semantic_search`` with the query and the validated chunk list to compute
         cosine similarities and return the ranked top-k results.
    EXAMPLE:
      >>> results = search_saved_embeddings(
      ...     "How does retrieval augmented generation work?",
      ...     "data/embedded_chunks.json",
      ...     top_k=3
      ... )
      >>> for r in results:
      ...     print(r["chunk_id"], r["score"], r["text"][:60])
      chunk_42  0.91  "RAG combines a retriever with a generative language model..."
      chunk_17  0.87  "Retrieval augmented generation grounds LLM responses in..."
      chunk_5   0.83  "The retriever fetches relevant documents before generation..."
    """
    embedding_path = embeddings_path
    embedded_chunks = load_json(embedding_path)
    if isinstance(embedded_chunks, list):
        print(f"Loaded {len(embedded_chunks)} embedded chunks from {embedding_path}")
    else:
        raise ValueError(f"Expected a list of embedded chunks, but got {type(embedded_chunks)}")
    for chunk in embedded_chunks:
        if "embedding" not in chunk:
            raise ValueError(f"Chunk with id {chunk['chunk_id']} does not have an embedding")
    results = semantic_search(query, embedded_chunks, top_k=top_k)
    return results

if __name__ == "__main__":
    print("Testing saved_embedding_search:")

    print("\nTest structure expected for embeddings file:")
    print("  [")
    print("    {")
    print("      'chunk_id': 'id',")
    print("      'text': 'chunk text',")
    print("      'embedding': [array of floats],")
    print("      'doc_id': 'doc_id',")
    print("      'filename': 'filename.txt'")
    print("    },")
    print("    ...")
    print("  ]")

    print("\nTesting error handling:")
    try:
        # This will fail unless the file exists, which is expected
        search_saved_embeddings("test query", "nonexistent_embeddings.json")
    except FileNotFoundError:
        print("  ✓ Correctly raises FileNotFoundError for missing embeddings file")
    except Exception as e:
        print(f"  Error (expected): {type(e).__name__}: {e}")
