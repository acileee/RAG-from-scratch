"""
WHAT: Compares the manual cosine-similarity retriever against the ChromaDB HNSW
      retriever for the same query and chunk set.
WHY: ChromaDB uses approximate nearest-neighbour search (HNSW) for speed, while
     the manual semantic_search does exact cosine similarity. Comparing them
     verifies that Chroma's approximation does not meaningfully degrade result
     quality, and surfaces any indexing or normalisation mismatches.
HOW: Runs both retrievers on the same query, collects their returned chunk_ids,
     computes the overlap (intersection) and checks whether both agree on the
     top-ranked result. Returns a structured comparison dict.
EXAMPLE: report = compare_retrievers("what is retrieval-augmented generation?",
             embedded_chunks, collection, top_k=3)
         print(report["top_match"])   # True if both retrievers agree on rank-1 chunk
         print(report["overlap"])     # chunk_ids returned by both
"""

from semantic_search import semantic_search
from chroma_store import chroma_search


def compare_retrievers(query: str, embedded_chunks: list[dict], collection, top_k: int = 3) -> dict:
    """
    WHAT: Runs manual semantic search and ChromaDB search on the same query and
          compares their results side by side.
    WHY: Before relying on ChromaDB in production we need to confirm its results
         agree with the exact-match baseline. Divergence indicates an indexing
         issue, an embedding normalisation mismatch, or a distance-metric
         discrepancy worth investigating.
    HOW: 1. Calls semantic_search(query, embedded_chunks, top_k) for exact results.
         2. Calls chroma_search(query, collection, top_k) for HNSW-approximate results.
         3. Extracts chunk_ids from both result lists.
         4. Computes set intersection (overlap) and checks whether the top-ranked
            chunk_id matches across both retrievers (top_match).
         5. Returns a comparison dict with both raw result lists and the overlap metrics.
    EXAMPLE: report = compare_retrievers(
                 "how do agentic systems retry failed steps?",
                 embedded_chunks, collection, top_k=3)
             # If both return "agents_000" as rank-1, report["top_match"] is True
             # and "agents_000" appears in report["overlap"].
    """
    manual_results = semantic_search(query, embedded_chunks, top_k=top_k)
    chroma_results = chroma_search(query, collection, top_k=top_k)
    manual_ids = [result["chunk_id"] for result in manual_results]
    chroma_ids = [result["chunk_id"] for result in chroma_results]
    overlap = set(manual_ids).intersection(set(chroma_ids))
    top_match = bool(manual_ids and chroma_ids and manual_ids[0] == chroma_ids[0])
    return {
        "query": query,
        "manual_chunk_ids": manual_ids,
        "chroma_chunk_ids": chroma_ids,
        "overlap": list(overlap),
        "top_match": top_match,
        "manual_results": manual_results,
        "chroma_results": chroma_results}


if __name__ == "__main__":
    print("Testing compare_retrievers (demo):")
    print("  This function compares semantic_search and chroma_search retrievers.")
    print("  It requires embedded chunks and a Chroma collection.")
    print("  Example output structure:")
    print("  {")
    print("      'query': str,")
    print("      'manual_chunk_ids': list,")
    print("      'chroma_chunk_ids': list,")
    print("      'overlap': list,")
    print("      'top_match': bool,")
    print("      'manual_results': list,")
    print("      'chroma_results': list")
    print("  }")
