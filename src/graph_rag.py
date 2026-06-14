"""
WHAT: Module that implements Graph RAG — a retrieval strategy that augments standard
      semantic search by using a knowledge graph to discover and add related chunks
      that the initial search might have missed.
WHY: Standard vector search returns chunks that are close to the query in embedding
     space, but can miss related chunks that use different vocabulary yet discuss the
     same concepts. The knowledge graph encodes entity co-occurrence relationships that
     let us bridge this vocabulary gap and return a richer, more complete context window.
HOW:
  1. Run first-stage semantic (or keyword) retrieval to get an initial set of chunks.
  2. For each retrieved chunk, look up its neighbours in the knowledge graph.
  3. Follow entity neighbours outward to find other chunks that share entities with
     the retrieved chunks ("entity-bridged" chunks).
  4. Merge the original and bridged chunks (deduplicating by chunk id) up to a
     configurable limit.
  5. Return the expanded set so the LLM generation step has broader context.
EXAMPLE: The query "how does RAG handle retrieval?" surfaces chunk c1 (about RAG) via
         embedding search. The graph shows that c1 mentions entity "retrieval", and
         chunk c9 also mentions "retrieval" (about query rewriting). Graph RAG adds c9
         to the context window, giving the LLM information about retrieval strategies
         it would otherwise have missed.
"""


def build_chunk_lookup(chunks: list[dict]) -> dict:
    """
    WHAT: Build a dict that maps each chunk's ``chunk_id`` to the full chunk dict,
          enabling O(1) lookup by id during graph traversal.
    WHY: Graph traversal discovers chunk ids (stored as node attributes). To actually
         return the full chunk text and metadata to the caller we need a fast way to
         go from an id back to the chunk dict without scanning the entire list each time.
    HOW:
      1. Initialise an empty dict.
      2. Iterate over the list of chunk dicts.
      3. Read ``chunks["chunk_id"]`` for each item and store the chunk dict at that key.
      4. Return the populated lookup dict.
    EXAMPLE:
      >>> chunks = [
      ...     {"chunk_id": "c1", "text": "RAG overview ...", "doc_id": "d1"},
      ...     {"chunk_id": "c2", "text": "Retrieval strategies ...", "doc_id": "d1"},
      ... ]
      >>> lookup = build_chunk_lookup(chunks)
      >>> lookup["c1"]["text"]
      "RAG overview ..."
      # Used during graph expansion to retrieve chunk content from a discovered chunk id.
    """
    lookup = {}
    for chunk in chunks:
        chunk_id = chunks["chunk_id"]
        lookup[chunk_id] = chunk
    return lookup


def expand_chunks_with_graph(retrieved_chunks: list[dict], graph, chunk_lookup: dict, max_extra_chunks: int = 3) -> list[dict]:
    """
    WHAT: Expand a list of retrieved chunks by traversing the knowledge graph to find
          additional chunks that share entities with the retrieved set, up to a
          configurable maximum number of extra chunks.
    WHY: The initially retrieved chunks cover the query well, but related chunks that
         discuss the same entities from a different angle can provide the LLM with
         complementary context. Graph expansion is a targeted way to broaden context
         without simply increasing ``top_k`` and returning low-relevance results.
    HOW:
      1. Start with a copy of the retrieved chunks and a set of already-seen chunk ids
         to prevent duplicates.
      2. For each retrieved chunk, find its node in the graph (``"chunk:<chunk_id>"``).
         If the chunk has no graph node, skip it.
      3. Iterate over the chunk node's neighbours; only consider entity-type neighbours.
      4. For each entity neighbour, iterate over *its* neighbours; only consider
         chunk-type neighbours (these are other chunks that share this entity).
      5. If the candidate chunk id is new (not in ``seen_chunk_ids``) and exists in
         ``chunk_lookup``, append its full dict to the expanded list, mark it seen,
         and increment the extra-chunks counter.
      6. Stop as soon as ``max_extra_chunks`` new chunks have been added.
      7. Return the expanded list (original + up to ``max_extra_chunks`` new chunks).
    EXAMPLE:
      retrieved_chunks = [chunk_c1]   # c1 mentions entities ["rag", "retrieval"]
      # In the graph, entity "retrieval" is also linked to chunk c9 (query rewriting).
      expanded = expand_chunks_with_graph(retrieved_chunks, graph, lookup, max_extra_chunks=2)
      # expanded = [chunk_c1, chunk_c9]
      # chunk_c9 was discovered because it shares the "retrieval" entity with chunk_c1.
    """
    expanded_chunks = []
    seen_chunk_ids = set()
    extra_count = 0
    for chunk in retrieved_chunks:
        expanded_chunks.append(chunk)
        chunk_id = chunk["chunk_id"]
        seen_chunk_ids.add(chunk_id)
    for retrieved_chunk in retrieved_chunks:
        chunk_id = retrieved_chunk["chunk_id"]
        chunk_node = f"chunk:{chunk_id}"
        if chunk_node not in graph:
            continue
        for neighbor in graph.neighbors(chunk_node):
            neighbor_data = graph.nodes[neighbor]
            if neighbor_data.get("type") != "entity":
                continue
            entity_neighbors = graph.neighbors(neighbor)
            for entity_neighbor in entity_neighbors:
                entity_neighbor_data = graph.nodes[entity_neighbor]
                if entity_neighbor_data.get("type") != "chunk":
                    continue
                candidate_chunk_id = entity_neighbor_data["chunk_id"]
                if candidate_chunk_id in seen_chunk_ids:
                    continue
                if candidate_chunk_id not in chunk_lookup:
                    continue
                expanded_chunks.append(chunk_lookup[candidate_chunk_id])
                seen_chunk_ids.add(candidate_chunk_id)
                extra_count += 1
                if extra_count >= max_extra_chunks:
                    return expanded_chunks
    return expanded_chunks


def graph_rag_search(query: str, retriever_fn, chunks: list[dict], graph, top_k: int = 3, max_extra_chunks: int = 3,) -> list[dict]:
    """
    WHAT: Perform a full Graph RAG search: retrieve an initial set of chunks with a
          standard retriever, build a lookup table, then expand the results using
          entity relationships in the knowledge graph.
    WHY: This is the top-level entry point that combines all the graph RAG components
         into one call. Callers do not need to know about the graph internals — they
         just pass a query and get back an enriched chunk list suitable for LLM
         generation.
    HOW:
      1. Call ``retriever_fn(query, chunks, top_k=top_k)`` to get the initial ranked
         chunks (e.g. via semantic search or BM25).
      2. Call ``build_chunk_lookup(chunks)`` to build an id-to-chunk mapping for fast
         lookups during graph traversal.
      3. Call ``expand_chunks_with_graph(retrieved_chunks, graph, chunk_lookup,
         max_extra_chunks)`` to add entity-bridged chunks from the knowledge graph.
      4. Return the expanded list directly.
    EXAMPLE:
      >>> from semantic_search import semantic_search
      >>> expanded = graph_rag_search(
      ...     "how does RAG use knowledge graphs?",
      ...     semantic_search,
      ...     embedded_chunks,
      ...     kg_graph,
      ...     top_k=3,
      ...     max_extra_chunks=2
      ... )
      # Stage 1: semantic_search returns chunks c1, c4, c7 (top-3 by cosine similarity).
      # Stage 2: graph traversal discovers c12 (shares "knowledge graphs" entity with c1)
      #          and c3 (shares "retrieval" entity with c4).
      # Result: [c1, c4, c7, c12, c3]  — 5 chunks for the LLM instead of 3.
    """
    retrieved_chunks = retriever_fn(query, chunks, top_k=top_k)
    chunk_lookup = build_chunk_lookup(chunks)
    expanded_chunks = expand_chunks_with_graph(retrieved_chunks, graph, chunk_lookup, max_extra_chunks)
    return expanded_chunks

if __name__ == "__main__":
    print("Testing graph_rag functions:")
    print("  build_chunk_lookup: Creates a lookup dictionary for chunks")
    print("  expand_chunks_with_graph: Expands retrieved chunks using knowledge graph")
    print("  graph_rag_search: Performs search with graph-based chunk expansion")
    print("\nNote: Requires a NetworkX graph with chunk and entity nodes.")
