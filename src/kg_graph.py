"""
WHAT: Module for constructing, summarising, and querying a NetworkX knowledge graph
      that links document chunks to the entities they mention.
WHY: A flat list of chunks with entity labels is not enough to do graph-based retrieval.
     We need an actual graph structure so we can traverse relationships — e.g. "given
     chunk A mentions entity X, find all *other* chunks that also mention X" — which is
     impossible with a simple list or dict.
HOW:
  1. Iterate over entity rows (produced by kg_extraction) and create one graph node
     per unique chunk and one per unique entity.
  2. Add a "mentions" edge between each chunk node and every entity it contains.
  3. Add a "co_occurs" edge between every pair of entities that appear in the same chunk.
  4. Expose helper functions to summarise the graph and to look up which chunks mention
     a given entity by name.
EXAMPLE: After processing 200 chunks about RAG and knowledge graphs, the graph contains
         ~200 chunk nodes, ~30 entity nodes, and hundreds of "mentions" and "co_occurs"
         edges. Querying ``find_chunks_for_entity(graph, "retrieval")`` instantly returns
         all chunk ids that discuss retrieval, without re-scanning any text.
"""


def build_knowledge_graph(entity_rows: list[dict]):
    """
    WHAT: Construct a NetworkX undirected graph from a list of entity rows, adding chunk
          nodes, entity nodes, "mentions" edges, and "co_occurs" edges.
    WHY: A graph structure makes it possible to traverse from a query entity to all
         chunks mentioning it, and from those chunks to related entities — enabling
         richer context expansion than flat keyword lookup allows.
    HOW:
      1. Create an empty ``nx.Graph()``.
      2. For each entity row, add a chunk node labelled ``"chunk:<chunk_id>"`` with
         ``type="chunk"`` and the associated metadata (doc_id, filename).
      3. For each entity in the row, add an entity node labelled ``"entity:<name>"``
         with ``type="entity"``, then add a ``"mentions"`` edge between the chunk node
         and the entity node.
      4. For every pair of entities in the same chunk, add a ``"co_occurs"`` edge
         between the two entity nodes (indicating they are semantically related through
         co-presence in the same passage).
      5. Return the fully populated graph.
    EXAMPLE:
      entity_rows = [
        {"chunk_id": "c1", "doc_id": "d1", "filename": "doc.txt",
         "entities": ["python", "programming"]},
      ]
      graph = build_knowledge_graph(entity_rows)
      # Nodes: "chunk:c1", "entity:python", "entity:programming"
      # Edges: chunk:c1 --mentions--> entity:python
      #        chunk:c1 --mentions--> entity:programming
      #        entity:python --co_occurs--> entity:programming
    """
    graph = nx.Graph()

    for entity_row in entity_rows:
        chunk_id = entity_row["chunk_id"]
        doc_id = entity_row.get("doc_id", "unknown")
        filename = entity_row.get("filename", "unknown")
        entities = entity_row.get("entities", [])

        # Create chunk node
        chunk_node = f"chunk:{chunk_id}"
        graph.add_node(
            chunk_node,
            type="chunk",           # Mark this as a chunk node
            chunk_id=chunk_id,
            doc_id=doc_id,
            filename=filename
        )

        # Add entity nodes and connections
        for entity in entities:
            entity_id = entity
            entity_node = f"entity:{entity_id}"

            # Create entity node
            graph.add_node(
                entity_node,
                type="entity",        # Mark as entity node
                name=entity
            )

            # Create "mentions" relationship: chunk --mentions--> entity
            graph.add_edge(chunk_node, entity_node, relation="mentions")

        # Connect co-occurring entities
        # If two entities appear in same chunk, they're related
        for i, entity_a in enumerate(entities):
            for entity_b in entities[i+1:]:
                entity_a_node = f"entity:{entity_a}"
                entity_b_node = f"entity:{entity_b}"
                # Create "co_occurs" relationship
                graph.add_edge(entity_a_node, entity_b_node, relation="co_occurs")

    return graph


def summarize_graph(graph) -> dict:
    """
    WHAT: Count the total nodes, edges, chunk nodes, and entity nodes in a knowledge
          graph and return the counts as a dict.
    WHY: After building a graph it is useful to verify its size and composition before
         running queries against it. A graph with zero entity nodes, for example,
         would indicate that entity extraction failed upstream.
    HOW:
      1. Iterate over all nodes with their data dicts.
      2. Increment ``chunks_count`` for every node whose ``type`` attribute is
         ``"chunk"``, and ``entities_count`` for every node whose ``type`` is
         ``"entity"``.
      3. Read ``graph.number_of_nodes()`` and ``graph.number_of_edges()`` for totals.
      4. Return all four counts in a single dict.
    EXAMPLE:
      >>> graph = build_knowledge_graph(entity_rows)   # 50 chunks, 10 entities
      >>> summarize_graph(graph)
      {"num_nodes": 60, "num_edges": 120,
       "num_chunk_nodes": 50, "num_entity_nodes": 10}
      # 50 chunk nodes + 10 entity nodes = 60 total nodes
    """
    chunks_count = 0
    entities_count = 0
    for node, data in graph.nodes(data=True):
            if data.get("type") == "chunk":
                chunks_count += 1
            elif data.get("type") == "entity":
                 entities_count += 1
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    return {
    "num_nodes": num_nodes,
    "num_edges": num_edges,
    "num_chunk_nodes": chunks_count,
    "num_entity_nodes": entities_count,}


def find_chunks_for_entity(graph, entity) -> list[str]:
    """
    WHAT: Look up an entity by name in the knowledge graph and return the ids of all
          chunk nodes that are directly connected to it via a "mentions" edge.
    WHY: This is the core graph-retrieval primitive. Instead of scanning every chunk's
         text for a keyword, we traverse the pre-built graph in O(degree) time to find
         all chunks that mention the entity — which is much faster and does not require
         re-loading or re-scanning text.
    HOW:
      1. Normalise the entity name using ``normalize_entity`` so that "Python", "PYTHON",
         and "python" all resolve to the same node key.
      2. Construct the expected node label ``"entity:<normalised_name>"``.
      3. If the node does not exist in the graph, return an empty list immediately.
      4. Iterate over the entity node's neighbours; for each neighbour whose ``type``
         attribute is ``"chunk"``, append its ``chunk_id`` to the result list.
      5. Return the list of chunk ids.
    EXAMPLE:
      >>> chunk_ids = find_chunks_for_entity(graph, "Python")
      ["c1", "c2", "c7"]
      # Chunks c1, c2, and c7 all contain the entity "python".
      # These ids can then be used to look up the full chunk text for generation.
    """
    normalized_entity = normalize_entity(entity)
    entity_node = f"entity:{normalized_entity}"
    chunk_ids = []
    if not graph.has_node(entity_node):
        return []
    for neighbor in graph.neighbors(entity_node):
        neighbor_data = graph.nodes[neighbor]
        if neighbor_data.get("type") == "chunk":
            chunk_ids.append(neighbor_data["chunk_id"])
    return chunk_ids

if __name__ == "__main__":
    print("Testing kg_graph functions:")

    print("\nTesting build_knowledge_graph:")
    entity_rows = [
        {"chunk_id": "c1", "doc_id": "d1", "filename": "f1.txt", "entities": ["Python", "programming"]},
        {"chunk_id": "c2", "doc_id": "d1", "filename": "f1.txt", "entities": ["Python", "language"]},
    ]

    graph = build_knowledge_graph(entity_rows)
    print(f"  Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

    print("\nTesting summarize_graph:")
    summary = summarize_graph(graph)
    print(f"  Nodes: {summary['num_nodes']}")
    print(f"  Edges: {summary['num_edges']}")
    print(f"  Chunk nodes: {summary['num_chunk_nodes']}")
    print(f"  Entity nodes: {summary['num_entity_nodes']}")

    print("\nTesting find_chunks_for_entity:")
    chunk_ids = find_chunks_for_entity(graph, "Python")
    print(f"  Chunks containing 'Python': {chunk_ids}")
