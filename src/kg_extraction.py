"""
WHAT: Module for extracting named entities and domain concepts from text chunks so they
      can be used to build a knowledge graph.
WHY: Plain keyword or vector search treats every word as an independent signal. Building
     a knowledge graph requires identifying the important *concepts* in each chunk —
     acronyms, known multi-word phrases, and useful domain terms — so that chunks can be
     connected by the entities they share rather than by raw word overlap.
HOW:
  1. Scan each text for acronyms (2+ consecutive capital letters like RAG, LLM).
  2. Match a curated list of known multi-word concepts against the lowercased text.
  3. Filter individual words against a set of high-value domain terms.
  4. Normalise and deduplicate all collected entities before returning them.
  5. Expose a batch function that processes a list of chunk dicts and returns one
     entity row per chunk, ready to be fed into the graph-building step.
EXAMPLE: Given a chunk "RAG leverages retrieval augmented generation with LLMs to ground
         responses in external context", the extractor returns entities such as
         ["rag", "llm", "retrieval augmented generation", "retrieval", "context"].
         These entities become the nodes and edges in the knowledge graph.
"""


def extract_entities_from_text(text: str) -> list[str]:
    """
    WHAT: Extract a deduplicated list of important entities and concepts from a single
          text string using three complementary strategies: acronym detection, known
          multi-word concept matching, and useful single-word term filtering.
    WHY: To build a knowledge graph we need to know *what* each chunk is about. A chunk
         about "RAG" and a chunk about "retrieval augmented generation" describe the same
         concept; normalising and extracting entities lets the graph connect them even
         if the surface words differ.
    HOW:
      1. Reject empty or whitespace-only input with a ValueError.
      2. Use a regex to find all sequences of 2+ uppercase letters (acronyms like RAG,
         LLM, AI) — these are almost always important named concepts.
      3. Lowercase the text and check for the presence of each entry in a hardcoded list
         of known important multi-word phrases (e.g. "knowledge graph",
         "machine learning").
      4. Split the text on whitespace, normalise each word, and keep any that appear in
         a predefined set of useful domain terms (e.g. "retrieval", "agents").
      5. Iterate over all collected entities, normalise each one, and append it to the
         result list only if it is non-empty and not already present.
      6. Return the deduplicated, normalised entity list.
    EXAMPLE:
      text = "RAG uses retrieval augmented generation with LLMs and knowledge graphs"

      Step 2 — acronyms:       ["RAG", "LLMs"]
      Step 3 — known phrases:  ["retrieval augmented generation", "knowledge graphs"]
      Step 4 — useful terms:   ["retrieval", "knowledge", "graphs"]
      Step 5 — deduplicated:   ["rag", "llms", "retrieval augmented generation",
                                 "knowledge graphs", "retrieval", "knowledge", "graphs"]
    """
    if text.strip() == "":
        raise ValueError("Text cannot be empty or whitespace")

    entities = []

    # Strategy 1: Extract acronyms (usually important concepts)
    # Look for 2+ capital letters like RAG, LLM, AI
    acronyms = re.findall(r"\b[A-Z]{2,}\b", text)
    entities.extend(acronyms)

    # Strategy 2: Look for known important multi-word concepts
    # These are phrases we know are domain-important
    known_concepts = [
        "retrieval augmented generation",
        "knowledge graph",
        "knowledge graphs",
        "large language model",
        "large language models",
        "agentic systems",
        "external context",
        "structured knowledge",
        "machine learning",
        "artificial intelligence",
    ]

    lower_text = text.lower()
    for concept in known_concepts:
        if concept in lower_text:
            entities.append(concept)

    # Strategy 3: Look for useful single words
    # Words that appear in domain-important documents
    useful_terms = {
        "chunks", "generation", "retrieval", "context", "entities",
        "relationships", "knowledge", "graphs", "python", "automation",
        "experiments", "models", "prompts", "tools", "agents",
    }

    for word in text.split():
        normalized_word = normalize_entity(word)
        if normalized_word in useful_terms:
            entities.append(normalized_word)

    # Strategy 4: Deduplicate and normalize
    # Remove duplicates and clean up format
    unique_entities = []
    for entity in entities:
        normalized = normalize_entity(entity)
        # Only add if: (1) not empty, (2) not already in list
        if normalized and normalized not in unique_entities:
            unique_entities.append(normalized)

    return unique_entities


def extract_entities_from_chunks(chunks: list[dict]) -> list[dict]:
    """
    WHAT: Process a list of chunk dicts and return one entity-row dict per chunk,
          each containing the chunk's metadata and the entities extracted from its text.
    WHY: The graph-building step needs a structured list mapping chunk ids to entities.
         This function bridges the raw chunked documents and the graph builder by
         applying entity extraction to every chunk in one call.
    HOW:
      1. Iterate over the input list of chunk dicts.
      2. For each chunk, assert that ``"chunk_id"`` and ``"text"`` fields are present;
         raise a descriptive ValueError if either is missing.
      3. Call ``extract_entities_from_text`` on the chunk's text to get its entity list.
      4. Build an output dict with ``chunk_id``, ``doc_id``, ``filename``, and
         ``entities`` keys (using ``"unknown"`` as the default for optional fields).
      5. Append the output dict to the results list and return it after all chunks are
         processed.
    EXAMPLE:
      >>> chunks = [
      ...     {"chunk_id": "c1", "text": "RAG retrieves relevant context for LLMs",
      ...      "doc_id": "d1", "filename": "intro.txt"},
      ...     {"chunk_id": "c2", "text": "Knowledge graphs store structured knowledge",
      ...      "doc_id": "d1", "filename": "intro.txt"},
      ... ]
      >>> extract_entities_from_chunks(chunks)
      [
        {"chunk_id": "c1", "doc_id": "d1", "filename": "intro.txt",
         "entities": ["rag", "llms", "retrieval augmented generation", "retrieval", "context"]},
        {"chunk_id": "c2", "doc_id": "d1", "filename": "intro.txt",
         "entities": ["knowledge graphs", "knowledge graphs", "knowledge"]},
      ]
    """
    rows = []
    for chunk in chunks:
        if "chunk_id" not in chunk:
            raise ValueError("Chunk is missing required field: chunk_id")
        if "text" not in chunk:
            raise ValueError(f"Chunk with id {chunk['chunk_id']} is missing text")
        entities = extract_entities_from_text(chunk["text"])
        rows.append({
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk.get("doc_id", "unknown"),
            "filename": chunk.get("filename", "unknown"),
            "entities": entities,
        })
    return rows

if __name__ == "__main__":
    print("Testing normalize_entity:")
    test_entities = [
        ("  Python  ", "python"),
        ("RETRIEVAL", "retrieval"),
        ("!!!AI!!!", "ai"),
    ]

    for entity, expected in test_entities:
        result = normalize_entity(entity)
        print(f"  normalize_entity('{entity}') = '{result}' (expected '{expected}')")

    print("\nTesting extract_entities_from_text:")
    test_text = "RAG uses retrieval augmented generation with LLMs and KGs for better context."
    entities = extract_entities_from_text(test_text)
    print(f"  Extracted {len(entities)} entities: {entities}")

    print("\nTesting extract_entities_from_chunks:")
    chunks = [
        {"chunk_id": "c1", "text": "RAG improves retrieval and LLM generation", "doc_id": "d1", "filename": "f1.txt"},
        {"chunk_id": "c2", "text": "Knowledge graphs store entities and relationships", "doc_id": "d1", "filename": "f1.txt"},
    ]
    results = extract_entities_from_chunks(chunks)
    print(f"  Processed {len(results)} chunks:")
    for r in results:
        print(f"    - {r['chunk_id']}: {len(r['entities'])} entities - {r['entities'][:3]}...")

    print("\nTesting error handling:")
    try:
        extract_entities_from_text("")
        print("  ERROR: Should raise ValueError for empty text")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

    try:
        extract_entities_from_chunks([{"chunk_id": "c1"}])
        print("  ERROR: Should raise ValueError for missing text")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
