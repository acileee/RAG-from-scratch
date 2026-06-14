"""
WHAT: Module for converting raw text strings and pre-built chunk dicts into
      dense vector embeddings using a locally-running sentence-transformer model.
WHY: Embeddings are the numerical representation of meaning that makes semantic
     search possible. Running the model locally avoids API costs, latency, and
     data-privacy concerns during development.
HOW: Loads the 'all-MiniLM-L6-v2' model from sentence-transformers (downloaded
     once and cached). embed_texts() encodes a list of strings into a 2-D array
     and returns it as a plain Python list. embed_chunks() wraps embed_texts(),
     attaching each resulting vector to a copy of its source chunk dict.
EXAMPLE: After chunking a 10-page PDF about transformer architectures into 40
         chunks, embed_chunks(chunks) returns the same 40 dicts each augmented
         with an "embedding" key — a 384-float list ready to be stored in a
         fake vector store or compared via cosine similarity.
"""

from sentence_transformers import SentenceTransformer


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    WHAT: Convert a list of plain-text strings into a list of 384-dimensional
          embedding vectors using the all-MiniLM-L6-v2 sentence-transformer model.

    WHY: Raw text cannot be compared mathematically. By encoding it into a dense
         vector we place every sentence in a shared geometric space where semantic
         similarity corresponds to geometric proximity. This is the step that turns
         keyword search into semantic search.

    HOW:
        1. Instantiate SentenceTransformer with 'all-MiniLM-L6-v2' (model is
           downloaded on first use and cached locally thereafter).
        2. Call model.encode(texts), which tokenises and forward-passes all strings
           in one batch, returning a NumPy array of shape (len(texts), 384).
        3. Convert to a plain Python list-of-lists so the result is serialisable
           and dependency-free for callers that don't import NumPy.

    EXAMPLE: embed_texts(["What is RAG?", "RAG stands for Retrieval-Augmented Generation"])
             returns two lists of 384 floats. The cosine similarity of those two
             vectors will be high (≈ 0.85+) because the sentences are on the same
             topic, which is exactly what the retrieval step exploits.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return embeddings.tolist()

if __name__ == "__main__":
    embeddings = embed_texts([
    "RAG retrieves relevant chunks.",
    "Knowledge graphs store entities."])
    print(len(embeddings))
    print(type(embeddings[0]))
    print(len(embeddings[0]))


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    WHAT: Attach a dense embedding vector to every chunk dict in a list, returning
          new dicts that are identical to the originals except for an added
          "embedding" key.

    WHY: The RAG pipeline stores chunk metadata (text, position, source file) and
         its embedding together so that vector_search can retrieve both the score
         and the original text in one step. Keeping them in the same dict avoids
         the need for a separate lookup after retrieval.

    HOW:
        1. Extract the "text" field from every chunk into a flat list.
        2. Pass that list to embed_texts() to encode all chunks in a single model
           forward pass (more efficient than encoding one-by-one).
        3. Zip the original chunks with their corresponding embeddings.
        4. For each pair, copy the chunk dict (to avoid mutating the caller's data)
           and add the embedding under the key "embedding".
        5. Return the list of augmented copies.

    EXAMPLE: A document about knowledge graphs is split into 15 chunks by
             chunk_text(). Calling embed_chunks(chunks) produces 15 new dicts, each
             carrying keys like chunk_id, text, start_char, end_char, and now also
             embedding — a 384-float vector. These enriched dicts are then passed
             directly to vector_search() or stored in a fake vector database.
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)
    copied_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        copied_chunk = chunk.copy()
        copied_chunk["embedding"] = embedding
        copied_chunks.append(copied_chunk)
    return copied_chunks

if __name__ == "__main__":
    chunks = [
    {
        "chunk_id": "rag_000",
        "doc_id": "rag",
        "filename": "rag.txt",
        "text": "RAG retrieves relevant chunks before generation.",
        "start_char": 0,
        "end_char": 50,
    },
    {
        "chunk_id": "kg_000",
        "doc_id": "kg",
        "filename": "kg.txt",
        "text": "Knowledge graphs store entities and relationships.",
        "start_char": 0,
        "end_char": 60,
    },
    {
        "chunk_id": "python_000",
        "doc_id": "python",
        "filename": "python.txt",
        "text": "Python helps automate AI experiments.",
        "start_char": 0,
        "end_char": 55,
    },
]
    embedded_chunks = embed_chunks(chunks)
    print(len(embedded_chunks))
    print("embedding" in embedded_chunks[0])
    print(type(embedded_chunks[0]["embedding"]))
    print(len(embedded_chunks[0]["embedding"]))
