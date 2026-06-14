"""
WHAT: Provides ChromaDB-backed storage and retrieval for embedded text chunks.
WHY: A persistent vector store is needed so embeddings survive between sessions
     and retrieval can scale beyond what fits in memory.
HOW: Wraps the chromadb Python client to create/query collections, add or upsert
     embedded chunks, and expose a retriever function compatible with the rest of
     the RAG pipeline.
EXAMPLE: After chunking and embedding a set of AI research documents, call
         get_chroma_client() then add_chunks_to_collection() to persist them,
         then chroma_search("what is RAG?", collection) to fetch the top matches.
"""

import chromadb

from local_embeddings import embed_texts


def get_chroma_client(path: str = "data/chroma_db"):
    """
    WHAT: Creates a ChromaDB persistent client pointed at a local directory.
    WHY: A persistent client saves the vector index to disk so embeddings do not
         have to be recomputed on every run.
    HOW: Calls chromadb.PersistentClient with the given path, which creates the
         directory if it does not already exist, then returns the client object.
    EXAMPLE: client = get_chroma_client("data/chroma_db")
             # The directory data/chroma_db/ now holds the SQLite + HNSW index.
    """
    client = chromadb.PersistentClient(path)
    return client


def get_or_create_collection(client, name: str = "rag_chunks"):
    """
    WHAT: Retrieves an existing ChromaDB collection by name or creates it if absent.
    WHY: Allows the same codebase to be run multiple times without duplicate
         collections — idempotent setup is important for experiment reproducibility.
    HOW: Delegates to client.get_or_create_collection(), which returns the
         existing collection when the name matches or allocates a fresh one.
    EXAMPLE: collection = get_or_create_collection(client, name="ai_docs")
             # Safe to call again later; returns the same collection.
    """
    collection = client.get_or_create_collection(name=name)
    return collection


def add_chunks_to_collection(collection, embedded_chunks: list[dict]) -> None:
    """
    WHAT: Inserts a list of embedded chunks into a ChromaDB collection.
    WHY: Bulk insertion is the fastest way to populate a new collection from a
         freshly embedded document set before any retrieval happens.
    HOW: Iterates over embedded_chunks, validates that each has an "embedding"
         field, then collects ids / metadatas / embeddings / documents into
         parallel lists and calls collection.add() in one batch.
    EXAMPLE: chunks = embed_chunks(chunk_documents(docs))
             add_chunks_to_collection(collection, chunks)
             # All chunks are now searchable inside ChromaDB.
    """
    ids = []
    metadatas = []
    embeddings = []
    documents = []
    for chunk in embedded_chunks:
        if "embedding" not in chunk:
            raise ValueError(f"Chunk with id {chunk['chunk_id']} does not have an embedding")
        ids.append(chunk["chunk_id"])
        metadatas.append({"chunk_id": chunk["chunk_id"], "text": chunk["text"], "filename": chunk["filename"], "start_char": chunk["start_char"], "end_char": chunk["end_char"], "doc_id": chunk["doc_id"]})
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])
    collection.add(ids=ids, metadatas=metadatas, embeddings=embeddings, documents=documents)
    return collection


def chroma_search(query: str, collection, top_k: int = 3) -> list[dict]:
    """
    WHAT: Performs semantic similarity search against a ChromaDB collection.
    WHY: ChromaDB's HNSW index makes approximate-nearest-neighbour lookup fast
         even over millions of chunks, making it practical for production RAG.
    HOW: Embeds the query string with embed_texts(), passes the resulting vector
         to collection.query(), converts the L2 distance returned by Chroma into
         a similarity score (score = 1 - distance), and returns a list of result
         dicts sorted by relevance.
    EXAMPLE: results = chroma_search("how does RAG work?", collection, top_k=3)
             # results[0] is the most semantically similar chunk, e.g. the RAG
             # overview paragraph from rag.txt with score ~0.92.
    """
    if query.strip() == "":
        raise ValueError("Query cannot be empty or whitespace")
    query_embedding = embed_texts([query])[0]
    queries = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    results = []
    for i, metadata in enumerate(queries["metadatas"][0]):
        distance = queries["distances"][0][i]
        score = 1 - distance
        results.append({
            "chunk_id": metadata["chunk_id"],
            "text": metadata["text"],
            "filename": metadata["filename"],
            "start_char": metadata["start_char"],
            "end_char": metadata["end_char"],
            "doc_id": metadata["doc_id"],
            "score": score
        })
    return results


def upsert_chunks_to_collection(collection, embedded_chunks: list[dict]):
    """
    WHAT: Inserts or updates embedded chunks in a ChromaDB collection.
    WHY: Upsert prevents duplicate-key errors when the same documents are
         re-processed (e.g. after re-chunking), making the ingestion pipeline
         safe to rerun without clearing the collection first.
    HOW: Validates required fields on each chunk, builds parallel id / metadata /
         embedding / document lists, then calls collection.upsert() so existing
         records are overwritten and new ones are created.
    EXAMPLE: # Re-run after changing chunk_size — no need to delete first.
             upsert_chunks_to_collection(collection, new_embedded_chunks)
             # Existing chunk IDs are updated; new IDs are inserted.
    """
    ids = []
    metadatas = []
    embeddings = []
    documents = []
    for chunk in embedded_chunks:
        required_fields = ["chunk_id", "text", "embedding"]
        for field in required_fields:
            if field not in chunk:
                raise ValueError(f"Chunk is missing required field: {field}")
        ids.append(chunk["chunk_id"])
        metadatas.append({ "chunk_id": chunk["chunk_id"], "filename": chunk.get("filename", "unknown"), "start_char": chunk.get("start_char", -1), "end_char": chunk.get("end_char", -1), "doc_id": chunk.get("doc_id", "unknown"),})
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])
    collection.upsert(ids=ids, metadatas=metadatas, embeddings=embeddings, documents=documents)
    return collection


def make_chroma_retriever(collection):
    """
    WHAT: Returns a retriever function that queries a ChromaDB collection.
    WHY: The rest of the pipeline expects a uniform retriever signature
         (query, chunks, top_k) so every retriever can be swapped in/out
         without changing calling code. This factory adapts Chroma to that
         interface.
    HOW: Defines an inner function retriever() that ignores the unused chunks
         argument and delegates to chroma_search(), then returns it as a
         callable.
    EXAMPLE: retriever = make_chroma_retriever(collection)
             results = retriever("what is a knowledge graph?", chunks=None, top_k=5)
             # Identical call signature to the manual semantic_search retriever.
    """
    def retriever(query: str, chunks=None, top_k: int = 3) -> list[dict]:
        return chroma_search(query, collection, top_k=top_k)
    return retriever


if __name__ == "__main__":
    print("Testing chroma_store functions:")
    print("  get_chroma_client: Creates a ChromaDB client")
    print("  get_or_create_collection: Gets or creates a named collection")
    print("  add_chunks_to_collection: Adds embedded chunks to a collection")
    print("  chroma_search: Performs semantic search using ChromaDB")
    print("  upsert_chunks_to_collection: Updates or inserts chunks")
    print("  make_chroma_retriever: Creates a retriever function")
    print("\nNote: These functions require ChromaDB to be installed and running.")
