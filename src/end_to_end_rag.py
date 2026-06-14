"""
WHAT: Provides a single-function end-to-end RAG pipeline that goes from raw
      documents to a generated answer.
WHY: Having one callable that handles the entire pipeline (chunk → embed → store →
     retrieve → answer) makes it easy to run experiments, build demos, and call
     the system from notebooks without manually wiring every stage together.
HOW: Orchestrates chunk_documents, embed_chunks, ChromaDB storage, the Chroma
     retriever, and a local Ollama LLM into a sequential pipeline. Returns a
     result dict that includes the answer, retrieved chunks, and pipeline metadata.
EXAMPLE: result = answer_from_documents(
             "What can agentic systems do?", documents, top_k=3)
         print(result["answer"])
         # "Agentic systems can plan steps, use tools, check results, and retry
         #  when the first attempt fails."
"""

from chunk_documents import chunk_documents
from local_embeddings import embed_chunks
from chroma_store import (
    get_chroma_client,
    get_or_create_collection,
    upsert_chunks_to_collection,
    make_chroma_retriever,
)
from local_llm import answer_question_local


def answer_from_documents(question: str, documents: list[dict], collection_name: str = "rag_chunks", chroma_path: str = "data/chroma_db", chunk_size: int = 500, overlap: int = 50, top_k: int = 3, model: str = "llama3.2:1b", min_score: float = 0.3, min_overlap: int = 1,) -> dict:
    """
    WHAT: Runs the complete RAG pipeline from raw documents to a final answer.
    WHY: A single end-to-end function lets callers skip manual pipeline assembly
         and makes it straightforward to evaluate the full system (not just
         individual components) with different configurations.
    HOW: 1. Chunks documents with chunk_documents() at the given chunk_size/overlap.
         2. Embeds all chunks with embed_chunks() using the local embedding model.
         3. Creates or connects to a ChromaDB client and collection at chroma_path.
         4. Upserts embedded chunks into the collection (safe to rerun).
         5. Builds a Chroma retriever via make_chroma_retriever().
         6. Calls answer_question_local() with the retriever to retrieve relevant
            chunks and generate an answer with the local LLM.
         7. Annotates the result with num_documents, num_chunks, and collection_name.
    EXAMPLE: result = answer_from_documents(
                 "What do knowledge graphs store?",
                 documents,
                 chunk_size=200, top_k=3, model="llama3.2:1b")
             # result["answer"] is the LLM's response grounded in the kg.txt chunk.
             # result["num_chunks"] shows how many chunks were created from the docs.
    """
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    embedded_chunks = embed_chunks(chunks)
    client = get_chroma_client(chroma_path)
    collection = get_or_create_collection(client, name=collection_name)
    upsert_chunks_to_collection(collection, embedded_chunks=embedded_chunks)
    retriever = make_chroma_retriever(collection)
    llama_answer = answer_question_local(
        question=question,
        retriever_fn=retriever,
        chunks=[],
        top_k=top_k,
        model=model,
        min_score=min_score,
        min_overlap=min_overlap,
    )
    llama_answer["num_documents"] = len(documents)
    llama_answer["num_chunks"] = len(chunks)
    llama_answer["collection_name"] = collection_name
    return llama_answer


if __name__ == "__main__":
    print("Testing end_to_end_rag:")
    print("  answer_from_documents: End-to-end RAG pipeline")
    print("    - Chunks documents")
    print("    - Embeds chunks with local embeddings")
    print("    - Stores in ChromaDB")
    print("    - Retrieves with Chroma retriever")
    print("    - Answers with local LLM")
    print("\nNote: Requires ChromaDB, embedding model, and local LLM to run.")
