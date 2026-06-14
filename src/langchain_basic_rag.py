"""
WHAT: Implements a complete RAG pipeline using LangChain abstractions.
WHY: LangChain provides high-level components (text splitters, vector stores,
     retrievers, LLM wrappers) that reduce boilerplate and make it easy to
     compare this approach with the hand-rolled pipeline in the rest of the repo.
HOW: Converts raw document dicts to LangChain Document objects, splits them
     with RecursiveCharacterTextSplitter, stores embeddings in a Chroma vector
     store via OllamaEmbeddings, retrieves relevant chunks, builds a prompt, and
     calls a local Ollama LLM to generate an answer.
EXAMPLE: Given a list of AI research documents, answer_with_langchain_rag(
         "What is retrieval-augmented generation?", documents) chunks them,
         embeds with nomic-embed-text, retrieves the top-3 chunks, and returns
         a structured answer dict including the LLM response and source docs.
"""

from langchain_core.documents import Document


def to_langchain_documents(documents: list[dict]) -> list:
    """
    WHAT: Converts a list of raw document dicts into LangChain Document objects.
    WHY: LangChain's pipeline components (splitters, vector stores, retrievers)
         only accept LangChain Document instances, so this conversion is the
         entry point into the LangChain ecosystem.
    HOW: Iterates over documents, extracts the "text" field as page_content and
         builds a metadata dict from "doc_id" and "filename", then wraps each in
         a langchain_core.documents.Document.
    EXAMPLE: docs = [{"text": "RAG retrieves chunks.", "doc_id": "rag",
                       "filename": "rag.txt"}]
             lc_docs = to_langchain_documents(docs)
             # lc_docs[0].page_content == "RAG retrieves chunks."
    """
    lc_documents = []
    for document in documents:
        page_content = document["text"]
        metadata = {"doc_id": document["doc_id"], "filename": document["filename"],}
        lc_document = Document(page_content=page_content,
                               metadata=metadata)
        lc_documents.append(lc_document)
    return lc_documents


from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_langchain_documents(lc_documents: list, chunk_size: int = 500, chunk_overlap: int = 50,) -> list:
    """
    WHAT: Splits LangChain Document objects into smaller overlapping chunks.
    WHY: LLMs have limited context windows, and smaller chunks improve retrieval
         precision — a chunk about one topic is more likely to match a narrow
         query than a large document covering many topics.
    HOW: Instantiates RecursiveCharacterTextSplitter with the given chunk_size
         and chunk_overlap, then calls split_documents() which recursively tries
         paragraph, sentence, and word boundaries to keep splits natural.
    EXAMPLE: chunks = split_langchain_documents(lc_docs, chunk_size=300,
                                                chunk_overlap=30)
             # A 900-char document becomes ~4 overlapping chunks of ≤300 chars.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,)
    chunks = splitter.split_documents(lc_documents)
    return chunks


from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def build_langchain_vectorstore(chunks: list, persist_directory: str = "data/langchain_chroma", collection_name: str = "langchain_rag",):
    """
    WHAT: Embeds document chunks and stores them in a persistent Chroma vector store.
    WHY: The vector store is the backbone of retrieval — it maps each chunk to a
         high-dimensional embedding so that semantically similar chunks can be
         found by cosine distance at query time.
    HOW: Creates an OllamaEmbeddings instance using the nomic-embed-text model,
         then calls Chroma.from_documents() which embeds every chunk and writes
         the resulting vectors plus metadata to disk at persist_directory.
    EXAMPLE: vectorstore = build_langchain_vectorstore(chunks,
                 persist_directory="data/langchain_chroma",
                 collection_name="ai_research")
             # Vectors are now persisted; future runs can reload without re-embedding.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,)
    return vectorstore


def build_langchain_retriever(vectorstore, k: int = 3):
    """
    WHAT: Wraps a Chroma vector store as a LangChain retriever object.
    WHY: LangChain retrievers expose a standard .invoke(query) interface that
         downstream chain components can call without knowing the underlying
         store type, enabling easy swapping of retrieval backends.
    HOW: Calls vectorstore.as_retriever() with search_kwargs={"k": k}, which
         returns the top-k most similar chunks for any query string.
    EXAMPLE: retriever = build_langchain_retriever(vectorstore, k=5)
             docs = retriever.invoke("how does attention work in transformers?")
             # Returns 5 LangChain Document objects ranked by similarity.
    """
    return vectorstore.as_retriever(search_kwargs={"k": k})


from langchain_ollama import OllamaLLM


def answer_with_langchain_rag(question: str, documents: list[dict], model: str = "llama3.2:1b", chunk_size: int = 500, chunk_overlap: int = 50, k: int = 3,) -> dict:
    """
    WHAT: Runs an end-to-end LangChain RAG pipeline: chunk → embed → retrieve → answer.
    WHY: Bundles the full pipeline into one callable so notebooks and scripts can
         get an answer from raw documents with a single function call, useful for
         quick experiments and baselines.
    HOW: 1. Converts documents to LangChain format via to_langchain_documents().
         2. Splits them into chunks via split_langchain_documents().
         3. Embeds and stores chunks in Chroma via build_langchain_vectorstore().
         4. Builds a retriever and retrieves top-k docs for the question.
         5. Joins retrieved content into a context string.
         6. Formats a strict prompt that instructs the LLM to answer only from context.
         7. Calls OllamaLLM.invoke() and returns a result dict.
    EXAMPLE: result = answer_with_langchain_rag(
                 "What can agentic systems do?", documents, k=3)
             # result["answer"] might be: "Agentic systems can plan steps, use
             # tools, check results, and retry when the first attempt fails."
    """
    if question.strip() == "":
        raise ValueError("Question cannot be empty or whitespace")
    lc_documents = to_langchain_documents(documents=documents)
    chunks = split_langchain_documents(lc_documents=lc_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vectors = build_langchain_vectorstore(chunks=chunks, persist_directory="data/langchain_chroma", collection_name="langchain_rag")
    retriever = build_langchain_retriever(vectorstore=vectors, k=k)
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = f"""Answer the question using only the context below.
    If the context does not contain the answer, say: "I don't know based on the provided context."
    Context: {context}
    Question: {question}
    Answer:"""
    llm = OllamaLLM(model=model)
    answer = llm.invoke(prompt)
    return {
        "question": question,
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "num_chunks": len(chunks),
    }


def langchain_docs_to_results(docs: list) -> list[dict]:
    """
    WHAT: Converts a list of LangChain Document objects into plain result dicts.
    WHY: The evaluation and comparison utilities in this repo expect a uniform
         dict schema (chunk_id, doc_id, filename, text, score). This adapter
         bridges LangChain's Document format to that shared contract.
    HOW: Iterates over docs, extracts metadata fields with safe fallbacks, sets
         score to 0.0 (LangChain retrievers do not always expose distances), and
         returns a list of result dicts.
    EXAMPLE: lc_docs = retriever.invoke("what is a knowledge graph?")
             results = langchain_docs_to_results(lc_docs)
             # results[0] == {"chunk_id": "langchain_0", "doc_id": "kg",
             #                "filename": "kg.txt", "text": "...", "score": 0.0}
    """
    results = []
    for i, doc in enumerate(docs):
        metadata = doc.metadata
        chunk_id = metadata.get("chunk_id", f"langchain_{i}")
        doc_id = metadata.get("doc_id", "unknown")
        filename = metadata.get("filename", "unknown")
        text = doc.page_content
        score = 0.0
        results.append({"chunk_id": chunk_id, "doc_id": doc_id, "filename": filename, "text": text, "score": score})
    return results


def make_langchain_retriever_fn(langchain_retriever):
    """
    WHAT: Wraps a LangChain retriever as a plain Python function with the
          standard (query, chunks, top_k) signature used throughout this repo.
    WHY: Experiment and evaluation utilities call retrievers via a uniform
         signature. This factory lets a LangChain retriever participate in those
         utilities without any changes to the calling code.
    HOW: Defines an inner retriever_fn that calls langchain_retriever.invoke(),
         converts the resulting LangChain Documents with langchain_docs_to_results(),
         slices to top_k, and returns the list.
    EXAMPLE: lc_retriever = build_langchain_retriever(vectorstore, k=5)
             retriever_fn = make_langchain_retriever_fn(lc_retriever)
             results = retriever_fn("how does RAG help LLMs?", chunks=None, top_k=3)
             # Identical call signature to chroma_search or semantic_search wrappers.
    """
    def retriever_fn(query: str, chunks=None, top_k: int = 3) -> list[dict]:
        docs = langchain_retriever.invoke(query)
        results = langchain_docs_to_results(docs)
        return results[:top_k]
    return retriever_fn


if __name__ == "__main__":
    print("Testing langchain_basic_rag functions:")
    print("  to_langchain_documents: Convert dict documents to LangChain format")
    print("  split_langchain_documents: Split documents with RecursiveCharacterTextSplitter")
    print("  build_langchain_vectorstore: Build Chroma vectorstore from LangChain docs")
    print("  build_langchain_retriever: Create retriever from vectorstore")
    print("  answer_with_langchain_rag: End-to-end LangChain RAG pipeline")
    print("  langchain_docs_to_results: Convert LangChain docs to result dicts")
    print("  make_langchain_retriever_fn: Wrap LangChain retriever for compatibility")
    print("\nNote: Requires LangChain, Chroma, and Ollama to be installed.")
