"""
WHAT: Core RAG pipeline module — formats retrieved chunks into context and builds LLM prompts.
WHY: A RAG system needs a clean, consistent way to turn raw retrieved text into structured
     prompts the LLM can act on. Centralising this logic makes the pipeline easy to swap
     out or extend (e.g. different prompt templates, context layouts).
HOW: Three composable functions — format_context assembles source-labelled text blocks,
     build_rag_prompt wraps them with an instruction template, and build_rag_input
     orchestrates retrieve → format → prompt in one call.
EXAMPLE: A user asks "What activation function does the paper recommend?" — build_rag_input
         retrieves the top-3 matching chunks from a machine-learning PDF corpus, formats
         them with source labels, and returns a ready-to-send LLM prompt grounded in those chunks.
"""

def format_context(retrieved_chunks: list[dict]) -> str:
    """
    WHAT: Converts a list of retrieved chunk dicts into a single, human-readable context string
          with source attribution headers above each chunk.
    WHY: The LLM needs context presented in a clear, labelled format so it can both use the text
         and reference where the information came from — enabling source traceability in answers.
    HOW:
        1. Return an empty string immediately if no chunks were provided.
        2. Iterate over each chunk dict.
        3. Raise ValueError if a chunk is missing the required "text" field.
        4. Extract "filename" and "chunk_id" metadata (defaulting to "unknown" if absent).
        5. Format each chunk as "[source:<filename> - chunk:<chunk_id>]\\n<text>".
        6. Join all formatted blocks with double newlines for visual separation.
    EXAMPLE: Given two chunks from "attention_paper.txt", the output looks like:
        [source:attention_paper.txt - chunk:attention_001]
        Attention is all you need...

        [source:attention_paper.txt - chunk:attention_002]
        The transformer architecture...
    """
    if not retrieved_chunks:
        return ""

    formatted_block = []
    for chunk in retrieved_chunks:
        if "text" not in chunk:
            raise ValueError(f"Chunk with id {chunk['chunk_id']} does not have 'text' field")
        else:
            filename = chunk.get("filename", "unknown")
            chunk_id = chunk.get("chunk_id", "unknown")
            block = f"[source:{filename} - chunk:{chunk_id}]\n{chunk['text']}"
            formatted_block.append(block)

    context = "\n\n".join(formatted_block)
    return context

def build_rag_prompt(question: str, context: str) -> str:
    """
    WHAT: Builds the final instruction prompt that tells the LLM to answer a question using
          only the supplied context string — no outside knowledge allowed.
    WHY: Without explicit instructions, an LLM will blend retrieved context with its own
         training-data knowledge, producing hallucinations. A strict "use only the context"
         instruction keeps answers grounded and auditable.
    HOW:
        1. Validate that question and context are both non-empty (raise ValueError otherwise).
        2. Embed the context and question into a fixed instruction template.
        3. The template tells the LLM: use only the context; if insufficient, say "I don't know".
        4. Return the fully assembled prompt string ready to send to any LLM.
    EXAMPLE: For question "What optimizer did the authors use?" and a context block from
             a research paper, the returned prompt will read:
             "Answer the question using only the provided context. If the context does not
              contain the answer, say: 'I don't know based on the provided context.'
              Context: [source:paper.txt - chunk:paper_003]\\nThe authors used Adam...
              Question: What optimizer did the authors use?
              Answer:"
    """
    if question.strip() == "":
        raise ValueError("Question cannot be empty or whitespace")
    if context.strip() == "":
        raise ValueError("Context cannot be empty or whitespace")

    prompt = f"""Answer the question using only the provided context. If the context does not contain the answer, say: "I don't know based on the provided context."

Context: {context}

Question: {question}

Answer:"""
    return prompt

def build_rag_input(question: str, retriever_fn, chunks: list[dict], top_k: int = 3) -> dict:
    """
    WHAT: Orchestrates the full RAG pipeline — retrieval, context formatting, and prompt
          construction — and returns a single dict containing every intermediate artifact.
    WHY: Bundling all pipeline stages into one call gives callers a single entry point and
         makes every intermediate result (retrieved chunks, raw context, final prompt)
         available for inspection, logging, or evaluation without extra work.
    HOW:
        1. RETRIEVE — call retriever_fn(question, chunks, top_k) to get the most relevant chunks.
        2. Raise ValueError if the retriever returns None.
        3. FORMAT — pass retrieved chunks to format_context() to produce a labelled context string.
        4. PROMPT — pass question + context to build_rag_prompt() to produce the final LLM prompt.
        5. RETURN — pack question, retrieved_chunks, context, and prompt into a dict.
    EXAMPLE: For a corpus of ML paper chunks and question "What is the learning rate used?",
             build_rag_input returns:
             {
               "question": "What is the learning rate used?",
               "retrieved_chunks": [{"chunk_id": "paper_002", "text": "lr=0.001...", ...}],
               "context": "[source:paper.txt - chunk:paper_002]\\nlr=0.001...",
               "prompt": "Answer the question using only the provided context..."
             }
    """
    retrieved_chunks = retriever_fn(question, chunks, top_k=top_k)
    if retrieved_chunks is None:
        raise ValueError("Retriever function returned None")

    context = format_context(retrieved_chunks)
    prompt = build_rag_prompt(question, context)

    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "context": context,
        "prompt": prompt,
    }

def build_chroma_rag_input(question: str, collection, top_k: int = 3) -> dict:
    """
    WHAT: Convenience wrapper around build_rag_input that uses a ChromaDB collection as the
          retrieval backend instead of an in-memory list of chunks.
    WHY: ChromaDB manages its own storage and vector search, so it cannot accept a flat chunk
         list. This wrapper adapts the generic pipeline to ChromaDB by creating a
         collection-specific retriever function, hiding that detail from callers.
    HOW:
        1. Import make_chroma_retriever from chroma_store (lazy import to avoid hard dependency).
        2. Wrap the ChromaDB collection in a retriever function via make_chroma_retriever.
        3. Delegate to build_rag_input with an empty chunks list (ChromaDB handles storage).
        4. Return the resulting RAG input dict.
    EXAMPLE: After indexing a set of research papers into a ChromaDB collection called
             "papers_collection", call build_chroma_rag_input("What is RLHF?", papers_collection)
             to retrieve the top-3 relevant chunks from Chroma and get a ready-to-use prompt dict.
    """
    from chroma_store import make_chroma_retriever
    chroma_retriever = make_chroma_retriever(collection)
    rag_input = build_rag_input(question, chroma_retriever, chunks=[], top_k=top_k)
    return rag_input

if __name__ == "__main__":
    test_chunks = [
        {"chunk_id": "1", "text": "Python is powerful", "start_char": 0, "end_char": 18, "doc_id": "d1", "filename": "f1.txt"},
        {"chunk_id": "2", "text": "Python is easy to learn", "start_char": 18, "end_char": 41, "doc_id": "d1", "filename": "f1.txt"},
    ]

    print("Testing format_context:")
    context = format_context(test_chunks)
    print(f"  Context length: {len(context)}")
    print(f"  Preview: {context[:100]}...")

    print("\nTesting build_rag_prompt:")
    question = "Is Python useful?"
    prompt = build_rag_prompt(question, context)
    print(f"  Prompt length: {len(prompt)}")
    print(f"  Contains question: {question in prompt}")

    print("\nTesting build_rag_input (with dummy retriever):")
    def dummy_retriever(q, chunks, top_k=3):
        return test_chunks[:top_k]

    rag_input = build_rag_input(question, dummy_retriever, test_chunks, top_k=2)
    print(f"  Keys: {list(rag_input.keys())}")
    print(f"  Question: {rag_input['question']}")
    print(f"  Retrieved chunks: {len(rag_input['retrieved_chunks'])}")
    print(f"  Context length: {len(rag_input['context'])}")

    print("\nTesting error handling:")
    try:
        build_rag_prompt("", "some context")
        print("  ERROR: Should raise ValueError for empty question")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

    try:
        build_rag_prompt("question", "")
        print("  ERROR: Should raise ValueError for empty context")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
