"""
WHAT: Answerability gating module — decides whether retrieved chunks are good enough to
      justify sending a prompt to the LLM, and builds a gated RAG input accordingly.
WHY: Without a quality gate, an LLM receives irrelevant or low-confidence context and
     produces confident-sounding but wrong answers (hallucinations). Checking answerability
     before generation lets us short-circuit with an honest "I don't know" instead.
HOW: is_answerable inspects each retrieved chunk's similarity score and token overlap with
     the query. build_gated_rag_input runs retrieval, calls is_answerable, then either
     returns a fallback response or a fully-formed RAG prompt dict.
EXAMPLE: A user asks "What is the dropout rate in the model?" but the indexed documents
         are all about dataset preprocessing. The low scores and zero token overlap cause
         is_answerable to return False, and build_gated_rag_input returns a safe fallback
         instead of a hallucinated answer about dropout.
"""

from reranking import simple_tokenize
from rag_pipeline import format_context, build_rag_prompt


def is_answerable(
    query: str,
    retrieved_chunks: list[dict],
    min_score: float = 0.3,
    min_overlap: int = 1,
) -> bool:
    """
    WHAT: Evaluates whether at least one retrieved chunk is relevant enough to the query
          to make a reliable LLM answer possible.
    WHY: Retrieval can return results even when nothing in the knowledge base matches the
         query — similarity search always returns something. This function acts as a
         quality gate so that only genuinely relevant context reaches the LLM.
    HOW:
        1. Iterate over each retrieved chunk.
        2. Read the chunk's relevance score, preferring "rerank_score" over "score"
           (reranking is more accurate than initial embedding similarity).
        3. Tokenize both the query and the chunk text using simple_tokenize.
        4. Compute token overlap as the size of the intersection of both token sets.
        5. If any chunk meets BOTH thresholds (score >= min_score AND overlap >= min_overlap),
           return True immediately.
        6. If no chunk passes, return False.
    EXAMPLE: Query "What is backpropagation?" against a chunk about "gradient descent and
             backprop" with rerank_score=0.75. simple_tokenize finds "backpropagation" in
             the chunk tokens (overlap=1), score >= 0.3, so is_answerable returns True.
    """
    for chunk in retrieved_chunks:
        score = chunk.get("rerank_score", chunk.get("score", 0))
        query_tokens = simple_tokenize(query)
        chunk_tokens = simple_tokenize(chunk["text"])
        overlap = len(query_tokens.intersection(chunk_tokens))
        if score >= min_score and overlap >= min_overlap:
            return True
    return False


def build_gated_rag_input(
    question: str,
    retriever_fn,
    chunks: list[dict],
    top_k: int = 3,
    min_score: float = 0.3,
    min_overlap: int = 1,
) -> dict:
    """
    WHAT: Runs the full retrieval pipeline with an answerability gate — returns either a
          complete RAG prompt dict (if answerable) or a structured fallback dict (if not).
    WHY: Separating "can we answer this?" from "what is the answer?" makes the pipeline
         safe by default. Callers receive a consistent dict shape in both cases, so
         downstream code (like local_llm.answer_from_rag_input) handles both uniformly.
    HOW:
        1. Call retriever_fn to fetch the top_k most relevant chunks.
        2. Pass retrieved chunks to is_answerable with the given score/overlap thresholds.
        3. If NOT answerable: return a dict with answerable=False, empty context,
           prompt=None, and a pre-written fallback_answer string.
        4. If answerable: format the chunks into a context string, build the LLM prompt,
           and return a dict with answerable=True and all pipeline artifacts populated.
    EXAMPLE: Question "What is the batch size used in training?" retrieves 3 chunks, one
             of which mentions "batch size 32" with rerank_score=0.82 and keyword overlap.
             is_answerable returns True, and build_gated_rag_input returns a full prompt
             dict ready for the LLM. If all chunks score below 0.3, the fallback dict is
             returned and no LLM call is needed.
    """
    retrieved_chunks = retriever_fn(question, chunks, top_k=top_k)
    answerable = is_answerable(question, retrieved_chunks, min_score, min_overlap)
    if answerable is False:
        return {
            "question": question,
            "answerable": False,
            "retrieved_chunks": retrieved_chunks,
            "context": "",
            "prompt": None,
            "fallback_answer": "I don't know based on the provided context.",
        }
    else:
        context = format_context(retrieved_chunks)
        prompt = build_rag_prompt(question, context)
    return {
        "question": question,
        "answerable": True,
        "retrieved_chunks": retrieved_chunks,
        "context": context,
        "prompt": prompt,
        "fallback_answer": None,
    }


if __name__ == "__main__":
    test_chunks = [
        {"chunk_id": "1", "text": "Python is a programming language", "start_char": 0, "end_char": 32, "doc_id": "doc1", "filename": "file1.txt", "score": 0.8, "rerank_score": 0.8},
        {"chunk_id": "2", "text": "Java is also a programming language", "start_char": 32, "end_char": 67, "doc_id": "doc1", "filename": "file1.txt", "score": 0.2, "rerank_score": 0.2},
    ]

    print("Testing is_answerable:")
    query = "What is Python?"
    answerable = is_answerable(query, test_chunks, min_score=0.3, min_overlap=1)
    print(f"  Query: '{query}'")
    print(f"  Answerable: {answerable}")

    print("\nTesting with low score chunks:")
    low_score_chunks = [
        {"chunk_id": "3", "text": "Something unrelated", "start_char": 0, "end_char": 19, "score": 0.1}
    ]
    answerable_low = is_answerable(query, low_score_chunks, min_score=0.5)
    print(f"  Low score chunks answerable: {answerable_low}")

    print("\nTesting edge cases:")
    empty_results = is_answerable("test query", [])
    print(f"  Empty chunks answerable: {empty_results}")
