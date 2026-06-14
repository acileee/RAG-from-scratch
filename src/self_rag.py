"""
WHAT: Self-RAG module — implements an iterative retrieval strategy that automatically
      rewrites the query and retries retrieval when the first attempt is not answerable.
WHY: Standard RAG uses a single retrieval pass. If the original query is phrased in a
     way that doesn't match the indexed documents, retrieval fails silently. Self-RAG
     adds a feedback loop: assess → rewrite → retry, giving the pipeline a second
     chance before admitting it cannot answer.
HOW: self_rag_retrieve loops up to max_attempts times, using the raw question on the
     first attempt and a rewritten version on subsequent ones, stopping as soon as an
     answerable set of chunks is found. build_self_rag_input wraps that into a full
     RAG input dict. answer_question_self_rag adds local LLM inference on top.
EXAMPLE: Query "transformer attention mechanism" fails to retrieve good chunks on the
         first pass (low scores). query_rewriting reformulates it as "how does the
         attention mechanism work in transformer models?" — the second pass retrieves
         high-scoring chunks and the question becomes answerable.
"""

from answerability import is_answerable
from query_rewriting import rewrite_query
from rag_pipeline import format_context, build_rag_prompt
from local_llm import answer_from_rag_input


def self_rag_retrieve(
    question: str,
    retriever_fn,
    chunks: list[dict],
    max_attempts: int = 2,
    top_k: int = 3,
    min_score: float = 0.3,
    min_overlap: int = 1,
) -> dict:
    """
    WHAT: Iteratively retrieves chunks for a question, rewriting the query on each
          failed attempt, and returns a summary of all attempts alongside the best
          chunks found.
    WHY: A single retrieval pass can miss relevant documents when the user's phrasing
         differs from the indexed text. Iterative query rewriting increases the chance
         of surfacing answerable content without requiring the user to reformulate
         manually.
    HOW:
        1. Raise ValueError if the question is empty.
        2. Loop from attempt 1 to max_attempts (inclusive).
        3. On attempt 1, use the original question as the query.
        4. On subsequent attempts, call rewrite_query(question) to generate an
           alternative phrasing of the same information need.
        5. Retrieve top_k chunks with retriever_fn(query, chunks, top_k).
        6. Call is_answerable to check whether the retrieved chunks meet the
           quality thresholds.
        7. Record the attempt details (attempt number, query used, answerability,
           retrieved chunks) in an attempts list.
        8. Break out of the loop as soon as an answerable set is found.
        9. Return a dict with the original question, final answerability flag,
           all attempt records, and the final set of chunks.
    EXAMPLE: Question "BERT pre-training objectives" retrieves low-quality chunks on
             attempt 1. rewrite_query produces "What tasks is BERT pre-trained on?",
             attempt 2 retrieves a high-scoring chunk about masked language modelling,
             is_answerable returns True, and the loop exits. The returned dict contains
             both attempt records so the caller can see exactly what happened.
    """
    if question.strip() == "":
        raise ValueError("question cannot be empty")
    attempts = []
    final_chunks = []
    final_answerable = False
    for attempt in range(1, max_attempts + 1):
        if attempt == 1:
            query = question
        else:
            query = rewrite_query(question)
        retrieved_chunks = retriever_fn(query, chunks, top_k=top_k)
        answerable = is_answerable(
            question,
            retrieved_chunks,
            min_score=min_score,
            min_overlap=min_overlap,
        )
        attempts.append({
            "attempt": attempt,
            "query": query,
            "answerable": answerable,
            "retrieved_chunks": retrieved_chunks,
        })
        final_chunks = retrieved_chunks
        final_answerable = answerable
        if answerable:
            break
    return {
        "question": question,
        "answerable": final_answerable,
        "attempts": attempts,
        "final_chunks": final_chunks,
    }


def build_self_rag_input(
    question: str,
    retriever_fn,
    chunks: list[dict],
    max_attempts: int = 2,
    top_k: int = 3,
    min_score: float = 0.3,
    min_overlap: int = 1,
) -> dict:
    """
    WHAT: Runs the Self-RAG retrieval loop and converts the result into a complete RAG
          input dict — either a fully-formed prompt (if answerable) or a structured
          fallback (if all attempts fail).
    WHY: Callers (such as answer_question_self_rag and the LLM layer) need a consistent
         dict interface regardless of how many retrieval attempts were made or whether
         the question was ultimately answerable. This function provides that consistent
         contract while preserving all attempt history for transparency.
    HOW:
        1. Call self_rag_retrieve to run the iterative retrieval loop.
        2. If the final result is not answerable, return a dict with answerable=False,
           empty context, prompt=None, a fallback_answer string, and the attempts list.
        3. If answerable, call format_context on the final chunks to build the context
           string, then call build_rag_prompt to create the LLM prompt.
        4. Return a dict with answerable=True, the context, the prompt, the attempts
           list, and the retrieved_chunks.
    EXAMPLE: For question "What loss function is used?", self_rag_retrieve runs two
             attempts and finds answerable chunks on the second. build_self_rag_input
             returns:
             {"question": "What loss function is used?", "answerable": True,
              "attempts": [{attempt:1, ...}, {attempt:2, ...}],
              "retrieved_chunks": [...], "context": "...", "prompt": "...",
              "fallback_answer": None}
    """
    retrieval_result = self_rag_retrieve(
        question=question,
        retriever_fn=retriever_fn,
        chunks=chunks,
        max_attempts=max_attempts,
        top_k=top_k,
        min_score=min_score,
        min_overlap=min_overlap,
    )
    attempts = retrieval_result["attempts"]
    final_chunks = retrieval_result["final_chunks"]
    if not retrieval_result["answerable"]:
        return {
            "question": question,
            "answerable": False,
            "attempts": attempts,
            "retrieved_chunks": final_chunks,
            "context": "",
            "prompt": None,
            "fallback_answer": "I don't know based on the provided context.",
        }
    context = format_context(final_chunks)
    prompt = build_rag_prompt(question, context)
    return {
        "question": question,
        "answerable": True,
        "attempts": attempts,
        "retrieved_chunks": final_chunks,
        "context": context,
        "prompt": prompt,
        "fallback_answer": None,
    }


def answer_question_self_rag(
    question: str,
    retriever_fn,
    chunks: list[dict],
    model: str = "llama3.2:1b",
    max_attempts: int = 2,
    top_k: int = 3,
    min_score: float = 0.3,
    min_overlap: int = 1,
) -> dict:
    """
    WHAT: Top-level Self-RAG entry point — combines iterative retrieval with local LLM
          inference and returns a final answer dict with full attempt history.
    WHY: This is the "production-ready" version of a RAG query for local inference. It
         gives the system two shots at finding relevant context before falling back, and
         produces an answer grounded in whatever it finds — all in one call.
    HOW:
        1. Call build_self_rag_input with all parameters to run the retrieval loop and
           produce a structured RAG input dict (answerable or not).
        2. Pass that dict to answer_from_rag_input, which either returns the fallback
           or calls the Ollama model to generate a grounded answer.
        3. Return the final answer dict (question, answer, answerable, retrieved_chunks).
    EXAMPLE: For a corpus of medical research papers, call:
             answer_question_self_rag(
                 "What side effects were reported in the trial?",
                 retriever_fn=bm25_retriever,
                 chunks=all_chunks,
                 max_attempts=2,
                 top_k=5,
                 model="llama3.2:1b"
             )
             If the first retrieval attempt fails, the query is rewritten and retried.
             If the second attempt succeeds, Ollama answers from the retrieved context.
    """
    rag_input = build_self_rag_input(
        question=question,
        retriever_fn=retriever_fn,
        chunks=chunks,
        max_attempts=max_attempts,
        top_k=top_k,
        min_score=min_score,
        min_overlap=min_overlap,
    )
    return answer_from_rag_input(rag_input, model=model)


if __name__ == "__main__":
    print("Testing self_rag functions:")

    test_chunks = [
        {"chunk_id": "1", "text": "Python is useful for AI", "start_char": 0, "end_char": 23, "doc_id": "py", "filename": "f1.txt", "score": 0.8, "rerank_score": 0.8},
        {"chunk_id": "2", "text": "Java is also powerful", "start_char": 23, "end_char": 44, "doc_id": "java", "filename": "f2.txt", "score": 0.3, "rerank_score": 0.3},
    ]

    def dummy_retriever(q, chunks, top_k=3):
        return chunks[:top_k]

    print("\nTesting self_rag_retrieve:")
    result = self_rag_retrieve("What is Python?", dummy_retriever, test_chunks, max_attempts=1)
    print(f"  Question: {result['question']}")
    print(f"  Answerable: {result['answerable']}")
    print(f"  Attempts: {len(result['attempts'])}")

    print("\nTesting build_self_rag_input:")
    rag_input = build_self_rag_input("What is Python?", dummy_retriever, test_chunks, max_attempts=1)
    print(f"  Question: {rag_input['question']}")
    print(f"  Answerable: {rag_input['answerable']}")
    print(f"  Keys: {list(rag_input.keys())}")

    print("\nTesting error handling:")
    try:
        self_rag_retrieve("", dummy_retriever, test_chunks)
        print("  ERROR: Should raise ValueError for empty question")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
