"""
WHAT: Local LLM inference module — sends RAG prompts to a locally running Ollama instance
      and returns structured answer dicts with full provenance information.
WHY: Running inference locally (via Ollama) keeps data private, removes API costs, and
     makes the RAG pipeline fully self-contained. This module bridges the RAG prompt
     construction layer and the actual LLM, handling the gated answerability check so
     the LLM is never called when retrieval quality is insufficient.
HOW: call_ollama posts a prompt to the Ollama HTTP API and extracts the response text.
     answer_from_rag_input wraps it with answerability gating — returning a fallback
     immediately if the RAG input is marked unanswerable. answer_question_local combines
     the entire pipeline (gated retrieval + local inference) in one call.
EXAMPLE: A user asks "What regularisation technique is described in section 3?" against
         a corpus of ML papers. answer_question_local retrieves relevant chunks, checks
         answerability, builds the prompt, sends it to llama3.2:1b running locally, and
         returns {"question": ..., "answer": "Dropout with p=0.5 ...", "answerable": True,
         "retrieved_chunks": [...]}.
"""

import requests


def call_ollama(prompt: str, model: str = "llama3.2:1b") -> str:
    """
    WHAT: Sends a text prompt to a locally running Ollama model and returns the model's
          response as a plain string.
    WHY: Ollama exposes a simple HTTP API that allows any Python code to run powerful
         open-source LLMs locally without GPU cloud costs or data-privacy concerns.
         This function abstracts the HTTP detail so callers only handle strings.
    HOW:
        1. Raise ValueError if the prompt is empty or whitespace-only.
        2. POST a JSON payload to Ollama's /api/generate endpoint at localhost:11434,
           with stream=False to get the full response in one shot.
        3. Raise RuntimeError (with a human-readable message) if the request fails,
           wrapping the original requests exception for debuggability.
        4. Parse the JSON response and raise RuntimeError if the "response" key is absent.
        5. Return the string value at data["response"].
    EXAMPLE: After indexing a biology textbook and building a RAG prompt asking
             "What is the role of mitochondria?", pass the prompt to call_ollama and
             receive a grounded answer like "According to the context, mitochondria are
             the powerhouse of the cell, producing ATP via oxidative phosphorylation."
    """
    if prompt.strip() == "":
        raise ValueError("prompt cannot be empty")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            "Failed to call Ollama. Make sure Ollama is running and the model is installed."
        ) from e

    data = response.json()
    if "response" not in data:
        raise RuntimeError(f"Ollama response did not contain 'response': {data}")
    answer = data["response"]
    return answer


def answer_from_rag_input(rag_input: dict, model: str = "llama3.2:1b") -> dict:
    """
    WHAT: Takes a fully-formed RAG input dict (as produced by build_gated_rag_input or
          build_self_rag_input) and either returns the pre-computed fallback answer or
          calls the local LLM to generate a grounded answer.
    WHY: Decoupling "build the prompt" from "call the LLM" makes each stage independently
         testable and lets the gating logic in answerability.py prevent unnecessary LLM
         calls when retrieved context is too weak to support a reliable answer.
    HOW:
        1. If rag_input["answerable"] is False, return a dict with the fallback_answer
           directly — no LLM call is made.
        2. Extract the "prompt" field; raise ValueError if it is missing or empty.
        3. Call call_ollama(prompt, model) to get the generated answer string.
        4. Return a dict containing question, answer, answerable=True, and retrieved_chunks.
    EXAMPLE: After build_gated_rag_input returns an answerable=True dict for the question
             "What dataset was used for pre-training?", pass it to answer_from_rag_input
             and receive {"question": "What dataset was used for pre-training?",
             "answer": "The model was pre-trained on The Pile...", "answerable": True,
             "retrieved_chunks": [...]}.
    """
    if rag_input.get("answerable") is False:
        return {
            "question": rag_input["question"],
            "answer": rag_input.get("fallback_answer", "I don't know based on the provided context."),
            "answerable": False,
            "retrieved_chunks": rag_input.get("retrieved_chunks", []),
        }

    prompt = rag_input.get("prompt")
    if prompt is None or prompt.strip() == "":
        raise ValueError("RAG input must contain a non-empty prompt")

    answer = call_ollama(prompt, model=model)

    return {
        "question": rag_input["question"],
        "answer": answer,
        "answerable": True,
        "retrieved_chunks": rag_input.get("retrieved_chunks", []),
    }


from answerability import build_gated_rag_input


def answer_question_local(
    question: str,
    retriever_fn,
    chunks: list[dict],
    top_k: int = 3,
    model: str = "llama3.2:1b",
    min_score: float = 0.3,
    min_overlap: int = 1,
) -> dict:
    """
    WHAT: End-to-end convenience function that runs gated retrieval and local LLM inference
          in a single call, returning a complete answer dict with full provenance.
    WHY: Most callers want a single function that goes from "question + corpus" to "answer"
         without manually orchestrating build_gated_rag_input and answer_from_rag_input.
         This function provides that high-level entry point while keeping each internal
         stage independently accessible for learning or debugging.
    HOW:
        1. Call build_gated_rag_input with all retrieval and answerability parameters
           to produce a gated RAG input dict.
        2. Pass that dict to answer_from_rag_input, which either returns the fallback
           or calls the local Ollama model.
        3. Return the answer dict (question, answer, answerable, retrieved_chunks).
    EXAMPLE: After chunking and embedding a collection of climate science papers, call:
             answer_question_local(
                 "What is the projected sea level rise by 2100?",
                 retriever_fn=bm25_retriever,
                 chunks=all_chunks,
                 top_k=5,
                 model="llama3.2:1b"
             )
             The function retrieves the 5 best chunks, gates on quality, and — if
             answerable — returns the Ollama-generated answer grounded in those chunks.
    """
    rag_input = build_gated_rag_input(
        question=question,
        retriever_fn=retriever_fn,
        chunks=chunks,
        top_k=top_k,
        min_score=min_score,
        min_overlap=min_overlap,
    )

    return answer_from_rag_input(rag_input, model=model)


if __name__ == "__main__":
    print("Testing local_llm functions:")

    print("\nTesting call_ollama:")
    print("  This function calls Ollama API at localhost:11434")
    print("  Requires: Ollama running and specified model installed")

    print("\nTesting error handling:")
    try:
        call_ollama("")
        print("  ERROR: Should raise ValueError for empty prompt")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

    print("\nTesting answer_from_rag_input:")
    test_rag_input_unanswerable = {
        "question": "What is Python?",
        "answerable": False,
        "fallback_answer": "I don't know based on the provided context.",
        "retrieved_chunks": []
    }

    result = answer_from_rag_input(test_rag_input_unanswerable)
    print(f"  Unanswerable question returns fallback: {result['answer']}")
    print(f"  Answerable: {result['answerable']}")

    print("\nTesting answer_question_local:")
    def dummy_retriever(q, chunks, top_k=3):
        return [{"chunk_id": "1", "text": "Python text", "doc_id": "d1"}]

    print("  This function combines gated retrieval with Ollama answering")
    print("  Requires: Ollama running with the specified model")
