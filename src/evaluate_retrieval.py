"""
WHAT: Provides a retrieval evaluation harness with a small built-in test corpus.
WHY: Measuring retrieval quality (did the right chunk come back?) is the first
     sanity check before evaluating answer quality — bad retrieval makes good
     generation impossible.
HOW: Defines a fixed set of chunks and labelled questions (with expected doc_ids
     and answerability flags), then provides evaluate_retriever() to score any
     retriever function and run_semantic_eval() as a convenience wrapper for the
     default semantic search backend.
EXAMPLE: results = run_semantic_eval(my_chunks, eval_questions, top_k=3)
         print(results["accuracy"])  # e.g. 0.857 — 6/7 answerable questions hit
"""

from torch import chunk


chunks = [
    {
        "chunk_id": "rag_000",
        "doc_id": "rag",
        "filename": "rag.txt",
        "text": "RAG retrieves relevant chunks before generation. It helps language models answer using external context.",
        "start_char": 0,
        "end_char": 100,
    },
    {
        "chunk_id": "kg_000",
        "doc_id": "kg",
        "filename": "kg.txt",
        "text": "Knowledge graphs store entities and relationships. They help systems represent structured knowledge.",
        "start_char": 0,
        "end_char": 100,
    },
    {
        "chunk_id": "python_000",
        "doc_id": "python",
        "filename": "python.txt",
        "text": "Python is used for AI experiments, automation, data processing, and building research prototypes.",
        "start_char": 0,
        "end_char": 100,
    },
    {
        "chunk_id": "llm_000",
        "doc_id": "llm",
        "filename": "llm.txt",
        "text": "Large language models generate text based on prompts and context. They can summarize, answer questions, and reason over text.",
        "start_char": 0,
        "end_char": 120,
    },
    {
        "chunk_id": "agents_000",
        "doc_id": "agents",
        "filename": "agents.txt",
        "text": "Agentic systems can plan steps, use tools, check results, and retry when the first attempt fails.",
        "start_char": 0,
        "end_char": 100,
    },
]

eval_questions = [
    {
        "question": "What does RAG retrieve before generation?",
        "expected_doc_id": "rag",
        "answerable": True,
    },
    {
        "question": "What helps language models answer using external context?",
        "expected_doc_id": "rag",
        "answerable": True,
    },
    {
        "question": "What do knowledge graphs store?",
        "expected_doc_id": "kg",
        "answerable": True,
    },
    {
        "question": "What helps systems represent structured knowledge?",
        "expected_doc_id": "kg",
        "answerable": True,
    },
    {
        "question": "What is Python used for in AI projects?",
        "expected_doc_id": "python",
        "answerable": True,
    },
    {
        "question": "What can large language models generate?",
        "expected_doc_id": "llm",
        "answerable": True,
    },
    {
        "question": "What can agentic systems do when the first attempt fails?",
        "expected_doc_id": "agents",
        "answerable": True,
    },
    {
        "question": "Who invented Python?",
        "expected_doc_id": None,
        "answerable": False,
    },
    {
        "question": "What is the capital of Japan?",
        "expected_doc_id": None,
        "answerable": False,
    },
    {
        "question": "When was AIT founded?",
        "expected_doc_id": None,
        "answerable": False,
    },
]


def evaluate_retriever(retriever_fn, questions: list[dict], chunks: list[dict], top_k: int = 3) -> dict:
    """
    WHAT: Scores a retriever function against a labelled question set.
    WHY: Quantifying retrieval accuracy separately from generation quality lets
         us diagnose whether failures are retrieval problems or LLM problems —
         the two most common failure modes in RAG systems.
    HOW: For each question, calls retriever_fn(question, chunks, top_k) and
         extracts the returned doc_ids. For answerable questions it checks whether
         the expected_doc_id is in the result set and counts hits. For unanswerable
         questions it records any returned chunks as false retrievals. Finally
         computes accuracy = correct / answerable_count.
    EXAMPLE: accuracy = evaluate_retriever(semantic_search, eval_questions,
                                           embedded_chunks, top_k=3)["accuracy"]
             # If semantic_search returns the RAG chunk for "What does RAG
             # retrieve?", that question counts as correct.
    """
    results = {}
    correct = 0
    answerable_count = 0
    unanswerable_count = 0
    failures = []
    false_retrievals = []
    for q in questions:
        question = q["question"]
        expected_doc_id = q["expected_doc_id"]
        answerable = q["answerable"]
        retrieved_chunks = retriever_fn(question, chunks, top_k=top_k)
        retrieved_doc_ids = [chunk["doc_id"] for chunk in retrieved_chunks]
        if answerable:
            success = expected_doc_id in retrieved_doc_ids
            answerable_count += 1
            if success:
                correct += 1
            else:
                failures.append({"question": question, "expected_doc_id": expected_doc_id, "retrieved_doc_ids": retrieved_doc_ids})
        else:
            unanswerable_count += 1
            if retrieved_doc_ids:
                false_retrievals.append({"question": question, "expected_doc_id": expected_doc_id, "retrieved_doc_ids": retrieved_doc_ids})
    total_questions = len(questions)
    accuracy = correct / answerable_count if answerable_count > 0 else 0.0
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total_questions": total_questions,
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
        "failures": failures,
        "false_retrievals": false_retrievals,
        "top_k": top_k,
    }
    return results


def run_semantic_eval(chunks: list[dict], eval_questions: list[dict], top_k: int = 3) -> dict:
    """
    WHAT: Convenience wrapper that embeds chunks then evaluates semantic search
          against the provided question set.
    WHY: Combining embedding and evaluation into one call reduces the boilerplate
         needed in experiment scripts that want a quick retrieval score for a
         given chunk set.
    HOW: Calls embed_chunks() to attach vectors to every chunk, then passes the
         semantic_search function and the embedded chunks to evaluate_retriever()
         and returns its result dict.
    EXAMPLE: results = run_semantic_eval(
                 chunk_documents(docs, chunk_size=200), eval_questions, top_k=3)
             print(results["accuracy"])
             # Shows how well 200-char chunks retrieve the right doc for each question.
    """
    from local_embeddings import embed_chunks
    embedded_chunks = embed_chunks(chunks)
    from semantic_search import semantic_search
    results = evaluate_retriever(semantic_search, eval_questions, embedded_chunks, top_k=top_k)
    return results


if __name__ == "__main__":
    print("Testing evaluate_retrieval functions:")

    print("\nTesting evaluate_retriever:")
    def dummy_retriever(query, chunks, top_k=3):
        return chunks[:top_k]

    test_chunks = [
        {"chunk_id": "1", "text": "RAG retrieves chunks", "doc_id": "rag", "filename": "f1.txt"},
        {"chunk_id": "2", "text": "KG stores knowledge", "doc_id": "kg", "filename": "f2.txt"},
    ]

    test_questions = [
        {"question": "What is RAG?", "expected_doc_id": "rag", "answerable": True},
        {"question": "What does KG do?", "expected_doc_id": "kg", "answerable": True},
        {"question": "Unknown topic?", "expected_doc_id": None, "answerable": False},
    ]

    results = evaluate_retriever(dummy_retriever, test_questions, test_chunks, top_k=2)
    print(f"  Accuracy: {results['accuracy']:.2%}")
    print(f"  Correct: {results['correct']}/{results['answerable_count']}")
    print(f"  Failures: {len(results['failures'])}")
    print(f"  False retrievals: {len(results['false_retrievals'])}")
