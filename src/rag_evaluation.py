"""
WHAT: Evaluates a complete RAG system across retrieval, answerability, and answer
      correctness dimensions.
WHY: Retrieval accuracy alone is insufficient — a system that retrieves the right
     chunk but generates the wrong answer (or wrongly claims it cannot answer)
     also fails. This module measures all three failure modes together.
HOW: Defines evaluate_rag_question() for single-question evaluation and
     evaluate_rag_system() to aggregate results over an entire question set,
     computing overall accuracy, retrieval accuracy, and answerability accuracy.
EXAMPLE: metrics = evaluate_rag_system(answer_from_documents, eval_questions,
                                        documents)
         print(metrics["overall_accuracy"])  # e.g. 0.8 — 8/10 questions fully correct
"""


def evaluate_rag_question(rag_answer_fn, question_item: dict, documents: list[dict], **rag_kwargs) -> dict:
    """
    WHAT: Evaluates a single RAG pipeline invocation against ground-truth labels.
    WHY: Per-question evaluation allows pinpointing exactly which questions fail
         and why (wrong retrieval vs. wrong answerability vs. empty answer), which
         is more actionable than aggregate metrics alone.
    HOW: 1. Extracts question, expected_doc_id, and answerable flag from question_item.
         2. Calls rag_answer_fn(question, documents, **rag_kwargs) to get the RAG output.
         3. Checks retrieval_success: expected_doc_id in retrieved doc_ids (or
            empty retrieval for unanswerable questions).
         4. Checks answerability_success: predicted flag matches expected flag.
         5. Checks has_answer: non-empty answer string.
         6. Computes overall_success based on whether the question is answerable.
    EXAMPLE: result = evaluate_rag_question(answer_from_documents,
                 {"question": "What is RAG?", "expected_doc_id": "rag",
                  "answerable": True}, documents)
             # result["overall_success"] is True if RAG retrieved the rag doc,
             # predicted answerable=True, and returned a non-empty answer.
    """
    question = question_item["question"]
    expected_doc_id = question_item["expected_doc_id"]
    expected_answerable = question_item["answerable"]
    result = rag_answer_fn(question=question, documents=documents, **rag_kwargs)
    retrieved_chunks = result.get("retrieved_chunks", [])
    retrieved_doc_ids = [chunk["doc_id"] for chunk in retrieved_chunks]
    answer = result.get("answer", "")
    predicted_answerable = result["answerable"]
    answerability_success = predicted_answerable == expected_answerable
    has_answer = bool(answer.strip())
    if expected_doc_id is not None:
        retrieval_success = expected_doc_id in retrieved_doc_ids
    else:
        retrieval_success = len(retrieved_doc_ids) == 0
    if expected_answerable:
        overall_success = retrieval_success and answerability_success and has_answer
    else:
        overall_success = (answerability_success and answer.strip() == "I don't know based on the provided context.")
    return {
        "question": question,
        "expected_doc_id": expected_doc_id,
        "expected_answerable": expected_answerable,
        "predicted_answerable": predicted_answerable,
        "retrieved_doc_ids": retrieved_doc_ids,
        "retrieval_success": retrieval_success,
        "answerability_success": answerability_success,
        "has_answer": has_answer,
        "overall_success": overall_success,
        "answer": answer}


def evaluate_rag_system(rag_answer_fn, questions: list[dict], documents: list[dict], **rag_kwargs) -> dict:
    """
    WHAT: Runs the full RAG evaluation suite over a list of labelled questions.
    WHY: Aggregate metrics (overall accuracy, retrieval accuracy, answerability
         accuracy) give a high-level picture of system health that can be tracked
         across experiment iterations or compared between RAG configurations.
    HOW: Calls evaluate_rag_question() for every question, accumulates per-question
         result dicts, then counts successes for each metric and divides by total
         questions to produce accuracy scores.
    EXAMPLE: metrics = evaluate_rag_system(answer_from_documents, eval_questions,
                                            documents, top_k=3, model="llama3.2:1b")
             # metrics["retrieval_accuracy"] might be 0.9 while
             # metrics["overall_accuracy"] is 0.7, revealing the LLM as the weak link.
    """
    results = []
    for question in questions:
        result = evaluate_rag_question(rag_answer_fn=rag_answer_fn, question_item=question, documents=documents, **rag_kwargs)
        results.append(result)
    total_questions = len(results)
    overall_correct = sum(1 for r in results if r["overall_success"])
    retrieval_correct = sum(1 for r in results if r["retrieval_success"])
    answerability_correct = sum(1 for r in results if r["answerability_success"])
    overall_accuracy = overall_correct / total_questions if total_questions > 0 else 0.0
    retrieval_accuracy = retrieval_correct / total_questions if total_questions > 0 else 0.0
    answerability_accuracy = answerability_correct / total_questions if total_questions > 0 else 0.0
    return {
        "total_questions": total_questions,
        "overall_correct": overall_correct,
        "overall_accuracy": overall_accuracy,
        "retrieval_accuracy": retrieval_accuracy,
        "answerability_accuracy": answerability_accuracy,
        "results": results}


if __name__ == "__main__":
    print("Testing rag_evaluation functions:")

    question_item = {
        "question": "What is Python?",
        "expected_doc_id": "py_doc",
        "answerable": True
    }

    print("\nTesting evaluate_rag_question (with mock):")
    def mock_rag_answer(question, documents, **kwargs):
        return {
            "question": question,
            "answer": "Python is a programming language.",
            "answerable": True,
            "retrieved_chunks": [{"doc_id": "py_doc", "text": "Python info"}]
        }

    result = evaluate_rag_question(mock_rag_answer, question_item, [])
    print(f"  Question: {result['question']}")
    print(f"  Overall success: {result['overall_success']}")
    print(f"  Retrieval success: {result['retrieval_success']}")
    print(f"  Answerability success: {result['answerability_success']}")
