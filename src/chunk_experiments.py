"""
WHAT: Runs systematic experiments to measure how chunk size affects retrieval accuracy.
WHY: Chunk size is one of the most impactful hyperparameters in RAG — too large
     and chunks contain mixed topics that hurt precision; too small and chunks
     lose surrounding context that the LLM needs. Empirical evaluation is the
     only reliable way to find the sweet spot for a given document set.
HOW: Iterates over a list of candidate chunk sizes, rechunks the documents at
     each size, embeds the chunks, and calls the semantic retrieval evaluator to
     measure accuracy. All results are collected into a list for comparison.
EXAMPLE: results = run_chunk_size_experiment(documents, eval_questions,
             chunk_sizes=[100, 200, 500], overlap=50, top_k=3)
         best = max(results, key=lambda r: r["accuracy"])
         print(best["chunk_size"])  # e.g. 200 — the size with highest retrieval hit rate
"""


def run_chunk_size_experiment(documents: list[dict], eval_questions: list[dict], chunk_sizes: list[int], overlap: int = 50, top_k: int = 3) -> list[dict]:
    """
    WHAT: Evaluates semantic retrieval accuracy across multiple chunk sizes.
    WHY: Running this experiment before choosing a chunk size saves time later —
         it shows which size produces the most accurate retrieval on the actual
         documents and questions used in the project, rather than relying on
         general rules of thumb.
    HOW: 1. Iterates over each chunk_size in chunk_sizes.
         2. Calls chunk_documents() with that size and the fixed overlap.
         3. Passes the resulting chunks to run_semantic_eval(), which embeds them
            and measures hit rate against eval_questions.
         4. Appends a result dict (chunk_size, overlap, num_chunks, accuracy,
            correct, failures, etc.) for that configuration.
         5. Returns the full list of result dicts for comparison.
    EXAMPLE: results = run_chunk_size_experiment(
                 documents, eval_questions,
                 chunk_sizes=[100, 200, 500, 1000], overlap=50, top_k=3)
             for r in results:
                 print(f"chunk_size={r['chunk_size']} → accuracy={r['accuracy']:.2%}")
             # Might show chunk_size=200 achieving 85% while 1000 only achieves 57%.
    """
    from chunk_documents import chunk_documents
    from evaluate_retrieval import run_semantic_eval
    experiment_results = []
    for chunk_size in chunk_sizes:
        chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
        eval_results = run_semantic_eval(chunks, eval_questions, top_k=top_k)
        experiment_results.append({
            "chunk_size": chunk_size,
            "overlap": overlap,
            "num_chunks": len(chunks),
            "accuracy": eval_results["accuracy"],
            "correct": eval_results["correct"],
            "answerable_count": eval_results["answerable_count"],
            "false_retrievals": eval_results["false_retrievals"],
            "failures": eval_results["failures"],
            "top_k": eval_results["top_k"],
        })
    return experiment_results


if __name__ == "__main__":
    print("Testing run_chunk_size_experiment (demo):")
    print("  This function requires external dependencies and is typically used in batch experiments.")
    print("  It would process documents with different chunk sizes and evaluate retrieval performance.")
