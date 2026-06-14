"""
WHAT: Runs systematic experiments to measure how the number of retrieved chunks
      (top_k) affects retrieval accuracy.
WHY: Retrieving too few chunks risks missing the relevant passage; retrieving too
     many adds noise to the LLM's context window. Finding the right top_k through
     empirical evaluation is more reliable than guessing.
HOW: Embeds a fixed chunk set once, then evaluates semantic_search at each
     top_k value in a provided list, collecting accuracy metrics for comparison.
EXAMPLE: results = run_top_k_experiment(chunks, eval_questions,
             top_k_values=[1, 3, 5, 10])
         # Shows accuracy rising from top_k=1 to a plateau, revealing diminishing returns.
"""

from evaluate_retrieval import evaluate_retriever
from local_embeddings import embed_chunks
from semantic_search import semantic_search


def run_top_k_experiment(chunks: list[dict], eval_questions: list[dict], top_k_values: list[int]) -> list[dict]:
    """
    WHAT: Evaluates retrieval accuracy for a range of top_k values on a fixed chunk set.
    WHY: top_k determines how much context the LLM sees. A systematic sweep lets
         us pick the value that maximises retrieval hit rate without unnecessarily
         large context windows that slow inference and dilute relevance.
    HOW: 1. Calls embed_chunks() once to attach vectors to all chunks — this is
            done outside the loop to avoid redundant embedding calls.
         2. For each top_k in top_k_values, calls evaluate_retriever() with
            semantic_search as the retriever function.
         3. Collects a result dict (top_k, num_chunks, accuracy, correct, failures,
            false_retrievals) for each configuration.
         4. Returns the list of result dicts for downstream comparison.
    EXAMPLE: results = run_top_k_experiment(chunks, eval_questions,
                 top_k_values=[1, 3, 5, 10])
             for r in results:
                 print(f"top_k={r['top_k']} → accuracy={r['accuracy']:.2%}")
             # top_k=3 might achieve 85% while top_k=1 is only 57%.
    """
    experiment_results = []
    embedded_chunks = embed_chunks(chunks)
    for top_k in top_k_values:
        eval_results = evaluate_retriever(semantic_search, eval_questions, embedded_chunks, top_k=top_k)
        experiment_results.append({
             "top_k": top_k,
             "num_chunks": len(embedded_chunks),
             "accuracy": eval_results["accuracy"],
             "correct": eval_results["correct"],
             "answerable_count": eval_results["answerable_count"],
             "false_retrievals": eval_results["false_retrievals"],
             "failures": eval_results["failures"]
        })
    return experiment_results


if __name__ == "__main__":
    print("Testing retrieval_experiments:")
    print("  run_top_k_experiment: Evaluates semantic search with different top_k values")
    print("\nThis function performs systematic experiments to evaluate retrieval performance")
    print("across different numbers of top-k results to find optimal settings.")
