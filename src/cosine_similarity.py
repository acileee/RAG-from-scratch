"""
WHAT: Module providing two implementations of cosine similarity for comparing
      embedding vectors: a pure-Python version and a NumPy-accelerated version.
WHY: Cosine similarity is the standard distance metric used in RAG vector search.
     Having a pure-Python implementation alongside the NumPy one makes the maths
     transparent for learners while the NumPy version handles real workloads.
HOW: cosine_similarity() computes the dot product and magnitudes manually using
     Python lists. cosine_similarity_numpy() delegates to np.dot and np.linalg.norm
     for speed. Both return a float in [0, 1] (or -1 for opposing vectors).
EXAMPLE: After embedding a user query "What is RAG?" and a document chunk
         "Retrieval-Augmented Generation combines search with LLMs", the RAG
         pipeline calls cosine_similarity(query_vec, chunk_vec) to score how
         closely the chunk matches the question. A score near 1.0 means strong
         semantic alignment.
"""

import numpy as np


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    WHAT: Compute the cosine similarity between two equal-length floating-point
          vectors using only Python built-ins — no external libraries required.

    WHY: In RAG, every text fragment is represented as a dense vector (embedding)
         produced by a language model. To rank chunks by relevance to a query we
         need a scalar measure of how "close" two vectors are in meaning. Cosine
         similarity is preferred over Euclidean distance because it is
         scale-invariant: a long document and a short one on the same topic yield
         the same score even though their raw vector magnitudes differ.

    HOW:
        1. Validate: both vectors must have equal length; return 0.0 for empty inputs.
        2. Iterate over paired elements, accumulating:
              - element-wise products  → list `a` (for dot product)
              - squared elements of v1 → even indices of list `b` (for |v1|)
              - squared elements of v2 → odd indices of list `b` (for |v2|)
        3. dot_product_sum = sum(a)
        4. magnitude_vec1  = sqrt(sum(b[::2]))   (every other element starting at 0)
        5. magnitude_vec2  = sqrt(sum(b[1::2]))  (every other element starting at 1)
        6. Guard against zero-magnitude vectors to prevent division by zero.
        7. Return dot_product_sum / (magnitude_vec1 * magnitude_vec2).

    EXAMPLE: A query embedding for "vector database" might be [0.6, 0.8] and a
             chunk embedding for "storing embeddings in FAISS" might be [0.55, 0.83].
             cosine_similarity([0.6, 0.8], [0.55, 0.83]) ≈ 0.9997, indicating the
             two texts are semantically very close and the chunk is a strong retrieval
             candidate.
    """
    a = []
    b = []

    # Validate vectors are same length
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length")
    if len(vec1) == 0 or len(vec2) == 0:
        return 0.0

    # Calculate components for dot product and magnitudes
    for v1, v2 in zip(vec1, vec2):
        dot_product = v1 * v2
        a.append(dot_product)              # Accumulate for dot product
        b.append(v1 * v1)                  # v1 component for magnitude_vec1
        b.append(v2 * v2)                  # v2 component for magnitude_vec2

    # Compute dot product (A · B)
    dot_product_sum = sum(a)

    # Compute magnitude of vec1: sqrt(sum of squares) - every other element in b
    magnitude_vec1 = sum(b[::2]) ** 0.5

    # Compute magnitude of vec2: sqrt(sum of squares) - every other element in b
    magnitude_vec2 = sum(b[1::2]) ** 0.5

    # Handle zero magnitude vectors (prevent division by zero)
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0

    # Final formula: (A · B) / (|A| × |B|)
    return dot_product_sum / (magnitude_vec1 * magnitude_vec2)


def cosine_similarity_numpy(vec1: list[float], vec2: list[float]) -> float:
    """
    WHAT: Compute cosine similarity between two vectors using NumPy, producing
          the same result as cosine_similarity() but faster for high-dimensional
          embedding vectors.

    WHY: Embedding models like all-MiniLM-L6-v2 produce 384-dimensional vectors.
         Scoring hundreds of chunks against a query embedding involves thousands of
         floating-point operations. NumPy's vectorised C routines are orders of
         magnitude faster than Python loops, making this version necessary for any
         real RAG workload beyond toy examples.

    HOW:
        1. Validate length equality and non-empty inputs; raise ValueError on violations.
        2. Convert both lists to np.array for vectorised operations.
        3. dot_product     = np.dot(v1, v2)
        4. magnitude_vec1  = np.linalg.norm(v1)
        5. magnitude_vec2  = np.linalg.norm(v2)
        6. Guard against zero-magnitude vectors.
        7. Return float(dot_product) / (magnitude_vec1 * magnitude_vec2).

    EXAMPLE: When fake_vector_search.py scores 50 embedded document chunks against
             a query embedding for "how does attention work in transformers?",
             cosine_similarity_numpy is called once per chunk. Each call takes
             microseconds rather than milliseconds, keeping the retrieval step
             imperceptible to the end user.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length")
    if len(vec1) == 0 or len(vec2) == 0:
        raise ValueError("Vectors must not be empty")
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    dot_product = np.dot(v1, v2)
    magnitude_vec1 = np.linalg.norm(v1)
    magnitude_vec2 = np.linalg.norm(v2)
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0
    return float(dot_product) / (magnitude_vec1 * magnitude_vec2)

if __name__ == "__main__":
    tests = [
        ([1, 0], [1, 0], 1.0),  # identical vectors
        ([1, 0], [0, 1], 0.0),  # orthogonal vectors
        ([1, 1], [1, 1], 1.0),  # identical vectors
        ([1, 2, 3], [4, 5, 6], 0.9746),  # general case
        ([2, 0], [1, 0], 1.0),  # aligned but different magnitude
    ]

    print("Testing cosine_similarity:")
    for v1, v2, expected in tests:
        result = cosine_similarity(v1, v2)
        print(f"  cosine_similarity({v1}, {v2}) = {result:.4f} (expected ~{expected})")

    print("\nTesting cosine_similarity_numpy:")
    for v1, v2, expected in tests:
        result = cosine_similarity_numpy(v1, v2)
        print(f"  cosine_similarity_numpy({v1}, {v2}) = {result:.4f} (expected ~{expected})")

    try:
        cosine_similarity([1, 2], [1, 2, 3])
        print("  ERROR: Should have raised ValueError for mismatched lengths")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
