"""
WHAT: Module for splitting long documents into smaller, overlapping text chunks.
WHY: LLMs have token limits and cannot process entire documents at once; chunking
     makes retrieval precise by letting the system fetch only the relevant slice of a document.
HOW: Exposes chunk_text(), which slides a fixed-size window across a string,
     stepping forward by (chunk_size - overlap) on each iteration so consecutive
     chunks share some text.
EXAMPLE: A 10 000-character research paper is chunked into ~17 overlapping pieces
         of 600 characters each (overlap=100), each piece stored with its position
         metadata so a RAG pipeline can later cite exact passages.
"""


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[dict]:
    """
    WHAT: Split a single string into a list of overlapping fixed-size text chunks,
          each annotated with its position in the original text.

    WHY: Embedding models and LLM context windows have hard token limits. By breaking
         a document into chunks we can embed each piece independently, store them in a
         vector index, and retrieve only the handful of chunks that are relevant to a
         query — rather than sending the whole document to the LLM every time.

    HOW:
        1. Validate that chunk_size > 0 and 0 <= overlap < chunk_size; raise
           ValueError otherwise so callers catch bad configurations early.
        2. Return an empty list immediately for blank input.
        3. Use a sliding window: start positions are 0, step, 2*step, …
           where step = chunk_size - overlap.  At each start position slice
           text[i : i + chunk_size].
        4. Wrap every slice in a dict carrying chunk_id (sequential index),
           the text itself, and start_char / end_char offsets into the original.

    EXAMPLE: Chunking a Wikipedia article on neural networks (≈4 000 chars) with
             chunk_size=600, overlap=100 produces chunks whose windows overlap by
             100 chars, so a sentence that spans a boundary appears in both adjacent
             chunks — preventing the retriever from missing a key fact that falls
             right at an edge.
    """
    chunks = []
    # Validate parameters - prevent invalid configs
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_size must be greater than 0, overlap must be non-negative and less than chunk_size")

    # Empty text edge case
    if text.strip() == "":
        return chunks

    # Slide window across text, stepping by (chunk_size - overlap) to create overlap
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i : i + chunk_size]
        chunks.append({
            "chunk_id": len(chunks),           # Sequential ID for this chunk
            "text": chunk,                     # The actual text content
            "start_char": i,                   # Position in original text
            "end_char": i + len(chunk)         # End position in original text
        })
    return chunks

if __name__ == "__main__":
    tests = [
        ("", 600, 100, 0),  # empty text
        ("Hello world", 100, 10, 1),  # short text fits in one chunk
        ("A" * 1000, 100, 20, 11),  # long text with overlap
    ]

    print("Testing chunk_text:")
    for text, chunk_size, overlap, expected_chunks in tests:
        result = chunk_text(text, chunk_size, overlap)
        print(f"  chunk_text(text_len={len(text)}, chunk_size={chunk_size}, overlap={overlap}) -> {len(result)} chunks (expected {expected_chunks})")
        if result:
            print(f"    First chunk: start={result[0]['start_char']}, end={result[0]['end_char']}, len={len(result[0]['text'])}")

    try:
        chunk_text("test", chunk_size=-1)
        print("  ERROR: Should raise ValueError for negative chunk_size")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

    try:
        chunk_text("test", chunk_size=100, overlap=100)
        print("  ERROR: Should raise ValueError for overlap >= chunk_size")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
