"""
WHAT: Document chunking module — splits a batch of documents into overlapping text chunks
      and attaches document-level metadata (doc_id, filename) to every chunk.
WHY: RAG retrieval operates on small, focused text segments rather than whole documents.
     Chunking with overlap ensures that information spanning a chunk boundary is not lost.
     Attaching metadata to each chunk makes it possible to trace every retrieved passage
     back to its source document.
HOW: chunk_documents iterates over a list of document dicts, validates required fields,
     delegates the actual splitting to chunk_text (from chunk_text.py), generates
     globally unique chunk IDs by combining doc_id with a zero-padded chunk number,
     and collects all chunks across all documents into a single flat list.
EXAMPLE: Two documents — "attention_paper.txt" (10 000 chars) and "bert_paper.txt"
         (8 000 chars) — are passed in. With chunk_size=600 and overlap=100, the function
         produces roughly 29 chunks total, each tagged with its source doc_id and filename,
         with IDs like "attention_paper_001", "attention_paper_002", "bert_paper_001", etc.
"""


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 600,
    overlap: int = 100,
) -> list[dict]:
    """
    WHAT: Processes a list of document dicts into a flat list of text chunk dicts,
          each carrying unique IDs and source document metadata.
    WHY: A RAG pipeline indexes chunks, not whole documents. Tracking doc_id and filename
         on every chunk enables source attribution in generated answers ("this came from
         bert_paper.txt, chunk 003") and supports filtering by document during retrieval.
    HOW:
        1. Import chunk_text lazily to avoid circular imports.
        2. Validate each document: must be a non-null dict with non-null string fields
           "doc_id", "filename", and a "text" field — raise ValueError for any violation.
        3. For each document, call chunk_text(doc["text"], chunk_size, overlap) to produce
           raw chunk dicts with integer chunk_id values.
        4. Convert each integer chunk_id to a zero-padded string (e.g., 1 -> "001").
        5. Prefix the doc_id to create a globally unique chunk ID (e.g., "bert_paper_001").
        6. Attach "doc_id" and "filename" to each chunk dict.
        7. Append the enriched chunk to the accumulator list.
        8. Return the flat list of all chunks from all documents.
    EXAMPLE: Given documents=[{"doc_id": "gpt4_report", "filename": "gpt4.txt",
             "text": "GPT-4 is a large multimodal model..."}] with chunk_size=600,
             the function returns chunks like:
             [{"chunk_id": "gpt4_report_001", "doc_id": "gpt4_report",
               "filename": "gpt4.txt", "text": "GPT-4 is a large...",
               "start_char": 0, "end_char": 600}, ...]
    """
    from chunk_text import chunk_text
    all_chunks = []

    for doc in documents:
        if doc is None or not isinstance(doc, dict):
            raise ValueError("Each document must be a non-null dictionary")

        doc_id = doc.get("doc_id")
        if doc_id is None or not isinstance(doc_id, str):
            raise ValueError("Each document must have a non-null string 'doc_id' field")

        filename = doc.get("filename")
        if filename is None or not isinstance(filename, str):
            raise ValueError("Each document must have a non-null string 'filename' field")

        if "text" not in doc:
            raise ValueError("Each document must have a 'text' field")

        for chunk in chunk_text(doc["text"], chunk_size, overlap):
            chunk_id = chunk["chunk_id"]

            formatted_chunk_id = str(chunk_id)
            padded_chunk_id = formatted_chunk_id.zfill(3)  # Pad with zeros: 1 -> 001
            chunk["chunk_id"] = doc_id + "_" + padded_chunk_id

            chunk.update({
                "doc_id": doc_id,
                "filename": filename,
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
            })

            all_chunks.append(chunk)

    return all_chunks


if __name__ == "__main__":
    print("Testing chunk_documents:")

    docs = [
        {"doc_id": "doc1", "filename": "file1.txt", "text": "A" * 800},
        {"doc_id": "doc2", "filename": "file2.txt", "text": "B" * 500},
    ]

    chunks = chunk_documents(docs, chunk_size=300, overlap=50)
    print(f"  Created {len(chunks)} chunks from {len(docs)} documents")
    for c in chunks[:5]:
        print(f"    - {c['chunk_id']}: len={len(c['text'])}, doc_id={c['doc_id']}")

    print("\nTesting error handling:")
    try:
        chunk_documents([None])
        print("  ERROR: Should raise ValueError for None document")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

    try:
        chunk_documents([{"doc_id": None, "filename": "f.txt", "text": "test"}])
        print("  ERROR: Should raise ValueError for missing doc_id")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

    try:
        chunk_documents([{"doc_id": "d1", "filename": None, "text": "test"}])
        print("  ERROR: Should raise ValueError for missing filename")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

    try:
        chunk_documents([{"doc_id": "d1", "filename": "f.txt"}])
        print("  ERROR: Should raise ValueError for missing text")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
