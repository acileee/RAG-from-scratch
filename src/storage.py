"""
WHAT: Persistent storage module — provides simple save and load functions for any
      JSON-serialisable data (chunk lists, embeddings metadata, pipeline outputs, etc.).
WHY: RAG pipelines process documents in a preparation phase (load → chunk → embed) that
     is expensive to repeat on every query. Persisting the results to JSON files lets
     subsequent runs skip the preparation phase and go straight to retrieval and answering.
HOW: save_json validates the .json extension, creates any missing parent directories,
     and writes data with pretty-printing and UTF-8 support. load_json validates the
     extension and file existence before reading and returning the parsed data.
EXAMPLE: After chunking 50 research papers into 2 000 chunks with chunk_documents(),
         call save_json(chunks, "data/chunks.json") to persist them. On the next run,
         load_json("data/chunks.json") restores the full chunk list in milliseconds,
         skipping all document loading and chunking work.
"""

import json
import pathlib as pl


def save_json(data, path: str) -> None:
    """
    WHAT: Serialises any JSON-compatible Python object to a .json file at the given path,
          creating parent directories automatically if they do not exist.
    WHY: Saving intermediate pipeline results (chunks, scored candidates, embeddings
         metadata) to disk avoids re-running expensive steps on every pipeline execution.
         Auto-creating directories removes the need for callers to manage folder setup.
    HOW:
        1. Convert path to a pathlib.Path object and verify the suffix is ".json";
           raise ValueError if not.
        2. If the parent directory does not exist, create it (and any ancestors) with
           parents=True, exist_ok=True.
        3. Open the file for writing with UTF-8 encoding.
        4. Call json.dump with ensure_ascii=False (preserves Unicode) and indent=4
           (human-readable formatting).
    EXAMPLE: After building 1 500 chunks from a corpus of NLP papers, persist them with:
             save_json(chunks, "data/processed/nlp_chunks.json")
             The function creates the "data/processed/" directory if it does not exist
             and writes a readable, indented JSON file ready to be loaded by load_json.
    """
    path_obj = pl.Path(path)
    if path_obj.suffix != '.json':
        raise ValueError("File extension must be .json")
    if not path_obj.parent.exists():
        path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_json(path: str):
    """
    WHAT: Reads a .json file from disk and returns the deserialised Python object
          (list, dict, or any other JSON-compatible type).
    WHY: Re-loading pre-processed data (chunks, cached retrieval results) from a JSON
         file is orders of magnitude faster than reprocessing raw documents. This
         function is the counterpart to save_json and completes the persistence round-trip.
    HOW:
        1. Convert path to a pathlib.Path object and verify the suffix is ".json";
           raise ValueError if not.
        2. Raise FileNotFoundError if the file does not exist at the given path.
        3. Open the file for reading with UTF-8 encoding.
        4. Call json.load to parse and return the deserialised data.
    EXAMPLE: At the start of a RAG query session, restore the pre-processed chunk list:
             chunks = load_json("data/processed/nlp_chunks.json")
             The returned list of dicts is ready to pass directly to a retriever function
             such as bm25_retriever(question, chunks, top_k=5).
    """
    path_obj = pl.Path(path)
    if path_obj.suffix != '.json':
        raise ValueError("File extension must be .json")
    if not path_obj.exists():
        raise FileNotFoundError(f"File {path} does not exist")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    import tempfile
    import os

    print("Testing save_json and load_json:")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_data = {
            "chunks": [
                {"id": 1, "text": "Sample chunk 1"},
                {"id": 2, "text": "Sample chunk 2"}
            ],
            "metadata": {"source": "test", "count": 2}
        }

        test_path = os.path.join(tmpdir, "test.json")

        print(f"  Saving to {test_path}")
        save_json(test_data, test_path)
        print(f"  ✓ File saved successfully")

        print(f"  Loading from {test_path}")
        loaded_data = load_json(test_path)
        print(f"  ✓ File loaded successfully")
        print(f"    - Chunks: {len(loaded_data['chunks'])}")
        print(f"    - Metadata: {loaded_data['metadata']}")

        assert loaded_data == test_data, "Loaded data doesn't match saved data"
        print(f"  ✓ Loaded data matches original")

    print("\nTesting error handling:")
    try:
        save_json({}, "test.txt")
        print("  ERROR: Should raise ValueError for non-.json file")
    except ValueError as e:
        print(f"  ✓ save_json correctly raised ValueError: {e}")

    try:
        load_json("nonexistent.json")
        print("  ERROR: Should raise FileNotFoundError")
    except FileNotFoundError as e:
        print(f"  ✓ load_json correctly raised FileNotFoundError: {e}")

    try:
        load_json("test.txt")
        print("  ERROR: Should raise ValueError for non-.json file")
    except ValueError as e:
        print(f"  ✓ load_json correctly raised ValueError: {e}")
