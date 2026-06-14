"""
WHAT: Document loading module — reads all .txt files from a given folder and returns
      them as a list of structured document dicts ready for chunking and indexing.
WHY: A RAG pipeline needs a consistent intake point for raw source material. Standardising
     on a simple folder-of-text-files format makes it easy to add new documents (just drop
     a .txt file in) without changing any pipeline code downstream.
HOW: load_documents validates the folder path, iterates over .txt files, reads each file
     as UTF-8 text, skips empty files silently, and returns a list of dicts containing
     doc_id (the filename stem), filename, and text.
EXAMPLE: A folder "documents/" contains "attention.txt", "bert.txt", and "gpt.txt".
         load_documents("documents/") returns:
         [{"doc_id": "attention", "filename": "attention.txt", "text": "..."},
          {"doc_id": "bert",      "filename": "bert.txt",      "text": "..."},
          {"doc_id": "gpt",       "filename": "gpt.txt",       "text": "..."}]
         These dicts are passed directly to chunk_documents() in the next pipeline stage.
"""

import pathlib as pl


def load_documents(folder: str) -> list[dict]:
    """
    WHAT: Scans a directory for .txt files and loads each non-empty file into a structured
          dict, returning the full collection as a list.
    WHY: Downstream pipeline stages (chunking, embedding, indexing) expect documents in a
         consistent dict format with doc_id, filename, and text fields. This function
         enforces that contract at the data-intake boundary so the rest of the pipeline
         never has to deal with raw file paths or I/O.
    HOW:
        1. Convert the folder string to a pathlib.Path object.
        2. Raise FileNotFoundError if the path does not exist.
        3. Raise NotADirectoryError if the path exists but is not a directory.
        4. Iterate over all items in the directory; process only files with a .txt suffix.
        5. Open each file with UTF-8 encoding and read its full content.
        6. Skip the file silently if the content is None or contains only whitespace.
        7. Append a dict with keys "doc_id" (file stem, e.g. "attention"),
           "filename" (full name, e.g. "attention.txt"), and "text" (file content).
        8. Return the list of all loaded document dicts.
    EXAMPLE: Given a "papers/" folder with "rag_survey.txt" (2 000 words) and
             "empty_draft.txt" (blank), load_documents("papers/") returns a list with
             one entry — the empty draft is skipped automatically:
             [{"doc_id": "rag_survey", "filename": "rag_survey.txt", "text": "RAG, or
               Retrieval-Augmented Generation, is..."}]
    """
    folder_path = pl.Path(folder)
    documents = []
    if folder_path.exists() is False:
        raise FileNotFoundError
    if folder_path.is_dir() is False:
        raise NotADirectoryError
    for file in folder_path.iterdir():
        if file.is_file() and file.suffix == '.txt':
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                if content is None or content.strip() == "":
                    continue
                documents.append({
                    "doc_id": file.stem,
                    "filename": file.name,
                    "text": content,
                })

    return documents


if __name__ == "__main__":
    folder = "documents"
    docs = load_documents(folder)
    for doc in docs:
        print(doc)
