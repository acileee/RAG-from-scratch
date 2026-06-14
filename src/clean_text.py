"""
WHAT: Module for normalising raw text before it enters the RAG pipeline.
WHY: Documents loaded from PDFs, web pages, or plain-text files often contain
     irregular whitespace, stray tabs, and blank lines that inflate token counts
     and confuse embedding models. A consistent cleaning step makes downstream
     processing more reliable.
HOW: Exposes clean_text(), which iterates line-by-line, collapses runs of spaces
     and tabs to a single space, strips leading/trailing whitespace from each line,
     drops empty lines, then rejoins the surviving lines with newlines.
EXAMPLE: A PDF-extracted paragraph like "RAG   retrieves\\n\\n\\nchunks" becomes
         "RAG retrieves\\nchunks" — ready to be passed to chunk_text() without
         wasting tokens on noise.
"""

import re


def clean_text(text) -> str:
    """
    WHAT: Collapse irregular whitespace in a multi-line string and remove blank lines,
          returning a clean version of the same text.

    WHY: Raw text harvested from PDFs, HTML pages, or copy-pasted sources is littered
         with extra spaces, tab characters, and runs of empty lines. These artefacts
         waste embedding-model tokens and can break simple tokenisers. Normalising
         before chunking ensures every chunk carries as much real information as
         possible within its character budget.

    HOW:
        1. Split the input on newlines to process line-by-line.
        2. For each line, replace every run of spaces or tabs ([ \\t]+) with a
           single space using a regex substitution.
        3. Strip any remaining leading/trailing whitespace from the line.
        4. Discard the line entirely if it is now empty (removes blank lines).
        5. Rejoin surviving lines with '\\n' to preserve paragraph structure.

    EXAMPLE: A chunk extracted from a PDF about RAG might arrive as
             "   Retrieval-Augmented   Generation\\t\\t(RAG)\\n\\n\\nimproves LLMs."
             After clean_text it becomes
             "Retrieval-Augmented Generation (RAG)\\nimproves LLMs."
             — preserving the two-sentence structure while eliminating all noise.
    """
    cleaned_lines = []
    for cleaned_line in text.splitlines():
        # Replace multiple spaces/tabs with single space
        cleaned_line = re.sub(r'[ \t]+', ' ', cleaned_line)
        # Remove leading/trailing whitespace
        cleaned_line = cleaned_line.strip()
        # Skip empty lines
        if cleaned_line == "":
            continue
        cleaned_lines.append(cleaned_line)
    # Rejoin with newlines to preserve document structure
    return '\n'.join(cleaned_lines)

if __name__ == "__main__":
    tests = [
        "   hello   world   ",
        "\n\nRAG is useful.\n\n\nKnowledge graphs store entities.\n\n",
        "Python\t\tis\tuseful.",
        "   RAG     uses retrieval.\n\n\n\tLLMs      generate answers.   ",
        "",
        "     \n\t   \n",
    ]

    for i, test in enumerate(tests, start=1):
        print(f"\n--- Test {i} ---")
        print(repr(clean_text(test)))
