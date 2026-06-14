# Contributing to Learn RAG From Scratch

Thank you for your interest in contributing.

This project is a learning-focused, open-source RAG implementation. The goal is to help people understand Retrieval-Augmented Generation from the ground up by building the core pieces manually before comparing them with tools like Chroma, Ollama, LangChain, Graph-RAG, and Self-RAG.

Contributions that make the project easier to understand, easier to run, or more useful for learners are welcome.

## Ways to Contribute

Good contribution ideas include:

* improving explanations or comments
* fixing unclear documentation
* adding examples
* adding or improving tests
* improving setup instructions
* improving evaluation examples
* adding better debugging tips
* improving retrieval experiments
* adding citations to answers
* improving Graph-RAG or Self-RAG examples
* fixing bugs

Please keep in mind that this repo is educational. Changes should make the system clearer, not just more complex.

## Getting Started

1. **Fork the repository** on GitHub.

2. **Clone your fork** locally:

   ```bash
   git clone https://github.com/YOUR_USERNAME/learn-rag-from-scratch.git
   cd learn-rag-from-scratch
   ```

3. **Create a virtual environment**:

   ```bash
   python -m venv venv
   ```

   On Windows:

   ```bash
   venv\Scripts\activate
   ```

   On macOS/Linux:

   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Install the Ollama models used by the project**:

   ```bash
   ollama pull llama3.2:1b
   ollama pull nomic-embed-text
   ```

6. **Start Ollama**:

   ```bash
   ollama serve
   ```

7. **Create a branch** for your changes:

   ```bash
   git checkout -b feature/your-feature-name
   ```

## Testing Your Changes

Before opening a pull request, test the files you changed.

You can run core files directly:

```bash
python src/chunk_documents.py
python src/local_embeddings.py
python src/semantic_search.py
python src/rag_pipeline.py
python src/end_to_end_rag.py
```

If you changed retrieval, evaluation, Graph-RAG, or Self-RAG logic, also test the related files:

```bash
python src/evaluate_retrieval.py
python src/rag_evaluation.py
python src/graph_rag.py
python src/self_rag.py
```

If the repo has automated tests available, run them as well:

```bash
pytest
```

## Code Style

Please follow the existing style of the project.

Use:

* clear variable names
* type hints where useful
* small functions with clear responsibilities
* readable control flow
* comments for learning-focused explanations

The code comments often follow this structure:

* **WHAT:** what the function or block does
* **WHY:** why this step exists in the RAG pipeline
* **HOW:** how the logic works
* **EXAMPLE:** how the function or concept is used

Comments should help learners understand the reasoning behind the code. Avoid comments that only repeat obvious syntax.

Good comment:

```python
# WHY: Overlap helps preserve context when useful information is split across chunk boundaries.
```

Less useful comment:

```python
# Create an empty list.
```

## Documentation Style

If you add or change functionality, update documentation when needed.

Useful files to update:

* `README.md`
* `LEARNING_GUIDE.md`
* comments/docstrings in the relevant source file

Documentation should stay beginner-friendly and explain both the **why** and the **how**.

## Pull Request Process

1. Make your changes in a separate branch.
2. Test the affected files.
3. Update documentation if needed.
4. Keep the pull request focused on one clear change when possible.
5. Open a pull request with a clear title and description.

In your pull request description, include:

* what you changed
* why you changed it
* how you tested it
* any limitations or follow-up ideas

Example:

```text
What changed:
Added a small example for query rewriting.

Why:
The learning guide mentioned query rewriting but did not show a concrete example.

How I tested:
Ran query_rewriting.py and checked that RAG expands to retrieval augmented generation.
```

## Issues and Suggestions

If you are not ready to make a pull request, you can still open an issue.

Good issue topics include:

* unclear explanations
* setup problems
* bugs
* confusing comments
* missing examples
* ideas for improving the learning guide
* suggestions for better tests

## Project Goal

The goal of this repo is not to be the most complex RAG system.

The goal is to make RAG understandable by showing the pipeline step by step:

```text
documents
→ chunks
→ embeddings
→ retrieval
→ reranking
→ answerability
→ prompt
→ local LLM answer
```

Please keep contributions aligned with that goal.

Thank you for helping make this project more useful for people learning RAG.
