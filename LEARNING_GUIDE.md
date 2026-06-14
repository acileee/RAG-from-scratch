# RAG System - Learning Guide

This is a learning-focused local RAG system built from the ground up.

The project is designed to help you understand how Retrieval-Augmented Generation works internally before relying on high-level frameworks. The core RAG components are implemented manually first, then extended or compared with tools like Chroma, Ollama, LangChain, NetworkX, Graph-RAG, and Self-RAG.

## What is RAG?

RAG stands for Retrieval-Augmented Generation.

A normal LLM answers from what it learned during training. That can be useful, but it can also lead to outdated answers, unsupported claims, or hallucinations.

A RAG system adds a retrieval step before generation.

```text
User question
    ↓
Retrieve relevant chunks from documents
    ↓
Format those chunks as context
    ↓
Build a prompt using the question and context
    ↓
Send the prompt to an LLM
    ↓
Generate an answer grounded in the retrieved text
```

The key idea is simple:

```text
RAG = retrieve first, generate second
```

Instead of asking the model to answer from memory, the system gives the model relevant external context first.

## What This Project Teaches

This project teaches RAG by building the pieces step by step:

* how documents are split into chunks
* how chunks are embedded
* how semantic search works
* how a retriever returns relevant context
* how prompts are built
* how answerability can be checked before generation
* how local LLM generation works with Ollama
* how vector databases like Chroma fit into the pipeline
* how query rewriting can improve retrieval
* how multi-query retrieval improves recall
* how reranking improves result quality
* how evaluation can measure retrieval and answer quality
* how LangChain abstracts the manual pipeline
* how knowledge graphs can expand retrieval
* how Self-RAG retries weak retrieval attempts

## Project Structure

### Core RAG Pipeline

These files implement the main RAG flow.

```text
src/chunk_documents.py
```

Splits raw documents into smaller overlapping chunks.

This is the first major step in RAG because retrieval works better on focused pieces of text than on entire documents.

```text
src/local_embeddings.py
```

Creates embeddings for text and chunks.

Embeddings turn text into vectors so the system can compare meaning instead of only matching exact words.

```text
src/semantic_search.py
```

Implements manual semantic search.

This file shows how query embeddings can be compared with chunk embeddings to find the most relevant chunks.

```text
src/rag_pipeline.py
```

Formats retrieved chunks into context and builds the final RAG prompt.

This is where retrieval output becomes something the LLM can use.

```text
src/local_llm.py
```

Calls Ollama locally and generates answers from a RAG prompt.

This allows the system to answer questions without using an external API.

```text
src/end_to_end_rag.py
```

Connects the whole pipeline from documents to final answer.

This is the main file to look at once you understand the individual pieces.

## Storage and Vector Database

```text
src/storage.py
```

Provides JSON save/load helpers.

This is useful for saving intermediate data such as chunks or embeddings.

```text
src/chroma_store.py
```

Adds Chroma as a persistent vector database.

Manual semantic search is implemented first, then Chroma is introduced to show how a vector database stores and retrieves embeddings more practically.

## Retrieval Improvements

```text
src/query_rewriting.py
```

Rewrites and normalizes user queries.

For example:

```text
RAG → retrieval augmented generation
LLM → large language model
AI → artificial intelligence
KG → knowledge graph
```

This helps retrieval when the user uses abbreviations but the documents use expanded terms.

```text
src/multi_query.py
```

Runs retrieval using multiple versions of the same query.

Instead of searching once, the system can search with:

```text
original question
rewritten query
simplified keyword query
```

The results are merged and deduplicated.

```text
src/reranking.py
```

Reranks retrieved chunks after the first retrieval step.

Retrieval finds candidate chunks. Reranking reorders those chunks using extra signals, such as token overlap with the query.

```text
src/answerability.py
```

Checks whether retrieved chunks are strong enough to answer the question.

If the retrieved context is too weak, the system returns:

```text
I don't know based on the provided context.
```

This helps avoid forcing the LLM to answer when the evidence is not there.

## Evaluation

```text
src/evaluate_retrieval.py
```

Evaluates retrieval quality.

This checks whether the retriever returned the expected document or chunk.

```text
src/rag_evaluation.py
```

Evaluates the full RAG system.

This checks things like:

* retrieval success
* answerability prediction
* whether an answer was produced
* whether unanswerable questions correctly returned the fallback answer

Evaluation matters because RAG systems can look like they work from one example, but fail on other questions.

## LangChain Comparison

```text
src/langchain_basic_rag.py
```

Rebuilds the basic RAG pipeline using LangChain.

The point is not to skip the manual work. The point is to compare the manual implementation with a framework-based version after understanding the internals.

This helps show how LangChain maps to the pieces already built manually:

```text
manual documents → LangChain Document
manual chunking → RecursiveCharacterTextSplitter
manual vector store → Chroma
manual LLM call → OllamaLLM
manual retriever → LangChain retriever
```

## Knowledge Graph and Graph-RAG

```text
src/kg_extraction.py
```

Extracts entities from chunks.

Entities are important concepts or terms found in the text.

```text
src/kg_graph.py
```

Builds a knowledge graph using NetworkX.

The graph connects chunks and entities:

```text
chunk → mentions → entity
entity → co-occurs with → entity
```

This creates structure on top of the retrieved text.

```text
src/graph_rag.py
```

Expands retrieval using the knowledge graph.

The system first retrieves chunks normally, then uses the graph to find related chunks through shared entities.

The flow is:

```text
retrieved chunk
    ↓
connected entity
    ↓
other chunks connected to that entity
```

This can add useful related context that vector search alone may miss.

## Self-RAG

```text
src/self_rag.py
```

Implements a simple Self-RAG retry loop.

The system tries retrieval, checks answerability, and retries with a rewritten query if the first attempt is weak.

```text
try original question
    ↓
check answerability
    ↓
if weak, rewrite query
    ↓
try again
    ↓
answer or return fallback
```

This adds a small agentic control loop around retrieval.

## The Main Pipeline

The basic pipeline is:

```text
documents
    ↓
chunk documents
    ↓
embed chunks
    ↓
retrieve relevant chunks
    ↓
format context
    ↓
build prompt
    ↓
call local LLM
    ↓
return answer
```

The fuller pipeline adds retrieval improvements and safety checks:

```text
documents
    ↓
chunk documents
    ↓
embed chunks
    ↓
store/search with Chroma
    ↓
rewrite query if needed
    ↓
retrieve chunks
    ↓
rerank chunks
    ↓
check answerability
    ↓
build prompt
    ↓
call Ollama
    ↓
return grounded answer or fallback
```

## Suggested Learning Path

Use this order if you want to understand the project from the ground up.

### 1. Start with chunking

Read:

```text
src/chunk_documents.py
```

Learn how documents become chunks.

Focus on:

* why chunks need IDs
* why overlap matters
* why metadata is preserved

### 2. Learn embeddings

Read:

```text
src/local_embeddings.py
```

Learn how text becomes vectors.

Focus on:

* embedding a single query
* embedding multiple chunks
* storing embeddings with chunk metadata

### 3. Understand manual semantic search

Read:

```text
src/semantic_search.py
```

Focus on:

* query embedding
* chunk embeddings
* similarity scores
* top-k retrieval

This is one of the most important files because it explains the core retrieval idea before Chroma is introduced.

### 4. Learn prompt construction

Read:

```text
src/rag_pipeline.py
```

Focus on:

* formatting retrieved chunks
* building a context block
* writing the final RAG prompt
* telling the LLM to answer only from context

### 5. Add local generation

Read:

```text
src/local_llm.py
```

Focus on:

* calling Ollama
* sending the prompt
* handling fallback answers
* connecting generation to retrieval

### 6. Run the full manual pipeline

Read:

```text
src/end_to_end_rag.py
```

This shows how the separate pieces connect into one working system.

### 7. Improve retrieval

Read these next:

```text
src/query_rewriting.py
src/multi_query.py
src/reranking.py
```

These files show how retrieval quality can be improved after the basic pipeline works.

### 8. Add answerability

Read:

```text
src/answerability.py
```

This teaches an important RAG safety idea: sometimes the right answer is not to answer.

### 9. Evaluate the system

Read:

```text
src/evaluate_retrieval.py
src/rag_evaluation.py
```

This shows how to test whether retrieval and answering are actually working.

### 10. Compare with LangChain

Read:

```text
src/langchain_basic_rag.py
```

Do this after understanding the manual version.

The goal is to see what LangChain abstracts, not to replace the learning process.

### 11. Explore graph-based retrieval

Read:

```text
src/kg_extraction.py
src/kg_graph.py
src/graph_rag.py
```

This introduces Graph-RAG by connecting chunks through shared entities.

### 12. Explore Self-RAG

Read:

```text
src/self_rag.py
```

This shows how a RAG system can retry retrieval when the first attempt is weak.

## Key Concepts Explained

### Chunking

Chunking splits documents into smaller pieces.

Why it matters:

* LLMs have context limits
* retrieval works better on focused text
* smaller chunks are easier to rank
* overlap helps preserve context across boundaries

Tradeoff:

```text
larger chunks = more context, less precision
smaller chunks = more precision, less surrounding context
```

### Embeddings

Embeddings are numerical representations of text meaning.

They allow the system to compare text by meaning instead of only exact words.

Example:

```text
"What is RAG?"
```

should be close to:

```text
"Retrieval-Augmented Generation combines retrieval with generation."
```

even if the wording is not exactly the same.

### Semantic Search

Semantic search compares the query embedding with chunk embeddings.

The system returns the chunks with the highest similarity scores.

This is the core retrieval mechanism.

### Chroma

Chroma is a vector database.

In this project, manual search is implemented first. Chroma is then added to make storage and search more practical.

Chroma handles:

* storing vectors
* storing metadata
* querying similar vectors
* persistent vector collections

### Query Rewriting

Query rewriting changes the user query into a form that may retrieve better results.

Example:

```text
"How do LLMs use RAG?"
```

can become:

```text
"how do large language models use retrieval augmented generation"
```

### Multi-Query Retrieval

Multi-query retrieval searches using more than one version of a query.

This helps because one query phrasing may miss useful chunks, while another phrasing may find them.

### Reranking

Reranking happens after retrieval.

Retrieval gets candidates. Reranking decides which candidates should appear first.

In this project, reranking uses a simple token-overlap signal plus the original retrieval score.

### Answerability

Answerability checks whether retrieved context is strong enough to answer.

This is important because retrieval can return weak or irrelevant chunks.

If the evidence is weak, the system should not force an answer.

### Graph-RAG

Graph-RAG adds graph structure to retrieval.

Instead of only retrieving by vector similarity, the system can also retrieve related chunks through shared entities.

This is useful when relevant information is connected indirectly.

### Self-RAG

Self-RAG adds a retry loop.

If the first retrieval attempt is weak, the system rewrites the query and tries again.

This makes retrieval more adaptive.

## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and run Ollama

Start Ollama:

```bash
ollama serve
```

Pull the local LLM model:

```bash
ollama pull llama3.2:1b
```

Pull the embedding model:

```bash
ollama pull nomic-embed-text
```

### 3. Run individual files

Many files can be read and tested individually.

Examples:

```bash
python src/chunk_documents.py
python src/semantic_search.py
python src/rag_pipeline.py
python src/end_to_end_rag.py
```

### 4. Run evaluations

```bash
python src/evaluate_retrieval.py
python src/rag_evaluation.py
```

## Experiments to Try

### Try different chunk sizes

Change chunk size and overlap in the pipeline.

Questions to observe:

* Do larger chunks improve answer quality?
* Do smaller chunks improve retrieval precision?
* Does overlap reduce missing context?

### Try different top-k values

Change how many chunks are retrieved.

Questions to observe:

* Is top_k=3 enough?
* Does top_k=10 add useful context or noise?
* Does answerability improve with more chunks?

### Compare manual search and Chroma

Use manual semantic search first, then compare it with Chroma.

Questions to observe:

* Do they return the same top chunks?
* Are the scores different?
* Does one retrieve better examples?

### Try query rewriting

Ask the same question with abbreviations and expanded terms.

Example:

```text
What is RAG?
```

versus:

```text
What is retrieval augmented generation?
```

### Try Graph-RAG expansion

Compare normal retrieval with graph-expanded retrieval.

Questions to observe:

* Does the graph add useful context?
* Does it add noise?
* Which entities connect the retrieved chunks?

### Try Self-RAG

Ask a question where the first retrieval attempt might be weak.

Observe:

* the first query
* the rewritten query
* whether answerability improves
* whether the system answers or falls back

## Debugging Tips

### Check if Ollama is running

```bash
curl http://localhost:11434/api/tags
```

If Ollama is running, this should return information about installed models.

### Check installed models

```bash
ollama list
```

You should see models such as:

```text
llama3.2:1b
nomic-embed-text
```

### If generation fails

Check:

* Ollama is running
* the model is installed
* the model name matches the code
* the prompt is not empty

### If retrieval is poor

Check:

* documents were chunked correctly
* chunks have embeddings
* query is not empty
* top_k is high enough
* chunk size is not too small or too large
* query rewriting is being applied if needed

### If Chroma gives strange results

Check:

* the collection contains data
* embeddings were upserted
* metadata includes chunk IDs and document IDs
* old test collections are not polluting results

## How to Read the Comments

The code includes explanatory comments to make the project easier to learn from.

Pay attention to comments that explain:

* why a step exists
* what problem it solves
* what the input and output look like
* how the component fits into the full RAG pipeline

Good RAG learning is not just knowing what each line does. It is understanding why each step exists.

## Contributing

The repo includes a contributing guide.

If you want to contribute, check:

```text
CONTRIBUTING.md
```

Good contribution ideas include:

* fixing unclear explanations
* improving comments
* adding examples
* adding tests
* improving evaluation
* adding citations
* adding hybrid search
* improving graph retrieval
* making setup easier

## To summarize

The most important thing to understand is:

```text
RAG is not just "LLM + documents."
```

A RAG system is a pipeline:

```text
prepare documents
retrieve useful evidence
verify whether the evidence is enough
build a grounded prompt
generate an answer from context
evaluate whether it worked
```

This project builds that pipeline manually first, then adds tools and advanced retrieval patterns after the core mechanics are clear.

## Happy Learning

This guide is meant to help you learn RAG from the ground up.

Start with the simple files, follow the pipeline step by step, then explore the advanced retrieval pieces once the basics make sense.
