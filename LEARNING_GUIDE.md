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

If you are learning RAG, debug the system like a pipeline instead of trying to fix everything at once.

Move through the system in order:

```text
documents → chunks → embeddings → retrieval → reranking → answerability → prompt → LLM answer
```

For each problem, ask:

* **IF:** What happened?
* **WHY:** What might be causing it?
* **HOW:** How can I check or fix it?
* **EXAMPLE:** What does this look like in practice?

---

### If Ollama connection fails

**WHY:**
The local Ollama server may not be running.

**HOW:**
Check whether Ollama is running:

```bash
curl http://localhost:11434/api/tags
```

If you get a connection error, start Ollama:

```bash
ollama serve
```

Then run the check again:

```bash
curl http://localhost:11434/api/tags
```

**EXAMPLE:**
If you see something like this:

```text
Failed to connect to localhost port 11434
```

it usually means Ollama is not running yet.

Start it with:

```bash
ollama serve
```

Then open another terminal and try your Python script again.

---

### If the model is not found

**WHY:**
The model name in the code may not match a model installed on your machine.

**HOW:**
Check installed models:

```bash
ollama list
```

If the LLM model is missing:

```bash
ollama pull llama3.2:1b
```

If the embedding model is missing:

```bash
ollama pull nomic-embed-text
```

Also check that the model name in the code matches exactly.

**EXAMPLE:**
If your code says:

```python
model = "llama3.2:1b"
```

but `ollama list` only shows:

```text
llama3
```

then the names do not match.

Either install the correct model:

```bash
ollama pull llama3.2:1b
```

or change the code to use the model you actually have installed.

---

### If embeddings fail

**WHY:**
The embedding model may not be installed, Ollama may not be running, or empty text may be passed into the embedding function.

**HOW:**
Test embeddings with a small example:

```python
from local_embeddings import embed_texts

embeddings = embed_texts(["hello world"])
print(len(embeddings))
print(len(embeddings[0]))
```

If this fails, check Ollama and the embedding model.

If this works, check the text being passed into the embedding function.

**EXAMPLE:**
Expected output may look like this:

```text
1
768
```

or another embedding dimension depending on the model.

This means:

* one text was embedded
* the embedding is a vector with many numbers

If you get an error instead, check:

```bash
ollama list
```

and make sure `nomic-embed-text` is installed.

---

### If chunks are not being created

**WHY:**
The input documents may not have the expected format.

**HOW:**
Check that each document looks like this:

```python
{
    "doc_id": "doc_1",
    "filename": "example.txt",
    "text": "Document text here."
}
```

Make sure:

* `text` exists
* `text` is not empty
* documents are stored inside a list
* `doc_id` and `filename` are present

**EXAMPLE:**
This is correct:

```python
documents = [
    {
        "doc_id": "doc_1",
        "filename": "notes.txt",
        "text": "RAG retrieves relevant context before generating an answer."
    }
]
```

This is wrong because `text` is missing:

```python
documents = [
    {
        "doc_id": "doc_1",
        "filename": "notes.txt"
    }
]
```

This is also wrong because the document is not inside a list:

```python
documents = {
    "doc_id": "doc_1",
    "filename": "notes.txt",
    "text": "RAG retrieves relevant context before generating an answer."
}
```

---

### If chunks are created but look wrong

**WHY:**
The chunk size or overlap may not be appropriate, or the document text may be too short, too messy, or empty.

**HOW:**
Print the first few chunks:

```python
for chunk in chunks[:3]:
    print(chunk["chunk_id"])
    print(chunk["text"])
    print("---")
```

Check:

* Are chunks empty?
* Are chunks too tiny?
* Are chunks too large?
* Is important context split apart?
* Are `doc_id`, `filename`, and `chunk_id` preserved?

**EXAMPLE:**
A useful chunk looks like this:

```python
{
    "chunk_id": "doc_1_chunk_0",
    "doc_id": "doc_1",
    "filename": "notes.txt",
    "text": "RAG retrieves relevant chunks from documents and uses them as context for the LLM.",
    "start_char": 0,
    "end_char": 95
}
```

A bad chunk may look like this:

```python
{
    "chunk_id": "doc_1_chunk_0",
    "text": ""
}
```

If chunks are empty, go back and check the original document text.

---

### If retrieval returns irrelevant chunks

**WHY:**
The chunks may be poor, embeddings may be missing, the query may be weak, or `top_k` may be too low.

**HOW:**
Print the retrieved chunks:

```python
for result in results:
    print(result["chunk_id"], result.get("score"))
    print(result["text"][:300])
    print("---")
```

Then check:

* Do the chunks contain useful information?
* Are the similarity scores low?
* Is the query too vague?
* Does the document actually contain the answer?
* Should `top_k` be increased?

**EXAMPLE:**
If the question is:

```text
What does RAG retrieve?
```

a good retrieved chunk might contain:

```text
RAG retrieves relevant chunks from documents before sending context to the LLM.
```

A bad retrieved chunk might contain:

```text
Python is a programming language used for many types of software development.
```

If retrieval returns bad chunks, the issue is probably before generation. Fix retrieval before blaming the LLM.

---

### If semantic search returns nothing useful

**WHY:**
The query embedding and chunk embeddings may not be comparable, the chunks may not contain the answer, or the query may be worded too differently from the documents.

**HOW:**
Check that chunks have embeddings:

```python
for chunk in embedded_chunks[:3]:
    print(chunk["chunk_id"])
    print("embedding" in chunk)
    print(len(chunk["embedding"]))
```

Also test a simple query that you know should match your documents:

```python
results = semantic_search(
    query="retrieval augmented generation",
    embedded_chunks=embedded_chunks,
    top_k=3
)

for result in results:
    print(result["chunk_id"], result.get("score"))
    print(result["text"][:200])
```

**EXAMPLE:**
If your document contains:

```text
Retrieval-Augmented Generation uses external context to improve LLM answers.
```

then this query should retrieve it:

```text
What does retrieval augmented generation use?
```

If it does not, check embeddings, chunking, or the semantic search function.

---

### If Chroma returns old or strange results

**WHY:**
You may be querying an old persistent collection.

**HOW:**
Check:

* the Chroma database path
* the collection name
* whether current chunks were upserted
* whether old local data is still being reused

When testing, try a new collection name:

```python
collection_name = "test_rag_chunks_v2"
```

If needed, clear the local Chroma database folder, but only if you do not need the stored data.

**EXAMPLE:**
If you changed your documents but Chroma keeps returning chunks from yesterday, you are probably using an old persistent collection.

Try:

```python
collection_name = "debug_collection_new"
```

Then rebuild the collection and run retrieval again.

---

### If query rewriting makes results worse

**WHY:**
The rewritten query may be too different from the original question or may not match the wording in your documents.

**HOW:**
Print the original and rewritten query:

```python
from query_rewriting import rewrite_query

query = "How do LLMs use RAG?"
print("Original:", query)
print("Rewritten:", rewrite_query(query))
```

Compare retrieval with and without rewriting.

**EXAMPLE:**
Good rewrite:

```text
Original: How do LLMs use RAG?
Rewritten: how do large language models use retrieval augmented generation
```

Bad rewrite:

```text
Original: How do LLMs use RAG?
Rewritten: models generation retrieval something
```

If rewriting makes the query awkward, test retrieval using the original query.

---

### If multi-query retrieval returns duplicate results

**WHY:**
The same chunk may be retrieved by multiple query variants.

**HOW:**
Check that results are deduplicated by `chunk_id`.

Each chunk should only appear once in the final result list.

Print the returned chunk IDs:

```python
for result in results:
    print(result["chunk_id"])
```

**EXAMPLE:**
Bad output:

```text
doc_1_chunk_0
doc_1_chunk_0
doc_1_chunk_2
```

Good output:

```text
doc_1_chunk_0
doc_1_chunk_2
doc_3_chunk_1
```

If duplicates still appear, inspect the merge logic inside `multi_query.py`.

---

### If reranking does not improve results

**WHY:**
Reranking only reorders already retrieved chunks. If retrieval found bad chunks, reranking cannot fully fix that.

**HOW:**
Print the reranking scores:

```python
for chunk in reranked_chunks:
    print(chunk["chunk_id"])
    print("original:", chunk.get("original_score"))
    print("overlap:", chunk.get("overlap_score"))
    print("rerank:", chunk.get("rerank_score"))
    print("---")
```

If overlap scores are zero, check tokenization, query wording, and whether the retrieved chunks share terms with the query.

**EXAMPLE:**
For the query:

```text
What is retrieval augmented generation?
```

a relevant chunk may have:

```text
overlap: 3
```

because it shares words like:

```text
retrieval
augmented
generation
```

An irrelevant chunk may have:

```text
overlap: 0
```

If all chunks have overlap `0`, the query and retrieved text may not share useful terms, or tokenization may be too aggressive.

---

### If the system keeps saying “I don't know based on the provided context.”

**WHY:**
The answerability gate probably decided the retrieved context was not strong enough.

**HOW:**
Inspect the retrieved chunks:

```python
for chunk in retrieved_chunks:
    print(chunk["chunk_id"])
    print(chunk.get("score"))
    print(chunk["text"][:300])
    print("---")
```

If the chunks are irrelevant, fix retrieval first.

If the chunks are relevant but the system still says it does not know, the thresholds may be too strict. Try lowering them temporarily:

```python
min_score = 0.1
min_overlap = 1
```

**EXAMPLE:**
If the question is:

```text
What does RAG use to answer questions?
```

and the retrieved chunk says:

```text
RAG uses retrieved document context to ground the LLM answer.
```

but answerability is still `False`, your thresholds may be too strict.

If the retrieved chunk says:

```text
Python supports functions, classes, and modules.
```

then answerability should be `False`, because the context does not answer the question.

---

### If the prompt is `None`

**WHY:**
The answerability gate likely decided that the retrieved context was not strong enough to answer the question.

**HOW:**
Check the answerability value and retrieved chunks:

```python
print(rag_input["answerable"])
print(rag_input["retrieved_chunks"])
```

If `answerable` is `False`, the system intentionally skipped prompt construction and returned the fallback answer.

**EXAMPLE:**
This is expected behavior:

```python
{
    "answerable": False,
    "prompt": None,
    "fallback_answer": "I don't know based on the provided context."
}
```

It means the system refused to build a prompt because the evidence was weak.

---

### If the LLM gives an unsupported answer

**WHY:**
The retrieved context may be noisy, the prompt may be too loose, or the answerability gate may be too permissive.

**HOW:**
Print the final prompt:

```python
print(rag_input["prompt"])
```

Check:

* Is the context relevant?
* Is there too much irrelevant context?
* Does the prompt tell the model to answer only from context?
* Should answerability thresholds be stricter?

**EXAMPLE:**
If your context says:

```text
RAG retrieves relevant chunks from documents.
```

but the LLM answers with unrelated information about model training, the prompt may need a stronger instruction like:

```text
Answer only using the context. If the context does not contain the answer, say you do not know.
```

---

### If local LLM generation times out

**WHY:**
The model may be slow on your machine, the prompt may be too long, or the timeout may be too short.

**HOW:**
Try:

* using a smaller model
* increasing the timeout
* reducing `top_k`
* shortening the retrieved context

Test Ollama directly:

```bash
ollama run llama3.2:1b
```

**EXAMPLE:**
If generation works in the terminal but fails in Python, the issue may be your Python request timeout.

If generation is slow everywhere, the model may be too heavy for your machine.

---

### If Graph-RAG does not add extra chunks

**WHY:**
The graph may not contain useful entities or connections.

**HOW:**
Print the graph summary:

```python
summary = summarize_graph(graph)
print(summary)
```

Check:

* Are there entity nodes?
* Are there chunk nodes?
* Are there edges?
* Do retrieved chunks share entities with other chunks?

**EXAMPLE:**
A useful graph summary may look like:

```python
{
    "num_nodes": 25,
    "num_edges": 40,
    "num_chunk_nodes": 8,
    "num_entity_nodes": 17
}
```

If you see:

```python
{
    "num_nodes": 8,
    "num_edges": 0,
    "num_chunk_nodes": 8,
    "num_entity_nodes": 0
}
```

then entity extraction probably failed or found no entities.

---

### If Self-RAG does not retry

**WHY:**
The first attempt may already be considered answerable, or `max_attempts` may be set to `1`.

**HOW:**
Print each attempt:

```python
for attempt in result["attempts"]:
    print(attempt["attempt"])
    print(attempt["query"])
    print(attempt["answerable"])
```

If there is only one attempt, check `max_attempts`.

If the first attempt is answerable, Self-RAG stops early on purpose.

**EXAMPLE:**
This means Self-RAG retried:

```text
1
What is RAG?
False

2
what is retrieval augmented generation
True
```

This means it stopped early because the first attempt worked:

```text
1
What is RAG?
True
```

---

### If evaluation accuracy is low

**WHY:**
The issue may be retrieval, answerability, generation, or expected test data.

**HOW:**
Inspect each evaluation result:

```python
for result in evaluation["results"]:
    print(result["question"])
    print("Expected:", result["expected_doc_id"])
    print("Returned:", result["retrieved_doc_ids"])
    print("Answerable:", result["predicted_answerable"])
    print("Success:", result["overall_success"])
    print("---")
```

If the expected document is not retrieved, debug retrieval.

If retrieval works but answerability fails, debug thresholds.

If retrieval and answerability work but answers are bad, inspect the prompt and LLM output.

**EXAMPLE:**
If you see:

```text
Expected: doc_2
Returned: ['doc_5', 'doc_1']
Success: False
```

the problem is retrieval.

If you see:

```text
Expected: doc_2
Returned: ['doc_2']
Answerable: False
Success: False
```

the problem is likely answerability.

If you see:

```text
Expected: doc_2
Returned: ['doc_2']
Answerable: True
Success: False
```

then inspect the generated answer and prompt.

---

## Best Debugging Habit

As a learner, do not ask only:

```text
Why is the final answer wrong?
```

Ask:

```text
What did this step receive?
What did this step return?
Does that output make sense for the next step?
```

This helps you find where the pipeline first breaks instead of guessing at the end.

The goal is that you do not just know what failed, but also understand why it failed and how that part of the RAG pipeline affects the final answer, Most RAG problems come from retrieval, chunking, embeddings, or prompt construction before the LLM ever answers, so it is better to debug the LLM as a later step.

## How to Read the Comments

The code includes learning-focused comments to make the project easier to follow for people learning RAG from the ground up.

The comments are organized around four labels:

- **WHAT:** what the function or code block does
- **WHY:** why this step exists in the RAG pipeline
- **HOW:** how the logic works step by step
- **EXAMPLE:** a concrete example of how the function or concept is used

Pay attention to comments that explain:

- why a step exists
- what problem it solves
- what the input and output look like
- how the component fits into the full RAG pipeline

These comments are meant to make the code easier to follow for people learning RAG from the ground up. The goal is not only to show working code, but also to explain the reasoning behind each component.

The goal is that you do not just know what the code does, but also understand why each step exists and how the pieces connect.

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
