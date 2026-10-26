# Chapter 5 — Retrieval-Augmented Generation (RAG)

## Overview

Retrieval-Augmented Generation (RAG) improves an LLM's answers by retrieving relevant information from an external knowledge source and adding it to the model's context. Instead of relying only on parameters learned during training, the agent can ground its response in retrieved documents or data.

A typical RAG pipeline has four stages:

1. **Ingest** source material.
2. **Split and index** the material into retrievable units.
3. **Retrieve** the most relevant units for a user query.
4. **Generate** an answer using the query and retrieved context.

Retrieval quality matters: irrelevant, incomplete, or poorly structured context can lead to weak answers even when the language model itself is capable.

## Search approaches

### Keyword-based search

Keyword search matches query terms against document text. It is simple, fast, and useful when exact words, names, identifiers, or phrases matter. Its limitation is that it may miss relevant passages that use different wording or synonyms.

### Vector search and embeddings

An embedding model converts text into a numerical vector. Texts with similar meanings are often represented by vectors that are close in the embedding space, making semantic retrieval possible even when the query and document do not share exact keywords.

Cosine similarity is commonly used to compare two vectors:

\[
\operatorname{cosine}(A,B)=\frac{A\cdot B}{\|A\|\,\|B\|}
\]

A higher cosine similarity generally indicates a smaller angle between the vectors and greater similarity in direction. Embeddings do not guarantee that two passages are factually equivalent; they provide a signal for ranking candidates.

### Graph-based search

Graph retrieval represents information as entities and relationships. It can help answer questions that require connecting facts across multiple entities or following relationships rather than finding one similar passage. Its usefulness depends on the quality of the graph and the relationships extracted from the source data.

### Structured search

Structured retrieval queries organized data, such as records, tables, fields, or databases. It is useful when a question asks for precise values or filtered results. The agent must understand the schema and formulate an appropriate query.

### Hybrid search

Hybrid search combines methods, often keyword and vector retrieval. Keyword search can capture exact terms while vector search captures semantic similarity. Combining their results can improve coverage, though the results may need normalization or reranking before selecting the final context.

## Chunking

Long documents are commonly divided into smaller pieces called **chunks** before indexing. Chunking makes retrieval more focused and helps keep the context passed to the model within a practical size.

Important considerations include:

- **Chunk size:** small chunks are precise but may lose context; large chunks preserve context but can include irrelevant material.
- **Overlap:** a small overlap between adjacent chunks can preserve information that crosses a boundary.
- **Structure-aware splitting:** headings, paragraphs, sections, or other natural boundaries can produce more meaningful chunks than splitting only by character count.
- **Metadata:** retain information such as source file, section, or page when it helps the system identify and cite retrieved material.

There is no universally best chunk size. It should be chosen based on the source format, query style, retrieval method, and context limits.

## Choosing and combining retrieval methods

Use the retrieval method that matches the data and the question:

| Method | Often useful for | Main limitation |
|---|---|---|
| Keyword search | Exact terms, names, IDs | Can miss paraphrases |
| Vector search | Semantic similarity | Similarity is not proof of relevance |
| Graph search | Relationships between entities | Requires a useful graph |
| Structured search | Exact fields, filters, calculations | Requires structured data and schema knowledge |
| Hybrid search | Queries needing both lexical and semantic matching | More components to tune and evaluate |

In practice, a system may route different questions to different retrieval tools or combine several methods in one pipeline.

## Working with files and datasets

When a task involves a dataset such as GAIA, first understand its file format and schema before processing it. Check how examples are represented, identify the fields that contain the task and expected answer, and handle JSON or JSONL correctly. JSON usually stores a complete JSON value, while JSONL stores one JSON value per line.

Keep file handling separate from reasoning where possible, and provide clear errors for missing files, malformed records, or unexpected fields. For evaluation, start with a small subset so that failures are easy to diagnose and any model/API costs remain controlled.

## Callbacks and human-in-the-loop workflows

Callbacks allow code to run at defined points in a workflow—for example, to log progress, inspect an event, or react to a tool result. They can make agent execution easier to observe and debug.

**Human-in-the-loop (HITL)** means involving a person at an appropriate point in the process. An agent might request approval before a consequential action, ask for clarification when information is insufficient, or let a person review a proposed result. The system should make it clear when it is waiting for human input and should not treat missing approval as approval.

## Compressing search results

Search tools can return many long passages, while an LLM has a limited context window. Search-result compression reduces the retrieved material to the information most relevant to the current question.

A useful compression step should:

- Preserve facts that directly support the answer.
- Keep important qualifiers, dates, units, and source information.
- Remove repetition and unrelated detail.
- Avoid inventing information or making uncertain statements sound certain.
- Retain enough provenance to trace important claims back to their sources.

Compression is not simply making text shorter: it is preserving the evidence needed for a reliable answer while reducing noise.

## Practical takeaways

1. RAG grounds generation in retrieved external information.
2. Keyword, vector, graph, and structured retrieval solve different problems; hybrid approaches can combine their strengths.
3. Embeddings and cosine similarity support semantic ranking, but retrieval results still need relevance checks.
4. Chunk size, overlap, boundaries, and metadata affect retrieval quality.
5. File formats and schemas should be understood before processing evaluation datasets.
6. Callbacks improve observability, and HITL can add review or approval to sensitive steps.
7. Compress retrieved context carefully so the model receives concise evidence without losing key facts or provenance.

## A simple mental model

**Prepare the knowledge → retrieve evidence → select and compress context → generate a grounded answer → involve a human when needed.**

RAG quality depends on the whole pipeline, not just the final prompt or the language model.
