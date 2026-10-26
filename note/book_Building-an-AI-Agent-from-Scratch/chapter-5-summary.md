# Chapter 5 Summary — Building an AI Agent from Scratch

## 1. Introduction: Why Do AI Agents Need Knowledge Bases?

Large language models (LLMs) have limitations:
- Their knowledge may be outdated.
- They may not know private or project-specific information.
- They can generate incorrect information when they lack relevant context.
- They have limited context windows, so providing too much information can be inefficient.

A **knowledge base** allows an AI agent to retrieve relevant information from external sources and use it to answer questions more accurately.

## 2. Retrieval-Augmented Generation (RAG)

**RAG** stands for Retrieval-Augmented Generation. It combines information retrieval with LLM-based answer generation.

The process has three main steps:

1. **Retrieval:** Search a knowledge base for information relevant to the user's question.
2. **Augmentation:** Add the retrieved information to the prompt as context.
3. **Generation:** Ask the LLM to generate an answer based on the question and retrieved context.

RAG can reduce hallucinations, improve factual accuracy, and make it possible to use private or frequently updated information without retraining the model.

## 3. Types of Search

Chapter 5 introduces four major search approaches.

### 3.1. Keyword-Based Search

Keyword search looks for exact words or phrases in documents.

**Advantages:**
- Simple to implement.
- Fast for exact matches.
- Useful for names, identifiers, and specific terms.

**Limitations:**
- May miss relevant information expressed using different words.
- Does not necessarily understand the meaning of a query.

### 3.2. Vector Search

Vector search represents text as numerical vectors called **embeddings**. It retrieves information based on semantic similarity rather than exact word matches.

For example, a search for "How can I reduce expenses?" may retrieve a passage about "cost optimization," even when the wording differs.

Vector search is especially useful for semantic search and RAG systems.

### 3.3. Graph-Based Search

Graph search represents information as entities and relationships.

For example:

- Alice works for Company A.
- Company A owns Product B.
- Product B is used by Customer C.

A graph-based system can follow these relationships to answer questions that require connecting multiple pieces of information.

### 3.4. Structure-Based Search

Structure-based search uses the organization and format of data to locate information.

Examples include:
- Searching file and folder names.
- Listing files inside an archive.
- Reading specific sections of a document.
- Extracting information from CSV and Excel files.
- Identifying relevant images, audio files, and PDFs.

This approach is useful when information is stored in structured files or when an agent must first explore an unfamiliar collection of files.

### 3.5. Hybrid Search

Hybrid search combines multiple search techniques, such as keyword search and vector search.

Combining approaches can improve retrieval quality because different techniques have different strengths.

## 4. Embeddings and Cosine Similarity

An **embedding** is a numerical representation of text that captures aspects of its meaning.

Texts with similar meanings often have embeddings that are close together in vector space.

A common method for measuring similarity is **cosine similarity**:

\[
\operatorname{similarity}(A,B)=
\frac{A\cdot B}{\|A\|\|B\|}
\]

Here:
- \(A\) and \(B\) are embedding vectors.
- \(A\cdot B\) is their dot product.
- \(\|A\|\) and \(\|B\|\) are their magnitudes.

A higher cosine similarity generally indicates greater directional similarity between two vectors.

Embeddings allow an agent to retrieve passages based on meaning rather than exact word matches.

## 5. Chunking and Vector Retrieval

Documents are often too long to process as a single unit. **Chunking** divides documents into smaller pieces.

The book demonstrates a configuration with:
- `chunk_size = 500`
- `chunk_overlap = 50`

The exact meaning of the chunk size depends on the implementation; it may refer to characters, tokens, or another unit.

Overlap preserves some context between neighboring chunks and reduces the chance of splitting important information across boundaries.

### Vector Search Pipeline

A typical vector retrieval pipeline works as follows:

1. Load documents.
2. Split documents into chunks.
3. Generate embeddings for the chunks.
4. Store the embeddings in a vector index or vector database.
5. Generate an embedding for the user's query.
6. Compare the query embedding with stored embeddings.
7. Retrieve the most relevant chunks, often using a parameter such as `top_k`.
8. Add the retrieved chunks to the LLM prompt.
9. Generate the final answer.

For large collections, approximate nearest neighbor (ANN) algorithms can make similarity search more efficient.

Examples of vector databases and search systems include FAISS, Chroma, and other vector database solutions.

## 6. The Chapter's Retrieval Exercise

The chapter demonstrates how retrieving only relevant information can reduce the amount of context sent to an LLM.

In the book's example, the reported token count decreased from 37,312 to 304, representing approximately a 99.2% reduction.

This result belongs to the specific example in the book; actual savings depend on the documents, retrieval method, query, and selected chunks.

The main lesson is that an agent does not always need to send an entire document or search result to the LLM. It can retrieve and use only the most relevant information.

## 7. Structure-Based Search with Files

The chapter explores working with attachments from the GAIA dataset.

The preparation process includes:
- Loading a dataset with `load_dataset()`.
- Downloading relevant files with `snapshot_download()`.
- Preparing or resetting a workspace.
- Exploring files and folders before deciding what to read.

The agent can use tools to inspect files and determine where relevant information is stored.

### File-Handling Tools

The chapter introduces four important tool concepts:

**1. `unzip_file`**

Extracts the contents of a ZIP archive so the agent can inspect its files.

**2. `list_files`**

Lists files and directories in a workspace. This helps the agent understand the available file structure.

**3. `read_file`**

Reads supported text-based or tabular files, such as:
- Text files.
- CSV files.
- Excel spreadsheets.

**4. `read_media_file`**

Processes media files, such as:
- Images.
- Audio.
- PDFs.

Depending on the implementation, this may involve extracting text from PDFs or using a vision-capable model to interpret visual content.

### Important Design Principle

An agent should inspect the available files before attempting to answer a question. It should identify the relevant files, select appropriate tools, extract the necessary information, and then produce an answer grounded in that information.

## 8. Callbacks

Callbacks allow developers to execute custom logic at specific stages of an agent's lifecycle.

The chapter covers callbacks associated with:
- Before an LLM call.
- After an LLM call.
- Before a tool call.
- After a tool call.
- After an agent run.

Callbacks can be used for logging, monitoring, validation, approval workflows, and result processing.

### Before-Tool Callback

The example uses the following behavior:

- Returning `None` allows the tool to execute normally.
- Returning a non-`None` value skips the tool execution and uses that value as the tool result.

This mechanism can be used to intercept tool calls, enforce policies, or request approval.

### After-Tool Callback

The example demonstrates that:

- Returning `None` preserves the original tool result.
- Returning a non-`None` value replaces the original result.

This is useful for cleaning, filtering, summarizing, or validating tool output.

### Early Exit

The chapter also explains that if a callback in a callback chain returns a non-`None` value, the chain may stop early, depending on the callback mechanism being used.

Always check the framework's callback contract before implementing this behavior in another codebase.

## 9. Human-in-the-Loop (HITL)

**Human-in-the-Loop (HITL)** means involving a human in an agent's decision-making or execution process.

For example, an agent may ask for approval before:
- Executing a sensitive tool.
- Deleting or modifying files.
- Performing an important external action.
- Running a potentially expensive operation.

The chapter demonstrates a simple approval mechanism using Python's `input()` function.

This is useful for understanding the concept, but `input()` is synchronous and blocks execution while waiting for a response. A production application may need a non-blocking approval workflow, a user interface, or an asynchronous request-and-resume mechanism.

## 10. Compressing Search Results

Search tools may return large amounts of information, even when only a small portion is relevant to the user's question.

The chapter demonstrates using a callback to compress search results.

The basic process is:

1. Execute a search tool.
2. Receive the full search results.
3. Compare the results with the original user query.
4. Select the most relevant chunks.
5. Keep only the most useful results, such as the top three chunks in the example.
6. Return the compressed output to the agent.

The implementation may use vector search to identify relevant chunks.

The callback must associate the search output with the correct tool call and query. In the book's example, the tool call identifier is used to connect the result with the corresponding call.

This technique can reduce token consumption, lower processing costs, and help the LLM focus on relevant information.

## 11. Applying Chapter 5 to My GitHub Project

**Repository:**

`https://github.com/Ghassem296/build-an-ai-agent-from-scratch`

**Local project directory:**

`C:\Users\Ghassem BAKHTIARI\Desktop\build-an-ai-agent-from-scratch`

### Current Project Structure

Important files include:

- `app/agent.py` — Agent logic and system prompt.
- `app/llm_client.py` — LLM communication.
- `app/mcp_runtime.py` — MCP tool runtime integration.
- `app/execution_context.py` — Execution context.
- `app/config.py` — Configuration.
- `app/main.py` — Application entry point.
- `tools/mcp_server.py` — MCP tools.

The project uses a local Ollama server at `http://127.0.0.1:11434` and the `qwen2.5:3b` model.

The MCP server already includes a `search_web` tool using `DDGS().text(...)`.

A weather tool named `get_weather(city)` has also been added to `tools/mcp_server.py`. It uses the Open-Meteo geocoding and forecast APIs.

The default system prompt in `app/agent.py` has been updated to direct weather-related queries to `get_weather`.

### Possible Future Improvements

After studying the chapter, potential improvements include:

1. Add a document-reading tool.
2. Add file-listing and archive-extraction tools.
3. Build a document chunking pipeline.
4. Generate embeddings for document chunks.
5. Implement vector-based retrieval.
6. Integrate RAG into the agent's workflow.
7. Add callbacks for logging and result processing.
8. Add human approval for sensitive tool calls.
9. Compress large search results before returning them to the LLM.

These are future possibilities, not changes that have already been implemented.

Before adding any of these features, inspect the existing project architecture and decide how the new functionality should integrate with it.

## 12. Key Takeaways

The most important lessons from Chapter 5 are:

- RAG allows agents to use relevant external information when answering questions.
- Keyword search and vector search solve different retrieval problems.
- Embeddings represent text numerically and enable semantic similarity search.
- Chunking makes large documents easier to retrieve and process.
- Structure-based search helps agents work with files, folders, archives, and media.
- Callbacks provide control over different stages of an agent's execution.
- HITL allows humans to approve or reject important actions.
- Search-result compression can reduce token usage and improve relevance.
- A reliable agent should retrieve the information it needs instead of blindly sending all available information to the LLM.

**Overall conclusion:** Chapter 5 explains how to make AI agents more useful and efficient by giving them access to external knowledge, teaching them to retrieve relevant information, and providing mechanisms to control and improve their execution.

---

## Chapter 5 — Teaching Outline and Learning Roadmap

### Teaching Instructions

Teach this chapter step by step, following the outline below. Each lesson should include:
- A clear explanation of the concept.
- A practical example, preferably in Python.
- A discussion of why the concept matters when building AI agents.
- A small exercise or coding task.
- A connection to the existing GitHub project when relevant.

Do not skip topics or assume prior knowledge of concepts that have not yet been explained. Clearly distinguish between concepts covered in the book and additional implementation ideas for our project.

### Lesson 1: Knowledge Bases and LLM Limitations
- What is a knowledge base?
- What information does an LLM already know?
- Limitations of an LLM's internal knowledge.
- Knowledge freshness, missing information, and hallucinations.
- Why an AI agent may need external knowledge sources.
- When external retrieval is useful.

### Lesson 2: Retrieval-Augmented Generation (RAG)
- What is Retrieval-Augmented Generation?
- The three main stages: retrieval, augmentation, and generation.
- How retrieved information is added to an LLM prompt.
- How RAG differs from relying only on the LLM's internal knowledge.
- Benefits and limitations of RAG.
- A simple end-to-end RAG example.

### Lesson 3: Search and Retrieval Methods
- Keyword-based search.
- Semantic search.
- Vector search.
- Graph-based search.
- Structure-based search.
- Hybrid search.
- Strengths and weaknesses of each method.
- Choosing a search method for a particular task.

### Lesson 4: Embeddings and Similarity
- What embeddings are.
- How text is represented as numerical vectors.
- How semantically similar texts can have similar embeddings.
- The role of embedding models.
- Cosine similarity and its intuition.
- Comparing text chunks using similarity scores.
- A practical Python example.
- Important limitations of embedding-based retrieval.

### Lesson 5: Document Chunking
- Why long documents are divided into smaller chunks.
- What a chunk is.
- Chunk size and chunk overlap.
- The effects of chunk size on retrieval quality and context.
- The trade-offs between small and large chunks.
- Example settings: `chunk_size=500` and `chunk_overlap=50`.
- How to choose and evaluate chunking settings.
- A practical document-chunking exercise.

### Lesson 6: Building a Basic Vector Retrieval Pipeline
- Loading and preparing documents.
- Splitting documents into chunks.
- Generating embeddings.
- Storing embeddings in a vector store.
- Embedding a user query.
- Retrieving the most relevant chunks.
- Supplying retrieved chunks to the LLM.
- Generating an answer grounded in retrieved information.
- Understanding the difference between a conceptual example and a production-ready implementation.

### Lesson 7: Improving Retrieval Quality and Reducing Token Usage
- Why sending an entire knowledge base to an LLM is inefficient.
- Selecting relevant information instead of sending everything.
- Ranking retrieved chunks by relevance.
- Choosing the top-k results.
- Understanding the trade-off between context size and answer quality.
- Measuring token usage before and after targeted retrieval.
- Interpreting the chapter's example of reducing the input from 37,312 tokens to 304 tokens.
- Why token reduction depends on the documents, query, and retrieval method.
- A practical retrieval optimization exercise.

### Lesson 8: Structure-Based Search
- What structure-based search means.
- When document structure is more useful than semantic similarity.
- Working with folders, files, archives, and structured datasets.
- Listing files and inspecting their organization.
- Finding relevant information before reading entire files.
- Understanding the chapter's GAIA dataset preparation example.
- Deciding when to use structure-based search instead of vector search.

### Lesson 9: Tools for Exploring and Reading Files
- The purpose of `unzip_file`.
- The purpose of `list_files`.
- The purpose of `read_file`.
- The purpose of `read_media_file`.
- How file tools help an agent inspect external information.
- How an agent chooses which tool to call.
- Tool inputs, outputs, and error handling.
- Safe file access and limitations.
- A practical exercise using file exploration tools.

### Lesson 10: Callbacks in AI Agents
- What callbacks are.
- Why callbacks are useful in an agent execution lifecycle.
- Before-LLM and after-LLM callbacks.
- Before-tool and after-tool callbacks.
- Before-agent and after-agent callbacks.
- Inspecting inputs and outputs.
- Logging, monitoring, validation, and policy enforcement.
- How callback return values can affect execution.
- The difference between allowing an operation, skipping it, and replacing its result.
- Why callback behavior must be verified against the specific framework or implementation.
- A practical callback example.

### Lesson 11: Human-in-the-Loop (HITL)
- What Human-in-the-Loop means.
- Why agents sometimes need human approval.
- Using `input()` as a simple demonstration.
- Pausing an agent before a sensitive action.
- Approving, rejecting, or modifying a proposed action.
- Human review of tool calls and their results.
- The difference between a learning example and a production approval workflow.
- Designing clear approval boundaries.
- A practical HITL exercise.

### Lesson 12: Compressing and Filtering Search Results
- Why retrieved results may contain irrelevant information.
- Selecting only the most useful chunks.
- Ranking and filtering search results.
- Returning the top three results when appropriate.
- Summarizing retrieved content without losing essential facts.
- Balancing relevance, completeness, and token cost.
- Using vector similarity or other ranking methods to improve selection.
- Evaluating whether compression preserves the information needed to answer a query.

### Lesson 13: Applying Chapter 5 to Our GitHub AI Agent
- Review the existing project architecture.
- Understand the responsibilities of `app/agent.py`.
- Understand the role of `app/llm_client.py`.
- Understand the MCP runtime in `app/mcp_runtime.py`.
- Understand the execution context and conversation history.
- Review the available tools in `tools/mcp_server.py`.
- Identify which Chapter 5 concepts are already supported.
- Identify which concepts are not yet implemented.
- Design a retrieval workflow that fits the current architecture.
- Decide whether to begin with file-based retrieval, keyword search, or vector search.
- Integrate retrieval without breaking the existing think-act loop.
- Consider token budgets, error handling, timeouts, and tool-output limits.
- Keep retrieval logic modular and testable.
- Document all new behavior and configuration.

### Lesson 14: Testing and Evaluating the Implementation
- Write tests for document loading and chunking.
- Test retrieval with known queries and expected results.
- Check whether retrieved chunks contain the required information.
- Test empty results and missing files.
- Test malformed inputs and tool failures.
- Measure token usage and response quality.
- Compare retrieval strategies.
- Test how the agent behaves when the knowledge base does not contain an answer.
- Verify that retrieved content is treated as data, not as trusted instructions.
- Update the project documentation and record known limitations.

### Final Review and Completion Checklist

By the end of this chapter, the learner should be able to:

- [ ] Explain why LLMs may need external knowledge.
- [ ] Describe the RAG pipeline and its three stages.
- [ ] Compare keyword, semantic, vector, graph, structure-based, and hybrid search.
- [ ] Explain embeddings and cosine similarity.
- [ ] Split documents into chunks and explain chunk size and overlap.
- [ ] Describe the steps of a basic vector retrieval pipeline.
- [ ] Explain how retrieval can reduce unnecessary token usage.
- [ ] Identify situations where structure-based search is appropriate.
- [ ] Explain the purpose of the file tools covered in the chapter.
- [ ] Describe callback execution and the effect of callback return values.
- [ ] Explain when and why Human-in-the-Loop approval is useful.
- [ ] Filter and compress retrieved information while preserving relevance.
- [ ] Identify which Chapter 5 concepts exist in our GitHub project and which require new implementation.
- [ ] Propose, implement, and test an appropriate retrieval improvement for the project.

**Teaching rule:** Complete one lesson at a time. Do not move to the next lesson until the current topic has been explained, demonstrated, and practiced. When implementation is relevant, inspect the current repository before proposing code changes. Never describe a feature as implemented unless the code confirms it.
