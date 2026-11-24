# Chapter 6 Summary — Adding Memory to Your Agent

> **Note:** This document is based on Chapter 6 of *Build an AI Agent (From Scratch)* and follows the same structure as the Chapter 5 summary. The "Applying to the GitHub Project" section assumes the project described in the Chapter 5 summary (Ollama + qwen2.5:3b + local MCP server). That section is a **roadmap and set of suggestions**, not a claim that anything has been implemented. Inspect the current code before making any changes.

## 1. Introduction: Why Do AI Agents Need Memory?

Without memory, an agent starts from scratch on every interaction. This creates three core problems:

- **Problem of space:** When solving complex problems, tool results accumulate rapidly and the context can exceed the model's window, causing the agent to crash before it finishes.
- **Problem of time:** Nothing is remembered between sessions; users must repeat information they already provided.
- **Lack of persistent knowledge:** Information such as "I'm a marketer" should remain available in future sessions.

Memory transforms an agent from a stateless tool into an assistant that learns and adapts.

## 2. The Anatomy of Agent Memory

Three foundational concepts:

| Concept | Definition |
|---|---|
| **Session** | A single continuous conversation between a user and an agent (like one chat window). |
| **Short-term memory** | Work done within a single session: conversation history, tool results, intermediate reasoning — stored in `ExecutionContext.events`. |
| **Long-term memory** | Information that persists after a session ends and is retrieved in future sessions. |

**How items enter:** Short-term memory automatically captures everything the agent does. Long-term memory is **selective** — at the end of each run, an extraction step decides which facts are worth saving.

### Three Challenges of the Current Memory Architecture

**Challenge 1: Context explosion** — Fine for one-off runs, but in complex problems with many tool calls, context grows rapidly. The question: how do we compress context *during* execution?

**Challenge 2: Lack of conversation continuity** — Without session management, each `run()` is independent and the conversation breaks (e.g., the user says "Project Alpha" and the agent doesn't know which project on the next message). In web services, servers are stateless; maintaining context requires explicit work: a unique `session_id`, saving session state to external storage when `run()` ends, and loading it when the next `run()` starts.

**Challenge 3: Lack of persistent knowledge** — Information must be shared *across different* `session_id`s. This differs fundamentally from Challenge 2: conversation continuity maintains context within the same session; persistent knowledge lets the agent remember the user when a *new* conversation starts.

## 3. Context Engineering and Memory

Chapter 1 defined five context-engineering strategies. Chapter 6 is a hub where three of them converge.

| Strategy | Description | Related Chapters |
|---|---|---|
| Generate | Constructing context with LLM-generated outputs | 7 |
| Retrieve | Bringing information from external sources into context | 3, 5, 6 |
| Write | Saving from context to external storage | 6, 8 |
| Reduce | Compressing and deleting context | 6 |
| Isolate | Separating tasks/tools into distinct environments | 8, 9 |

The chapter's three use patterns map to these strategies:

| Pattern | Primary Strategy | Description |
|---|---|---|
| **Pattern 1: In-run context management** | Reduce + Write | Managing extensive context via sliding window, compaction, summarization |
| **Pattern 2: Continuous execution** | Reduce + Write | Maintaining context across multiple `run()` calls within the same session |
| **Pattern 3: Long-term memory** | Write + Retrieve | Extracting and storing memories, then searching via a tool in a new session |

Patterns 1 and 2 are grouped because they both solve the problem of managing the context sent to the LLM *now*. Pattern 1 works with events inside `ExecutionContext`; Pattern 2 loads events stored in a `Session` and processes them the same way.

## 4. Managing Context During Execution (Pattern 1)

### 4.1 Separating Storage from Presentation

The first principle: **all events in `ExecutionContext.events` remain immutable; only the `LlmRequest.contents` sent to the LLM are processed.**

Why this matters:
- Without the original history, tracing the cause of failures is difficult.
- Some tools may reference results from previous operations.
- You can change how the LLM sees context without migrating or modifying stored data.

Implementation is via `before_llm_callback`: it receives both `ExecutionContext` and `LlmRequest`, reads state from the context, and modifies the request. Returning `None` lets the LLM call proceed; returning an `LlmResponse` skips the call and uses that response instead.

A factory function `create_optimizer_callback(apply_optimization, threshold)` generates a callback that applies the optimization only when the token count exceeds the threshold.

### 4.2 Sliding-Window Strategy

The simplest approach: keep only the most recent *n* items.

Three steps:
1. Find the user's original question in the contents.
2. Preserve everything up to the user message (so the agent never loses the problem).
3. Keep only the most recent `window_size` items from the remainder.

**Limitations:** If *n* is small you lose important context; if *n* is large it doesn't solve the problem. Message count is not a good indicator of context size — a single file-read result can be tens of thousands of tokens.

### 4.3 Token Counting

The LLM's real limit is determined by token count. `tiktoken` (OpenAI's tokenizer library) gives exact counts for GPT models and a rough estimate for others.

`count_tokens(request)` works by converting the request to API message format and counting:
- Per-message overhead (~4 tokens each)
- Content tokens
- `tool_calls` tokens (function name + arguments)
- Tool definition tokens (JSON-serialized)

**Why optimize even below the limit:**
- Cost and latency grow with token count.
- Irrelevant information distracts the model — the "lost in the middle" phenomenon.
- You must reserve space for the model's response.
- Research (context rot) shows performance degrades with input length even on simple tasks.

Rule of thumb: minimize context to what's required. The book uses a small threshold (~20,000 tokens) to make the behaviors observable.

### 4.4 Compaction Strategy

Core idea: **replace already-processed data with lightweight references.** The agent sees "Search results processed. Re-search if needed." and can re-call the tool if necessary.

Two types:
- **ToolCall compression (write operations):** e.g., `create_file` — remove the large `content` argument and replace it with a reference message like `"[Content saved to file]"`.
- **ToolResult compression (read operations):** e.g., `read_file`, `search_web` — replace the original data with a reference such as `"File content from {file_path}. Re-read if needed."`

**Trade-off:** Re-calling the tool isn't free — it costs API fees, latency, and (for search) may return slightly different results. Compaction pays off when the discarded blob is large relative to refetch costs.

**Implementation:** A single loop over `request.contents` processes both `ToolCall` and `ToolResult`. Because `ToolCall` always appears before its corresponding `ToolResult`, arguments can be saved when the call is encountered and used to compress the result later. Two rule tables control behavior: `TOOLCALL_COMPACTION_RULES` and `TOOLRESULT_COMPACTION_RULES`.

**Caching consideration:** Context optimization invalidates prompt caching. The solution is threshold-based optimization — keep context stable below the threshold to preserve caching, and modify only when overflow cost exceeds cache-miss cost.

### 4.5 Summarization Strategy

Compaction only targets specific tool calls/results. If general conversation or reasoning becomes lengthy, summarization is the last resort.

**Target selection:** Summarize the *oldest* intermediate messages first, but preserve the user's original question. The agent must not forget what it's currently solving.

**Summary prompt** asks the LLM to extract:
1. Key findings
2. Tools used
3. Current status (what's done, what remains)

**Implementation (`apply_summarization`)** has four steps:
1. Find the user-message position.
2. Keep the most recent `keep_recent` messages in original form.
3. Summarize the intermediate range via the LLM.
4. Reconstruct as "user question + summary message (appended to instructions) + recent messages."

Recording `last_summary_idx` in `context.state` prevents re-summarizing already-summarized portions.

**Compaction vs. Summarization:**

| | Compaction | Summarization |
|---|---|---|
| Nature | Deterministic (predictable from tool names) | Semantic (LLM decides what matters) |
| Cost/speed | Cheaper, faster | Additional LLM call |
| Reversibility | Reversible (re-call the tool) | Irreversible (original reasoning lost) |

Therefore: try compaction first; use summarization only when compaction is insufficient.

### 4.6 Hierarchical Context Management

Recommended order:
1. **Measure tokens.** Below threshold → do nothing (preserves prompt caching).
2. **Apply compaction.** Free and fast; in most cases file reads/search results are the main culprits, so this alone often suffices.
3. **Re-measure.** Still above → apply summarization (costs an LLM call but better than failing).

The `ContextOptimizer` class integrates all three stages as a `before_llm_callback`. It re-measures after each stage and returns early if below threshold.

**Registration example:** the optimization LLM uses a cheaper model (e.g., GPT-5 mini) while the main agent uses a stronger model (e.g., GPT-5).

## 5. Continuous Execution: Session and State Management (Pattern 2)

### 5.1 The Session Class

A session is a container for everything that must persist across multiple `run()` calls. It serves two purposes: storing conversation history so previous conversations can be continued, and storing arbitrary execution state so interrupted tasks can be resumed.

**Relationship with ExecutionContext:**
- `ExecutionContext` holds everything during a *single* `run()`; it is discarded when execution ends.
- A `Session` persists across *multiple* `run()` calls — it has a longer lifecycle.
- When each `run()` completes, necessary information is saved to the session; when the next `run()` begins, information is restored.

**`Session` fields:**
- `session_id` — uniquely identifies the session
- `user_id` — optional user identifier
- `events` — all events accumulated across `run()` calls
- `state` — arbitrary execution state (`dict[str, Any]`)
- `created_at` / `updated_at` — timestamps

The versatility of `state` is notable: it can hold a pending tool call awaiting approval, or the progress of a long-running task, without modifying the `Session` class itself.

### 5.2 Managing Sessions with SessionManager

The `Session` is just a data container; a separate manager creates, retrieves, and saves sessions.

**Core responsibilities:**
- **Create** — make a new session
- **Get** — find an existing session by `session_id`
- **Get or create** — return existing or create new
- **Save** — persist changes back to storage

**Design:** `BaseSessionManager` is an abstract class with async methods (`create`, `get`, `save`). `get_or_create` is a regular method built on `get` and `create`. Declaring methods async in the base class lets you swap in database-backed implementations without changing calling code.

**`InMemorySessionManager`** stores sessions in a Python dict — suitable for development and testing; all sessions are lost when the program terminates. `create` raises `ValueError` if the `session_id` already exists.

### 5.3 Integrating Sessions into the Agent

**Extending ExecutionContext:** Add `session` and `session_manager` fields. Backward compatibility is preserved with properties: `events` and `state` return `session.events` and `session.state` when a session exists, so existing code works unchanged. `add_event` also updates the session's `updated_at`.

**Modifying Agent initialization:** Accept a `SessionManager`; default to `InMemorySessionManager`.

**Modifying `run()`:** Three additions:
1. Accept a `session_id` parameter.
2. Load or create the session at the start (`get_or_create`).
3. Save the session when execution completes.

### 5.4 Multiturn Conversation Example

- **Without `session_id`** (or with different IDs each call): each `run()` is independent; the agent has no memory of the first conversation.
- **With the same `session_id`:** both calls share the same `Session`, so the agent correctly recalls information.

**Internal flow with the same `session_id`:**
1. First `run()` → `get_or_create("alice_session")` creates a new session.
2. Conversation is stored in `session.events`; `save()` persists it.
3. Second `run()` → `get_or_create` returns the existing session.
4. `ExecutionContext` references this session, so `context.events` includes the previous conversation.
5. The LLM generates its response with full access to previous context.

In practice: use sessions for conversational applications; for one-off tasks (like GAIA problems), independent `run()` calls are appropriate.

### 5.5 Asynchronous Human-in-the-Loop (HITL)

This demonstrates that sessions provide value far beyond chat history: they enable pausing and resuming execution at arbitrary points.

**Data structures:**

- **`PendingToolCall`** — a tool call awaiting approval. Fields: `tool_call` (the original `ToolCall`) and `confirmation_message` (what to show the user).
- **`ToolConfirmation`** — the user's decision. Fields: `tool_call_id`, `reason` (optional), `approved` (bool), `modified_arguments` (optional, lets the user change arguments).
- **`AgentResult`** extended with:
  - `status`: `"complete"` | `"pending_confirmation"` | `"error"`
  - `pending_tool_calls`: list of `PendingToolCall`

**Serializing state with Pydantic:** Python dicts can't store Pydantic objects directly. Use `model_dump()` to convert to dict for storage and `model_validate()` to restore:
```python
context.state["pending_tool_calls"] = [p.model_dump() for p in pending_calls]
# later...
pending_calls = [PendingToolCall.model_validate(d) for d in raw_pending]
```
This pattern works for many kinds of execution state: long-running task progress, checkpoints, intermediate results.

### 5.6 Extending Tools for Confirmation

Add two attributes to `BaseTool`:
- `requires_confirmation: bool = False`
- `confirmation_message_template` — uses `{name}` and `{arguments}` placeholders

`get_confirmation_message(arguments)` fills the template. The `@tool` decorator is extended to accept `requires_confirmation` and `confirmation_message` parameters.

Example: `delete_file` is configured with `requires_confirmation=True` and a custom message that includes `{arguments[filename]}`; `get_weather` is read-only and requires no approval.

### 5.7 Implementing Pause and Resume

**Internal features vs. external extensions:** Callbacks are *external* extension points injected by users. Tool confirmation is a *core feature* the framework provides. Therefore, confirmation handling is implemented inside `_prepare_llm_request()` rather than as a callback — this ensures confirmation results are converted to `ToolResult`s and appear naturally in the conversation flow. There is also a timing reason: callbacks receive an already-constructed `LlmRequest`, so newly added events wouldn't appear in `contents` if the logic lived in a callback.

**Two core design principles:**
- **Pre-execution blocking** — a tool requiring approval must receive approval *before* execution. Executing first and asking afterward is meaningless for irreversible actions (file deletion, payments).
- **Whitelist approval** — users can approve only some of the pending tool calls. Only tools in the approval list are executed; others are rejected automatically.

**Modifying `act()`:** For each tool call, check `requires_confirmation`. If true, create a `PendingToolCall` and add it to `pending_calls`, then `continue` *without* creating a `ToolResult`. Tools not requiring confirmation execute normally. Pending calls are serialized into `context.state["pending_tool_calls"]`.

**Modifying `run()`:**
- Accept an optional `tool_confirmations: List[ToolConfirmation]` parameter; when resuming, register it in `context.state`.
- After each `step()`, check `context.state.get("pending_tool_calls")`. If present, return immediately with `status="pending_confirmation"`.
- When resuming, `user_input` is optional — only `tool_confirmations` is required.

**Modifying `_prepare_llm_request()`** (now async):
- If both `pending_tool_calls` and `tool_confirmations` are in state, call `_process_confirmations()`, add results as an event, and clear the processed state.
- *Then* flatten events into contents — this ordering ensures the LLM sees approval/rejection results in the next step.

**`_process_confirmations()` logic:**
1. Restore pending calls from state.
2. Build a lookup of confirmations by `tool_call_id`.
3. For each pending call:
   - If approved: merge original arguments with `modified_arguments` (`{**original, **modified}`), execute the tool, append a success `ToolResult`.
   - Else (explicit rejection or absent from the approval list): append an error `ToolResult` with the rejection reason.
4. Return all results.

**Key points:**
- **Whitelist approval** — only approved=True entries execute.
- **Argument merging** — modified arguments overwrite original ones, allowing selective changes (e.g., updating only the file path).
- **Clear feedback** — rejected tools also produce `ToolResult`s, because the LLM expects results for tool calls and can then adapt (suggest alternatives, ask for clarification).

**Complete flow summary:**

*First `run()`:*
1. Add `user_input`.
2. `step()` → `think()` → LLM returns tool calls.
3. `step()` → `act()` → discovers `requires_confirmation` tool → stores in `context.state['pending_tool_calls']` → no `ToolResult` created.
4. `run()` loop detects pending.
5. Returns `AgentResult(status='pending_confirmation')`.

*Second `run()` after user approval:*
1. Register `tool_confirmations` in `context.state`.
2. `step()` → `_prepare_llm_request()` → calls `_process_confirmations()` → executes approved tools, error messages for rejected → adds `ToolResult` as event → clears pending/confirmations state.
3. LLM sees results and decides the next action.
4. Returns `AgentResult(status='complete')` on final response.

### 5.8 Complete HITL Example

**Setup:** Define a `delete_file` tool with `requires_confirmation=True` and a message like "Delete '{arguments[filename]}'? This cannot be undone." Create an agent with session support (`InMemorySessionManager`).

**Step 1 — Initial request:**
```python
result = await agent.run("Please delete the file named temp.txt", session_id="file_session")
# result.status == "pending_confirmation"
# result.pending_tool_calls[0].tool_call.name == "delete_file"
# result.pending_tool_calls[0].tool_call.arguments == {'filename': 'temp.txt'}
```
The agent immediately returns with `status="pending_confirmation"`.

**Step 2 — User approval:**
```python
confirmation = ToolConfirmation(
    tool_call_id=result.pending_tool_calls[0].tool_call.tool_call_id,
    approved=True
)
result = await agent.run(session_id="file_session", tool_confirmations=[confirmation])
# result.status == "complete"
# result.output == "I've deleted the file temp.txt as requested."
```
The agent resumes, executes the approved tool, and generates a final response.

## 6. Long-Term Memory: Accumulating Knowledge Across Sessions (Pattern 3)

Session and SessionManager allow conversations to continue within the same `session_id`. When a session ends, its information is lost. Long-term memory accumulates and uses knowledge *beyond* session boundaries.

This is a prime example of combining context engineering's **write** (storage) and **retrieve** (search) strategies.

### 6.1 Structure of Long-Term Memory

**SessionManager vs. MemoryManager:**

| | SessionManager | MemoryManager |
|---|---|---|
| Problem solved | Continuity *within* a session | Knowledge *across* sessions |
| Mechanism | Multiple `run()` calls with the same `session_id` | Storing important info after a session ends, retrieving in future sessions |
| Analogy | Horizontal continuity (conversation flow) | Vertical continuity (knowledge accumulation) |

**Two core operations:** storage (Write) and retrieval.

**Approach:** The simplest method is converting text directly to vector embeddings and finding similar vectors at retrieval. But storing entire conversations introduces noise and important information can get buried.

Recent memory libraries (Mem0, Graphiti) use LLMs to extract only information worth preserving. Four operations manage conflicts with existing memories:
- **ADD** — store new information
- **UPDATE** — modify existing memories
- **DELETE** — remove contradictory memories
- **SKIP** — ignore information already stored

In this chapter, only **ADD and SKIP** are implemented. Rationale:
- User profile information (name, occupation, preferences) can change over time and would require UPDATE/DELETE.
- Problem-solving records are different: different approaches or results for the same problem are worth storing as *separate* records (ADD); identical attempts can be ignored (SKIP).
- Past failure records don't need modification/deletion — keeping them helps avoid repeating mistakes.
- This immutability of memory simplifies implementation.

### 6.2 Information Extraction: Structured Output

**Extending LlmClient:** Add an `ask()` method that takes a prompt and returns text or JSON. When `response_format` is provided, it returns output in the requested format:
```python
answer = await llm_client.ask("What is 2 + 2?")
data = await llm_client.ask(prompt, response_format=GaiaOutput)
```

**TaskMemory schema** (Pydantic):
- `task_summary` — what the problem asked
- `approach` — methods and tools used
- `final_answer` — the agent's submitted answer
- `is_correct` — whether it was right
- `error_analysis` — why the attempt failed (optional)

`to_embedding_text()` returns only the `task_summary` for vector search, so similar *problems* can be found during retrieval. The approach and answer are stored as metadata and used after retrieval.

**Extraction prompt:** `TASK_MEMORY_EXTRACTION_PROMPT` instructs the LLM to analyze the complete execution trace (reasoning, tool calls, intermediate results, correctness feedback) and produce a structured `TaskMemory`.

**Duplicate check:** `DUPLICATE_CHECK_PROMPT` asks the LLM to compare a new memory against existing ones and respond with:
- **ADD** — new information
- **SKIP** — similar information already exists

Judgment criteria:
- Same problem with a *different approach or different result* = new information
- Same problem with the *same approach and same result* = duplicate

`DuplicateCheckResult` schema captures both the decision and the reasoning.

### 6.3 Building a Vector Store with ChromaDB

Install: `uv add chromadb`. ChromaDB is lightweight, runs locally, and needs no separate server — ideal for prototyping and learning.

**Core concept: collection** — groups related documents, similar to a table in a relational database. Each collection stores document text, embedding vectors, and metadata together.

**Basic use:**
```python
client = chromadb.Client()
embedding_fn = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
collection = client.get_or_create_collection(name="my_memories", embedding_function=embedding_fn)
collection.add(ids=[...], documents=[...], metadatas=[...])
results = collection.query(query_texts=["Where is the cat resting"], n_results=1)
```
When an `embedding_function` is specified, document text is embedded automatically on `add()`, and `query_texts` are embedded automatically on `query()`. Embedding generation and cosine similarity are handled internally.

**Metadata:** stores supplementary information for each document (category, creation time, source). In the memory store, fields like `approach` and `is_correct` are stored as metadata.

### 6.4 Implementing TaskMemoryManager

Stores and retrieves GAIA problem-solving records using ChromaDB. The pipeline: extract structured memory from execution history → check for duplicates → store in ChromaDB.

**Methods:**

| Method | Purpose |
|---|---|
| `__init__` | Wires up the LLM client and ChromaDB collection |
| `_format_execution_history` | Flattens the event list into a text block the LLM can read |
| `_extract_memory` | Asks the LLM to produce a structured `TaskMemory` |
| `_is_duplicate` | Asks the LLM whether a new memory duplicates retrieved neighbors |
| `save` | Orchestrates formatting, extraction, duplicate checking, and storage |
| `search` | Embeds a query and returns top-k `TaskMemory` neighbors |

**`__init__`:** Receives an `LlmClient` via injection (used for extraction and duplicate checking). Sets up `chromadb.Client()` and a collection with `OpenAIEmbeddingFunction(model_name="text-embedding-3-small")`.

**`_format_execution_history`:** Iterates through each event's content and formats `Message`s, `ToolCall`s, and `ToolResult`s separately (tool results truncated to 500 chars).

**`_extract_memory`:** Passes `response_format=TaskMemory` so the LLM generates matching JSON; `LlmClient` parses it and returns a `TaskMemory` instance — no separate JSON parsing needed.

**`_is_duplicate`:** If no existing search results → `False`. Converts existing memories to text, includes them and the new memory in `DUPLICATE_CHECK_PROMPT`, and returns `True` if `result.decision == "SKIP"`. On exception, returns `False` (proceed with storage).

**`save`:**
1. Convert execution history to text.
2. Extract structured memory via the LLM.
3. Search for similar existing memories (using the same text used for storage) — `n_results=3`.
4. If duplicate → return `None`; otherwise store in ChromaDB with a UUID and `model_dump()` metadata (with `None` values converted to `""`).

**`search`:** `collection.query()` handles embedding, similarity, and sorting; converts result metadata to `TaskMemory` objects for type-safe access.

### 6.5 Integrating MemoryManager into the Agent

Add `memory_manager: TaskMemoryManager | None = None` to the `Agent` constructor. In `run()`, pass it to `ExecutionContext` so tools can access it via `context.memory_manager`.

### 6.6 Retrieving Memories

**Tool-based retrieval:** `MemoryTool` (name: `recall_memory`) searches the memory store and formats results (problem, approach, answer, correctness, error analysis). The agent decides when to call it.

**Automatic injection with `process_llm_request`:** Some memories should always be retrieved before the LLM sees a request (e.g., user profile, checking for similar past problems).

Instead of a separate callback, override `process_llm_request()` in `BaseTool`. This method runs before each LLM request and can modify the request directly. It *reuses* `execute()` logic rather than duplicating search/formatting code.

**Visibility control via `tool_definition`:**
- If `tool_definition` is a dict → the LLM can call the tool explicitly.
- If `tool_definition is None` → the tool doesn't appear in the LLM's tool list, but `process_llm_request()` still runs before every LLM call.

`MemoryTool` sets `tool_definition=None` for automatic injection. `process_llm_request` finds the latest user message, calls `execute()`, and appends the result to instructions wrapped in `<PAST_EXPERIENCES>` tags with guidance to reference successful approaches and avoid failed ones.

**Integrating into the Agent:** Modify `_prepare_llm_request()` (now async):
1. Filter out tools with `tool_definition is None` (automatic-injection-only tools) from the LLM's tool list.
2. Call each tool's `process_llm_request()` method on the constructed request.

## 7. Key Takeaways

- **Memory transforms an agent** from a stateless tool into an intelligent assistant that learns and adapts across interactions.
- **Three use patterns** address different challenges: Pattern 1 (in-run context management), Pattern 2 (continuous execution across `run()` calls), Pattern 3 (long-term memory across sessions).
- **Context management is hierarchical:** measure tokens → apply compaction (deterministic, cheap, reversible) → use summarization as a last resort (semantic, costly, irreversible). Keep context stable below a threshold to preserve prompt caching.
- **Separate storage from presentation:** `ExecutionContext` remains immutable; `LlmRequest` is optimized per call. This is where context engineering happens.
- **Session and SessionManager** enable multiturn conversations by persisting events and state across `run()` calls. The same architecture supports **asynchronous HITL** workflows — execution pauses for approval and resumes when confirmation arrives, with no blocking `input()` call.
- **Tool confirmation is a core framework feature**, not a user callback. It belongs in `_prepare_llm_request()` for correct event ordering. Two principles: pre-execution blocking and whitelist approval. Pydantic's `model_dump()`/`model_validate()` allow storing complex objects in session state.
- **Long-term memory** combines write and retrieve strategies. Structured information is extracted from execution history, checked for duplicates, stored in a vector database (ChromaDB), and retrieved when relevant.
- **Memory retrieval** can be an explicit tool (`recall_memory`) or automatic injection via `process_llm_request()` with `tool_definition=None`.
- **The ADD/SKIP-only policy** works for problem-solving records because different attempts at the same problem are valuable as separate records, and past failures help avoid repeating mistakes.

---

# Chapter 6 — Teaching Outline and Learning Roadmap

### Teaching Instructions

Teach this chapter step by step, following the outline below. Each lesson should include:
- A clear explanation of the concept.
- A practical example, preferably in Python.
- A discussion of why the concept matters when building AI agents.
- A small exercise or coding task.
- A connection to the existing GitHub project when relevant.

Do not skip topics or assume prior knowledge of concepts that have not yet been explained. Clearly distinguish between concepts covered in the book and additional implementation ideas for our project.

### Lesson 1: Why Agents Need Memory
- The three core problems: space, time, persistent knowledge.
- Short-term vs. long-term memory.
- Session as a unit of continuity.
- The difference between conversation continuity and persistent knowledge.
- When memory is unnecessary (one-off tasks).

### Lesson 2: Context Engineering and Memory
- The five context-engineering strategies (generate, retrieve, write, reduce, isolate).
- Mapping the three memory patterns to strategies.
- Why Chapter 6 is a "hub" where reduce, write, and retrieve converge.
- The distinction between in-run context management and cross-session memory.

### Lesson 3: Separating Storage from Presentation
- Why `ExecutionContext.events` must remain immutable.
- The role of `LlmRequest.contents` as a curated projection.
- The `before_llm_callback` mechanism.
- The `create_optimizer_callback` factory and threshold-based activation.
- Why original history is essential for debugging.

### Lesson 4: Sliding-Window Strategy
- Basic implementation: preserve the user message, keep the most recent *n* items.
- Why the user's original question must be preserved.
- Limitations: message count ≠ context size; small *n* loses context; large *n* doesn't solve the problem.

### Lesson 5: Token Counting
- Why the real limit is tokens, not messages.
- Using `tiktoken` (`encoding_for_model`, fallback to `o200k_base`).
- Counting per-message overhead, content, tool calls, and tool definitions.
- Why optimization matters even below the limit: cost, latency, lost-in-the-middle, response space, context rot.
- The book's small threshold (~20,000 tokens) for observability.

### Lesson 6: Compaction Strategy
- Core idea: replace processed data with lightweight references.
- ToolCall compression (write operations) vs. ToolResult compression (read operations).
- Implementation with a single loop and rule tables (`TOOLCALL_COMPACTION_RULES`, `TOOLRESULT_COMPACTION_RULES`).
- The refetch trade-off: API fees, latency, different results.
- Prompt caching and threshold-based optimization.

### Lesson 7: Summarization Strategy
- When compaction is insufficient.
- Selecting targets: oldest intermediate messages, preserving the user question.
- The structured summary prompt (key findings, tools used, current status).
- Implementation: `last_summary_idx` tracking to avoid re-summarization.
- Compaction vs. summarization: deterministic vs. semantic, reversible vs. irreversible.
- Why compaction should be tried first.

### Lesson 8: Hierarchical Context Management
- The recommended order: measure → compaction → re-measure → summarization.
- Why "do nothing below threshold" preserves caching.
- The `ContextOptimizer` class as a `before_llm_callback`.
- Using a cheaper model for optimization and a stronger model for the main agent.
- Practical exercise: implement and test `ContextOptimizer` with different thresholds.

### Lesson 9: The Session Class
- Definition and purpose: persistence across `run()` calls.
- `Session` fields: `session_id`, `user_id`, `events`, `state`, `created_at`, `updated_at`.
- The lifecycle relationship with `ExecutionContext`.
- The versatility of `state: dict[str, Any]`.
- The difference between horizontal (within-session) and vertical (cross-session) continuity.

### Lesson 10: SessionManager
- Core responsibilities: create, get, get-or-create, save.
- The abstract base class and why methods are async.
- `InMemorySessionManager` for development and testing.
- `ValueError` on duplicate `session_id` in `create`.
- How to swap in a database-backed implementation without changing calling code.

### Lesson 11: Integrating Sessions into the Agent
- Extending `ExecutionContext` with `session` and `session_manager`.
- Backward compatibility via properties (`events`, `state`).
- Updating `updated_at` in `add_event`.
- Modifying `Agent.__init__` and `run()`.
- The internal flow of a multiturn conversation with the same `session_id`.
- Why one-off tasks don't need sessions.

### Lesson 12: Data Structures for HITL
- `PendingToolCall` (`tool_call`, `confirmation_message`).
- `ToolConfirmation` (`tool_call_id`, `reason`, `approved`, `modified_arguments`).
- Extending `AgentResult` with `status` and `pending_tool_calls`.
- Serializing state with Pydantic: `model_dump()` and `model_validate()`.
- Why this pattern generalizes to long-running tasks and checkpoints.

### Lesson 13: Extending Tools for Confirmation
- Adding `requires_confirmation` and `confirmation_message_template` to `BaseTool`.
- `get_confirmation_message()` and placeholder filling (`{name}`, `{arguments}`).
- Extending the `@tool` decorator.
- Example: `delete_file` (requires confirmation) vs. `get_weather` (read-only).
- Custom messages with specific argument values.

### Lesson 14: Implementing Pause and Resume
- Internal features vs. external extensions — why confirmation lives in `_prepare_llm_request()`.
- The timing problem with callbacks (new events wouldn't appear in `contents`).
- Two design principles: pre-execution blocking and whitelist approval.
- Modifying `act()`: create `PendingToolCall`, skip `ToolResult`, serialize to state.
- Modifying `run()`: accept `tool_confirmations`, check for pending after each step, return `pending_confirmation`.
- Modifying `_prepare_llm_request()`: process confirmations *before* flattening events.
- `_process_confirmations()`: restore, look up, merge arguments, execute or reject, return results.
- Argument merging with `{**original, **modified}`.
- Why rejected tools still produce `ToolResult`s.
- The complete two-`run()` flow.
- A practical HITL exercise: file deletion with approval.

### Lesson 15: Long-Term Memory — Structure and Policy
- SessionManager vs. MemoryManager: horizontal vs. vertical continuity.
- Two core operations: write and retrieve.
- Why storing entire