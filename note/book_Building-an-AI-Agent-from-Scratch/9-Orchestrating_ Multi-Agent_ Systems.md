# Chapter 9 Summary — Orchestrating Multi-Agent Systems

**Note:** This document is based on Chapter 9 of *Build an AI Agent (From Scratch)* and follows the same structure as the Chapter 6 and Chapter 8 summaries.

## 1. Introduction: Why Multi-Agent Systems?

So far, we have built a single agent that can perform diverse tasks by assigning roles through system prompts and providing tools. However, real-world problems are more complex. As an agent's responsibilities grow, system prompts become longer, and the number of required tools increases. Packing multiple roles (researcher, coder, writer, reviewer) into a single agent leads to several problems:

- **Prompt confusion:** A long system prompt makes it easy for the LLM to miss what is essential. When it should be writing, it might focus on review guidelines.
- **Tool overload:** With more than 10 tools to choose from, it becomes harder to judge which tool to use, and incorrect tool selections become more frequent.
- **Context overload:** The more irrelevant information fills the context window, the harder it is for the LLM to focus on what is needed.

**The solution:** Assemble a team of specialists. A multi-agent system divides responsibilities among multiple agents, each with a focused prompt and dedicated tools.

**Benefits of multi-agent systems:**
- **Separation of roles:** Each agent focuses on one role, making system prompts short and clear.
- **Specialization:** Prompts and tools can be optimized for each domain.
- **Reusability:** A specialist agent (e.g., a Researcher) can be reused across multiple systems.
- **Scalability:** New roles can be added by simply adding a new agent, without modifying existing ones.

## 2. Three Collaboration Patterns

When designing a multi-agent system, two key questions arise:

1. **Who decides the next agent?** The developer (predictable, stable) or the LLM (flexible, variable).
2. **Is the context shared?** Agents can share the same conversation history, or they can work in isolated environments and pass only results.

These two criteria distinguish three collaboration patterns:

| Pattern | Order decision | Context sharing | Best use case |
| :--- | :--- | :--- | :--- |
| **Workflow** | Developer | Developer's choice | When the workflow is predictable |
| **Agent as Tool** | LLM | Isolated | When synthesizing multiple expert results |
| **Transfer** | LLM | Shared | When domains are clearly separated |

**How to choose:**
- If the execution order is known in advance (e.g., research → write → review), use **Workflow**.
- If the orchestrator should coordinate multiple specialists and synthesize their results, use **Agent as Tool**.
- If the request clearly belongs to one specialist who should take over entirely, use **Transfer**.

## 3. Workflow: Defining Order in Code

In the Workflow pattern, the developer explicitly defines the execution order in code. The simplest approach is to call agents in a `for` loop, but this makes composition difficult. The solution is to make workflows inherit from the `Agent` class and implement the same `run()` method. This design enables **composability**: a workflow can be used anywhere a single agent is expected.

### 3.1 Sequential Workflow

Executes agents in order and passes the previous agent's context to the next. The structure is like a pipeline where data flows through one stage at a time.

**Key implementation points:**
- Only the first agent receives `user_input`; subsequent agents already have it in their context.
- Reset `final_result` and `current_step` before each agent execution.
- Extract the context from each agent's result and pass it to the next.

### 3.2 Parallel Workflow

Runs multiple agents concurrently on the same input and merges the results. This is ideal when you need multiple independent perspectives or analyses at once (e.g., optimistic, pessimistic, and realistic viewpoints).

**Challenges in merging results:**
- **Prevent duplicate user events:** The same `user_input` is passed to all agents, so duplicates must be removed.
- **Preserve existing context:** When Parallel runs inside a Sequential, events from previous stages must be kept.
- **Distinguish each agent's results:** The `event.author` field identifies which agent generated each event.

### 3.3 Loop Workflow

Runs agents repeatedly until a specific condition is met. This is appropriate when the quality of the output can be measured (e.g., a score, pass/fail criteria, or keyword presence) and iterative refinement can improve the result.

**Key implementation points:**
- The `stop_condition` function receives the current result and iteration count and returns whether to terminate.
- `max_iterations` prevents infinite loops.
- As in Sequential, only the first agent in the first iteration receives `user_input`.

### 3.4 Composing Workflows

All three workflow patterns inherit from `Agent`, so they can be nested freely. For example, a `ParallelWorkflow` can be placed inside a `SequentialWorkflow`, and a `LoopWorkflow` can be one stage of a larger pipeline. From the outside, the entire structure behaves like a single agent.

## 4. Agent as Tool: Calling Agents as Tools

The Workflow pattern excels when the execution order is known in advance. But some requests require dynamic routing: which specialists are needed depends on what the user asks.

**The core idea:** Make agents into tools. From the LLM's perspective, a tool has a name, description, and parameters. Whether it is `web_search`, a calculator, or another agent, the structure is identical. The LLM does not need to know whether the researcher is an agent or a simple function.

### 4.1 Implementing AgentTool

`AgentTool` is an adapter that wraps an `Agent` with the `BaseTool` interface. Because it is fully compatible with the existing tool system, regular tools and agent tools can be mixed.

**Two core parts:**
- `__init__` converts the agent's name and description into a tool definition.
- `execute` takes input, runs the agent, and returns only the result.

By default, `AgentTool` takes a single string parameter called `request`. For more precise control, a Pydantic model can be specified with `input_schema` to receive structured input.

### 4.2 Context Isolation

The biggest difference between Agent as Tool and Workflow is how context is shared. In Sequential Workflow, agents use the same context, and each agent can see in detail what previous agents did. In Agent as Tool, child agents run in isolated environments. The orchestrator requests "do research" but cannot know how many web searches the researcher did or which sites it visited. Only the final result returns.

**Advantages of isolation:**
- **Abstraction:** The orchestrator needs only the research result, not the internal process.
- **Context efficiency:** Even if a child agent goes through 10 tool calls, only one line of the final result is added to the parent context.
- **Independence:** The child agent's tool use or failures do not directly affect the parent agent.

**Disadvantage:** When debugging, a separate inspection is needed to see what happened inside a child agent. When an intermediate state needs to be shared, a workflow is more suitable.

## 5. Agent Transfer Pattern

In the Agent as Tool pattern, the main agent calls specialists and always maintains control. In the Transfer pattern, control itself is handed over. The previous conversation remains intact, but a different agent is responding.

**Three key differences from Agent as Tool:**
- **Location of control:** In Agent as Tool, the main agent always maintains control. In Transfer, control is handed over to another agent.
- **Method of context sharing:** In Agent as Tool, the child agent runs in isolation. In Transfer, the same context is inherited.
- **Conversation continuity:** In Agent as Tool, only the result is returned. In Transfer, the transferred agent takes over the conversation and responds directly to follow-up questions.

### 5.1 Transfer Is Also a Tool

The core idea is simple: make the agent transfer into a tool. When the agent decides that a different specialist is needed, it calls the `transfer_to_agent` tool. From the LLM's perspective, it is no different from any other tool.

**Four main components:**
1. **Agent tree composition:** Define which agent can transfer to which agent.
2. **Transfer tool creation:** Dynamically create the `transfer_to_agent` tool.
3. **Transfer detection and execution:** Hand over control when the tool is called.
4. **Context preservation:** Ensure that the previous conversation continues after the transfer.

### 5.2 Agent Tree

The agent tree defines the relationships between agents. Each agent can transfer to:
- **Child agents:** A parent agent can delegate work to its sub-agents.
- **Parent agent:** A sub-agent can return to the parent when work is complete.
- **Sibling agents:** Agents with the same parent can transfer directly to each other (optional, controlled by `disallow_transfer_to_peers`).

**Validation rules:**
- Names cannot be duplicated within the same tree.
- A single agent cannot have multiple parents.

**Common structures:**
- **Supervisory:** One router manages multiple specialists.
- **Hierarchical:** Agents delegate step by step (e.g., CEO → Manager → Specialist).
- **Network:** Agents can call each other directly.

## 6. A2A: Collaborating Across Networks

All the multi-agent systems implemented so far operate within the same process. But in practice, you may need to use an agent that another team operates, connect to a partner company's agent service, or integrate agents built with different languages or frameworks. In these cases, you cannot directly import agent objects. You need a standardized way to communicate over the network.

**A2A (Agent-to-Agent) protocol:** An open protocol introduced by Google that defines how agents built with different frameworks and languages can discover each other and collaborate over the network through standardized HTTP-based communication.

**Two core concepts:**
- **Agent card:** A JSON document where an agent introduces itself, provided at the `/.well-known/agent.json` path. It includes the agent's name, description, URL, and capabilities.
- **Task request/response:** Clients send task requests to the URL from the agent card, and servers execute the agent and return results.

**Server side:** An `AgentExecutor` adapter receives A2A-formatted requests, runs the agent, and returns results in A2A format. The server is run with a configuration that includes the agent card and the request handler.

**Client side:** A `RemoteAgent` class hides HTTP communication behind the agent interface. During initialization, it queries the agent card to set `name` and `description`. When `run()` is called, it sends task requests to the remote server. From the caller's perspective, there is no way to distinguish it from a local agent.

**Integration with existing patterns:** Because `RemoteAgent` inherits from `Agent`, it can be used with Agent as Tool and Transfer patterns. Whether local or remote, the way you work with agents is identical. This is the core value of A2A.

**Security considerations:** When deploying agents across networks, consider authentication between agents (e.g., API keys or OAuth tokens), communication encryption (HTTPS/TLS), and access control.

## 7. Key Takeaways

- Single agents face limitations as responsibilities grow. Longer system prompts and more tools make it harder for the LLM to focus, leading to confusion and incorrect tool selection.
- Multi-agent systems address this by assigning each agent a specialized role with focused prompts and dedicated tools.
- Three collaboration patterns exist, distinguished by two criteria: who decides the next agent (developer or LLM) and whether context is shared or isolated.
- The Workflow pattern has the developer define execution order in code. By making workflows inherit from `Agent`, Sequential, Parallel, and Loop patterns can be composed freely.
- The Agent as Tool pattern wraps agents with the `BaseTool` interface using `AgentTool`. The orchestrator calls specialists like functions, receiving only final results while intermediate processes remain hidden.
- The Transfer pattern lets the LLM decide which agent should handle a request by calling a `transfer_to_agent` tool. Unlike Agent as Tool, control is fully handed over, and context is shared.
- The agent tree structure defines transfer relationships through `sub_agents` and `parent` references. Each agent can transfer to children, parent, and optionally siblings.
- A2A (Agent-to-Agent) protocol enables agents to collaborate across network boundaries. Servers expose agents through agent cards and process task requests, while clients use `RemoteAgent` to call remote agents with the same interface as local ones.
- These patterns are not mutually exclusive. Real systems often combine Workflow for overall structure, Agent as Tool for specialist consultation, and Transfer for domain-based routing, with A2A enabling cross-system integration.

---

# Chapter 9 — Teaching Outline and Learning Roadmap

### Teaching Instructions

Teach this chapter step by step, following the outline below. Each lesson should include:

- A clear explanation of the concept.
- A practical example, preferably in Python.
- A discussion of why the concept matters when building AI agents.
- A small exercise or coding task.
- A connection to the existing GitHub project when relevant.

Do not skip topics or assume prior knowledge of concepts that have not yet been explained. Clearly distinguish between concepts covered in the book and additional implementation ideas for your project.

### Lesson 1: Why Multi-Agent Systems?

- The limitations of a single agent as responsibilities grow: prompt confusion, tool overload, context overload.
- The four benefits of multi-agent systems: separation of roles, specialization, reusability, scalability.
- The analogy of assembling a team of specialists.
- **Exercise:** Take a complex task (e.g., writing a technical blog post) and break it down into four specialist roles. Write a short system prompt for each.

### Lesson 2: Three Collaboration Patterns

- The two key questions: who decides the next agent, and is context shared?
- The comparison table of Workflow, Agent as Tool, and Transfer.
- How to choose the right pattern for a given use case.
- **Exercise:** For three different user requests (e.g., "Summarize this paper," "Build a data dashboard," "Fix this bug"), decide which collaboration pattern is most appropriate and explain why.

### Lesson 3: Workflow Design Principles

- Why workflows should inherit from `Agent`.
- The concept of composability: nesting workflows inside each other.
- The `run()` interface that makes workflows behave like agents.
- **Exercise:** Design a workflow that combines a `SequentialWorkflow` containing a `ParallelWorkflow` and a `LoopWorkflow`. Sketch the structure.

### Lesson 4: Sequential Workflow

- The pipeline structure: data flows through one stage at a time.
- Key implementation points: only the first agent receives `user_input`, reset state before each agent, pass context to the next.
- A practical example: researcher → writer → editor.
- **Exercise:** Implement a `SequentialWorkflow` with three agents that collaborate to write a blog post.

### Lesson 5: Parallel Workflow

- The concurrent execution model: same input distributed to multiple agents.
- Challenges in merging results: duplicate user events, preserving existing context, distinguishing agent results.
- The `asyncio.gather` approach and the `seen_user_event` flag.
- **Exercise:** Implement a `ParallelWorkflow` with three agents that analyze a business idea from optimistic, pessimistic, and realistic perspectives.

### Lesson 6: Loop Workflow

- The iterative execution model: run agents repeatedly until a condition is met.
- The `stop_condition` function and `max_iterations`.
- The `is_first_agent` flag to pass `user_input` only once.
- A practical example: writer and reviewer iterating until quality threshold is reached.
- **Exercise:** Implement a `LoopWorkflow` where a writer drafts marketing copy and a reviewer scores it until the score reaches 80 or higher.

### Lesson 7: Composing Workflows

- Nesting workflows inside each other.
- The external interface is identical to a single `Agent`.
- A practical example: research → parallel analysis → write/review loop → final editing.
- **Exercise:** Compose a workflow that combines all three patterns (Sequential, Parallel, Loop) into a single pipeline.

### Lesson 8: Agent as Tool

- The core idea: make agents into tools.
- Implementing `AgentTool` as an adapter.
- Context isolation: the child agent runs in a separate context and returns only the result.
- The advantages of isolation: abstraction, context efficiency, independence.
- **Exercise:** Implement an orchestrator agent that uses a `researcher` agent and a `coder` agent as tools.

### Lesson 9: Agent Transfer Pattern

- The difference between Agent as Tool and Transfer: location of control, context sharing, conversation continuity.
- The analogy of forwarding a phone call to another department.
- The four components: agent tree, transfer tool creation, transfer detection, context preservation.
- **Exercise:** Implement a router agent with `math_agent` and `search_agent` as sub-agents. Test the transfer flow.

### Lesson 10: Agent Tree

- The three types of transfer targets: children, parent, siblings.
- The validation rules: unique names, single parent.
- The `_get_transfer_targets` and `_find_agent` methods.
- Common structures: Supervisory, Hierarchical, Network.
- **Exercise:** Build a hierarchical agent tree with a CEO agent, two manager agents, and four specialist agents. Verify the transfer targets for each agent.

### Lesson 11: Implementing the Transfer Tool

- The `create_transfer_tool` function.
- Including agent descriptions in the tool definition.
- The `enum` constraint on the `agent_name` parameter.
- Setting `context.transfer_to` without performing the actual transfer.
- **Exercise:** Implement the transfer tool and integrate it into the `Agent` class. Test a transfer from a router to a specialist.

### Lesson 12: A2A Protocol

- The need for cross-network collaboration.
- The two core concepts: agent card and task request/response.
- The server side: `AgentExecutor` and `A2AStarletteApplication`.
- The client side: `RemoteAgent` and `httpx`.
- **Exercise:** Design an agent card for a hypothetical math expert agent. What fields would it include?

### Lesson 13: Integrating A2A with Existing Patterns

- Using `RemoteAgent` with Agent as Tool and Transfer patterns.
- The core value of A2A: local and remote agents are used identically.
- Security considerations: authentication, encryption, access control.
- **Exercise:** Build an orchestrator that uses both a local agent and a remote `RemoteAgent` as tools. Explain how the orchestrator handles them identically.