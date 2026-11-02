# Chapter 7 Summary — Planning and Reflection for Complex Tasks

> **Note:** This document is based on Chapter 7 of *Build an AI Agent (From Scratch)* and follows the same structure as the Chapter 5 and Chapter 6 summaries. The "Applying to the GitHub Project" section assumes the project described in the Chapter 5 summary (Ollama + qwen2.5:3b + local MCP server). That section is a **roadmap and set of suggestions**, not a claim that anything has been implemented. Inspect the current code before making any changes.

## 1. Introduction: Giving Agents Time to Think

The reactive ReAct agent observes the current situation, decides which tool to call, observes the result, and repeats. This works well for simple tasks but struggles with complex problems requiring multiple steps. Even with memory (Chapter 6), the agent can lose direction, fail to use gathered information, or get stuck repeating failed attempts.

Memory helps the agent *remember* what happened. **Planning and reflection address these limitations by giving agents time to think:**

- **Planning** breaks complex problems into manageable tasks *before* execution.
- **Reflection** pauses to check progress and adjust direction *when things go wrong*.

Both give agents a form of **metacognition**: the ability to examine their own thinking process.

## 2. The Limitations of ReAct

ReAct is like driving without GPS: at every intersection you go in whichever direction looks best right now. Fine when the destination is close; easy to get lost on a complex route.

**Typical failure modes of ReAct agents in complex problems:**

| # | Failure Mode | Description |
|---|---|---|
| 1 | **Losing direction** | Trying to do too much at the same time; executing multiple searches and failing to use the results properly. |
| 2 | **Forgetting work in progress** | Failing to use previously found information and repeating searches. |
| 3 | **Hasty completion** | Declaring "I found the answer" with only partial information. |
| 4 | **Failure to recover** | When a tool fails, repeating the same attempt until `max_steps` instead of analyzing the cause and finding alternatives. |

**Common root cause:** The agent thinks only about *what to do right now*; it doesn't consider *how to approach the problem overall* or *whether things are progressing well*.

## 3. How Human Experts Work

A skilled researcher asked "How many hours to reach the Moon at Kipchoge's marathon pace?" does not start searching immediately. They:

1. **Decompose** the question (What's the record? What's the distance? Which distance value?).
2. **Investigate one thing at a time**, organizing along the way.
3. **Adjust** when unexpected results appear (e.g., choose perigee vs. average distance based on the question).
4. **Check for anything missing** before the final answer.

The same pattern appears in software development: experienced developers write a spec/PRD first, implement and test against it, and revisit the spec when problems arise. Coding agents like Claude Code and Cursor follow this method too. Anthropic's agent development guide emphasizes: *"Inspiration for these practices came from knowing what effective software engineers do every day."*

## 4. Why Time to Think Matters

A pattern repeatedly confirmed in LLM research: **giving models time to think improves performance** compared with requiring immediate answers.

- Chain-of-thought prompting ("Let's think step by step") significantly improves math accuracy by making the model generate intermediate reasoning steps explicitly.
- Reasoning models took this further with thinking tokens before answers.

**Planning and reflection apply this principle to agents:**
- **Planning** = time to think *before* acting ("What should I do? In what order?").
- **Reflection** = time to look back *after* acting ("What have I learned? Am I on the right track? Should I change direction?").

**From a context-engineering perspective:** Neither performs tasks directly. Neither searches, calculates, or reads files. Instead, they **generate text signals that control the flow of work**:

- Planning adds **"what to do next"** to the context.
- Reflection adds **"evaluation and direction so far"** to the context.

Among the five context-engineering strategies (Chapter 1), both belong to the **Generation** strategy — the LLM generates text within the context itself to control the workflow (unlike Retrieve, which brings in external information, or Write, which stores information externally).

## 5. Planning: Setting Direction

The core of planning is **decomposing complex problems into manageable units**. Like a to-do list: "write a report" → "gather materials → write draft → review and revise."

**Kipchoge example:**

*Original question:* "If Kipchoge could maintain his world record marathon pace indefinitely, how many thousand hours would it take him to reach the Moon?"

*If the agent starts searching immediately:* searches "Kipchoge moon hours," gets nothing useful, tries different terms, loses direction.

*Decomposed tasks:*
1. Research Kipchoge's marathon record → calculate pace.
2. Research the distance to the Moon (perigee).
3. Calculate time = distance ÷ speed.
4. Round to thousands of hours.

Each step becomes clear, and completion criteria are obvious.

### 5.1 When Planning Is Necessary

**Necessary:**
- Problems requiring multiple steps of research (e.g., Kipchoge problem).
- Problems requiring combining multiple pieces of information (e.g., comparing two companies' stock prices).

**Unnecessary:**
- Simple questions solvable with a single search (e.g., "What's the weather in Seoul today?").
- Tasks with obvious procedures (e.g., "Translate this text").

Creating a plan for "What's the weather today?" wastes tokens and time. To let agents make this distinction themselves, the planning tool's description must clearly state when to use it and when not to.

### 5.2 Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Single plan or multiple plans? | **Single plan per question** | Most cases suffice; managing multiple plans requires extra logic to decide which plan to reference. |
| Partial updates or full regeneration? | **Full regeneration each time** | Partial updates ("change task 3's status") save tokens but require precise state tracking and are bug-prone (wrong indices, lost synchronization). For 5–10 tasks, the extra token cost is minimal. |
| How to represent progress status? | **`pending`, `in_progress`, `completed`** | Makes it easy for the LLM to determine what to do next. |

### 5.3 Task Data Structure

```python
class Task(BaseModel):
    content: str
    status: Literal["pending", "in_progress", "completed"]

    def __str__(self):
        if self.status == "pending":
            return f"[ ] {self.content}"
        elif self.status == "in_progress":
            return f"[>] **{self.content}**"
        elif self.status == "completed":
            return f"[x] ~~{self.content}~~"
        return self.content
```

- Pending → empty checkbox `[ ]`
- In progress → `[>]` with bold text
- Completed → `[x]` with strikethrough

This is a common to-do-list format, so the LLM understands it easily.

### 5.4 The `create_tasks` Tool

A simple tool that receives a task list and returns it as a string, which is recorded in the context as a `ToolResult`. The **description is the key information** the LLM uses to judge when to use it:

- **WHEN TO USE:** Complex queries requiring multiple steps of research; questions that need to combine information from different sources.
- **WHEN NOT TO USE:** Simple questions answerable with a single search; tasks with obvious, straightforward procedures.
- **HOW TO USE:** Regenerate the entire task list with updated statuses; mark completed tasks as `completed`; mark the next task as `in_progress`; keep future tasks as `pending`.

**Why so simple?** LLMs already have powerful language understanding and generation. Instead of complex logic in code, the tool records the plan in the context; the LLM handles planning and state management.

### 5.5 Planning Tool Example

When the agent receives the Kipchoge question, it calls `create_tasks`:

```
Step 1: create_tasks call
Result:
[ ] Find Kipchoge marathon world record time
[ ] Calculate pace in km/h
[ ] Find Earth-Moon distance at perigee
[ ] Calculate time and convert to thousands of hours
```

Then it updates status as it progresses:

```
Step 2: search_web("Kipchoge marathon world record") call

Step 3: create_tasks call (state update)
Result:
[x] ~~Find Kipchoge marathon world record time~~
[>] **Calculate pace in km/h**
[ ] Find Earth-Moon distance at perigee
[ ] Calculate time and convert to thousands of hours
```

The LLM references the plan in subsequent steps, like a person looking at a to-do list. The checkbox format shows at a glance what's complete, in progress, and remaining.

**Granularity tip:** If tasks are too granular, `create_tasks` calls for status updates become excessive, increasing execution time. Group tasks in meaningful units. The LLM decides granularity each time, but you steer it via the tool's docstring (the `WHEN TO USE` / `WHEN NOT TO USE` sections set the threshold) and can bias toward coarser plans with hints like "Group related lookups into a single task."

### 5.6 Extension Directions

The simple tool suffices for most cases. For complex tasks, consider:

- **Completion criteria** — Explicitly define when each task is complete (e.g., "Confirm Kipchoge's marathon record time and distance" instead of "Research Kipchoge's record").
- **Hierarchical structure (task/action)** — Decompose large tasks into smaller actions.
- **Dependency management (`deps`)** — Express order relationships between tasks (e.g., "Calculate time" requires both research tasks to be complete), enabling parallel execution.

These extensions significantly increase complexity (dependency graphs, hierarchical tracking, parallel coordination). Add them incrementally only when the simple tool fails.

**The key is questions, not structure.** More important than complex dependency graphs is guiding the LLM to ask itself:
- What do I need to know to solve this problem?
- What order would be efficient?
- How will I know when each step is complete?

## 6. Reflection: Checking and Correcting

If planning is making a plan before acting, reflection is **stopping in the middle to check**. A skilled researcher naturally pauses when results differ from expectations, when synthesizing multiple sources, or before the final answer.

ReAct agents lack this checking habit. They think only about the next step and don't ask, "Is this going well so far?"

**Reflection makes agents ask:**
- What have I learned so far?
- How close am I to the original goal?
- Is the current approach effective?
- Does anything have to change?

If planning adds **"what to do next"**, reflection adds **"evaluation and direction so far"** to the context.

### 6.1 When Reflection Is Necessary

**Necessary:**

| Situation | Description |
|---|---|
| **Meaningful step completed** | Natural pause point; check progress and prepare for the next step. Most predictable checkpoint. |
| **Error in a tool** | When a search fails or an API doesn't respond, think "Why isn't this working? I should find another way." Without reflection, the agent repeats failures until `max_steps`. Particularly important. |
| **Synthesizing multiple pieces of information** | When a lot of material is gathered or information is contradictory (e.g., "average distance 384,400 km" vs. "perigee 356,500 km"). Judge which value to use. |
| **Review before the final answer** | Check "nothing is missing." Prevents hasty "I found the answer" declarations. Most important but easily overlooked. |

**Unnecessary:**
- After every single tool call (excessive overhead).
- During simple, straightforward operations.
- When everything is proceeding as expected.

**Reflection vs. Summarization (Chapter 6):**

| | Summarization | Reflection |
|---|---|---|
| Trigger | Automatic when context exceeds a threshold | LLM decides selectively when checking is necessary |
| Purpose | Reduce token count (compression) | Check direction and make corrections |

### 6.2 Implementing the Reflection Tool

As with planning, implemented simply — using the LLM's judgment rather than complex code.

**Parameters:**
- `analysis: str` — free-form text. Different content is required depending on the situation (error analysis, information synthesis, progress checking), so forcing structure would reduce flexibility.
- `need_replan: bool = False` — coordination signal with the planning tool. When the current plan is no longer valid, setting this to `True` encourages modifying the plan in the next step.

**Tool description specifies four use situations:**

1. **PROGRESS REVIEW** — After completing a meaningful step. *Example:* "Kipchoge's record found: 2:01:09. Moving to moon distance research."
2. **ERROR ANALYSIS** — When a tool fails or returns unexpected results. *Example:* "Wikipedia tool failed. Cause: service unavailable. Alternative: use web search."
3. **RESULT SYNTHESIS** — When combining information from multiple sources. *Example:* "Two different moon distances found. Problem asks for closest approach, so using perigee: 356,500km."
4. **SELF CHECK** — Before providing the final answer. *Example:* "Have all required data: marathon pace 20.81km/h, moon distance 356,500km. Ready to calculate."

**WHEN NOT TO USE:** After every single tool call (excessive overhead); during simple, straightforward operations; when everything is proceeding as expected.

**Implementation:** Receives `analysis`, returns it as a string (with a `REPLAN NEEDED` prefix if `need_replan=True`). Recorded as a `ToolResult`, referenced by the LLM in subsequent decisions.

**Summary:** If planning records "what to do next," reflection records "evaluation so far." Both are Generation strategies that add LLM-generated text to the context to influence subsequent decisions.

### 6.3 The Real Value of Reflection: Failure Recovery

Reflection's value appears **not when everything is going smoothly, but when problems arise.**

**Demonstration:** Create a Wikipedia tool that always fails (`raise RuntimeError("Wikipedia service is temporarily unavailable")`). In reality, external APIs can fail for various reasons: server overload, network issues, API changes.

**With reflection:**
1. Agent calls `get_wikipedia_page` → receives `RuntimeError`.
2. Agent calls `reflection` with an analysis like "Wikipedia tool failed. Cause: service unavailable. Alternative: use web search." (the ERROR ANALYSIS pattern).
3. The recorded analysis steers the next step toward `search_web` instead of retrying Wikipedia.

**Without reflection** (or without precise error information for the LLM to analyze): the agent likely repeats the failing call until `max_steps`.

### 6.4 Running an Agent That Uses Reflection for Research Synthesis

Reflection is useful not only for failure recovery but also for **synthesizing multiple pieces of information**.

**Example:** "Research recent developments in quantum computing." After multiple searches, the agent synthesizes results through reflection. Without reflection, it might keep adding searches or answer without organizing the collected information. Through reflection, the judgment "I've gathered enough information" is explicitly recorded in the context, guiding the next step to generate an answer.

## 7. Integrating Planning and Reflection

Planning and reflection don't operate separately — they form a **single cycle**:

1. The agent decomposes tasks and establishes order through **planning**.
2. It executes work by calling tools according to the plan.
3. After execution, it checks progress through **reflection**.
4. If things are going well → move to the next task.
5. If a problem occurred → modify the plan (replanning).

**The `need_replan` parameter is the connecting link.** When reflection determines the current plan is no longer valid, it sets `need_replan=True`. The LLM then calls `create_tasks` again in the next step to modify the plan.

**Example — Wikipedia failure in the Kipchoge problem:**

| Scenario | Analysis | `need_replan` | Reason |
|---|---|---|---|
| Wikipedia service unavailable | "Need to switch to web search" | **False** | The plan itself (research Moon distance) is still valid; only the method changes. |
| Search reveals Kipchoge holds the *half*-marathon record, not the marathon record | "Premise of the question changed" | **True** | The entire plan must be reconsidered. |

### 7.1 How Planning and Reflection Address the Four ReAct Failure Modes

| Failure Mode | Planning Solution | Reflection Solution |
|---|---|---|
| **Trying too much at the same time** | Decompose tasks; proceed one at a time. | — |
| **Premature completion declaration** | Specify completion criteria. | Self-verification (SELF CHECK). |
| **Losing direction / forgetting** | Record status in the plan. | Progress check (PROGRESS REVIEW). |
| **Unable to recover from failure** | — | Cause analysis; suggest alternatives (ERROR ANALYSIS). |

**Summary:** Planning solves mainly **"looking ahead"** problems; reflection solves **"looking back"** problems. Some problems are solved through collaboration.

A skilled researcher doesn't search immediately. They:
- Decompose the question (**planning**),
- Investigate one thing at a time while organizing along the way (**reflection**),
- Adjust their approach when unexpected results appear (**replanning**).

The integration of planning and reflection implements this natural working style in agents.

## 8. Connection to the Next Chapter

Chapter 8 explores giving agents code-execution capabilities through a sandbox environment. Code expresses procedures for performing tasks in a programming language, which inherently shares characteristics with the planning covered in this chapter. The sandbox becomes a new workspace for the agent, much as computers are for humans.

## 9. Key Takeaways

- **Planning and reflection give agents time to think.** Instead of reacting moment by moment, agents plan before acting and check after acting. This grants metacognition: the ability to examine their own process.
- **Planning decomposes complex problems** into clear, manageable units. When "1. Research Kipchoge's record, 2. Research Moon distance, 3. Calculate time" is recorded in the context, the LLM references this plan to maintain direction across multiple steps.
- **Reflection is valuable when problems arise**, not when everything goes smoothly. When tools fail or results are unexpected, reflection enables cause analysis and alternative strategies instead of repeating the same failures.
- **Planning and reflection form a complementary cycle.** Planning provides direction; reflection checks that direction and, when necessary, triggers replanning. Neither works in isolation.
- **From a context-engineering perspective, both strategies are Generation strategies.** Planning adds "what to do next" to the context; reflection adds "evaluation and direction so far." These texts influence the LLM's subsequent decisions.
- **Implementation is intentionally simple.** The tools record LLM-generated text in the context; the LLM handles the actual planning, state management, and judgment. Complex logic in code (dependency graphs, hierarchical tracking) is added only when the simple tools fail.
- **`need_replan` connects the two tools.** It distinguishes "the method failed but the plan is valid" from "the premise changed and the plan must be reconsidered."
- **Tool descriptions are critical.** The `WHEN TO USE` / `WHEN NOT TO USE` sections in the docstrings steer the LLM to call the tools at appropriate times and avoid excessive overhead.

---

# Chapter 7 — Teaching Outline and Learning Roadmap

### Teaching Instructions

Teach this chapter step by step, following the outline below. Each lesson should include:
- A clear explanation of the concept.
- A practical example, preferably in Python.
- A discussion of why the concept matters when building AI agents.
- A small exercise or coding task.
- A connection to the existing GitHub project when relevant.

Do not skip topics or assume prior knowledge of concepts that have not yet been explained. Clearly distinguish between concepts covered in the book and additional implementation ideas for our project.

### Lesson 1: Why Reactive Agents Struggle with Complex Tasks
- The ReAct loop and its adaptability.
- The "driving without GPS" analogy.
- The four failure modes: losing direction, forgetting work, hasty completion, failure to recover.
- The common root cause: thinking only about the current step.
- A practical exercise: run the current agent on a multi-step question and identify failures.

### Lesson 2: How Human Experts Work
- Decomposition, one-thing-at-a-time investigation, adjustment, final checking.
- The software-development analogy (spec/PRD → implement → test → revisit).
- Anthropic's guidance on designing agents inspired by effective engineers.
- Mapping human behaviors to planning and reflection.

### Lesson 3: Why Time to Think Matters
- Chain-of-thought prompting and reasoning models.
- Planning = time to think before acting; reflection = time to look back after acting.
- Metacognition for agents.
- The context-engineering view: both are Generation strategies.
- The distinction from Retrieve (external information) and Write (external storage).

### Lesson 4: Decomposing Complex Problems
- The Kipchoge example: what happens without decomposition vs. with it.
- Why decomposition makes completion criteria obvious.
- When planning is necessary vs. unnecessary.
- Why the tool description must state both.

### Lesson 5: Planning Design Decisions
- Single plan vs. multiple plans.
- Partial updates vs. full regeneration (token cost vs. bug risk).
- Progress status representation: `pending`, `in_progress`, `completed`.
- A practical exercise: design a `Task` data structure and its string representation.

### Lesson 6: Implementing the Planning Tool
- The simple `create_tasks` tool.
- The critical role of the docstring (`WHEN TO USE`, `WHEN NOT TO USE`, `HOW TO USE`).
- Why the implementation is intentionally simple (LLM handles the actual planning).
- Running the Kipchoge example and observing the plan in the context.
- Task granularity and how to steer it via the docstring.

### Lesson 7: Extending the Planning Tool
- Completion criteria.
- Hierarchical structures (task/action).
- Dependency management (`deps`) and parallel execution.
- The complexity trade-off.
- Guiding questions instead of complex structure.
- A practical exercise: add completion criteria to the Kipchoge plan.

### Lesson 8: When Reflection Is Necessary
- The four situations: progress review, error analysis, result synthesis, self check.
- When reflection is unnecessary (excessive overhead).
- Reflection vs. summarization: selective vs. automatic; direction-checking vs. compression.
- A practical exercise: identify reflection points in a multi-step task.

### Lesson 9: Implementing the Reflection Tool
- The `analysis: str` parameter (free-form for flexibility).
- The `need_replan: bool` coordination signal.
- The four use situations in the tool description.
- Why the implementation is simple.
- A practical exercise: write the reflection tool and its docstring.

### Lesson 10: Failure Recovery with Reflection
- The always-failing Wikipedia tool demonstration.
- How reflection analyzes the cause and suggests alternatives.
- What happens without reflection (repeating failures until `max_steps`).
- A practical exercise: build a failing tool, run the agent with and without reflection, compare traces.

### Lesson 11: Research Synthesis with Reflection
- Running multiple searches and synthesizing results.
- How reflection records "I've gathered enough information."
- Preventing unnecessary additional searches.
- A practical exercise: research a topic and observe the synthesis step.

### Lesson 12: Integrating Planning and Reflection
- The complementary cycle: plan → execute → reflect → next task / replan.
- The `need_replan` link.
- The Wikipedia example: `need_replan=False` (method changed) vs. `need_replan=True` (premise changed).
- Mapping the cycle to the four ReAct failure modes.
- A practical exercise: run the Kipchoge problem with both tools and trace the cycle.

### Lesson 13: Applying Chapter 7 to the GitHub Project
- Review the existing project architecture (`app/agent.py`, `app/llm_client.py`, `app/mcp_runtime.py`, `app/execution_context.py`, `tools/mcp_server.py`).
- Identify where planning and reflection tools would fit.
- Decide how to register the tools (regular tools vs. MCP tools).
- Consider prompt design: when should the agent plan? When should it reflect?
- Consider `max_steps` implications (planning and reflection add steps).
- Consider token budgets (plan and reflection text accumulate in the context).
- Keep the tools modular and testable.
- Do not break the existing think-act loop.
- Document all new behavior and configuration.

### Lesson 14: Testing and Evaluating the Implementation
- Test the planning tool with multi-step questions and single-step questions.
- Verify that the agent does *not* plan for simple questions.
- Test the reflection tool with failing tools.
- Verify that the agent recovers from failures instead of repeating them.
- Test the `need_replan` path: introduce a premise change and verify replanning.
- Measure token usage with and without planning/reflection.
- Compare success rates on a small benchmark (e.g., GAIA level-1 subset).
- Update the project documentation and record known limitations.

### Final Review and Completion Checklist

By the end of this chapter, the learner should be able to:

- [ ] Explain why reactive ReAct agents fail on complex tasks.
- [ ] Describe how human experts approach complex problems.
- [ ] Explain why giving agents time to think improves performance.
- [ ] Decompose a complex problem into manageable tasks.
- [ ] Decide when planning is necessary and when it is not.
- [ ] Implement a `Task` data structure with status representation.
- [ ] Implement a `create_tasks` tool with a clear docstring.
- [ ] Explain the trade-offs of partial updates vs. full regeneration.
- [ ] Identify the four situations where reflection is necessary.
- [ ] Implement a `reflection` tool with `analysis` and `need_replan`.
- [ ] Explain how reflection enables failure recovery.
- [ ] Explain how reflection supports research synthesis.
- [ ] Describe the planning-reflection cycle and the role of `need_replan`.
- [ ] Map planning and reflection to the four ReAct failure modes.
- [ ] Identify which Chapter 7 concepts exist in the GitHub project and which require new implementation.
- [ ] Propose, implement, and test planning and reflection improvements for the project.

**Teaching rule:** Complete one lesson at a time. Do not move to the next lesson until the current topic has been explained, demonstrated, and practiced. When implementation is relevant, inspect the current repository before proposing code changes. Never describe a feature as implemented unless the code confirms it.