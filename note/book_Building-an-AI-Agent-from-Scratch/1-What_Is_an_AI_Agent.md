# Chapter 1 Summary — What Is an AI Agent?

**Note:** This document is based on Chapter 1 of *Build an AI Agent (From Scratch)* and follows the same structure as the Chapter 6, Chapter 8, Chapter 9, and Chapter 10 summaries.

## 1. Introduction: The Age of AI Agents

AI agents are rapidly transforming how we interact with technology. Before diving into how to build them, it is important to survey the landscape of AI agents to understand what we are working toward:

- **Personal AI agents:** Services such as ChatGPT, Claude, and Gemini started as conversational chatbots but have evolved dramatically. Modern personal agents can analyze uploaded documents, search the web, generate images, write and execute code, and help with shopping decisions.
- **Customer-facing agents:** These agents operate on behalf of businesses, interacting directly with customers in real time. They can access company policies, retrieve customer data, process transactions, and make decisions based on business rules.
- **Specialized agents:** These agents tackle domain-specific tasks that require deep expertise or extended processing time. Coding agents like Claude Code, Cursor, and Codex CLI can navigate codebases, implement features, fix bugs, and refactor code. Deep Research agents can conduct comprehensive investigations across hundreds of sources.

Despite their different applications, these agents share a common foundation: they are powered by LLMs. In this book, the terms **AI agent** and **LLM agent** are used interchangeably because virtually all modern AI agents are powered by LLMs.

## 2. Understanding LLM Agents

### 2.1 What Is an LLM?

An LLM is a language model trained on nearly all publicly available text from the internet. The core principle is simple: learning to predict the next word by processing massive amounts of text data. Through large-scale training, LLMs develop an understanding of language structure, context, and meaning. This ability to apply learned patterns to new examples is a form of **generalization**.

Recently, **reasoning models** have emerged that analyze tasks or formulate plans before producing results. The reasoning model does not answer immediately; first, it generates thinking tokens. This advance enables LLM agents to tackle complex and unfamiliar problems effectively.

### 2.2 What Is an LLM Agent?

A representative example of an LLM agent is a research agent. When a user asks, "Summarize the 2024 Nobel Physics Prize winners' research," the agent gathers information from multiple sources, analyzes findings, and produces a comprehensive report.

What is worth noticing is **who decided each step**: which source to check first, whether the search results were sufficient, what to look for next, and when to start writing the report. These decisions were not hardcoded; the LLM made them on the fly, based on the context at each moment.

**Core definition: Autonomous control flow.** An LLM agent is a program that hands part of its control flow to an LLM, letting the model decide what actions to take and when to stop based on its current context and goals.

**Elements that realize autonomy: LLM + Tool + Loop:**
- **LLM:** The agent's brain. It understands the current situation and decides what to do next.
- **Tools:** Extend the agent's action space into the external world (web search, code execution, database access).
- **Loop:** The execution structure that lets these decisions happen repeatedly until the goal is achieved.

**How the agent loop works:**
1. The LLM evaluates the current context and determines whether a tool is needed.
2. The selected tool is executed.
3. The tool's results are added back to the LLM's context.
4. The LLM decides whether to continue or stop.

## 3. Workflow vs. Agent

When integrating LLMs into applications, developers face a fundamental architectural choice: how much control of the execution flow the LLM should have. There are seven distinct levels of agency, progressing from left to right with increasing autonomy.

These seven levels can be broadly categorized into two architectural approaches: **workflows** and **agents**.

### 3.1 Workflow: Developer-Defined Flow

A workflow is a system in which developers explicitly design the sequence of operations, with LLMs executing specific steps within that predefined structure. The key characteristic is **predictability**: given the same input, the system follows the same path.

| Pattern | Description |
| :--- | :--- |
| **LLM call** | A single LLM call where one prompt goes in and one response comes out. |
| **Chain** | Multiple LLM calls in a predefined sequence, with the output of one step becoming the input for the next. |
| **Router** | Conditional logic that the LLM uses to decide which predefined path to take next. |

### 3.2 Agent: LLM-Directed Flow

An agent is a system in which the LLM dynamically directs its own processes and tool use, maintaining control of how it accomplishes tasks. Rather than follow a predetermined path, the agent decides what to do next based on the current context and its progress toward the goal.

| Pattern | Description |
| :--- | :--- |
| **Tool use with a multistep loop** | The ability to use external tools and the autonomy to continue operating until the task is complete. |
| **Tool creation** | The agent creates new tools by generating code to implement functionality that does not exist in the predefined toolkit. |

### 3.3 Combining Workflows and Agents in Practice

The distinction between workflows and agents is not always binary in production systems. The most effective architectures often combine both approaches, using workflows to provide structure and predictability, and embedding agents at specific points where flexibility is needed.

**Why combine?**
- **Contains complexity:** Limiting where agents operate makes the overall system easier to reason about and debug.
- **Optimizes costs:** Agent loops can be expensive due to multiple LLM calls, so using them only where flexibility is genuinely needed keeps operational costs manageable.
- **Enables graceful degradation:** If an agent component fails, the workflow structure allows for fallback behaviors or human escalation.

## 4. Tasks That Require Agents

Before jumping into agent development, two key decisions shape whether an agent is the appropriate solution:

1. **Does this task require an LLM?**
2. **If an LLM is needed, does it require an agent, or will a workflow suffice?**

### 4.1 Tasks That Require an LLM

You use two main criteria to make this decision:
- **Tasks involving unstructured data:** If a task requires analyzing text, images, or audio, it is a strong candidate for LLM use.
- **Input diversity:** LLMs shine when inputs are unpredictable and varied, requiring flexible interpretation that cannot be anticipated.

### 4.2 Conditions for Using Agents

Agents come with inherent tradeoffs:
- **Costs are higher:** Multiple LLM calls multiply API expenses.
- **Latency increases:** Response time accumulates with each reasoning step.
- **Errors propagate:** Mistakes in early steps cascade through the entire process.

When should you use an agent? You can decide based on three criteria:
- **Task complexity:** If it is difficult to predict how many steps a task will require, agents have the advantage.
- **Task value:** The value of completing the task should outweigh the additional cost and latency.
- **Error cost and detectability:** If errors lead to critical consequences, an agent may not be the best choice.

### 4.3 GAIA: An Agent Gym

GAIA is a benchmark that systematically collects tasks requiring agents. It consists of question-and-answer pairs, each requiring multistep reasoning, web searches, calculations, and more. The questions are difficult to solve with a single LLM call, and it is not easy to predefine a clear workflow.

**Why GAIA is used in this book:**
- **Clear answers enable fast feedback:** When answers are clear, you can verify immediately whether you are heading in the right direction.
- **It is ideal for practicing the agent development cycle:** Determine whether an agent would help, develop a prototype, observe failures, analyze the cause, and improve.
- **No domain knowledge is required:** Most GAIA problems center on web search and information synthesis.

## 5. Context Engineering

**Prompt vs. context:**
- A **prompt** is the input text that a user sends to an LLM. It can be divided into system prompts (how the model should respond) and user prompts (the user's actual requests).
- **Context** is a broader concept that encompasses all the information an LLM references when generating a response: system prompt, user messages, conversation history, tool execution results, and retrieved documents.

Think of context as the LLM's working memory. An LLM predicts the next token based solely on the information within its context window.

### 5.1 Why Agents Fail

Agent failures can be attributed to two main causes:
- **Insufficient model intelligence:** The model fails to solve complex mathematical problems or makes errors in logical reasoning.
- **Lack of necessary information:** Even if a model is intelligent enough, it cannot produce correct answers if the information needed to complete the task is not in the context.

A significant portion of real-world agent failures stems from information deficiency. An LLM can be only as smart as the context it is given.

### 5.2 From Prompt Engineering to Context Engineering

In the early days of LLM adoption, prompt engineering was the primary focus. But as agents began using tools, context began to change dynamically. This shift raised a new question: "How do we maintain only the essential information needed for task completion in the context?"

**Context engineering** is the discipline of providing the information an LLM needs to perform its tasks at the right time and in the right form.

### 5.3 Bigger Context Isn't Always Better

Recent research has reported that model performance degrades as context length increases. This phenomenon is sometimes called **context rot**. The **"lost in the middle" effect** is particularly well documented: models tend to miss important information located in the middle of long contexts.

The key insight: to maximize model performance, we should not simply provide more information but selectively provide only highly relevant information.

### 5.4 Five Context-Engineering Strategies

| Strategy | Description | Related chapters |
| :--- | :--- | :--- |
| **Generate** | Constructing context with LLM-generated outputs (plans, reflections). | Chapter 7 |
| **Retrieve** | Bringing information from external sources into context (web search, RAG, memory). | Chapters 3, 5, 6 |
| **Write** | Saving from context to external storage (long-term memory, workspace). | Chapters 6, 8 |
| **Reduce** | Compressing and deleting context (summarization, compaction). | Chapter 6 |
| **Isolate** | Separating tasks/tools into distinct environments (sandbox, multi-agent). | Chapters 8, 9 |

### 5.5 From Context Engineering to Harness Engineering

As the field matured, attention widened one step outward. Everything that supports an agent from outside the model is called the **harness**, and designing it is **harness engineering**.

The harness is more than just components such as memory and tools. The user's working environment in which the agent runs—the files, projects, and systems it can reach—is part of the harness too.

**In one line:** Agent = Model + Harness.

Context engineering is part of this harness as well. A new term does not replace it; it takes its place within the larger picture surrounding the model.

### 5.6 The Journey of This Book

| Chapters | Focus |
| :--- | :--- |
| **Chapters 2–4** | Building the basic agent loop (LLM, tools, agent loop). |
| **Chapters 5–9** | Enhancing the agent with context-engineering strategies (knowledge base, memory, planning and reflection, agent workspace, multi-agent system). |
| **Chapter 10** | Evaluating agents. |

## 6. Prerequisites for Reading This Book

- **Basic knowledge of Python:** Variables, data types, lists, dictionaries, control flow, functions, basic OOP.
- **Development-environment setup:** Latest version of Python (recommended: Python 3.12 or later), a code editor or IDE (recommended: Visual Studio Code), and familiarity with `pip` or `uv`.
- **API keys and external services:** OpenAI API key, Tavily API key, and a basic understanding of REST API concepts.
- **Cost considerations:** The total cost of completing all exercises should be less than $10. Tavily's free tier includes up to 1,000 searches per month.

## 7. Key Takeaways

- AI agents represent a wide spectrum, from personal assistants to customer-facing agents and specialized tools. These agents have a common foundation: LLMs are their decision-making core.
- An LLM agent consists of three elements: the LLM (brain), tools (means of interacting with the external world), and a loop (an iterative process that continues until goal completion).
- Workflows are developer-defined execution flows in which LLMs perform specific steps. Agents are LLM-directed flows in which the model determines its own process dynamically. Production systems often combine both approaches.
- Use agents when tasks require multiple unpredictable steps, provide sufficient value to justify costs, and allow error detection. The GAIA benchmark provides ideal practice problems for agent development.
- Context engineering is the discipline of providing the right information at the right time. Five strategies (generate, retrieve, write, reduce, and isolate) form the framework for building effective agents throughout this book.
- Harness engineering extends context engineering by considering everything that supports an agent from outside the model, including the user's working environment.

---

# Chapter 1 — Teaching Outline and Learning Roadmap

### Teaching Instructions

Teach this chapter step by step, following the outline below. Each lesson should include:

- A clear explanation of the concept.
- A practical example, preferably in Python.
- A discussion of why the concept matters when building AI agents.
- A small exercise or coding task.
- A connection to the existing GitHub project when relevant.

Do not skip topics or assume prior knowledge of concepts that have not yet been explained. Clearly distinguish between concepts covered in the book and additional implementation ideas for your project.

### Lesson 1: The Landscape of AI Agents

- Three categories: personal AI agents, customer-facing agents, and specialized agents.
- The common foundation: LLMs as the decision-making core.
- Examples of each category and their key capabilities.
- **Exercise:** List three AI agents you have used or heard of. For each, identify whether it is personal, customer-facing, or specialized, and describe its key capabilities.

### Lesson 2: What Is an LLM?

- The core principle: predicting the next word.
- Generalization: applying learned patterns to new examples.
- Reasoning models: generating thinking tokens before answering.
- **Exercise:** Explain in your own words how an LLM can answer a question about a topic it was not explicitly trained on.

### Lesson 3: What Is an LLM Agent?

- The research agent example: gathering information from multiple sources.
- The core definition: autonomous control flow.
- The three elements: LLM, tools, and loop.
- **Exercise:** For a customer support agent, identify the LLM's role, the tools it needs, and describe the loop it follows to resolve a refund request.

### Lesson 4: Workflow vs. Agent

- The seven levels of agency.
- Workflow patterns: LLM call, chain, router.
- Agent patterns: tool use with a multistep loop, tool creation.
- **Exercise:** For each of the following tasks, decide whether a workflow or an agent is more appropriate and explain why: (a) classifying customer emails, (b) researching a market and writing a report, (c) translating a document.

### Lesson 5: Combining Workflows and Agents

- The hybrid approach: agents as nodes in a workflow.
- Why combine: contains complexity, optimizes costs, enables graceful degradation.
- **Exercise:** Design a document-processing pipeline with four stages. Identify which stages should be workflows and which should be agents. Explain your reasoning.

### Lesson 6: Tasks That Require Agents

- Two key decisions: does the task require an LLM, and does it require an agent?
- Criteria for LLM use: unstructured data and input diversity.
- Three criteria for agent use: task complexity, task value, and error cost/detectability.
- **Exercise:** For each of the following tasks, decide whether it requires an LLM and whether it requires an agent: (a) sorting a CSV file by date, (b) summarizing a news article, (c) planning a multi-city trip with budget constraints.

### Lesson 7: The GAIA Benchmark

- What GAIA is and why it is used in this book.
- The three difficulty levels.
- Why GAIA is ideal for practicing the agent development cycle.
- **Exercise:** Look at the Kipchoge problem from the chapter. List the steps required to solve it and identify which tools are needed.

### Lesson 8: Context Engineering

- The difference between prompt and context.
- Why agents fail: insufficient model intelligence vs. lack of necessary information.
- The shift from prompt engineering to context engineering.
- **Exercise:** For a research agent that needs to answer "What are the latest trends in AI agents?", list all the information that should be in its context. Identify which information is essential and which is optional.

### Lesson 9: Bigger Context Isn't Always Better

- The context rot phenomenon.
- The "lost in the middle" effect.
- The key insight: selectively provide only highly relevant information.
- **Exercise:** Describe a scenario where adding more information to the context would degrade the agent's performance. Explain why.

### Lesson 10: Five Context-Engineering Strategies

- Generate, retrieve, write, reduce, and isolate.
- How each strategy addresses a different challenge.
- The connection between strategies and the chapters of the book.
- **Exercise:** For each of the five strategies, give one example of how it could be applied to a customer support agent.

### Lesson 11: From Context Engineering to Harness Engineering

- The definition of harness and harness engineering.
- The formula: Agent = Model + Harness.
- How context engineering fits into the larger picture.
- **Exercise:** List three components of the harness for a coding agent (e.g., Claude Code or Cursor). Explain why each component matters.

### Lesson 12: Prerequisites and Setup

- Python basics, development environment, API keys, and cost considerations.
- The tools used throughout the book: `uv`, OpenAI API, Tavily, LiteLLM, MCP, Pydantic, ChromaDB, E2B.
- **Exercise:** Set up your development environment. Clone the repository, install dependencies with `uv sync`, and configure your `.env` file with the required API keys.