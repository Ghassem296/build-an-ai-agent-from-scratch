# Chapter 8 Summary — Empowering Agents with Code Execution

**Note:** This document is based on Chapter 8 of *Build an AI Agent (From Scratch)* and follows the same structure as the Chapter 6 summary.

## 1. Introduction: Why Do AI Agents Need Code Execution?

So far, our agents have been built on the **predefined-tool approach**. The developer defines functions such as `search_web` and `read_file` in advance, and the LLM only selects from that list. This approach reveals clear limitations when applied to complex real-world problems:

- **Limited action space:** The agent cannot perform any operation that does not exist in the developer-defined list. If there is no `list_files` tool, the agent cannot fulfill a request like "Show me the files in the current folder."
- **Inefficient task execution:** The predefined-tool approach operates on the basic unit of "one LLM call, one tool execution, one observation." Processing 10 files requires multiple LLM calls, creating significant latency and cost.
- **Tool proliferation and context costs:** As MCP servers are added, the number of tools grows rapidly. Long tool definitions occupy a significant portion of the context window, increasing token costs and degrading the LLM's ability to select the right tool.

**The solution:** Instead of creating individual functions as tools, provide an execution environment in which the LLM can write and run code directly. This is what it means to give an agent a computer.

## 2. What Code Execution Brings

Code inherently has rich expressive power. Three core capabilities increase the possibility of solving more complex problems:

| Capability | Description | Example |
| :--- | :--- | :--- |
| **Control flow** | Selecting different paths based on conditions, repeating tasks, and handling exceptions. | A `for` loop processing 31 Wikipedia pages in a single execution. |
| **Data flow** | Storing intermediate results in variables and reusing them without recalculation. | `cleaned_data = raw_data[raw_data['amount'] > 0]` |
| **Tool composition** | Combining small functions to create new, more complex functionality. | Building `analyze_company` from `search_news` and `get_price`. |

**Empirical evidence (CodeAct):** Research published in 2024 shows that expressing actions as Python code (CodeAct) achieves a success rate improvement of more than 20% and reduces interactions by 30% compared with JSON or text formats. LLMs use variables to hold intermediate results, conditionals to branch, and loops to repeat — all of which lead to better reasoning.

## 3. Connecting Code Environments: Sandboxes

Executing LLM-generated code is risky (wrong file deletion, infinite loops, memory exhaustion). A **sandbox** is an isolated environment that provides three layers of protection:

- **Security:** Blocks access to sensitive files on the host system (e.g., `.ssh/` or `/etc/passwd`).
- **Resource management:** Limits CPU, memory, and execution time so an infinite loop does not freeze the entire system.
- **Recoverability:** If something goes wrong, the environment can easily be restored to its initial state.

**E2B platform:** This book uses E2B, which provides cloud sandboxes with a simple API. A key advantage of E2B is **state persistence**: variables and functions defined in one execution can be used directly in subsequent executions within the same sandbox (similar to running Jupyter Notebook cells in sequence).

**Connecting to the agent:**
1. Add a `code_env` field to `ExecutionContext` to store the sandbox instance.
2. Add a `code_execution="e2b"` parameter to the `Agent` constructor.
3. In `run()`, create the sandbox and clean it up in a `try...finally` block using `kill()`.
4. The `execute_python` tool is automatically registered so the LLM can execute code.

## 4. Porting Tools to Sandboxes

The agent can now write code, but it cannot directly call the tools we built in previous chapters from inside the sandbox. The solution:

- **Source code extraction:** Use `inspect.getsource()` to extract the function's source code and remove the `@tool` decorator.
- **Registration in the sandbox:** Execute the extracted source code in the sandbox so the function is defined in that environment.
- **The `sandbox_executable` flag:** Added to `FunctionTool`. Only pure functions that do not need `ExecutionContext` can be ported.

**Practical example:** A GAIA problem about counting Twitter references across 31 Wikipedia pages. Instead of 31 separate LLM calls, the LLM writes a single program with a loop that runs inside the sandbox and returns only the final count to the context. This dramatically reduces token usage.

## 5. Workspace: Filesystems and the CLI

Code execution alone is not enough. A real developer saves files, runs terminal commands, and stores intermediate results. A **workspace** combines code execution, a filesystem, and a CLI.

- **`bash_tool`:** Executes shell commands in the sandbox via `sandbox.commands.run()`. It formats output into readable stdout, stderr, and exit code.
- **`upload_file`:** Uploads local files to the sandbox via `sandbox.files.write()`.

**Practical example:** Analyzing an Excel file. The agent uploads the file, installs the necessary library with `pip install`, analyzes the data with `pandas`, and returns the answer. Unlike the approach in Chapter 5 (building a `_read_excel` tool in the local environment), the agent has full freedom here, and the host system remains secure.

## 6. Agent Skills

The problem of tool proliferation remains. If an agent has 100 tools, should detailed definitions for all 100 always be in the prompt? **Agent skills** (introduced by Anthropic in 2025) solve this through **progressive disclosure of information**.

**Folder-based structure:** Each skill is a folder containing a `SKILL.md` file (overall description) and, if needed, scripts.

**Three levels of disclosure:**

| Level | Content | Token cost | When loaded |
| :--- | :--- | :--- | :--- |
| **Level 1** | Only skill names and one-line descriptions in the system prompt. | A few hundred tokens | Always |
| **Level 2** | When the LLM determines a skill is needed, it reads that skill's `SKILL.md` with `bash_tool`. | A few thousand tokens | When needed |
| **Level 3** | Additional documents or execution of scripts inside the skill folder. | Tens of thousands of tokens | When needed |

**Comparison:** Traditional approach with 100 tools (500 tokens each) = 50,000 tokens per request. Agent skills approach: 100 skills (50 tokens each) + reading documentation for 2 skills (2,000 tokens each) = about 9,000 tokens. **More than 80% savings.**

## 7. Hierarchical Tool Structure

By the end of the chapter, tools are divided into three layers:

- **Level 1 (Core tools):** Always in the prompt (`execute_python`, `bash_tool`, `upload_file`). These are gateways to other capabilities.
- **Level 2 (Installable tools - skills):** Only summaries are in the prompt; detailed definitions live in the workspace filesystem.
- **Level 3 (Generated tools):** The LLM writes code on the spot and executes it with `execute_python`. The Python ecosystem enables virtually unlimited functionality with a single tool.

**Key insight:** The essence of this hierarchical structure is **offloading tool definitions from the prompt to a workspace**. From a context-engineering perspective, the workspace is a powerful method for removing tool information and tool-execution results that do not require LLM judgment.

## 8. Key Takeaways

- The predefined-tool approach has three limitations: limited action space, inefficient execution, and tool proliferation with context costs.
- Code execution provides three core capabilities: control flow, data flow, and tool composition.
- Sandboxes are isolated environments for safely executing LLM-generated code, providing security, resource management, and recoverability.
- Porting tools to sandboxes is done by extracting source code and registering it in the remote environment, maximizing token efficiency when processing large amounts of data.
- A workspace combines code execution, a filesystem, and a CLI, giving the agent the same working environment as a human developer.
- Agent skills optimize token use through progressive disclosure: only names and one-line descriptions are in the prompt, and details are read when needed.
- The hierarchical tool structure (core, installable, generated) offloads tool definitions from the prompt to the workspace.

---

# Chapter 8 — Teaching Outline and Learning Roadmap

### Teaching Instructions

Teach this chapter step by step, following the outline below. Each lesson should include:

- A clear explanation of the concept.
- A practical example, preferably in Python.
- A discussion of why the concept matters when building AI agents.
- A small exercise or coding task.
- A connection to the existing GitHub project when relevant.

Do not skip topics or assume prior knowledge of concepts that have not yet been explained. Clearly distinguish between concepts covered in the book and additional implementation ideas for your project.

### Lesson 1: Limitations of the Predefined-Tool Approach

- The three main limitations: limited action space, inefficient execution, tool proliferation.
- Why it is impossible for developers to anticipate every need.
- How a large number of tools affects LLM selection accuracy and token cost.
- **Exercise:** Design a hypothetical agent with 50 tools and calculate how many tokens the tool definitions occupy in the context window.

### Lesson 2: Capabilities of Code Execution

- The three capabilities: control flow, data flow, tool composition.
- Practical examples for each capability.
- CodeAct research results: higher success rates and fewer interactions.
- **Exercise:** Design a multistep task (e.g., analyzing 10 files) once with a tool-based approach and once with a code-based approach, and compare the number of LLM calls.

### Lesson 3: Sandboxes and Their Necessity

- The risks of executing LLM code: file deletion, infinite loops, memory exhaustion.
- The three layers of sandbox protection: security, resource management, recoverability.
- Introduction to E2B and the advantage of state persistence.
- **Exercise:** Consider a hypothetical dangerous code snippet (e.g., `os.remove` with a wrong path) and explain how a sandbox protects the host system.

### Lesson 4: Connecting the Sandbox to the Agent

- Adding the `code_env` field to `ExecutionContext`.
- The `code_execution` parameter in the `Agent` constructor.
- Managing the sandbox lifecycle in `run()` with `try...finally`.
- Implementing the `execute_python` tool.
- **Exercise:** Build an agent with code execution and ask it to compute the 100th Fibonacci number.

### Lesson 5: Porting Tools to Sandboxes

- Extracting source code with `inspect.getsource()` and removing the decorator.
- Registering the code in the sandbox.
- The `sandbox_executable` flag and the pure-function constraint.
- **Exercise:** Port a simple function (e.g., factorial) to the sandbox and call it from LLM-generated code.

### Lesson 6: Workspace

- The concept of a workspace: combining code execution, a filesystem, and a CLI.
- Implementing `bash_tool` and `upload_file`.
- The Excel file analysis example.
- **Exercise:** Build an agent that uploads a local CSV file, analyzes it with `pandas`, and returns the mean of a column.

### Lesson 7: Agent Skills

- The tool proliferation problem and the need for progressive disclosure.
- The folder-based structure and the `SKILL.md` file with YAML front matter.
- The three levels of information disclosure.
- Comparing token costs between the traditional approach and agent skills.
- **Exercise:** Create a simple skill (e.g., `pdf-merge`) and connect it to the agent.

### Lesson 8: Hierarchical Tool Structure

- The three layers: core tools, installable tools (skills), generated tools.
- The role of the workspace in offloading tool definitions from the prompt.
- **Exercise:** Design an agent that uses all three layers: a core tool (`execute_python`), a skill (e.g., image processing), and a generated tool (e.g., prime number calculation).