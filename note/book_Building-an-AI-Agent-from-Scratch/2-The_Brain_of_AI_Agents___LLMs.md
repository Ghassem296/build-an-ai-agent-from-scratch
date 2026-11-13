# Chapter 2 Summary — The Brain of AI Agents: LLMs

**Note:** This document is based on Chapter 2 of *Build an AI Agent (From Scratch)* and follows the same structure as the Chapter 1, 6, 8, 9, and 10 summaries.

## 1. Introduction: The LLM as the Agent's Brain

Throughout this book, we are building a research agent that can interpret requests, gather information from multiple sources, and synthesize findings into comprehensive answers. At the heart of this agent is the large language model (LLM), which serves as its decision-making brain.

Before building an effective agent, we must answer fundamental questions: Which LLM should we use, and how do we communicate with it programmatically? What can an LLM do on its own, and where does it fall short? This chapter addresses these questions head-on, starting with how to choose an LLM, then getting hands-on with APIs using LiteLLM, examining prompt engineering principles, and finally testing LLMs on the GAIA benchmark to directly experience their limitations.

## 2. Choosing LLMs for Agents

When starting agent development, the first question is: "Which LLM should I use?" This choice directly affects the agent's performance, cost, and development speed.

### 2.1 Starting with Closed LLMs

LLMs fall into two broad categories:
- **Open LLMs:** Models whose weights can be downloaded, run locally, and fine-tuned (e.g., Meta's Llama, Mistral AI's Mistral, Alibaba's Qwen, DeepSeek).
- **Closed LLMs:** Models accessible only through APIs (e.g., OpenAI's GPT, Anthropic's Claude, Google's Gemini).

This book focuses on building agents with **closed LLMs** for three reasons:

- **Rapid experimentation and validation:** You can start immediately with only an API key. No GPU servers or model infrastructure are needed, so you can focus directly on implementing agent logic.
- **Top performance:** The best-performing models today are still closed LLMs. They maintain an edge in complex reasoning and tool-use accuracy.
- **Production-ready features:** Closed LLM providers offer essential features through stable APIs: tool calling, structured output, and cost optimization (prompt caching, batch processing).

### 2.2 Expanding to Open LLMs

When you have validated your agent's design with closed LLMs, open LLMs become worth considering for:
- **Cost optimization:** When handling high-volume traffic, self-hosting can be cheaper than using APIs.
- **Stability and control:** Self-hosted open LLMs eliminate external API dependency and may be required when regulations mandate sensitive data stays in your infrastructure.

A realistic approach is to validate the agent's design with closed LLMs first, then port the same design to open LLMs and test whether performance degradation falls within acceptable bounds.

### 2.3 Essential LLM Capabilities for Agents

Regardless of which LLM you choose, certain capabilities are essential for agent development:

| Capability | Description |
| :--- | :--- |
| **Tool calling (function calling)** | The ability to appropriately select and call tools such as web search, code execution, and database queries. |
| **Structured output** | The ability to generate responses matching JSON or specific schemas for integration with real software systems. |
| **Long context support** | A sufficiently long context window to process conversation history, tool-execution results, and retrieved documents. |

**Designing for model agility:** The LLM market evolves at an astonishing pace. New models are released monthly, and performance rankings shift regularly. This is why we use **LiteLLM** throughout this book. With the same code, you can easily switch among different providers' models.

**Our choice for this book: OpenAI API.** For simple examples, we use GPT-5 mini to reduce costs, and agent implementations center on GPT-5. The OpenAI API is widely used, provides stable support for tool calling and structured output, and offers an easy-to-use Python library. Still, we design our code to avoid vendor lock-in using LiteLLM.

## 3. LLM API Basics for Building Agents

### 3.1 Setting Up the Development Environment

All code examples are available on GitHub. The book uses `uv` as the Python package manager because it is faster than `pip` and handles virtual environment management and dependency resolution in one step.

**Setup steps:**
1. Clone the repository: `git clone https://github.com/shangrilar/ai-agent-from-scratch`
2. Install `uv` if not already installed.
3. Run `uv sync` to create the virtual environment and install all dependencies.
4. Launch Jupyter Lab: `uv run jupyter lab`
5. Configure API keys in a `.env` file (at minimum, an OpenAI API key and a Hugging Face token for GAIA).

To load environment variables, use `python-dotenv`:
```python
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
```
The `find_dotenv()` function searches parent directories to find the `.env` file, so it works regardless of which subdirectory you run your code from.

### 3.2 Getting Started with the OpenAI API

OpenAI's Chat Completions API has become the industry standard. Most LLM providers offer similar interfaces, so understanding the OpenAI API makes it easy to work with other providers.

**Basic API call:**
```python
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)
print(response.choices[0].message.content)
```

**Three message roles:**
- **developer (previously system):** Defines the model's overall behavior.
- **user:** Messages from the user.
- **assistant:** Responses generated by the model.

**Key parameters:**
- **`temperature`:** Ranges from 0 to 2. Lower values produce more deterministic responses; higher values generate more creative outputs. Use 0–0.3 for accuracy tasks and 0.7–1.0 for creative tasks.
- **`max_tokens`:** Limits the number of tokens generated.

Other providers work similarly. For example, Anthropic's API uses `client.messages.create()` and returns `response.content[0].text`. Managing these differences across providers makes development cumbersome.

### 3.3 Unifying Providers with LiteLLM

LiteLLM is a library that lets you call more than 100 LLMs through a unified interface. A single `completion` function handles all providers:
```python
from litellm import completion
response = completion(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
response = completion(
    model="claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Hello!"}]
)
```
Change the model name, and the response format is unified to the OpenAI format. API keys for each provider are read automatically from environment variables.

### 3.4 Conversation Management: Handling Stateless APIs

LLM APIs are **stateless**: each API call is independent and has no memory of previous calls. To maintain conversation continuity, you must manage the conversation history manually.

**The pattern:**
```python
messages = []
# First exchange
messages.append({"role": "user", "content": "My name is Jungjun."})
response1 = completion(model="gpt-5-mini", messages=messages)
assistant_message1 = response1.choices[0].message.content
messages.append({"role": "assistant", "content": assistant_message1})
# Second exchange - includes previous conversation history
messages.append({"role": "user", "content": "What is my name?"})
response2 = completion(model="gpt-5-mini", messages=messages)
```
We accumulate all conversation content in the `messages` list and pass the entire history with each call. This approach is even more important in agent development because agents must add tool-call results, search results, and intermediate reasoning steps to the context.

### 3.5 Generating Structured Output

The natural-language text generated by LLMs is great for humans but inconvenient for programs to process. To call tools, an agent must use a structured format to output the tool to call and the arguments to use.

**Pydantic** is a Python library for data validation that lets you define expected data structures as classes:
```python
from pydantic import BaseModel
from litellm import completion
class ExtractedInfo(BaseModel):
    name: str
    email: str
    phone: str | None = None
response = completion(
    model="gpt-5-mini",
    messages=[{
        "role": "user",
        "content": "My name is John Smith, my email is john@example.com, and my phone is 555-1234."
    }],
    response_format=ExtractedInfo
)
result = response.choices[0].message.content
```
When you pass a Pydantic model to `response_format`, the LLM generates JSON matching that schema. Pydantic automatically validates data against the schema, raising clear errors if the data doesn't match.

Structured output plays a crucial role in agent development. In tool calling, the LLM must use a structured format to output which tool to call and which arguments to use.

### 3.6 Asynchronous Calls

When developing agents, you often process multiple LLM requests simultaneously (e.g., evaluating dozens of problems in benchmark tests, comparing responses from multiple models, or running multiple agents concurrently).

Python handles asynchronous programming with `async`/`await` syntax and the `asyncio` library. LiteLLM supports asynchronous calls through the `acompletion` function:
```python
import asyncio
from litellm import acompletion
async def get_response(prompt: str) -> str:
    response = await acompletion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
prompts = [
    "What is 2 + 2?",
    "What is the capital of Japan?",
    "Who wrote Romeo and Juliet?"
]
tasks = [get_response(p) for p in prompts]
results = await asyncio.gather(*tasks)
```
Functions defined with `async def` are called **coroutines**. The `await` keyword waits for an asynchronous operation to complete and allows other tasks to run in the meantime. `asyncio.gather` executes multiple coroutines concurrently and returns all results together.

**Rate limiting and retries:** When you send many requests simultaneously, two problems arise: rate limits from the API provider and transient failures. LiteLLM's `num_retries` parameter handles transient failures with automatic exponential backoff. For rate limiting, use Python's `asyncio.Semaphore`:
```python
semaphore = asyncio.Semaphore(10)
async def call_llm(prompt: str) -> str:
    async with semaphore:
        response = await acompletion(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            num_retries=3
        )
        return response.choices[0].message.content
```
The semaphore ensures that even if we create 100 tasks at the same time, only 10 API calls run simultaneously. The `return_exceptions=True` argument in `asyncio.gather` prevents a single failure from canceling all other tasks.

## 4. Enhancing Agent Intelligence: Prompt Engineering

After selecting an LLM and learning to use its API, you must teach the LLM to function properly as an agent. This is the domain of prompt design.

**Prompts fall into two main categories:**
- **User prompts:** Messages that users type in the chat interface, changing every turn and staying beyond the developer's control.
- **System prompts:** Developer-defined instructions that persist throughout the conversation, setting the agent's personality, permissions, constraints, and tool-use policies.

This book focuses on **system prompt design** because a well-crafted system prompt transforms a general-purpose LLM into a reliable agent for specific tasks.

### 4.1 The Role of System Prompts

System prompts are particularly important for agents because agents act autonomously across multiple steps. From a context-engineering perspective, a system prompt is information that is always included in the context.

Analyzing Claude's system prompts (published by Anthropic) reveals four main roles:

| Role | Description |
| :--- | :--- |
| **Defining product identity** | Specifying "who" the agent is and what it can do. |
| **Specifying output format and style** | Instructions about tone, formatting, and when to use lists vs. prose. |
| **Setting behavioral boundaries** | Defining what not to do (e.g., refusing harmful requests, limiting refund authority). |
| **Clarifying knowledge limits** | Instructing the model to honestly say it cannot know about events after its knowledge cutoff. |

### 4.2 Guidelines for Agent Prompts

A chatbot converses with users, whereas an agent works on their behalf. This difference calls for a fundamentally different approach to prompt design.

**Acting autonomously:** Agents should act before asking questions. When the user's intent is clear, execute immediately without asking for permission. The key principle is: "If it's not ambiguous, don't ask; just execute." When the user's intent is too ambiguous, clarification is necessary.

**Using tools strategically:**
- **Setting the default attitude:** "Use when necessary" is vague; "use proactively, without asking permission" is clearer.
- **Defining trigger patterns:** Provide guidance that helps the LLM identify when to invoke tools based on linguistic signals (e.g., "What we discussed..." triggers a conversation search).
- **Specifying negative guides:** Knowing when not to use tools is equally important. For a search tool: "Don't search for timeless information, fundamental concepts, definitions, or well-established technical facts."
- **Anchoring with examples:** Concrete input-output pairs help the LLM understand expected behavior much better than abstract rules.

**Using tool information in the prompt:** For an agent to use tools effectively, you must explain what each tool does, what parameters it requires, and what results it returns. The quality of these descriptions directly affects the LLM's tool-selection accuracy.

## 5. Experiencing LLM Limitations with GAIA

### 5.1 Why GAIA: Setting Goals for Agent Development

Before diving into the experiment, we need a clear target to aim for. Without a concrete benchmark, you might spend weeks tweaking prompts and adding features without knowing whether you are making progress.

GAIA provides test cases that meet three criteria:
- Have clear, verifiable answers (so we can measure accuracy objectively).
- Represent realistic user requests so improvements translate to real-world value.
- Cover a range of difficulty levels (so we can track progress as we add capabilities).

**Thinking of GAIA as simulated user inputs:** Imagine each GAIA problem as a request from a real user of our research agent. The user expects the agent to figure out how to answer, not to ask where to find the Kipchoge's pace or the Earth-Moon distance.

### 5.2 Loading the GAIA Dataset

GAIA is a gated dataset, so you need a Hugging Face account and must accept the dataset's terms of use. We focus on level-1 problems in this chapter to establish a baseline for what LLMs can do without tools.
```python
from datasets import load_dataset
level1_problems = load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")
print(f"Number of Level 1 problems: {len(level1_problems)}")
```

**GAIA problem structure:**
```python
{
    'task_id': 'e1fc63a2-da7a-432f-be78-7c4a95598703',
    'Question': 'If Eliud Kipchoge could maintain his record-making marathon pace indefinitely...',
    'Level': 1,
    'Final answer': '17',
    'file_name': '',
    'Annotator Metadata': {
        'Steps': '1. Googled Eliud Kipchoge marathon pace...',
        'Tools': '1. A web browser.\n2. A search engine.\n3. A calculator.',
        'Number of tools': '3'
    }
}
```

**Three difficulty levels:**
- **Level 1:** Generally requires no tools, or at most one tool, with no more than five steps.
- **Level 2:** Involves more steps (roughly 5–10) and requires combining tools.
- **Level 3:** Designed for a near-perfect general assistant, requiring arbitrarily long sequences of actions.

### 5.3 Experiment Setup

**Defining the response format:** We use structured output with a Pydantic model:
```python
from pydantic import BaseModel
class GaiaOutput(BaseModel):
    is_solvable: bool
    unsolvable_reason: str = ""
    final_answer: str = ""
```

**Building the evaluation pipeline:** We use GAIA's standard evaluation prompt, which instructs models to provide answers in a consistent format. The prompt includes rules such as "don't use a comma to write your number" and "don't use units such as $ or a percent sign."

**Managing API rate limits:** Different providers impose different rate limits. We create a separate semaphore for each provider:
```python
PROVIDER_SEMAPHORES = {
    "openai": asyncio.Semaphore(30),
    "anthropic": asyncio.Semaphore(10),
}
```

**The core solving function:**
```python
async def solve_problem(model: str, question: str) -> GaiaOutput:
    provider = get_provider(model)
    async with PROVIDER_SEMAPHORES[provider]:
        response = await acompletion(
            model=model,
            messages=[
                {"role": "system", "content": gaia_prompt},
                {"role": "user", "content": question},
            ],
            response_format=GaiaOutput,
            num_retries=2,
        )
        finish_reason = response.choices[0].finish_reason
        content = response.choices[0].message.content
        if finish_reason == "refusal" or content is None:
            return GaiaOutput(
                is_solvable=False,
                unsolvable_reason=f"Model refused to answer (finish_reason: {finish_reason})",
                final_answer=""
            )
    return GaiaOutput.model_validate_json(content)
```

**Validating answers:** GAIA uses exact-match evaluation, ignoring case and whitespace:
```python
def is_correct(prediction: str | None, answer: str) -> bool:
    if prediction is None:
        return False
    return prediction.strip().lower() == answer.strip().lower()
```

### 5.4 Results and Analysis: Why LLMs Need Tools

Let's examine what happened when we ran four models against 20 level-1 GAIA problems:

| Model | Accuracy | Judged solvable |
| :--- | :--- | :--- |
| GPT-5 | 10/20 (50%) | 11/20 (55%) |
| GPT-5 mini | 4/20 (20%) | 8/20 (40%) |
| Claude Sonnet 4.5 | 3/20 (15%) | 4/20 (20%) |
| Claude Haiku 4.5 | 3/20 (15%) | 4/20 (20%) |

**Only five problems are truly solvable without tools.** This means the theoretical maximum accuracy for any LLM without tools is 25%. Yet we see models attempting far more problems and sometimes succeeding. The likely explanation is training data: modern LLMs are trained on vast amounts of web content, including Wikipedia articles and reference materials. But this "knowledge" is unreliable for agent development because the model cannot distinguish between information it knows accurately and information it is confabulating.

**Model refusals:** Some models refused to answer certain problems, returning a refusal rather than attempting an answer. For example, one GAIA problem contains text written backward. When reversed, it simply asks the model to write the opposite of "left" as the answer. But some models flagged it as a potential attempt to bypass safety filters through obfuscation and refused to engage with it.

**Why 15 problems remain unsolvable:**

| Missing capability | Count |
| :--- | :--- |
| Web search and web page access | 11 |
| File reading | 6 |

The Kipchoge problem exemplifies why web access matters. Even if the model happens to know Kipchoge's marathon pace from training, the problem explicitly requires checking "the minimum perigee value on the Wikipedia page for the Moon." The user expects the agent to look this information up, not to guess it from training data.

### 5.5 Key Takeaways

This experiment reveals three critical insights for agent development:

1. **LLMs have reasonable self-awareness of their limitations.** When properly prompted, models can often identify when they cannot solve a problem. This capability is valuable because an agent that knows what it cannot do can use appropriate tools instead.
2. **Training data is no substitute for tools.** Even when models "know" information from training, that knowledge is unreliable for precise, current, or verifiable answers. Agents need tools to access authoritative sources.
3. **Most real-world questions require external capabilities.** In our sample, 75% of even the easiest GAIA problems required tools. For a research agent to be genuinely useful, it must be able to search the web, read files, and perform calculations.

## 6. Key Takeaways

- When starting agent development, closed LLMs offer practical advantages: out-of-the-box features such as tool calling and structured output, stable APIs that let you focus on agent logic, and top-tier performance for complex reasoning tasks. Open LLMs become relevant later for cost optimization.
- LiteLLM provides a unified interface across providers, enabling model flexibility in a rapidly evolving landscape. Key API patterns include conversation management, structured output with Pydantic, and asynchronous calls with rate limiting.
- System prompts define agent behavior: product identity, output format, behavioral boundaries, and knowledge limits. For agents, prompts should emphasize taking autonomous action and using tools strategically rather than asking for permission.
- The GAIA benchmark reveals that even capable LLMs solve only 25% of easy problems without tools. The other 75% of problems require external capabilities such as web search and file reading.
- Models show reasonable self-awareness of limitations when properly prompted, but training-data knowledge is unreliable for precise, current, or verifiable answers. Agents need tools to access authoritative sources.

---

# Chapter 2 — Teaching Outline and Learning Roadmap

### Teaching Instructions

Teach this chapter step by step, following the outline below. Each lesson should include:

- A clear explanation of the concept.
- A practical example, preferably in Python.
- A discussion of why the concept matters when building AI agents.
- A small exercise or coding task.
- A connection to the existing GitHub project when relevant.

Do not skip topics or assume prior knowledge of concepts that have not yet been explained. Clearly distinguish between concepts covered in the book and additional implementation ideas for your project.

### Lesson 1: Choosing LLMs for Agents

- Open vs. closed LLMs: definitions and examples.
- Three reasons to start with closed LLMs: rapid experimentation, top performance, production-ready features.
- When to consider open LLMs: cost optimization, stability, and control.
- **Exercise:** For a prototype agent that answers customer questions, decide whether to start with a closed or open LLM. Explain your reasoning based on the chapter's criteria.

### Lesson 2: Essential LLM Capabilities

- Three essential capabilities: tool calling, structured output, and long context support.
- Designing for model agility: why LiteLLM matters in a rapidly changing landscape.
- The book's choice: OpenAI API with LiteLLM for vendor flexibility.
- **Exercise:** List three LLM providers and compare their support for tool calling and structured output. Which would you choose for an agent that needs both?

### Lesson 3: Setting Up the Development Environment

- Cloning the repository and installing `uv`.
- Running `uv sync` and launching Jupyter Lab.
- Configuring API keys in a `.env` file.
- Loading environment variables with `python-dotenv` and `find_dotenv()`.
- **Exercise:** Set up your development environment. Verify that you can load environment variables and make a basic API call.

### Lesson 4: Getting Started with the OpenAI API

- The Chat Completions API and the `client.chat.completions.create()` method.
- The three message roles: developer, user, and assistant.
- Key parameters: `temperature` and `max_tokens`.
- **Exercise:** Make a basic API call to GPT-5 mini and ask it to classify a sentence as positive, negative, or neutral. Experiment with different `temperature` values and observe the differences.

### Lesson 5: Unifying Providers with LiteLLM

- The problem: different providers have different interfaces.
- The solution: LiteLLM's unified `completion` function.
- How to switch models by changing the model name.
- **Exercise:** Write a script that uses LiteLLM to call both GPT-5 mini and Claude Sonnet 4.5 with the same prompt. Compare their responses.

### Lesson 6: Conversation Management

- The stateless nature of LLM APIs.
- The pattern: accumulate messages in a list and pass the entire history with each call.
- Why this matters for agent development: agents must add tool-call results and intermediate reasoning steps.
- **Exercise:** Write a script that maintains a conversation with an LLM across three turns. Verify that the LLM remembers information from the first turn.

### Lesson 7: Generating Structured Output

- The problem: natural-language text is inconvenient for programs to process.
- Pydantic: defining expected data structures as classes.
- Passing a Pydantic model to `response_format`.
- **Exercise:** Define a Pydantic model for a product review (with fields such as `rating`, `sentiment`, and `key_points`). Use it to extract structured information from a review text.

### Lesson 8: Asynchronous Calls

- The problem: sequential requests are slow.
- The solution: `async`/`await` and `asyncio.gather`.
- Rate limiting with `asyncio.Semaphore` and retries with `num_retries`.
- **Exercise:** Write an asynchronous script that sends 20 requests concurrently with a semaphore limit of 5. Measure the total execution time and compare it with sequential execution.

### Lesson 9: The Role of System Prompts

- The four roles of system prompts: product identity, output format, behavioral boundaries, and knowledge limits.
- Why system prompts are particularly important for agents.
- **Exercise:** Write a system prompt for a customer support agent that handles refund requests. Include all four roles.

### Lesson 10: Guidelines for Agent Prompts

- Acting autonomously: "If it's not ambiguous, don't ask; just execute."
- Using tools strategically: default attitude, trigger patterns, negative guides, anchoring with examples.
- The importance of tool descriptions.
- **Exercise:** Write a system prompt for a research agent with a web search tool. Include at least one trigger pattern and one negative guide.

### Lesson 11: The GAIA Benchmark

- What GAIA is and why it is used in this book.
- The three difficulty levels and the problem structure.
- How to load the dataset and handle gated access.
- **Exercise:** Load the GAIA level-1 dataset and examine three problems. For each, identify the required tools and the number of steps.

### Lesson 12: Experiment Setup and Evaluation Pipeline

- Defining the response format with a Pydantic model.
- Managing API rate limits with provider-specific semaphores.
- The core solving function and answer validation.
- **Exercise:** Implement the `solve_problem` function and the `is_correct` validation function. Test them on a single GAIA problem.

### Lesson 13: Results and Analysis

- The experiment results across four models.
- Why only 25% of problems are solvable without tools.
- Model refusals and their implications for production agents.
- Why 15 problems remain unsolvable: web search and file reading.
- **Exercise:** Run the experiment on 20 level-1 problems and analyze the failures. For each failure, categorize it as either "requires web search" or "requires file reading."

### Lesson 14: Key Insights and Next Steps

- Three critical insights: LLM self-awareness, training data is no substitute for tools, and most real-world questions require external capabilities.
- The baseline: without tools, even capable LLMs solve only a fraction of level-1 problems.
- Next chapter: giving the agent the ability to use tools.
- **Exercise:** Based on the experiment results, list three tools you would add to the agent first. Explain why each tool would address a major failure mode.