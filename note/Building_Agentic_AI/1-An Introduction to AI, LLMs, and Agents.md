# Chapter 1 Summary — An Introduction to AI, LLMs, and Agents

**Note:** This document is based on Chapter 1 of *Building Agentic AI* by Sinan Ozdemir (Pearson, 2026) and follows the same structure as the summaries for *Build an AI Agent (From Scratch)*.

## 1. Introduction: The Building Blocks of AI Systems

AI, and more recently large language models (LLMs), went from novelty to required software system building blocks in what felt like the blink of an eye. This chapter provides the bare minimum you need to know to build AI systems with confidence. It covers what an LLM is, how tokens and context windows shape behavior, why size and architecture matter, and where alignment shows up in real systems.

The chapter also introduces the two primary types of LLM pipelines used throughout the book: more deterministic **predefined workflows** and autonomous decision-making **agents** that can use tools and carve their own paths forward.

## 2. The Basics of Large Language Models

In this book, most of the "AI" discussed refers to a **language model**—a probabilistic engine tasked with filling in the blanks from a given piece of text. The blanks are filled with **tokens**—a unit of semantic meaning created by breaking down a sentence or piece of text into smaller units.

### 2.1 Two Types of Language Models

| Type | Description | Examples | Best for |
| :--- | :--- | :--- | :--- |
| **Autoregressive** | Trained to predict the next token in a sentence, based only on previous tokens. The "writers" of LLMs. | GPT, Mistral, Llama, Claude, DeepSeek, Qwen | Text generation, chat, reasoning |
| **Autoencoding** | Trained to reconstruct a piece of text from a corrupted version. The "readers" of LLMs. | BERT, ModernBERT | Embedding, classification, retrieval |

**LLMs** are language models that are either autoregressive, autoencoding, or a combination of the two. Modern LLMs are usually (but not always) based on the **Transformer architecture**, a deep learning architecture put forward in 2017 that revolutionized both AI and the public's interest in AI.

### 2.2 Parameter Count

**Parameters** are the weights the model learns during training. More parameters generally mean more capacity to learn patterns and relationships in data, but they also require more computing power to train and run. Parameter count isn't everything—architecture, training data, and fine-tuning techniques all play a big role.

### 2.3 Context Windows

Language models don't have memory in the way humans do. They can't recall past conversations unless you explicitly feed them that information again in their **context window**—the chunk of text (measured by the token count) they're allowed to process at once, both input and output.

The context window is often referred to as the **short-term memory** of an LLM. Once the window is full and you need to start a new thread, the LLM will forget everything from that thread.

### 2.4 The Needle-in-the-Haystack Problem

As language models included increasingly larger context windows, a problem emerged. If you fill the context window more and more, the LLM has trouble finding little bits of relevant information, effectively forgetting something that's right in the context window. This is called the **needle-in-the-haystack problem**.

The condition in which the needle is placed in the middle of the haystack proved the most challenging. That's why this problem is sometimes referred to as the **"lost in the middle" problem**. The formal name is **positional bias**—the LLM is paying more "attention" to items at the beginning and the end of the prompt due to the architecture of the Transformer.

**The solution:** Instead of filling a prompt with information, retrieve relevant information before augmenting the prompt. This technique is called **Retrieval Augmented Generation (RAG)**.

## 3. The Family Tree of LLM Tasks

Just as LLMs can be split into a family tree based on whether they are more autoencoding or more autoregressive, the types of tasks we want to solve with LLMs can be broken down into categories:

| Category | Task Type | Description |
| :--- | :--- | :--- |
| **Generative** | Multiple choice | Reasoning through a question and a set of predefined choices to pick one or more correct answers. |
| **Generative** | Free text response | Allowing the model to generate free text responses without being bounded by predefined options. |
| **Understanding** | Embeddings | Encoding data to vectors for clustering, recommendations, or another purpose. |
| **Understanding** | Classification | Fine-tuning a model specifically to distinguish between predefined classes. |

## 4. Alignment

Alignment refers to the process of instilling behavior into an AI model that is in line with the human user's expectations. There are four basic types of alignment:

| Type | Description | Example |
| :--- | :--- | :--- |
| **Instructional alignment** | Ensures AI responses match the user's intent, keeping things relevant and conversational. | The model answers the question asked, not a different one. |
| **Chat alignment** | Ensures the AI can follow instructions over multiple "turns" of a conversation. | The model remembers context from earlier in the conversation. |
| **Behavior alignment** | Focuses on what the AI model is allowed or not allowed to do. | The model refuses to generate harmful content. |
| **Style alignment** | Shapes how an AI model communicates, adjusting tone and personality. | Grok's "fun mode" vs. "regular mode." |
| **Value alignment** | Concerned with the *why* behind an AI model. | GPT's default value system reflects Western ideals. |

## 5. Prompt Engineering

### 5.1 Prompt Ordering

An autoregressive LLM reads input from left to right. If you end a prompt with crucial information that contextualizes the information before, the AI won't be able to use that crucial information while reading everything that came before.

**Recommended prompt structure (from top to bottom):**
1. **Asks + Guardrails:** The overall ask and any warnings/guardrails.
2. **Documentation/In-context learning:** Static information the AI needs (company info, database schemas, few-shot examples).
3. **Live context for the system:** Information that changes with the current ask (retrieved documents for RAG).
4. **The current ask of the system:** The dynamic user request.

**Rule of thumb:** If understanding of prompt section A changes given prompt section B, then section B should come before section A.

### 5.2 Chain of Thought (CoT)

By asking an LLM to provide its reasoning before the final answer, we can often get more precise results. This is called **chain-of-thought (CoT) prompting**. Reasoning models are trained to specifically output a reasoning step before giving the final answer, but non-reasoning models can also perform this way with a nudge.

### 5.3 Few-Shot Learning

**Few-shot prompting** is the "show, don't tell" approach to prompting. We give the LLM literal examples of tasks being solved so it can see what to do.

**Best practice:** Include few-shot examples in the **system prompt**, not the user prompt. Foundation labs specifically train LLMs to follow system prompts, so this more accurately replicates the training process.

### 5.4 Prompt Chaining

**Prompt chaining** is the act of creating a string of LLM calls using the output of one LLM call as the input to another. This is the basis of workflows.

There are two ways to chain prompts:
- **Two separate prompts:** Prompt 2's input relies entirely on Prompt 1's output.
- **A single growing prompt:** The same prompt is called multiple times, with each call adding to the prompt.

## 6. Special LLM Features

### 6.1 Inference Parameters

| Parameter | Description | Effect |
| :--- | :--- | :--- |
| **temperature** (float) | Controls randomness. Lower values (< 1) make the model more confident and less random. Higher values make it more random. | 0 = deterministic, 1 = default, >1 = more creative (not recommended). |
| **top_k** (int) | A hard limit on how many tokens the LLM considers when generating. | top_k=7 means the model selects from only the top 7 tokens. |
| **top_p** (float) | Considers tokens from only the top X% of confidences. A softer threshold than top_k. | top_p=0.9 means the model considers the smallest set of tokens that exceeds 90% cumulative probability. |

**Achieving true determinism:** The only way is to set `top_k=1`, which means selecting only the highest-probability token at every step. Not every LLM provider allows this.

### 6.2 Prompt Caching

**Prompt caching** stores computations for static prompt elements (documentation, examples, guardrails) ahead of time, allowing them to be retrieved on future LLM calls with the same static information.

**Why order matters for caching:** If a provider allows prompt caching and you have a dynamic piece of information that changes often, that piece could ruin the cache for everything that comes after. Place static content first and dynamic content last.

**Experiment results:** Across 50 trivia questions, non-reasoning models (GPT-4.1-nano, GPT-4.1-mini, GPT-4.1) showed faster latencies and lower costs over time due to caching. The reasoning model (o4-mini) did not show consistent improvement because its reasoning tokens continually break the cache.

### 6.3 Structured Outputs

**Structured outputs** encourage (but don't technically force) LLM responses to follow a specific JSON schema. This allows for consistently formatted responses that can be reliably parsed by the system surrounding the LLM.

### 6.4 Tool/Function Calling

**Tool calls** provide an LLM with access to tools that may affect an external environment.

**Important note:** The LLM itself cannot and does not actually invoke a tool; instead, it simply recommends a tool to be called with specific parameters. The system around the LLM must actually invoke the tool and provide the tool's response to the LLM.

## 7. LLM Workflows

An **LLM workflow** is a series of LLM calls with predefined pathways and edge-cases with clear start and stop triggers. The developer defines the flow, and the LLM executes specific steps within that structure.

## 8. AI Agents

An **AI agent** is an autonomous LLM with access to tools and the ability to act with agency and without predefined paths.

**Components of an agentic deployment:**
- **Environment:** An external "world" with which the agent interacts.
- **Conversational system:** Through which the agent converses with a user or takes in one-off commands.
- **LLM:** The engine of the operation, tasked with breaking down tasks into subtasks and suggesting tools to use.
- **Prompt:** Which can include few-shot elements, chains-of-thought asks, guardrails, rules, and so on.
- **Tools:** The methods the agent uses to interact with the environment (changing something or retrieving something).
- **Memory:** How far back the agent can recall past events. The basic implementation is the conversational system itself.

### 8.1 ReAct Agents

A **ReAct (Reasoning + Action)** agent follows a basic pattern:
1. **Reason** through what tools it may or may not want to use.
2. **Act** by suggesting a tool to use and writing the input to the tool.
3. The system invokes that tool and passes the observation back to the LLM.
4. Repeat until the LLM decides to no longer call a tool and writes a message to the user about the completion of the task.

**The key distinction between workflows and agents:** LLM workflows are predefined pathways, whereas agents have agency. The LLM can decide to ignore rules, try different approaches, and choose its own path.

## 9. Key Takeaways

- **Model types matter.** Autoregressive models excel at generation, autoencoders shine at reading, and Transformers still dominate most real-world work.
- **Size is only one signal.** The parameter count hints at the model's capability, but context window limits, training data quality, and alignment choices shape its actual performance.
- **Prompt engineering and crafting deliver results.** Prompt ordering, few-shot examples, chain of thought, and tool calls can turn a plain text box into a reliable system.
- **Caches and inference knobs save resources.** Use structured outputs, temperature, top_k, and prompt caching to control cost and latency.
- **Workflows follow rails, whereas agents choose their own path.** Production setups usually combine both.

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

### Lesson 1: What Is a Language Model?

- The definition: a probabilistic engine tasked with filling in the blanks.
- Tokens: the basic unit of semantic meaning.
- The two types: autoregressive (writers) vs. autoencoding (readers).
- **Exercise:** Take a sentence (e.g., "The capital of France is ____") and explain how an autoregressive model would fill in the blank. Then explain how an autoencoding model would fill in a blank in the middle of a sentence.

### Lesson 2: Parameter Count and Context Windows

- Parameters as the weights learned during training.
- More parameters = more capacity, but also more computing power.
- Context windows: the chunk of text the model can process at once.
- The context window as the LLM's "short-term memory."
- **Exercise:** If an LLM has a context window of 100,000 tokens and your prompt is 95,000 tokens, what happens if you ask the model to generate 10,000 tokens? Explain why.

### Lesson 3: The Needle-in-the-Haystack Problem

- The problem: LLMs forget information in the middle of long contexts.
- Positional bias: the LLM pays more attention to the beginning and end.
- The "lost in the middle" effect.
- The solution: Retrieval Augmented Generation (RAG).
- **Exercise:** Design a simple experiment to test the needle-in-the-haystack problem with a model of your choice. Insert a single fact in the middle of a long prompt and see if the model can retrieve it.

### Lesson 4: The Family Tree of LLM Tasks

- Four categories: multiple choice, free text response, embeddings, and classification.
- Generative tasks vs. understanding tasks.
- **Exercise:** For each of the following tasks, identify which category it belongs to: (a) translating a document, (b) classifying customer emails as positive or negative, (c) finding similar documents in a database, (d) answering a multiple-choice question.

### Lesson 5: Alignment

- Five types: instructional, chat, behavior, style, and value.
- Why alignment matters: it dictates which models to consider and how they behave.
- How alignment types can conflict (e.g., behavior alignment vs. style alignment).
- **Exercise:** Write a system prompt for a customer service agent that demonstrates all five types of alignment.

### Lesson 6: Prompt Ordering

- The left-to-right reading of autoregressive LLMs.
- The recommended prompt structure: asks + guardrails → documentation → live context → current ask.
- The rule: if understanding of section A changes given section B, section B should come first.
- **Exercise:** Take a prompt that puts the user's question at the top and the documentation at the bottom. Rearrange it according to the recommended structure and explain why the new order is better.

### Lesson 7: Chain of Thought and Few-Shot Learning

- Chain of thought: asking the LLM to reason before answering.
- Few-shot learning: providing examples in the prompt.
- Best practice: put few-shot examples in the system prompt, not the user prompt.
- **Exercise:** Write a system prompt for a classification task that includes three few-shot examples. Then write the same prompt without few-shot examples and compare the outputs.

### Lesson 8: Prompt Chaining

- The concept: using the output of one LLM call as the input to another.
- Two approaches: two separate prompts vs. a single growing prompt.
- **Exercise:** Design a two-step prompt chain that first detects the language of a text and then translates it into English. Implement it in Python.

### Lesson 9: Inference Parameters

- Temperature, top_k, and top_p.
- The trade-off between determinism and creativity.
- How to achieve true determinism (top_k=1).
- **Exercise:** Write a Python script that calls an LLM with the same prompt three times at temperature=0, temperature=0.7, and temperature=1.0. Compare the outputs.

### Lesson 10: Prompt Caching

- How caching works: storing computations for static prompt elements.
- Why order matters for caching: dynamic content breaks the cache.
- The experiment results: non-reasoning models benefit from caching, reasoning models do not.
- **Exercise:** Design a prompt with a large static system prompt and a small dynamic user message. Explain how you would structure it to maximize cache hits.

### Lesson 11: Structured Outputs and Tool Calling

- Structured outputs: encouraging JSON schema compliance.
- Tool calling: the LLM recommends a tool, but the system executes it.
- The multi-step process: LLM generates tool call → system executes → result fed back to LLM → LLM generates final response.
- **Exercise:** Write a Python function that takes an LLM's tool call and executes it. Then feed the result back to the LLM for a final response.

### Lesson 12: LLM Workflows vs. AI Agents

- Workflows: developer-defined pathways with clear start and stop triggers.
- Agents: autonomous LLMs with access to tools and agency.
- The ReAct pattern: reason → act → observe → repeat.
- The key distinction: workflows follow rails, agents choose their own path.
- **Exercise:** For each of the following tasks, decide whether a workflow or an agent is more appropriate and explain why: (a) classifying customer emails, (b) researching a market and writing a report, (c) translating a document.

### Lesson 13: ReAct Agents in Practice

- The four steps: reason, act, observe, repeat.
- The difference between a workflow and a ReAct agent for the same task.
- **Exercise:** Implement a simple ReAct agent in Python that can search the web and calculate. Test it on a question that requires both tools.