# Chapter 10 Summary — Evaluating Agents

**Note:** This document is based on Chapter 10 of *Build an AI Agent (From Scratch)* and follows the same structure as the Chapter 6, Chapter 8, and Chapter 9 summaries.

## 1. Introduction: Why Is Agent Evaluation Essential?

An agent goes beyond merely suggesting an answer. It decides for itself what information to seek, which tools to use, in what sequence to execute tasks, and when to stop. This autonomy is powerful, but a single small mistake can cascade into real-world costs and risks: payments, procurement, permission changes, and external communications. Rigorous evaluation is essential to prevent agents from causing unexpected harm and to protect users from significant losses.

**Why evaluation is difficult:**
- **The evaluation space is enormous.** An agent's output is formed through a chain of multistep reasoning and actions that interlock and accumulate. If you evaluate only the final output, it becomes difficult to distinguish a correct result that happened by luck from one produced by consistently sound reasoning.
- **Defining good performance is not straightforward.** For tasks with clear ground truth, evaluation is relatively simple. For open-ended tasks such as writing or problem solving, the first challenge is establishing an explicit rubric: what counts as success and by what criteria quality should be judged.
- **Nondeterminism.** Even with the same input, an agent can reach an answer through different paths each time. An agent that got lucky cannot guarantee the same quality next time.

**The solution:** Automation is essential, but building an automated evaluation is itself extremely difficult. The chapter walks step by step through the procedures required to automate evaluation: observation, dataset creation, error analysis, rubric design, LLM-as-a-judge, and CI/CD integration.

## 2. Observing an Agent: The Foundation of Evaluation

Observability is tightly coupled to evaluation because the data collected during system operation becomes the input for evaluation. Observability means making every layer of an agent visible, from input handling to reasoning, tool use, and final output.

**Three pillars of observability:**

| Pillar | Description | Examples |
| :--- | :--- | :--- |
| **Metrics** | Numerical measurements of a system's state at a point in time. | Latency per tool call, total task-completion time, token use per interaction, cost per task, LLM call error rate. |
| **Traces** | The full journey of a single request through the system, subdivided into spans and events. | The entire workflow from interpreting a user request to invoking tools and generating the final response. |
| **Logs** | Information such as authentication failures, error messages, and tool invocation inputs and outputs. | API call errors, service start/stop events, tool invocation details. |

**Spans and events:**
- A **span** is a unit of work within a trace (an LLM call, a tool invocation, a memory read, or a planning step).
- An **event** is a notable instantaneous occurrence within a span (an exception, the start of a retry, user feedback). Events have no duration—only timestamps.

**OpenTelemetry (OTel):** An open-source, vendor-neutral standard for collecting and exporting telemetry data. Its biggest advantage is vendor neutrality: instrumenting with OTel eliminates lock-in to any proprietary monitoring solution. The OpenTelemetry Protocol (OTLP) specifies how telemetry data is structured and transmitted. Commercial tools such as Langfuse, LangSmith, Arize AI, and Weights & Biases provide dashboards and evaluation tools on top of the collected data.

**What to instrument:** Input prompt, intermediate reasoning, tool-call logs, tool outputs, final output, runtime/cost.

## 3. Building Datasets and Establishing Evaluation Criteria

Agent quality is not merely about whether an answer sounds plausible. Because agents can call tools that change real-world state, evaluation must include not only the agent's words but also the consequences of its actions.

**The evaluation design process:**
1. Build a dataset containing representative user requests.
2. Run the agent on the dataset and collect full execution traces (tool calls, intermediate reasoning, outputs, errors).
3. Perform error analysis by reviewing traces and identifying recurring failure patterns.
4. Group similar errors into a failure taxonomy.
5. Derive evaluation criteria and organize them into a rubric.
6. Define metrics that translate the criteria into measurable, trackable numbers.

### 3.1 Three Perspectives of Agent Evaluation

| Perspective | Question | Example |
| :--- | :--- | :--- |
| **Final output** | Does the delivered result satisfy the requirements? | Does the summary accurately capture key points without omissions? Does it follow the required format? |
| **Component** | Did the individual modules perform their roles correctly? | Did the search tool return relevant documents? Did the LLM interpret the results correctly? |
| **Trajectory** | Was the path the agent took reasonable and efficient? | Did it reach the answer via a minimal path without unnecessary tool calls? |

### 3.2 Creating a Dataset

| Type | Description | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Benchmark** | Standardized public evaluation sets (e.g., GAIA, SWE-bench, HotpotQA). | Comparable across systems; validated difficulty. | May diverge from real use; risk of overfitting. |
| **Custom** | Internally constructed sets reflecting real scenarios. | Tailored to product needs; domain-specific. | High build cost. |
| **Production** | Real user requests and feedback collected in production. | Most realistic; reflects real use patterns. | Often lacks ground-truth labels. |

**Three categories for a custom test set:**
- **Representative core use cases:** Key scenarios most users execute most frequently.
- **Complex and challenging edge cases:** Ambiguous inputs, multiple constraints, small errors with large downstream impact.
- **Adversarial or stress-testing inputs:** Malicious inputs, attempts to bypass policies, prompts designed to exploit weaknesses.

### 3.3 Analyzing Errors

**Step 1 — Open coding:** Evaluators read traces and describe problems in their own words rather than choosing from predefined categories. This work is performed by product engineers, domain specialists, or QA engineers. Notes are collected in a spreadsheet with fields such as input, agent output, tool trace, and evaluation note.

**Step 2 — Build a failure taxonomy:** Similar observations are grouped into broader categories. For example, "Citations do not match the source," "Sources unclear," and "Answer sounds plausible but doesn't answer the question" can be grouped into evidence/citation failures, reasoning/synthesis failures, and format/requirement compliance failures.

**Step 3 — Repeat:** If no new failure types emerge in new traces, the first taxonomy pass is complete. Repeat the process whenever there are major changes. Review 50–100 samples per cycle.

### 3.4 Designing Rubrics and Metrics

**Rubric:** A scoring guide that makes the criteria for good/bad explicit. Binary-first designs (PASS/FAIL) are generally more stable than Likert scales.

**Example rubric items:**
- **Answer relevancy:** PASS if it directly answers the user's core question; FAIL if it provides a plausible summary but does not answer the core question.
- **Citation reliability:** PASS if key claims are supported by sources; FAIL if citations differ from the original text or are unrelated to the claims.
- **Requirement compliance:** PASS if it follows the requested format; FAIL if it breaks the format or omits required criteria.

**Metrics:** Aggregate judgments into trackable numbers:
- **Overall pass rate (OPR):** The fraction of cases that PASS on all rubric items.
- **Answer relevancy rate:** PASS rate for the answer relevancy rubric item.
- **Citation reliability rate:** PASS rate for the citation reliability rubric item.
- **Requirement compliance pass rate:** PASS rate for the requirement compliance rubric item.

## 4. Evaluating with LLM-as-a-Judge

LLM-as-a-judge uses an LLM to evaluate the outputs of another system. The main motivation is scalability.

**Four categories of evaluators:**

| Evaluator | Description | Advantages | Limitations |
| :--- | :--- | :--- | :--- |
| **Code-based** | Automatically grades using predefined rules or programs. | Objective, reproducible, cheap, fast. | Works only when criteria can be precisely expressed in code. |
| **LLM judge** | Uses another language model to evaluate the output. | Scalable, can evaluate thousands of cases quickly. | Incurs additional cost; may not match human judgment perfectly. |
| **Human evaluation** | Professional graders or domain experts review outputs. | Most trustworthy for complex context and subjective quality. | Expensive, slow, difficult to scale. |
| **End-user feedback** | Thumbs-up/down ratings, star scores, follow-up questions. | Reflects real user satisfaction; scales easily. | Noisy; most satisfied users leave no feedback. |

**Best practice:** Use a stronger model as the judge than the one being evaluated. For example, if an agent was built with Claude Haiku, choose Claude Opus as the judge.

### 4.1 Types of LLM-as-a-Judge

**Single-output grading:** The judge evaluates one output at a time and assigns a score according to the rubric. If an expected output exists, it can be provided alongside the agent response to improve scoring consistency.

**Pairwise comparison:** The judge compares two candidate outputs for the same input and decides which one better satisfies the evaluation criteria. This is often simpler and more reliable for subjective judgments such as fluency, clarity, or helpfulness. The judge selects A, B, or Tie, and results are aggregated into win rates.

### 4.2 Building a Rubric-Based Evaluation System

**Input data:** User prompt, output to evaluate, rubric, and if available, a reference answer.

**Evaluator prompt structure:**
- **System prompt:** Defines the evaluator's role and critical rules (e.g., "Base your judgment ONLY on the provided evidence_pack," "Do not reward verbosity," "Output MUST be a single JSON object only").
- **User prompt:** Provides the query, final answer, evidence pack, and rubric.
- **Output JSON schema:** Specifies the expected structure with `overall_pass`, `rubric` (with PASS/FAIL for each item), and `failure_reasons` (with problematic spans and explanations).
- **Decision rule:** `overall_pass` is PASS only if ALL rubric items are PASS.

## 5. Operations: CI/CD

In production systems, agent behavior can change even with small updates such as prompt edits, tool integrations, and model upgrades. Without systematic checks, these changes can silently degrade answer quality. Evaluation must become part of the operational pipeline.

### 5.1 Evaluation-Gated Deployment

The core principle: without quality verification, there is no production deployment.

**Stage 1 — Premerge integration (CI):** Runs automatically when a PR is opened. Executes fast, repeatable checks such as unit tests, linting, and dependency scans. Uses a small test set (around 20–50 cases) to confirm that the branch is healthy and performance has not regressed.

**Stage 2 — Staging validation (CD):** After a change passes CI and is merged, the Ops/platform team packages the agent and deploys it to staging. Runs heavier, more realistic validation: load testing, long-run stability testing, remote-service integration tests, and evaluation using a larger test set.

### 5.2 Improving the Test Set and Evaluator: Agent Quality Flywheel

Evaluation is also a product, and it must continuously improve after deployment. An agent's failure patterns evolve as use environments, user behavior, and external tools/APIs change. A single test set and rubric are not sufficient to maintain quality in the long term.

**The four-step flywheel:**
1. **Define quality:** Establish the criteria for good/bad agent behavior.
2. **Instrument for visibility:** Collect telemetry data from real-world behavior.
3. **Evaluate the process:** Analyze the collected data to identify failure patterns.
4. **Architect the feedback loop:** Update datasets, rubrics, and metrics based on insights, feeding back to step 1.

## 6. Key Takeaways

- As agent autonomy increases, failures shift from minor bugs to real-world effects with meaningful cost.
- Evaluation must cover not only the final output but also the components and trajectory to separate luck from reproducible success.
- Instrumentation and observability are required because evaluation is impossible without execution records.
- Strong evaluation starts by converting failures to a shared language via trace review, open-coded error logging, and a failure taxonomy.
- Convert the most frequent and severe failures into a PASS/FAIL-first rubric. Metrics are a dashboard that turns rubrics into numbers to catch regressions and guide improvement.
- Embed automated evaluation in the deployment pipeline.
- Treat evaluation as a continuous flywheel because environments, users, and tools/APIs change: operational data → error analysis → taxonomy → rubric/metrics → test set updates.
- Building an agent is not a one-time act of engineering but a continuous cycle of improvement, in which every deployment teaches you something new about what your agent can and cannot do.

---

# Chapter 10 — Teaching Outline and Learning Roadmap

### Teaching Instructions

Teach this chapter step by step, following the outline below. Each lesson should include:

- A clear explanation of the concept.
- A practical example, preferably in Python.
- A discussion of why the concept matters when building AI agents.
- A small exercise or coding task.
- A connection to the existing GitHub project when relevant.

Do not skip topics or assume prior knowledge of concepts that have not yet been explained. Clearly distinguish between concepts covered in the book and additional implementation ideas for your project.

### Lesson 1: Why Agent Evaluation Is Essential

- The shift from minor bugs to real-world effects with meaningful cost.
- The three challenges: enormous evaluation space, difficulty defining good performance, nondeterminism.
- Why evaluating only the final output is insufficient: separating luck from reproducible success.
- **Exercise:** List three failure modes of an agent that could cause real-world harm. For each, explain why evaluating only the final output would miss it.

### Lesson 2: The Three Pillars of Observability

- Metrics, traces, and logs: what each one tells you.
- Spans and events: units of work within a trace.
- The difference between traditional monitoring and agent observability.
- **Exercise:** Design a telemetry schema for an agent that performs web search and code execution. List the metrics, traces, and logs you would collect.

### Lesson 3: OpenTelemetry and Instrumentation

- The concept of instrumentation: inserting code to emit telemetry.
- OpenTelemetry's vendor neutrality and the OTLP protocol.
- What to instrument: input prompt, intermediate reasoning, tool-call logs, tool outputs, final output, runtime/cost.
- **Exercise:** Identify the five most important spans in an agent's execution and explain why each one matters for evaluation.

### Lesson 4: Three Perspectives of Agent Evaluation

- Final output: does the result satisfy the requirements?
- Component: did individual modules perform their roles correctly?
- Trajectory: was the path reasonable and efficient?
- **Exercise:** For a research agent that summarizes papers, write one evaluation question for each perspective.

### Lesson 5: Creating an Evaluation Dataset

- Three sources: benchmarks, custom datasets, production data.
- Pros and cons of each source.
- Three categories for a custom test set: representative core cases, complex edge cases, adversarial inputs.
- **Exercise:** Design a 30-case test set for a customer support agent. Allocate 15 cases to core scenarios, 10 to edge cases, and 5 to adversarial inputs. Describe one example for each category.

### Lesson 6: Error Analysis — Open Coding and Failure Taxonomy

- Open coding: reading traces and describing problems in your own words.
- Who performs open coding: product engineers, domain specialists, QA engineers.
- Building a failure taxonomy: grouping similar observations into broader categories.
- The repeat cycle: review 50–100 samples per cycle.
- **Exercise:** Given three example traces with the following notes — "Tool failed but agent claims success," "Citation does not match source," "Answer is plausible but does not answer the question" — group them into a failure taxonomy and name each category.

### Lesson 7: Designing Rubrics and Metrics

- The difference between Likert scales and binary decisions.
- Why binary-first designs are more stable.
- Example rubric items: answer relevancy, citation reliability, requirement compliance.
- Metrics: overall pass rate, answer relevancy rate, citation reliability rate, requirement compliance pass rate.
- **Exercise:** Write a PASS/FAIL rubric for a code-generation agent. Include at least three rubric items and define the overall pass rate metric.

### Lesson 8: LLM-as-a-Judge

- The motivation: scalability.
- Four categories of evaluators: code-based, LLM judge, human evaluation, end-user feedback.
- Best practice: use a stronger model as the judge.
- **Exercise:** For each of the four evaluator categories, give one example of a task where that category is the best choice. Explain why.

### Lesson 9: Types of LLM-as-a-Judge

- Single-output grading: evaluate one output at a time with a rubric.
- Pairwise comparison: compare two outputs and select A, B, or Tie.
- When to use each type.
- **Exercise:** Design a pairwise comparison prompt for evaluating two customer support responses. Include the evaluation criteria and the expected output format.

### Lesson 10: Building a Rubric-Based Evaluation System

- The input data: user prompt, output to evaluate, rubric, reference answer.
- The evaluator prompt structure: system prompt, user prompt, output JSON schema, decision rule.
- Critical rules: base judgment only on evidence, do not reward verbosity, output must be a single JSON object.
- **Exercise:** Write the system prompt for a research agent evaluator. Include at least four critical rules and define the output JSON schema.

### Lesson 11: Evaluation-Gated Deployment

- The core principle: without quality verification, there is no production deployment.
- Stage 1 (CI): fast, repeatable checks with a small test set (20–50 cases).
- Stage 2 (CD): heavier validation with a larger test set and realistic operational concerns.
- **Exercise:** Design a CI/CD pipeline for an agent. Specify what checks run at each stage, how many test cases are used, and what threshold triggers a deployment block.

### Lesson 12: The Agent Quality Flywheel

- Evaluation as a product that must continuously improve.
- The four-step flywheel: define quality, instrument for visibility, evaluate the process, architect the feedback loop.
- Why a single test set and rubric are not sufficient in the long term.
- **Exercise:** For a deployed research agent, describe how you would collect production data, analyze failures, update the rubric, and feed insights back into the evaluation dataset.