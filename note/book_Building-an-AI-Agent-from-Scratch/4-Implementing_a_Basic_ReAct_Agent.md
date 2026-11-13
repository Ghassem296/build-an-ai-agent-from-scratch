# Chapter 4 Summary — Implementing a Basic ReAct Agent

**Note:** This document is based on Chapter 4 of *Build an AI Agent (From Scratch)* and follows the same structure as the Chapter 1, 2, 3, 6, 8, 9, and 10 summaries.

## 1. Introduction: Combining Reasoning and Action

We have explored how LLMs work and given them the ability to use tools. Now it is time to combine these pieces into something more powerful: an agent that can reason about what it needs, take action to get it, and continue until it solves the problem.

We build an agent using the **ReAct pattern**, chaining multistep reasoning and actions to solve problems end to end, and adding robust error handling. ReAct (reasoning and acting) is not a framework but a way of designing agents that mirrors how humans solve problems: assess the situation, decide what information or action is required, execute that action, observe the results, and repeat.

By the end of this chapter, you will have a working agent that can search the web for real-time information, chain multiple steps of reasoning to solve problems, handle errors gracefully, and return structured output in any format you specify.

## 2. How ReAct Agents Work

Think about how you would answer this question: "If marathon runner Eliud Kipchoge could maintain his world record pace indefinitely, how long would it take him to reach the Moon?" You would not try to answer immediately. Instead, you would think, "I need Kipchoge's marathon pace and the Earth-Moon distance." Then you would search for that information, calculate the result, and formulate your answer.

This natural cycle of thinking and acting is exactly how ReAct agents operate.

### 2.1 The Think-Act Cycle

A ReAct agent would tackle the Kipchoge problem as follows:

- **Thought:** I need to find Kipchoge's marathon world record pace.
- **Action:** `search_web("Kipchoge marathon world record")`
- **Observation:** Kipchoge's record is 2:01:09 for 42.195 km (Berlin 2022).
- **Thought:** Now I need the Earth-Moon distance at closest approach.
- **Action:** `search_wikipedia("Moon perigee distance")`
- **Observation:** The Moon's perigee (closest approach) is 356,500 km.
- **Thought:** I can now calculate the time needed.
- **Action:** `calculator("356500 / (42.195 / 2.0186) * 1")`
- **Observation:** 17,034 hours.
- **Thought:** I have enough information to answer.
- **Final Answer:** Approximately 17,000 hours.

At each step, the agent explicitly states what it is thinking, takes a concrete action, and observes the result. This cycle repeats until the agent has gathered enough information to provide a final answer.

What makes this approach powerful is its adaptability. If the first search returned unhelpful results, the agent could try a different query. If the calculation seemed wrong, the agent could verify it by searching for additional information. Unlike a fixed script, which breaks when something unexpected happens, a ReAct agent adjusts its approach based on what it discovers.

### 2.2 From Text Parsing to Tool Calling

Early ReAct implementations relied entirely on text parsing. The agent had to output responses in an exact format, such as `Action: search["query"]`, and any deviation would break the system. A missing bracket, an extra space, or a slightly different keyword could cause parsing failures.

**Tool calling eliminates this fragility.** Instead of parsing free-form text, the LLM generates structured outputs that specify exactly which tool to call and which arguments to use. The format is guaranteed by the API.

If there is no explicit "thought" step now, is it still ReAct? Yes. The reasoning did not disappear; it moved inside the LLM. When the LLM selects a tool and specifies its arguments, it has already performed the reasoning internally. The pattern remains the same: reason about what is needed, act to get it, observe the result, and repeat.

## 3. Agent Architecture Overview

### 3.1 The Completed Agent

The finished agent will look like this in action:
```python
from scratch_agents import Agent, LlmClient
from scratch_agents.tools import calculator, search_web
agent = Agent(
    model=LlmClient(model="gpt-5-mini"),
    tools=[calculator, search_web],
    instructions="You are a helpful assistant"
)
result = await agent.run("What is 1234 * 5678?")
```
This simple interface hides significant complexity. The agent receives a user question, decides whether to use tools, executes them if necessary, and returns a final answer.

### 3.2 Information Flow: The Core Design

An agent's job is straightforward in concept: receive user input, ask the LLM what to do next, execute tools if asked to do so, feed the results back to the LLM, and repeat until done.

**Why messages alone are not enough:** Running an agent requires tracking much more information than a simple list of messages:
- **`events`:** The history of all interactions, including user messages, LLM responses, and tool-execution results.
- **`current_step`:** A counter that prevents infinite loops.
- **`execution_id`:** A unique identifier for debugging and logging.
- **`state`:** A scratch pad for dynamic data during execution.
- **`final_result`:** A flag that indicates when execution is complete.

The solution is to consolidate everything in a single container: **`ExecutionContext`**. All methods receive this one container, and when new information is required, we modify only the container definition.

**The ExecutionContext-centered design:**
- When the agent needs to call the LLM, it extracts relevant information from `ExecutionContext` into an `LlmRequest`.
- When the LLM responds, the response flows back into `ExecutionContext`.
- When tools execute, the agent passes `ExecutionContext` to them so that they can access any execution state they need, and their results are recorded back in `ExecutionContext`.

This architecture provides two key benefits:
1. It simplifies method signatures because every method receives only the context.
2. It enables context propagation, in which the agent can pass `ExecutionContext` to a tool that needs access to execution state.

Notice the separation between `LlmRequest` and `ExecutionContext`. Whereas `ExecutionContext` stores everything, `LlmRequest` is a curated subset containing only what the LLM needs for a specific call. This separation is where **context engineering** happens.

### 3.3 Components We Have to Build

| Order | Component | Role |
| :--- | :--- | :--- |
| 1 | **ExecutionContext** | Central storage that manages all information during execution. |
| 2 | **Tool abstraction** | Unifies tools from chapter 3 under a consistent interface that can receive context. |
| 3 | **LLM communication layer** | `LlmRequest` selects information for the LLM; `LlmClient` handles API calls; `LlmResponse` standardizes responses. |
| 4 | **Agent** | Orchestrator that creates context, coordinates information flow, and implements the think-act loop. |

## 4. ExecutionContext: The Agent's Central Storage

### 4.1 What Happens During Agent Execution

When an agent solves the Kipchoge problem, a series of events unfolds:
1. User asks, "How long would it take Kipchoge to run to the Moon?"
2. LLM decides that it needs information and requests the `search_web` tool.
3. Tool executes and returns "Kipchoge's record is 2:01:09 for 42.195km."
4. LLM needs more information and requests the `search_wikipedia` tool.
5. Tool executes and returns "Moon's perigee is 356,500 km."
6. LLM can calculate now and requests the `calculator` tool.
7. The tool executes and returns "17034."
8. LLM provides the final answer: "Approximately 17,000 hours."

We can identify three distinct types of occurrences:
- **Messages:** Text exchanges in a conversation.
- **Tool calls:** Calls that occur when the LLM decides to use a tool.
- **Tool results:** Results that capture what happens when tools execute.

These occurrences are defined as data types:
```python
class Message(BaseModel):
    type: Literal["message"] = "message"
    role: Literal["system", "user", "assistant"]
    content: str
class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    name: str
    arguments: dict
class ToolResult(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    status: Literal["success", "error"]
    content: list
ContentItem = Union[Message, ToolCall, ToolResult]
```

For debugging and analysis, we also need to know who produced each piece of content and when. We wrap content items with metadata using an **`Event`** class:
```python
class Event(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    author: str
    content: List[ContentItem] = Field(default_factory=list)
```
The `execution_id` groups all events from a single agent run, enabling us to trace an entire problem-solving session. The `author` field distinguishes between user input and agent actions.

### 4.2 ExecutionContext Implementation

```python
@dataclass
class ExecutionContext:
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: List[Event] = field(default_factory=list)
    current_step: int = 0
    state: Dict[str, Any] = field(default_factory=dict)
    final_result: Optional[str | BaseModel] = None
    def add_event(self, event: Event):
        self.events.append(event)
    def increment_step(self):
        self.current_step += 1
```

Each field serves a specific purpose:
- **`execution_id`:** Unique identifier for this execution session.
- **`events`:** Chronological list of all events that occur during execution.
- **`current_step`:** Counter to prevent infinite loops.
- **`state`:** Flexible key-value store for custom data that tools may need.
- **`final_result`:** Holds the agent's answer when execution completes.

## 5. Tool Abstraction

In chapter 3, we built tools as simple Python functions and created a utility to convert them to tool definitions. This approach worked well for learning the basics, but as we prepare to build a complete agent framework, several practical challenges emerge.

### 5.1 Why We Need a Unified Tool Interface

The function and its metadata exist as separate entities. When building an agent, we pass definitions to the LLM and map names back to functions for execution, keeping everything in sync manually. With tool abstraction, everything stays bundled together:
```python
@tool
def search_web(query: str, max_results: int = 5) -> str:
    return tavily_client.search(query, max_results=max_results)
search_web.tool_definition
search_web.execute(query="Python tutorials")
```

### 5.2 BaseTool: The Foundation

Every tool in our framework inherits from `BaseTool`, an abstract base class that defines what it means to be a tool:
```python
class BaseTool(ABC):
    def __init__(self, name: str = None, description: str = None, tool_definition: Dict[str, Any] = None):
        self.name = name or self.__class__.__name__
        self.description = description or self.__doc__ or ""
        self._tool_definition = tool_definition
    @property
    def tool_definition(self) -> Dict[str, Any] | None:
        return self._tool_definition
    @abstractmethod
    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        pass
    async def __call__(self, context: ExecutionContext, **kwargs) -> Any:
        return await self.execute(context, **kwargs)
```
The `execute` method always receives `context` as its first parameter. This is the key design decision that enables **context propagation**: the Agent passes `ExecutionContext` to every tool, and tools can use it if they need access to execution state.

### 5.3 FunctionTool: Wrapping Functions

`FunctionTool` acts as an adapter that wraps existing functions with the `BaseTool` interface. The key challenge is context propagation. The agent passes `ExecutionContext` to every tool, but most functions do not expect a context parameter. The solution is to inspect each function's signature at initialization. If the function has a context parameter, we pass it; if not, we omit it.
```python
class FunctionTool(BaseTool):
    def __init__(self, func: Callable, name: str = None, description: str = None, tool_definition: Dict[str, Any] = None):
        self.func = func
        self.needs_context = 'context' in inspect.signature(func).parameters
        name = name or func.__name__
        description = description or (func.__doc__ or "").strip()
        tool_definition = tool_definition or self._generate_definition()
        super().__init__(name=name, description=description, tool_definition=tool_definition)
    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        if self.needs_context:
            result = self.func(context=context, **kwargs)
        else:
            result = self.func(**kwargs)
        if inspect.iscoroutine(result):
            return await result
        return result
```

**The `@tool` decorator:** For convenience, we provide a decorator that creates `FunctionTool` instances:
```python
@tool
def calculator(expression: str) -> float:
    """Calculate mathematical expressions."""
    return eval(expression)
```
The decorator also accepts optional parameters that override the tool's metadata:
```python
@tool(name="web_search", description="Search the internet for information")
def search_web(query: str) -> str:
    return tavily_client.search(query)
```

### 5.4 Integrating MCP Tools

MCP tools live on external servers and are executed via `session.call_tool()`, whereas our agent expects `BaseTool` instances with an `execute()` method. The solution is to create a wrapper function for each MCP tool that calls the server and then wraps that function with `FunctionTool`:
```python
async def load_mcp_tools(connection: dict) -> list[BaseTool]:
    tools = []
    async with stdio_client(StdioServerParameters(**connection)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()
        for mcp_tool in mcp_tools.tools:
            func_tool = _create_mcp_tool(mcp_tool, connection)
            tools.append(func_tool)
    return tools
```
With this adapter, MCP tools integrate seamlessly with our agent. The agent does not know or care that some tools are local functions and others call remote MCP servers.

## 6. LLM Communication Layer

With `ExecutionContext` storing our execution state and tools unified under `BaseTool`, we need one more piece: a way to communicate with the LLM.

### 6.1 Reasons for Adding a Communication Layer

`ExecutionContext` contains a list of `Event`s, each holding `ContentItem`s such as `Message`, `ToolCall`, and `ToolResult`. But LLM APIs expect a specific message format: a list of dictionaries with roles and content. Someone must translate between these representations. We also face the challenge of provider diversity. The solution involves three components working together:

- **`LlmRequest`:** Packages what we want to send: instructions, conversation contents, and available tools.
- **`LlmClient`:** Handles the API communication. It transforms `LlmRequest` into the format that LLM APIs expect, makes the call, and converts the response back.
- **`LlmResponse`:** Standardizes what we receive. Regardless of which provider we use, responses come back in the same format that our agent can process.

### 6.2 LlmRequest: Selecting What to Send

`LlmRequest` is the outbound gate from our agent to the LLM. It holds everything required for a single LLM call organized in four components:
```python
class LlmRequest(BaseModel):
    instructions: List[str] = Field(default_factory=list)
    contents: List[ContentItem] = Field(default_factory=list)
    tools: List[BaseTool] = Field(default_factory=list)
    tool_choice: Optional[str] = None
```

Notice what `LlmRequest` does not contain: it has no reference to `ExecutionContext` or `Event`s. The agent is responsible for extracting relevant information from `ExecutionContext` and packaging it into `LlmRequest`. This separation is intentional. It makes `LlmRequest` a simple data container while keeping context-selection logic in the agent, where it belongs. This is where **context engineering** happens.

### 6.3 LlmResponse: Standardizing What We Receive

`LlmResponse` is the inbound gate, standardizing LLM responses regardless of which provider generated them:
```python
class LlmResponse(BaseModel):
    content: List[ContentItem] = Field(default_factory=list)
    error_message: Optional[str] = None
    usage_metadata: Dict[str, Any] = Field(default_factory=dict)
```
- **`content`:** Stores what the LLM produced, represented as `ContentItem` objects.
- **`error_message`:** Captures failures. When an API call fails, we store the error here rather than raise an exception.
- **`usage_metadata`:** Tracks token consumption.

### 6.4 LlmClient: The Provider Adapter

`LlmClient` bridges `LlmRequest` and `LlmResponse` with LLM APIs. Thanks to LiteLLM, we can support multiple providers through a single implementation:
```python
class LlmClient:
    def __init__(self, model: str, **config):
        self.model = model
        self.config = config
    async def generate(self, request: LlmRequest) -> LlmResponse:
        try:
            messages = self._build_messages(request)
            tools = [t.tool_definition for t in request.tools] if request.tools else None
            response = await acompletion(
                model=self.model,
                messages=messages,
                tools=tools,
                **({"tool_choice": request.tool_choice} if request.tool_choice else {}),
                **self.config
            )
            return self._parse_response(response)
        except Exception as e:
            return LlmResponse(error_message=str(e))
```

**Building messages:** The `_build_messages` method transforms `LlmRequest` contents into the message format that LLM APIs expect. The main complexity is that OpenAI's API requires assistant messages and their tool calls to appear together in a single message object while we store them as separate `ContentItem`s.

**Parsing responses:** The `_parse_response` method converts API responses back to our standard format. When the LLM returns text, we wrap it in a `Message`; when it requests tools, we create `ToolCall` objects.

### 6.5 Putting It Together

```python
from scratch_agents import LlmClient, LlmRequest, Message
client = LlmClient(model="gpt-5-mini")
request = LlmRequest(
    instructions=["You are a helpful assistant."],
    contents=[Message(role="user", content="What is 2 + 2?")],
)
response = await client.generate(request)
for item in response.content:
    if isinstance(item, Message):
        print(item.content)  # "4"
```
The communication layer provides clean separation of concerns. `LlmRequest` packages what to send. `LlmClient` handles how to send it. `LlmResponse` standardizes what comes back.

## 7. Implementing the Agent

With `ExecutionContext` storing our state, tools unified under `BaseTool`, and `LlmClient` handling LLM communication, we are ready to build the `Agent` class that orchestrates everything.

### 7.1 Agent Class Structure

```python
class Agent:
    def __init__(
        self,
        model: LlmClient,
        tools: List[BaseTool] = None,
        instructions: str = "",
        max_steps: int = 10,
        name: str = "agent",
    ):
        self.model = model
        self.instructions = instructions
        self.max_steps = max_steps
        self.name = name
        self.tools = self._setup_tools(tools or [])
```
The parameters serve distinct purposes:
- **`model`:** The `LlmClient` instance that handles LLM communication.
- **`tools`:** List of `BaseTool` instances the agent can use.
- **`instructions`:** System prompt that defines the agent's behavior.
- **`max_steps`:** Safety limit to prevent infinite loops.

### 7.2 The run() Method

The `run()` method is the entry point for agent execution. It creates the execution environment, manages the think-act loop, and returns the result. To give callers access to the full execution trace, we define `AgentResult`, which bundles the final output with its `ExecutionContext`:
```python
@dataclass
class AgentResult:
    output: str | BaseModel
    context: ExecutionContext
```
```python
async def run(self, user_input: str, context: ExecutionContext = None) -> AgentResult:
    if context is None:
        context = ExecutionContext()
    user_event = Event(
        execution_id=context.execution_id,
        author="user",
        content=[Message(role="user", content=user_input)]
    )
    context.add_event(user_event)
    while not context.final_result and context.current_step < self.max_steps:
        await self.step(context)
        last_event = context.events[-1]
        if self._is_final_response(last_event):
            context.final_result = self._extract_final_result(last_event)
    return AgentResult(output=context.final_result, context=context)
```

The helper methods handle completion detection and result extraction. `_is_final_response()` checks whether an event represents a final answer by examining its contents. An event is final when it contains neither tool calls nor tool results, meaning that the LLM provided a direct answer. `_extract_final_result()` iterates through the event's content to find the assistant's message.

### 7.3 The step() Method

The `step()` method performs one complete think-act cycle. It prepares a request for the LLM, gets the LLM's decision, and executes any tools that the LLM requests:
```python
async def step(self, context: ExecutionContext):
    llm_request = self._prepare_llm_request(context)
    llm_response = await self.think(llm_request)
    response_event = Event(
        execution_id=context.execution_id,
        author=self.name,
        content=llm_response.content,
    )
    context.add_event(response_event)
    tool_calls = [c for c in llm_response.content if isinstance(c, ToolCall)]
    if tool_calls:
        tool_results = await self.act(context, tool_calls)
        tool_event = Event(
            execution_id=context.execution_id,
            author=self.name,
            content=tool_results,
        )
        context.add_event(tool_event)
    context.increment_step()
```

The `_prepare_llm_request()` method extracts information from `ExecutionContext` and packages it in an `LlmRequest`. It flattens all events into a list of content items, combines the items with the agent's instructions and available tools, and sets the tool choice to "auto" so the LLM can decide whether to use tools. This is where context engineering happens. We could summarize old messages, omit irrelevant tool results, or inject additional information by modifying how we build `LlmRequest` without touching the original context.

### 7.4 The think() and act() Methods

The `think()` method is deliberately simple. It takes the prepared `LlmRequest` and passes it to the `LlmClient`:
```python
async def think(self, llm_request: LlmRequest) -> LlmResponse:
    return await self.model.generate(llm_request)
```

The `act()` method executes the tools that the LLM requested. It builds a dictionary mapping tool names to tool instances for efficient lookup, iterates through each `ToolCall`, retrieves the corresponding tool, executes it with the provided arguments, and collects the results. Each execution is wrapped in a try-except block, capturing failures as error results rather than crashing the entire agent:
```python
async def act(self, context: ExecutionContext, tool_calls: List[ToolCall]) -> List[ToolResult]:
    tools_dict = {tool.name: tool for tool in self.tools}
    results = []
    for tool_call in tool_calls:
        if tool_call.name not in tools_dict:
            results.append(ToolResult(
                tool_call_id=tool_call.tool_call_id,
                name=tool_call.name,
                status="error",
                content=[f"Tool '{tool_call.name}' not found"],
            ))
            continue
        tool = tools_dict[tool_call.name]
        try:
            output = await tool(context, **tool_call.arguments)
            results.append(ToolResult(
                tool_call_id=tool_call.tool_call_id,
                name=tool_call.name,
                status="success",
                content=[output],
            ))
        except Exception as e:
            results.append(ToolResult(
                tool_call_id=tool_call.tool_call_id,
                name=tool_call.name,
                status="error",
                content=[str(e)],
            ))
    return results
```
When tool execution fails, the error message becomes part of the conversation history. The LLM sees this failure in the next step and adapts its approach. This graceful error handling is one of the advantages of the ReAct pattern: the agent can recover from failures rather than stop entirely.

## 8. Adding Structured Output

The agent we built returns only free-form text. Many production scenarios require structured data that downstream systems can parse reliably. The key insight is that we can use our existing tool-calling infrastructure to enforce structured outputs. Instead of having the LLM generate free text and then trying to parse it into a structure, we create a special tool that accepts our desired output schema as its input.

**The approach: Tools as output formatters.** When the agent has to provide its final answer, it calls this tool with properly structured data, guaranteeing type safety and format compliance. This approach reuses our existing tool-calling mechanism, keeping the implementation simple and consistent.

**Modifying the agent:** We add an `output_type` parameter to the constructor. When provided, this parameter should be a Pydantic model class that defines the expected output structure:
```python
def _setup_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
    if self.output_type is not None:
        @tool(
            name="final_answer",
            description="Return the final structured answer matching the required schema."
        )
        def final_answer(output: self.output_type) -> self.output_type:
            return output
        tools = list(tools)
        tools.append(final_answer)
        self.output_tool_name = "final_answer"
    return tools
```

When structured output is required, we must ensure that the agent uses that `final_answer` tool instead of providing a free-text response. We modify `_prepare_llm_request()` to set `tool_choice="required"` to force the LLM to use a tool. We also modify `_is_final_response()` to check for a successful result from our `final_answer` tool, and `_extract_final_result()` to extract the structured data from tool results.

**Using structured output in practice:**
```python
from pydantic import BaseModel
from typing import Literal, List
class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    key_phrases: List[str]
agent = Agent(
    model=LlmClient(model="gpt-5-mini"),
    tools=[],
    instructions="Analyze the sentiment of the provided text.",
    output_type=SentimentAnalysis
)
result = await agent.run("This product exceeded my expectations! Highly recommend.")
print(f"Sentiment: {result.output.sentiment}")      # "positive"
print(f"Confidence: {result.output.confidence}")      # 0.92
```
The agent processes the input and reasons about the sentiment; when it is ready to respond, it calls the `final_answer` tool with a properly structured `SentimentAnalysis` object. The agent's output is the Pydantic model itself, not a string, and it is available through `result.output`, so you can access fields with full type safety.

## 9. Testing with the GAIA Benchmark

In chapter 2, we tested LLMs directly on GAIA problems and observed their limitations. Most problems required information that the LLM did not have, leading to low accuracy. Now let's see how our agent performs when it is equipped with tools.

**From LLM to agent:** The evaluation setup remains largely the same as in chapter 2. The key difference lies in what solves each problem. Instead of making a direct LLM call, now we use an agent with tools:
```python
def create_gaia_agent(model: str, tools: list) -> Agent:
    return Agent(
        model=LlmClient(model=model),
        tools=tools,
        instructions=gaia_prompt,
        output_type=GaiaOutput,
        max_steps=15,
    )
async def solve_problem(agent: Agent, question: str) -> AgentResult:
    async with SEMAPHORE:
        return await agent.run(question)
```

**Debugging with execution traces:** When problems fail, we need to understand why. Because `AgentResult` includes the full `ExecutionContext`, we can examine what happened during execution:
```python
from scratch_agents.utils import display_trace
agent = create_gaia_agent(models[0], mcp_tools)
result = await agent.run(
    "If Eliud Kipchoge could maintain his marathon pace, "
    "how many thousand hours to reach the Moon?"
)
print(f"Answer: {result.output.final_answer}")
print(f"Steps: {result.context.current_step}")
display_trace(result.context)
```
This trace reveals the agent's reasoning: it identified the need for two pieces of information, searched for each piece, and produced the final answer. When debugging failures, you can pinpoint where things went wrong.

**The effect of web search tools:** We ran a controlled experiment comparing agents with and without search tools on 10 GAIA level-1 problems that require web access:

| Condition | Correct | Total cost |
| :--- | :--- | :--- |
| With web search | 7/10 | $1.89 |
| Without web search | 2/10 | $0.26 |

The results demonstrate a **50 percentage-point improvement** when web search tools are available. Without search capabilities, the agent could answer only questions involving information that happened to exist in its training data. With search tools, the agent successfully retrieved current information from Wikipedia, academic papers, and other web sources.

Examining failures reveals interesting patterns. The agent without web search either admitted that it couldn't solve the problem (returning `is_solvable=False`) or hallucinated incorrect answers. When it was asked about Mercedes Sosa's studio albums from 2000 to 2009, the agent with search correctly found three albums on Wikipedia; the agent without search guessed four.

**Remaining challenges:** Although web search dramatically improves performance on information retrieval tasks, many GAIA problems require additional capabilities. A significant portion of level-1 problems includes attached files (such as PDFs, spreadsheets, images, and audio files) that the agent must process to find the answer. Our current agent cannot handle these file-based problems. Chapter 5 will extend our agent with file-processing tools.

## 10. Key Takeaways

- ReAct is the foundational pattern behind modern AI agents, in which the LLM alternates between reasoning about what it needs and taking actions to get it. Modern implementations use tool calling instead of fragile text parsing.
- `ExecutionContext` centralizes all execution state, including events, step count, execution ID, and flexible state storage. This design simplifies method signatures and enables context propagation to tools that need access to execution state.
- Tool abstraction through `BaseTool` and `FunctionTool` provides a consistent interface for all tools. The `@tool` decorator wraps simple functions, whereas `load_mcp_tools()` integrates external MCP servers, allowing the agent to use both local and remote tools seamlessly.
- The LLM communication layer separates concerns cleanly. `LlmRequest` packages what to send, `LlmClient` handles API communication via LiteLLM, and `LlmResponse` standardizes what comes back. This separation is where context engineering happens.
- The `Agent` class orchestrates everything through five core methods. `_setup_tools()` prepares available tools, `run()` manages the execution loop, `step()` performs one think-act cycle, `think()` calls the LLM, and `act()` executes requested tools.
- Structured output enables agents to return type-safe Pydantic models instead of free-form text. By creating a dynamic `final_answer` tool from the output schema, we use existing tool-calling infrastructure to guarantee format compliance.

---

# Chapter 4 — Teaching Outline and Learning Roadmap

### Teaching Instructions

Teach this chapter step by step, following the outline below. Each lesson should include:

- A clear explanation of the concept.
- A practical example, preferably in Python.
- A discussion of why the concept matters when building AI agents.
- A small exercise or coding task.
- A connection to the existing GitHub project when relevant.

Do not skip topics or assume prior knowledge of concepts that have not yet been explained. Clearly distinguish between concepts covered in the book and additional implementation ideas for your project.

### Lesson 1: The Think-Act Cycle

- The ReAct pattern: reasoning, acting, observing, repeating.
- The Kipchoge problem walkthrough: thought, action, observation, thought, action, observation, final answer.
- Why adaptability is the key advantage over fixed scripts.
- **Exercise:** Write out the think-act cycle for a question that requires two searches and one calculation. Identify each thought, action, and observation.

### Lesson 2: From Text Parsing to Tool Calling

- The fragility of early ReAct implementations: exact formatting was required, and any deviation broke the system.
- Tool calling eliminates this fragility by using structured outputs.
- The reasoning did not disappear; it moved inside the LLM's tool-selection process.
- **Exercise:** Compare the two approaches for a simple question ("What is the weather in Paris?"). Write the text-parsing version and the tool-calling version. Explain why the tool-calling version is more reliable.

### Lesson 3: Agent Architecture Overview

- The completed agent interface: `Agent(model, tools, instructions)`.
- Why messages alone are not enough: the need for `events`, `current_step`, `execution_id`, `state`, and `final_result`.
- The ExecutionContext-centered design and its two key benefits: simplified method signatures and context propagation.
- The separation between `ExecutionContext` (stores everything) and `LlmRequest` (curated subset).
- **Exercise:** Draw a diagram of the information flow. Show how `ExecutionContext` connects to `LlmRequest`, `LlmClient`, `LlmResponse`, and tools.

### Lesson 4: Content Types and Events

- The three content types: `Message`, `ToolCall`, and `ToolResult`.
- The `ContentItem` union type.
- The `Event` class: adding metadata (ID, execution_id, timestamp, author) to content items.
- **Exercise:** Create three example events from the Kipchoge problem: one for the user's question, one for the LLM's tool call, and one for the tool result. Write each as an `Event` object.

### Lesson 5: ExecutionContext Implementation

- The five fields: `execution_id`, `events`, `current_step`, `state`, and `final_result`.
- The `add_event()` and `increment_step()` methods.
- How `ExecutionContext` is passed to tools that need access to execution state.
- **Exercise:** Implement the `ExecutionContext` class. Create an instance and add three events to it. Verify that `current_step` increments correctly.

### Lesson 6: BaseTool and FunctionTool

- The `BaseTool` abstract class: name, description, tool_definition, and execute.
- The `execute` method always receives `context` as its first parameter.
- The `FunctionTool` adapter: inspecting the function signature to decide whether to pass context.
- The `@tool` decorator for cleaner syntax.
- **Exercise:** Implement `BaseTool` and `FunctionTool`. Wrap a simple function (e.g., `add(a, b)`) and verify that it works both with and without a context parameter.

### Lesson 7: Integrating MCP Tools

- The challenge: MCP tools live on external servers and are executed via `session.call_tool()`.
- The solution: create a wrapper function that calls the server and wrap it with `FunctionTool`.
- The `load_mcp_tools` function and the `_create_mcp_tool` helper.
- **Exercise:** Implement `load_mcp_tools` and load tools from the Tavily MCP server. Combine them with a local calculator tool and verify that the agent can use both