# Chapter 3 Summary — Enabling Actions: Tool Use

**Note:** This document is based on Chapter 3 of *Build an AI Agent (From Scratch)* and follows the same structure as the Chapter 1, 2, 6, 8, 9, and 10 summaries.

## 1. Introduction: Why LLMs Need Tools

On their own, large language models (LLMs) cannot access external data or interact with external systems. They need tools and the ability to select and use those tools, which is central to implementing agents. A basic agent performs tasks step by step by repeatedly selecting the next action to take, and tools make this possible.

In this chapter, we explore how to build tools and inform the LLM how to use them, how the LLM chooses the appropriate tool for a given task, and how tool executions are relayed back to the LLM. We then examine the **Model Context Protocol (MCP)** by Anthropic, an initiative that standardizes the development and use of tools.

## 2. LLM Tools

Despite their impressive capabilities, LLMs have inherent limitations that prevent them from functioning as fully autonomous agents. These limitations fall into three categories, all of which tools address:

| Limitation | Description | Tool type that addresses it |
| :--- | :--- | :--- |
| **Temporal** | LLMs cannot access information beyond their training data or retrieve data that changes in real time. | Information-augmenting tools (web search, domain-specific retrieval, real-time APIs). |
| **Interaction** | LLMs cannot influence the external world directly. | Action-executing tools (booking systems, communication tools, IoT controllers). |
| **Functional** | Tasks requiring precise computation, code execution, or media generation exceed what language models can reliably accomplish. | Domain-specialized tools (calculators, code interpreters, format converters, image generators). |

### 2.1 Types of LLM Tools

Tools are categorized as **custom-built** by the developer or **predefined** by the LLM provider.

- **Custom tools:** Developed or adapted by the developer to fit a specific environment (e.g., APIs for internal company systems, specialized data analysis scripts). The developer is responsible for building, maintaining, and securing them.
- **Noncustom tools:** Predefined and offered directly by the LLM provider (e.g., OpenAI's web search, Anthropic's web search tool). They can be used immediately, but cannot be easily modified or extended.

**Why this book focuses on custom tools:**
1. Understanding how tools work internally is essential for debugging agent failures. Built-in tools are black boxes that obscure these details.
2. Real-world agents often require specialized capabilities that providers do not offer (e.g., querying a company's internal database, integrating with proprietary APIs).

## 3. How LLMs Use Tools

How does a model that can only generate text select and execute a tool? The answer lies in a mechanism called **tool calling**.

### 3.1 How Tool Calling Works

Tool calling involves a multistep process:

1. **Specify tool definitions:** The developer sends a message (representing user input) to the LLM, along with tool definitions. A tool definition is a structured description that tells the LLM what tools are accessible, what parameters they expect, and what they do.
2. **LLM generates a tool call:** The LLM reasons about which tool is appropriate and generates a tool call—a structured output that includes the name of the tool and the arguments required to invoke it.
3. **Execute the tool:** The LLM does not execute tools itself. The host system (the agent application) executes the tool based on the LLM's tool call.
4. **Feed the result back:** The tool's result is added back to the LLM's context, enriching its understanding.
5. **Generate the final response:** The LLM interprets the result and continues the conversation or produces the final answer.

**Step 1: Specify tool definitions.** A tool definition has three essential components:
- **`name`:** The tool's identifier that the LLM will use to call it.
- **`description`:** When and why to use this tool.
- **`parameters`:** The specification of inputs required to use the tool (following a JSON schema format).

Example calculator tool definition:
```python
calculator_tool_definition = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform basic arithmetic operations.",
        "parameters": {
            "type": "object",
            "properties": {
                "operator": {
                    "type": "string",
                    "description": "Arithmetic operation to perform",
                    "enum": ["add", "subtract", "multiply", "divide"]
                },
                "first_number": {"type": "number", "description": "First number for the calculation"},
                "second_number": {"type": "number", "description": "Second number for the calculation"}
            },
            "required": ["operator", "first_number", "second_number"],
        },
    },
}
```

**Step 2: Set up the tool.** Define a Python function such as `calculator()` that matches the input schema.

**Step 3: Execute the tool calling.** The LLM receives the tool definitions and the user's question. For a question like "What's the capital of South Korea?", the model provides a direct answer without invoking any tools. For "What is 1234 x 5678?", the model generates a tool call with the appropriate parameters.

**Step 4: Run the tool.** The host system extracts the function name and arguments from the tool call and executes the corresponding function.

**Step 5: Feed results back to the LLM.** Append the tool result as a message with a role of `"tool"` and include the output in the content. The updated context is passed back to the LLM, which generates the final response.

### 3.2 How an LLM Chooses Tools

During training, LLMs are exposed to datasets containing tool definitions paired with appropriate tool calls. The model develops the ability to:
1. Recognize when a tool is needed based on the user's request.
2. Select the appropriate tool from the available options.
3. Extract the correct parameter values from the user's input.
4. Format the tool call in the expected structure.

### 3.3 Guidelines for Effective Tool Calling

The following best practices are adapted from OpenAI's official documentation:

**Clear and explicit function definitions:** Define the tool's purpose, parameter types, and expected outputs explicitly. A good rule of thumb: if an intern can understand and use the tool from the schema alone, so can an LLM.

**Applying software engineering best practices:** Avoid ambiguous or logically conflicting parameters. For example, `toggle_light(on: bool, off: bool)` allows contradictory inputs. Replace mutually exclusive Booleans with a single enum.

**Reducing model load and using code effectively:** Do not make the model resupply identifiers you already have. Let code remember context, and let the model focus on the decision itself. For example, if an `order_id` was retrieved in a previous step, it is more efficient to provide a parameterless function such as `submit_refund()` and pass the `order_id` automatically through code.

**Limiting the number of functions:** As a general guideline, keep the number of tools under 20. Using too many tools can lead to selection errors.

## 4. Building Tools and Tool Definitions for LLMs

### 4.1 Implementing a Web Search Tool

For web search, this book uses **Tavily** because it is optimized for LLMs and delivers results that are easier to process. Tavily allows up to 1,000 free API calls per month.

**Starting simple:**
```python
import os
from tavily import TavilyClient
from dotenv import load_dotenv
load_dotenv()
tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))
def search_web(query: str, max_results: int = 2) -> list:
    response = tavily_client.search(query, max_results=max_results)
    return response.get("results")
```

**Adding search options:** Tavily's API supports parameters such as `topic` ("general", "news", "finance"), `time_range` (filter by recency), and `country`.

**Handling errors gracefully:** Wrap the API call in a try-except block and return an error string if anything goes wrong:
```python
def search_web(query: str, max_results: int = 5, topic: str = "general", time_range: str | None = None) -> list | str:
    try:
        response = tavily_client.search(query, max_results=max_results, topic=topic, time_range=time_range)
        return response.get("results")
    except Exception as e:
        return f"Error: Search failed - {e}"
```

### 4.2 Converting to Tool Definitions

To make functions accessible to an LLM, we must define them using a standardized tool-definition format. Because functions may change frequently during development, manual updates are error-prone. We create a utility function that automatically converts a Python function to a tool definition using the `inspect` module.

**Extracting tool metadata:**
```python
import inspect
def example_tool(input_1: str, input_2: int = 1):
    """docstring for example_tool"""
    return
print(f"function name: {example_tool.__name__}")
print(f"function docstring: {example_tool.__doc__}")
print(f"function signature: {inspect.signature(example_tool)}")
```

**Generating the input schema:** The `function_to_input_schema` utility maps Python types to JSON Schema types and determines which parameters are required based on whether they have default values.

**Converting functions to tool definitions:**
```python
def function_to_tool_definition(func) -> dict:
    return format_tool_definition(
        func.__name__,
        func.__doc__ or "",
        function_to_input_schema(func)
    )
```

### 4.3 End-to-End Tool Execution

We need two components: a function to execute tools based on LLM output and a control loop to manage the conversation between the LLM and tools.

**Tool execution utility:**
```python
def tool_execution(tool_box, tool_call):
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)
    tool_result = tool_box[function_name](**function_args)
    return tool_result
```

**Simple agent loop:**
```python
def simple_agent_loop(system_prompt, question):
    tools = [search_web]
    tool_box = {tool.__name__: tool for tool in tools}
    tool_definitions = [function_to_tool_definition(tool) for tool in tools]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    while True:
        response = completion(model="gpt-5-mini", messages=messages, tools=tool_definitions)
        assistant_message = response.choices[0].message
        if assistant_message.tool_calls:
            messages.append(assistant_message)
            for tool_call in assistant_message.tool_calls:
                tool_result = tool_execution(tool_box, tool_call)
                messages.append({
                    "role": "tool",
                    "content": str(tool_result),
                    "tool_call_id": tool_call.id
                })
        else:
            return assistant_message.content
```

### 4.4 The Challenges of Custom Tools

Building the web search tool was straightforward, but as you develop more tools and scale to real-world projects, several challenges emerge:

| Challenge | Description |
| :--- | :--- |
| **Understanding unfamiliar APIs** | Most services are not built with LLMs in mind. Each service has its own API conventions, rate-limiting policies, pagination patterns, and response formats. |
| **Handling errors robustly** | Production tools must handle network timeouts, rate limits, service outages, input validation, and provide meaningful error messages. |
| **Duplicating efforts across teams** | Without a standard way to share tools, each team builds their own implementation, wasting engineering resources. |
| **Carrying the ongoing burden of maintenance** | APIs evolve, and tools must evolve with them. Each tool becomes a maintenance responsibility. |

These challenges are exactly what **MCP** addresses.

## 5. MCP: Standardizing Tools

In November 2024, Anthropic introduced **Model Context Protocol (MCP)** to bring standardization to LLM tool integration. Just as REST APIs established a common language for developers to communicate with services, MCP established a common language for LLM agents to communicate with tools.

### 5.1 The Core of MCP: Server-Client Architecture

MCP separates the brain of the agent from its arms and legs (tools) through a three-component architecture:

| Component | Role |
| :--- | :--- |
| **Host** | The application that interacts with users and makes decisions using an LLM (e.g., Claude Desktop, an IDE with AI features, the agent application you are building). |
| **Client** | The bridge between the host and the MCP servers. It handles communication, requests available tool definitions from servers, and forwards tool-execution requests. Each client maintains a connection to one MCP server. |
| **Server** | Hosts the tools and executes them on demand. It responds to two types of requests: "What tools do you have?" and "Execute this tool with these arguments." |

**Two standardized interfaces:**
- **Tool discovery:** The client can request available tool definitions, and the server returns them in a standardized format.
- **Tool execution:** The client sends a tool-execution request to the server, which handles execution and returns the result in a consistent format.

**Transport mechanisms:**
- **Stdio (standard I/O):** The client launches the server as a subprocess and communicates through standard input/output streams. Ideal for local development.
- **Streamable HTTP:** Enables remote communication over the network.

### 5.2 Hands-On: Running an MCP Server

The book demonstrates running the official Tavily MCP server using the MCP Inspector, a browser-based tool for testing MCP servers:
```bash
npx @modelcontextprotocol/inspector npx -y tavily-mcp@latest
```
With a single command, we downloaded and ran a production-ready MCP server, connected to it using a standard client interface, discovered available tools automatically, and executed a tool.

### 5.3 Understanding the MCP Client

Production agents need to interact with MCP servers programmatically. The MCP client handles protocol details such as establishing connections, requesting tool definitions, executing tools, and parsing results.
```python
import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
load_dotenv()
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "tavily-mcp@latest"],
    env={"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")}
)
async with stdio_client(server_params) as (read_stream, write_stream):
    async with ClientSession(read_stream, write_stream) as session:
        await session.initialize()
        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            print(f" - {tool.name}: {tool.description[:60]}...")
        result = await session.call_tool("tavily_search", arguments={"query": "2025 Nobel Physics"})
        print(result.content)
```

**Converting MCP tools for LLM use:** MCP tools use MCP's schema format. To use these tools with LLMs, we convert them to the format the LLM provider expects:
```python
def mcp_tools_to_openai_format(mcp_tools) -> list[dict]:
    return [
        format_tool_definition(
            name=tool.name,
            description=tool.description,
            parameters=tool.inputSchema,
        )
        for tool in mcp_tools.tools
    ]
```

### 5.4 Hands-On: Implementing an MCP Server

Using the FastMCP library, you can convert an existing Python function to an MCP-compatible tool by adding a decorator:
```python
import os
from tavily import TavilyClient
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
load_dotenv()
tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))
mcp = FastMCP("custom-tavily-search")
@mcp.tool()
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily API."""
    try:
        response = tavily_client.search(query, max_results=max_results)
        results = response.get("results", [])
        return "\n\n".join(
            f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}"
            for r in results
        )
    except Exception as e:
        return f"Error searching web: {str(e)}"
if __name__ == "__main__":
    mcp.run(transport='stdio')
```
Notice how little code we added to our original function. The core logic remains unchanged; we simply wrapped it with the MCP infrastructure.

## 6. Key Takeaways

- Tools extend LLMs' capabilities, enabling them to access APIs, perform calculations, and retrieve external information. Tools fall into three categories: information-augmenting, action-executing, and domain-specialized.
- Tool calling is the mechanism by which LLMs generate structured outputs that specify which tool to use and with what parameters. The LLM acts as a mediator between user requests and available tools but does not execute the tools.
- Tool execution is handled externally. The LLM generates specifications that a separate system executes, with the results being fed back into the LLM's context for further reasoning.
- Tool definitions are structured schemas that describe available tools, their parameters, and expected outputs. Clear, explicit definitions are essential for reliable tool selection by the LLM.
- Building custom tools involves implementing the tool function, creating tool definitions, and building execution infrastructure. Although using them is straightforward for simple cases, custom tools create a maintenance burden and inconsistencies as projects scale.
- MCP standardizes tool development through a server-client architecture. Servers host and execute tools; clients discover and invoke them. This separation enables tool reuse across projects and providers.

---

# Chapter 3 — Teaching Outline and Learning Roadmap

### Teaching Instructions

Teach this chapter step by step, following the outline below. Each lesson should include:

- A clear explanation of the concept.
- A practical example, preferably in Python.
- A discussion of why the concept matters when building AI agents.
- A small exercise or coding task.
- A connection to the existing GitHub project when relevant.

Do not skip topics or assume prior knowledge of concepts that have not yet been explained. Clearly distinguish between concepts covered in the book and additional implementation ideas for your project.

### Lesson 1: Why LLMs Need Tools

- The three limitations of LLMs: temporal, interaction, and functional.
- The three types of tools that address each limitation: information-augmenting, action-executing, and domain-specialized.
- **Exercise:** For each of the three limitations, give one example of a real-world task where an LLM would fail without a tool. Identify which type of tool would solve the problem.

### Lesson 2: Types of LLM Tools

- Custom tools vs. noncustom tools: definitions, examples, and trade-offs.
- Why this book focuses on custom tools: debugging and specialized capabilities.
- **Exercise:** List three built-in tools offered by OpenAI or Anthropic. For each, describe a scenario where a custom tool would be a better choice.

### Lesson 3: How Tool Calling Works

- The five-step process: specify tool definitions, LLM generates a tool call, execute the tool, feed the result back, generate the final response.
- The three essential components of a tool definition: name, description, and parameters.
- **Exercise:** Write a tool definition for a `get_weather` function that takes a city name and returns the current temperature. Include the name, description, and parameters.

### Lesson 4: How an LLM Chooses Tools

- How training on tool-use examples enables the model to recognize when a tool is needed.
- The four capabilities: recognize, select, extract, and format.
- **Exercise:** Given a tool definition for a calculator and a user question ("What is 1234 x 5678?"), write the expected tool call in JSON format. Then write the expected tool call for a question that does not require a tool.

### Lesson 5: Guidelines for Effective Tool Calling

- Clear and explicit function definitions.
- Applying software engineering best practices: avoiding conflicting parameters.
- Reducing model load: letting code remember context.
- Limiting the number of functions to under 20.
- **Exercise:** Take the following antipattern and improve it: `toggle_light(on: bool, off: bool)`. Write the improved tool definition and explain why it is better.

### Lesson 6: Implementing a Web Search Tool

- Introduction to Tavily and its advantages for LLM applications.
- Starting with a simple implementation.
- Adding search options: `topic`, `time_range`, `country`.
- Handling errors gracefully with try-except.
- **Exercise:** Implement the `search_web` function with error handling. Test it with a query and verify that it returns results. Then test it with an invalid API key and verify that it returns an error message.

### Lesson 7: Converting to Tool Definitions

- The `inspect` module: extracting name, docstring, and signature.
- Mapping Python types to JSON Schema types.
- Determining required parameters.
- **Exercise:** Write a function called `translate_text` that takes a text and a target language. Use the `function_to_tool_definition` utility to convert it to a tool definition. Print the result.

### Lesson 8: End-to-End Tool Execution

- The tool execution utility.
- The simple agent loop: sending the question, checking for tool calls, executing tools, feeding results back.
- **Exercise:** Implement the `simple_agent_loop` function with the `search_web` tool. Test it with a question that requires current information ("Who won the 2025 Nobel Prize in Physics?").

### Lesson 9: The Challenges of Custom Tools

- Understanding unfamiliar APIs.
- Handling errors robustly.
- Duplicating efforts across teams.
- Carrying the ongoing burden of maintenance.
- **Exercise:** For each of the four challenges, describe a scenario where it would arise in a real project. Explain how MCP would address it.

### Lesson 10: MCP Architecture

- The three components: host, client, and server.
- The two standardized interfaces: tool discovery and tool execution.
- The two transport mechanisms: stdio and streamable HTTP.
- **Exercise:** Draw a diagram of the MCP architecture. Label the host, client, server, and the requests/responses exchanged between them.

### Lesson 11: Running an MCP Server

- Using the MCP Inspector to test a server.
- The command to launch the Tavily MCP server.
- Exploring the available tools: `tavily_search`, `tavily_extract`, `tavily_crawl`, `tavily_map`.
- **Exercise:** Run the MCP Inspector with the Tavily MCP server. List the available tools and execute one tool with a query of your choice.

### Lesson 12: Understanding the MCP Client

- Connecting to an MCP server: `StdioServerParameters`, `stdio_client`, `ClientSession`.
- Listing available tools with `session.list_tools()`.
- Executing a tool with `session.call_tool()`.
- Converting MCP tools to OpenAI format.
- **Exercise:** Write a client that connects to the Tavily MCP server, lists the available tools, and executes a search for "2025 Nobel Prize." Print the results.

### Lesson 13: Implementing an MCP Server

- Using FastMCP to wrap an existing Python function.
- The `@mcp.tool()` decorator.
- Running the server with `mcp.run(transport='stdio')`.
- **Exercise:** Take the `search_web` function you implemented in Lesson 6 and wrap it with FastMCP. Connect to it with the client from Lesson 12 and verify that it works.