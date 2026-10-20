
import os
import json
import inspect
import traceback

from dotenv import load_dotenv
from litellm import completion
from ddgs import DDGS


# ==================================================
# Configuration
# ==================================================

load_dotenv()

MODEL_NAME = "openai/gemini-2.5-flash"
API_BASE = "https://1xai.ir/v1"
MAX_STEPS = 5


# ==================================================
# 1. Web search tool
# ==================================================

def search_web(
    query: str,
    max_results: int = 3,
) -> list:
    """Search the web for information and return relevant results."""

    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                query,
                max_results=max_results,
            )

        return [
            {
                "title": result.get("title"),
                "url": result.get("href") or result.get("url"),
                "content": result.get("body"),
            }
            for result in results
        ]

    except Exception as e:
        return [
            {
                "error": f"Web search failed: {e}"
            }
        ]


# ==================================================
# 2. Convert function parameters to JSON Schema
# ==================================================

def function_to_input_schema(func) -> dict:
    """Convert a Python function's parameters into a JSON schema."""

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    signature = inspect.signature(func)
    parameters = {}

    for param in signature.parameters.values():
        param_type = type_map.get(
            param.annotation,
            "string",
        )

        parameters[param.name] = {
            "type": param_type,
        }

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect.Parameter.empty
    ]

    return {
        "type": "object",
        "properties": parameters,
        "required": required,
    }


# ==================================================
# 3. Convert a function to an LLM tool definition
# ==================================================

def function_to_tool_definition(func) -> dict:
    """Convert a Python function into an LLM tool definition."""

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": function_to_input_schema(func),
        },
    }


# ==================================================
# 4. Execute a tool requested by the LLM
# ==================================================

def tool_execution(tool_box, tool_call):
    """Execute a tool requested by the LLM."""

    function_name = tool_call.function.name

    function_args = json.loads(
        tool_call.function.arguments or "{}"
    )

    if function_name not in tool_box:
        raise ValueError(
            f"Unknown tool requested: {function_name}"
        )

    print(f"[DEBUG] Executing tool: {function_name}")
    print(f"[DEBUG] Arguments: {function_args}")

    tool_result = tool_box[function_name](
        **function_args
    )

    return tool_result


# ==================================================
# 5. Agent loop
# ==================================================

def simple_agent_loop(
    system_prompt: str,
    question: str,
    max_steps: int = MAX_STEPS,
) -> str:
    """Manage the conversation between the LLM and tools."""

    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")

    api_key = os.getenv("XAI_1_API_KEY")

    if not api_key:
        raise RuntimeError(
            "XAI_1_API_KEY was not found. Check your .env file."
        )

    tools = [search_web]

    tool_box = {
        tool.__name__: tool
        for tool in tools
    }

    tool_definitions = [
        function_to_tool_definition(tool)
        for tool in tools
    ]

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    for step in range(max_steps):
        print(
            f"\n[DEBUG] Agent step: {step + 1}/{max_steps}"
        )

        response = completion(
            model=MODEL_NAME,
            api_base=API_BASE,
            api_key=api_key,
            messages=messages,
            tools=tool_definitions,
        )

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        if not tool_calls:
            print("\n[DEBUG] No tool call was returned.")

            print(
                "[DEBUG] Finish reason:",
                response.choices[0].finish_reason,
            )

            print(
                "[DEBUG] Assistant response:",
                repr(assistant_message.content),
            )

            return assistant_message.content or ""

        print("\n[DEBUG] The model requested a tool.")

        messages.append(assistant_message)

        for tool_call in tool_calls:
            try:
                tool_result = tool_execution(
                    tool_box,
                    tool_call,
                )

            except Exception as e:
                print(
                    f"[DEBUG] Tool execution failed: {e}"
                )

                tool_result = {
                    "error": str(e)
                }

            messages.append({
                "role": "tool",
                "content": json.dumps(
                    tool_result,
                    ensure_ascii=False,
                    default=str,
                ),
                "tool_call_id": tool_call.id,
            })

    # The step limit has been reached.
    print(
        f"\n[DEBUG] Maximum step limit reached: {max_steps}"
    )

    messages.append({
        "role": "system",
        "content": (
            "The tool execution limit has been reached. "
            "Do not request or execute any more tools. "
            "Provide the best final answer you can using "
            "the conversation and tool results already available. "
            "Be transparent about any missing information."
        ),
    })

    # Request a final answer without allowing more tool calls.
    try:
        final_response = completion(
            model=MODEL_NAME,
            api_base=API_BASE,
            api_key=api_key,
            messages=messages,
            tools=tool_definitions,
            tool_choice="none",
        )

        final_message = final_response.choices[0].message
        final_content = final_message.content or ""

        print("\n[DEBUG] Forced final response received.")

        return final_content

    except Exception as e:
        print(
            f"[ERROR] Failed to generate final response: {e}"
        )

        return (
            "The agent reached its maximum step limit, "
            "but the final response could not be generated. "
            f"Error: {e}"
        )


# ==================================================
# 6. Test the agent manually
# ==================================================

if __name__ == "__main__":

    system_prompt = """
You are a helpful research assistant.

You have access to a web search tool named search_web.

For questions about current events or facts that may have changed,
you must call search_web before answering.

Use the search results to construct your final answer.
Do not invent facts that are not supported by the results.
"""

    question = (
        "Who won the 2025 Nobel Prize in Physics? "
        "Search the web and identify the recipients."
    )

    print("=" * 60)
    print("AGENT TOOL EXECUTION TEST")
    print("=" * 60)

    print(f"Model: {MODEL_NAME}")
    print(f"Question: {question}")
    print("\nRunning agent...\n")

    try:
        result = simple_agent_loop(
            system_prompt,
            question,
        )

        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)
        print(result)

    except Exception:
        traceback.print_exc()
