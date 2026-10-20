import inspect


def example_tool(input_1: str, input_2: int = 1):
    """An example tool."""
    return


def function_to_input_schema(func) -> dict:
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
        param_type = type_map.get(param.annotation, "string")

        parameters[param.name] = {
            "type": param_type
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


def format_tool_definition(
    name: str,
    description: str,
    parameters: dict,
) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def function_to_tool_definition(func) -> dict:
    return format_tool_definition(
        func.__name__,
        func.__doc__ or "",
        function_to_input_schema(func),
    )


if __name__ == "__main__":
    print(function_to_input_schema(example_tool))
    print(function_to_tool_definition(example_tool))