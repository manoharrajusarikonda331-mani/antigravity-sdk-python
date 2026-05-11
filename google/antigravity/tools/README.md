# Google Antigravity SDK Tools

The `tools` package provides utilities for managing and executing tools within the Google Antigravity SDK. Tools are Python callables that the agent can invoke to perform actions or fetch information.

## Core Concepts

### `ToolRunner`

`ToolRunner` is a registry and executor for in-process Python tools. It maintains a mapping of tool names to callables and handles their execution, including running synchronous functions in a separate thread to avoid blocking the async event loop.

Key features:
- **Registration**: Register tools with `register(tool, name=None)`.
- **Execution**: Execute a tool by name with `execute(tool_name, **kwargs)`.
- **Batch Processing**: Process a list of `ToolCall` objects and return `ToolResult` objects with `process_tool_calls(tool_calls)`.
- **Support for Sync and Async**: Automatically detects if a tool is async or sync and handles it appropriately.

### `ToolWithSchema`

A wrapper class for callables that have an explicit JSON Schema defined. This is useful for tools that come from external sources like MCP servers.

## Available Tools and Utilities

## Usage Example

```python
from google.antigravity.tools.tool_runner import ToolRunner

def my_custom_tool(param: str) -> str:
    """A custom tool that does something."""
    return f"Processed: {param}"

tool_runner = ToolRunner(tools=[my_custom_tool])

# Execute directly
result = await tool_runner.execute("my_custom_tool", param="input value")
print(result)

# Process structured tool calls
from google.antigravity import types
calls = [types.ToolCall(name="my_custom_tool", args={"param": "another input"})]
results = await tool_runner.process_tool_calls(calls)
print(results[0].result)
```

## Files

- `tool_runner.py`: Defines `ToolRunner` and `ToolWithSchema`.
