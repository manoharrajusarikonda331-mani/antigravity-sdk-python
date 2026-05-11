# Google Antigravity SDK MCP Integration

The `mcp` package provides integration with the Model Context Protocol (MCP), allowing the agent to use tools exposed by external MCP servers.

## Core Concepts

### `McpBridge`

`McpBridge` simplifies the lifecycle of MCP client sessions and tool registration. It allows connecting to MCP servers and automatically registering their tools with the SDK's `ToolRunner`.

Supported connection types:
- **stdio**: For connecting to local MCP servers running as subprocesses.
- **SSE (Server-Sent Events)**: For connecting to remote MCP servers over HTTP.
- **Streamable HTTP**: For connecting to remote MCP servers with advanced timeout and connection management.

### `get_mcp_tools`

A helper function that fetches tools from an MCP `ClientSessionGroup` and returns them as `ToolWithSchema` objects.

## Usage Example

```python
from google.antigravity.mcp.bridge import McpBridge
from google.antigravity.tools.tool_runner import ToolRunner

tool_runner = ToolRunner()
bridge = McpBridge()

# Connect to a local MCP server
await bridge.connect_stdio(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-everything"]
)

# Tools from the MCP server are now registered in tool_runner
# and available to the agent.

# Cleanup when done
await bridge.stop()
```

## Files

- `bridge.py`: Defines `McpBridge` and `get_mcp_tools`.
