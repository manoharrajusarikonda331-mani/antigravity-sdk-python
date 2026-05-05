# Google Antigravity SDK

The Google Antigravity SDK is a Python SDK for building AI agents powered by
Antigravity and Gemini. It provides a secure, scalable, and stateful
infrastructure layer that abstracts the agentic loop, letting you focus on what
your agent *does* rather than how it runs.

## Installation

```sh
pip install google-antigravity
```

## Quickstart

### Simple Agent

The `Agent` class is the easiest way to get started. It manages the full
lifecycle — binary discovery, tool wiring, hook registration, and policy
defaults — behind a single async context manager.

The `system_instructions` parameter is optional.

```python
import asyncio
from google.antigravity import Agent

async def main():
    async with Agent(system_instructions="You are an expert assistant for codebase navigation.", api_key="GEMINI_API_KEY") as agent:
        response = await agent.chat("What files are in the current directory?")
        print(response)

asyncio.run(main())
```

By default, `Agent` runs in **read-only mode** for safety. Pass
`read_only=False` to enable write operations.

### Interactive Loop

```python
async with Agent(system_instructions="You are a helpful assistant.", api_key="GEMINI_API_KEY", read_only=False) as agent:
    await agent.run_interactive_loop()
```

### Advanced Usage with Conversation

For full control over the connection lifecycle, use `Conversation` with a
`ConnectionStrategy` directly. `Conversation` is a stateful session that
accumulates step history, provides a `chat()` convenience method, and exposes
state introspection:

```python
import asyncio
from google.antigravity.connections.local_connection import LocalConnectionStrategy
from google.antigravity.conversation.conversation import Conversation
from google.antigravity.tools.tool_runner import ToolRunner
from google.antigravity.types import GeminiConfig

async def main():
    tool_runner = ToolRunner()
    strategy = LocalConnectionStrategy(
        binary_path="/path/to/localharness",
        tool_runner=tool_runner,
        gemini_config=GeminiConfig(api_key="GEMINI_API_KEY"),
    )

    async with Conversation.create(strategy) as conversation:
        # High-level: one-call send + collect
        response = await conversation.chat("What files are here?")
        print(response.text)

        # Step history accumulates automatically
        print(f"Total steps: {len(conversation.history)}")
        print(f"Turns: {conversation.turn_count}")
        print(f"Last response: {conversation.last_response}")

        # Low-level: streaming steps
        await conversation.send("Tell me more.")
        async for step in conversation.receive_steps():
            if step.is_complete_response:
                print(step.content)

asyncio.run(main())
```

## Features

### Multimodal Ingestion

Pass rich multimedia file attachments (images, videos, audio, and documents) to the agent alongside textual instruction prompt lists:

```python
from google.antigravity import Agent, Part

async with Agent(system_instructions="You are an expert software architect.") as agent:
    # Load local multimedia assets seamlessly via Part.from_file
    image_part = Part.from_file("diagram.png", description="System design chart")
    
    # Send a mixed content list
    prompt = ["Analyze this architecture chart and list three security vulnerabilities:", image_part]
    response = await agent.chat(prompt)
    print(response)
```

### Custom Tools

Register Python functions as tools that the agent can call:

```python
def get_weather(city: str) -> str:
    """Returns the current weather for a city."""
    return f"It's sunny in {city}."

async with Agent(system_instructions="You are a helpful assistant.", tools=[get_weather]) as agent:
    response = await agent.chat("What's the weather in Tokyo?")
```

### MCP Integration

Connect to external [MCP](https://modelcontextprotocol.io/) servers and expose
their tools to the agent:

```python
from google.antigravity import Agent

async with Agent(
    system_instructions="You are a helpful assistant.",
    mcp_servers=[{"type": "stdio", "command": "npx", "args": ["my-mcp-server"]}],
) as agent:
    response = await agent.chat("Use the MCP tools to help me.")
```

### Hooks and Policies

Control agent behavior with a declarative policy system:

```python
from google.antigravity import Agent
from google.antigravity.hooks.policy import deny, allow, ask_user, enforce

policies = [
    deny("*"),                          # Block all tools by default
    allow("view_file"),                 # Allow reading files
    ask_user("run_command", handler=my_handler),  # Ask before running commands
]

async with Agent(system_instructions="You are a helpful assistant.", policies=policies) as agent:
    await agent.run_interactive_loop()
```

### Triggers

Run background tasks that react to external events and push messages into the
agent:

```python
from google.antigravity import Agent
from google.antigravity.triggers import every

async def check_status(ctx):
    ctx.send("Check the deployment status.")

async with Agent(system_instructions="You are a helpful assistant.", triggers=[every(60, check_status)]) as agent:
    await agent.run_interactive_loop()
```

## Architecture

The SDK follows a three-layer architecture:

| Layer | Purpose | Key Classes |
|:------|:--------|:------------|
| **Layer 1** — Simplified | High-level, batteries-included entry point | `Agent` |
| **Layer 2** — Session | Stateful session with history and convenience methods | `Conversation`, `ChatResponse`, `Step`, `ToolCall`, `HookRunner`, `ToolRunner`, `TriggerRunner` |
| **Layer 3** — Adapter | Transport and backend abstraction | `Connection`, `ConnectionStrategy`, `LocalConnection` |

## Component Documentation

For more detailed documentation on specific components, see:

-   [Agent](agent.py) — High-level, batteries-included entry point.
-   [Connections](connections/README.md) — Transport and backend abstraction.
-   [Conversation](conversation/README.md) — Stateful session management.
-   [Hooks](hooks/README.md) — Agent lifecycle interception and policies.
-   [MCP](mcp/README.md) — Model Context Protocol integration.
-   [Tools](tools/README.md) — In-process tool execution.
-   [Triggers](triggers/README.md) — Background tasks and external events.

## Examples

See the [`examples/`](examples/) directory for complete, runnable examples:

- **Local connection** — custom tools, MCP, and policy hooks with
  `LocalConnectionStrategy`
- **MCP server** — example MCP server with "pirate math" tools
- **Agent examples** — simple agent, custom tools + MCP, policies, hooks and
  triggers

## License

[Apache License 2.0](LICENSE)
