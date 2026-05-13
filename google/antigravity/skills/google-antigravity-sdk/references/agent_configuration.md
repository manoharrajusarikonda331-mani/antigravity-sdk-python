<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
# Advanced Agent Configuration Guide

This guide provides instructions on how to perform advanced configuration for
Google Antigravity SDK agents.

## Model Selection

### Default Model

Google Antigravity SDK's default model is `gemini-3.5-flash`.

### Finding Valid Models

To find the most up-to-date list of valid Gemini model identifiers, refer to the
official documentation: -
[Google AI Studio Documentation](https://ai.google.dev/gemini-api/docs/models/gemini)

## CRITICAL RULE: Never Assume Valid Model Identifiers

> [!IMPORTANT] **Do not assume valid model identifiers.** Avoid guessing model
> names or assuming they follow a specific pattern. Always verify the valid
> identifiers from official documentation or user context before using them.

> [!IMPORTANT] **Avoid setting the model explicitly unless requested.** It is
> generally better to leave the model unset to use the default behavior, unless
> the user has explicitly requested a specific model.

## Advanced Configuration Examples

Here are small code snippets demonstrating advanced configurations using
`LocalAgentConfig`.

### Basic Configuration with Model Selection

```python
from google.antigravity import Agent, LocalAgentConfig

config = LocalAgentConfig(
    model="gemini-3.5-flash",
)
async with Agent(config=config) as agent:
    # Use the agent
    pass
```

### System Instructions and Personas

You can configure system instructions directly in the `LocalAgentConfig`:
`python config = LocalAgentConfig( system_instructions="You are an expert
software architect.", )` For a more detailed guide and complex persona examples,
see [persona_config.md](../../examples/getting_started/persona_config.md).

### Custom Tools

You can add custom tools to your agent: ```python from google.antigravity import
Agent, LocalAgentConfig

config = LocalAgentConfig( tools=[my_custom_tool_function], ) ``` For a full
guide on creating and using custom tools, see
[custom_tool.md](../../examples/getting_started/custom_tool.md).

### MCP Integration

To configure Model Context Protocol (MCP) servers: `python config =
LocalAgentConfig( mcp_servers={"my_mcp_server": "http://localhost:8080"}, )` For
more details, see [mcp_integration.md](mcp_integration.md).
