# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Example demonstrating end-to-end flow with LocalConnection.

This example shows how to:
1. Define a custom Python tool and register it with a ToolRunner.
2. Connect an MCP server (pirate math tools) via McpBridge.
3. Configure hook-based tool approval policy with CLI interaction.
4. Start a LocalConnectionStrategy that launches the LocalConnection binary.
5. Run an interactive conversation loop using the Conversation API.

To run:
  python local_connection_example.py

Tip: Pass --alsologtostderr to see execution steps in detail.
"""

import asyncio
from collections.abc import Sequence
import os
import sys

from absl import app
from absl import flags
from absl import logging

from google.antigravity import types
from google.antigravity.agent import Agent
from google.antigravity.connections.local.local_connection_config import LocalAgentConfig
from google.antigravity.hooks import cli
from google.antigravity.hooks import hook_runner as hooks_runner
from google.antigravity.hooks import policy
from google.antigravity.mcp.bridge import McpBridge
from google.antigravity.tools.tool_runner import ToolRunner

_MODEL_NAME = flags.DEFINE_string(
    "model_name", "gemini-3-flash-preview", "Gemini model name."
)
_SYSTEM_INSTRUCTION = flags.DEFINE_string(
    "system_instruction", None, "System instruction text for the agent."
)
_DISABLE_RUN_COMMAND = flags.DEFINE_bool(
    "disable_run_command",
    False,
    "Whether to disable the run_command tool.",
)
_SHOW_USAGE = flags.DEFINE_bool(
    "show_usage",
    False,
    "Whether to display token usage and trajectory after each turn.",
)


def read_file_upside_down(path: str) -> str:
  """Reads the file at the given path and returns its content with lines inverted.

  Args:
      path: The path to the file to read.

  Returns:
      The file content with lines in reverse order.
  """
  logging.info("Tool read_file_upside_down called with path: %s", path)
  with open(path, "r") as f:
    lines = f.readlines()
  return "".join(reversed(lines))


def _add(cur: int | None, val: int | None) -> int | None:
  """Adds two nullable ints, preserving None when both are absent."""
  if val is None:
    return cur
  return (cur or 0) + val


def _print_telemetry(
    turn_usage: types.UsageMetadata | None,
    cumul: types.UsageMetadata,
    history: list[types.Step],
) -> None:
  """Prints telemetry data for the current turn."""
  print("\n--- Turn Token Usage ---")
  if turn_usage:
    print(f"  Prompt tokens:   {turn_usage.prompt_token_count}")
    print(f"  Cached tokens:   {turn_usage.cached_content_token_count}")
    print(f"  Output tokens:   {turn_usage.candidates_token_count}")
    print(f"  Thinking tokens: {turn_usage.thoughts_token_count}")
    print(f"  Total tokens:    {turn_usage.total_token_count}")
  else:
    print("  Usage data not available for this turn.")

  # Cumulative session usage.
  print("\n--- Session Cumulative Usage ---")
  print(f"  Prompt tokens:   {cumul.prompt_token_count}")
  print(f"  Cached tokens:   {cumul.cached_content_token_count}")
  print(f"  Output tokens:   {cumul.candidates_token_count}")
  print(f"  Thinking tokens: {cumul.thoughts_token_count}")
  print(f"  Total tokens:    {cumul.total_token_count}")

  # Trajectory summary.
  print(f"\n--- Trajectory ({len(history)} steps) ---")
  for i, s in enumerate(history):
    label = f"    [{i}] {s.type.value} ({s.source.value}) - {s.status.value}"
    if s.tool_calls:
      names = ", ".join(tc.name for tc in s.tool_calls)
      label += f" [{names}]"
    print(label)
  print()


async def run():
  """Runs the example."""
  try:

    mcp_server_path = os.path.join(
        os.path.dirname(__file__), "mcp_server.par"
    )

    config = LocalAgentConfig(
        tools=[read_file_upside_down],
        mcp_servers=[types.McpStdioServer(
            command=mcp_server_path,
            args=["--transport=stdio"],
        )],
        policies=[policy.ask_user("*", handler=cli.ask_user_handler)],
        hooks=[cli.AskQuestionHook()],
        capabilities=types.CapabilitiesConfig(
            disabled_tools=(
                [types.BuiltinTools.RUN_COMMAND]
                if _DISABLE_RUN_COMMAND.value
                else None
            ),
        ),
    )
    config.gemini_config = types.GeminiConfig(
        models=types.ModelConfig(
            default=types.ModelEntry(name=_MODEL_NAME.value),
        ),
    )
    config.system_instructions = _SYSTEM_INSTRUCTION.value

    logging.info("Starting agent...")
    async with Agent(config) as agent:

      print("\nGoogle Antigravity SDK Demo")
      print("Type your message and press Enter • Ctrl+C to exit\n")

      while True:
        try:
          user_input = await asyncio.to_thread(input, "\n→ ")
          user_input = user_input.strip()
          if not user_input:
            continue
          if user_input.lower() in ("exit", "quit"):
            print("\nGoodbye! 👋")
            break

          response = await agent.chat(user_input)

          # Stream the response to stdout
          async for chunk in response:
            sys.stdout.write(chunk)
            sys.stdout.flush()
          print()

          if _SHOW_USAGE.value:
            assert agent._conversation is not None
            _print_telemetry(
                response.usage_metadata,
                agent.total_usage,
                agent._conversation.history,
            )

        except (KeyboardInterrupt, asyncio.CancelledError, EOFError):
          print("\nGoodbye! 👋")
          break

  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"An error occurred: {e}", file=sys.stderr)
    logging.exception("Error running example: %s", e)

  # asyncio.to_thread(input) spawns a thread that blocks on stdin. This thread
  # cannot be interrupted in CPython, so asyncio.run() will hang during executor
  # shutdown. os._exit() is the standard workaround for this scenario.
  os._exit(0)


def main(argv: Sequence[str]) -> None:
  del argv
  logging.set_verbosity(logging.INFO)
  asyncio.run(run())


if __name__ == "__main__":
  app.run(main)
