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
from google.antigravity.connections.local.local_connection import LocalConnectionStrategy
from google.antigravity.conversation.conversation import Conversation
from google.antigravity.hooks import cli
from google.antigravity.hooks import hook_runner as hooks_runner
from google.antigravity.hooks import policy
from google.antigravity.mcp.bridge import McpBridge
from google.antigravity.tools.tool_runner import ToolRunner
from google.antigravity.utils import cli_utils

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


async def run():
  """Runs the example."""
  strategy = None
  mcp_bridge = None
  try:
    # In-process Python tools via ToolRunner ---
    # ToolRunner executes Python functions directly in this process.
    # These tools are dispatched by the Connection when the agent calls them.
    tool_runner = ToolRunner(tools=[read_file_upside_down])

    # McpBridge connects to a separate MCP server process over stdio.
    # Its tools are merged into the ToolRunner so both paths are available
    # to the agent.
    mcp_bridge = McpBridge(tool_runner)
    mcp_server_path = os.path.join(
        os.path.dirname(__file__), "mcp_server.par"
    )
    await mcp_bridge.connect_stdio(mcp_server_path, ["--transport=stdio"])
    logging.info("MCP server connected (pirate math tools available).")

    hr = hooks_runner.HookRunner()
    hr.pre_tool_call_decide_hooks.append(
        policy.enforce([policy.ask_user("*", handler=cli.ask_user_handler)])
    )
    hr.on_interaction_hooks.append(cli.AskQuestionHook())

    # Initialize LocalConnectionStrategy with ToolRunner and binary path
    strategy = LocalConnectionStrategy(
        tool_runner=tool_runner,
        hook_runner=hr,
        gemini_config=types.GeminiConfig(
            models=types.ModelConfig(
                default=types.ModelEntry(name=_MODEL_NAME.value),
            ),
        ),
        system_instructions=_SYSTEM_INSTRUCTION.value,
        capabilities_config=types.CapabilitiesConfig(
            disabled_tools=(
                [types.BuiltinTools.RUN_COMMAND]
                if _DISABLE_RUN_COMMAND.value
                else None
            ),
        ),
    )

    # Create Conversation
    logging.info("Starting connection and creating conversation...")
    async with Conversation.create(strategy) as conversation:

      cli_utils.print_cli_header("Antigravity SDK Demo")

      while True:
        try:
          user_input = await asyncio.to_thread(input, cli_utils.INPUT_PROMPT)
          user_input = user_input.strip()
          if not user_input:
            continue
          if user_input.lower() in ("exit", "quit"):
            print(cli_utils.GOODBYE_MSG)
            break

          await conversation.send(user_input)

          try:
            async for step in conversation.receive_steps():
              if step.is_complete_response:
                print(f"\n{step.content}\n")
          except asyncio.CancelledError:
            print("\nCanceling current request...")
            await conversation.cancel()

        except (KeyboardInterrupt, asyncio.CancelledError, EOFError):
          print(cli_utils.GOODBYE_MSG)
          break

  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"An error occurred: {e}", file=sys.stderr)
    logging.exception("Error running example: %s", e)
  finally:
    if mcp_bridge is not None:
      logging.info("Stopping McpBridge...")
      await mcp_bridge.stop()

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
