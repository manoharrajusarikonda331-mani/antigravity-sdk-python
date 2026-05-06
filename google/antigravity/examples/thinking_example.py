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

r"""Example demonstrating model thinking visibility via the SDK.

This example shows how to access the model's reasoning/thinking content
as a first-class field on each Step. When thinking is enabled via
GenerationConfig.thinking_level, the `step.thinking` field is populated
with the model's internal reasoning, separate from `step.content`.

To run:
  python thinking_example.py

Override the thinking level:
  python thinking_example.py --thinking_level=high
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
from google.antigravity.utils import cli_utils

_MODEL_NAME = flags.DEFINE_string(
    "model_name", "gemini-3-flash-preview", "Gemini model name."
)
_THINKING_LEVEL = flags.DEFINE_enum_class(
    "thinking_level",
    types.ThinkingLevel.LOW,
    types.ThinkingLevel,
    "Thinking level (minimal, low, medium, high).",
)


async def run():
  """Runs the thinking example."""
  try:
    strategy = LocalConnectionStrategy(
        gemini_config=types.GeminiConfig(
            models=types.ModelConfig(
                default=types.ModelEntry(
                    name=_MODEL_NAME.value,
                    generation=types.GenerationConfig(
                        thinking_level=_THINKING_LEVEL.value,
                    ),
                ),
            ),
        ),
        capabilities_config=types.CapabilitiesConfig(
            disabled_tools=[types.BuiltinTools.RUN_COMMAND],
        ),
    )

    logging.info(
        "Starting connection (model: %s, thinking: %s)...",
        _MODEL_NAME.value,
        _THINKING_LEVEL.value,
    )
    async with Conversation.create(strategy) as conversation:

      cli_utils.print_cli_header("Thinking Example")
      print("Ask a question to see the model's reasoning process.\n")

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
              if step.thinking:
                print(f"\n  💭 Thinking: {step.thinking}")
              if step.is_complete_response:
                print(f"\n  💬 Response: {step.content}\n")
          except asyncio.CancelledError:
            print("\nCanceling current request...")
            await conversation.cancel()

        except (KeyboardInterrupt, asyncio.CancelledError, EOFError):
          print(cli_utils.GOODBYE_MSG)
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
