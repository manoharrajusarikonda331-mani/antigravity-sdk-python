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

r"""Example demonstrating streaming delta fields on Step.

The SDK provides two parallel representations for streaming model output:

  Full state:  step.content / step.thinking
    Accumulated text, always available, always complete up to this point.

  Delta:       step.content_delta / step.thinking_delta
    Incremental suffix since the last update for this step index.
    Contains only the new tokens — suitable for token-by-token rendering.

For production token-by-token rendering, the delta pattern is simply:

  async for step in conversation.receive_steps():
      if step.content_delta:
          sys.stdout.write(step.content_delta)

To run:
  python3 streaming_delta_example.py
"""

import asyncio
from collections.abc import Sequence
import os
import sys

from absl import app
from absl import logging

from google.antigravity import types
from google.antigravity.connections.local.local_connection import LocalConnectionStrategy
from google.antigravity.conversation.conversation import Conversation
from google.antigravity.hooks import hook_runner as hooks_runner
from google.antigravity.hooks import policy
from google.antigravity.utils import cli_utils


async def run_prompt(conversation: Conversation, prompt: str) -> None:
  """Sends a prompt and streams the response using deltas."""
  print(f"\n{'='*60}")
  print(f"--- Sending: {prompt!r} ---")
  print(f"{'='*60}")
  await conversation.send(prompt)

  delta_chars = 0
  full_chars = 0

  async for step in conversation.receive_steps():
    # Stream tokens to stdout as they arrive.
    if step.content_delta:
      sys.stdout.write(step.content_delta)
      sys.stdout.flush()
      delta_chars += len(step.content_delta)

    if step.thinking_delta:
      print(f"  [thinking] {step.thinking_delta}")

    full_chars += len(step.content or "")

    if step.is_complete_response:
      print()  # Newline after streamed content.

  print(f"\n--- Delta efficiency: {delta_chars} delta chars vs "
        f"{full_chars} full chars "
        f"({full_chars / max(delta_chars, 1):.1f}x) ---")


async def run():
  """Runs the streaming delta example."""
  # Auto-approve all tool calls so the agent can run without human
  # confirmation prompts.
  hr = hooks_runner.HookRunner()
  hr.pre_tool_call_decide_hooks.append(
      policy.enforce([policy.allow("*")])
  )

  strategy = LocalConnectionStrategy(
      hook_runner=hr,
      gemini_config=types.GeminiConfig(
          models=types.ModelConfig(
              default=types.ModelEntry(
                  name="gemini-3-flash-preview",
                  generation=types.GenerationConfig(
                      thinking_level=types.ThinkingLevel.LOW,
                  ),
              ),
          ),
      ),
  )

  logging.info("Starting connection...")
  async with Conversation.create(strategy) as conversation:
    cli_utils.print_cli_header("Streaming Delta Example")

    # Demonstrate delta streaming with a prompt that produces text.
    await run_prompt(conversation, "Write a short haiku about streaming data.")

    # Interactive loop for further exploration.
    while True:
      try:
        user_input = await asyncio.to_thread(input, cli_utils.INPUT_PROMPT)
        user_input = user_input.strip()
        if not user_input:
          continue
        if user_input.lower() in ("exit", "quit"):
          print(cli_utils.GOODBYE_MSG)
          break
        await run_prompt(conversation, user_input)
      except (KeyboardInterrupt, asyncio.CancelledError, EOFError):
        print(cli_utils.GOODBYE_MSG)
        break

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
