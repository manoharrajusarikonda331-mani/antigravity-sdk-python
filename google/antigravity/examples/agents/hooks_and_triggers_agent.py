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

"""Agent example with custom hooks and triggers."""

import asyncio
import logging
from typing import Any
from google.antigravity import types
from google.antigravity.agent import Agent
from google.antigravity.connections.local.local_connection_config import LocalAgentConfig
from google.antigravity.hooks import hooks
from google.antigravity.triggers import triggers as triggers_module
from google.antigravity.triggers.helpers import every


class MyPreTurnHook(hooks.PreTurnHook):
  """Logs the prompt before the turn starts."""

  async def run(
      self, context: hooks.HookContext, data: Any
  ) -> types.HookResult:
    logging.info("PRE-TURN HOOK: Prompt is: %s", data)
    return types.HookResult(allow=True)


async def ping_callback(ctx: triggers_module.TriggerContext):
  """Callback for the periodic trigger."""
  logging.info("TRIGGER: Pinging agent...")
  await ctx.send(
      "Keep-alive ping from trigger",
  )


async def main():
  logging.basicConfig(level=logging.INFO)

  my_hook = MyPreTurnHook()
  # Ping every 5 seconds for the example
  my_trigger = every(5, ping_callback)

  print("Creating agent...")
  config = LocalAgentConfig(
      system_instructions="You are a helpful assistant.",
      hooks=[my_hook],
      triggers=[my_trigger],
  )
  async with Agent(config) as agent:

    print("\nChatting with agent...")
    response = await agent.chat("Hello! Tell me a short joke.")
    print(f"Agent: {response.text}\n")

    # Wait a bit to let the trigger fire
    print("Waiting for trigger to fire...")
    await asyncio.sleep(10)


if __name__ == "__main__":
  asyncio.run(main())
