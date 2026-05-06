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

"""Example demonstrating stateful session resumption.

Runs two independent agent sessions sharing the same save_dir. The first
session establishes context ("my favorite color is blue"), then shuts
down. The second session provides the conversation_id from the first
run and asks the agent to recall the earlier context, confirming that
the prior trajectory was restored.
"""

import asyncio
import logging
import tempfile


from google.antigravity.agent import Agent
from google.antigravity.connections.local.local_connection_config import LocalAgentConfig


async def main():
  logging.basicConfig(level=logging.INFO)

  save_dir = tempfile.mkdtemp(prefix="agent_session_")
  print(f"Save directory: {save_dir}")

  # --- Session 1: establish context ---
  print("\n=== Session 1: establishing context ===")
  config = LocalAgentConfig(
      system_instructions="You are a helpful assistant.",
      save_dir=save_dir,
  )
  async with Agent(config) as agent:
    response = await agent.chat(
        "Remember this: my favorite color is blue."
    )
    print(f"Agent: {response.text}")

    # Read back the conversation_id assigned by the runtime.
    conversation_id = agent.conversation_id
    print(f"Assigned conversation ID: {conversation_id}")
  print("Session 1 ended.\n")

  # --- Session 2: resume and verify recall ---
  print("=== Session 2: resuming and verifying recall ===")
  config = LocalAgentConfig(
      system_instructions="You are a helpful assistant.",
      conversation_id=conversation_id,
      save_dir=save_dir,
  )
  async with Agent(config) as agent:
    response = await agent.chat(
        "What is my favorite color?"
    )
    print(f"Agent: {response.text}")
  print("Session 2 ended.")


if __name__ == "__main__":
  asyncio.run(main())
