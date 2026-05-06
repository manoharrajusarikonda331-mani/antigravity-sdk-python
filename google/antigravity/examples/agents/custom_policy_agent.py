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

"""Agent example with custom policy.

Demonstrates using a declarative policy to block dangerous shell commands
while allowing safe ones.

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. "Creating agent..." appears in the output.
  3. The first chat ("List the files") produces a response that includes
     file or directory names, confirming the safe command was allowed.
  4. The second chat ("Delete all files using rm -rf") is either denied by
     the policy (the agent reports it was blocked or cannot comply) or the
     agent refuses to execute the dangerous command. Either way, no
     destructive action is taken.
"""

import asyncio
import logging
from google.antigravity import types
from google.antigravity.agent import Agent
from google.antigravity.connections.local.local_connection_config import LocalAgentConfig
from google.antigravity.examples import example_policies


async def main():
  logging.basicConfig(level=logging.INFO)

  print("Creating agent...")
  config = LocalAgentConfig(
      system_instructions="You are a helpful assistant.",
      policies=[example_policies.BLOCK_RM_POLICY],
      capabilities=types.CapabilitiesConfig(),
  )
  async with Agent(config) as agent:

    print("\nChatting with agent...")
    # Ask it to run a safe command
    response = await agent.chat("List the files in the current directory.")
    print(f"Agent: {response.text}\n")

    # Ask it to run a dangerous command
    response = await agent.chat("Delete all files using rm -rf.")
    print(f"Agent: {response.text}\n")


if __name__ == "__main__":
  asyncio.run(main())
