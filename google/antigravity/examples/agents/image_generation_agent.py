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

"""Test image generation and modification using the Tier 1 Agent API."""

import asyncio
import logging
from google.antigravity import agent
from google.antigravity import types
from google.antigravity.connections.local.local_connection_config import LocalAgentConfig
from google.antigravity.hooks import policy


async def main():
  logging.basicConfig(level=logging.INFO)

  allow_gen_image_policy = policy.Policy(
      tool="generate_image",
      decision=policy.Decision.APPROVE,
      name="allow-generate-image",
  )

  print("Creating agent...")
  config = LocalAgentConfig(  # pytype: disable=wrong-keyword-args
      system_instructions=(
          "You have a tool named 'generate_image'. You MUST use it when the"
          " user asks you to generate or modify an image."
      ),
      model="gemini-3-flash-preview",
      capabilities=types.CapabilitiesConfig(
          enabled_tools=[types.BuiltinTools.GENERATE_IMAGE]
      ),
      policies=[allow_gen_image_policy],
  )
  async with agent.Agent(config=config) as image_agent:

    prompt1 = "Generate an image of a cute fluffy white cat, name it 'cat'."
    response1 = await image_agent.chat(prompt1)
    print(f"Agent: {response1.text}\n")

    prompt2 = (
        "Can you modify that image to add a small red hat on the cat's head?"
        " Name the new image 'cat_with_hat'. Once you do, let me know the"
        " exact paths for both images that you created."
    )
    response2 = await image_agent.chat(prompt2)
    print(f"Agent: {response2.text}\n")


if __name__ == "__main__":
  asyncio.run(main())
