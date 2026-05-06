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

"""Example demonstrating TemplatedSystemInstructions."""

import asyncio
import logging
from google.antigravity import types
import google.antigravity.agent as agent_module
from google.antigravity.connections.local.local_connection_config import LocalAgentConfig
from google.antigravity.examples import example_policies


# Define a custom tool for the security agent
def check_vulnerability_database(library_name: str) -> str:
  """Checks if a given library has known vulnerabilities."""
  if library_name == "insecure-lib":
    return "Found CVE-2026-XXXX: Remote Code Execution."
  return "No known vulnerabilities found."


async def main():
  logging.basicConfig(level=logging.INFO)

  print("=== Templated System Instructions Example ===")

  # Override the Identity (Persona)
  # This replaces the default identity.
  identity = (
      "You are a specialized Code Security Auditor agent.\n"
      "Your role is to analyze code for vulnerabilities, secret leaks, and poor"
      " security practices."
  )

  # Add General Guidelines
  guidelines_section = types.SystemInstructionSection(
      title="security_guidelines",
      content=(
          "- Always flag hardcoded credentials or API keys.\n"
          "- Prefer environment variables over local config files."
      ),
  )

  # Add Tool-Specific Instructions
  # This guides the agent on how to use the specific tool we are providing.
  tool_instructions = types.SystemInstructionSection(
      title="tool_guidelines_check_vulnerability_database",
      content=(
          "When using the `check_vulnerability_database` tool:\n"
          "- Always report the CVE number if found.\n"
          "- If no vulnerabilities are found, still advise caution if the"
          " library is old."
      ),
  )

  templated_si = types.TemplatedSystemInstructions(
      identity=identity,
      sections=[guidelines_section, tool_instructions],
  )

  print("Creating agent with advanced templated instructions...")
  config = LocalAgentConfig(
      system_instructions=templated_si,
      tools=[check_vulnerability_database],
      capabilities=types.CapabilitiesConfig(),
      policies=[example_policies.BLOCK_RM_POLICY],
  )
  async with agent_module.Agent(config) as agent:
    print("\nChatting with agent...")
    response = await agent.chat(
        "Should I use the library 'insecure-lib' in my project?"
    )
    print(f"Agent: {response.text}\n")


if __name__ == "__main__":
  asyncio.run(main())
